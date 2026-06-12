#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn as nn
import psutil
import traceback
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss
from gaussian_renderer import render, network_gui
import sys # <-- 确保 sys 已导入
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vq_utils import load_2d_language_feature, load_2d_lmm_feature, ResidualVectorQuantizationWithClustering
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt

print(f"DEBUG: sys.argv = {sys.argv}")


def log_stage(message):
    print(f"\n[STAGE] {message}", flush=True)


def load_checkpoint_compat(checkpoint_path):
    """Load checkpoints across PyTorch versions where weights_only default changed."""
    try:
        return torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path)


def get_gt_feature_for_training(viewpoint_cam, dataset, opt):
    """Load GT feature/mask for the current camera and indicate whether it is usable."""
    if not opt.include_feature:
        return None, None, True

    if not opt.llm_feature:
        gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(
            language_feature_dir=dataset.lf_path,
            feature_level=dataset.feature_level,
        )
    else:
        gt_language_feature, language_feature_mask = viewpoint_cam.get_llm_feature(
            language_feature_dir=dataset.lf_path,
            feature_level=dataset.feature_level,
        )

    has_valid_mask = bool(language_feature_mask.any().item())
    return gt_language_feature, language_feature_mask, has_valid_mask



def _resize_chw(tensor, target_hw, mode):
    """Resize a [C,H,W] tensor with a torch interpolation mode."""
    if tensor.shape[1:] == target_hw:
        return tensor
    kwargs = {"size": target_hw, "mode": mode}
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor.unsqueeze(0).float(), **kwargs).squeeze(0).to(dtype=tensor.dtype)


def _resize_mask_chw(mask, target_hw):
    """Resize a [1,H,W] mask with nearest interpolation and keep it boolean-like."""
    if mask.shape[1:] == target_hw:
        return mask.bool()
    resized = F.interpolate(mask.unsqueeze(0).float(), size=target_hw, mode="nearest").squeeze(0)
    return resized > 0.5


def _center_crop_chw(tensor, target_hw):
    """Center-crop [C,H,W] tensor to target_hw."""
    target_h, target_w = target_hw
    _, H, W = tensor.shape
    h = min(H, target_h)
    w = min(W, target_w)
    top = max(0, (H - h) // 2)
    left = max(0, (W - w) // 2)
    return tensor[:, top:top + h, left:left + w]


def align_feature_supervision(language_feature, gt_language_feature, language_feature_mask, args):
    """
    Align rendered and GT feature maps for reconstruction-resolution ablations.

    The old crop-only fallback is kept as an explicit mode for 1px legacy drift, but
    RF/down_scale mismatch experiments should use resize_gt_to_render or
    resize_render_to_gt so the supervision grid is intentional and reproducible.
    """
    if language_feature.shape[1:] == gt_language_feature.shape[1:]:
        return language_feature, gt_language_feature, language_feature_mask

    mode = args.feature_align_mode
    interp = args.feature_interp
    pred_hw = language_feature.shape[1:]
    gt_hw = gt_language_feature.shape[1:]

    if mode == "strict":
        raise RuntimeError(
            f"Feature resolution mismatch: rendered={pred_hw}, gt={gt_hw}. "
            "Use --feature_align_mode resize_gt_to_render/resize_render_to_gt/crop_only."
        )

    if mode == "resize_gt_to_render":
        gt_language_feature = _resize_chw(gt_language_feature, pred_hw, interp)
        language_feature_mask = _resize_mask_chw(language_feature_mask, pred_hw)
    elif mode == "resize_render_to_gt":
        language_feature = _resize_chw(language_feature, gt_hw, interp)
    elif mode == "crop_only":
        h = min(pred_hw[0], gt_hw[0])
        w = min(pred_hw[1], gt_hw[1])
        target_hw = (h, w)
        language_feature = _center_crop_chw(language_feature, target_hw)
        gt_language_feature = _center_crop_chw(gt_language_feature, target_hw)
        language_feature_mask = _center_crop_chw(language_feature_mask, target_hw).bool()
    else:
        raise ValueError(f"Unknown --feature_align_mode: {mode}")

    if args.log_feature_alignment:
        print(
            f"[ALIGN] {mode}: rendered {pred_hw} vs gt {gt_hw} -> "
            f"rendered {tuple(language_feature.shape[1:])}, gt {tuple(gt_language_feature.shape[1:])}",
            flush=True,
        )
    return language_feature, gt_language_feature, language_feature_mask


def masked_cos_loss(network_output, gt, mask):
    """Cosine loss on valid pixels without materialising full masked copies."""
    valid = mask.bool().squeeze(0)
    if not valid.any():
        raise RuntimeError("masked_cos_loss received an empty valid mask")
    pred = network_output[:, valid]
    target = gt[:, valid]
    return 1 - F.cosine_similarity(pred, target, dim=0).mean()


def masked_l1_loss(network_output, gt, mask):
    """L1 loss on valid pixels without materialising full masked copies."""
    valid = mask.bool().squeeze(0)
    if not valid.any():
        raise RuntimeError("masked_l1_loss received an empty valid mask")
    return torch.abs(network_output[:, valid] - gt[:, valid]).mean()


def llm_crop_native_supervision_loss(viewpoint_cam, dataset, language_feature, args):
    """
    Compare rendered LLaVA features against native 27x27 crop features.

    The dense-map path expands every crop to full image resolution before resizing
    back to the render grid. For 3584-dim LLaVA features this is CPU-bound and can
    make the GPU appear idle. This path keeps supervision on the stored crop grid:
    crop rendered features by bbox, resize that prediction to the native crop
    tensor size, and average crop losses.
    """
    encoded = viewpoint_cam.get_llm_scale_data(dataset.lf_path, dataset.feature_level)
    if not encoded or not encoded.get('crop_features'):
        raise RuntimeError(
            f"No encoded LLM crop features for {viewpoint_cam.image_name} "
            f"at feature_level={dataset.feature_level}"
        )

    image_w, image_h = encoded['image_size']
    _, render_h, render_w = language_feature.shape
    device = language_feature.device
    dtype = language_feature.dtype
    weighted_loss = language_feature.new_tensor(0.0)
    total_weight = 0

    for crop_data in encoded['crop_features']:
        gt_feature = crop_data['feature']
        if not torch.is_tensor(gt_feature):
            gt_feature = torch.from_numpy(gt_feature)
        if gt_feature.numel() == 0:
            continue

        x, y, w, h = crop_data['bbox']
        x1 = max(0, min(render_w, int(round(float(x) * render_w / image_w))))
        x2 = max(0, min(render_w, int(round(float(x + w) * render_w / image_w))))
        y1 = max(0, min(render_h, int(round(float(y) * render_h / image_h))))
        y2 = max(0, min(render_h, int(round(float(y + h) * render_h / image_h))))
        if x2 <= x1 or y2 <= y1:
            continue

        gt_chw = gt_feature.permute(2, 0, 1).contiguous().to(device=device, dtype=dtype)
        pred_crop = language_feature[:, y1:y2, x1:x2]
        interp_kwargs = {
            "size": gt_chw.shape[1:],
            "mode": args.feature_interp,
        }
        if args.feature_interp in ("bilinear", "bicubic"):
            interp_kwargs["align_corners"] = False
        pred_native = F.interpolate(pred_crop.unsqueeze(0).float(), **interp_kwargs).squeeze(0).to(dtype=dtype)

        crop_weight = int(gt_chw.shape[1] * gt_chw.shape[2])
        crop_loss = language_feature.new_tensor(0.0)
        if args.cos_loss:
            crop_loss = crop_loss + (1 - F.cosine_similarity(pred_native, gt_chw, dim=0).mean())
        if args.l1_loss:
            crop_loss = crop_loss + torch.abs(pred_native - gt_chw).mean()
        weighted_loss = weighted_loss + crop_loss * crop_weight
        total_weight += crop_weight

    if total_weight == 0:
        raise RuntimeError(
            f"No valid LLM crop supervision regions for {viewpoint_cam.image_name} "
            f"at feature_level={dataset.feature_level}"
        )
    return weighted_loss / total_weight


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    log_stage("Preparing output, scene, Gaussian model, and optimizer")
    tb_writer = prepare_output_and_logger(dataset)
    # 根据特征类型设置特征维度（LLM特征为 3584 维）
    language_feature_dim = 3584 if opt.llm_feature else 512
    gaussians = GaussianModel(dataset.sh_degree, language_feature_dim=language_feature_dim)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        # Handle PyTorch version differences:
        # - torch>=2.6: default weights_only=True; force weights_only=False
        # - older torch: no weights_only arg; fall back without it
        (model_params, first_iter) = load_checkpoint_compat(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        log_stage(f"Restored checkpoint from {checkpoint}; first_iter={first_iter}")
    
    # Initialize language feature codebooks
    if opt.include_feature and first_iter == 0:
      if not opt.llm_feature:
        log_stage("Loading 2D language features and fitting CLIP codebook")
        device = torch.device("cuda")
        features = load_2d_language_feature(dataset.lf_path, device)
        rvq = ResidualVectorQuantizationWithClustering(opt.vq_layer_num, opt.codebook_size, features.shape[1], device).to(device)
        rvq.fit_quantizers(features)
        codebooks = torch.stack(rvq.quantizers, dim=0).to(device) # [vq_layer_num, codebook_size, feature_dim] e.g. [1,64,512]
        with torch.no_grad():
            gaussians._language_feature_codebooks.data.copy_(codebooks)
      else: # opt.llm_feature == True
        # try load from the pre-cached codebooks
        device = torch.device("cuda")

        # 码本文件名包含 feature_level 和 vq_layer_num；避免 L=1/L=2 ablation 复用错误缓存。
        level_name = ['small', 'medium', 'large'][dataset.feature_level]
        codebook_filename = (
            f"llm_codebooks_L{opt.vq_layer_num}_K{opt.codebook_size}"
            f"_level{dataset.feature_level}_{level_name}.pt"
        )
        codebook_path = os.path.join(dataset.lf_path, codebook_filename)
        legacy_codebook_filename = f"llm_codebooks_{opt.codebook_size}_level{dataset.feature_level}_{level_name}.pt"
        legacy_codebook_path = os.path.join(dataset.lf_path, legacy_codebook_filename)
        if opt.vq_layer_num == 1 and not os.path.exists(codebook_path) and os.path.exists(legacy_codebook_path):
            codebook_path = legacy_codebook_path
            codebook_filename = legacy_codebook_filename
        log_stage(
            f"Preparing LLM codebook for feature_level={dataset.feature_level} "
            f"({level_name}); expected cache={codebook_path}"
        )

        if os.path.exists(codebook_path):
            log_stage(f"Loading cached LLM codebook: {codebook_filename}")
            codebooks = load_checkpoint_compat(codebook_path).to(device)
            expected_shape = (opt.vq_layer_num, opt.codebook_size, gaussians.language_feature_dim)
            if tuple(codebooks.shape) != expected_shape:
                raise RuntimeError(
                    f"Cached codebook shape {tuple(codebooks.shape)} does not match expected {expected_shape}: {codebook_path}"
                )
            with torch.no_grad():
                gaussians._language_feature_codebooks.data.copy_(codebooks)
        else:
            log_stage(f"{codebook_filename} not found; fitting LLM codebook from feature files")

            # 根据feature_level选择对应的scale特征
            load_func = level_name  # 'small', 'medium', 或 'large'
            log_stage(f"Preparing {load_func} scale feature stream for level {dataset.feature_level}")

            features = load_2d_lmm_feature(dataset.lf_path, device, load_func=load_func)
            first_block = features.peek()
            feature_dim = first_block.shape[1] if first_block.size else 0
            log_stage(f"Initialising RVQ with feature_dim={feature_dim}, codebook_size={opt.codebook_size}")
            rvq = ResidualVectorQuantizationWithClustering(
                opt.vq_layer_num, opt.codebook_size, feature_dim, device
            ).to(device)
            log_stage("Fitting quantizers; this can be slow and CPU-heavy")
            rvq.fit_quantizers(features)
            log_stage("Stacking learned quantizers and saving codebook cache")
            codebooks = torch.stack(rvq.quantizers, dim=0).to(device) # [vq_layer_num, codebook_size, feature_dim] e.g. [1,64,512]
            print(codebooks.shape)
            print(f"Codebooks device: {codebooks.device}, dtype: {codebooks.dtype}")
            # save the codebooks for future use
            torch.save(codebooks.cpu(), codebook_path)
            print(f"Saved codebooks to {codebook_path}")
            # 重新加载到 GPU（因为上面 .cpu() 可能影响了原始张量）
            codebooks = codebooks.to(device)
            with torch.no_grad():
                gaussians._language_feature_codebooks.data.copy_(codebooks)
            print(f"Copied codebooks to gaussians, shape: {gaussians._language_feature_codebooks.shape}")

    log_stage("Entering main training loop")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_record = []
    iter_record = []
    smooth_loss = None
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Memory logging
        if iteration % 100 == 0:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"\n[ITER {iteration}] RAM: {mem_info.rss / 1024 ** 2:.2f} MB", end="")
            if torch.cuda.is_available():
                print(f" | GPU Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | GPU Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB", end="")
            print("")

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = None
        gt_language_feature = None
        language_feature_mask = None
        while viewpoint_stack:
            candidate_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            if opt.include_feature and opt.llm_feature and args.llm_supervision_mode == "crop_native":
                has_valid_mask = candidate_cam.has_llm_feature(dataset.lf_path, dataset.feature_level)
            else:
                gt_language_feature, language_feature_mask, has_valid_mask = get_gt_feature_for_training(
                    candidate_cam, dataset, opt
                )
            if has_valid_mask:
                viewpoint_cam = candidate_cam
                break
            print(
                f"[WARN] Skipping camera {candidate_cam.image_name}: empty GT feature mask "
                f"at feature_level={dataset.feature_level}",
                flush=True,
            )
        if viewpoint_cam is None:
            raise RuntimeError(
                f"No valid GT feature mask found for feature_level={dataset.feature_level} "
                f"across the available training cameras."
            )
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        opt.topk = args.topk
        try:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
            image, language_feature_weight_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["language_feature_weight_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            # Loss
            if opt.include_feature:
                # In this paper, we select layer_num = 1
                layer_num, _, _ = gaussians.get_language_feature_codebooks.shape
                layer_idx = min(int(iteration / 10000 * layer_num), layer_num - 1)
                language_feature = gaussians.compute_layer_feature_map(language_feature_weight_map, layer_idx)
                if args.normalize:
                    language_feature = language_feature / (language_feature.norm(dim=0, keepdim=True) + 1e-10)

                if opt.llm_feature and args.llm_supervision_mode == "crop_native":
                    loss = llm_crop_native_supervision_loss(viewpoint_cam, dataset, language_feature, args)
                else:
                    # Align spatial grids for RF resolution vs LLaVA down_scale ablations.
                    language_feature, gt_language_feature, language_feature_mask = align_feature_supervision(
                        language_feature,
                        gt_language_feature,
                        language_feature_mask,
                        args,
                    )
                    # 这里理论上不应再出现空 mask：view 已在采样前筛选过
                    if not language_feature_mask.any():
                        raise RuntimeError(
                            f"Training camera {viewpoint_cam.image_name} lost all valid GT pixels after crop alignment "
                            f"at feature_level={dataset.feature_level}."
                        )

                    loss = 0
                    if args.cos_loss:
                        cosloss = masked_cos_loss(language_feature, gt_language_feature, language_feature_mask)
                        loss += cosloss
                    if args.l1_loss:
                        Ll1 = masked_l1_loss(language_feature, gt_language_feature, language_feature_mask)
                        loss += Ll1

            else:
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
        except Exception as e:
            print(f"\n[ERROR] Crash at iteration {iteration} with camera {viewpoint_cam.image_name}")
            if torch.cuda.is_available():
                print(f"[ERROR] Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
            raise e
        iter_end.record()
        
        iter_record.append(iteration)
        if smooth_loss is None:
            smooth_loss = loss.item()
        else:
            smooth_loss = smooth_loss * 0.99 + loss.item() * 0.01
        loss_record.append(smooth_loss)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            if (iteration in saving_iterations):
                # print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if (iteration < opt.iterations) and (iteration % args.accum_iter == 0):
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                # print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    torch.save((gaussians.capture(opt.include_feature), opt.iterations), scene.model_path + "/chkpnt" + str(opt.iterations) + ".pth")        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    # print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--cos_loss', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument(
        '--feature_align_mode',
        type=str,
        default='resize_gt_to_render',
        choices=['resize_gt_to_render', 'resize_render_to_gt', 'crop_only', 'strict'],
        help='How to align rendered feature maps with GT LLaVA maps when RF -r and feature down_scale differ.',
    )
    parser.add_argument(
        '--feature_interp',
        type=str,
        default='bilinear',
        choices=['nearest', 'bilinear', 'bicubic'],
        help='Interpolation mode for feature-map resizing in --feature_align_mode resize_* modes.',
    )
    parser.add_argument(
        '--log_feature_alignment',
        action='store_true',
        default=False,
        help='Print every feature-resolution alignment operation for debugging ablations.',
    )
    parser.add_argument(
        '--llm_supervision_mode',
        type=str,
        default='dense_map',
        choices=['dense_map', 'crop_native'],
        help='dense_map decodes LLaVA crops to a full map; crop_native supervises on stored crop grids to avoid CPU-heavy dense decode.',
    )
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print(f"DEBUG: args.source_path = {args.source_path}")
    print(f"DEBUG: lp.extract(args).source_path = {lp.extract(args).source_path}")
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    except Exception:
        print("\n\nTraining crashed! Saving traceback to crash_log.txt")
        traceback.print_exc()
        with open("crash_log.txt", "w") as f:
            traceback.print_exc(file=f)
    # All done
    # print("\nTraining complete.")
