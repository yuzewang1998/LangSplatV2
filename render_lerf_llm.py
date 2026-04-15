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
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import time
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pathlib import Path
import cv2
import logging

from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
import eval.colormaps as colormaps
import json
import glob
from collections import defaultdict
from typing import Dict, Union
import sys
sys.path.append("eval")
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import numpy as np
from utils.vq_utils import get_weights_and_indices

import torch.nn.functional as F

# 导入特征对比可视化
from visualize_comparison_simple_rgb import visualize_comparison


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    print(json_folder)
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), '*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), '*.jpg')))
    gt_qa = {}
    for js_path in gt_json_paths:
        with open(js_path, 'r') as f:
            gt_data = json.load(f)

        # 原始格式：用于分割评估，不是问答
        idx = js_path.split('.')[0].split('/')[-1]
        print(idx)
        gt_qa[f'{idx}'] = gt_data
    return gt_qa, img_paths

def smooth_cuda(mask_pred:torch.Tensor):
    scale = 7
    avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=3, count_include_pad=False).to(mask_pred.device)
    avg_filtered = avg_pool(mask_pred.float().unsqueeze(0).unsqueeze(0))
    mask = (avg_filtered > 0.5).type(torch.uint8).squeeze(0).squeeze(0)
    return mask

def segmentation_process_cuda(sem_map:torch.tensor, clip_model, thresh, img_ann, prompts, output_path=None, frame_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across_quick(sem_map)
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    iou_all = {}
    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head).to(device)
        mask_lvl = torch.zeros((n_head, h, w)).to(device)
        for i in range(n_head):
            scale = 29
            avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output > thresh).type(torch.uint8)
            mask_pred = smooth_cuda(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = torch.from_numpy(img_ann[prompts[k]]['mask'].astype(np.uint8)).to(device)

            # calculate iou
            intersection = torch.sum(torch.logical_and(mask_gt, mask_pred))
            union = torch.sum(torch.logical_or(mask_gt, mask_pred))
            iou = torch.sum(intersection) / torch.sum(union)
            iou_lvl[i] = iou

        iou_all[prompts[k]] = iou_lvl.tolist()
        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        # Save prediction result
        if output_path is not None and frame_idx is not None:
            pred_mask = mask_lvl[chosen_lvl].cpu().numpy().astype(np.uint8)
            save_path = Path(output_path) / 'pred' / f'frame_{frame_idx:0>5}' / f'{prompts[k]}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(pred_mask, save_path)

        chosen_iou_list.append(iou_lvl[chosen_lvl].cpu().numpy().item())
        chosen_lvl_list.append(chosen_lvl.cpu().numpy().item())

    return chosen_iou_list, chosen_lvl_list

def localization_process_cuda(sem_map:torch.tensor, clip_model, img_ann):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across_quick(sem_map)
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    select_level, scores_all = {}, {}
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(n_prompt):
        select_output = valid_map[:, k]
        scale = 29
        avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
        avg_filtered = avg_pool(select_output.unsqueeze(1)).squeeze(1)
        
        score_lvl = torch.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[i].max()
            coord = torch.nonzero((avg_filtered[i] == score).type(torch.uint8))
            score_lvl[i] = score
            coord_lvl.append(coord)

        selec_head = torch.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        scores_all[positives[k]] = score_lvl.tolist()
        select_level[positives[k]] = selec_head.item()
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[1] >= x_min and cord_list[1] <= x_max and 
                    cord_list[0] >= y_min and cord_list[0] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

    return acc_num

def render_language_feature_map(gaussians:GaussianModel, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        language_feature_map = gaussians.compute_final_feature_map(language_feature_weight_map) #[1152, H, W]

    return language_feature_map

def render_language_feature_map_quick(gaussians:GaussianModel, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(3, 64, H, W).view(3, 64, H*W)
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        feature_dim = gaussians._language_feature_codebooks.shape[2]
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(3, feature_dim, H, W)
        # language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)

    return language_feature_map


def load_checkpoint_compat(checkpoint_path):
    """Load checkpoints across PyTorch versions where weights_only default changed."""
    try:
        return torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path)


def render_all(dataset:ModelParams, pipeline:PipelineParams, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # load test data
    gt_ann, image_paths = eval_gt_lerfdata(Path(args.json_folder), Path(args.output_path))
    eval_index_list = [idx for idx in list(gt_ann.keys())]

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0

    for i, idx in enumerate(tqdm(eval_index_list)):
      
      
        image_path = os.path.join(args.json_folder, f'{idx}.jpg')
        rgb_img = cv2.imread(image_path)[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)
        image_name = Path(args.output_path) / f'{idx}'
        image_name.mkdir(exist_ok=True, parents=True)
        img_ann = gt_ann[f'{idx}']
        # 直接使用-m传入的model_path（不循环，不hardcode）
        # dataset.model_path已经通过-m参数设置
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        views = scene.getTrainCameras()

        # 找到匹配的view
        view = None
        for v in views:
            print(v, idx)
            if v.image_name == idx.split('/')[-1]:
                view = v
                break
        if view == None:
            raise ValueError(f'View not found for {idx}')

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 使用-m传入的model_path加载checkpoint
        checkpoint = os.path.join(dataset.model_path, f'chkpnt{args.checkpoint}.pth')
        (model_params, first_iter) = load_checkpoint_compat(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        language_feature_image = render_language_feature_map(gaussians, view, pipeline, background, args)
        language_feature_image = language_feature_image.detach()  # [3584, H, W]
        language_feature_image = language_feature_image.permute(1, 2, 0)  # [H, W, 3584]

        # 保持4D格式 [1, H, W, 3584] 以兼容现有代码
        restored_feat = language_feature_image.unsqueeze(0)
        img_ann = gt_ann[f'{idx}']

        # save feature map
        rendered_feature_path = image_name / f'feature_map_{idx}.pt'
        torch.save(restored_feat.cpu(), rendered_feature_path)

        # 特征对比可视化（如果启用）
        if args.visualize_comparison and hasattr(args, 'gt_feature_dir') and args.gt_feature_dir:
            try:
                # 构建 GT 特征路径（使用idx而不是idx+1）
                gt_feature_path = Path(args.gt_feature_dir) / f'{idx}.pth'
                if gt_feature_path.exists():
                    # 使用正确的图像路径（与前面一致）
                    original_image_path = image_path

                    # 输出路径
                    comparison_output_path = image_name / f'{idx}_comparison.png'

                    logger.info(f"Generating comparison visualization for frame {idx}...")

                    # 调用可视化函数
                    visualize_comparison(
                        rendered_path=str(rendered_feature_path),
                        gt_path=str(gt_feature_path),
                        image_path=original_image_path,
                        output_path=str(comparison_output_path),
                        scale_name=args.comparison_scale
                    )
                else:
                    logger.warning(f"GT feature not found: {gt_feature_path}")
            except Exception as e:
                logger.error(f"Error generating comparison for frame {idx}: {e}")
                import traceback
                traceback.print_exc()

    return

def render_all_quick(dataset:ModelParams, pipeline:PipelineParams, args):
  #TODO: not implemented yet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # load test data
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(args.json_folder), Path(args.output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0

    for i, idx in enumerate[int](tqdm(eval_index_list)):
        # 构建正确的图片路径（使用idx而不是i）
        image_path = os.path.join(args.json_folder, f'frame_{idx:05d}.jpg')
        rgb_img = cv2.imread(image_path)[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        image_name = Path(args.output_path) / f'frame_{idx:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = []
        language_feature_weights = []
        language_feature_indices = []
        language_feature_codebooks = []
        combined_gaussians = GaussianModel(dataset.sh_degree)
        dataset.model_path = args.ckpt_paths[0]
        scene = Scene(dataset, combined_gaussians, shuffle=False)
        views = scene.getTrainCameras()
        # views是0-indexed数组，但idx是帧号（1-based），所以需要减1
        view = views[idx - 1]
        checkpoint = os.path.join(args.ckpt_paths[0], f'chkpnt{args.checkpoint}.pth')
        (model_params, first_iter) = load_checkpoint_compat(checkpoint)
        combined_gaussians.restore(model_params, args, mode='test')
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        for level_idx in range(3):
            # restore gaussian model
            gaussians = GaussianModel(dataset.sh_degree)
            checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
            (model_params, first_iter) = load_checkpoint_compat(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            feature_dim = gaussians._language_feature_codebooks.shape[2]
            language_feature_codebooks.append(gaussians._language_feature_codebooks.view(-1, feature_dim))
            weights, indices = get_weights_and_indices(gaussians._language_feature_logits, 4)
            language_feature_weights.append(weights)
            language_feature_indices.append(indices + int(level_idx * gaussians._language_feature_codebooks.shape[1]))
        language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
        language_feature_weights = torch.cat(language_feature_weights, dim=1)
        language_feature_indices = torch.cat(language_feature_indices, dim=1)
        combined_gaussians._language_feature_codebooks = language_feature_codebooks
        combined_gaussians._language_feature_weights = language_feature_weights
        combined_gaussians._language_feature_indices = torch.from_numpy(language_feature_indices.detach().cpu().numpy()).to(combined_gaussians._language_feature_weights.device)
        
        language_feature_image = render_language_feature_map_quick(combined_gaussians, view, pipeline, background, args)
        restored_feat = language_feature_image.permute(0, 2, 3, 1)
        c_iou_list, c_lvl = segmentation_process_cuda(restored_feat, clip_model, args.mask_thresh, img_ann, list(img_ann.keys()), args.output_path, idx)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)
        acc_num_img = localization_process_cuda(restored_feat, clip_model, img_ann)
        acc_num += acc_num_img

    logger.info(f'checkpoint: {args.checkpoint}')
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {args.mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')

    return

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    #------------------------------------------------------------
    # arguments for gaussian model
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    #------------------------------------------------------------
    #------------------------------------------------------------
    # arguments for evaluation and output
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=1)
    #------------------------------------------------------------
    # 新增：特征对比可视化参数
    parser.add_argument("--visualize_comparison", action="store_true",
                       help="Enable feature comparison visualization")
    parser.add_argument("--gt_feature_dir", type=str, default=None,
                       help="Directory containing GT features (.pth files)")
    parser.add_argument("--comparison_scale", type=str, default="Medium",
                       choices=["Small", "Medium", "Large"],
                       help="GT feature scale to use for comparison")
    #------------------------------------------------------------

    args = get_combined_args(parser)

    # Note: 现在render_lerf_llm.py直接使用-m传入的单个model_path，不再需要ckpt_paths列表
    # 保留这一行是为了向后兼容，但实际不再使用
    args.ckpt_paths = [os.path.join(args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}") for level in [0, 1, 2]]
    # 新目录结构：output_dir已包含实验索引和level信息，只需加dataset_name
    # 例如：eval_result/{INDEX}/level{0,1,2}/{dataset_name}/
    args.output_path = os.path.join(args.output_dir, args.dataset_name)
    args.json_folder = os.path.join(args.json_folder, args.dataset_name)
    
    os.makedirs(args.output_path, exist_ok=True)
    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(args.output_path, exist_ok=True)
    log_file = os.path.join(args.output_path, f'{timestamp}.log')
    logger = get_logger(f'{args.dataset_name}', log_file=log_file, log_level=logging.INFO)
    
    safe_state(args.quiet)
    print(args)
    with torch.no_grad():
        if args.quick_render:
            render_all_quick(model.extract(args), pipeline.extract(args), args)
        else:
            render_all(model.extract(args), pipeline.extract(args), args)
    print("EVAL COMPLETE")
