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
import time
from pathlib import Path

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _read_compiled_language_feature_channels(default=128):
    """Return the rasterizer capacity from the checked-out CUDA config.

    The extension uses this value as a compile-time constant. Cache it at module
    import so a long-running process remains consistent even if another Exp10
    job recompiles the extension for a different K*L after this process has
    already imported diff_gaussian_rasterization._C.
    """
    config_path = (
        Path(__file__).resolve().parents[1]
        / "submodules"
        / "efficient-langsplat-rasterization"
        / "cuda_rasterizer"
        / "config.h"
    )
    try:
        for line in config_path.read_text().splitlines():
            if "NUM_CHANNELS_language_feature" in line and line.lstrip().startswith("#define"):
                parts = line.split()
                return int(parts[2])
    except Exception:
        pass
    return default


_COMPILED_LANGUAGE_FEATURE_CHANNELS = _read_compiled_language_feature_channels()


def _resolve_raster_feature_channels(opt):
    requested = getattr(opt, "raster_feature_channels", -1)
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = -1
    if requested > 0:
        return requested
    return _COMPILED_LANGUAGE_FEATURE_CHANNELS

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """ 
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=opt.include_feature,
        quick_render=opt.quick_render
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if opt.quick_render:
        assert pc._language_feature_weights is not None and pc._language_feature_indices is not None, "None Value Error"
        language_feature_weights_quick = pc._language_feature_weights
        # language_feature_indices = torch.from_numpy(pc._language_feature_indices.detach().cpu().numpy()).to(pc._language_feature_weights.device)
        language_feature_indices = pc._language_feature_indices
        # print(language_feature_indices)
        language_feature_weights = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)

    elif opt.include_feature:
        language_feature_weights = pc.get_render_weights(opt.topk) # [N, L*K]
        # The CUDA rasterizer is compiled with a fixed language-feature channel
        # count (currently 128). K/L ablations can produce fewer channels
        # (e.g. L=1,K=64); pad before rasterization so the kernel never reads
        # past the input tensor. Downstream code slices back to the active
        # codebook channels, so padded channels carry zero signal/gradient.
        raster_feature_channels = _resolve_raster_feature_channels(opt)
        if language_feature_weights.shape[1] > raster_feature_channels:
            raise ValueError(
                f"language feature channels {language_feature_weights.shape[1]} exceed "
                f"compiled rasterizer capacity {raster_feature_channels}."
            )
        if language_feature_weights.shape[1] < raster_feature_channels:
            pad = torch.zeros(
                (language_feature_weights.shape[0], raster_feature_channels - language_feature_weights.shape[1]),
                dtype=language_feature_weights.dtype,
                device=language_feature_weights.device,
            )
            language_feature_weights = torch.cat([language_feature_weights, pad], dim=1)
        language_feature_weights_quick = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        language_feature_indices = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
    
    else:
        language_feature_weights = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        language_feature_weights_quick = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        language_feature_indices = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # start_time = time.time()

    rendered_image, language_feature_weight_map, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        language_feature_precomp = language_feature_weights,
        language_feature_weights_quick = language_feature_weights_quick,
        language_feature_indices = language_feature_indices,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # end_time = time.time()
    # print('render_init_rasterizer程序运行时间为: %s Seconds'%(end_time-start_time))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    return {"render": rendered_image, #[3, w, h]
            "language_feature_weight_map": language_feature_weight_map, #[64, w, h]
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
