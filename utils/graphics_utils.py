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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# ============================================================
# 3D-to-2D 投影与可见性判断工具函数
#
# 重要: 这些函数使用与 CUDA 光栅化器相同的矩阵约定。
# CUDA 的 transformPoint4x4 以行主序读取 float[16] 矩阵:
#   x' = M[0]*x + M[4]*y + M[8]*z + M[12]
#   y' = M[1]*x + M[5]*y + M[9]*z + M[13]
#   z' = M[2]*x + M[6]*y + M[10]*z + M[14]
#   w' = M[3]*x + M[7]*y + M[11]*z + M[15]
# Camera 的 full_proj_transform 和 world_view_transform 是列主序 PyTorch 张量，
# 但其底层 float[16] 内存布局恰好等于 CUDA 行主序读取所需的布局。
# 因此我们不能使用 PyTorch 的 @ 或 matmul 做列向量乘法，而必须使用
# 与 CUDA 相同的逐元素索引模式。
#
# CUDA in_frustum 条件: p_view.z <= 0.2 (点在相机前方至少 0.2 个单位)
# 透视除法使用 1/w （非 w）。
# ============================================================

def _cuda_transform_4x4(points_3d, matrix):
    """
    CUDA 兼容的 4x4 变换（对应 CUDA transformPoint4x4）。
    使用行主序索引读取 matrix 的 float[16] 内存布局。

    Args:
        points_3d: [N, 3] torch.Tensor
        matrix: [4, 4] torch.Tensor，Camera 的 full_proj_transform（列主序存储，
                但函数以 CUDA 行主序方式读取其底层 float[16]）。

    Returns:
        result: [N, 4] torch.Tensor，变换后的齐次坐标。
    """
    x = matrix[0, 0] * points_3d[:, 0] + matrix[1, 0] * points_3d[:, 1]         + matrix[2, 0] * points_3d[:, 2] + matrix[3, 0]
    y = matrix[0, 1] * points_3d[:, 0] + matrix[1, 1] * points_3d[:, 1]         + matrix[2, 1] * points_3d[:, 2] + matrix[3, 1]
    z = matrix[0, 2] * points_3d[:, 0] + matrix[1, 2] * points_3d[:, 1]         + matrix[2, 2] * points_3d[:, 2] + matrix[3, 2]
    w = matrix[0, 3] * points_3d[:, 0] + matrix[1, 3] * points_3d[:, 1]         + matrix[2, 3] * points_3d[:, 2] + matrix[3, 3]
    return torch.stack([x, y, z, w], dim=1)


def _cuda_transform_4x3(points_3d, matrix):
    """
    CUDA 兼容的 4x3 变换（对应 CUDA transformPoint4x3，忽略第 4 行）。
    用于计算 view-space 坐标。

    Args:
        points_3d: [N, 3] torch.Tensor
        matrix: [4, 4] torch.Tensor，Camera 的 world_view_transform。

    Returns:
        result: [N, 3] torch.Tensor，变换后的 3D 坐标。
    """
    x = matrix[0, 0] * points_3d[:, 0] + matrix[1, 0] * points_3d[:, 1]         + matrix[2, 0] * points_3d[:, 2] + matrix[3, 0]
    y = matrix[0, 1] * points_3d[:, 0] + matrix[1, 1] * points_3d[:, 1]         + matrix[2, 1] * points_3d[:, 2] + matrix[3, 1]
    z = matrix[0, 2] * points_3d[:, 0] + matrix[1, 2] * points_3d[:, 1]         + matrix[2, 2] * points_3d[:, 2] + matrix[3, 2]
    return torch.stack([x, y, z], dim=1)


def world_to_clip(points_3d, full_proj_transform):
    """
    将 3D 世界坐标点转换到 clip 空间（CUDA 兼容语义）。

    Args:
        points_3d: [N, 3] torch.Tensor，世界坐标系下的 3D 点。
        full_proj_transform: [4, 4] torch.Tensor，camera.full_proj_transform。

    Returns:
        clip_pts: [N, 4] torch.Tensor，clip 空间齐次坐标。
    """
    return _cuda_transform_4x4(points_3d, full_proj_transform)


def world_to_view(points_3d, world_view_transform):
    """
    将 3D 世界坐标点转换到 view（相机）空间（CUDA 兼容语义）。

    Args:
        points_3d: [N, 3] torch.Tensor，世界坐标系下的 3D 点。
        world_view_transform: [4, 4] torch.Tensor，camera.world_view_transform。

    Returns:
        view_pts: [N, 3] torch.Tensor，view 空间坐标。
    """
    return _cuda_transform_4x3(points_3d, world_view_transform)


def clip_to_ndc_cuda(clip_pts):
    """
    CUDA 风格的透视除法，但使用 1/|w| 修正 z_sign=1.0 的符号问题。

    原始 CUDA rasterizer 使用 1/w:
        float p_w = 1.0f / (p_hom.w + 0.0000001f);
    但由于 getProjectionMatrix 中 z_sign=1.0，clip.w = z_view（前方点为负值），
    直接 1/w 会得到符号错误的 p_proj（右侧点映射为负 ndc_x，左侧点映射为正 ndc_x）。
    CUDA rasterizer 中所有高斯被一致镜像，渲染结果仍自洽，但独立的投影函数
    需要正确的几何映射，因此使用 1/|w| 修正符号。

    Args:
        clip_pts: [N, 4] torch.Tensor，clip 空间齐次坐标。

    Returns:
        p_proj: [N, 3] torch.Tensor，透视除法后的投影坐标。
        与 GT 小孔成像的误差 < 1 像素（来自 ndc2Pix 的 -0.5 子像素偏移）。
    """
    p_w = 1.0 / (clip_pts[:, 3:4].abs() + 1e-7)  # [N, 1]，使用 |w| 修正符号
    return clip_pts[:, :3] * p_w  # [N, 3]


def p_proj_to_pixel(p_proj, image_width, image_height):
    """
    将透视除法后的投影坐标转换为图像像素坐标。

    等价于 CUDA forward.cu 中的 ndc2Pix 变换:
        ndc2Pix(v, S) = ((v + 1.0) * S - 1.0) * 0.5
    展开: pixel = (v + 1.0) * 0.5 * S - 0.5
    x 和 y 轴使用完全相同的公式（CUDA rasterizer 不翻转 y 轴）。

    注: 公式中的 -0.5 是 CUDA 的像素边界对齐约定，与标准小孔成像
    （像素中心对齐）相差 0.5 像素。如需对齐小孔成像坐标系，
    可在返回值上加 0.5: p_proj_to_pixel(...) + 0.5。

    Args:
        p_proj: [N, 3] torch.Tensor，透视除法后的投影坐标。
        image_width: int，图像宽度。
        image_height: int，图像高度。

    Returns:
        pixel_xy: [N, 2] torch.Tensor，像素坐标 (x, y)。
    """
    ndc_x = p_proj[:, 0]
    ndc_y = p_proj[:, 1]
    # ndc2Pix: ((v + 1.0) * S - 1.0) * 0.5 = (v + 1.0) * 0.5 * S - 0.5
    pixel_x = (ndc_x + 1.0) * 0.5 * image_width - 0.5
    pixel_y = (ndc_y + 1.0) * 0.5 * image_height - 0.5
    return torch.stack([pixel_x, pixel_y], dim=1)

def project_points_to_screen(points_3d, full_proj_transform, image_width, image_height):
    """
    将 3D 世界坐标点投影到屏幕坐标（CUDA 兼容管线）。

    管线: world -> clip (transformPoint4x4) -> 1/|w| divide -> p_proj -> pixel

    Args:
        points_3d: [N, 3] torch.Tensor。
        full_proj_transform: [4, 4] torch.Tensor，camera.full_proj_transform。
        image_width: int。
        image_height: int。

    Returns:
        pixel_xy: [N, 2] torch.Tensor，像素坐标。
        p_proj: [N, 3] torch.Tensor，透视除法后的投影坐标。
    """
    clip_pts = world_to_clip(points_3d, full_proj_transform)
    p_proj = clip_to_ndc_cuda(clip_pts)
    pixel_xy = p_proj_to_pixel(p_proj, image_width, image_height)
    return pixel_xy, p_proj


def check_points_in_view(points_3d, world_view_transform, margin=0.0):
    """
    判断 3D 点是否在相机前方（与 CUDA in_frustum 一致）。

    CUDA 条件: p_view.z <= 0.2
    这里允许通过 margin 参数调整阈值（默认与 CUDA 一致为 0.2，
    使用 margin 参数名以避免与图像边界 margin 混淆）。

    Args:
        points_3d: [N, 3] torch.Tensor，世界坐标 3D 点。
        world_view_transform: [4, 4] torch.Tensor，camera.world_view_transform。
        margin: float，相机前方阈值（默认 0.2，与 CUDA 一致）。
              负值允许点在相机后方仍算"可见"。

    Returns:
        in_view: [N] bool torch.Tensor，True 表示点在相机前方。
        p_view: [N, 3] torch.Tensor，view-space 坐标。
    """
    p_view = world_to_view(points_3d, world_view_transform)
    # CUDA in_frustum: p_view.z <= 0.2
    # 在 view space 中，相机前方 z 为负，所以 z <= 0.2 包含所有前方点
    in_view = p_view[:, 2] <= max(0.2, margin)
    return in_view, p_view


def check_visibility(points_3d, full_proj_transform, world_view_transform,
                     image_width, image_height, margin=0.0, z_threshold=0.2):
    """
    综合判断 3D 点是否从当前相机视角可见。

    可见条件（与 CUDA rasterizer 一致）:
        1. p_view.z <= z_threshold（点在相机前方，默认 0.2）

    额外的屏幕边界检查（CUDA rasterizer 不检查此条件）:
        2. 投影后的像素坐标在 [0, W) x [0, H) 范围内

    Args:
        points_3d: [N, 3] torch.Tensor。
        full_proj_transform: [4, 4] torch.Tensor。
        world_view_transform: [4, 4] torch.Tensor。
        image_width: int。
        image_height: int。
        margin: float，边界裕量（像素），传递给边界检查。
        z_threshold: float，view-space z 阈值（默认 0.2，与 CUDA 一致）。

    Returns:
        visible: [N] bool torch.Tensor。
        pixel_xy: [N, 2] torch.Tensor。
        p_proj: [N, 3] torch.Tensor。
        p_view: [N, 3] torch.Tensor。
    """
    # View-space check (CUDA in_frustum)
    p_view = world_to_view(points_3d, world_view_transform)
    in_frustum_cuda = p_view[:, 2] <= z_threshold

    # Project to screen
    clip_pts = world_to_clip(points_3d, full_proj_transform)
    p_proj = clip_to_ndc_cuda(clip_pts)
    pixel_xy = p_proj_to_pixel(p_proj, image_width, image_height)

    # Screen bounds check
    on_screen = (
        (pixel_xy[:, 0] >= -margin) & (pixel_xy[:, 0] < image_width + margin) &
        (pixel_xy[:, 1] >= -margin) & (pixel_xy[:, 1] < image_height + margin)
    )

    visible = in_frustum_cuda & on_screen
    return visible, pixel_xy, p_proj, p_view
