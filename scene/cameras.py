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
import sys
import pickle
import torch
from torch import nn
import numpy as np

# 添加项目根目录到 Python 路径，使得可以直接运行此文件进行测试
if __name__ == "__main__":
    # 获取当前文件所在目录的父目录（即项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# 本地实现的crop特征解码函数（替代已删除的CropFeatureCodec）
def _decode_crop_features_to_full_map(scale_data, overlap_mode='average'):
    """
    从crop-level存储解码为完整特征图（使用no_interp策略）

    Args:
        scale_data: {'crop_features': [...], 'image_size': (W, H)}
        overlap_mode: 'average' (遗留参数，实际使用last-write-wins)

    Returns:
        feature_map: [H, W, C] numpy array
        valid_mask: [H, W] bool numpy array
    """
    crop_features = scale_data['crop_features']
    image_size = scale_data['image_size']  # (W, H)

    H, W = image_size[1], image_size[0]

    C = crop_features[0]['feature'].shape[-1]
    feature_map = np.zeros((H, W, C), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    for crop_data in crop_features:
        feature = crop_data['feature']
        bbox = crop_data['bbox']  # (x, y, w, h)

        if torch.is_tensor(feature):
            feature_np = feature.numpy()
        else:
            feature_np = np.asarray(feature)

        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        crop_h, crop_w = int(y2 - y), int(x2 - x)

        # 如果尺寸不匹配，使用nearest neighbor resize
        if feature_np.shape[0] != crop_h or feature_np.shape[1] != crop_w:
            feature_torch = torch.from_numpy(feature_np).permute(2, 0, 1).unsqueeze(0)
            feature_torch = torch.nn.functional.interpolate(
                feature_torch, size=(crop_h, crop_w), mode='nearest'
            )
            feature_np = feature_torch.squeeze(0).permute(1, 2, 0).numpy()

        # 直接覆盖（last-write-wins）
        feature_map[y:y2, x:x2] = feature_np
        mask[y:y2, x:x2] = True

    return feature_map, mask

_CODEC_OK = True  # 使用本地实现，不再依赖外部codec

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0) #[N, 512]
            mask = mask[0:1].reshape(1, self.image_height, self.image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, self.image_height, self.image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, self.image_height, self.image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, self.image_height, self.image_width)
        else:
            raise ValueError("feature_level=", feature_level)
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
        return point_feature.cuda(), mask.cuda() # [512,512,512],[1,512,512]
    def get_llm_feature(self, language_feature_dir, feature_level):
        """
        读取并解码 3584 维 LLaVA crop-level 特征。
        新格式文件: <language_feature_dir>/<image_name>.pth
        顶层键: 'feature_maps'，每个 scale: {'crop_features': [...], 'image_size': (W,H)}
        feature_level: 0->Small, 1->Medium, 2->Large
        返回: [C,H,W] float32, [1,H,W] bool mask（均在 CUDA 上）
        """
        assert _CODEC_OK, "CropFeatureCodec not available; cannot decode LLM features"

        file_path = os.path.join(language_feature_dir, self.image_name + ".pth")
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(file_path, map_location='cpu')
        feature_maps = data["feature_maps"]

        if feature_level == 0:
            scale_name = "Small"
        elif feature_level == 1:
            scale_name = "Medium"
        elif feature_level == 2:
            scale_name = "Large"
        else:
            raise ValueError("feature_level=", feature_level)

        # 若指定尺度缺失：不回退，直接返回空掩码（训练时将跳过该帧）
        if scale_name not in feature_maps:
            H, W = int(self.image_height), int(self.image_width)
            feat = torch.zeros(3584, H, W, dtype=torch.float32, device='cuda')
            msk = torch.zeros(1, H, W, dtype=torch.bool, device='cuda')
            return feat, msk

        encoded = feature_maps[scale_name]
        feature_map, valid_mask = _decode_crop_features_to_full_map(encoded, overlap_mode='average')

        # Convert to torch tensors: [H,W,C] -> [C,H,W]; mask -> [1,H,W]
        feature_map_torch = torch.from_numpy(feature_map) if isinstance(feature_map, np.ndarray) else feature_map
        valid_mask_torch = torch.from_numpy(valid_mask) if isinstance(valid_mask, np.ndarray) else valid_mask

        point_feature = feature_map_torch.permute(2, 0, 1).contiguous()
        mask = valid_mask_torch.bool()[None, :, :]

        return point_feature.cuda(), mask.cuda()

    def get_llm_feature_tile(self, language_feature_dir, feature_level, ys, ye, xs, xe, overlap_mode='average'):
        """
        Decode a spatial tile of the 3584-dim LLaVA crop-level feature map to avoid full-image allocation.
        Returns [C, h, w] and [1, h, w] on CUDA where h=ye-ys, w=xe-xs.
        """
        assert _CODEC_OK, "CropFeatureCodec not available; cannot decode LLM features"

        file_path = os.path.join(language_feature_dir, self.image_name + ".pth")
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        except TypeError:
            data = torch.load(file_path, map_location='cpu')
        feature_maps = data["feature_maps"]

        if feature_level == 0:
            scale_name = "Small"
        elif feature_level == 1:
            scale_name = "Medium"
        elif feature_level == 2:
            scale_name = "Large"
        else:
            raise ValueError("feature_level=", feature_level)

        if scale_name not in feature_maps:
            h = max(0, int(ye) - int(ys))
            w = max(0, int(xe) - int(xs))
            empty_feat = torch.zeros(3584, h, w, dtype=torch.float32, device='cuda')
            empty_mask = torch.zeros(1, h, w, dtype=torch.bool, device='cuda')
            return empty_feat, empty_mask

        encoded = feature_maps[scale_name]

        img_w, img_h = encoded['image_size']  # (W, H)
        ys = max(0, int(ys)); xe = min(int(xe), img_w)
        xs = max(0, int(xs)); ye = min(int(ye), img_h)
        h = max(0, ye - ys); w = max(0, xe - xs)
        if h == 0 or w == 0:
            empty_feat = torch.zeros(3584, max(h, 0), max(w, 0), dtype=torch.float32, device='cuda')
            empty_mask = torch.zeros(1, max(h, 0), max(w, 0), dtype=torch.bool, device='cuda')
            return empty_feat, empty_mask

        # Accumulators on CPU to reduce GPU memory; move to CUDA at the end for loss
        tile_feature = torch.zeros(h, w, 3584, dtype=torch.float32)
        tile_count = torch.zeros(h, w, dtype=torch.int32)
        tile_mask = torch.zeros(h, w, dtype=torch.bool)

        for crop in encoded['crop_features']:
            cf = crop['feature']  # [27, 27, 3584]
            x, y, cw, ch = crop['bbox']
            x2, y2 = x + cw, y + ch

            # Compute overlap with tile
            ox1, oy1 = max(xs, x), max(ys, y)
            ox2, oy2 = min(xe, x2), min(ye, y2)
            if ox1 >= ox2 or oy1 >= oy2:
                continue

            # Resize crop feature to its bbox size once
            resized = CropFeatureCodec._resize_feature_map(cf, (ch, cw))  # [ch, cw, 3584]

            # Region within crop
            cx1, cy1 = ox1 - x, oy1 - y
            cx2, cy2 = ox2 - x, oy2 - y
            region = resized[cy1:cy2, cx1:cx2, :]  # [rh, rw, 3584]

            # Region within tile
            tx1, ty1 = ox1 - xs, oy1 - ys
            tx2, ty2 = ox2 - xs, oy2 - ys

            tile_feature[ty1:ty2, tx1:tx2, :] += region
            tile_count[ty1:ty2, tx1:tx2] += 1
            tile_mask[ty1:ty2, tx1:tx2] = True

        if overlap_mode == 'average':
            overlap = tile_count > 0
            tile_feature[overlap] = tile_feature[overlap] / tile_count[overlap].unsqueeze(-1).float()

        # [H,W,C] -> [C,H,W]
        point_feature = tile_feature.permute(2, 0, 1).contiguous()
        mask = tile_mask[None, :, :]
        return point_feature.cuda(), mask.cuda()
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

if __name__== "__main__":
    test_cam = Camera(0,
                        R = np.eye(3),
                        T = np.zeros(3),  # 修复：应该是一维数组，不是 (3,1)
                        FoVx = 60.0,
                        FoVy = 60.0,
                        image = torch.randn((3,512,512)),
                        gt_alpha_mask = None,
                        image_name = "frame_00001",
                        uid = 0
                        )
    print("Camera 对象创建成功!")
    print(f"图像尺寸: {test_cam.image_width} x {test_cam.image_height}")
    print(f"相机中心: {test_cam.camera_center}")
    # 注意：下面这行需要实际的特征目录才能运行
    # get_language_feature = test_cam.get_language_feature(language_feature_dir="/mnt/data/wangyz/lerf_ovs/teatime/language_features", feature_level=0)
    get_language_feature = test_cam.get_llm_feature(language_feature_dir="/mnt/data/wangyz/lerf_ovs/teatime/llava_features_multiscale", feature_level=0)
    
