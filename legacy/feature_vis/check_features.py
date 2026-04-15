#!/usr/bin/env python3
"""检查特征文件格式"""
import torch
import sys

# 检查 GT 特征格式
gt_path = '/home/wangyz/data/lerf_ovs/teatime/llava_features_3584_multiscale/frame_00002.pth'
print("=" * 80)
print("GT特征详细格式:")
print("=" * 80)
gt_data = torch.load(gt_path, map_location='cpu')

if isinstance(gt_data, dict):
    print("顶层Keys:", list(gt_data.keys()))
    print(f"\nimage_path: {gt_data.get('image_path', 'N/A')}")
    print(f"overlap_mode: {gt_data.get('overlap_mode', 'N/A')}")

    if 'feature_maps' in gt_data:
        print("\nfeature_maps keys:", list(gt_data['feature_maps'].keys()))
        for scale_name, scale_data in gt_data['feature_maps'].items():
            print(f"\n{scale_name}:")
            for k, v in scale_data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"  {k}: list of {len(v)} items")
                    if len(v) > 0 and isinstance(v[0], dict):
                        print(f"    First item keys: {list(v[0].keys())}")
                        for kk, vv in v[0].items():
                            if isinstance(vv, torch.Tensor):
                                print(f"      {kk}: {vv.shape}, dtype={vv.dtype}")
                            elif isinstance(vv, (list, tuple)):
                                print(f"      {kk}: {type(vv).__name__} of length {len(vv)}")
                            else:
                                print(f"      {kk}: {type(vv).__name__}")
                else:
                    print(f"  {k}: {type(v).__name__}")

    if 'crops_info' in gt_data:
        crops_info = gt_data['crops_info']
        print(f"\ncrops_info: {type(crops_info).__name__}")
        if isinstance(crops_info, dict):
            print(f"  Keys: {list(crops_info.keys())}")

print("\n" + "=" * 80)
print("渲染特征格式:")
print("=" * 80)
rendered_path = '/home/wangyz/project/0working/LangSplatV2/eval_result/teatime_test/frame_00003/feature_map_frame_00003.pt'
rendered_data = torch.load(rendered_path, map_location='cpu')
print(f"Type: {type(rendered_data)}")
if isinstance(rendered_data, torch.Tensor):
    print(f"Shape: {rendered_data.shape}")  # [1, H, W, 3584]
    print(f"Dtype: {rendered_data.dtype}")
    print(f"Min: {rendered_data.min():.4f}, Max: {rendered_data.max():.4f}")
    print(f"Mean: {rendered_data.mean():.4f}, Std: {rendered_data.std():.4f}")
