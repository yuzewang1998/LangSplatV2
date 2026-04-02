#!/usr/bin/env python3
"""
测试 Ollama VLM 图像质量检查功能
对单张或少量图片进行测试
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from filter_images_with_vlm import OllamaVLMFilter


def main():
    parser = argparse.ArgumentParser(
        description="测试 Ollama VLM 图像质量检查"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="测试图像路径"
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default="localhost",
        help="Ollama 服务地址"
    )
    parser.add_argument(
        "--ollama_port",
        type=int,
        default=11434,
        help="Ollama 服务端口"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava:7b",
        help="VLM 模型名称"
    )
    parser.add_argument(
        "--landmark_name",
        type=str,
        default="Brandenburg Gate",
        help="地标名称"
    )

    args = parser.parse_args()

    # 检查图像是否存在
    if not Path(args.image_path).exists():
        print(f"错误：图像文件不存在: {args.image_path}")
        return 1

    print(f"测试图像: {args.image_path}")
    print(f"Ollama 服务: {args.ollama_host}:{args.ollama_port}")
    print(f"模型: {args.model_name}")
    print(f"地标: {args.landmark_name}")
    print("-" * 80)

    # 创建过滤器
    filter_obj = OllamaVLMFilter(
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        model_name=args.model_name,
        landmark_name=args.landmark_name
    )

    # 显示使用的 prompt
    print("\n使用的 Prompt:")
    print("-" * 80)
    print(filter_obj.quality_prompt)
    print("-" * 80)

    # 检查图像质量
    print("\n正在检查图像质量...")
    passed, answer = filter_obj.check_image_quality(args.image_path)

    # 显示结果
    print("\n" + "=" * 80)
    print("检查结果:")
    print(f"通过: {'✓ Yes' if passed else '✗ No'}")
    print(f"模型回复: {answer}")
    print("=" * 80)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
