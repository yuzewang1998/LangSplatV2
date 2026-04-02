#!/usr/bin/env python3
"""
使用 Ollama VLM 模型过滤图像数据集
根据图像质量标准（高清、无遮挡物、天气晴朗、无滤镜、无模糊）筛选图片
"""

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm


class OllamaVLMFilter:
    """使用 Ollama VLM 进行图像质量过滤"""

    def __init__(
        self,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        model_name: str = "llava:7b",
        landmark_name: str = "Brandenburg Gate"
    ):
        """
        初始化 Ollama VLM 过滤器

        Args:
            ollama_host: Ollama 服务地址
            ollama_port: Ollama 服务端口
            model_name: 使用的 VLM 模型名称
            landmark_name: 地标名称（用于 prompt）
        """
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/generate"
        self.model_name = model_name
        self.landmark_name = landmark_name

        # 构建质量检查 prompt
        self.quality_prompt = self._build_quality_prompt()

    def _build_quality_prompt(self) -> str:
        """构建图像质量检查的 prompt"""
        prompt = f"""Look at this image of {self.landmark_name}.

Answer "Yes" ONLY if ALL of these conditions are absolutely true:
- NO human/person/tourist/pedestrian anywhere in the image (not even tiny, distant, or partially visible)
- NO vehicles of any kind (cars, buses, bikes, motorcycles)
- NO flags, banners, signs, posters, or advertisements
- NO construction equipment, scaffolding, or barriers
- NO temporary objects or obstructions
- Clear and sharp image quality

CRITICAL: If you see ANY person, even as a small dot in the distance, answer "No".
CRITICAL: If you see ANY flag or banner, answer "No".
CRITICAL: If you have ANY doubt about any object, answer "No".

Answer only "Yes" or "No":"""
        return prompt

    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def check_image_quality(self, image_path: str) -> Tuple[bool, str]:
        """
        使用 VLM 检查图像质量

        Args:
            image_path: 图像文件路径

        Returns:
            (是否通过质量检查, 模型回复)
        """
        try:
            # 编码图像
            image_base64 = self.encode_image(image_path)

            # 构建请求
            payload = {
                "model": self.model_name,
                "prompt": self.quality_prompt,
                "images": [image_base64],
                "stream": False
            }

            # 调用 Ollama API
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            answer = result.get("response", "").strip()

            # 判断是否通过（检查回答中是否包含 Yes）
            passed = "yes" in answer.lower()

            return passed, answer

        except Exception as e:
            print(f"错误：处理图像 {image_path} 时出现异常: {e}")
            return False, str(e)

    def filter_dataset(
        self,
        tsv_path: str,
        output_path: str,
        image_dir: str,
        resume_from: int = 0
    ) -> Dict[str, int]:
        """
        过滤整个数据集

        Args:
            tsv_path: 输入 TSV 文件路径
            output_path: 输出 TSV 文件路径
            image_dir: 图像文件所在目录
            resume_from: 从第几行开始处理（用于断点续传）

        Returns:
            统计信息字典
        """
        # 读取输入 TSV
        with open(tsv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        # 准备输出文件
        output_mode = 'a' if resume_from > 0 else 'w'
        output_file = open(output_path, output_mode, newline='')
        writer = csv.DictWriter(
            output_file,
            fieldnames=['filename', 'id', 'split', 'dataset'],
            delimiter='\t'
        )

        # 如果是新文件，写入表头
        if resume_from == 0:
            writer.writeheader()

        # 统计信息
        stats = {
            'total': len(rows),
            'processed': resume_from,
            'passed': 0,
            'failed': 0,
            'error': 0
        }

        # 如果从中间恢复，先统计已处理的通过数量
        if resume_from > 0:
            with open(output_path, 'r') as f:
                stats['passed'] = sum(1 for _ in f) - 1  # 减去表头

        try:
            # 遍历所有图像
            for idx, row in enumerate(tqdm(rows[resume_from:], initial=resume_from, total=len(rows))):
                filename = row['filename']
                image_path = os.path.join(image_dir, filename)

                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    print(f"警告：图像文件不存在: {image_path}")
                    stats['error'] += 1
                    continue

                # 检查质量
                passed, answer = self.check_image_quality(image_path)

                stats['processed'] += 1

                if passed:
                    # 通过质量检查，写入输出文件
                    writer.writerow(row)
                    output_file.flush()  # 立即写入磁盘
                    stats['passed'] += 1
                    tqdm.write(f"✓ {filename}: {answer}")
                else:
                    stats['failed'] += 1
                    tqdm.write(f"✗ {filename}: {answer}")

                # 每处理 10 张图片打印一次统计信息
                if (idx + 1) % 10 == 0:
                    pass_rate = stats['passed'] / stats['processed'] * 100
                    tqdm.write(f"当前进度: {stats['processed']}/{stats['total']}, "
                              f"通过率: {pass_rate:.1f}% ({stats['passed']}/{stats['processed']})")

        finally:
            output_file.close()

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="使用 Ollama VLM 模型过滤图像数据集"
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        required=True,
        help="输入 TSV 文件路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 TSV 文件路径"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图像文件所在目录"
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
    parser.add_argument(
        "--resume_from",
        type=int,
        default=0,
        help="从第几行开始处理（用于断点续传）"
    )

    args = parser.parse_args()

    # 创建过滤器
    filter_obj = OllamaVLMFilter(
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        model_name=args.model_name,
        landmark_name=args.landmark_name
    )

    print(f"开始过滤数据集...")
    print(f"输入文件: {args.tsv_path}")
    print(f"输出文件: {args.output_path}")
    print(f"图像目录: {args.image_dir}")
    print(f"Ollama 服务: {args.ollama_host}:{args.ollama_port}")
    print(f"模型: {args.model_name}")
    print(f"地标: {args.landmark_name}")
    print("-" * 80)

    # 执行过滤
    stats = filter_obj.filter_dataset(
        tsv_path=args.tsv_path,
        output_path=args.output_path,
        image_dir=args.image_dir,
        resume_from=args.resume_from
    )

    # 打印统计信息
    print("\n" + "=" * 80)
    print("过滤完成！")
    print(f"总计: {stats['total']} 张图片")
    print(f"已处理: {stats['processed']} 张")
    print(f"通过: {stats['passed']} 张 ({stats['passed']/stats['processed']*100:.1f}%)")
    print(f"未通过: {stats['failed']} 张 ({stats['failed']/stats['processed']*100:.1f}%)")
    if stats['error'] > 0:
        print(f"错误: {stats['error']} 张")
    print("=" * 80)


if __name__ == "__main__":
    main()
