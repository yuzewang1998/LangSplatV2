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
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


class OllamaVLMFilter:
    """使用 Ollama VLM 进行图像质量过滤"""

    def __init__(
        self,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        model_name: str = "llava:7b",
        landmark_name: str = "Brandenburg Gate",
        request_timeout: int = 180,
    ):
        """
        初始化 Ollama VLM 过滤器

        Args:
            ollama_host: Ollama 服务地址
            ollama_port: Ollama 服务端口
            model_name: 使用的 VLM 模型名称
            landmark_name: 地标名称（用于 prompt）
            request_timeout: 单次请求超时时间（秒）
        """
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/generate"
        self.model_name = model_name
        self.landmark_name = landmark_name
        self.request_timeout = request_timeout

        # 构建质量检查 prompt
        self.quality_prompt = self._build_quality_prompt()

    def _build_quality_prompt(self) -> str:
        """构建图像质量检查的 prompt"""
        prompt = f"""You are filtering training images of {self.landmark_name}.

Be extremely strict. Reject the image if there is ANY doubt.

We only want images that satisfy ALL rules:
1. No real people at all: no human, tourist, pedestrian, crowd, body part, reflection of a person, or tiny distant person.
   Permanent sculptures, reliefs, carved figures, or statues that are part of the landmark architecture are allowed.
   Specifically for Brandenburg Gate, the Quadriga statue on top of the gate is allowed and must NOT be counted as a real person, horse, vehicle, or obstruction.
2. No vehicles at all: no car, bus, truck, bike, bicycle, motorcycle, scooter, or carriage.
3. No flags, banners, signs, posters, billboards, text overlays, watermarks, or advertisements.
4. No obstructions or temporary objects that block or distract from the landmark:
   barriers, construction equipment, scaffolding, fences, umbrellas, tents, poles, or heavy foreground clutter.
5. Natural appearance only:
   no heavy filter, no strong HDR look, no extreme saturation, no sepia/black-and-white stylization,
   no strong color cast, no unusual editing, and no obviously weird appearance.
6. Clear and sharp image quality.

Important:
- If you are unsure whether a tiny object is a person, vehicle, flag, or obstruction, mark it as present.
- Do NOT count fixed sculptures or statues that are part of the landmark itself as people.
- Do NOT count the Quadriga statue at Brandenburg Gate as a person, horse, vehicle, or obstruction.
- If the appearance looks edited, stylized, heavily color-graded, or strange, reject it.
- Prefer false negatives over false positives.

Return ONLY valid JSON with this exact schema:
{{
  "people_or_body_parts": true,
  "vehicles": false,
  "flags_banners_signs_text": false,
  "obstructions_or_foreground_clutter": false,
  "filtered_or_stylized_appearance": false,
  "sharp_and_clear": true,
  "keep": false,
  "reason": "short reason"
}}

Set keep=true ONLY when all negative categories are false AND sharp_and_clear=true."""
        return prompt

    def _extract_json_object(self, text: str) -> Optional[Dict]:
        """从模型响应中提取 JSON 对象。"""
        text = text.strip()
        if not text:
            return None

        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidates.append(text[start:end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _to_bool(value) -> bool:
        """将模型返回值稳健转换为布尔值。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return False

    def _normalize_decision(self, parsed: Dict, raw_answer: str) -> Dict:
        """标准化模型判定，并使用保守规则重新计算 keep。"""
        decision = {
            "people_or_body_parts": self._to_bool(parsed.get("people_or_body_parts")),
            "vehicles": self._to_bool(parsed.get("vehicles")),
            "flags_banners_signs_text": self._to_bool(parsed.get("flags_banners_signs_text")),
            "obstructions_or_foreground_clutter": self._to_bool(parsed.get("obstructions_or_foreground_clutter")),
            "filtered_or_stylized_appearance": self._to_bool(parsed.get("filtered_or_stylized_appearance")),
            "sharp_and_clear": self._to_bool(parsed.get("sharp_and_clear")),
            "reason": str(parsed.get("reason", "")).strip() or raw_answer.strip(),
            "raw_answer": raw_answer.strip(),
        }

        negatives = [
            decision["people_or_body_parts"],
            decision["vehicles"],
            decision["flags_banners_signs_text"],
            decision["obstructions_or_foreground_clutter"],
            decision["filtered_or_stylized_appearance"],
        ]
        decision["keep"] = (not any(negatives)) and decision["sharp_and_clear"]
        return decision

    @staticmethod
    def _format_decision(decision: Dict) -> str:
        """将结构化判定整理为简短字符串。"""
        issues = []
        if decision.get("people_or_body_parts"):
            issues.append("people")
        if decision.get("vehicles"):
            issues.append("vehicles")
        if decision.get("flags_banners_signs_text"):
            issues.append("flags/signs/text")
        if decision.get("obstructions_or_foreground_clutter"):
            issues.append("obstructions")
        if decision.get("filtered_or_stylized_appearance"):
            issues.append("stylized")
        if not decision.get("sharp_and_clear", False):
            issues.append("not_sharp")

        issue_text = ", ".join(issues) if issues else "clean"
        reason = decision.get("reason", "")
        return f"keep={decision.get('keep', False)} | issues={issue_text} | reason={reason}"

    def encode_image(self, image_path: str) -> str:
        """将图像编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def check_image_quality(self, image_path: str) -> Tuple[bool, str, Dict]:
        """
        使用 VLM 检查图像质量

        Args:
            image_path: 图像文件路径

        Returns:
            (是否通过质量检查, 可读回复, 结构化判定)
        """
        try:
            # 编码图像
            image_base64 = self.encode_image(image_path)

            # 构建请求
            payload = {
                "model": self.model_name,
                "prompt": self.quality_prompt,
                "images": [image_base64],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0
                }
            }

            # 调用 Ollama API
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            answer = result.get("response", "").strip()

            parsed = self._extract_json_object(answer)
            if parsed is None:
                fallback_passed = "yes" in answer.lower()
                fallback_decision = {
                    "people_or_body_parts": False,
                    "vehicles": False,
                    "flags_banners_signs_text": False,
                    "obstructions_or_foreground_clutter": False,
                    "filtered_or_stylized_appearance": False,
                    "sharp_and_clear": fallback_passed,
                    "keep": fallback_passed,
                    "reason": answer,
                    "raw_answer": answer,
                }
                return fallback_passed, answer, fallback_decision

            decision = self._normalize_decision(parsed, answer)
            return decision["keep"], self._format_decision(decision), decision

        except Exception as e:
            print(f"错误：处理图像 {image_path} 时出现异常: {e}")
            error_decision = {
                "people_or_body_parts": False,
                "vehicles": False,
                "flags_banners_signs_text": False,
                "obstructions_or_foreground_clutter": False,
                "filtered_or_stylized_appearance": False,
                "sharp_and_clear": False,
                "keep": False,
                "reason": str(e),
                "raw_answer": str(e),
            }
            return False, str(e), error_decision

    def filter_dataset(
        self,
        tsv_path: str,
        output_path: str,
        image_dir: str,
        resume_from: int = 0,
        decision_log_path: Optional[str] = None,
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

        log_file = None
        if decision_log_path:
            log_mode = 'a' if resume_from > 0 else 'w'
            log_file = open(decision_log_path, log_mode, encoding='utf-8')

        # 如果从中间恢复，先统计已处理的通过数量
        if resume_from > 0:
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    existing_lines = sum(1 for _ in f)
                stats['passed'] = max(existing_lines - 1, 0)  # 减去表头
            else:
                stats['passed'] = 0

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
                passed, answer, decision = self.check_image_quality(image_path)

                stats['processed'] += 1

                if log_file is not None:
                    log_record = {
                        'filename': filename,
                        'image_path': image_path,
                        'passed': passed,
                        'decision': decision,
                    }
                    log_file.write(json.dumps(log_record, ensure_ascii=False) + '\n')
                    log_file.flush()

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
            if log_file is not None:
                log_file.close()

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
        "--request_timeout",
        type=int,
        default=180,
        help="单次 VLM 请求超时时间（秒）"
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=0,
        help="从第几行开始处理（用于断点续传）"
    )
    parser.add_argument(
        "--decision_log_path",
        type=str,
        default=None,
        help="可选：逐图保存结构化判定日志（jsonl）"
    )

    args = parser.parse_args()

    # 创建过滤器
    filter_obj = OllamaVLMFilter(
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        model_name=args.model_name,
        landmark_name=args.landmark_name,
        request_timeout=args.request_timeout
    )

    print(f"开始过滤数据集...")
    print(f"输入文件: {args.tsv_path}")
    print(f"输出文件: {args.output_path}")
    print(f"图像目录: {args.image_dir}")
    print(f"Ollama 服务: {args.ollama_host}:{args.ollama_port}")
    print(f"模型: {args.model_name}")
    print(f"地标: {args.landmark_name}")
    print(f"超时: {args.request_timeout}s")
    if args.decision_log_path:
        print(f"判定日志: {args.decision_log_path}")
    print("-" * 80)

    # 执行过滤
    stats = filter_obj.filter_dataset(
        tsv_path=args.tsv_path,
        output_path=args.output_path,
        image_dir=args.image_dir,
        resume_from=args.resume_from,
        decision_log_path=args.decision_log_path
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
