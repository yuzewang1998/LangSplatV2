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
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests
from tqdm import tqdm


class OllamaVLMFilter:
    """使用 Ollama VLM 进行图像质量过滤"""

    SCENE_ALLOWED_TERMS = {
        "fence", "fences", "railing", "railings", "gate", "gates",
        "sculpture", "sculptures", "statue", "statues", "relief", "reliefs",
        "fountain", "fountains", "water", "basin", "basins", "obelisk", "obelisks",
        "column", "columns", "door", "doors", "window", "windows", "roof", "roofs",
        "pediment", "dome", "domes", "minaret", "minarets", "memorial",
        "shibi", "coat of arms", "oceanus", "horse", "horses",
    }

    TEMPORARY_OBSTRUCTION_TERMS = {
        "temporary", "construction", "scaffold", "scaffolding", "barrier", "barricade",
        "umbrella", "tent", "pole", "clutter", "foreground clutter",
    }

    PERMISSIVE_BENCHMARK_ALLOWED_TERMS = SCENE_ALLOWED_TERMS | {
        "people", "person", "tourist", "pedestrian", "crowd", "distant", "tiny",
        "vehicle", "vehicles", "car", "bus", "truck", "bike", "bicycle", "motorcycle",
        "flag", "flags", "banner", "banners", "sign", "signs", "text",
    }

    def __init__(
        self,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        model_name: str = "llava:7b",
        landmark_name: str = "Brandenburg Gate",
        request_timeout: int = 180,
        filter_mode: str = "strict",
    ):
        """
        初始化 Ollama VLM 过滤器

        Args:
            ollama_host: Ollama 服务地址
            ollama_port: Ollama 服务端口
            model_name: 使用的 VLM 模型名称
            landmark_name: 地标名称（用于 prompt）
            request_timeout: 单次请求超时时间（秒）
            filter_mode: strict 使用原始严过滤；benchmark_permissive 允许少量远处游客/车辆/文字，
                但仍拒绝遮挡主体、低质、强滤镜图，便于构造多场景审核 benchmark。
        """
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/generate"
        self.model_name = model_name
        self.landmark_name = landmark_name
        self.request_timeout = request_timeout
        self.filter_mode = filter_mode

        # 构建质量检查 prompt
        self.quality_prompt = self._build_quality_prompt()

    def _build_quality_prompt(self) -> str:
        """构建图像质量检查的 prompt"""
        if self.filter_mode == "benchmark_permissive":
            return self._build_benchmark_permissive_prompt()
        if self.filter_mode == "benchmark_balanced_strict":
            return self._build_benchmark_balanced_strict_prompt()
        if self.filter_mode == "benchmark_reconstruction_clean":
            return self._build_benchmark_reconstruction_clean_prompt()
        if self.filter_mode == "benchmark_reconstruction_clean_strict_appearance":
            return self._build_benchmark_reconstruction_clean_prompt(strict_appearance=True)

        prompt = f"""You are filtering training images of {self.landmark_name}.

Be extremely strict. Reject the image if there is ANY doubt.

We only want images that satisfy ALL rules:
1. No real people at all: no human, tourist, pedestrian, crowd, body part, reflection of a person, or tiny distant person.
   Permanent sculptures, reliefs, carved figures, or statues that are part of the landmark architecture are allowed.
   Specifically for Brandenburg Gate, the Quadriga statue on top of the gate is allowed and must NOT be counted as a real person, horse, vehicle, or obstruction.
2. No vehicles at all: no car, bus, truck, bike, bicycle, motorcycle, scooter, or carriage.
3. No flags, banners, signs, posters, billboards, text overlays, watermarks, or advertisements.
4. No obstructions or temporary objects that block or distract from the landmark:
   construction equipment, scaffolding, temporary barriers, umbrellas, tents, poles, or heavy foreground clutter.
   Permanent landmark elements such as decorative fences/railings, fountains, obelisks, sculptures, reliefs,
   statues, water basins, gates, roofs, columns, doors, windows, and other fixed architectural parts are allowed
   when they belong to the landmark scene and do not hide the main subject.
5. Natural appearance only:
   no heavy filter, no strong HDR look, no extreme saturation, no sepia/black-and-white stylization,
   no strong color cast, no unusual editing, and no obviously weird appearance.
6. Clear and sharp image quality.

Important:
- If you are unsure whether a tiny object is a person, vehicle, flag, or obstruction, mark it as present.
- Do NOT count fixed sculptures, reliefs, fountains, obelisks, decorative fences/railings, or statues that are
  part of the landmark itself as people, vehicles, or obstructions.
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

    def _build_benchmark_permissive_prompt(self) -> str:
        """构建 benchmark 选图用的宽松质量检查 prompt。"""
        return f"""You are selecting clean benchmark/review images of {self.landmark_name}.

Keep the image if the landmark is clearly visible, sharp, natural-looking, and useful for visual question answering.

Reject the image if:
1. The landmark or main architectural parts are heavily blocked or hard to see.
2. The image is blurry, low-resolution, overexposed/underexposed, or strongly cropped so the landmark cannot be understood.
3. The image has heavy filters, strong HDR, unusual color grading, black-and-white stylization, text overlays, watermarks, or obvious editing.
4. Foreground people, vehicles, signs, poles, umbrellas, tents, scaffolding, or construction equipment substantially distract from or cover the landmark.

Allowed:
- Small or distant tourists/vehicles are acceptable if they do not cover important landmark details.
- Fixed sculptures, statues, reliefs, decorative fences/railings, fountains, water basins, obelisks, roofs, doors, windows, columns, and other permanent landmark parts are allowed.
- Normal sky, plaza, pavement, or surrounding context is allowed.

Return ONLY valid JSON with this exact schema:
{{
  "people_or_body_parts": false,
  "vehicles": false,
  "flags_banners_signs_text": false,
  "obstructions_or_foreground_clutter": false,
  "filtered_or_stylized_appearance": false,
  "sharp_and_clear": true,
  "keep": true,
  "reason": "short reason"
}}

        Set keep=true when the landmark is clear and useful for benchmark QA, even if there are tiny non-blocking people or vehicles."""

    def _build_benchmark_balanced_strict_prompt(self) -> str:
        """构建 benchmark 重建/评测用的平衡严格 prompt。"""
        return f"""You are filtering images of {self.landmark_name} for 3D reconstruction and benchmark evaluation.

Goal: keep clean, natural, reconstruction-friendly landmark images. Be stricter than a casual VQA dataset.

Reject the image if ANY of these are true:
1. People:
   - any large, medium, foreground, central, or clearly visible person/tourist/crowd;
   - any person overlapping, touching, standing in front of, or visually covering the landmark;
   - any body part, selfie foreground, parade, queue, or dense crowd.
   Only allow tiny distant background people when they are clearly not overlapping the landmark and do not distract.
2. Vehicles:
   - any large, medium, foreground, central, or clearly visible car, bus, truck, bicycle, motorcycle, carriage, tram, or traffic;
   - any vehicle overlapping, touching, parked in front of, or visually covering the landmark.
   Only allow tiny distant vehicles when they are clearly not overlapping the landmark and do not distract.
3. Other visual clutter or occlusion:
   - scaffolding, construction, temporary barriers, tents, umbrellas, poles in front of the landmark,
     foreground signs, flags, banners, billboards, text overlays, watermarks, or heavy foreground clutter;
   - anything that hides important geometry of the gate, columns, roof, Quadriga, or main facade.
4. Extreme or inconsistent appearance:
   - heavy filter, strong HDR, oversaturation, sepia, black-and-white, strong color cast, artistic editing,
     watermark/text overlay, fisheye/distortion, collage, drawing/rendering;
   - extreme weather or atmosphere such as heavy snow, rain, fog, smoke, haze, storm, very dark night,
     dramatic artificial lighting, or severe over/under-exposure.
   Normal daylight, mild overcast sky, and ordinary shadows are allowed.
5. Image quality:
   - blurry, low resolution, severe motion blur, out of focus, very compressed, or too tightly cropped to understand the landmark.

Allowed:
- Fixed architectural parts of the landmark scene, including the Brandenburg Gate columns, roof, Quadriga statue,
  decorative railings/fences, pavement, plaza, sky, and permanent surrounding context.
- For Brandenburg Gate, the Quadriga statue is part of the landmark. Do NOT count it as real people, horses, vehicles, or an obstruction.

Important decision policy:
- Prefer false negatives over false positives.
- If a person/vehicle is visible and you are not sure whether it is tiny and harmless, reject.
- If the image looks like a postcard/filter/HDR/monochrome/night/extreme-weather shot, reject.
- Keep=true ONLY for natural, mostly unobstructed, sharp images with no meaningful people/vehicle/clutter.

Return ONLY valid JSON with this exact schema:
{{
  "people_or_body_parts": false,
  "vehicles": false,
  "flags_banners_signs_text": false,
  "obstructions_or_foreground_clutter": false,
  "filtered_or_stylized_appearance": false,
  "sharp_and_clear": true,
  "keep": true,
  "reason": "short reason"
}}

Set people_or_body_parts=true for visible large/medium/foreground/central/overlapping people or crowds.
Set vehicles=true for visible large/medium/foreground/central/overlapping vehicles or traffic.
Set flags_banners_signs_text=true for visible distracting flags, banners, signs, text overlays, watermarks, or ads.
Set obstructions_or_foreground_clutter=true for any meaningful occlusion/clutter, including people/vehicles that cover landmark geometry.
Set filtered_or_stylized_appearance=true for extreme weather, night/dark artificial lighting, heavy HDR/filter/color cast, monochrome, or edited appearance.
Set sharp_and_clear=false for blur, low resolution, severe exposure problems, or unusable crop."""

    def _build_benchmark_reconstruction_clean_prompt(self, strict_appearance: bool = False) -> str:
        """构建 3D 重建干净图像集用的严格 prompt。"""
        landmark_lower = self.landmark_name.lower()
        special_notes = ""
        if "brandenburg" in landmark_lower:
            special_notes = (
                "\n- For Brandenburg Gate, the Quadriga statue on top of the gate is part of the landmark. "
                "Do NOT count it as real people, horses, vehicles, or clutter."
            )
        elif "trevi" in landmark_lower:
            special_notes = (
                "\n- For Trevi Fountain, permanent fountain sculptures, carved figures, horses, rocks, and water "
                "are part of the landmark. Do NOT count fixed sculptures as real people or clutter."
            )
        elif "taj" in landmark_lower:
            special_notes = (
                "\n- For Taj Mahal, the reflecting pool, fixed garden geometry, minarets, domes, arches, and "
                "permanent architectural ornament are part of the landmark."
            )
        elif "notre" in landmark_lower:
            special_notes = (
                "\n- For Notre Dame, fixed facade statues, carved figures, portals, towers, rose windows, and "
                "architectural ornament are part of the landmark."
            )
        elif "sacre" in landmark_lower or "sacré" in landmark_lower:
            special_notes = (
                "\n- For Sacre Coeur, domes, towers, stairs, fixed railings, facade statues, and architectural "
                "ornament are part of the landmark."
            )
        elif "pantheon" in landmark_lower:
            special_notes = (
                "\n- For Pantheon exterior, columns, pediment, dome, fountain/obelisk if part of the fixed scene, "
                "and permanent architectural context are allowed."
            )
        elif "buckingham" in landmark_lower:
            special_notes = (
                "\n- For Buckingham Palace, fixed gates, railings, statues, memorials, facade elements, and "
                "permanent architectural context are allowed."
            )
        elif "temple" in landmark_lower or "nara" in landmark_lower:
            special_notes = (
                "\n- For the Nara temple scene, fixed temple architecture, roofs, gates, lanterns, statues, "
                "stonework, and permanent garden/path elements are allowed."
            )
        strict_appearance_notes = ""
        if strict_appearance:
            strict_appearance_notes = """

Strict appearance addendum for this benchmark rebuild:
- Be especially strict about extreme weather, night, dusk/dawn with artificial lighting, fog/haze/smoke, heavy rain/snow, storm sky, strong backlight, severe underexposure/overexposure, and low-visibility atmosphere.
- Reject postcard-like photos, heavy HDR, Instagram-style filters, high saturation/color grading, sepia, black-and-white, strong warm/cold color cast, fantasy/dramatic lighting, fisheye/distortion, collage, drawing/rendering, or edited/AI-looking images.
- Normal daylight, mild overcast, ordinary clouds, ordinary shadows, and natural water/sky colors are allowed.
- Set filtered_or_stylized_appearance=true whenever the rejection is due to weather, night/darkness, filter/color grading, stylization, artificial lighting, severe exposure, or non-natural appearance."""
        return f"""You are filtering images of {self.landmark_name} for clean 3D reconstruction.

Use a VERY STRICT policy. Keep only clean, natural, mostly empty landmark photos.

Reject the image if ANY of these are visible anywhere in the image:
1. Any real person, tourist, pedestrian, crowd, face, body, body part, selfie subject, or reflection of a person.
   This includes tiny, distant, background, partial, blurry, or non-overlapping people.
2. Any vehicle or traffic object: car, bus, truck, bicycle, motorcycle, scooter, carriage, taxi, van, tram, parked vehicle,
   or even a small/distant/partial vehicle.
3. Any movable or temporary clutter: umbrellas, tents, temporary barriers, cones, construction, scaffolding, tripods,
   foreground poles crossing the landmark, large street lamps dominating the view, flags, banners, signs, billboards,
   readable text, ads, watermarks, or posters.
4. Any extreme or non-natural appearance: heavy filter, strong HDR, oversaturation, sepia, black-and-white,
   strong color cast, artistic editing, postcard-like effect, fisheye/distortion, collage, drawing/rendering,
   night/dark artificial lighting, heavy rain/snow/fog/smoke/haze, storm sky, severe overexposure or underexposure.
5. Poor image quality or bad crop: blur, low resolution, severe compression, motion blur, out of focus,
   or crop that misses important parts of the gate, columns, roof, main facade, or Quadriga.

Allowed:
- Permanent architecture and fixed landmark elements only: landmark facade, columns, roofs, domes, towers, arches,
  doors, windows, fixed decorative railings/fences/gates, stairs, pavement/plaza/path, water basins/fountains,
  permanent sculptures/statues/reliefs/carved figures, normal sky, vegetation, and permanent surrounding buildings.
- Do NOT count fixed sculptures, statues, reliefs, carved figures, fountain figures, decorative animals, or other
  permanent landmark ornament as real people, vehicles, or clutter.{special_notes}
- Normal daylight, mild overcast, ordinary clouds, and ordinary shadows are allowed.

Inspection checklist before answering:
- Look at the base/plaza/path/street area, landmark openings, foreground edges, left/right edges, and background
  context for tiny people or vehicles.
- Look for color grading/HDR/monochrome/night/extreme weather.
- If you are unsure whether a small object is a person or vehicle, reject.{strict_appearance_notes}

Return ONLY valid JSON with this exact schema:
{{
  "people_or_body_parts": false,
  "vehicles": false,
  "flags_banners_signs_text": false,
  "obstructions_or_foreground_clutter": false,
  "filtered_or_stylized_appearance": false,
  "sharp_and_clear": true,
  "keep": true,
  "reason": "short reason"
}}

Set keep=true ONLY if there are no real people, no vehicles, no temporary clutter/signage, natural appearance, and sharp clear landmark geometry.
Prefer false negatives over false positives."""

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

        # Some landmark benchmark scenes contain fixed fences, fountains,
        # obelisks, sculptures, water basins, etc. LLaVA often reports these as
        # "obstructions" despite the prompt. Correct that conservative
        # over-rejection only when the reason names permanent landmark elements
        # and does not name temporary/foreground obstruction terms.
        reason_lower = decision["reason"].lower()
        has_scene_allowed = any(term in reason_lower for term in self.SCENE_ALLOWED_TERMS)
        has_temporary_obstruction = any(term in reason_lower for term in self.TEMPORARY_OBSTRUCTION_TERMS)
        if decision["obstructions_or_foreground_clutter"] and has_scene_allowed and not has_temporary_obstruction:
            decision["obstructions_or_foreground_clutter"] = False
            decision["reason"] = f"{decision['reason']} [normalized: fixed landmark element allowed]"

        negatives = [
            decision["people_or_body_parts"],
            decision["vehicles"],
            decision["flags_banners_signs_text"],
            decision["obstructions_or_foreground_clutter"],
            decision["filtered_or_stylized_appearance"],
        ]
        decision["keep"] = (not any(negatives)) and decision["sharp_and_clear"]
        if self.filter_mode == "benchmark_permissive":
            decision = self._normalize_permissive_decision(decision)
        return decision

    def _normalize_permissive_decision(self, decision: Dict) -> Dict:
        """宽松 benchmark 模式：允许小/远/不遮挡的游客车辆等上下文。"""
        reason_lower = decision["reason"].lower()
        blocking_terms = {
            "block", "blocked", "blocking", "obscure", "obscured", "cover", "covered",
            "covering", "distract", "distracting", "foreground", "close-up", "hard to see",
            "not clear", "cannot be understood",
        }
        tiny_terms = {"tiny", "small", "distant", "background", "not blocking", "do not cover", "does not cover", "no obstruction"}
        has_blocking = any(term in reason_lower for term in blocking_terms)
        has_allowed_context = any(term in reason_lower for term in self.PERMISSIVE_BENCHMARK_ALLOWED_TERMS)
        has_tiny_context = any(term in reason_lower for term in tiny_terms)
        if decision["sharp_and_clear"] and not decision["filtered_or_stylized_appearance"]:
            if has_allowed_context and not has_blocking:
                decision["people_or_body_parts"] = False
                decision["vehicles"] = False
                decision["flags_banners_signs_text"] = False
                decision["obstructions_or_foreground_clutter"] = False
                decision["reason"] = f"{decision['reason']} [normalized: acceptable benchmark context]"
            elif has_tiny_context and not has_blocking:
                decision["people_or_body_parts"] = False
                decision["vehicles"] = False
                decision["flags_banners_signs_text"] = False
                decision["obstructions_or_foreground_clutter"] = False
                decision["reason"] = f"{decision['reason']} [normalized: small/distant context allowed]"
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
        priority_filenames: Optional[Sequence[str]] = None,
        skip_existing_decisions: bool = False,
    ) -> Dict[str, int]:
        """
        过滤整个数据集

        Args:
            tsv_path: 输入 TSV 文件路径
            output_path: 输出 TSV 文件路径
            image_dir: 图像文件所在目录
            resume_from: 从第几行开始处理（用于断点续传）
            decision_log_path: 可选判定日志路径
            priority_filenames: 优先处理的文件名列表，用于先筛 benchmark/标注候选图
            skip_existing_decisions: 根据 decision_log_path 跳过已处理文件，支持更稳健断点续跑

        Returns:
            统计信息字典
        """
        # 读取输入 TSV
        with open(tsv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        existing_decisions: Set[str] = set()
        existing_kept: Set[str] = set()
        if skip_existing_decisions:
            existing_decisions = self._read_existing_decision_filenames(decision_log_path)
            existing_kept = self._read_existing_output_filenames(output_path)

        # 准备输出文件
        should_append = resume_from > 0 or (
            skip_existing_decisions and os.path.exists(output_path) and os.path.getsize(output_path) > 0
        )
        output_mode = 'a' if should_append else 'w'
        output_file = open(output_path, output_mode, newline='')
        writer = csv.DictWriter(
            output_file,
            fieldnames=['filename', 'id', 'split', 'dataset'],
            delimiter='\t'
        )

        # 如果是新文件，写入表头
        if not should_append:
            writer.writeheader()
            output_file.flush()

        # 统计信息
        stats = {
            'total': len(rows),
            'processed': len(existing_decisions) if skip_existing_decisions else resume_from,
            'passed': len(existing_kept) if skip_existing_decisions else 0,
            'failed': 0,
            'error': 0
        }

        log_file = None
        if decision_log_path:
            log_mode = 'a' if should_append else 'w'
            log_file = open(decision_log_path, log_mode, encoding='utf-8')

        # 如果从中间恢复，先统计已处理的通过数量
        if resume_from > 0 and not skip_existing_decisions:
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    existing_lines = sum(1 for _ in f)
                stats['passed'] = max(existing_lines - 1, 0)  # 减去表头
            else:
                stats['passed'] = 0

        row_by_filename = {row.get('filename'): row for row in rows}
        ordered_rows: List[Dict[str, str]] = []
        seen_order: Set[str] = set()
        if priority_filenames:
            for name in priority_filenames:
                name = name.strip()
                if not name or name in seen_order:
                    continue
                row = row_by_filename.get(name)
                if row is not None:
                    ordered_rows.append(row)
                    seen_order.add(name)
        for row in rows[resume_from:]:
            name = row.get('filename')
            if name in seen_order:
                continue
            ordered_rows.append(row)
            if name:
                seen_order.add(name)

        try:
            # 遍历所有图像
            progress_initial = min(stats['processed'], len(rows))
            for idx, row in enumerate(tqdm(ordered_rows, initial=progress_initial, total=len(rows))):
                filename = row['filename']
                if skip_existing_decisions and filename in existing_decisions:
                    continue
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
                existing_decisions.add(filename)

                if passed:
                    # 通过质量检查，写入输出文件
                    if filename not in existing_kept:
                        writer.writerow(row)
                        output_file.flush()  # 立即写入磁盘
                        existing_kept.add(filename)
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

    @staticmethod
    def _read_existing_decision_filenames(decision_log_path: Optional[str]) -> Set[str]:
        """从判定日志中读取已处理文件名。"""
        filenames: Set[str] = set()
        if not decision_log_path or not os.path.exists(decision_log_path):
            return filenames
        with open(decision_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                filename = record.get('filename')
                if filename:
                    filenames.add(str(filename))
        return filenames

    @staticmethod
    def _read_existing_output_filenames(output_path: str) -> Set[str]:
        """从已有输出 TSV 中读取已保留文件名。"""
        filenames: Set[str] = set()
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return filenames
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                filename = row.get('filename')
                if filename:
                    filenames.add(str(filename))
        return filenames


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
        "--filter_mode",
        type=str,
        default="strict",
        choices=["strict", "benchmark_permissive", "benchmark_balanced_strict", "benchmark_reconstruction_clean", "benchmark_reconstruction_clean_strict_appearance"],
        help="过滤模式：strict 用于训练集；benchmark_permissive 用于审核 benchmark 选图；benchmark_balanced_strict 用于较严格选图；benchmark_reconstruction_clean 用于无真人/车辆的重建干净集；benchmark_reconstruction_clean_strict_appearance 在同一次 VLM 判定内进一步收紧极端天气/滤镜/黑夜等 appearance"
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
    parser.add_argument(
        "--priority_filenames",
        type=str,
        default=None,
        help="可选：优先处理的文件名列表，每行一个文件名"
    )
    parser.add_argument(
        "--skip_existing_decisions",
        action="store_true",
        help="根据 decision_log_path 跳过已处理文件，便于优先队列/断点续跑"
    )

    args = parser.parse_args()

    # 创建过滤器
    filter_obj = OllamaVLMFilter(
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        model_name=args.model_name,
        landmark_name=args.landmark_name,
        request_timeout=args.request_timeout,
        filter_mode=args.filter_mode,
    )

    print(f"开始过滤数据集...")
    print(f"输入文件: {args.tsv_path}")
    print(f"输出文件: {args.output_path}")
    print(f"图像目录: {args.image_dir}")
    print(f"Ollama 服务: {args.ollama_host}:{args.ollama_port}")
    print(f"模型: {args.model_name}")
    print(f"地标: {args.landmark_name}")
    print(f"过滤模式: {args.filter_mode}")
    print(f"超时: {args.request_timeout}s")
    if args.decision_log_path:
        print(f"判定日志: {args.decision_log_path}")
    priority_filenames = None
    if args.priority_filenames:
        with open(args.priority_filenames, 'r', encoding='utf-8', errors='ignore') as f:
            priority_filenames = [line.strip() for line in f if line.strip()]
        print(f"优先文件列表: {args.priority_filenames} ({len(priority_filenames)} files)")
    if args.skip_existing_decisions:
        print("断点模式: skip_existing_decisions=True")
    print("-" * 80)

    # 执行过滤
    stats = filter_obj.filter_dataset(
        tsv_path=args.tsv_path,
        output_path=args.output_path,
        image_dir=args.image_dir,
        resume_from=args.resume_from,
        decision_log_path=args.decision_log_path,
        priority_filenames=priority_filenames,
        skip_existing_decisions=args.skip_existing_decisions,
    )

    # 打印统计信息
    print("\n" + "=" * 80)
    print("过滤完成！")
    print(f"总计: {stats['total']} 张图片")
    print(f"已处理: {stats['processed']} 张")
    processed = max(stats['processed'], 1)
    print(f"通过: {stats['passed']} 张 ({stats['passed']/processed*100:.1f}%)")
    print(f"未通过: {stats['failed']} 张 ({stats['failed']/processed*100:.1f}%)")
    if stats['error'] > 0:
        print(f"错误: {stats['error']} 张")
    print("=" * 80)


if __name__ == "__main__":
    main()
