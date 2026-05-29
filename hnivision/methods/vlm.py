"""Vision-Language Model method (Method 5) — Qwen via OpenAI (OpenAI-compatible).

Sends image + structured prompt, parses JSON response into HNIResult.
This is the only method that fills ALL 4 HNI levels coherently +
image-level summary + overall HNI judgement + confidence.

Requirements:
  - `pip install "hnivision[vlm]"`
  - OPENAI_API_KEY env var (or .env at project root)
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.schema import (
    ActivityLevel,
    HNIResult,
    HumanLevel,
    MeaningLevel,
    NatureLevel,
)


# OpenAI uses its default endpoint; no base_url override needed.


# --- The HNI prompt (1:1 from GPT_HNI Review.ipynb Cell 0 - the "new logic" 4-dim version) ---

HNI_PROMPT = """\
You are an expert in urban nature research, landscape architecture, and human–nature interaction (HNI).

Analyze the image using ONLY visually supported evidence.

This task focuses on four dimensions of image-based HNI evidence:

1. Human presence
Definition: Whether people are visible in the image.
Examples: individuals, groups, crowd, children, visitors.

2. Nature detection
Definition: What type of natural or park-related environment is visible in the image.
Examples: woodland, lawn, flower garden, waterfront, wetland, beach, open field, park path.

3. Activity evidence
Definition: What type of visible activity or event is taking place in the image.
Examples: walking, playing, resting, gathering, watching, learning.

4. Cultural / experiential meaning
Definition: What broader experiential or cultural value the scene may suggest.
Examples: recreation, aesthetic appreciation, social interaction, education, relaxation, cultural participation.

Important rules:
- Only describe what is visually supported by the image.
- Do not invent invisible intentions, emotions, or background information.
- If the evidence is weak or unclear, use "unclear".
- Use controlled vocabulary whenever possible.
- Output ONLY valid JSON.
- Do not include markdown.

Controlled vocabulary:

human_presence:
["yes", "no", "unclear"]

human_presence_type:
["none", "individual", "group", "crowd", "children", "visitors", "unclear"]

estimated_people_count:
["0", "1", "2-5", "6-20", "21+", "unclear"]

nature_detection:
["woodland", "lawn", "flower garden", "waterfront", "wetland", "beach", "open field", "park path", "urban greenery", "wildlife habitat", "indoor nature display", "no clear natural setting", "unclear"]

activity_evidence:
["walking", "playing", "resting", "gathering", "watching", "learning", "performing", "sports", "gardening", "volunteering", "wildlife observation", "landscape viewing", "none", "unclear"]

cultural_experiential_meaning:
["recreation", "aesthetic appreciation", "social interaction", "education", "relaxation", "cultural participation", "wildlife appreciation", "environmental stewardship", "none", "unclear"]

confidence:
["high", "medium", "low"]

Return JSON with exactly this structure:

{
  "image_level_summary": "",
  "human_presence": {
    "visible": "",
    "presence_type": [],
    "estimated_people_count": "",
    "visual_evidence": ""
  },
  "nature_detection": {
    "nature_types": [],
    "dominant_nature_type": "",
    "visual_evidence": ""
  },
  "activity_evidence": {
    "activity_types": [],
    "dominant_activity": "",
    "visual_evidence": ""
  },
  "cultural_experiential_meaning": {
    "meaning_types": [],
    "dominant_meaning": "",
    "visual_evidence": ""
  },
  "hni_overall_interpretation": {
    "is_hni_present": "",
    "hni_strength": "none/weak/moderate/strong/unclear",
    "reason": ""
  },
  "confidence": "",
  "uncertainty_notes": ""
}
"""


# --- Output schema ---

class VLMOutput(BaseModel):
    """Output of VLM on one image."""
    raw_response: str = ""
    parsed: Dict[str, Any] = Field(default_factory=dict)
    parse_error: Optional[str] = None
    tokens_used: int = 0


# --- The Method ---

class VLM(BaseHNIMethod):
    """Vision-Language Model method via OpenAI (Qwen).

    Uses Alibaba's OpenAI-compatible API to call Qwen vision models.
    Sends an image + structured HNI prompt and parses the JSON response
    directly into a complete HNIResult.

    Example:
        >>> vlm = VLM()
        >>> out = vlm.extract("park.jpg")
        >>> hni = vlm.to_hni(out)
        >>> hni.summary
        'Two children at a stone overlook with the Hudson River in view.'
        >>> hni.nature.dominant
        'waterfront'
    """

    name = "vlm"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4.1-mini",
        max_image_size: int = 768,
        image_quality: int = 75,
        temperature: float = 0.0,
        max_output_tokens: int = 1500,
    ):
        """
        Args:
            api_key: OpenAI API key. If None, reads OPENAI_API_KEY from
                env (and tries .env at project root).
            model_name: OpenAI vision model. Default 'gpt-4.1-mini'.
                Alternatives: 'qwen-vl-max-latest', 'qwen-vl-plus-latest'.
            max_image_size: Longest side after resize (smaller = cheaper).
            image_quality: JPEG quality (1-100). Default 75 = good balance.
            temperature: 0.0 for deterministic output.
            max_output_tokens: Cap on response length.
        """
        self.model_name = model_name
        self.max_image_size = max_image_size
        self.image_quality = image_quality
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # --- Resolve API key ---
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            try:
                from dotenv import load_dotenv
                here = Path(__file__).resolve()
                for parent in [here.parent, *here.parents]:
                    env_file = parent / ".env"
                    if env_file.exists():
                        load_dotenv(env_file)
                        break
            except ImportError:
                pass

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Either:\n"
                "  1. Pass api_key=... to VLM(...)\n"
                "  2. Set in shell: export OPENAI_API_KEY=sk-...\n"
                "  3. Put it in .env at the project root"
            )

        # --- Lazy import openai ---
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "VLM requires `openai`. Install with:\n"
                "    pip install 'hnivision[vlm]'"
            ) from e

        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def _image_to_data_url(self, image: ImageInput) -> str:
        """Compress image to JPEG base64 data URL."""
        img = self.load_image(image)
        img.thumbnail((self.max_image_size, self.max_image_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.image_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _parse_json(self, content: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Strip markdown fences and parse JSON. Returns (parsed_dict, error_or_none)."""
        cleaned = content.strip()
        # Strip ```json ... ``` if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n", 1)
            if len(lines) == 2:
                cleaned = lines[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"

    def extract(self, image: ImageInput) -> VLMOutput:
        """Send image + HNI prompt to Qwen, parse JSON response."""
        data_url = self._image_to_data_url(image)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": HNI_PROMPT},
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        raw = response.choices[0].message.content or ""
        parsed, err = self._parse_json(raw)

        return VLMOutput(
            raw_response=raw,
            parsed=parsed or {},
            parse_error=err,
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )

    def to_hni(self, output: VLMOutput) -> HNIResult:
        """Map parsed JSON directly into HNIResult (most natural mapping of all 5 methods)."""
        result = HNIResult()
        if not output.parsed:
            return result

        p = output.parsed

        # Top-level summary
        result.summary = p.get("image_level_summary") or None

        # --- Human ---
        hp = p.get("human_presence", {})
        result.human = HumanLevel(
            present=(hp.get("visible", "").lower() == "yes"),
            count=hp.get("estimated_people_count") or None,
            tags=hp.get("presence_type") or [],
            evidence=hp.get("visual_evidence") or None,
        )

        # --- Nature ---
        nd = p.get("nature_detection", {})
        result.nature = NatureLevel(
            tags=nd.get("nature_types") or [],
            dominant=nd.get("dominant_nature_type") or None,
            evidence=nd.get("visual_evidence") or None,
        )

        # --- Activity ---
        ae = p.get("activity_evidence", {})
        result.activity = ActivityLevel(
            tags=ae.get("activity_types") or [],
            dominant=ae.get("dominant_activity") or None,
            evidence=ae.get("visual_evidence") or None,
        )

        # --- Meaning ---
        cem = p.get("cultural_experiential_meaning", {})
        result.meaning = MeaningLevel(
            tags=cem.get("meaning_types") or [],
            dominant=cem.get("dominant_meaning") or None,
            evidence=cem.get("visual_evidence") or None,
        )

        # --- HNI overall interpretation ---
        hni_overall = p.get("hni_overall_interpretation", {})
        is_present = hni_overall.get("is_hni_present", "").lower()
        if is_present in ("yes", "no", "unclear"):
            result.hni_present = is_present
        strength = hni_overall.get("hni_strength", "").lower()
        if strength in ("weak", "moderate", "strong"):
            result.hni_strength = strength

        # --- Confidence ---
        conf = p.get("confidence", "").lower()
        if conf in ("low", "medium", "high"):
            result.confidence = conf

        return result
