"""Render YOLO bboxes + SAM2 instance masks onto images.

Used by precompute_examples.py (gallery PNGs) and space/app.py (live viz).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BBOX_COLOR = (37, 99, 235)  # blue
BBOX_WIDTH = 3
LABEL_FONT_SIZE = 18


def _load_font(size: int = LABEL_FONT_SIZE):
    for candidate in [
        "/System/Library/Fonts/Helvetica.ttc",                # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",    # Linux / HF Space
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_detection_overlay(image_path, boxes, output_path) -> Path:
    """Draw YOLO bboxes from JSON-schema dicts onto image."""
    img = Image.open(image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _load_font()
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        label = f"{b['class_name']} {b['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=BBOX_WIDTH)
        tx, ty = x1, max(y1 - 22, 0)
        bb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle([bb[0]-3, bb[1]-3, bb[2]+3, bb[3]+3], fill=BBOX_COLOR)
        draw.text((tx, ty), label, fill="white", font=font)
    output_path = Path(output_path)
    img.save(output_path)
    return output_path


def render_segmentation_overlay(image_path, raw_masks, output_path,
                                  alpha: float = 0.55, seed: int = 42) -> Path:
    """Render SAM2 instance masks as semi-transparent colored overlay.
    
    `raw_masks` is the output of SAM2AutomaticMaskGenerator.generate() —
    a list of dicts each with 'segmentation' (HxW bool) and 'area'.
    """
    img = Image.open(image_path).convert("RGB")
    base = np.array(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    # Paint largest masks first so small ones layer on top
    for m in sorted(raw_masks, key=lambda m: m["area"], reverse=True):
        mask = m["segmentation"]
        if mask.dtype != bool:
            mask = mask.astype(bool)
        color = rng.integers(40, 255, size=3).astype(np.float32)
        for c in range(3):
            base[..., c] = np.where(
                mask, base[..., c] * (1 - alpha) + color[c] * alpha, base[..., c]
            )
    output_path = Path(output_path)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(output_path)
    return output_path
