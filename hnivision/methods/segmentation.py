"""Method 4: Hybrid SAM2 + SegFormer (ADE20K) instance-level segmentation.

Two-stage pipeline matching the user's NHI_SAM2+ADK20.ipynb research workflow:
  Stage 1: SAM2.1-hiera-small generates N instance masks (auto mode)
  Stage 2: SegFormer-b0 on ADE20K labels each mask via majority vote
  Stage 3: Aggregate per-mask labels into pixel_shares for HNI mapping

Output is richer than either model alone — SAM2 gives precise object
boundaries, ADE20K gives class names. Each mask carries both pieces.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.schema import HNIResult, HumanLevel, NatureLevel


# ============================================================
# Output schema
# ============================================================
class MaskInstance(BaseModel):
    """One SAM2 mask with its ADE20K semantic label (majority vote)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mask_id: int
    area: int                     # pixel count
    pixel_share: float            # area / total image pixels
    bbox: List[int]               # [x, y, w, h] from SAM2
    sam2_iou: float               # SAM2 predicted IoU
    sam2_stability: float         # SAM2 stability score
    ade20k_label: str             # majority class name
    ade20k_label_idx: int         # ADE20K class id (0-149)
    ade20k_confidence: float      # fraction of mask pixels matching dominant label


class SegmentationOutput(BaseModel):
    """Hybrid SAM2 + SegFormer segmentation output."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    masks: List[MaskInstance] = Field(default_factory=list)
    n_masks: int = 0
    image_size: Tuple[int, int] = (0, 0)  # (W, H)

    # Backward-compat: aggregated pixel shares per ADE20K class
    pixel_shares: Dict[str, Any] = Field(
        default_factory=lambda: {"shares": {}, "image_size": [0, 0]}
    )
    n_classes_detected: int = 0

    coverage_ratio: float = 0.0   # total SAM2 mask px / image px (>1 = overlap)
    method_meta: Dict[str, str] = Field(default_factory=dict)


# ============================================================
# Class → HNI category mappings
# ============================================================
NATURE_FIRST_WORDS = {
    "sky", "tree", "grass", "plant", "flower", "earth", "field",
    "mountain", "rock", "sea", "water", "river", "lake", "sand",
    "snow", "waterfall", "fountain", "hill", "land", "vegetation",
}
HUMAN_KEYWORDS = ("person", "individual", "soul", "people")


# ============================================================
# Segmentation method
# ============================================================
class Segmentation(BaseHNIMethod):
    """Hybrid SAM2 + SegFormer (ADE20K) — paper config."""

    def __init__(
        self,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        segformer_model: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cpu",
        # SAM2 params matching user's notebook
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.6,
        stability_score_thresh: float = 0.6,
        min_mask_region_area: int = 100,
    ):
        super().__init__()
        if sam2_checkpoint is None:
            sam2_checkpoint = str(
                Path.home() / ".cache/hnivision/sam2/sam2.1_hiera_small.pt"
            )
        if not Path(sam2_checkpoint).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {sam2_checkpoint}. "
                f"Download: curl -L --continue-at - -o {sam2_checkpoint} "
                f"https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt"
            )

        self.device = device

        # Lazy imports
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        # === Load SAM2 ===
        print(f"  Loading SAM2 ({Path(sam2_checkpoint).name}, device={device})...")
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self._sam2_gen = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

        # === Load SegFormer ===
        print(f"  Loading SegFormer ({segformer_model}, device={device})...")
        self._processor = SegformerImageProcessor.from_pretrained(segformer_model)
        self._segformer = (
            SegformerForSemanticSegmentation.from_pretrained(segformer_model)
            .to(device)
            .eval()
        )
        self._id2label = self._segformer.config.id2label

        self._sam2_model_name = "facebook/sam2.1-hiera-small"
        self._segformer_model_name = segformer_model

    # ------------------------------------------------------------
    @staticmethod
    def _load_image(image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return np.array(img)

    # ------------------------------------------------------------
    def extract(self, image: ImageInput) -> SegmentationOutput:
        img_np = self._load_image(image)
        H, W = img_np.shape[:2]
        total_px = float(H * W)

        # === Stage 1: SAM2 instance masks ===
        sam2_masks = self._sam2_gen.generate(img_np)

        # === Stage 2: SegFormer per-pixel ADE20K labels ===
        pil_img = Image.fromarray(img_np)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._segformer(**inputs)
        logits = outputs.logits  # (1, 150, h, w)
        upsampled = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        ade20k_map = (
            upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
        )  # (H, W)

        # === Stage 3: per-mask majority vote ===
        masks_out: List[MaskInstance] = []
        agg: Dict[str, float] = defaultdict(float)

        for mid, m in enumerate(sam2_masks):
            seg = m["segmentation"]  # bool (H, W)
            labels_in_mask = ade20k_map[seg]
            if labels_in_mask.size == 0:
                continue

            counts = np.bincount(labels_in_mask, minlength=150)
            majority_idx = int(counts.argmax())
            majority_count = int(counts[majority_idx])
            confidence = majority_count / labels_in_mask.size
            label_name = self._id2label.get(majority_idx, f"class_{majority_idx}")

            masks_out.append(
                MaskInstance(
                    mask_id=mid,
                    area=int(m["area"]),
                    pixel_share=m["area"] / total_px,
                    bbox=[int(x) for x in m["bbox"]],
                    sam2_iou=float(m["predicted_iou"]),
                    sam2_stability=float(m["stability_score"]),
                    ade20k_label=label_name,
                    ade20k_label_idx=majority_idx,
                    ade20k_confidence=confidence,
                )
            )
            agg[label_name] += m["area"] / total_px

        # Normalize aggregated shares (sum to 1 after overlap correction)
        total = sum(agg.values())
        normalized = {k: v / total for k, v in agg.items()} if total > 0 else {}
        sorted_shares = dict(
            sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        )
        coverage = sum(m["area"] for m in sam2_masks) / total_px

        return SegmentationOutput(
            masks=masks_out,
            n_masks=len(masks_out),
            image_size=(W, H),
            pixel_shares={"shares": sorted_shares, "image_size": [W, H]},
            n_classes_detected=len(sorted_shares),
            coverage_ratio=coverage,
            method_meta={
                "sam2_model": self._sam2_model_name,
                "ade20k_model": self._segformer_model_name,
            },
        )

    # ------------------------------------------------------------
    def to_hni(self, output: SegmentationOutput) -> HNIResult:
        result = HNIResult(
            source_method="segmentation",
            raw_output={
                "n_masks": output.n_masks,
                "n_classes": output.n_classes_detected,
            },
        )
        shares = output.pixel_shares.get("shares", {})
        if not shares:
            return result

        human_tags, nature_tags = [], []
        nature_pixel_shares: Dict[str, float] = {}
        human_share = 0.0

        for label, share in shares.items():
            label_lower = label.lower()
            first = label_lower.split(",")[0].strip()
            if any(k in label_lower for k in HUMAN_KEYWORDS):
                human_tags.append(first)
                human_share += share
            if first in NATURE_FIRST_WORDS:
                nature_tags.append(first)
                nature_pixel_shares[first] = round(share, 4)

        if human_tags:
            result.human = HumanLevel(
                present=True,
                tags=list(dict.fromkeys(human_tags)),
                evidence=f"Detected person in SAM2 masks ({human_share*100:.1f}% pixel share)",
            )

        if nature_tags:
            dominant = max(nature_pixel_shares.items(), key=lambda x: x[1])[0]
            ev_parts = [f"{k} {v*100:.1f}%" for k, v in list(nature_pixel_shares.items())[:5]]
            result.nature = NatureLevel(
                tags=list(dict.fromkeys(nature_tags)),
                dominant=dominant,
                pixel_shares=nature_pixel_shares,
                evidence=f"Top natural classes: {', '.join(ev_parts)}",
            )

        return result
