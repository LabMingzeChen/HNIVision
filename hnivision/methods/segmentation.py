"""Image segmentation method (Method 4) — SegFormer-b0 on ADE20K.

Per-pixel semantic labeling across 150 ADE20K classes, aggregated to
per-class pixel shares. Primary HNI contribution: nature composition
(sky/tree/grass/water shares) and human presence (person mask area).

Note: This is the v0.1 SegFormer-only backend. A SAM2-based mask-level
enhancement is planned for v0.2 (requires arm64 native Python env).
"""

from __future__ import annotations

# Apple Silicon (mps) safety: enable CPU fallback for any unsupported ops
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.schema import HNIResult, HumanLevel, NatureLevel


# --- ADE20K class buckets for HNI mapping (canonical-name based) ---
# Match against the first synonym (e.g., "mountain" from "mountain, mount")

NATURE_CANONICAL = {
    "sky", "tree", "grass", "earth", "mountain", "plant", "water",
    "sea", "field", "rock", "sand", "river", "flower", "hill",
    "palm", "dirt track", "land", "waterfall", "lake",
}

HUMAN_CANONICAL = {"person"}


def _canonical_name(full_label: str) -> str:
    """Extract the first synonym from ADE20K's comma-separated label.

    Examples:
        'person, individual, someone, ...' → 'person'
        'mountain, mount' → 'mountain'
        'sky' → 'sky'
    """
    return full_label.split(",")[0].strip().lower()


# --- Output schemas ---

class PixelShares(BaseModel):
    """Per-class pixel share for one image."""
    shares: Dict[str, float] = Field(default_factory=dict)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)

    def top_n(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return the top N classes sorted by pixel share."""
        return sorted(self.shares.items(), key=lambda x: -x[1])[:n]

    def dominant(self) -> Optional[str]:
        if not self.shares:
            return None
        return max(self.shares.items(), key=lambda x: x[1])[0]


class SegmentationOutput(BaseModel):
    """Output of Segmentation on one image."""
    pixel_shares: PixelShares = Field(default_factory=PixelShares)
    semantic_map_shape: Tuple[int, int] = (0, 0)  # (H, W)
    n_classes_detected: int = 0


# --- Method ---

class Segmentation(BaseHNIMethod):
    """Image segmentation via SegFormer-b0 fine-tuned on ADE20K.

    Returns per-pixel semantic classes (150 categories), aggregated to per-class
    pixel shares of the image.

    Requirements:
      - `pip install "hnivision[segmentation]"`

    Example:
        >>> seg = Segmentation()
        >>> out = seg.extract("park.jpg")
        >>> out.pixel_shares.top_n(3)
        [('sky', 0.42), ('tree', 0.18), ('grass', 0.09)]
        >>> hni = seg.to_hni(out)
        >>> hni.nature.tags
        ['sky', 'tree', 'grass']
    """

    name = "segmentation"

    def __init__(
        self,
        segformer_model: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: Optional[str] = None,
        min_share_for_hni: float = 0.005,
    ):
        """
        Args:
            segformer_model: HuggingFace model ID for SegFormer ADE20K checkpoint.
            device: 'cuda', 'mps', 'cpu', or None for auto-detect.
            min_share_for_hni: Minimum pixel share (0-1) to count a class
                in the HNI mapping. Default 0.005 = 0.5% of image area.
        """
        self.model_name = segformer_model
        self.min_share_for_hni = min_share_for_hni

        try:
            import torch
        except ImportError as e:
            raise ImportError("Segmentation requires PyTorch") from e

        if device is None:
            # NOTE: MPS gives numerically wrong results on SegFormer (transformer
            # ops aren't fully supported as of PyTorch 2.2). Force CPU on Mac.
            # CUDA still preferred when available.
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self._torch = torch

        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "Segmentation requires `transformers`. Install with:\n"
                "    pip install 'hnivision[segmentation]'"
            ) from e

        self._processor = AutoImageProcessor.from_pretrained(segformer_model)
        self._model = SegformerForSemanticSegmentation.from_pretrained(segformer_model)
        self._model.to(device)
        self._model.eval()
        self._id2label = self._model.config.id2label

    def extract(self, image: ImageInput) -> SegmentationOutput:
        """Run SegFormer and compute per-class pixel shares."""
        img = self.load_image(image)
        width, height = img.size
        total_pixels = width * height

        inputs = self._processor(images=img, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process: resize logits back to original image size
        seg_map = self._processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(height, width)],
        )[0].cpu().numpy()

        class_ids, pixel_counts = np.unique(seg_map, return_counts=True)
        shares: Dict[str, float] = {}
        for cls_id, count in zip(class_ids, pixel_counts):
            label = self._id2label[int(cls_id)]
            shares[label] = float(count) / total_pixels

        return SegmentationOutput(
            pixel_shares=PixelShares(shares=shares, image_size=(width, height)),
            semantic_map_shape=tuple(seg_map.shape),
            n_classes_detected=len(class_ids),
        )

    def to_hni(self, output: SegmentationOutput) -> HNIResult:
        """Map per-class pixel shares into HNI levels."""
        result = HNIResult()

        # Filter classes above the HNI threshold
        relevant = {
            cls: share
            for cls, share in output.pixel_shares.shares.items()
            if share >= self.min_share_for_hni
        }

        # --- Human ---
        person_share = sum(
            share for cls, share in relevant.items()
            if _canonical_name(cls) in HUMAN_CANONICAL
        )
        if person_share > 0:
            result.human = HumanLevel(
                present=True,
                evidence=f"SegFormer detected 'person' covering {person_share:.1%} of image",
            )

        # --- Nature ---
        nature_pairs: List[Tuple[str, float]] = sorted(
            [
                (_canonical_name(cls), share)
                for cls, share in relevant.items()
                if _canonical_name(cls) in NATURE_CANONICAL
            ],
            key=lambda x: -x[1],
        )
        if nature_pairs:
            result.nature = NatureLevel(
                tags=[name for name, _ in nature_pairs],
                dominant=nature_pairs[0][0],
                pixel_shares=dict(nature_pairs),
                evidence=(
                    f"SegFormer detected {len(nature_pairs)} nature class(es): "
                    + ", ".join(f"{n} ({s:.1%})" for n, s in nature_pairs[:5])
                ),
            )

        # Activity, Meaning: segmentation doesn't capture these
        return result
