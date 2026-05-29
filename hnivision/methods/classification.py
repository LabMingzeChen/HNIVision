"""Image classification method (Method 2) — ResNet-50 fine-tuned on 7 HNI scenarios.

Predicts a single scenario category per image, with softmax probabilities
across all 7 classes. Each class maps to a multi-level HNI evidence pattern.

The 7 classes were defined by the project's HNI taxonomy:
  - Cultural Performance
  - Garden Vegetation
  - Social and Informal Recreation
  - Sport and Exercise Activities
  - Waterscape and Built Settings
  - Wildlife Encounters
  - Woodland Settings
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.schema import (
    ActivityLevel,
    HNIResult,
    HumanLevel,
    MeaningLevel,
    NatureLevel,
)

# --- The 7 classes, in ImageFolder alphabetical order ---
CLASS_NAMES: List[str] = [
    "Cultural Performance",            # idx 0
    "Garden Vegetation",               # idx 1
    "Social and Informal Recreation",  # idx 2
    "Sport and Exercise Activities",   # idx 3
    "Waterscape and Built Settings",   # idx 4
    "Wildlife Encounters",             # idx 5
    "Woodland Settings",               # idx 6
]


# --- Per-class HNI mapping ---
# Each class implies a specific pattern of HNI evidence
CLASS_TO_HNI: Dict[str, Dict] = {
    "Cultural Performance": {
        "human": True,
        "human_tags": ["groups", "crowd"],
        "nature_tags": [],
        "activity_tags": ["watching performance", "performing"],
        "meaning_tags": ["cultural experience", "aesthetic appreciation"],
    },
    "Garden Vegetation": {
        "human": False,
        "human_tags": [],
        "nature_tags": ["garden", "flowers", "plants", "manicured vegetation"],
        "activity_tags": [],
        "meaning_tags": ["aesthetic appreciation"],
    },
    "Social and Informal Recreation": {
        "human": True,
        "human_tags": ["groups", "families"],
        "nature_tags": ["park", "lawn"],
        "activity_tags": ["picnicking", "socializing", "relaxing"],
        "meaning_tags": ["social bonding", "leisure"],
    },
    "Sport and Exercise Activities": {
        "human": True,
        "human_tags": ["individuals"],
        "nature_tags": [],
        "activity_tags": ["sport", "exercise", "running", "cycling"],
        "meaning_tags": ["physical activity", "wellbeing"],
    },
    "Waterscape and Built Settings": {
        "human": True,
        "human_tags": ["individuals"],
        "nature_tags": ["water", "waterfront"],
        "activity_tags": ["walking", "viewing"],
        "meaning_tags": ["aesthetic appreciation", "contemplation"],
    },
    "Wildlife Encounters": {
        "human": True,
        "human_tags": ["individuals"],
        "nature_tags": ["wildlife", "animals", "natural area"],
        "activity_tags": ["observing wildlife", "photographing"],
        "meaning_tags": ["nature appreciation"],
    },
    "Woodland Settings": {
        "human": False,
        "human_tags": [],
        "nature_tags": ["woodland", "forest", "trees"],
        "activity_tags": ["walking", "hiking"],
        "meaning_tags": ["nature immersion", "restoration"],
    },
}


# --- Default weights location ---
DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "hnivision" / "resnet" / "best_resnet50_model.pth"

# HF Hub fallback (used when local cache missing, e.g. on HF Space deployments)
HF_REPO_ID = "Mingze/HNIVision-ResNet50"
HF_WEIGHTS_FILENAME = "best_resnet50_model.pth"


def _resolve_weights_path(weights_path: Optional[str]) -> Path:
    """Resolve ResNet weights path with HF Hub fallback.

    Resolution order:
      1. If `weights_path` explicitly passed, use it (must exist).
      2. If DEFAULT_WEIGHTS_PATH exists locally, use it.
      3. Otherwise download from HF Hub repo Mingze/HNIVision-ResNet50.
    """
    if weights_path is not None:
        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(f"ResNet weights not found at explicit path: {wp}")
        return wp

    if DEFAULT_WEIGHTS_PATH.exists():
        return DEFAULT_WEIGHTS_PATH

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"ResNet weights not in local cache ({DEFAULT_WEIGHTS_PATH}) and "
            "huggingface_hub is not installed. Install: pip install huggingface_hub"
        ) from e

    print(f"  ResNet weights not in local cache; downloading from {HF_REPO_ID}...")
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS_FILENAME))


# --- Output schemas ---

class ClassificationOutput(BaseModel):
    """Output of one classification inference."""
    predicted_class: str
    predicted_class_idx: int
    confidence: float  # softmax probability of the top class
    all_probs: Dict[str, float] = Field(default_factory=dict)  # all 7 class probabilities

    def top_n(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return top N classes sorted by probability."""
        return sorted(self.all_probs.items(), key=lambda x: -x[1])[:n]


# --- Method ---

class Classification(BaseHNIMethod):
    """Image classification via ResNet-50 fine-tuned on 7 HNI scenarios.

    Architecture matches the training notebook:
        ResNet-50 backbone (ImageNet pretrained) →
        fc = Sequential(
            Linear(2048, 512),
            ReLU(),
            Dropout(0.3),
            Linear(512, 7)
        )

    Requirements:
      - `pip install "hnivision[classification]"`
      - Weights file at ~/.cache/hnivision/resnet/best_resnet50_model.pth
        (or pass weights_path= explicitly)

    Example:
        >>> clf = Classification()
        >>> out = clf.extract("park.jpg")
        >>> out.predicted_class
        'Waterscape and Built Settings'
        >>> out.confidence
        0.78
        >>> hni = clf.to_hni(out)
        >>> hni.nature.tags
        ['water', 'waterfront']
    """

    name = "classification"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        input_size: int = 224,
    ):
        """
        Args:
            weights_path: Path to fine-tuned .pth file. If None, uses
                ~/.cache/hnivision/resnet/best_resnet50_model.pth.
            device: 'cuda', 'mps', 'cpu', or None for auto-detect (cuda > mps > cpu).
            input_size: Square input size (default 224 to match training).
        """
        self.input_size = input_size
        self.model_name = "resnet50-finetuned-hni-7class"

        # Resolve weights path
        wp = _resolve_weights_path(weights_path)
        self.weights_path = str(wp)

        # Lazy imports
        try:
            import torch
            import torchvision
            from torchvision.transforms import (
                CenterCrop, Compose, Normalize, Resize, ToTensor,
            )
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "Classification requires `torchvision`. Install with:\n"
                "    pip install 'hnivision[classification]'"
            ) from e

        # Device detect
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self._torch = torch

        # Build model architecture (must match the trained one exactly)
        model = torchvision.models.resnet50(weights=None)  # weights=None: no ImageNet pretrain
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_NAMES)),  # 7 classes
        )

        # Load fine-tuned weights
        state_dict = torch.load(self.weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self._model = model

        # Preprocessing (matches notebook's data_transforms but with CenterCrop
        # instead of RandomCrop, and no horizontal flip — deterministic inference)
        self._transform = Compose([
            Resize((256, 256)),
            CenterCrop(input_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def extract(self, image: ImageInput) -> ClassificationOutput:
        """Run ResNet on one image, return class + probabilities."""
        img = self.load_image(image)
        tensor = self._transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with self._torch.no_grad():
            logits = self._model(tensor)
            probs = self._torch.nn.functional.softmax(logits, dim=1)[0]

        probs_list = probs.cpu().tolist()
        pred_idx = int(probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx].item())

        all_probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs_list)}

        return ClassificationOutput(
            predicted_class=pred_class,
            predicted_class_idx=pred_idx,
            confidence=confidence,
            all_probs=all_probs,
        )

    def to_hni(self, output: ClassificationOutput) -> HNIResult:
        """Map predicted class → all 4 HNI levels via CLASS_TO_HNI lookup."""
        result = HNIResult()
        cls = output.predicted_class
        mapping = CLASS_TO_HNI.get(cls, {})
        evidence_base = f"ResNet classified scene as '{cls}' (confidence {output.confidence:.2f})"

        # Human
        if mapping.get("human"):
            result.human = HumanLevel(
                present=True,
                tags=mapping.get("human_tags", []),
                evidence=evidence_base,
            )

        # Nature
        if mapping.get("nature_tags"):
            tags = mapping["nature_tags"]
            result.nature = NatureLevel(
                tags=tags,
                dominant=tags[0],
                evidence=evidence_base,
            )

        # Activity
        if mapping.get("activity_tags"):
            tags = mapping["activity_tags"]
            result.activity = ActivityLevel(
                tags=tags,
                dominant=tags[0],
                evidence=evidence_base,
            )

        # Meaning
        if mapping.get("meaning_tags"):
            tags = mapping["meaning_tags"]
            result.meaning = MeaningLevel(
                tags=tags,
                dominant=tags[0],
                evidence=evidence_base,
            )

        # Overall confidence bucket (used by HNIResult meta)
        if output.confidence >= 0.7:
            result.confidence = "high"
        elif output.confidence >= 0.4:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        return result
