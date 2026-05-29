"""HNIVision method implementations.

Each method subclasses hnivision.BaseHNIMethod and implements:
    extract(image) → method-specific raw output
    to_hni(output) → unified HNIResult
"""

from hnivision.methods.classification import (
    Classification,
    ClassificationOutput,
)
from hnivision.methods.detection import (
    BoundingBox,
    Detection,
    DetectionOutput,
)
from hnivision.methods.labeling import (
    Label,
    Labeling,
    LabelingOutput,
)
from hnivision.methods.segmentation import (
    PixelShares,
    Segmentation,
    SegmentationOutput,
)
from hnivision.methods.vlm import (
    VLM,
    VLMOutput,
)

__all__ = [
    # Detection (Method 3)
    "Detection",
    "DetectionOutput",
    "BoundingBox",
    # Labeling (Method 1)
    "Labeling",
    "LabelingOutput",
    "Label",
    # Segmentation (Method 4)
    "Segmentation",
    "SegmentationOutput",
    "PixelShares",
    # Classification (Method 2)
    "Classification",
    "ClassificationOutput",
    # VLM (Method 5)
    "VLM",
    "VLMOutput",
]
