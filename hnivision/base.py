"""BaseHNIMethod — the abstract contract every method follows.

Each concrete method (Labeling, Classification, Detection, Segmentation, VLM)
subclasses BaseHNIMethod and implements:

  1. extract(image)       — run inference, return method-specific raw output
  2. to_hni(raw_output)   — translate raw output to a unified HNIResult

Subclasses get extract_hni(image) — which does both — for free.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from PIL import Image

from hnivision.hni.schema import HNIResult

# Accept file paths, PIL images, or anything PIL.Image.open can read
ImageInput = Union[str, Path, Image.Image]


class BaseHNIMethod(ABC):
    """Abstract base class for all HNI extraction methods."""

    # Subclasses should override these
    name: str = "base"            # e.g., "detection", "labeling"
    model_name: str = "unknown"   # e.g., "yolov26-s", "google-vision-v1"

    def __init__(self, **kwargs):
        """Subclasses use __init__ to load model weights, init API clients, etc.

        Heavy lifting (downloading model, opening API connection) belongs here,
        NOT inside extract() — so that running .extract() on 1,000 images
        doesn't re-init the model 1,000 times.
        """
        pass

    @abstractmethod
    def extract(self, image: ImageInput) -> Any:
        """Run the method on a single image.

        Returns method-specific raw output. Each subclass defines what this means:
          - Labeling     → List[Tuple[str, float]]   (tags + confidence)
          - Classification → Tuple[str, float]        (category + confidence)
          - Detection    → List[BoundingBox]
          - Segmentation → PixelShares
          - VLM          → Dict[str, Any]             (JSON-style)
        """
        ...

    @abstractmethod
    def to_hni(self, raw_output: Any) -> HNIResult:
        """Translate raw method output into a unified HNIResult.

        Each method fills only the HNI levels it can actually speak to.
        E.g., Detection fills `human.count`, leaves nature/activity/meaning empty.
        """
        ...

    def extract_hni(self, image: ImageInput) -> HNIResult:
        """Convenience: extract + to_hni in one call.

        Annotates the result with the source method name automatically.
        """
        raw = self.extract(image)
        result = self.to_hni(raw)
        result.source_method = self.name
        result.raw_output = raw
        return result

    @staticmethod
    def load_image(image: ImageInput) -> Image.Image:
        """Normalize input to a PIL Image in RGB mode.

        Use this at the top of every extract() implementation so the rest of
        your code doesn't have to handle 3 different input types.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        raise TypeError(
            f"Unsupported image input: {type(image).__name__}. "
            f"Expected str, Path, or PIL.Image.Image."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_name!r})"
