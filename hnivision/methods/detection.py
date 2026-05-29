"""Object detection method (Method 3) — wraps YOLO via Ultralytics.

For HNI mapping, only 'person' detections contribute to HumanLevel. Other COCO
classes are preserved in raw_output for downstream analysis but not mapped.
"""

from __future__ import annotations

# Apple Silicon (mps) workaround: PyTorch's MPS backend has not yet
# implemented torchvision::nms (the NMS op YOLO uses). Enable CPU fallback
# so MPS works for the operators it DOES support, and NMS quietly falls
# back to CPU. Ref: https://github.com/pytorch/pytorch/issues/77764
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.schema import HNIResult, HumanLevel


# --- Helpers ---------------------------------------------------------------

def _auto_device() -> str:
    """Pick best available inference device (cuda > mps > cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _person_count_bucket(n: int) -> str:
    """Map a person count integer → the HNI presence bucket string."""
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 10:
        return "6-10"
    return "10+"


def _person_presence_tag(n: int) -> Optional[str]:
    """Map person count → presence-type tag."""
    if n <= 0:
        return None
    if n == 1:
        return "individuals"
    if n <= 5:
        return "groups"
    return "crowd"


# --- Method-specific output schemas ----------------------------------------

class BoundingBox(BaseModel):
    """A single detected object."""
    class_name: str          # e.g. "person", "dog", "bench"
    class_id: int            # COCO class id
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.5


class DetectionOutput(BaseModel):
    """Raw output of one image's object detection."""
    boxes: List[BoundingBox] = Field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)

    @property
    def total_objects(self) -> int:
        return len(self.boxes)

    @property
    def is_abnormal_count(self) -> bool:
        return len(self.boxes) > 50  # likely false positives if 50+

    def count_class(self, class_name: str, min_confidence: float = 0.0) -> int:
        """Count objects of a class above a confidence threshold."""
        return sum(
            1 for b in self.boxes
            if b.class_name == class_name and b.confidence >= min_confidence
        )

    def by_class(self) -> dict:
        """Group boxes by class name (returns dict[str, list[BoundingBox]])."""
        from collections import defaultdict
        groups = defaultdict(list)
        for b in self.boxes:
            groups[b.class_name].append(b)
        return dict(groups)


# --- The Method class ------------------------------------------------------

class Detection(BaseHNIMethod):
    """Object detection via YOLO (Ultralytics).

    Example:
        >>> det = Detection()
        >>> output = det.extract("park.jpg")
        >>> output.count_class("person", min_confidence=0.5)
        2
        >>> hni = det.to_hni(output)
        >>> hni.human.count
        '2-5'
    """

    name = "detection"

    def __init__(
        self,
        model_name: str = "yolo26s.pt",
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
        person_conf_threshold: float = 0.5,
    ):
        """
        Args:
            model_name: Ultralytics model name (auto-downloads) or path to .pt.
            conf_threshold: Minimum confidence to keep a detection in extract().
            imgsz: YOLO input image size.
            device: 'cuda' / 'mps' / 'cpu' / None for auto-detect.
            person_conf_threshold: Higher threshold used in to_hni() — only
                count persons above this confidence as truly "present". Matches
                the 0.5 cutoff used in the HNI_Yolo26s.ipynb pipeline.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device or _auto_device()
        self.person_conf_threshold = person_conf_threshold

        # Lazy import — only triggered when Detection is actually used
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Detection requires `ultralytics`. Install with:\n"
                "    pip install 'hnivision[detection]'"
            ) from e

        self._model = YOLO(model_name)

    def extract(self, image: ImageInput) -> DetectionOutput:
        """Run YOLO on one image."""
        img = self.load_image(image)

        results = self._model.predict(
            source=img,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        boxes: List[BoundingBox] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                boxes.append(BoundingBox(
                    class_name=self._model.names[cls_id],
                    class_id=cls_id,
                    confidence=float(box.conf[0]),
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                ))

        return DetectionOutput(boxes=boxes, image_size=img.size)

    def to_hni(self, output: DetectionOutput) -> HNIResult:
        """Translate DetectionOutput → HNIResult (fills only HumanLevel)."""
        n_persons = output.count_class("person", self.person_conf_threshold)

        result = HNIResult()
        if n_persons > 0:
            presence_tag = _person_presence_tag(n_persons)
            result.human = HumanLevel(
                present=True,
                count=_person_count_bucket(n_persons),
                tags=[presence_tag] if presence_tag else [],
                evidence=(
                    f"YOLO detected {n_persons} 'person' object(s) "
                    f"with confidence ≥ {self.person_conf_threshold}"
                ),
            )
        # nature, activity, meaning all remain at default (empty)
        return result
