"""Image labeling method (Method 1) — wraps Google Cloud Vision API.

Returns multi-label tags with confidence scores. Filters through a hybrid
WordNet + manual whitelist/blacklist pipeline (matches HNI_google_vision.ipynb).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional, Set

from pydantic import BaseModel, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.schema import ActivityLevel, HNIResult, HumanLevel, MeaningLevel, NatureLevel


# --- Filter configuration (1:1 from HNI_google_vision.ipynb) ---

KEEP_WORDS: Set[str] = {
    "garden", "park", "botanical garden", "pond", "fountain", "water feature",
    "tree", "plant", "grass", "flower", "building", "architecture",
    "person", "people", "sky", "water", "river", "lake", "bench", "road",
    "path", "walkway", "bridge", "playground", "lawn", "vegetation", "landscape",
}

REMOVE_WORDS: Set[str] = {
    "reflection", "autumn", "winter", "summer", "spring", "season",
    "tourism", "travel", "vacation",
    "morning", "evening", "daytime", "night", "shadow", "light", "symmetry",
}

CONCRETE_LEXNAMES: Set[str] = {
    "noun.artifact", "noun.object", "noun.person", "noun.animal",
    "noun.plant", "noun.location", "noun.group", "noun.substance",
    "noun.body", "noun.food", "noun.event", "noun.act",
}

ABSTRACT_LEXNAMES: Set[str] = {
    "noun.time", "noun.attribute", "noun.state", "noun.feeling",
    "noun.cognition", "noun.process", "noun.phenomenon", "noun.motive",
    "noun.relation",
}

# --- HNI level mappings for to_hni() ---

HUMAN_TAGS: Set[str] = {
    "person", "people", "human", "child", "children", "crowd", "family",
    "people in nature",
}

NATURE_TAGS: Set[str] = {
    "tree", "garden", "park", "lawn", "flower", "water", "river", "lake",
    "vegetation", "landscape", "grass", "plant", "sky", "cloud", "cumulus",
    "pond", "fountain", "waterfront", "woodland", "forest", "beach",
    "wetland", "natural environment", "meteorological phenomenon",
}

MEANING_TAGS: Set[str] = {"leisure", "recreation"}

ACTIVITY_TAGS: Set[str] = {
    "walking", "hiking", "running", "jogging", "cycling", "biking",
    "sitting", "standing", "playing", "swimming", "boating", "fishing",
    "climbing", "picnicking", "picnic", "skating", "rowing", "dancing",
    "gardening", "exercise", "photography",
}


# --- Output schema ---

class Label(BaseModel):
    """A single Google Vision label annotation."""
    description: str
    confidence: float
    rank_original: int  # 1-based rank in the raw API response
    kept_by_filter: bool = True  # passed the WordNet/whitelist/blacklist filter


class LabelingOutput(BaseModel):
    """Output of Labeling on one image."""
    labels: List[Label] = Field(default_factory=list)

    def kept_labels(self) -> List[Label]:
        return [l for l in self.labels if l.kept_by_filter]

    def removed_labels(self) -> List[Label]:
        return [l for l in self.labels if not l.kept_by_filter]

    def top_kept(self, n: int = 10) -> List[Label]:
        return self.kept_labels()[:n]


# --- The Method ---

class Labeling(BaseHNIMethod):
    """Image labeling via Google Cloud Vision API.

    Requirements:
      - `pip install "hnivision[labeling]"`
      - NLTK data: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`
      - GOOGLE_APPLICATION_CREDENTIALS env var (or .env file at project root)

    Example:
        >>> lab = Labeling()
        >>> out = lab.extract("park.jpg")
        >>> [l.description for l in out.top_kept(5)]
        ['Cloud', 'Park', 'Walkway', 'Cumulus', 'Fence']
        >>> hni = lab.to_hni(out)
        >>> hni.nature.tags
        ['cloud', 'park', 'cumulus']
    """

    name = "labeling"

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        top_n: int = 10,
        max_results: int = 50,
    ):
        """
        Args:
            credentials_path: Path to GCP service account JSON. If None,
                tries to load from .env at the project root.
            top_n: How many kept labels to retain for HNI mapping.
            max_results: Max labels to request from the API per image.
        """
        self.top_n = top_n
        self.max_results = max_results
        self.model_name = "google-cloud-vision-v1"

        # --- Resolve credentials ---
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        else:
            # Auto-load .env by walking up from this file (robust regardless of cwd)
            try:
                from dotenv import load_dotenv
                here = Path(__file__).resolve()
                # Search up to 4 levels up for a .env file
                for parent in [here.parent, *here.parents]:
                    env_file = parent / ".env"
                    if env_file.exists():
                        load_dotenv(env_file)
                        break
            except ImportError:
                pass

        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS not set. Set it via:\n"
                "  1. Labeling(credentials_path='/path/to/key.json')\n"
                "  2. export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json\n"
                "  3. Add to .env at the project root"
            )

        # --- Lazy imports ---
        try:
            from google.cloud import vision
        except ImportError as e:
            raise ImportError(
                "Labeling requires `google-cloud-vision`. Install with:\n"
                "    pip install 'hnivision[labeling]'"
            ) from e

        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("tree")  # trigger lazy load to catch missing data
        except (ImportError, LookupError) as e:
            raise ImportError(
                "Labeling requires NLTK WordNet. Run:\n"
                "    python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
            ) from e

        self._vision = vision
        self._client = vision.ImageAnnotatorClient()
        self._wn = wn

    # --- Filter (matches notebook's is_meaningful_visual_label) ---

    def _is_meaningful(self, label_name: str) -> bool:
        name = (label_name or "").strip().lower()
        if not name:
            return False
        if name in KEEP_WORDS:
            return True
        if name in REMOVE_WORDS:
            return False

        wn_key = name.replace(" ", "_")
        synsets = self._wn.synsets(wn_key, pos=self._wn.NOUN)
        if not synsets and " " in name:
            synsets = self._wn.synsets(name.split()[-1], pos=self._wn.NOUN)
        if not synsets:
            return True  # unknown → default keep

        lexnames = {s.lexname() for s in synsets[:3]}
        if lexnames & CONCRETE_LEXNAMES:
            return True
        if lexnames & ABSTRACT_LEXNAMES:
            return False
        return True

    # --- BaseHNIMethod interface ---

    def extract(self, image: ImageInput) -> LabelingOutput:
        """Call Google Vision label_detection on one image."""
        # Vision API wants raw bytes
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                content = f.read()
        else:
            # PIL Image — encode to JPEG bytes
            img = image.convert("RGB") if hasattr(image, "convert") else image
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            content = buf.getvalue()

        vision_image = self._vision.Image(content=content)
        response = self._client.label_detection(
            image=vision_image,
            max_results=self.max_results,
        )

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        labels: List[Label] = []
        for rank, ann in enumerate(response.label_annotations, start=1):
            desc = ann.description.strip()
            labels.append(Label(
                description=desc,
                confidence=round(float(ann.score), 4),
                rank_original=rank,
                kept_by_filter=self._is_meaningful(desc),
            ))

        return LabelingOutput(labels=labels)

    def to_hni(self, output: LabelingOutput) -> HNIResult:
        """Map filtered labels into HNI levels via the tag lookup tables."""
        kept_lower = [l.description.lower() for l in output.top_kept(self.top_n)]

        result = HNIResult()

        # --- Human ---
        human_matches = [t for t in kept_lower if t in HUMAN_TAGS]
        if human_matches:
            result.human = HumanLevel(
                present=True,
                tags=human_matches,
                evidence=f"Google Vision tagged: {', '.join(human_matches)}",
            )

        # --- Nature ---
        nature_matches = [t for t in kept_lower if t in NATURE_TAGS]
        if nature_matches:
            result.nature = NatureLevel(
                tags=nature_matches,
                dominant=nature_matches[0],  # highest-confidence kept tag first
                evidence=f"Google Vision tagged: {', '.join(nature_matches)}",
            )

        # --- Meaning ---
        meaning_matches = [t for t in kept_lower if t in MEANING_TAGS]
        if meaning_matches:
            result.meaning = MeaningLevel(
                tags=meaning_matches,
                dominant=meaning_matches[0],
                evidence=f"Google Vision tagged: {', '.join(meaning_matches)}",
            )

        # --- Activity ---
        activity_matches = [t for t in kept_lower if t in ACTIVITY_TAGS]
        if activity_matches:
            result.activity = ActivityLevel(
                tags=activity_matches,
                dominant=activity_matches[0],
                evidence=f"Google Vision tagged: {', '.join(activity_matches)}",
            )

        return result
