"""HNI evidence schema: the 4-level data model used across all methods.

Every method's output is translated into this schema by its `to_hni()` method.
This is what makes the 5 methods comparable.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HumanLevel(BaseModel):
    """Level 1: Is there a person, and how?"""
    present: bool = False
    count: Optional[str] = None  # "1", "2-5", "6-10", "10+"
    tags: List[str] = Field(default_factory=list)  # ["children", "couple", "crowd"]
    evidence: Optional[str] = None  # free-text justification


class NatureLevel(BaseModel):
    """Level 2: What natural environment is shown?"""
    tags: List[str] = Field(default_factory=list)  # ["lawn", "woodland", "waterfront"]
    dominant: Optional[str] = None
    evidence: Optional[str] = None
    pixel_shares: Dict[str, float] = Field(default_factory=dict)  # segmentation-only


class ActivityLevel(BaseModel):
    """Level 3: What are people doing?"""
    tags: List[str] = Field(default_factory=list)  # ["walking", "viewing", "playing"]
    dominant: Optional[str] = None
    evidence: Optional[str] = None


class MeaningLevel(BaseModel):
    """Level 4: What cultural / experiential meaning does the scene suggest?"""
    tags: List[str] = Field(default_factory=list)  # ["recreation", "aesthetic appreciation"]
    dominant: Optional[str] = None
    evidence: Optional[str] = None


class HNIResult(BaseModel):
    """Unified 4-level Human-Nature Interaction extraction result.

    Each level is independently fillable — some methods only populate some levels.
    Use `HNIResult.merge([result_from_method_a, result_from_method_b, ...])`
    to combine multiple methods' outputs into a single, richer HNIResult.
    """

    # --- The four HNI levels ---
    human: HumanLevel = Field(default_factory=HumanLevel)
    nature: NatureLevel = Field(default_factory=NatureLevel)
    activity: ActivityLevel = Field(default_factory=ActivityLevel)
    meaning: MeaningLevel = Field(default_factory=MeaningLevel)

    # --- Top-level scene summary (optional, usually from VLM) ---
    summary: Optional[str] = None

    # --- HNI present judgement ---
    hni_present: Optional[Literal["yes", "no", "unclear"]] = None
    hni_strength: Optional[Literal["weak", "moderate", "strong"]] = None
    confidence: Optional[Literal["low", "medium", "high"]] = None

    # --- Provenance: which method produced this ---
    source_method: Optional[str] = None  # e.g., "detection", "vlm"
    raw_output: Optional[Any] = None  # original method-specific output (kept for debugging)
