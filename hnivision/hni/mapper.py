"""Aggregation logic to merge 5-method outputs into one HNIResult.

The mapper is the climax of HNIVision's architecture: it takes the
per-method HNIResults and produces a single, richer HNIResult that
combines the strengths of each method.

Merge strategy:
  - Tags: union across methods, deduplicated (lowercased, order-preserving)
  - Dominant: trust hierarchy (VLM > method-specific best)
  - Pixel shares: from segmentation only
  - Summary + HNI meta fields: from VLM only
  - Evidence: concatenated per-method, annotated with method name
"""

from __future__ import annotations

from typing import Dict, List, Optional

from hnivision.hni.schema import (
    ActivityLevel,
    HNIResult,
    HumanLevel,
    MeaningLevel,
    NatureLevel,
)


# Trust hierarchy for picking `dominant` field, per level
DOMINANT_PREFERENCE: Dict[str, List[str]] = {
    "human":    ["vlm", "detection", "labeling", "classification"],
    "nature":   ["vlm", "segmentation", "labeling", "classification"],
    "activity": ["vlm", "labeling", "classification"],
    "meaning":  ["vlm", "labeling", "classification"],
}


def _pick_dominant(
    per_method: Dict[str, HNIResult],
    level: str,
) -> Optional[str]:
    """Pick the dominant field for a level using trust hierarchy."""
    for method_name in DOMINANT_PREFERENCE.get(level, []):
        if method_name not in per_method:
            continue
        level_obj = getattr(per_method[method_name], level)
        if level_obj.dominant:
            return level_obj.dominant
    return None


def _collect_tags(per_method: Dict[str, HNIResult], level: str) -> List[str]:
    """Collect and deduplicate tags from all methods (case-insensitive, order-preserving)."""
    seen = set()
    ordered: List[str] = []
    for r in per_method.values():
        for tag in getattr(r, level).tags:
            t = (tag or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


def _collect_evidence(per_method: Dict[str, HNIResult], level: str) -> Optional[str]:
    """Concatenate per-method evidence with method-name annotation."""
    parts = []
    for method, r in per_method.items():
        ev = getattr(r, level).evidence
        if ev:
            parts.append(f"[{method}] {ev}")
    return " | ".join(parts) if parts else None


def merge_hni_results(per_method: Dict[str, HNIResult]) -> HNIResult:
    """Merge per-method HNIResults into one composite HNIResult.

    Args:
        per_method: Dict mapping method_name (e.g. "vlm", "detection") to HNIResult.

    Returns:
        A new HNIResult that combines all methods' evidence.
    """
    merged = HNIResult()

    # --- HUMAN ---
    human_tags = _collect_tags(per_method, "human")
    human_present = any(r.human.present for r in per_method.values())
    if human_tags or human_present:
        # Count: prefer VLM > Detection > Labeling
        count: Optional[str] = None
        for method in ["vlm", "detection", "labeling"]:
            if method in per_method and per_method[method].human.count:
                count = per_method[method].human.count
                break
        merged.human = HumanLevel(
            present=human_present,
            count=count,
            tags=human_tags,
            evidence=_collect_evidence(per_method, "human"),
        )

    # --- NATURE ---
    nature_tags = _collect_tags(per_method, "nature")
    pixel_shares: Dict[str, float] = {}
    if "segmentation" in per_method:
        pixel_shares = dict(per_method["segmentation"].nature.pixel_shares)
    if nature_tags or pixel_shares:
        merged.nature = NatureLevel(
            tags=nature_tags,
            dominant=_pick_dominant(per_method, "nature"),
            pixel_shares=pixel_shares,
            evidence=_collect_evidence(per_method, "nature"),
        )

    # --- ACTIVITY ---
    activity_tags = _collect_tags(per_method, "activity")
    if activity_tags:
        merged.activity = ActivityLevel(
            tags=activity_tags,
            dominant=_pick_dominant(per_method, "activity"),
            evidence=_collect_evidence(per_method, "activity"),
        )

    # --- MEANING ---
    meaning_tags = _collect_tags(per_method, "meaning")
    if meaning_tags:
        merged.meaning = MeaningLevel(
            tags=meaning_tags,
            dominant=_pick_dominant(per_method, "meaning"),
            evidence=_collect_evidence(per_method, "meaning"),
        )

    # --- META FIELDS (from VLM) ---
    if "vlm" in per_method:
        vlm = per_method["vlm"]
        merged.summary = vlm.summary
        merged.hni_present = vlm.hni_present
        merged.hni_strength = vlm.hni_strength
        merged.confidence = vlm.confidence

    # Provenance
    merged.source_method = "merged(" + ",".join(sorted(per_method.keys())) + ")"

    return merged
