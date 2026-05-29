"""HNI schema, mapper, and (future) evaluator."""

from hnivision.hni.mapper import merge_hni_results
from hnivision.hni.schema import (
    ActivityLevel,
    HNIResult,
    HumanLevel,
    MeaningLevel,
    NatureLevel,
)

__all__ = [
    "HNIResult",
    "HumanLevel",
    "NatureLevel",
    "ActivityLevel",
    "MeaningLevel",
    "merge_hni_results",
]
