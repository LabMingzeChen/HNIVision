"""HNIVision — a unified Python toolkit for human-nature interaction extraction.

Five computer vision methods (labeling, classification, detection, segmentation,
vision-LLM) translated into a single 4-level HNI evidence schema, with a
top-level Pipeline that runs all 5 and merges their outputs.
"""

from hnivision.base import BaseHNIMethod
from hnivision.hni import (
    ActivityLevel,
    HNIResult,
    HumanLevel,
    MeaningLevel,
    NatureLevel,
    merge_hni_results,
)
from hnivision.pipeline import Pipeline, PipelineOutput

__version__ = "0.1.0-pre"
__author__ = "Mingze Chen"
__license__ = "MIT"

__all__ = [
    "BaseHNIMethod",
    "HNIResult",
    "Pipeline",
    "PipelineOutput",
    "HumanLevel",
    "NatureLevel",
    "ActivityLevel",
    "MeaningLevel",
    "merge_hni_results",
]
