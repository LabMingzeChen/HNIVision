"""HNIVision Pipeline: orchestrator that runs all 5 methods + merger.

Example:
    >>> from hnivision import Pipeline
    >>> pipe = Pipeline()
    >>> out = pipe.extract("park.jpg")
    >>> out.merged.summary
    'Two individuals stand at...'
    >>> out.per_method["detection"].human.count
    '2-5'
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from hnivision.base import BaseHNIMethod, ImageInput
from hnivision.hni.mapper import merge_hni_results
from hnivision.hni.schema import HNIResult


class PipelineOutput(BaseModel):
    """Output of running the full HNIVision pipeline on one image."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    per_method: Dict[str, HNIResult] = Field(default_factory=dict)
    merged: HNIResult = Field(default_factory=HNIResult)


class Pipeline:
    """Run multiple HNIVision methods on an image and merge results.

    Methods loaded eagerly at __init__. Failed initializations are logged
    (not raised) by default — useful when e.g. you don't have a DASHSCOPE
    key but want to run the other 4 methods.

    Example:
        >>> pipe = Pipeline()  # all 5
        >>> out = pipe.extract("park.jpg")
        >>> out.merged.nature.dominant
        'waterfront'

        >>> pipe = Pipeline(methods=["detection", "vlm"])  # selective
        >>> out = pipe.extract("park.jpg")
    """

    DEFAULT_METHODS = ["detection", "labeling", "segmentation", "classification", "vlm"]

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        skip_failed_init: bool = True,
        method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            methods: List of method names to load. Default: all 5.
            skip_failed_init: If True, prints warning instead of raising on init failure.
            method_kwargs: Optional per-method __init__ kwargs.
                Example: {"vlm": {"model_name": "qwen-vl-max-latest"}}
        """
        if methods is None:
            methods = list(self.DEFAULT_METHODS)
        method_kwargs = method_kwargs or {}

        self._methods: Dict[str, BaseHNIMethod] = {}

        for name in methods:
            try:
                kwargs = method_kwargs.get(name, {})
                self._methods[name] = self._build_method(name, **kwargs)
            except Exception as e:
                if skip_failed_init:
                    print(f"⚠️  Skipping method '{name}': {type(e).__name__}: {e}")
                else:
                    raise

    def _build_method(self, name: str, **kwargs) -> BaseHNIMethod:
        """Lazy-import each method module so we don't load all models on import."""
        if name == "detection":
            from hnivision.methods import Detection
            return Detection(**kwargs)
        if name == "labeling":
            from hnivision.methods import Labeling
            return Labeling(**kwargs)
        if name == "segmentation":
            from hnivision.methods import Segmentation
            return Segmentation(**kwargs)
        if name == "classification":
            from hnivision.methods import Classification
            return Classification(**kwargs)
        if name == "vlm":
            from hnivision.methods import VLM
            return VLM(**kwargs)
        raise ValueError(
            f"Unknown method: {name!r}. "
            f"Valid: {self.DEFAULT_METHODS}"
        )

    def extract(self, image: ImageInput) -> PipelineOutput:
        """Run all loaded methods on the image and merge results."""
        per_method: Dict[str, HNIResult] = {}
        for name, method in self._methods.items():
            try:
                per_method[name] = method.extract_hni(image)
            except Exception as e:
                print(f"⚠️  Method '{name}' failed during extract: {type(e).__name__}: {e}")

        merged = merge_hni_results(per_method)
        return PipelineOutput(per_method=per_method, merged=merged)

    def methods(self) -> List[str]:
        """List the methods this pipeline runs."""
        return list(self._methods.keys())

    def __repr__(self) -> str:
        return f"Pipeline(methods={self.methods()})"
