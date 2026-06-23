"""Reusable inference runners independent of input/output storage."""
from __future__ import annotations
from typing import Callable, Dict, Iterable, List


class ImageInferenceRunner:
    """Run an image inference function over decoded image objects."""

    def __init__(
        self,
        infer_one: Callable[..., Dict[str, object]],
    ) -> None:
        self.infer_one = infer_one

    def run_one(
        self,
        image,
        *,
        index: int = 0,
        **infer_kwargs,
    ) -> Dict[str, object]:
        """Run inference on one decoded image."""
        return self.infer_one(image, index=index, **infer_kwargs)

    def run_batch(
        self,
        images: Iterable[object],
        **infer_kwargs,
    ) -> List[Dict[str, object]]:
        """Run inference on already loaded images."""
        return [
            self.run_one(image, index=index, **infer_kwargs)
            for index, image in enumerate(images)
        ]
