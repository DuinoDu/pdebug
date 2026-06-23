"""Convenience image IO helpers built on Piata Input and Output."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .input import Input
from .output import Output


@dataclass
class ImageBatch:
    """Decoded images plus optional source table metadata."""

    images: List[np.ndarray]
    metadata: List[Dict[str, object]] = field(default_factory=list)
    table: Optional[object] = None
    source_type: str = ""


def read_image_batch(
    input_path: Union[str, Path, np.ndarray],
    *,
    image_col: str = "camera_left",
    lance_kwargs: Optional[Dict[str, object]] = None,
    to_rgb: bool = True,
) -> ImageBatch:
    """Read image, image directory, URL, ndarray, or Lance dataset."""
    if isinstance(input_path, np.ndarray):
        reader = Input(input_path, name="image", to_rgb=to_rgb).get_reader()
        image = next(reader)
        return ImageBatch(
            images=[image],
            metadata=[{"image_name": "array"}],
            source_type="image",
        )

    input_str = str(input_path)
    path = Path(input_str)
    if path.suffix == ".lance":
        return load_lance_batch(
            input_str,
            image_col=image_col,
            reader_kwargs=lance_kwargs,
        )

    if path.is_dir():
        reader = Input(input_str, name="imgdir", to_rgb=to_rgb).get_reader()
        metadata = []
        images = []
        for _ in range(len(reader)):
            image = next(reader)
            images.append(image)
            metadata.append({"image_name": reader.filename})
        reader.reset()
        return ImageBatch(
            images=images,
            metadata=metadata,
            source_type="imgdir",
        )

    reader = Input(input_str, name="image", to_rgb=to_rgb).get_reader()
    image = next(reader)
    return ImageBatch(
        images=[image],
        metadata=[{"image_name": input_str}],
        source_type="image",
    )


def load_lance_batch(
    dataset_path: Union[str, Path],
    *,
    image_col: str = "camera_left",
    reader_kwargs: Optional[Dict[str, object]] = None,
) -> ImageBatch:
    """Load a Lance dataset into memory with decoded frames."""
    parameters = dict(reader_kwargs or {})
    image_col = str(parameters.pop("image_col", image_col))
    reader = Input(
        str(dataset_path),
        name="lance_vita",
        image_col=image_col,
        **parameters,
    ).get_reader()

    metadata: List[Dict[str, object]] = []
    images: List[np.ndarray] = []
    for _ in range(len(reader)):
        entry = next(reader)
        metadata.append(entry)
        images.append(entry["image"])
    reader.reset()

    table = getattr(reader, "_table", None)
    if table is None:  # pragma: no cover - Lance reader exposes _table.
        import lance

        table = lance.dataset(str(dataset_path)).to_table()

    return ImageBatch(
        images=images,
        metadata=metadata,
        table=table,
        source_type="lance",
    )


def write_json_result(output_path: Union[str, Path], payload) -> None:
    """Write JSON output through Piata Output."""
    Output(payload, name="json").save(str(output_path))


def write_lance_column(
    table,
    column_name: str,
    column_values,
    output_path: Union[str, Path],
) -> Path:
    """Write one result column into a Lance dataset through Piata Output."""
    Output(
        table,
        name="lance_column",
        column_name=column_name,
        column_values=column_values,
    ).save(str(output_path))
    return Path(output_path)
