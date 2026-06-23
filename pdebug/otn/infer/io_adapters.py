"""Input/output adapters for image inference nodes."""
from __future__ import annotations
from typing import Dict, Optional, Union

from pdebug.otn.inference import ImageInferenceRunner
from pdebug.piata import (
    read_image_batch,
    write_json_result,
    write_lance_column,
)


def run_image_input(
    runner: ImageInferenceRunner,
    input_path: str,
    output: Optional[str] = None,
    *,
    output_key: str,
    lance_image_col: str = "camera_left",
    lance_kwargs: Optional[Dict[str, object]] = None,
    empty_message: str = "No images found",
    missing_output_message: str = (
        "`output` is required when writing Lance results."
    ),
    **infer_kwargs,
) -> Union[Dict[str, object], str]:
    """Run an image runner over path/dir/URL input or a Lance dataset."""
    batch = read_image_batch(
        input_path,
        image_col=lance_image_col,
        lance_kwargs=lance_kwargs,
    )

    if batch.source_type == "lance":
        if output is None:
            raise ValueError(missing_output_message)
        results = runner.run_batch(batch.images, **infer_kwargs)
        written = write_lance_column(
            batch.table,
            output_key,
            results,
            output,
        )
        return str(written)

    if not batch.images:
        raise RuntimeError(f"{empty_message} at {input_path}")
    result = runner.run_one(batch.images[0], index=0, **infer_kwargs)
    if output:
        write_json_result(output, result)
        return output
    return result
