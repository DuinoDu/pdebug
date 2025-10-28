import json
import os
from pathlib import Path
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input

import cv2
import tqdm
import typer


@otn_manager.NODE.register(name="spatialmp4-to-imgdir")
def main(
    input_path: str,
    fps: int = None,
    output: str = "tmp_spatialmp4",
    cache: bool = True,
    camera: str = None,
):
    """Parse spatialmp4 to imgdir.

    Args:
        input_path: Path to spatialmp4 file
        fps: Target frames per second for output (if None, use original fps)
        output: Output directory path
        cache: Whether to use caching (skip if output already exists)
        camera: Json file to save camera parameter from spatialmp4

    Returns:
        Path: Output directory path
    """
    reader = Input(input_path, name="spatialmp4").get_reader()
    typer.echo(
        typer.style(
            f"camera intrinsic {reader.intrinsic.data}", fg=typer.colors.GREEN
        )
    )

    output = Path(output)
    output.mkdir(exist_ok=True)

    if fps and fps > 0:
        sample_interval = reader.fps // fps
        print(f"fps: {reader.fps} => {fps}")
    else:
        sample_interval = 1

    output_num = len(reader) // sample_interval
    if cache and len(os.listdir(output)) >= output_num:
        print(f"{output} exists, skip")
        return output

    t = tqdm.tqdm(total=len(reader))
    for frame in reader:
        t.update()
        if reader.cnt % sample_interval != 0:
            continue
        cv2.imwrite(str(output / reader.filename), frame)

    if camera:
        camera_data = {
            "cam_K": reader.intrinsic.data.tolist(),
            "depth_scale": 1.0,
        }
        with open(camera, "w") as fid:
            json.dump(camera_data, fid, indent=2)

    print(f"saved to {output}")


if __name__ == "__main__":
    typer.run(main)
