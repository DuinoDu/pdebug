#!/usr/bin/env python
from typing import List, Optional

from pdebug.data_types import NerfCamera, Open3dCameraTrajectory

import numpy as np
import typer


def test_o3d(filepath):
    try:
        import open3d as o3d
    except ModuleNotFoundError as e:
        return
    trajectory = o3d.io.read_pinhole_camera_trajectory(filepath)
    print(trajectory)


def nerf_to_open3d(nerf_json, output):
    if not output:
        output = nerf_json[:-5] + "_open3d.json"
    typer.echo(typer.style(f"loading {nerf_json}", fg=typer.colors.GREEN))

    nerf_camera = NerfCamera.fromfile(nerf_json)
    print(f"camera length: {len(nerf_camera)}")
    o3d_camera = Open3dCameraTrajectory.from_nerf(nerf_camera)
    o3d_camera.dump(output)
    typer.echo(typer.style(f"save to {output}", fg=typer.colors.GREEN))
    test_o3d(output)


def open3d_to_nerf(jsonfile, output):
    if not output:
        output = jsonfile[:-5] + "_nerf.json"
    typer.echo(typer.style(f"loading {jsonfile}", fg=typer.colors.GREEN))
    test_o3d(jsonfile)

    o3d_camera = Open3dCameraTrajectory.fromfile(jsonfile)
    print(f"open3d trajectory length: {len(o3d_camera)}")
    nerf_camera = NerfCamera.from_o3d_trajectory(o3d_camera)
    nerf_camera.dump(output)
    typer.echo(typer.style(f"save to {output}", fg=typer.colors.GREEN))


# @task(name="my-tool")
def main(
    jsonfile: str,
    to_open3d: bool = False,
    to_nerf: bool = False,
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Convert nerf transforms json to open3d camera trajectory json."""
    assert to_open3d + to_nerf == 1
    if to_open3d:
        nerf_to_open3d(jsonfile, output)
    elif to_nerf:
        open3d_to_nerf(jsonfile, output)
    else:
        raise ValueError("set `--to-open3d` or `--to-nerf`.")


if __name__ == "__main__":
    typer.run(main)
