#!/usr/bin/env python
import os
from typing import List, Optional

from pdebug.data_types import Camera, NerfCamera, PointcloudTensor
from pdebug.utils.decorator import mp

import cv2
import tqdm
import typer


# @task(name="my-tool")
def main(
    camera_file: str,
    points_file: str,
    num_workers: int = 6,
    output: Optional[str] = typer.Option("output", help="output name"),
):
    """Capture image using camera and points."""
    if not os.path.exists(output):
        os.makedirs(output)
    pcd = PointcloudTensor.from_open3d(points_file)
    nerf_camera = NerfCamera.fromfile(camera_file)
    cameras: List[Camera] = Camera.from_nerf_camera(
        nerf_camera, use_half_wh_as_cxcy=True
    )

    @mp(nums=num_workers)
    def _process(data):
        t = tqdm.tqdm(total=len(data))
        for camera, frame in data:
            t.update()
            image = camera.capture(pcd.data)
            savename = os.path.join(
                output, os.path.basename(frame["file_path"])
            )
            cv2.imwrite(savename, image)

    data = [(c, f) for c, f in zip(cameras, nerf_camera.frames)]
    _process(data)


if __name__ == "__main__":
    typer.run(main)
