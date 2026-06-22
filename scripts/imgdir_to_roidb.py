#!/usr/bin/env python
import os
import pickle
from typing import Optional

import cv2
import numpy as np
import tqdm
import typer


# @task(name="my-tool")
def main(
    imgdir: str,
    ext: Optional[str] = typer.Option("jpg", help="image file ext"),
    add_boxes: Optional[bool] = typer.Option(False, help="add fake boxes"),
    add_keypoints: Optional[bool] = typer.Option(
        False, help="add fake keypoints"
    ),
    num_keypoints: Optional[int] = typer.Option(
        17, help="fake keypoints number"
    ),
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Generate roidb from imgdir."""
    # typer.echo(typer.style(f"hello, tool", fg=typer.colors.GREEN))

    if output is None:
        output = os.path.splitext(imgdir)[0] + "_roidb.pkl"
    roidb = []
    imgfiles = sorted(
        [
            os.path.join(imgdir, x)
            for x in sorted(os.listdir(imgdir))
            if x.endswith(ext)
        ]
    )
    t = tqdm.tqdm(total=len(imgfiles))

    for ind, imgfile in enumerate(imgfiles):
        t.update()
        roi = dict()
        roi["image"] = imgfile
        roi["image_name"] = os.path.basename(imgfile)
        img = cv2.imread(imgfile)
        h, w = img.shape[:2]
        roi["image_height"] = h
        roi["image_width"] = w
        if add_boxes:
            roi["boxes"] = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        if add_keypoints:
            fake_keypoints = np.ones((1, num_keypoints, 3), dtype=np.float32)
            fake_keypoints[:, :, 2] = 2.0
            roi["keypoints"] = fake_keypoints.reshape(1, -1)
        roidb.append(roi)
    print("saved in %s" % output)
    with open(output, "wb") as fid:
        pickle.dump(roidb, fid)


if __name__ == "__main__":
    typer.run(main)
