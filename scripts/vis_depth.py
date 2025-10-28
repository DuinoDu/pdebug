#!/usr/bin/env python
import os
from glob import glob
from pathlib import Path
from typing import Optional

from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer


# @task(name="my-tool")
def main(
    imgdir: Path,
    depdir: Path,
    image_ext: str = ".jpg",
    depth_ext: str = ".png",
    generate_split: bool = False,
    norm_depth_replica: bool = False,
    split_str: str = " ",
    sync_file: str = None,
    cmap: str = "magma_r",  # "magma, jet, magma_r"
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Visualize depth."""
    if not output:
        output = "vis_depth_output"

    imgfiles = sorted(glob(str(imgdir / f"*{image_ext}")))
    depfiles = sorted(glob(str(depdir / f"*{depth_ext}")))

    if not imgfiles or not depfiles:
        print("find empty files, please check filename.")
        return

    if len(imgfiles) != len(depfiles):
        print(
            f"len of images {len(imgfiles)} != len of depth files({len(depfiles)})"
        )

    if generate_split:
        if not output.endswith(".txt"):
            output += ".txt"
        with open(output, "w") as fid:
            t = tqdm.tqdm(total=len(imgfiles))
            for imgfile, depfile in zip(imgfiles, depfiles):
                t.update()
                fid.write(f"{imgfile}{split_str}{depfile}\n")
        print(f"saved to {output}")
        return

    if sync_file:
        print(f"loading sync file: {sync_file}")
        # [rgb, left, right, depth]
        sync_info = [
            line.strip().split(" ")
            for line in open(sync_file, "r").readlines()
        ]
        imgfiles = [
            os.path.join(imgdir, os.path.basename(l[0])) for l in sync_info
        ]
        depfiles = [
            os.path.join(depdir, os.path.basename(l[3])) for l in sync_info
        ]

    os.makedirs(output, exist_ok=True)
    t = tqdm.tqdm(total=len(imgfiles))
    for imgfile, depfile in zip(imgfiles, depfiles):
        t.update()
        image = cv2.imread(imgfile)
        depth = cv2.imread(depfile, cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))

        if norm_depth_replica:
            depth = depth.astype(np.float32) * 16 / 256

        vis_depth = draw.depth(
            depth, image, vertical=False, max_depth=1000, cmap=cmap, hist=True
        )
        savename = os.path.join(output, os.path.basename(imgfile))
        cv2.imwrite(savename, vis_depth)


if __name__ == "__main__":
    typer.run(main)
