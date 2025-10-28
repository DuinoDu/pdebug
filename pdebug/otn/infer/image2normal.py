import os
import sys
from pathlib import Path

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.env import TORCH_INSTALLED
from pdebug.utils.fileio import download_file
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer
from PIL import Image

if TORCH_INSTALLED:
    import torch


@otn_manager.NODE.register(name="stable_normal")
def main(
    input_path: str = None,
    output: str = "tmp_output",
    repo: str = None,
    unittest: bool = False,
    cache: bool = True,
    topk: int = -1,
):
    """Compute normal from rgb images.

    Args:
        input_path: input images folder
        output: output path
        repo: path to https://github.com/hugoycj/StableNormal repo
    """
    output = Path(output)
    output.mkdir(exist_ok=True)

    if unittest:
        import imageio.v3 as iio

        url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        download_file(url, "/tmp/lenna.png")
        input_files = ["/tmp/lenna.png"]
    else:
        input_path = Path(input_path)
        input_files = (
            Input(input_path, name="imgdir", topk=topk).get_reader().imgfiles
        )

    typer.echo(
        typer.style(
            f"loading {len(input_files)} images", fg=typer.colors.GREEN
        )
    )

    sys.path.insert(0, repo)
    from hubconf import StableNormal_turbo

    predictor = StableNormal_turbo(yoso_version="yoso-normal-v1-5")

    t = tqdm.tqdm(total=len(input_files))
    for image_file in input_files:
        t.update()

        input_image = Image.open(image_file)
        normal_image = predictor(input_image)

        savename = output / os.path.basename(image_file)
        if cache and savename.exists():
            continue
        normal_image.save(str(savename))
    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
