import json
import os
import random
from typing import List

from pdebug.debug.nerf import vis_pose

import numpy as np
import typer

"""
Coffee:
  train: 0-107, 107-203, 203~-1
  valid: 0-7, 7-19, 19~-1
  test:  0-7, 7-18, 18~-1
Car:
  train: 0-108, 108-204, 204~-1
  valid: 0-7, 7-19, 19~-1
  test: 0-7, 7-18, 18~-1
Easyship:
  train: 0-105, 105-203, 203~-1
  valid: 0-7, 7-16, 16~-1
  test: 0-10, 10-22, 22~-1

"""

# @task(name="my-tool")
def main(
    input_json: str,
    start: int = None,
    end: int = None,
    output: str = "output.json",
):
    """Visualize nerf-format camera pose."""
    typer.echo(typer.style(f"loading {input_json}", fg=typer.colors.GREEN))
    with open(input_json, "r") as fid:
        data = json.load(fid)

    data["frames"] = data["frames"][start:end]

    outdir = os.path.dirname(output)
    os.makedirs(outdir, exist_ok=True)
    with open(output, "w") as fid:
        json.dump(data, fid)


if __name__ == "__main__":
    typer.run(main)
