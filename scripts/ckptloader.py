#!/usr/bin/env python
import os
import sys
from typing import Optional

import ppico
import typer

DEPTH_ROOT = os.path.join(
    ppico.__path__[0], "../3rdparty/Monocular-Depth-Estimation-Toolbox"
)
sys.path.append(DEPTH_ROOT)

from pdebug.types import Checkpoint


# @task(name="my-tool")
def main(
    ckpt: str,
    debug: bool = False,
    compare: str = None,
):
    """Load checkpoint and play with it."""
    typer.echo(typer.style(f"loading {ckpt}", fg=typer.colors.GREEN))
    checkpoint = Checkpoint(ckpt)
    print(checkpoint)

    # checkpoint -= "human_seg"
    # bbb = Checkpoint("b.ckpt")
    # bbb -= "backbone"
    # bbb -= "mono_depth"
    # checkpoint.update(bbb)
    # checkpoint.save("merged.ckpt")

    if compare:
        bbb = Checkpoint(compare)
        print(bbb)
        checkpoint -= "human_seg"
        checkpoint -= "mono_depth"
        bbb -= "human_seg"
        bbb -= "mono_depth"
        if checkpoint != bbb:
            checkpoint.compare(bbb, save_hist=False)

    ckpt = checkpoint.checkpoint
    if debug:
        __import__("IPython").embed()


if __name__ == "__main__":
    typer.run(main)
