#!/usr/bin/env python
from typing import Optional

from pdebug.data_types import PointcloudTensor

import typer


# @task(name="my-tool")
def main(
    txtfile: str,
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Convert Point3d in colmap to open3d pcd."""
    if not output:
        output = txtfile[:-4] + ".pcd"
    pcd = PointcloudTensor.from_colmap_point3d(txtfile)
    pcd.to_open3d(output)
    print(f"save to {output}")


if __name__ == "__main__":
    typer.run(main)
