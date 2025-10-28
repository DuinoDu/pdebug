import glob
import os
from collections import defaultdict

import cv2
import numpy as np
import tqdm
import typer
from tabulate import tabulate

os.environ["PDEBUG_DISABLE_PHOTOID"] = "1"
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.semantic_types import load_categories, load_colors
from pdebug.visp import draw


@otn_manager.NODE.register(name="statistic_semseg")
def statistic_semseg(
    path: str,
    name: str = None,
    show_label_hist: bool = False,
    output: str = None,
    vis_output: str = "tmp_statistic_semseg",
):
    """Statistic data."""
    typer.echo(typer.style(f"loading {path}", fg=typer.colors.GREEN))

    input_kwargs = {}
    batch_size = 1
    if name is None and path.endswith(".parquet"):
        name = "cruise_rgbd_semseg"
        batch_size = 16
        input_kwargs["batch_size"] = batch_size
        input_kwargs["num_readers"] = 8
        input_kwargs["num_workers"] = 8
    if "*" in path:
        path = glob.glob(path)

    reader = Input(path, name=name, **input_kwargs).get_reader()

    print(f"data length: {len(reader) * batch_size}")

    if show_label_hist:
        t = tqdm.tqdm(total=len(reader), desc="label hist")
        label_hist = defaultdict(int)

        for data_dict in reader:
            t.update()
            for label_sample in data_dict["label"].numpy():
                # 0 is VOID for semanticpcd, same as ADE20k
                for label_value in np.unique(label_sample):
                    label_hist[label_value] += 1

        table_data = [(k, v) for k, v in label_hist.items()]
        table_data = sorted(table_data, key=lambda input: input[0])
        print(
            "\n"
            + tabulate(
                table_data,
                headers=["Label Value", "Num Images"],
                tablefmt="fancy_grid",
            )
        )

    return ""


if __name__ == "__main__":
    typer.run(statistic_semseg)
