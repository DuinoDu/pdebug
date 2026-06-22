#!/usr/bin/env python3
from pathlib import Path
from typing import List, Optional

from pdebug.visp import plotly as draw

import numpy as np
import tqdm
import typer

app = typer.Typer()


@app.command()
def showlog(
    logfiles: List[Path],
    output: Optional[str] = typer.Option("result.html", help="output name"),
):
    """Analysis aipack log"""
    logfiles = logfiles[1:]

    class Metric:
        def __init__(self, lines):
            self._lines = lines
            assert len(lines) > 3
            self.dataset_names = [l for l in lines[0].split(" ") if l != ""][
                1:
            ]
            data_list = []
            self.class_names = []
            for line in lines[1:]:
                data_list.append(
                    [float(l) for l in line.split(" ")[1:] if l != ""]
                )
                self.class_names.append(line.split(" ")[0])
            self.data = np.asarray(data_list)
            assert self.data.shape[1] == len(self.dataset_names)
            assert self.data.shape[0] == len(self.class_names)
            self.num_datasets = len(self.dataset_names)

        def print(self):
            for l in self._lines:
                print(l)

        def get_value(self, dataset_name, class_name):
            assert (
                dataset_name in self.dataset_names
            ), f"{dataset_name} not in {self.dataset_names}"
            assert (
                class_name in self.class_names
            ), f"{class_name} not in {self.class_names}"
            dataset_idx = self.dataset_names.index(dataset_name)
            class_idx = self.class_names.index(class_name)
            return self.data[class_idx][dataset_idx]

        @classmethod
        def create_metrics(cls, lines):
            metrics = []
            new_metric = []
            for idx, l in enumerate(lines):
                if l.startswith("class     semi_test"):
                    new_metric = [l.strip()]
                    continue

                if (
                    idx - 1 > 0
                    and (idx - 1) < len(lines)
                    and lines[idx - 1].startswith("others")
                ):
                    metrics.append(cls(new_metric))
                    new_metric = []
                if new_metric:
                    new_metric.append(l.strip())
            return metrics

    data, names = [], []
    class_names = ["miou", "ceiling", "wall", "floor", "human"]

    t = tqdm.tqdm(total=len(logfiles))
    for ind, logfile in enumerate(logfiles):
        t.update()
        lines = open(logfile, "r").readlines()
        metrics = Metric.create_metrics(lines)
        for dataset_name in metrics[0].dataset_names:
            for class_name in class_names:
                item = [m.get_value(dataset_name, class_name) for m in metrics]
                data.append(item)
                names.append(f"exp_{ind}_{dataset_name}_{class_name}")

    draw.lines(
        data,
        names,
        xlabel="epoch",
        ylabel="iou",
        title="semseg analysis",
        output=output,
    )
    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    app()
