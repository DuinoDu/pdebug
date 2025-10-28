import enum
import os
import random
import shutil
from typing import Dict, List, Union

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.piata.registry import (
    OUTPUT_REGISTRY,
    ROIDB_REGISTRY,
    SOURCE_REGISTRY,
)
from pdebug.utils.decorator import mp
from pdebug.utils.fileio import load_python_config

import tqdm
import typer

__all__ = ["SINGLE_NODE_CONFIG_TEMPLATE", "main"]


SINGLE_NODE_CONFIG_TEMPLATE = """\
Input = {}
Input["path"] = TODO
Input["name"] = TODO

Output = {}
Output["path"] = TODO
Output["name"] = TODO

# SourceInput

# VisOutput

# Main

# TopK = -1

# Shuffle = False

# num_workers = 0

def main(**kwargs) -> None:
    TODO
"""


class InputType(enum.Enum):
    Roidb = 1
    Reader = 2


@otn_manager.NODE.register(name="single_node")
def main(config: Union[str, Dict]):
    """Run single node config. Single node config is for simle and custom data
    processing based on piata.

    """
    if isinstance(config, str):
        config = load_python_config(config)
    assert "Input" in config, "`Input` is requred in config"
    file_path = config["Input"].pop("path")
    name = config["Input"].get("name", "default")
    if name == "coco":
        import pdebug.piata.coco
    input_inst = Input(file_path, **config["Input"])
    config["Input"]["path"] = file_path

    kwargs = {}
    loop_length = 0
    if name in ROIDB_REGISTRY:
        roidb = input_inst.get_roidb()
        reader = None
        # kwargs["roi"] = None
        loop_length = len(roidb)
        input_type = InputType.Roidb
    elif name in SOURCE_REGISTRY:
        roidb = None
        reader = input_inst.get_reader()
        # kwargs["image"] = None
        # kwargs["image_file"] = None
        # kwargs["imgdir"] = None
        loop_length = len(reader)
        input_type = InputType.Reader
    else:
        raise RuntimeError(f"Unknown name: {name}")

    source_reader = None
    if "SourceInput" in config:
        file_path = config["SourceInput"].pop("path")
        source_name = config["SourceInput"].get("name")
        assert source_name in SOURCE_REGISTRY
        source_reader = Input(file_path, **config["SourceInput"]).get_reader()
        config["SourceInput"]["path"] = file_path
        assert reader == None, "Don't support two reader in input."
        assert roidb != None, "SourceInput should used together with Roidb."
        # kwargs["image"] = None
        # kwargs["image_file"] = None
        # kwargs["imgdir"] = None

    vis_output = None
    if "VisOutput" in config:
        vis_output = os.path.expanduser(config["VisOutput"])
        if os.path.exists(vis_output):
            shutil.rmtree(vis_output)
        os.makedirs(vis_output, exist_ok=True)
        kwargs["vis_output"] = vis_output

    top_k = int(config.get("TopK", 0))
    if top_k and roidb and len(roidb) > top_k:
        roidb = roidb[:top_k]

    if config.get("Shuffle", False) and roidb:
        random.shuffle(roidb)

    if "main" in config:
        main_func = config["main"]
    elif "Main" in config:
        # TODO: Get more Main object from NODE.
        # main_func = eval(config["Main"])
        main_func = config[config["Main"]]
    else:
        typer.echo(
            typer.style(
                f"No main function found in {config_file}", fg=typer.colors.RED
            )
        )
        return
    typer.echo(typer.style(f"Run {main_func.__name__}", fg=typer.colors.GREEN))
    num_workers = config.get("num_workers", 0)

    result_list = []
    idx_list = list(range(loop_length))

    @mp(nums=num_workers)
    def _process(idx_list):
        t = tqdm.tqdm(total=len(idx_list))
        for idx in idx_list:
            t.update()
            if roidb:
                roi = roidb[idx]
                kwargs["roi"] = roi
                if source_reader and "image_name" in roi:
                    kwargs["image"] = source_reader.imread(roi["image_name"])
                    kwargs["imgdir"] = source_reader._imgdir
                    kwargs["image_file"] = os.path.join(
                        imgdir, roi["image_name"]
                    )
            if reader:
                assert not source_reader
                kwargs["image"] = reader.__next__()
                kwargs["image_file"] = reader.filename
                kwargs["imgdir"] = getattr(reader, "imgdir", None)

            kwargs["ctx"] = config
            res = main_func(**kwargs)
            if res is not None:
                assert (
                    num_workers == 0
                ), "num_workers > 0 dont support return value."
                result_list.append(res)

    _process(idx_list)

    if "Output" in config and result_list:
        savepath = config["Output"].pop("path")
        output_kwargs = config["Output"]
        if config["Output"]["name"] == "coco":
            # coco_categories is required for saving coco json.
            assert "Extras" in config
            assert "coco_categories" in config["Extras"]
            objects = config["Extras"]["coco_categories"]
            output_kwargs["categories"] = [
                obj["label_zh"] + " | " + obj["label"] for obj in objects
            ]

        writer = Output(result_list, **output_kwargs)
        writer.save(savepath)
        typer.echo(typer.style(f"saved to {savepath}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
