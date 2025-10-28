import inspect
import os
import re
from ast import Try
from functools import partial
from typing import Dict, Optional, Union

# from numpy.distutils.misc_util import as_list
from pdebug.otn import manager as otn_manager
from pdebug.pdag.io import DataCatalog, MemoryDataSet
from pdebug.pdag.pipeline import node, pipeline
from pdebug.pdag.runner import SequentialRunner
from pdebug.utils.fileio import load_python_config
from pdebug.utils.hdfs_utils import is_list_of

import typer

__all__ = ["main"]


def camel_case_to_snake_case(camel_case: str) -> str:
    snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", camel_case)
    return snake_case.lower().strip("_")


def get_args_without_defaults(func):
    signature = inspect.signature(func)
    args_without_defaults = []
    for param in signature.parameters.values():
        if param.default is inspect.Parameter.empty and param.name != "kwargs":
            args_without_defaults.append(param.name)
    return args_without_defaults


@otn_manager.NODE.register(name="dag")
def main(config: Union[str, Dict]):
    """Run dag from config."""
    if isinstance(config, str):
        config = load_python_config(config)
    assert isinstance(config, dict)

    if os.environ.get("OTN_DEBUG", "0") == "1":
        debug_dir = os.environ.get("OTN_DEBUG_DIR", "./OTN_DEBUG")
        os.makedirs(debug_dir, exist_ok=True)
        typer.echo(
            typer.style(
                f"In DEBUG mode, output to {debug_dir}", fg=typer.colors.GREEN
            )
        )

    nodes = []
    for Name, cfg in config.items():
        if Name in ("Input", "Output", "Main"):
            continue
        if not Name[0].isupper():
            # only first character is Upper regarded as a node
            continue

        if "__" in Name:
            Name2, suffix = Name.split("__")
        else:
            Name2 = Name
            suffix = ""
        name = camel_case_to_snake_case(Name2)
        node_obj = otn_manager.create(name)
        assert isinstance(cfg, dict)

        if os.environ.get("OTN_DEBUG", "0") == "1":
            # DEBUG mode, set output to debug folder.
            if "output" in cfg:
                cfg["output"] = os.path.join(
                    debug_dir, os.path.basename(cfg["output"])
                )

        # node inputs
        if "inputs" in cfg:
            inputs = cfg.pop("inputs")
        else:
            inputs_args = get_args_without_defaults(node_obj)
            inputs = [cfg.pop(i) for i in inputs_args]

            # find "xxx.outputs" in cfg
            other_inputs = {}
            for k in cfg:
                if isinstance(cfg[k], str) and cfg[k].endswith(".outputs"):
                    other_inputs[k] = cfg[k]
                elif (
                    is_list_of(cfg[k], str)
                    and cfg[k]
                    and cfg[k][0].endswith(".outputs")
                ):
                    other_inputs[k] = cfg[k]
                else:
                    pass
            if other_inputs:
                # add to inputs
                inputs.extend(other_inputs.values())
                inputs = list(dict.fromkeys(inputs))
                # pop from cfg
                _ = [cfg.pop(k) for k in other_inputs]

        # node outputs
        outputs = cfg.pop("outputs", f"outputs")
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs = [f"{Name}.{o}" for o in outputs]
        if len(outputs) == 1:
            outputs = outputs[0]

        # node params
        if callable(node_obj):
            node_obj_filling_kwargs = partial(node_obj, **cfg)
        else:
            node_obj_filling_kwargs = node_obj

        node_inst = node(
            node_obj_filling_kwargs, inputs=inputs, outputs=outputs, name=Name
        )
        nodes.append(node_inst)

    pipe = pipeline(nodes)
    runner = SequentialRunner()

    data_catalog = DataCatalog()
    if "Input" in config:
        Input = config["Input"]
        data_catalog.add("Input", MemoryDataSet(Input))
        if isinstance(Input, dict):
            for k, v in config["Input"].items():
                data_catalog.add(f"Input.{k}", MemoryDataSet(v))

    return runner.run(pipe, data_catalog)


@otn_manager.NODE.register(name="node_y")
def node_y(*args, **kwargs):
    """Y node, used for get output from two input."""
    return kwargs.get("output", "")


@otn_manager.NODE.register(name="join_path")
def join_path(path: str, append: str = ""):
    """Join path.

    Args:
        path: input path.
        append: append path, default is "".

    Returns:
        output: output path.
    """
    return os.path.join(path, append)


if __name__ == "__main__":
    typer.run(main)
