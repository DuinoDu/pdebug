"""
Utilities used with torch.profiler
"""
import subprocess as commands
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from pdebug.utils.env import RICH_INSTALLED


class OpType(Enum):
    aten = "aten"
    cuda = "cuda"
    at_native = "at::native"
    cudnn = "cudnn"
    function = "function"
    others = "others"
    custom = "custom op"


@dataclass
class ProfileItem:
    name: str
    count: int

    def __repr__(self):
        return f"{self.name}, count: {self.count}"


class pretty_print_list(list):
    def __str__(self):
        _str = super(pretty_print_list, self).__str__()
        items = [i.strip(",") for i in _str[1:-1].split(" ")]
        return "\n".join(items)
        return _str


def search_name_in_code(name: str, code_path: str) -> List[str]:
    """Do `rg` search in given code path. You need to install ripgrep."""
    cmd = f"cd {code_path}; rg {name} -g '*.py' -g '*.cu' -g '*.cpp' -g '*.c' "
    (status, output) = commands.getstatusoutput(cmd)
    if output == "":
        return None
    else:
        output = output.split("\n")
    return output


def _find_custom(result: Dict) -> None:
    """Find custom op using "rg"."""
    BLACK_LIST = ["Memset (Device)", "model_inference"]

    for key in [OpType.function, OpType.others]:
        for item in result[key]:
            if item.name in BLACK_LIST:
                continue
            ret = search_name_in_code(item.name, "./")
            if ret:
                result[OpType.custom].append(item)


def classify(
    prof_result: "torch.autograd.profiler_util.EventList",
    keep: Optional[List[str]] = None,
    print_result: bool = False,
    print_sort_by: Optional[str] = "count",
) -> Dict[OpType, List[ProfileItem]]:
    """Classify profile result by op type.

    Args:
        prof_result: profile result, usually by `prof.key_averages()`
        keep: keep key name list. Options:
                key
                count
                input_shapes
                self_cuda_time
                cuda_time
                self_cpu_time
                cpu_time
                cpu_memory_usage
                cuda_memory_usage

    Example:
        >>> from torch.profiler import profile, record_function, ProfilerActivity
        >>> with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        >>>     with record_function("model_inference"):
        >>>         _ = model(data)
        >>> result_table = prof.key_averages().table(sort_by="cuda_time_total")

        >>> from pdebug.utils import pytorch_profiler
        >>> prof_result = pytorch_profiler.classify(prof.key_averages(), print_result=True)

        >>> from thop import profile
        >>> total_ops, total_params = profile(model, inputs=inputs)
        >>> print("%.2f M Params | %.2f G " % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))

    """
    assert hasattr(prof_result, "__iter__")
    if keep:
        raise NotImplementedError

    result = defaultdict(list)
    for item in prof_result:
        name = item.key
        count = item.count
        new_item = ProfileItem(name, count)

        if name.startswith("aten"):
            result[OpType.aten].append(new_item)
        elif name.startswith("cuda"):
            result[OpType.cuda].append(new_item)

        elif "(" in name and ")" in name:
            function_name = name.split("(")[0].split("<")[0]
            if " " in function_name:
                function_name = function_name.split(" ")[1]
            if function_name == "":
                function_name = name
            new_item.name = function_name

            if function_name.startswith("at::native"):
                result[OpType.at_native].append(new_item)
            elif "cudnn" in function_name:
                result[OpType.cudnn].append(new_item)
            else:
                result[OpType.function].append(new_item)
        else:
            result[OpType.others].append(new_item)

    _find_custom(result)

    assert print_sort_by in ProfileItem.__dataclass_fields__
    for op_type in OpType:
        result[op_type] = sorted(
            result[op_type], key=lambda x: getattr(x, print_sort_by)
        )

    if print_result:
        assert RICH_INSTALLED, "rich is required."

        from rich.console import Console
        from rich.table import Table

        for op_type in OpType:
            op_result = result[op_type]
            table = Table(title=op_type.value)
            table.add_column("name", style="magenta")
            table.add_column("count", style="magenta")

            for profile_item in result[op_type]:
                table.add_row(profile_item.name, str(profile_item.count))
            console = Console()
            console.print(table)

    return result
