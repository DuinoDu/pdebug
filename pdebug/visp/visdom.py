from typing import List, Optional

from pdebug.data_types import Tensor, x_to_ndarray
from pdebug.utils.env import VISDOM_INSTALLED
from pdebug.utils.visdom_utils import get_global_visdom

import numpy as np

__all__ = ["histogram"]


def histogram(
    data: Tensor,
    *,
    bins: int = 30,
    ignore: Optional[List] = None,
    title: Optional[str] = None,
) -> None:
    """Visualize data via histogram."""
    data = x_to_ndarray(data)
    data = data.flatten()

    # remove inf
    finite_idx = np.isfinite(data)
    if finite_idx.sum() != data.shape[0]:
        print(f"change inf to zero: {len(data)} => {sum(finite_idx)}")
        output[~finite_idx] = 0

    if ignore and len(ignore) > 0:
        for value in ignore:
            if isinstance(value, str):
                assert "=" not in value, "Not implemented yet."
                if ">" in value:
                    value = float(value.split(">")[1])
                    idx = np.where(data <= value)[0]
                    if len(data) != len(idx):
                        print(
                            f"ignore value > {value}, "
                            f"nums: {len(data) - len(idx)}"
                        )
                elif "<" in value:
                    value = float(value.split("<")[1])
                    idx = np.where(data >= value)[0]
                    if len(data) != len(idx):
                        print(
                            f"ignore value < {value}, "
                            f"nums: {len(data) - len(idx)}"
                        )
                else:
                    raise ValueError(
                        f"Found bad ignore value: {value}, "
                        "'>' or '<' should be in string"
                    )
            else:
                idx = np.where(data != value)[0]
                if len(data) != len(idx):
                    print(
                        f"ignore value = {value}, "
                        f"nums: {len(data) - len(idx)}"
                    )
            data = data[idx]
    if len(data) == 0:
        print("data is empty, skip draw histogram")
        return

    if VISDOM_INSTALLED:
        vis = get_global_visdom()
        opts = {"numbins": bins}
        if title:
            opts["title"] = title
        vis.histogram(data, opts=opts)
    else:
        raise NotImplementedError
