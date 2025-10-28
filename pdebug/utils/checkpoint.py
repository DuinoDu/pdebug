from typing import List, Union

from pdebug.visp import draw

import cv2
import torch


class Checkpoint:

    """Easy-to-use checkpoint tool.

    Args:
        ckpt_file: checkpoint file.

    Example:
        >>> from pdebug.types import Checkpoint
        >>> ckpt = Checkpoint("sample.ckpt")
        >>> ckpt -= "backbone"
        >>> ckpt.save("updated.ckpt")
    """

    def __init__(self, ckpt_file: str):
        self._ckpt_file = ckpt_file
        self.checkpoint = torch.load(
            ckpt_file, map_location=torch.device("cpu")
        )

    @property
    def state_dict(self):
        if "state_dict" in self.checkpoint:
            return self.checkpoint["state_dict"]
        else:
            return self.checkpoint

    @state_dict.setter
    def state_dict(self, item):
        if "state_dict" in self.checkpoint:
            self.checkpoint["state_dict"] = item
        else:
            self.checkpoint = item

    def update(self, item):
        self.state_dict.update(item.state_dict)

    def save(self, output: str):
        torch.save(self.checkpoint, output)

    def compare(
        self,
        item,
        output="tmp_compare_result",
        save_hist=False,
        print_func=print,
    ):
        if self == item:
            print_func("all close.")
            return
        if save_hist:
            os.makedirs(output, exist_ok=True)
        for k in self.state_dict.keys():
            v1 = self.state_dict[k]
            v2 = item.state_dict[k]
            if not torch.allclose(v1, v2):
                if save_hist:
                    hist_diff = draw.tensor_hist([v1, v2])
                    savename = os.path.join(output, f"{k}.jpg")
                    cv2.imwrite(savename, hist_diff)
                    print_func(f"saved to {savename}")
                else:
                    print_func(f"{k} differs")

    def __isub__(self, name: Union[str, List[str]]):
        """Remove name keys in state_dict."""
        if isinstance(name, str):
            name = [name]
        for name_i in name:
            self.state_dict = {
                k: self.state_dict[k]
                for k in self.state_dict.keys()
                if name_i not in k
            }
        return self

    def __repr__(self):
        _str = "Checkpoint: \n"
        _str += f"  file: {self._ckpt_file}\n"
        _str += f"  keys: {list(self.checkpoint.keys())[:10]}\n"
        return _str

    def __eq__(self, item):
        if len(self.state_dict) != len(item.state_dict):
            print(
                f"different state_dict length ({len(self.state_dict)} != {len(item.state_dict)})"
            )
            return False

        for k in self.state_dict.keys():
            v1 = self.state_dict[k]
            v2 = item.state_dict[k]
            if not torch.allclose(v1, v2):
                print(f"find different state: {k}, {v1} != {v2}")
                return False
        return True

    def __len__(self):
        return len(self.state_dict)
