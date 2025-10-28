import random
from typing import Tuple

from matplotlib import pyplot as plt

__all__ = ["Colormap"]


class Colormap:
    """
    Colormap Class.

    You can get colormap name like below:
    >>> from matplotlib._cm import datad
    >>> print(datad.keys())

    """

    def __init__(self, size, name="jet", hex_mode=False, shuffle=False):
        cm = plt.get_cmap(name)
        c_list = [i * 1.0 / size for i in range(size)]
        self.colors = []
        for i in c_list:
            color = [
                int(cm(i)[0] * 255),
                int(cm(i)[1] * 255),
                int(cm(i)[2] * 255),
            ]
            self.colors.append(color)
        if shuffle:
            random.shuffle(self.colors)
        self.hex_mode = hex_mode

    def __getitem__(self, i):
        color = self.colors[i]
        if self.hex_mode:
            color = Colormap.to_hex(tuple(color))
        return color

    def __len__(self):
        return len(self.colors)

    @staticmethod
    def to_hex(rgb: Tuple[int, int, int]):
        """Convert color ([r, g, b]) to hex string."""
        return "0x%02x%02x%02x" % rgb
