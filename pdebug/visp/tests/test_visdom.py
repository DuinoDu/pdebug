from pdebug.visp import visdom

import numpy as np


def test_histogram():
    data = np.random.rand(100, 100)
    visdom.histogram(data, title="test", bins=100)

    data = np.arange(12)
    visdom.histogram(data, title="test", ignore=[4,5])

    data = np.arange(12)
    visdom.histogram(data, title="test", ignore=["<5", 9])
