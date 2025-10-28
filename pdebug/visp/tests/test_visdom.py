from pdebug.utils.env import VISDOM_INSTALLED
from pdebug.visp import visdom

import numpy as np
import pytest


@pytest.mark.skipif(not VISDOM_INSTALLED, reason="visdom is required")
def test_histogram():
    data = np.random.rand(100, 100)
    visdom.histogram(data, title="test", bins=100)

    data = np.arange(12)
    visdom.histogram(data, title="test", ignore=[4, 5])

    data = np.arange(12)
    visdom.histogram(data, title="test", ignore=["<5", 9])
