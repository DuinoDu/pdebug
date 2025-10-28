import time
from collections import defaultdict
from enum import Enum
from functools import partial

import numpy as np


class STATUS(Enum):

    RESET = "reset"
    STOP = "stop"
    RUNNING = "running"


class SingleTimer:
    def __init__(self, name=None):
        self._time_table = []
        self._status = STATUS.RESET
        self._name = name

    @property
    def name(self):
        return self._name

    def tic(self):
        assert self._status == STATUS.RESET, "please reset timer first."
        self._status = STATUS.RUNNING
        self._start = time.time()

    def toc(self):
        assert self._status == STATUS.RUNNING, "please call `tic` before `toc`"
        self._end = time.time()
        self._status = STATUS.STOP
        self._time_table.append((self._start, self._end))
        self.reset()

    def reset(self):
        self._start, self._end = 0, 0
        self._status = STATUS.RESET

    def __repr__(self):
        times, cost_mean, _, _ = self.metric()
        _str = f" times: {times}\n cost: {cost_mean:.3f} sec"
        return _str

    def metric(self):
        assert (
            self._status == STATUS.RESET
        ), "timer should be in `RESET` when do metric."
        times = len(self._time_table)
        time_table = np.asarray(self._time_table)
        costs = time_table[:, 1] - time_table[:, 0]
        cost_mean = np.mean(costs)
        cost_max = np.max(costs)
        cost_min = np.min(costs)
        return times, cost_mean, cost_max, cost_min


class Timeit:
    def __init__(self, name, timer):
        self._name = name
        self._timer = timer

    def __enter__(self):
        self._timer.start(self._name)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._timer.stop(self._name)


class Timer:

    GLOBAL_STATE = "global"

    def __init__(self, enable=True):
        self._enable = enable
        self.timeit = partial(Timeit, timer=self)
        self._timer_pool = defaultdict(SingleTimer)

    def start(self, name=None):
        if not name:
            name = self.GLOBAL_STATE
        self._timer_pool[name].tic()

    def stop(self, name=None):
        if not name:
            name = self.GLOBAL_STATE
        self._timer_pool[name].toc()

    def report(self):
        """Report time cost result to report."""
        # by stdout
        for name, timer in self._timer_pool.items():
            print(name)
            print(timer)

        # by visdom
        pass
