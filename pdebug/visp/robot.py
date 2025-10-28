from collections import defaultdict

import numpy as np

from ..data_types import x_to_ndarray
from .plotly import lines

__all__ = ["VisRobot"]


class _VisRobot:
    def __init__(self):
        self._data = defaultdict(list)

    def write(self, name: str, action, env_id=0):
        """
        action or torque: [num_envs, num_dof]
        """
        actions = x_to_ndarray(action)
        if actions.ndim == 2:
            actions = actions[env_id]
        elif actions.ndim > 2:
            raise NotImplementedError
        assert actions.ndim == 1

        self._data[name].append(actions)

    def save(self, title="title", output="output.html"):
        if len(self._data) == 0:
            return

        data_list = []
        name_list = []
        for name, value in self._data.items():
            value = np.asarray(value)  # [len_of_sequence, num_items]
            for i in range(value.shape[1]):
                data_list.append(value[:, i])
                name_list.append(f"{name}_{i}")
        lines(
            data_list,
            name_list,
            xlabel="num_steps",
            ylabel="value",
            title=title,
            output=output,
        )
        print(f"VisRobot is saved to {output}.")


VisRobot = _VisRobot()
