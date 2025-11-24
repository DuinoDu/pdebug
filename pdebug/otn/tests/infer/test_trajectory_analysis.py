# Created by AI
import math
from pathlib import Path

import pytest

from pdebug.otn.infer import trajectory_analysis_node as traj_node


def _yaw_to_quat(z_angle: float) -> tuple[float, float, float, float]:
    half_angle = z_angle / 2.0
    return (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))


def _write_tum_file(path: Path, yaws: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, yaw in enumerate(yaws):
            qx, qy, qz, qw = _yaw_to_quat(yaw)
            f.write(
                f"{idx:.1f} {float(idx)} 0.0 0.0 {qx} {qy} {qz} {qw}\n"
            )


def _write_json_file(path: Path, yaws: list[float]) -> None:
    data = {}
    for idx, yaw in enumerate(yaws):
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        data[f"{idx:.1f}"] = [float(idx), 0.0, 0.0, qx, qy, qz, qw]
    path.write_text(traj_node.json.dumps(data))


def test_state_classification_and_output(tmp_path: Path) -> None:
    yaws = [0.0, math.radians(30), math.radians(30), math.radians(30), math.radians(30)]
    tum_path = tmp_path / "traj.txt"
    state_path = tmp_path / "states.txt"
    image_path = tmp_path / "viz.png"
    _write_tum_file(tum_path, yaws)

    result = traj_node._main(
        input_path=str(tum_path),
        output=str(image_path),
        states=str(state_path),
        no_viz=False,
        skip=1,
        threshold=0.05,
        future=1,
    )

    assert state_path.exists()
    assert image_path.exists()
    assert result["states"][:3] == ["forward_left", "forward", "forward"]

    lines = [
        line
        for line in state_path.read_text().splitlines()
        if not line.startswith("#")
    ]
    velocities = [float(line.split()[11]) for line in lines]
    assert all(v > 0 for v in velocities[:-1])

    points = traj_node.read_trajectory_txt(str(tum_path))
    traj_node.classify_state(points, future_points=1, velocity_threshold=0.05)
    future_diffs = [p.future_yaw_diff for p in points]
    assert [d for d in future_diffs if d is not None][:2] == pytest.approx(
        [30.0, 0.0], abs=1e-3
    )


def test_json_input_support(tmp_path: Path) -> None:
    yaws = [0.0, 0.0, 0.0]
    json_path = tmp_path / "traj.json"
    _write_json_file(json_path, yaws)

    result = traj_node._main(
        input_path=str(json_path),
        states=str(tmp_path / "states2.txt"),
        no_viz=True,
        skip=1,
        threshold=0.05,
        future=1,
    )

    assert result["states"] == ["forward", "forward", "stop"]
    assert result["angle_stats"]["yaw"]["max"] >= 0.0
