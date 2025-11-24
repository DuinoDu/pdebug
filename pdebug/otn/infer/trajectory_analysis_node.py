# Created by AI
"""Trajectory state classification OTN infer node."""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from pdebug.otn import manager as otn_manager
from pdebug.visp import draw_trajectory

STATE_THRESHOLD_DEG = 15.0
ARROW_LENGTH = 2.0
DEFAULT_VELOCITY_THRESHOLD = 0.1


@dataclass
class TrajectoryPoint:
    """Trajectory point enriched with yaw/state information."""

    timestamp: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw: float | None = None
    state: str | None = None
    future_yaw_diff: float | None = None

    def calculate_yaw(self) -> None:
        """Calculate yaw (rotation around z-axis) from quaternion."""

        self.yaw = math.atan2(
            2 * (self.qw * self.qz + self.qx * self.qy),
            1 - 2 * (self.qy**2 + self.qz**2),
        )


def _normalize_angle(angle: float) -> float:
    """Normalize angle to ``[-pi, pi]``."""

    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def read_trajectory_txt(filename: str) -> List[TrajectoryPoint]:
    """Read TUM format trajectory file."""

    points: List[TrajectoryPoint] = []
    for line in Path(filename).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        timestamp, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
        point = TrajectoryPoint(timestamp, x, y, z, qx, qy, qz, qw)
        point.calculate_yaw()
        points.append(point)
    return points


def read_trajectory_json(filename: str) -> List[TrajectoryPoint]:
    """Read JSON trajectory file produced by odometry pipelines."""

    data: Dict[str, Sequence[float]] = json.loads(Path(filename).read_text())
    points: List[TrajectoryPoint] = []
    for timestamp, odo in data.items():
        qx, qy, qz, qw = odo[3:7]
        point = TrajectoryPoint(float(timestamp), odo[0], odo[1], 0.0, qx, qy, qz, qw)
        point.calculate_yaw()
        points.append(point)
    return points


def classify_state(
    points: List[TrajectoryPoint],
    future_points: int = 3,
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
) -> None:
    """Assign movement state for each trajectory point."""

    if not points:
        return

    for i, point in enumerate(points):
        if i >= len(points) - future_points:
            point.state = "stop"
            continue

        current_yaw = point.yaw or 0.0
        future_point = points[i + future_points]
        future_yaw = future_point.yaw or 0.0
        yaw_diff = _normalize_angle(future_yaw - current_yaw)
        yaw_diff_deg = math.degrees(yaw_diff)
        point.future_yaw_diff = yaw_diff_deg

        dx = points[i + 1].x - point.x if i < len(points) - 1 else 0.0
        dy = points[i + 1].y - point.y if i < len(points) - 1 else 0.0
        velocity = math.hypot(dx, dy)

        if velocity < velocity_threshold:
            point.state = "stop"
            continue

        movement_angle = math.atan2(dy, dx)
        orientation_diff = abs(movement_angle - current_yaw)
        if orientation_diff > math.pi:
            orientation_diff = 2 * math.pi - orientation_diff
        is_forward = orientation_diff < math.pi / 2

        if abs(yaw_diff_deg) <= STATE_THRESHOLD_DEG:
            point.state = "forward" if is_forward else "backward"
        elif yaw_diff_deg > STATE_THRESHOLD_DEG:
            point.state = "forward_left" if is_forward else "backward_left"
        else:
            point.state = "forward_right" if is_forward else "backward_right"


def calculate_angle_statistics(points: Iterable[TrajectoryPoint]) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics for yaw and yaw differences."""

    yaw_angles = [math.degrees(p.yaw) for p in points if p.yaw is not None]
    future_yaw_diffs = [p.future_yaw_diff for p in points if p.future_yaw_diff is not None]

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "values": [],
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }
        return {
            "values": values,
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
            "median": float(statistics.median(values)),
        }

    return {"yaw": _stats(yaw_angles), "future_yaw_diff": _stats(future_yaw_diffs)}


def write_states_to_file(points: Sequence[TrajectoryPoint], output_filename: str) -> None:
    """Write classified trajectory states to disk."""

    lines = ["# timestamp x y z qx qy qz qw yaw_deg state future_yaw_diff_deg velocity\n"]
    for i, point in enumerate(points):
        if i < len(points) - 1:
            dx = points[i + 1].x - point.x
            dy = points[i + 1].y - point.y
            velocity = math.hypot(dx, dy)
        else:
            velocity = 0.0

        yaw_deg = math.degrees(point.yaw) if point.yaw is not None else 0.0
        future_diff = point.future_yaw_diff if point.future_yaw_diff is not None else 0.0
        lines.append(
            f"{point.timestamp} {point.x} {point.y} {point.z} {point.qx} {point.qy} {point.qz} {point.qw} "
            f"{yaw_deg:.6f} {point.state} {future_diff:.6f} {velocity:.6f}\n"
        )

    Path(output_filename).write_text("".join(lines))


@otn_manager.NODE.register(name="trajectory_analysis")
def _main(
    input_path: str,
    output: str | None = None,
    states: str | None = None,
    no_viz: bool = False,
    skip: int = 10,
    threshold: float = DEFAULT_VELOCITY_THRESHOLD,
    future: int = 3,
) -> Dict[str, object]:
    """Run trajectory classification and optional visualization.

    Args:
        input_path: Path to trajectory file (``.txt`` or ``.json``).
        output: Output image path for visualization.
        states: Output text path for serialized states.
        no_viz: Disable visualization generation when ``True``.
        skip: Interval between orientation arrows when visualizing.
        threshold: Velocity threshold for stop detection.
        future: Offset used to compare future orientations.

    Returns:
        Mapping containing states list and angle statistics.
    """

    input_path = str(Path(input_path))
    if output is None:
        output = "trajectory_analysis.png"
    if states is None:
        states = "trajectory_states.txt"

    if input_path.endswith(".txt"):
        points = read_trajectory_txt(input_path)
    elif input_path.endswith(".json"):
        points = read_trajectory_json(input_path)
    else:
        raise ValueError("Unsupported trajectory format")

    classify_state(points, future_points=future, velocity_threshold=threshold)
    write_states_to_file(points, states)
    angle_stats = calculate_angle_statistics(points)

    if not no_viz:
        draw_trajectory.visualize_trajectory(
            points,
            output_filename=output,
            skip_points=skip,
            future_points=future,
            arrow_length=ARROW_LENGTH,
            state_threshold_deg=STATE_THRESHOLD_DEG,
        )

    states_list = [p.state for p in points]
    return {"states": states_list, "angle_stats": angle_stats}
