# Created by AI
"""Trajectory visualization helpers using matplotlib."""

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

DEFAULT_STATE_COLORS: Mapping[str, str] = {
    "stop": "red",
    "forward": "green",
    "backward": "blue",
    "forward_left": "orange",
    "forward_right": "purple",
    "backward_left": "cyan",
    "backward_right": "magenta",
}


def visualize_trajectory(
    points: Sequence[object],
    output_filename: str,
    skip_points: int,
    future_points: int,
    arrow_length: float,
    state_threshold_deg: float,
    state_colors: Mapping[str, str] | None = None,
) -> None:
    """Render trajectory, states, and angle statistics.

    The function expects each point to provide ``x``, ``y``, ``yaw``,
    ``state``, and optional ``future_yaw_diff`` attributes.
    """

    colors = state_colors or DEFAULT_STATE_COLORS
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(
        2, 2, hspace=0.3, wspace=0.3, width_ratios=[2, 1], height_ratios=[1, 1]
    )
    ax_traj = fig.add_subplot(gs[0, :])
    ax_states = fig.add_subplot(gs[1, 0])
    ax_angles = fig.add_subplot(gs[1, 1])

    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]

    ax_traj.set_title("Trajectory Analysis - State Classification")
    ax_traj.set_xlabel("X Position")
    ax_traj.set_ylabel("Y Position")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis("equal")

    ax_traj.plot(x_coords, y_coords, "k-", alpha=0.3, linewidth=1, label="Trajectory")
    for point in points:
        color = colors.get(point.state, "black")
        ax_traj.scatter(point.x, point.y, c=color, s=20, alpha=0.7)

    for state, color in colors.items():
        ax_traj.scatter([], [], c=color, label=state, s=50)
    ax_traj.legend()

    for i in range(0, len(points), max(1, skip_points)):
        if i >= len(points) - future_points:
            continue
        point = points[i]
        future_point = points[i + future_points]
        dx1 = arrow_length * math.cos(point.yaw)
        dy1 = arrow_length * math.sin(point.yaw)
        dx2 = arrow_length * math.cos(future_point.yaw)
        dy2 = arrow_length * math.sin(future_point.yaw)
        ax_traj.arrow(
            point.x,
            point.y,
            dx1,
            dy1,
            head_width=0.5,
            head_length=0.5,
            fc="blue",
            ec="blue",
            alpha=0.7,
            label="Current" if i == 0 else "",
        )
        ax_traj.arrow(
            point.x,
            point.y,
            dx2,
            dy2,
            head_width=0.5,
            head_length=0.5,
            fc="red",
            ec="red",
            alpha=0.7,
            label="Future" if i == 0 else "",
        )
        if getattr(point, "future_yaw_diff", None) is not None:
            ax_traj.annotate(
                f"{point.future_yaw_diff:.1f}°",
                (point.x + 1, point.y + 1),
                fontsize=8,
                alpha=0.7,
            )

    state_values = {state: idx for idx, state in enumerate(colors.keys())}
    y_values = [state_values.get(p.state, -1) for p in points]
    point_colors = [colors.get(p.state, "black") for p in points]
    ax_states.set_title("State Distribution Over Time")
    ax_states.set_xlabel("Time Index")
    ax_states.set_ylabel("State")
    ax_states.scatter(range(len(points)), y_values, c=point_colors, s=20, alpha=0.7)
    ax_states.set_yticks(list(state_values.values()))
    ax_states.set_yticklabels(list(state_values.keys()))
    ax_states.grid(True, alpha=0.3)

    state_counts = {state: 0 for state in colors.keys()}
    for p in points:
        if p.state in state_counts:
            state_counts[p.state] += 1
    stats_text = "State Distribution:\n"
    total_points = len(points) if points else 1
    for state, count in state_counts.items():
        if count:
            stats_text += f"{state}: {count} ({(count/total_points)*100:.1f}%)\n"
    ax_traj.text(
        0.02,
        0.9,
        stats_text,
        transform=ax_traj.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )
    ax_traj.text(
        0.02,
        0.97,
        f"Future points used: {future_points}",
        transform=ax_traj.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        fontsize=10,
    )

    yaw_diffs = [p.future_yaw_diff for p in points if getattr(p, "future_yaw_diff", None) is not None]
    if yaw_diffs:
        ax_angles.hist(
            yaw_diffs,
            bins=50,
            alpha=0.7,
            color="orange",
            edgecolor="black",
            label="Future Yaw Difference",
        )
        mean_diff = sum(yaw_diffs) / len(yaw_diffs)
        sorted_diff = sorted(yaw_diffs)
        mid = len(sorted_diff) // 2
        median = (
            (sorted_diff[mid - 1] + sorted_diff[mid]) / 2.0
            if len(sorted_diff) % 2 == 0
            else sorted_diff[mid]
        )
        ax_angles.axvline(mean_diff, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_diff:.2f}°")
        ax_angles.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median: {median:.2f}°")
        ax_angles.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5, label="Zero")
        ax_angles.axvline(
            state_threshold_deg,
            color="blue",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label=f"Threshold: ±{state_threshold_deg}°",
        )
        ax_angles.axvline(-state_threshold_deg, color="blue", linestyle=":", linewidth=1, alpha=0.5)
        ax_angles.set_xlabel("Future Yaw Difference (degrees)")
        ax_angles.set_ylabel("Frequency")
        ax_angles.set_title("Future Yaw Difference Distribution")
        ax_angles.legend(fontsize=8)
        ax_angles.grid(True, alpha=0.3)

    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
