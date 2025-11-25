# Created by AI
"""Classify odometry states from a Lance dataset and write them back to Lance."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from pdebug.otn.infer import trajectory_analysis_node as traj
from pdebug.otn.infer.lance_utils import load_lance_batch, write_lance_with_column


def _build_reader_kwargs(
    *,
    timestamp_col: str | None,
    video_id_col: str | None,
    frame_num_col: str | None,
    row_limit: int | None,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if timestamp_col:
        kwargs["timestamp_col"] = timestamp_col
    if video_id_col:
        kwargs["video_id_col"] = video_id_col
    if frame_num_col:
        kwargs["frame_num_col"] = frame_num_col
    if row_limit:
        kwargs["row_limit"] = int(row_limit)
    return kwargs


def _odo_values(entry: object) -> Sequence[float]:
    if isinstance(entry, dict):
        for key in ("values", "value", "pose", "odo", "data"):
            if key in entry:
                return entry[key]
        # dictionary with explicit pose components
        if {"x", "y", "z", "qx", "qy", "qz", "qw"}.issubset(entry):
            return [
                entry["x"],
                entry["y"],
                entry["z"],
                entry["qx"],
                entry["qy"],
                entry["qz"],
                entry["qw"],
            ]
    if isinstance(entry, (list, tuple)):
        return list(entry)
    raise TypeError(f"Unsupported odo entry type: {type(entry)}")


def _to_points(odo_entries: Iterable[object], timestamps: Iterable[float]) -> List[traj.TrajectoryPoint]:
    points: List[traj.TrajectoryPoint] = []
    for idx, (odo, timestamp) in enumerate(zip(odo_entries, timestamps)):
        values = list(_odo_values(odo))
        if len(values) < 7:
            raise ValueError(f"Expected at least 7 values for odo at index {idx}, got {len(values)}")
        x, y = float(values[0]), float(values[1])
        z = float(values[2]) if len(values) >= 3 else 0.0
        qx, qy, qz, qw = (float(v) for v in values[3:7])
        point = traj.TrajectoryPoint(float(timestamp), x, y, z, qx, qy, qz, qw)
        point.calculate_yaw()
        points.append(point)
    return points


def _timestamps(table, timestamp_col: str | None) -> List[float]:
    if timestamp_col is None or timestamp_col not in table.column_names:
        return [float(i) for i in range(len(table))]
    return [float(v) for v in table[timestamp_col].to_pylist()]


def _update_odo_entries(original: Sequence[object], states: Sequence[str | None]) -> List[object]:
    updated: List[object] = []
    for entry, state in zip(original, states):
        if isinstance(entry, dict):
            next_entry = dict(entry)
            next_entry["state"] = state
        elif isinstance(entry, (list, tuple)):
            next_entry = {"values": list(entry), "state": state}
        else:
            next_entry = {"value": entry, "state": state}
        updated.append(next_entry)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify odometry states within a Lance dataset.")
    parser.add_argument("dataset", type=Path, help="Path to the input Lance dataset directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Lance dataset directory (defaults to <dataset>_with_states)",
    )
    parser.add_argument("--odo-col", default="odo", help="Name of the odo column to read and update")
    parser.add_argument("--timestamp-col", default=None, help="Optional timestamp column name")
    parser.add_argument("--video-id-col", default=None, help="Optional video id column for the reader")
    parser.add_argument("--frame-num-col", default=None, help="Optional frame number column for the reader")
    parser.add_argument("--row-limit", type=int, default=None, help="Limit rows read from the dataset")
    parser.add_argument("--future", type=int, default=3, help="Future point offset for classification")
    parser.add_argument(
        "--threshold",
        type=float,
        default=traj.DEFAULT_VELOCITY_THRESHOLD,
        help="Velocity threshold for stop detection",
    )
    args = parser.parse_args()

    reader_kwargs = _build_reader_kwargs(
        timestamp_col=args.timestamp_col,
        video_id_col=args.video_id_col,
        frame_num_col=args.frame_num_col,
        row_limit=args.row_limit,
    )

    lance_batch = load_lance_batch(args.dataset, reader_kwargs=reader_kwargs)
    table = lance_batch.table
    if args.odo_col not in table.column_names:
        raise KeyError(f"Column '{args.odo_col}' not found in Lance dataset")

    odo_entries = table[args.odo_col].to_pylist()
    timestamps = _timestamps(table, args.timestamp_col)
    points = _to_points(odo_entries, timestamps)
    traj.classify_state(points, future_points=args.future, velocity_threshold=args.threshold)
    states = [p.state for p in points]

    output_path = args.output or args.dataset.with_name(f"{args.dataset.name}_with_states")
    updated_odo = _update_odo_entries(odo_entries, states)
    write_lance_with_column(table, args.odo_col, updated_odo, output_path)
    print(f"Wrote classified odometry to {output_path}")


if __name__ == "__main__":
    main()
