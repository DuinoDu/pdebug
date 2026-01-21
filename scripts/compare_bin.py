#!/usr/bin/env python3
"""Compare two binary files as float32 arrays.

Usage:
  python compare_bin.py a.bin b.bin [--atol 1e-5]
"""

from __future__ import annotations

import argparse
import array
import math
from pathlib import Path
from typing import Tuple


def load_float32(path: Path) -> array.array:
    buf = array.array("f")
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size {len(data)} is not a multiple of 4 bytes for float32")
    buf.frombytes(data)
    return buf


def compare_float_arrays(lhs: array.array, rhs: array.array) -> Tuple[float, float]:
    if len(lhs) != len(rhs):
        raise ValueError(f"array length mismatch ({len(lhs)} vs {len(rhs)})")
    max_abs = 0.0
    mse = 0.0
    for a, b in zip(lhs, rhs):
        diff = abs(a - b)
        max_abs = max(max_abs, diff)
        mse += diff * diff
    rms = math.sqrt(mse / len(lhs)) if lhs else 0.0
    return max_abs, rms


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two binary files as float32 arrays.")
    ap.add_argument("file_a", type=Path, help="Path to the first binary file")
    ap.add_argument("file_b", type=Path, help="Path to the second binary file")
    ap.add_argument("--atol", type=float, default=1e-5, help="Acceptable absolute tolerance for float32 diffs")
    args = ap.parse_args()

    if not args.file_a.exists():
        raise SystemExit(f"File not found: {args.file_a}")
    if not args.file_b.exists():
        raise SystemExit(f"File not found: {args.file_b}")

    if args.file_a.stat().st_size != args.file_b.stat().st_size:
        print(f"[FAIL] Byte size mismatch: {args.file_a} ({args.file_a.stat().st_size}) vs {args.file_b} ({args.file_b.stat().st_size})")
        return

    lhs_arr = load_float32(args.file_a)
    rhs_arr = load_float32(args.file_b)

    max_abs, rms = compare_float_arrays(lhs_arr, rhs_arr)
    status = "OK" if max_abs <= args.atol else "DIFF"
    print(f"[{status}] max_abs={max_abs:.6g}, rms={rms:.6g}, count={len(lhs_arr)}")


if __name__ == "__main__":
    main()
