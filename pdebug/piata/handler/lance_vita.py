"""Lance dataset reader utilities for VITA pipelines."""
import io
import os
from typing import Any, Dict, Optional, Sequence

from pdebug.utils.env import PYARROW_INSTALLED

import cv2
import numpy as np
from PIL import Image

from ..registry import SOURCE_REGISTRY
from .source import Reader


@SOURCE_REGISTRY.register(name="lance_vita")
class LanceVitaReader(Reader):
    """Reader for Lance datasets tailored to VITA/Cortexia pipelines."""

    def __init__(
        self,
        dataset_path: str,
        *,
        image_col: str = "camera_left",
        timestamp_col: Optional[str] = "timestamp",
        video_id_col: Optional[str] = None,
        frame_num_col: Optional[str] = None,
        row_limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        to_rgb: bool = True,
    ):
        super().__init__()
        if not PYARROW_INSTALLED:
            raise ImportError("pyarrow is required for the lance_vita reader.")

        try:
            import lance  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError(
                "lance package is required for the lance_vita reader."
            ) from exc

        dataset_path = str(dataset_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{dataset_path} not exist.")

        self.dataset_path = dataset_path
        self.image_col = image_col
        self.timestamp_col = timestamp_col
        self.video_id_col = video_id_col
        self.frame_num_col = frame_num_col
        self.to_rgb = to_rgb

        required_columns = {image_col}
        for col in (timestamp_col, video_id_col, frame_num_col):
            if col:
                required_columns.add(col)

        if columns is not None:
            merged = list(
                dict.fromkeys(
                    list(columns)
                    + [col for col in required_columns if col is not None]
                )
            )
            selected_columns: Optional[Sequence[str]] = merged
        else:
            selected_columns = None

        self._dataset = lance.dataset(dataset_path)
        table = self._dataset.to_table(columns=selected_columns)

        missing = [
            col
            for col in required_columns
            if col and col not in table.column_names
        ]
        if missing:
            raise ValueError(
                f"Columns {missing} are required but missing in Lance table {dataset_path}."
            )

        if row_limit is not None and row_limit > 0:
            row_limit = min(int(row_limit), len(table))
            table = table.slice(0, row_limit)

        self._table = table
        self._length = len(self._table)
        self._imgfiles = [self._frame_name(i) for i in range(self._length)]
        self._name_to_index = {
            name: idx for idx, name in enumerate(self._imgfiles)
        }

    def __len__(self) -> int:
        return self._length

    def __next__(self) -> Dict[str, Any]:
        if self._cur >= self._length:
            raise StopIteration
        row_idx = self._cur
        row = self._table.slice(row_idx, 1)
        entry = self._row_to_entry(row, row_idx)
        self._cur += 1
        self._cur_filename = entry["image_name"]
        return entry

    next = __next__

    def imread(self, identifier: Any):
        if isinstance(identifier, str):
            if identifier not in self._name_to_index:
                raise KeyError(f"{identifier} not found in Lance dataset.")
            index = self._name_to_index[identifier]
        else:
            index = int(identifier)
        if index < 0 or index >= self._length:
            raise IndexError(
                f"index {index} out of bounds for Lance dataset with length {self._length}"
            )
        row = self._table.slice(index, 1)
        entry = self._row_to_entry(row, index)
        return entry["image"]

    def _frame_name(self, idx: int) -> str:
        return f"lance_{idx:06d}.png"

    def _row_to_entry(self, row, row_idx: int) -> Dict[str, Any]:
        entry: Dict[str, Any] = {}
        for name in row.column_names:
            value = row[name][0]
            if hasattr(value, "as_py"):
                value = value.as_py()
            entry[name] = value

        if self.image_col not in entry:
            raise KeyError(
                f"Column '{self.image_col}' not found when reading Lance row."
            )

        image_bytes = self._ensure_bytes(entry[self.image_col])
        image = self._decode_image(image_bytes)
        entry["image"] = image
        if image is None:
            raise ValueError("Failed to decode image from Lance dataset.")

        entry["image_height"], entry["image_width"] = image.shape[:2]
        entry["image_name"] = self._imgfiles[row_idx]
        entry["frame_index"] = row_idx

        frame_number = (
            entry.get(self.frame_num_col) if self.frame_num_col else None
        )
        if frame_number is None:
            frame_number = row_idx
        entry["frame_number_resolved"] = int(frame_number)

        source_video = (
            entry.get(self.video_id_col) if self.video_id_col else None
        )
        if source_video is None:
            source_video = "lance_vita"
        entry["source_video_id_resolved"] = str(source_video)

        timestamp_raw = (
            entry.get(self.timestamp_col) if self.timestamp_col else None
        )
        entry["timestamp_raw"] = timestamp_raw
        entry["timestamp_seconds"] = self._timestamp_to_seconds(
            timestamp_raw, entry["frame_number_resolved"]
        )

        return entry

    @staticmethod
    def _ensure_bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        raise TypeError(f"Unsupported image storage type: {type(value)}")

    def _decode_image(self, buffer: bytes) -> np.ndarray:
        if buffer is None:
            raise ValueError(
                "Empty image buffer encountered in Lance dataset."
            )
        if Image is not None:
            with Image.open(io.BytesIO(buffer)) as im:
                frame = np.array(im.convert("RGB"))
        else:
            if cv2 is None:
                raise RuntimeError(
                    "Either pillow or opencv-python is required to decode Lance images."
                )
            frame = cv2.imdecode(
                np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError("Failed to decode image buffer via OpenCV.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame is None:
            raise ValueError("Decoded image is None.")
        if not self.to_rgb:
            if cv2 is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = frame[..., ::-1]
        return frame

    @staticmethod
    def _timestamp_to_seconds(
        timestamp_value: Any, frame_number: int
    ) -> Optional[float]:
        if timestamp_value is None:
            return frame_number / 30.0
        if isinstance(timestamp_value, float):
            return timestamp_value
        if isinstance(timestamp_value, int):
            if abs(timestamp_value) > 10_000_000_000:
                return timestamp_value / 1_000_000_000.0
            return float(timestamp_value)
        return None
