from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence

__all__ = ["FileCache", "ResultCache"]


class FileCache:
    """
    Easy to use file cache.

    Example:
    >>> with FileCache(list, "cache.pkl") as data:
    >>>     if len(data) == 0:
    >>>         data.append("1")

    """

    def __init__(self, class_type, cache_file, quiet=False, use_cache=True):
        self._data = class_type()
        self._cache_file = cache_file
        self.quiet = quiet
        self.use_cache = use_cache

    def __enter__(self):
        if self.exists() and self.use_cache:
            with open(self._cache_file, "rb") as fid:
                self._data = pickle.load(fid)
            if not self.quiet:
                print(f"load cache from {self._cache_file}.")
        return self._data

    def __exit__(self, type, value, traceback):
        if not self.exists():
            with open(self._cache_file, "wb") as fid:
                pickle.dump(self._data, fid)
            if not self.quiet:
                print(f"dump cache to {self._cache_file}.")

    def exists(self):
        return os.path.exists(self._cache_file)


class ResultCache:
    """FIFO cache that spills algorithm outputs to disk in fixed-size chunks."""

    def __init__(
        self,
        stage_name: str,
        *,
        chunk_size: int = 200,
        tmp_root: Optional[Path] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.stage_name = stage_name
        self.chunk_size = chunk_size
        if tmp_root is not None:
            tmp_root = tmp_root.resolve()
            tmp_root.mkdir(parents=True, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(
                prefix=f"pdebug_{stage_name}_", dir=str(tmp_root)
            )
        else:
            tmp_dir = tempfile.mkdtemp(prefix=f"pdebug_{stage_name}_")
        self._tmpdir = Path(tmp_dir)
        self._queue: Deque[Dict[str, object]] = deque()
        self._chunk_paths: List[Path] = []
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def tmpdir(self) -> Path:
        return self._tmpdir

    def append_many(self, payloads: Sequence[Dict[str, object]]) -> None:
        for payload in payloads:
            self._queue.append(dict(payload))
            self._count += 1
            if len(self._queue) >= self.chunk_size:
                self._flush()

    def append(self, payload: Dict[str, object]) -> None:
        self.append_many([payload])

    def finalize(self) -> None:
        self._flush()

    def iter_results(self) -> Iterator[Dict[str, object]]:
        for path in self._chunk_paths:
            with path.open("rb") as handle:
                chunk = pickle.load(handle)
            for item in chunk:
                yield item

    def _flush(self) -> None:
        if not self._queue:
            return
        index = len(self._chunk_paths)
        path = self._tmpdir / f"{self.stage_name}_{index:05d}.pkl"
        with path.open("wb") as handle:
            pickle.dump(list(self._queue), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._chunk_paths.append(path)
        self._queue.clear()

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass
