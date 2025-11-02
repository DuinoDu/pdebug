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
        persist_path: Optional[Path] = None,
        resume: bool = False,
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
        elif persist_path is None:
            tmp_dir = tempfile.mkdtemp(prefix=f"pdebug_{stage_name}_")
        else:
            persist_path = persist_path.resolve()
            if persist_path.exists() and not resume:
                shutil.rmtree(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
            tmp_dir = str(persist_path)
        self._tmpdir = Path(tmp_dir)
        self._persistent = persist_path is not None
        self._manifest_path: Optional[Path] = (
            self._tmpdir / "manifest.pkl" if self._persistent else None
        )
        self._queue: Deque[Dict[str, object]] = deque()
        self._chunk_paths: List[Path] = []
        self._chunk_sizes: List[int] = []
        self._count = 0
        if self._persistent and resume:
            self._load_persistent_state()

    @property
    def count(self) -> int:
        return self._count

    @property
    def tmpdir(self) -> Path:
        return self._tmpdir

    @property
    def persistent(self) -> bool:
        return self._persistent

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
        chunk = list(self._queue)
        with path.open("wb") as handle:
            pickle.dump(chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._chunk_paths.append(path)
        self._chunk_sizes.append(len(chunk))
        self._queue.clear()
        if self._persistent:
            self._save_manifest()

    def cleanup(self) -> None:
        if self._persistent:
            return
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    def _save_manifest(self) -> None:
        if not self._persistent or self._manifest_path is None:
            return
        manifest_payload = {
            "count": self._count,
            "chunks": [
                {"filename": path.name, "count": size}
                for path, size in zip(self._chunk_paths, self._chunk_sizes)
            ],
        }
        with self._manifest_path.open("wb") as handle:
            pickle.dump(
                manifest_payload, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    def _load_persistent_state(self) -> None:
        if not self._tmpdir.exists():
            self._tmpdir.mkdir(parents=True, exist_ok=True)
            return
        if self._manifest_path and self._manifest_path.exists():
            try:
                with self._manifest_path.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception:
                payload = {}
            chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
            for entry in chunks:
                filename = entry.get("filename")
                count = entry.get("count", 0)
                if not filename:
                    continue
                path = self._tmpdir / filename
                if not path.exists():
                    continue
                try:
                    count_int = int(count)
                except Exception:
                    continue
                self._chunk_paths.append(path)
                self._chunk_sizes.append(count_int)
                self._count += count_int
            # If manifest references no files, fall back to scan
            if self._chunk_paths:
                return
        for path in sorted(self._tmpdir.glob("*.pkl")):
            try:
                with path.open("rb") as handle:
                    chunk = pickle.load(handle)
            except Exception:
                continue
            if not isinstance(chunk, list):
                continue
            count = len(chunk)
            self._chunk_paths.append(path)
            self._chunk_sizes.append(count)
            self._count += count
        self._save_manifest()
