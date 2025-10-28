import os
import pickle

__all__ = ["FileCache"]


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
