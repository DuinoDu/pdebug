import os
import pickle
import threading

from pdebug.runnb.utils import IS_PYTHON_VERSION_NEWER_THAN

if IS_PYTHON_VERSION_NEWER_THAN("3.9"):
    from functools import _NOT_FOUND
    from functools import cached_property as _cached_property
else:
    _NOT_FOUND = "0"
    from cached_property import cached_property as _cached_property

__all__ = ["cached_property"]


class cached_property(_cached_property):
    """Caching cached_property to file."""

    WHITE_LIST_FOR_INSTANCE_ID = ("mode",)

    cache_dir = ".cached_property"

    def __init__(self, func):
        super(cached_property, self).__init__(func)
        self.lock = threading.RLock()

    def get_cache_filename(self, instance, include_address=False):
        prefix = str(instance)[1:-2]
        if not include_address:
            prefix = prefix.split(" at ")[0]

        for prop in self.WHITE_LIST_FOR_INSTANCE_ID:
            if hasattr(instance, prop):
                prefix += "__" + f"{prop}=" + getattr(instance, prop)

        return prefix + "__" + self.attrname

    def __get__(self, instance, owner=None):
        cache = instance.__dict__

        if IS_PYTHON_VERSION_NEWER_THAN("3.9"):
            attrname = self.attrname
        else:
            attrname = self.func.__name__
            self.attrname = attrname
        val = cache.get(attrname, _NOT_FOUND)

        if val is not _NOT_FOUND:
            return val

        with self.lock:
            os.makedirs(self.cache_dir, exist_ok=True)

        cache_filename = self.get_cache_filename(instance)
        cache_filepath = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"load cached_property from {cache_filepath}")
            with open(cache_filepath, "rb") as fid:
                val = pickle.load(fid)
            cache[self.attrname] = val
            return val

        val = super(cached_property, self).__get__(instance, owner)

        with self.lock:
            with open(cache_filepath, "wb") as fid:
                pickle.dump(val, fid)
                print(f"dump cached_property to {cache_filepath}")

        return val
