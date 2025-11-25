# Created by AI
"""Infer node registrations and helpers for OTN."""

# Lazily expose infer nodes to avoid circular imports during package load.
from importlib import import_module

__all__ = ["trajectory_analysis_node"]


def __getattr__(name):
    if name == "trajectory_analysis_node":
        module = import_module(".trajectory_analysis_node", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
