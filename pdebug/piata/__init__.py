"""Piata is a data package, providing all data processing utilities you need."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.0.2"
__author__ = "duinodu"
__licence__ = ""
__homepage__ = ""
__docs__ = ""
__long_docs__ = ""

from .input import Input
from .output import Output

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    from . import handler as handler_module  # noqa: F401


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodules on first access."""
    if name == "handler":
        module = import_module("pdebug.piata.handler")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
