from distutils.version import LooseVersion, StrictVersion
from platform import python_version

__all__ = ["is_notebook", "is_kaggle"]


def is_notebook() -> bool:
    """If running in notebook mode."""
    from IPython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def is_kaggle() -> bool:
    """If running in kaggle."""
    import os

    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def IS_PYTHON_VERSION_NEWER_THAN(version: str) -> bool:
    """If python version newer than given version."""
    return LooseVersion(python_version()) > LooseVersion(version)
