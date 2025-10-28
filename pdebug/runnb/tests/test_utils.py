import shutil

from pdebug.runnb.utils import IS_PYTHON_VERSION_NEWER_THAN


def test_python_version():
    assert IS_PYTHON_VERSION_NEWER_THAN("2.7")
    assert IS_PYTHON_VERSION_NEWER_THAN("3.5")
    assert not IS_PYTHON_VERSION_NEWER_THAN("3.11")
