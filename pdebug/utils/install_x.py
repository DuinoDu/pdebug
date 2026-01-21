"""Utilities for install-x tool."""

import os
import sys
from typing import Union

from termcolor import cprint

__all__ = ["print_and_exit", "get_repo", "install", "INSTALL_X_CACHE"]

INSTALL_X_CACHE = os.getenv(
    "INSTALL_X_CACHE", os.path.expanduser("~/.cache/install-x")
)


def print_and_exit(msg):
    cprint(msg, "red")
    sys.exit()


def _try_to_find_install_x():
    install_root = os.getenv("INSTALL", None)
    if not install_root:
        print_and_exit(
            "`INSTALL` env is not found. Please update `scripts` "
            "and source scripts/envrc/command.sh"
        )

    if not os.path.exists(install_root):
        print_and_exit(
            f"{install_root} not found. Please install install-x from "
            "http://github.com/duinodu/install-x"
        )
    return install_root


def get_repo(name) -> Union[str, bool]:
    repo_path = os.path.join(INSTALL_X_CACHE, name)
    if os.path.exists(repo_path):
        return repo_path
    else:
        return None


def install(name: str):
    root = _try_to_find_install_x()
    bash_filename = f"{name}.sh"
    bash_file = None
    for dirpath, _, filenames in os.walk(root):
        if bash_filename in filenames:
            bash_file = os.path.join(dirpath, bash_filename)
            break
    if not bash_file:
        print_and_exit(f"{bash_filename} not found under {root}")
    os.system(bash_file)
