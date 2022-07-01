# -*- coding: utf-8 -*-
# mypy: ignore-errors

import io
import re

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    readme = f.read()

with open("LICENSE", "r") as f:
    license = f.read()

with open("requirements/build.txt", "r") as f:
    requires = []
    for line in f:
        line = line.strip()
        if not line.startswith("#"):
            requires.append(line)

with io.open("pdebug/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

setup(
    name="pdebug",
    version=version,
    description="TODO",
    long_description=readme,
    author="user",
    author_email="duino472365351@gmail.com",
    url="https://github.com/user/pdebug",
    license=license,
    platform="linux",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=requires,
    entry_points={"console_scripts": ["pdebug = pdebug.cli:main"]},
)
