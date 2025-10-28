"""
OTN is short for One Thousand Nodes, containing more than 1k data process nodes.

A node can be several types:

    1. typer cli file
    2. function
    3. class with __call__ interface

OTN provide a manager to manage these 1k+ nodes, can also be used as node factory.

"""

from . import manager
