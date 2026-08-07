"""
blks - Building blocks and utilities for Python projects.
"""

from importlib.metadata import version

__version__ = version("blks")


def hello() -> str:
    return "Hello from blks!"
