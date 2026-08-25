"""
blks - Building blocks and utilities for Python projects.
"""

from importlib.metadata import version

from blks.adamw import AdamW
from blks.layer_norm import LayerNorm
from blks.muon import Muon
from blks.rms_norm import RMSNorm

__version__ = version("blks")

__all__ = ["AdamW", "LayerNorm", "Muon", "RMSNorm"]


def hello() -> str:
    return "Hello from blks!"
