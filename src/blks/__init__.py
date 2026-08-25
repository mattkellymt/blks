"""
blks - Building blocks and utilities for Python projects.
"""

from importlib.metadata import version

from blks.adamw import AdamW
from blks.attention import Attention
from blks.layer_norm import LayerNorm
from blks.muon import Muon
from blks.rms_norm import RMSNorm
from blks.rope import Rope

__version__ = version("blks")

__all__ = ["AdamW", "Attention", "LayerNorm", "Muon", "RMSNorm", "Rope"]


def hello() -> str:
    return "Hello from blks!"
