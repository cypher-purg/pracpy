"""
pracpy
==========
A collection of physics & chemistry simulation functions backed by a
fast C++ core.  Results are returned as plain NumPy arrays so you can
plot, analyse, or pass them on however you like.

Quick start
-----------
>>> import pracpy
>>> res = pracpy.expone(N0=10000, lam=0.3, t_end=10)
>>> res["t_half"]
2.310...
"""

from pracpy._core import single_decay, chain_decay  # noqa: F401

__all__ = ["single_decay", "chain_decay"]
__version__ = "0.1.0"
