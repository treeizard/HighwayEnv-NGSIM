"""Small runtime compatibility shims for third-party packages.

Python imports this module automatically when the repository is on PYTHONPATH.
The W&B 0.16 service process still references NumPy aliases removed in NumPy 2.
"""

from __future__ import annotations

try:
    import numpy as np
except Exception:
    np = None

if np is not None:
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128  # type: ignore[attr-defined]
