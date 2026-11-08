"""Calendar helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def period_end_flags(index: pd.DatetimeIndex, freq: str = "monthly") -> np.ndarray:
    """True on the last trading day of each month/quarter/year present in the index (last row excluded)."""
    if freq == "monthly":
        key = index.year * 100 + index.month
    elif freq == "quarterly":
        key = index.year * 10 + (index.month - 1) // 3
    elif freq == "annual":
        key = index.year
    else:
        raise ValueError(f"Unknown frequency {freq!r}")
    k = np.asarray(key)
    flags = np.zeros(len(index), dtype=bool)
    flags[:-1] = k[1:] != k[:-1]
    return flags
