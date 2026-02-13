from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) Data model (uniform signals)
# -----------------------------

@dataclass(frozen=True)
class TimeSeries:
    """Simple NxD time series container."""
    t: np.ndarray          # (N,) seconds
    x: np.ndarray          # (N, D)
    units: str = ""
    frame: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.t.ndim != 1:
            raise ValueError("t must be shape (N,)")
        if self.x.ndim == 1:
            object.__setattr__(self, "x", self.x.reshape(-1, 1))
        if self.x.shape[0] != self.t.shape[0]:
            raise ValueError("x must have same length as t")
        

@dataclass(frozen=True)
class ChunkedTimeSeries:
    base: TimeSeries
    # list of (start_idx, end_idx) half-open intervals into base arrays
    spans: List[Tuple[int, int]]
    meta: dict = field(default_factory=dict)

    def iter_chunks(self):
        t = self.base.t
        x = self.base.x
        for i0, i1 in self.spans:
            yield TimeSeries(t=t[i0:i1], x=x[i0:i1],
                             units=self.base.units, frame=self.base.frame,
                             meta={**self.base.meta, "chunk": (i0, i1)})