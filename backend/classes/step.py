from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from scipy.signal import butter, sosfiltfilt, sosfilt

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries, ChunkedTimeSeries


@dataclass(kw_only=True)
class Step:
    name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    plot_keys: Tuple[str, ...] = ()

    def run(self, ws: Workspace) -> None:
        """Implement in subclasses."""
        raise NotImplementedError
    

@dataclass
class FilterStep(Step):
    """Very basic low-pass via moving average (placeholder)."""
    fc_hz: float
    btype: str = "low"

    def run(self, ws: Workspace) -> None:
        ts: TimeSeries = ws[self.inputs[0]]
        fs_hz = ts.meta["fs_hz"]

        sos = butter(N=4, Wn=self.fc_hz, btype=self.btype, fs=fs_hz, output="sos")

        xf = sosfiltfilt(sos, ts.x, axis=0)

        ws[self.outputs[0]] = TimeSeries(
            t=ts.t,
            x=xf,
            units=ts.units,
            frame=ts.frame,
            meta={**ts.meta, f"{self.name}_fc_hz": self.fc_hz, f"{self.name}_btype": self.btype},
        )


@dataclass
class ChunkStep(Step):
    """Break time series into chunks"""
    chunk_t_s: float

    def run(self, ws: Workspace) -> None:
        ts: TimeSeries = ws[self.inputs[0]]

        span_len = int(self.chunk_t_s * ts.meta["fs_hz"])
        spans = [(i, i + span_len) for i in range(0, len(ts.x), span_len)]
        if (spans[-1][1] - spans[-1][0]) < span_len:
            spans = spans[:-1]

        for i in range(len(spans)):
            if (spans[i][1] - spans[i][0]) != span_len:
                print(i)

        print(len(spans), "chunks of", self.chunk_t_s, "s each (", span_len, "samples )")
        
        ws[self.outputs[0]] = ChunkedTimeSeries(
            base=ts,
            spans=spans,
            meta={**ts.meta, "chunk_t_s": self.chunk_t_s},
        )