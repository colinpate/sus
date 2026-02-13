from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Literal
import json

import numpy as np
import matplotlib.pyplot as plt

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step

PlotKind = Literal["timeseries", "scatter", "hist"]

@dataclass(frozen=True)
class PlotSpec:
    kind: PlotKind
    key: str
    title: str = ""
    bins: int = 50                # for hist
    alpha: float = 0.7            # for scatter/hist

@dataclass
class Runner:
    out_dir: Path
    write_cache: bool = True
    read_cache: bool = False
    make_plots: bool = True

    def _cache_path(self, step_name: str) -> Path:
        return self.out_dir / "cache" / f"{step_name}.npz"

    def _save_cache(self, step_name: str, ws: Workspace, keys: Tuple[str, ...]) -> None:
        self._cache_path(step_name).parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for k in keys:
            v = ws[k]
            if isinstance(v, TimeSeries):
                payload[f"{k}__t"] = v.t
                payload[f"{k}__x"] = v.x
            elif isinstance(v, np.ndarray):
                payload[k] = v
        np.savez_compressed(self._cache_path(step_name), **payload)

    def _load_cache(self, step_name: str, ws: Workspace, keys: Tuple[str, ...]) -> bool:
        p = self._cache_path(step_name)
        if not p.exists():
            return False
        data = np.load(p, allow_pickle=False)
        for k in keys:
            t_key = f"{k}__t"
            x_key = f"{k}__x"
            if t_key in data and x_key in data:
                ws[k] = TimeSeries(t=data[t_key], x=data[x_key])
        return True

    def _plot_timeseries(self, ts: TimeSeries, title: str, path: Path) -> None:
        fig = plt.figure()
        plt.plot(ts.t, ts.x)
        plt.title(title)
        plt.xlabel("t (s)")
        plt.ylabel(ts.units or "value")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _plot_data(self, data: np.ndarray, plot_spec: PlotSpec, path: Path):
        if plot_spec.kind == "scatter":
            fig = plt.figure()
            x = data[:, 0]
            for y in range(1, data.shape[1]):
                plt.scatter(x, data[:, y], alpha=plot_spec.alpha)
            plt.title(plot_spec.title)
            fig.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            plt.close(fig)

    def run(self, ws: Workspace, steps: List[Step]) -> Workspace:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for step in steps:
            # Cache outputs if available
            loaded = False
            if self.read_cache and step.outputs:
                loaded = self._load_cache(step.name, ws, step.outputs)

            if not loaded:
                # Sanity: ensure inputs exist
                missing = [k for k in step.inputs if k not in ws]
                if missing:
                    raise KeyError(f"Step '{step.name}' missing inputs: {missing}")

                step.run(ws)

                if self.write_cache and step.outputs:
                    self._save_cache(step.name, ws, step.outputs)

            # Artifacts (plots)
            if self.make_plots:
                plot_keys = step.plot_keys or step.outputs
                for k in plot_keys:
                    if isinstance(k, PlotSpec):
                        self._plot_data(
                            ws.get(k.key),
                            k,
                            path=self.out_dir / "plots" / step.name / f"{k.key.replace('/', '_')}.png",
                        )
                    else:
                        v = ws.get(k)
                        if isinstance(v, TimeSeries):
                            self._plot_timeseries(
                                v,
                                title=f"{k} ({step.name})",
                                path=self.out_dir / "plots" / step.name / f"{k.replace('/', '_')}.png",
                            )

        return ws
