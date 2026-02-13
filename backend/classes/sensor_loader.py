from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from classes.time_series import TimeSeries

Workspace = Dict[str, Any]


# -----------------------------
# 2) Sensor I/O layer (OOP edges)
# -----------------------------

class SensorLoader(Protocol):
    def load(self) -> Workspace:
        """Return workspace entries, e.g. {'accel/a': TimeSeries(...)}"""
        ...

@dataclass
class AccelLoader:
    """Loads accelerometer data from a DataFrame with columns like 't', 'a_x', 'a_y', 'a_z'."""
    path: str
    sensor_id: str

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        cols = [f"{self.sensor_id}_x", f"{self.sensor_id}_y", f"{self.sensor_id}_z"]
        accel = df[cols].values * 9.81 / 1000
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        return {
            f"accel/{self.sensor_id}": TimeSeries(
                t=t,
                x=accel,
                units="m/s^2",
                frame="sensor",
                meta={"sensor_id": self.sensor_id, "fs_hz": fs_hz},
            )
        }
    

@dataclass
class MagLoader:
    """Loads magnetometer data from a DataFrame with columns like 't', 'a_x', 'a_y', 'a_z'."""
    path: str
    sensor_id: str

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        cols = [f"{self.sensor_id}_x", f"{self.sensor_id}_y", f"{self.sensor_id}_z"]
        accel = df[cols].values * 9.81 / 1000
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        return {
            f"accel/{self.sensor_id}": TimeSeries(
                t=t,
                x=accel,
                units="m/s^2",
                frame="sensor",
                meta={"sensor_id": self.sensor_id, "fs_hz": fs_hz},
            )
        }