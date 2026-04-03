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
    scale: float = 9.81 / 1000

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        cols = [f"{self.sensor_id}_x", f"{self.sensor_id}_y", f"{self.sensor_id}_z"]
        accel = df[cols].values * self.scale
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        if np.mean(accel[:, 0]) > 0:
            # Rotate 180 around Z so gravity is negative on X axis
            accel = accel @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

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
class GyroLoader:
    """Loads gyroscope data from a DataFrame with columns like 't', 'g_x', 'g_y', 'g_z'."""
    path: str
    sensor_id: str
    scale: float = 0.1

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        cols = [f"{self.sensor_id}_dps10_x", f"{self.sensor_id}_dps10_y", f"{self.sensor_id}_dps10_z"]
        if not all(col in df.columns for col in cols):
            print(f"Warning: Not all gyroscope columns {cols} found in {self.path}. Filling with zeros.")
            gyro = np.zeros((len(df), 3))
        else:
            gyro = df[cols].values * self.scale
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        return {
            f"gyro/{self.sensor_id}": TimeSeries(
                t=t,
                x=gyro,
                units="deg/s",
                frame="sensor",
                meta={"sensor_id": self.sensor_id, "fs_hz": fs_hz},
            )
        }


@dataclass
class MagLoader:
    """Loads magnetometer data from a DataFrame with columns like 't', 'mmc_mG*'"""
    path: str
    lag: int = 0
    cols: Tuple[str, str, str] = ("mmc_mG_x", "mmc_mG_y", "mmc_mG_z")
    data_name: str = "mag"

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        if not all(col in df.columns for col in self.cols):
            print(f"Warning: Not all magnetometer columns {self.cols} found in {self.path}. Filling with zeros.")
            x = np.zeros((len(df), 3))
        else:
            x = df[list(self.cols)].values
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        if self.lag != 0:
            x = np.roll(x, shift=-self.lag, axis=0)

        return {
            self.data_name: TimeSeries(
                t=t,
                x=x,
                units="milli-Gauss",
                frame="sensor",
                meta={"fs_hz": fs_hz},
            )
        }


@dataclass
class LISMagLoader(MagLoader):
    """Special mag loader for LIS3MDL"""
    cols: Tuple[str, str, str] = ("lis3mdl_mG_x", "lis3mdl_mG_y", "lis3mdl_mG_z")
    data_name: str = "mag_lis"


@dataclass
class AngleLoader:
    """Loads angle data from a DataFrame with columns"""
    path: str
    lag: int = 0

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        angle_raw = df["angle_raw"].values
        x = angle_raw * np.pi * 2 / 4096
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        if self.lag != 0:
            x = np.roll(x, shift=-self.lag, axis=0)
            
        return {
            f"angle": TimeSeries(
                t=t,
                x=x,
                units="radians",
                frame="sensor",
                meta={"fs_hz": fs_hz},
            )
        }