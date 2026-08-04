from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from angle_corruption import (
    ANGLE_RAW_PAD_SAMPLES,
    find_corrupt_angle_samples,
    interpolate_masked_signal,
)
from classes.time_series import TimeSeries
from sensor_orientation import resolve_signal_orientation

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
    signal_config: Optional[Dict[str, Any]] = None

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        if not all(col in df.columns for col in self.cols):
            print(f"Warning: Not all magnetometer columns {self.cols} found in {self.path}. Filling with zeros.")
            x = np.zeros((len(df), 3))
        else:
            x = df[list(self.cols)].values
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))

        signal_config = self.signal_config or {}
        lag = int(signal_config.get("lag", self.lag))
        if lag != 0:
            x = np.roll(x, shift=-lag, axis=0)

        offset = signal_config.get("offset")
        if offset is not None:
            offset_vec = np.asarray(offset, dtype=float)
            if offset_vec.shape != (3,):
                raise ValueError(f"{self.data_name} offset must be length-3, got shape {offset_vec.shape}")
            x = x - offset_vec
            print(f"Applying {self.data_name} offset {offset_vec}")

        orientation_config = resolve_signal_orientation(signal_config)
        orientation_matrix = None
        if orientation_config is not None:
            orientation_matrix = np.asarray(orientation_config, dtype=float)
            if orientation_matrix.shape != (3, 3):
                raise ValueError(
                    f"{self.data_name} sensor_to_pod_matrix must be 3x3, got shape {orientation_matrix.shape}"
                )
            x = (orientation_matrix @ x.T).T
            print(f"Applying {self.data_name} sensor-to-pod orientation {orientation_matrix.tolist()}")

        frame = "pod" if orientation_matrix is not None else "sensor"
        meta = {"fs_hz": fs_hz, "offset": offset, "lag": lag}
        if orientation_matrix is not None:
            meta.update(
                {
                    "orientation_preset": signal_config.get("orientation_preset"),
                    "sensor_to_pod_matrix": orientation_matrix.tolist(),
                }
            )

        return {
            self.data_name: TimeSeries(
                t=t,
                x=x,
                units="milli-Gauss",
                frame=frame,
                meta=meta,
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
    interpolate_bad: bool = True
    offset: int = 0
    mark_bad_samples: bool = True
    bad_mask_pad_samples: int = ANGLE_RAW_PAD_SAMPLES
    allow_degenerate: bool = False

    def load(self) -> Workspace:
        df = pd.read_csv(self.path)
        angle_raw = df["angle_raw"].to_numpy()
        t = np.array(df["t_s"].values)
        fs_hz = 1 / np.median(np.diff(t))
        if self.mark_bad_samples:
            bad_mask = find_corrupt_angle_samples(angle_raw, pad_samples=self.bad_mask_pad_samples)
        else:
            bad_mask = np.zeros_like(angle_raw, dtype=bool)

        x_raw = ((angle_raw + self.offset) % 4096) * np.pi * 2 / 4096
        if self.interpolate_bad:
            interpolated_bad_samples = False
            good_samples = int(np.sum(~bad_mask))
            if np.any(bad_mask) and good_samples < 2:
                if not self.allow_degenerate:
                    raise ValueError("Need at least two good angle samples to interpolate corrupted regions")
                print(
                    f"Warning: only {good_samples} good angle samples found in {self.path}; "
                    "using un-interpolated angle data"
                )
                x = x_raw.copy()
            else:
                x = interpolate_masked_signal(x_raw, bad_mask, sample_pos=t)
                interpolated_bad_samples = np.any(bad_mask)
            if interpolated_bad_samples:
                print(
                    f"Interpolated {np.sum(bad_mask)} corrupted angle samples "
                    f"({np.mean(bad_mask) * 100:.2f}%) from {self.path}"
                )
        else:
            x = x_raw

        if self.lag != 0:
            x = np.roll(x, shift=-self.lag, axis=0)
            bad_mask = np.roll(bad_mask, shift=-self.lag, axis=0)

        source_path = str(Path(self.path).resolve())
            
        return {
            f"angle": TimeSeries(
                t=t,
                x=x,
                units="radians",
                frame="sensor",
                meta={
                    "fs_hz": fs_hz,
                    "source_path": source_path,
                    "angle_bad_pct": float(np.mean(bad_mask) * 100.0),
                },
            ),
            "angle/bad_mask": TimeSeries(
                t=t,
                x=bad_mask,
                units="bool",
                frame="sensor",
                meta={"fs_hz": fs_hz, "source_path": source_path},
            ),
        }
