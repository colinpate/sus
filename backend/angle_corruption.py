from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


ANGLE_RAIL_VALUES = (0, 4095)
ANGLE_RAW_PAD_SAMPLES = 3
ANGLE_ERROR_HALO_S = 0.08


def find_corrupt_angle_samples(
    angle_raw: np.ndarray,
    *,
    rail_values: tuple[int, ...] = ANGLE_RAIL_VALUES,
    pad_samples: int = ANGLE_RAW_PAD_SAMPLES,
) -> np.ndarray:
    angle_raw = np.asarray(angle_raw)
    bad_mask = np.isin(angle_raw, rail_values)
    if pad_samples <= 0 or not np.any(bad_mask):
        return bad_mask.astype(bool)

    structure = np.ones((pad_samples * 2) + 1, dtype=bool)
    return binary_dilation(bad_mask, structure=structure)


def interpolate_masked_signal(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    sample_pos: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError(f"Values and mask must align, got {values.shape} vs {mask.shape}")
    if not np.any(mask):
        return values.copy()

    good_mask = ~mask
    if np.sum(good_mask) < 2:
        raise ValueError("Need at least two good angle samples to interpolate corrupted regions")

    if sample_pos is None:
        sample_pos = np.arange(len(values), dtype=float)
    else:
        sample_pos = np.asarray(sample_pos, dtype=float).reshape(-1)
        if sample_pos.shape != values.shape:
            raise ValueError(f"Sample positions and values must align, got {sample_pos.shape} vs {values.shape}")

    cleaned = values.copy()
    cleaned[mask] = np.interp(sample_pos[mask], sample_pos[good_mask], values[good_mask])
    return cleaned


def project_mask_to_timeline(
    source_t: np.ndarray,
    source_mask: np.ndarray,
    target_t: np.ndarray,
    *,
    halo_s: float = ANGLE_ERROR_HALO_S,
) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=float).reshape(-1)
    source_mask = np.asarray(source_mask, dtype=bool).reshape(-1)
    target_t = np.asarray(target_t, dtype=float).reshape(-1)

    if source_t.shape != source_mask.shape:
        raise ValueError(f"Source time and mask must align, got {source_t.shape} vs {source_mask.shape}")
    if len(target_t) == 0 or not np.any(source_mask):
        return np.zeros(len(target_t), dtype=bool)

    bad_idx = np.flatnonzero(source_mask)
    split_idx = np.where(np.diff(bad_idx) > 1)[0]
    run_starts = np.r_[bad_idx[0], bad_idx[split_idx + 1]]
    run_ends = np.r_[bad_idx[split_idx], bad_idx[-1]]

    projected = np.zeros(len(target_t), dtype=bool)
    for start_idx, end_idx in zip(run_starts, run_ends):
        start_t = source_t[start_idx] - halo_s
        end_t = source_t[end_idx] + halo_s
        start = np.searchsorted(target_t, start_t, side="left")
        end = np.searchsorted(target_t, end_t, side="right")
        projected[start:end] = True

    return projected
