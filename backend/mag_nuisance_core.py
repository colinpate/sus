"""Production continuous body/world magnetometer correction utilities.

The state is ``[body_field, world_field_in_body]``. The body component follows
a random walk. The world component is first rotated by gyro1, then given a
small random-walk allowance for gyro/model error and real ambient-field change.

Given travel, this is a linear Gaussian smoothing problem, so an RTS smoother
is both faster and easier to inspect than a general nonlinear least-squares
solver. Travel and field correction are coupled with a small outer iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# Verified independently on pod-v1 and pod-v2 strong-magnet rotation logs.
# gyro-frame XYZ = PRIMARY_MAG_TO_GYRO @ recorded-MMC XYZ.
PRIMARY_MAG_TO_GYRO = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
)


@dataclass(frozen=True)
class MagSolverWeights:
    """Noise scales for a 10 Hz correction-state/measurement grid."""

    mag_update_threshold: float = 1500.0  # mG, applied to predicted magnet field
    world_rw: float = 1.5  # mG / sqrt(second)
    body_rw: float = 1.0  # mG / sqrt(second)
    mag_sigma: float = 40.0  # mG per axis at an effective 10 Hz measurement rate
    body_initial_sigma: float = 300.0  # mG
    world_initial_sigma: float = 500.0  # mG

    def validate(self) -> None:
        for name in (
            "mag_update_threshold",
            "world_rw",
            "body_rw",
            "mag_sigma",
            "body_initial_sigma",
            "world_initial_sigma",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LinearXYZModel:
    """Constrained low-field model ``field_xyz = intercept + slope * travel``."""

    slope: np.ndarray
    intercept: np.ndarray
    travel_min: float
    travel_max: float
    bin_count: int

    def __post_init__(self) -> None:
        slope = np.asarray(self.slope, dtype=float)
        intercept = np.asarray(self.intercept, dtype=float)
        if slope.shape != (3,) or intercept.shape != (3,):
            raise ValueError("slope and intercept must both be length-3")
        if float(slope @ slope) < 1e-12:
            raise ValueError("XYZ model slope is too small to invert")
        if self.travel_max <= self.travel_min:
            raise ValueError("travel_max must exceed travel_min")
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "intercept", intercept)

    def predict(self, travel: np.ndarray | float) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return self.intercept + travel[..., np.newaxis] * self.slope

    def infer(self, field_xyz: np.ndarray, clip: bool = True) -> np.ndarray:
        field_xyz = np.asarray(field_xyz, dtype=float)
        travel = (field_xyz - self.intercept) @ self.slope / float(
            self.slope @ self.slope
        )
        if clip:
            travel = np.clip(travel, self.travel_min, self.travel_max)
        return travel

    def covers(self, travel: np.ndarray) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return (travel >= self.travel_min) & (travel <= self.travel_max)

    def weak(self, travel: np.ndarray, threshold_mg: float) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return self.covers(travel) & (
            np.linalg.norm(self.predict(travel), axis=-1) <= threshold_mg
        )


@dataclass
class MagCorrectionResult:
    travel: np.ndarray
    body_field: np.ndarray
    world_field: np.ndarray
    correction: np.ndarray
    corrected_mag_weak: np.ndarray
    corrected_mag_all: np.ndarray
    update_mask: np.ndarray
    iteration_change_mm: list[float]


@dataclass(frozen=True)
class ScalarParameterizedXYZModel:
    """Dense travel-to-XYZ path with nearest-path inversion."""

    travel_grid: np.ndarray
    xyz_grid: np.ndarray
    scalar_center: float
    scalar_scale: float
    coefficients: np.ndarray
    bin_count: int

    def __post_init__(self) -> None:
        travel_grid = np.asarray(self.travel_grid, dtype=float)
        xyz_grid = np.asarray(self.xyz_grid, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        if travel_grid.ndim != 1 or len(travel_grid) < 2:
            raise ValueError("travel_grid must be a one-dimensional grid")
        if xyz_grid.shape != (len(travel_grid), 3):
            raise ValueError("xyz_grid must have shape (len(travel_grid), 3)")
        if not np.all(np.diff(travel_grid) > 0):
            raise ValueError("travel_grid must be strictly increasing")
        if not (
            np.all(np.isfinite(travel_grid))
            and np.all(np.isfinite(xyz_grid))
            and np.all(np.isfinite(coefficients))
        ):
            raise ValueError("XYZ-model arrays must be finite")
        object.__setattr__(self, "travel_grid", travel_grid)
        object.__setattr__(self, "xyz_grid", xyz_grid)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "_tree", cKDTree(xyz_grid))

    @property
    def travel_min(self) -> float:
        return float(self.travel_grid[0])

    @property
    def travel_max(self) -> float:
        return float(self.travel_grid[-1])

    def predict(self, travel: np.ndarray | float) -> np.ndarray:
        travel_array = np.asarray(travel, dtype=float)
        flat = travel_array.reshape(-1)
        predicted = np.column_stack(
            [
                np.interp(flat, self.travel_grid, self.xyz_grid[:, axis])
                for axis in range(3)
            ]
        )
        if travel_array.ndim == 0:
            return predicted[0]
        return predicted.reshape(travel_array.shape + (3,))

    def infer(self, field_xyz: np.ndarray, clip: bool = True) -> np.ndarray:
        del clip  # The dense path is already bounded by travel_grid.
        field_xyz = np.asarray(field_xyz, dtype=float)
        if field_xyz.shape == (3,):
            return np.asarray(self.travel_grid[self._tree.query(field_xyz)[1]])
        if field_xyz.ndim < 2 or field_xyz.shape[-1] != 3:
            raise ValueError("field_xyz must have final dimension three")
        flat = field_xyz.reshape(-1, 3)
        inferred = self.travel_grid[self._tree.query(flat)[1]]
        return inferred.reshape(field_xyz.shape[:-1])

    def covers(self, travel: np.ndarray) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return (travel >= self.travel_min) & (travel <= self.travel_max)

    def weak(self, travel: np.ndarray, threshold_mg: float) -> np.ndarray:
        travel = np.asarray(travel, dtype=float)
        return self.covers(travel) & (
            np.linalg.norm(self.predict(travel), axis=-1) <= threshold_mg
        )


def invert_scalar_travel_model(
    travel: np.ndarray,
    coefficients: np.ndarray,
    offset_mm: float,
    *,
    soft_mg: float = 50.0,
) -> np.ndarray:
    """Invert the front scalar magnet-to-travel model."""

    x0, y_scale, power = np.asarray(coefficients, dtype=float)
    if y_scale <= 0 or power <= 0:
        raise ValueError(
            f"Expected positive scalar-model scale/power, got {y_scale}, {power}"
        )
    normalized = (np.asarray(travel, dtype=float) - offset_mm) / y_scale
    delta = np.sign(normalized) * (
        (np.abs(normalized) + soft_mg**power) ** (1.0 / power) - soft_mg
    )
    return x0 + delta


def predict_scalar_travel(
    scalar_mag: np.ndarray,
    coefficients: np.ndarray,
    offset_mm: float,
    *,
    soft_mg: float = 50.0,
) -> np.ndarray:
    """Apply the front scalar magnet-to-travel model and absolute offset."""

    x0, y_scale, power = np.asarray(coefficients, dtype=float)
    delta = np.asarray(scalar_mag, dtype=float) - x0
    softened = (np.abs(delta) + soft_mg) ** power - soft_mg**power
    return np.sign(delta) * softened * y_scale + offset_mm


def fit_scalar_parameterized_xyz(
    scalar_mag: np.ndarray,
    mag_xyz: np.ndarray,
    scalar_coefficients: np.ndarray,
    scalar_offset_mm: float,
    *,
    scalar_bin_mg: float = 100.0,
    degree: int = 2,
    travel_max_mm: float = 210.0,
    travel_step_mm: float = 0.25,
    min_bin_samples: int = 5,
) -> ScalarParameterizedXYZModel:
    """Fit XYZ against the independently calibrated scalar coordinate."""

    scalar_mag = np.asarray(scalar_mag, dtype=float).reshape(-1)
    mag_xyz = np.asarray(mag_xyz, dtype=float)
    if mag_xyz.shape != (len(scalar_mag), 3):
        raise ValueError("scalar_mag and mag_xyz must have shapes (N,) and (N, 3)")
    if degree not in (1, 2):
        raise ValueError("degree must be one or two")
    if scalar_bin_mg <= 0 or travel_max_mm <= 0 or travel_step_mm <= 0:
        raise ValueError("bin size, maximum travel, and grid step must be positive")
    if min_bin_samples < 1:
        raise ValueError("min_bin_samples must be positive")

    finite = np.isfinite(scalar_mag) & np.all(np.isfinite(mag_xyz), axis=1)
    bin_id = np.floor(scalar_mag / scalar_bin_mg).astype(int)
    scalar_centers: list[float] = []
    xyz_medians: list[np.ndarray] = []
    for value in np.unique(bin_id[finite]):
        selected = finite & (bin_id == value)
        if np.sum(selected) < min_bin_samples:
            continue
        scalar_centers.append(float(np.median(scalar_mag[selected])))
        xyz_medians.append(np.median(mag_xyz[selected], axis=0))
    if len(scalar_centers) < degree + 2:
        raise ValueError("Not enough populated scalar-field bins for XYZ fit")

    scalar_centers_array = np.asarray(scalar_centers)
    xyz_medians_array = np.asarray(xyz_medians)
    center = float(np.median(scalar_centers_array))
    scale = max(float(np.std(scalar_centers_array)), scalar_bin_mg)
    normalized = (scalar_centers_array - center) / scale
    design = np.column_stack(
        [normalized**order for order in range(degree + 1)]
    )
    coefficients = np.linalg.lstsq(design, xyz_medians_array, rcond=None)[0]

    travel_grid = np.arange(
        0.0, travel_max_mm + 0.5 * travel_step_mm, travel_step_mm
    )
    scalar_grid = invert_scalar_travel_model(
        travel_grid, scalar_coefficients, scalar_offset_mm
    )
    normalized_grid = (scalar_grid - center) / scale
    xyz_grid = sum(
        normalized_grid[:, np.newaxis] ** order * coefficients[order]
        for order in range(degree + 1)
    )
    return ScalarParameterizedXYZModel(
        travel_grid=travel_grid,
        xyz_grid=xyz_grid,
        scalar_center=center,
        scalar_scale=scale,
        coefficients=coefficients,
        bin_count=len(scalar_centers_array),
    )


def _validate_series(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    travel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    gyro_dps = np.asarray(gyro_dps, dtype=float)
    mag_xyz = np.asarray(mag_xyz, dtype=float)
    travel = np.asarray(travel, dtype=float).reshape(-1)
    n = len(time_s)
    if gyro_dps.shape != (n, 3) or mag_xyz.shape != (n, 3) or len(travel) != n:
        raise ValueError(
            "Expected time/travel shape (N,) and gyro/mag shape (N, 3); "
            f"got {time_s.shape}, {travel.shape}, {gyro_dps.shape}, {mag_xyz.shape}"
        )
    if n < 2 or not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing and contain at least two samples")
    if not (
        np.all(np.isfinite(time_s))
        and np.all(np.isfinite(gyro_dps))
        and np.all(np.isfinite(mag_xyz))
        and np.all(np.isfinite(travel))
    ):
        raise ValueError("Inputs must be finite")
    return time_s, gyro_dps, mag_xyz, travel


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_from_vector(rotation_vector_rad: np.ndarray) -> np.ndarray:
    """Rodrigues rotation for a length-3 rotation vector in radians."""

    vector = np.asarray(rotation_vector_rad, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"rotation vector must be length-3, got {vector.shape}")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + skew(vector)
    axis_skew = skew(vector / angle)
    return (
        np.eye(3)
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    )


def integrate_gyro(time_s: np.ndarray, gyro_dps: np.ndarray) -> np.ndarray:
    """Return body-to-initial-frame rotations from body-frame gyro samples."""

    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    gyro_dps = np.asarray(gyro_dps, dtype=float)
    if gyro_dps.shape != (len(time_s), 3):
        raise ValueError("gyro_dps must have shape (len(time_s), 3)")
    rotations = np.empty((len(time_s), 3, 3), dtype=float)
    rotations[0] = np.eye(3)
    for index in range(1, len(time_s)):
        dt = float(time_s[index] - time_s[index - 1])
        omega_dps = 0.5 * (gyro_dps[index - 1] + gyro_dps[index])
        delta = rotation_from_vector(np.deg2rad(omega_dps) * dt)
        rotations[index] = rotations[index - 1] @ delta
    return rotations


def fit_linear_xyz_model(
    travel: np.ndarray,
    mag_xyz: np.ndarray,
    training_mask: np.ndarray,
    *,
    threshold_mg: float = 1500.0,
    bin_mm: float = 5.0,
    min_bin_samples: int = 5,
) -> LinearXYZModel:
    """Fit a density-balanced line to low-field travel bins.

    Samples are reduced to a median XYZ value per travel bin so the abundant
    low-travel samples do not dominate. Selection is refined using predicted
    rather than measured magnitude, avoiding an ambient-field-dependent gate.
    """

    travel = np.asarray(travel, dtype=float).reshape(-1)
    mag_xyz = np.asarray(mag_xyz, dtype=float)
    training_mask = np.asarray(training_mask, dtype=bool).reshape(-1)
    if mag_xyz.shape != (len(travel), 3) or len(training_mask) != len(travel):
        raise ValueError("travel, mag_xyz, and training_mask lengths must match")
    finite = training_mask & np.isfinite(travel) & np.all(np.isfinite(mag_xyz), axis=1)
    if np.sum(finite) < min_bin_samples * 3:
        raise ValueError("Not enough finite training samples for a low-field XYZ model")

    bin_id = np.floor(travel / bin_mm).astype(int)
    centers: list[float] = []
    medians: list[np.ndarray] = []
    for value in np.unique(bin_id[finite]):
        selected = finite & (bin_id == value)
        if np.sum(selected) < min_bin_samples:
            continue
        centers.append(float(np.median(travel[selected])))
        medians.append(np.median(mag_xyz[selected], axis=0))
    centers_arr = np.asarray(centers)
    medians_arr = np.asarray(medians)
    if len(centers_arr) < 3:
        raise ValueError("Fewer than three populated travel bins")

    selected_bins = np.linalg.norm(medians_arr, axis=1) <= threshold_mg
    if np.sum(selected_bins) < 3:
        selected_bins[np.argsort(np.linalg.norm(medians_arr, axis=1))[:3]] = True

    design_all = np.column_stack((centers_arr, np.ones(len(centers_arr))))
    for _ in range(3):
        coefficients = np.linalg.lstsq(
            design_all[selected_bins], medians_arr[selected_bins], rcond=None
        )[0]
        predicted = design_all @ coefficients
        refined = np.linalg.norm(predicted, axis=1) <= threshold_mg
        if np.sum(refined) < 3 or np.array_equal(refined, selected_bins):
            break
        selected_bins = refined

    selected_travel = centers_arr[selected_bins]
    return LinearXYZModel(
        slope=coefficients[0],
        intercept=coefficients[1],
        travel_min=float(np.min(selected_travel) - 0.5 * bin_mm),
        travel_max=float(np.max(selected_travel) + 0.5 * bin_mm),
        bin_count=int(np.sum(selected_bins)),
    )


def continuous_residuals(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    expected_mag: np.ndarray,
    update_mask: np.ndarray,
    body_field: np.ndarray,
    world_field: np.ndarray,
    weights: MagSolverWeights,
) -> np.ndarray:
    """Flattened residuals equivalent to the smoother's Gaussian model."""

    weights.validate()
    n = len(time_s)
    expected_mag = np.asarray(expected_mag, dtype=float)
    update_mask = np.asarray(update_mask, dtype=bool).reshape(-1)
    body_field = np.asarray(body_field, dtype=float)
    world_field = np.asarray(world_field, dtype=float)
    if any(array.shape != (n, 3) for array in (mag_xyz, expected_mag, body_field, world_field)):
        raise ValueError("mag/expected/body/world fields must all have shape (N, 3)")
    rotations = integrate_gyro(time_s, gyro_dps)
    residuals: list[np.ndarray] = [
        (body_field[0] / weights.body_initial_sigma).ravel(),
        (world_field[0] / weights.world_initial_sigma).ravel(),
    ]
    for index in range(n):
        if update_mask[index]:
            residuals.append(
                (
                    mag_xyz[index]
                    - expected_mag[index]
                    - body_field[index]
                    - world_field[index]
                )
                / weights.mag_sigma
            )
        if index == 0:
            continue
        dt = float(time_s[index] - time_s[index - 1])
        delta = rotations[index - 1].T @ rotations[index]
        expected_world = world_field[index - 1] @ delta
        residuals.append(
            (body_field[index] - body_field[index - 1])
            / (weights.body_rw * np.sqrt(dt))
        )
        residuals.append(
            (world_field[index] - expected_world)
            / (weights.world_rw * np.sqrt(dt))
        )
    return np.concatenate([np.asarray(value).ravel() for value in residuals])


def smooth_body_world_fields(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    expected_mag: np.ndarray,
    update_mask: np.ndarray,
    weights: MagSolverWeights,
    measurement_covariances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman/RTS solution of the continuous body/world field model.

    ``measurement_covariances`` may supply a different 3x3 covariance at each
    sample.  This is useful for curve-normal measurements: uncertainty can be
    kept small perpendicular to the expected magnet curve and made large along
    its tangent, where travel error and additive field are locally ambiguous.
    """

    weights.validate()
    n = len(time_s)
    expected_mag = np.asarray(expected_mag, dtype=float)
    update_mask = np.asarray(update_mask, dtype=bool).reshape(-1)
    if expected_mag.shape != (n, 3) or len(update_mask) != n:
        raise ValueError("expected_mag and update_mask must align with time_s")
    if measurement_covariances is not None:
        measurement_covariances = np.asarray(measurement_covariances, dtype=float)
        if measurement_covariances.shape != (n, 3, 3):
            raise ValueError(
                "measurement_covariances must have shape (len(time_s), 3, 3)"
            )
        if not np.all(np.isfinite(measurement_covariances)):
            raise ValueError("measurement_covariances must be finite")

    rotations = integrate_gyro(time_s, gyro_dps)
    filtered_state = np.empty((n, 6), dtype=float)
    filtered_cov = np.empty((n, 6, 6), dtype=float)
    predicted_state = np.empty((n, 6), dtype=float)
    predicted_cov = np.empty((n, 6, 6), dtype=float)
    transitions = np.empty((n, 6, 6), dtype=float)

    state = np.zeros(6, dtype=float)
    covariance = np.diag(
        [weights.body_initial_sigma**2] * 3
        + [weights.world_initial_sigma**2] * 3
    )
    identity6 = np.eye(6)
    measurement_design = np.column_stack((np.eye(3), np.eye(3)))
    default_measurement_cov = np.eye(3) * weights.mag_sigma**2

    for index in range(n):
        if index == 0:
            transition = identity6
        else:
            dt = float(time_s[index] - time_s[index - 1])
            delta = rotations[index - 1].T @ rotations[index]
            transition = np.zeros((6, 6), dtype=float)
            transition[:3, :3] = np.eye(3)
            # Column-vector state: w_i = delta.T @ w_(i-1).
            transition[3:, 3:] = delta.T
            process_cov = np.diag(
                [weights.body_rw**2 * dt] * 3
                + [weights.world_rw**2 * dt] * 3
            )
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_cov

        transitions[index] = transition
        predicted_state[index] = state
        predicted_cov[index] = covariance
        if update_mask[index]:
            measurement_cov = (
                default_measurement_cov
                if measurement_covariances is None
                else measurement_covariances[index]
            )
            observation = mag_xyz[index] - expected_mag[index]
            innovation = observation - measurement_design @ state
            innovation_cov = (
                measurement_design @ covariance @ measurement_design.T
                + measurement_cov
            )
            gain = np.linalg.solve(
                innovation_cov, measurement_design @ covariance
            ).T
            state = state + gain @ innovation
            covariance = (identity6 - gain @ measurement_design) @ covariance
            covariance = 0.5 * (covariance + covariance.T)
        filtered_state[index] = state
        filtered_cov[index] = covariance

    smoothed_state = filtered_state.copy()
    smoothed_cov = filtered_cov.copy()
    for index in range(n - 2, -1, -1):
        smoother_gain = np.linalg.solve(
            predicted_cov[index + 1],
            transitions[index + 1] @ filtered_cov[index],
        ).T
        smoothed_state[index] = filtered_state[index] + smoother_gain @ (
            smoothed_state[index + 1] - predicted_state[index + 1]
        )
        smoothed_cov[index] = filtered_cov[index] + smoother_gain @ (
            smoothed_cov[index + 1] - predicted_cov[index + 1]
        ) @ smoother_gain.T

    return smoothed_state[:, :3], smoothed_state[:, 3:]


def curve_tangent_covariances(
    xyz_model: object,
    travel: np.ndarray,
    normal_sigma_mg: float,
    tangent_sigma_ratio: float,
) -> np.ndarray:
    """Build XYZ measurement covariances elongated along the model tangent."""

    if normal_sigma_mg <= 0.0:
        raise ValueError("normal_sigma_mg must be positive")
    if tangent_sigma_ratio < 1.0 or not np.isfinite(tangent_sigma_ratio):
        raise ValueError("tangent_sigma_ratio must be finite and at least one")

    travel = np.asarray(travel, dtype=float).reshape(-1)
    if hasattr(xyz_model, "slope"):
        tangents = np.broadcast_to(
            np.asarray(xyz_model.slope, dtype=float), (len(travel), 3)
        ).copy()
    else:
        grid = np.asarray(xyz_model.travel_grid, dtype=float)
        xyz_grid = np.asarray(xyz_model.xyz_grid, dtype=float)
        derivative_grid = np.gradient(xyz_grid, grid, axis=0)
        tangents = np.column_stack(
            [
                np.interp(travel, grid, derivative_grid[:, axis])
                for axis in range(3)
            ]
        )

    norms = np.linalg.norm(tangents, axis=1)
    usable = norms > 1e-12
    tangent_unit = np.zeros_like(tangents)
    tangent_unit[usable] = tangents[usable] / norms[usable, np.newaxis]
    tangent_unit[~usable] = np.array([1.0, 0.0, 0.0])

    normal_variance = normal_sigma_mg**2
    extra_tangent_variance = normal_variance * (tangent_sigma_ratio**2 - 1.0)
    return (
        normal_variance * np.eye(3)[np.newaxis, :, :]
        + extra_tangent_variance
        * tangent_unit[:, :, np.newaxis]
        * tangent_unit[:, np.newaxis, :]
    )


def curve_slope_covariances(
    xyz_model: object,
    travel: np.ndarray,
    mag_sigma_mg: float,
    travel_sigma_mm: float,
    *,
    normal_slope_fraction: float = 0.0,
) -> np.ndarray:
    """Propagate travel uncertainty through a travel-to-XYZ curve.

    ``normal_slope_fraction=0`` is the first-order covariance
    ``R_mag + J sigma_t^2 J.T``. Positive values additionally inflate the two
    curve-normal directions as a heuristic for tangent-direction and model
    error that grows with local curve slope. A value of one is isotropic.
    """

    if mag_sigma_mg <= 0.0:
        raise ValueError("mag_sigma_mg must be positive")
    if travel_sigma_mm < 0.0 or not np.isfinite(travel_sigma_mm):
        raise ValueError("travel_sigma_mm must be finite and nonnegative")
    if not 0.0 <= normal_slope_fraction <= 1.0:
        raise ValueError("normal_slope_fraction must be between zero and one")

    travel = np.asarray(travel, dtype=float).reshape(-1)
    if hasattr(xyz_model, "slope"):
        tangents = np.broadcast_to(
            np.asarray(xyz_model.slope, dtype=float), (len(travel), 3)
        ).copy()
    else:
        grid = np.asarray(xyz_model.travel_grid, dtype=float)
        xyz_grid = np.asarray(xyz_model.xyz_grid, dtype=float)
        derivative_grid = np.gradient(xyz_grid, grid, axis=0)
        tangents = np.column_stack(
            [
                np.interp(travel, grid, derivative_grid[:, axis])
                for axis in range(3)
            ]
        )

    slopes = np.linalg.norm(tangents, axis=1)
    usable = slopes > 1e-12
    tangent_unit = np.zeros_like(tangents)
    tangent_unit[usable] = tangents[usable] / slopes[usable, np.newaxis]
    tangent_unit[~usable] = np.array([1.0, 0.0, 0.0])

    travel_induced_sigma = slopes * travel_sigma_mm
    tangent_variance = mag_sigma_mg**2 + travel_induced_sigma**2
    normal_variance = mag_sigma_mg**2 + (
        normal_slope_fraction * travel_induced_sigma
    ) ** 2
    return (
        normal_variance[:, np.newaxis, np.newaxis]
        * np.eye(3)[np.newaxis, :, :]
        + (tangent_variance - normal_variance)[:, np.newaxis, np.newaxis]
        * tangent_unit[:, :, np.newaxis]
        * tangent_unit[:, np.newaxis, :]
    )


def solve_iterative_correction(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    initial_travel: np.ndarray,
    xyz_model: LinearXYZModel,
    weights: MagSolverWeights | None = None,
    *,
    iterations: int = 4,
    tangent_sigma_ratio: float = 1.0,
    travel_sigma_mm: float | None = None,
    normal_slope_fraction: float = 0.0,
) -> MagCorrectionResult:
    """Alternate field smoothing and low-field XYZ-to-travel inversion.

    ``tangent_sigma_ratio=1`` retains the original isotropic full-vector
    observation. Larger values progressively downweight residuals parallel to
    the current travel-to-XYZ curve. Alternatively, ``travel_sigma_mm``
    propagates an assumed travel uncertainty through the local curve slope.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if tangent_sigma_ratio < 1.0 or not np.isfinite(tangent_sigma_ratio):
        raise ValueError("tangent_sigma_ratio must be finite and at least one")
    if travel_sigma_mm is not None:
        if travel_sigma_mm < 0.0 or not np.isfinite(travel_sigma_mm):
            raise ValueError("travel_sigma_mm must be finite and nonnegative")
        if tangent_sigma_ratio != 1.0:
            raise ValueError(
                "Use either tangent_sigma_ratio or travel_sigma_mm, not both"
            )
    if not 0.0 <= normal_slope_fraction <= 1.0:
        raise ValueError("normal_slope_fraction must be between zero and one")
    if travel_sigma_mm is None and normal_slope_fraction != 0.0:
        raise ValueError("normal_slope_fraction requires travel_sigma_mm")
    weights = weights or MagSolverWeights()
    time_s, gyro_dps, mag_xyz, initial_travel = _validate_series(
        time_s, gyro_dps, mag_xyz, initial_travel
    )
    travel = initial_travel.copy()
    changes: list[float] = []
    body = np.zeros_like(mag_xyz)
    world = np.zeros_like(mag_xyz)
    update_mask = np.zeros(len(time_s), dtype=bool)

    for _ in range(iterations):
        expected = xyz_model.predict(travel)
        fit_mask = xyz_model.weak(travel, weights.mag_update_threshold)
        if travel_sigma_mm is not None and travel_sigma_mm > 0.0:
            measurement_covariances = curve_slope_covariances(
                xyz_model,
                travel,
                weights.mag_sigma,
                travel_sigma_mm,
                normal_slope_fraction=normal_slope_fraction,
            )
        elif tangent_sigma_ratio != 1.0:
            measurement_covariances = curve_tangent_covariances(
                xyz_model,
                travel,
                weights.mag_sigma,
                tangent_sigma_ratio,
            )
        else:
            measurement_covariances = None
        body, world = smooth_body_world_fields(
            time_s,
            gyro_dps,
            mag_xyz,
            expected,
            fit_mask,
            weights,
            measurement_covariances=measurement_covariances,
        )
        corrected = mag_xyz - body - world
        inferred = xyz_model.infer(corrected)
        update_mask = fit_mask & (
            np.linalg.norm(mag_xyz, axis=1) <= weights.mag_update_threshold
        )
        next_travel = initial_travel.copy()
        next_travel[update_mask] = inferred[update_mask]
        changes.append(float(np.sqrt(np.mean((next_travel - travel) ** 2))))
        travel = next_travel

    update_mask = xyz_model.weak(travel, weights.mag_update_threshold) & (
        np.linalg.norm(mag_xyz, axis=1) <= weights.mag_update_threshold
    )
    correction = body + world
    corrected_weak = mag_xyz.copy()
    corrected_weak[update_mask] -= correction[update_mask]
    return MagCorrectionResult(
        travel=travel,
        body_field=body,
        world_field=world,
        correction=correction,
        corrected_mag_weak=corrected_weak,
        corrected_mag_all=mag_xyz - correction,
        update_mask=update_mask,
        iteration_change_mm=changes,
    )
