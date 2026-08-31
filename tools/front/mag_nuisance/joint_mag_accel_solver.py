"""Sparse joint travel and body/world magnetic-field optimizer.

The nonlinear magnetic path is linearized around the current latent travel.
Each Gauss-Newton step jointly updates travel, a slowly changing body-fixed
field, and a gyro-transported world-fixed field.  Short accelerometer windows
provide relative-travel factors that do not require an absolute travel label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsmr

from tools.front.mag_nuisance.mag_correction_solver import (
    MagSolverWeights,
    integrate_gyro,
)


@dataclass(frozen=True)
class RelativeTravelFactor:
    center_index: int
    sample_index: int
    displacement_mm: float
    weight_scale: float = 1.0


@dataclass(frozen=True)
class JointSolverWeights:
    field: MagSolverWeights = field(default_factory=MagSolverWeights)
    tangent_sigma_ratio: float = 5.0
    accel_sigma_mm: float = 5.0
    mag_huber_sigma: float = 4.0
    travel_prior_sigma_mm: float = 1.0
    travel_prior_stride_s: float = 0.1
    travel_correction_rw: float = 0.25  # mm / sqrt(second)
    travel_min_mm: float = 0.0
    travel_max_mm: float = 210.0
    max_travel_step_mm: float = 30.0
    max_field_step_mg: float = 600.0

    def validate(self) -> None:
        self.field.validate()
        for name in (
            "tangent_sigma_ratio",
            "accel_sigma_mm",
            "mag_huber_sigma",
            "travel_prior_sigma_mm",
            "travel_prior_stride_s",
            "travel_correction_rw",
            "travel_max_mm",
            "max_travel_step_mm",
            "max_field_step_mg",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tangent_sigma_ratio < 1.0:
            raise ValueError("tangent_sigma_ratio must be at least one")
        if self.travel_max_mm <= self.travel_min_mm:
            raise ValueError("travel_max_mm must exceed travel_min_mm")


@dataclass
class JointSolverResult:
    travel: np.ndarray
    body_field: np.ndarray
    world_field: np.ndarray
    iteration_diagnostics: list[dict[str, float | int]]

    @property
    def correction(self) -> np.ndarray:
        return self.body_field + self.world_field


class SparseRows:
    def __init__(self, n_variables: int) -> None:
        self.n_variables = n_variables
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.target: list[float] = []

    @property
    def row_count(self) -> int:
        return len(self.target)

    def add(self, columns: list[int], values: list[float], target: float) -> None:
        row = self.row_count
        self.rows.extend([row] * len(columns))
        self.columns.extend(columns)
        self.values.extend(values)
        self.target.append(float(target))

    def matrix_and_target(self) -> tuple[coo_matrix, np.ndarray]:
        matrix = coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(self.row_count, self.n_variables),
        ).tocsr()
        return matrix, np.asarray(self.target, dtype=float)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float).copy()
    norms = np.linalg.norm(vectors, axis=1)
    bad = norms < 1e-9
    vectors[bad] = np.array([1.0, 0.0, 0.0])
    norms[bad] = 1.0
    return vectors / norms[:, np.newaxis]


def _indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    travel = np.arange(n)
    body = n + np.arange(3 * n).reshape(n, 3)
    world = 4 * n + np.arange(3 * n).reshape(n, 3)
    return travel, body, world


def _mag_whitener(
    tangent_unit: np.ndarray,
    normal_sigma_mg: float,
    tangent_sigma_mg: float,
) -> np.ndarray:
    tangent_outer = np.outer(tangent_unit, tangent_unit)
    return (
        (np.eye(3) - tangent_outer) / normal_sigma_mg
        + tangent_outer / tangent_sigma_mg
    )


def build_linearized_system(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    initial_travel: np.ndarray,
    travel: np.ndarray,
    body_field: np.ndarray,
    world_field: np.ndarray,
    expected_xyz: np.ndarray,
    travel_derivative_xyz: np.ndarray,
    covariance_tangent_xyz: np.ndarray,
    accel_factors: list[RelativeTravelFactor],
    travel_prior_mask: np.ndarray | None,
    weights: JointSolverWeights,
) -> tuple[coo_matrix, np.ndarray, dict[str, float]]:
    """Construct ``A delta = target`` for one Gauss-Newton iteration."""

    weights.validate()
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    gyro_dps = np.asarray(gyro_dps, dtype=float)
    mag_xyz = np.asarray(mag_xyz, dtype=float)
    initial_travel = np.asarray(initial_travel, dtype=float).reshape(-1)
    travel = np.asarray(travel, dtype=float).reshape(-1)
    body_field = np.asarray(body_field, dtype=float)
    world_field = np.asarray(world_field, dtype=float)
    expected_xyz = np.asarray(expected_xyz, dtype=float)
    derivative = np.asarray(travel_derivative_xyz, dtype=float)
    tangent_unit = _unit_rows(covariance_tangent_xyz)
    n = len(time_s)
    if any(array.shape != (n, 3) for array in (
        gyro_dps,
        mag_xyz,
        body_field,
        world_field,
        expected_xyz,
        derivative,
        tangent_unit,
    )):
        raise ValueError("All XYZ inputs must have shape (len(time_s), 3)")
    if len(initial_travel) != n or len(travel) != n:
        raise ValueError("travel arrays must align with time_s")
    if travel_prior_mask is None:
        travel_prior_mask = np.ones(n, dtype=bool)
    else:
        travel_prior_mask = np.asarray(travel_prior_mask, dtype=bool).reshape(-1)
        if len(travel_prior_mask) != n:
            raise ValueError("travel_prior_mask must align with time_s")

    travel_index, body_index, world_index = _indices(n)
    rows = SparseRows(7 * n)
    field_weights = weights.field
    tangent_sigma_mg = field_weights.mag_sigma * weights.tangent_sigma_ratio

    mag_residual = mag_xyz - expected_xyz - body_field - world_field
    whitened_mag_norms: list[float] = []
    for index in range(n):
        whitener = _mag_whitener(
            tangent_unit[index], field_weights.mag_sigma, tangent_sigma_mg
        )
        whitened_residual = whitener @ mag_residual[index]
        residual_norm = float(np.linalg.norm(whitened_residual))
        whitened_mag_norms.append(residual_norm)
        robust_scale = min(
            1.0,
            np.sqrt(weights.mag_huber_sigma / max(residual_norm, 1e-12)),
        )
        whitened_derivative = robust_scale * whitener @ derivative[index]
        whitened_target = robust_scale * whitened_residual
        whitened_identity = robust_scale * whitener
        for axis in range(3):
            columns = [int(travel_index[index])]
            values = [float(whitened_derivative[axis])]
            for component in range(3):
                value = float(whitened_identity[axis, component])
                if abs(value) < 1e-15:
                    continue
                columns.extend(
                    [
                        int(body_index[index, component]),
                        int(world_index[index, component]),
                    ]
                )
                values.extend([value, value])
            rows.add(columns, values, float(whitened_target[axis]))

    rotations = integrate_gyro(time_s, gyro_dps)
    for index in range(1, n):
        dt = float(time_s[index] - time_s[index - 1])
        body_sigma = field_weights.body_rw * np.sqrt(dt)
        world_sigma = field_weights.world_rw * np.sqrt(dt)
        delta_rotation = rotations[index - 1].T @ rotations[index]
        world_transition = delta_rotation.T
        for component in range(3):
            body_residual = body_field[index, component] - body_field[index - 1, component]
            rows.add(
                [int(body_index[index, component]), int(body_index[index - 1, component])],
                [1.0 / body_sigma, -1.0 / body_sigma],
                -body_residual / body_sigma,
            )
            columns = [int(world_index[index, component])]
            values = [1.0 / world_sigma]
            for previous_component in range(3):
                transition_value = float(world_transition[component, previous_component])
                if abs(transition_value) < 1e-15:
                    continue
                columns.append(int(world_index[index - 1, previous_component]))
                values.append(-transition_value / world_sigma)
            world_residual = (
                world_field[index, component]
                - world_transition[component] @ world_field[index - 1]
            )
            rows.add(columns, values, -world_residual / world_sigma)

    for component in range(3):
        rows.add(
            [int(body_index[0, component])],
            [1.0 / field_weights.body_initial_sigma],
            -body_field[0, component] / field_weights.body_initial_sigma,
        )
        rows.add(
            [int(world_index[0, component])],
            [1.0 / field_weights.world_initial_sigma],
            -world_field[0, component] / field_weights.world_initial_sigma,
        )

    accel_residuals: list[float] = []
    for factor in accel_factors:
        sigma = weights.accel_sigma_mm / max(factor.weight_scale, 1e-12)
        residual = (
            travel[factor.sample_index]
            - travel[factor.center_index]
            - factor.displacement_mm
        )
        accel_residuals.append(float(residual))
        rows.add(
            [
                int(travel_index[factor.sample_index]),
                int(travel_index[factor.center_index]),
            ],
            [1.0 / sigma, -1.0 / sigma],
            -residual / sigma,
        )

    correction = travel - initial_travel
    correction_rw_residuals: list[float] = []
    for index in range(1, n):
        dt = float(time_s[index] - time_s[index - 1])
        sigma = weights.travel_correction_rw * np.sqrt(dt)
        residual = correction[index] - correction[index - 1]
        correction_rw_residuals.append(float(residual))
        rows.add(
            [int(travel_index[index]), int(travel_index[index - 1])],
            [1.0 / sigma, -1.0 / sigma],
            -residual / sigma,
        )

    prior_indices: list[int] = []
    last_prior_time = -np.inf
    for index in np.flatnonzero(travel_prior_mask):
        if time_s[index] - last_prior_time < weights.travel_prior_stride_s:
            continue
        prior_indices.append(int(index))
        last_prior_time = float(time_s[index])
    for index in prior_indices:
        residual = travel[index] - initial_travel[index]
        rows.add(
            [int(travel_index[index])],
            [1.0 / weights.travel_prior_sigma_mm],
            -residual / weights.travel_prior_sigma_mm,
        )

    matrix, target = rows.matrix_and_target()
    diagnostics = {
        "mag_whitened_rms": float(np.sqrt(np.mean(np.square(whitened_mag_norms)))),
        "accel_factor_rms_mm": (
            float(np.sqrt(np.mean(np.square(accel_residuals))))
            if accel_residuals
            else float("nan")
        ),
        "rows": rows.row_count,
        "variables": 7 * n,
        "travel_prior_factors": len(prior_indices),
        "travel_correction_rw_rms_mm": float(
            np.sqrt(np.mean(np.square(correction_rw_residuals)))
        ),
    }
    return matrix, target, diagnostics


def solve_joint_mag_accel(
    time_s: np.ndarray,
    gyro_dps: np.ndarray,
    mag_xyz: np.ndarray,
    initial_travel: np.ndarray,
    xyz_model: object,
    tangent_fn: Callable[[np.ndarray], np.ndarray],
    accel_factors: list[RelativeTravelFactor],
    weights: JointSolverWeights | None = None,
    *,
    travel_prior_mask: np.ndarray | None = None,
    iterations: int = 4,
    damping: float = 1.0,
    lsmr_maxiter: int = 2000,
    lsmr_tolerance: float = 1e-5,
) -> JointSolverResult:
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    if lsmr_tolerance <= 0.0:
        raise ValueError("lsmr_tolerance must be positive")
    weights = weights or JointSolverWeights()
    weights.validate()
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    initial_travel = np.asarray(initial_travel, dtype=float).reshape(-1)
    n = len(time_s)
    travel = np.clip(
        initial_travel.copy(), weights.travel_min_mm, weights.travel_max_mm
    )
    body = np.zeros((n, 3), dtype=float)
    world = np.zeros((n, 3), dtype=float)
    iteration_diagnostics: list[dict[str, float | int]] = []

    for iteration in range(iterations):
        expected = xyz_model.predict(travel)
        base_derivative = np.gradient(
            np.asarray(xyz_model.xyz_grid, dtype=float),
            np.asarray(xyz_model.travel_grid, dtype=float),
            axis=0,
        )
        derivative = np.column_stack(
            [
                np.interp(travel, xyz_model.travel_grid, base_derivative[:, axis])
                for axis in range(3)
            ]
        )
        covariance_tangent = np.asarray(tangent_fn(travel), dtype=float)
        tangent_unit = _unit_rows(covariance_tangent)
        derivative_sign = np.sum(derivative * tangent_unit, axis=1)
        tangent_unit[derivative_sign < 0.0] *= -1.0
        derivative = tangent_unit * np.linalg.norm(derivative, axis=1)[:, np.newaxis]

        matrix, target, diagnostics = build_linearized_system(
            time_s,
            gyro_dps,
            mag_xyz,
            initial_travel,
            travel,
            body,
            world,
            expected,
            derivative,
            tangent_unit,
            accel_factors,
            travel_prior_mask,
            weights,
        )
        # The travel and field-state columns use very different physical units
        # and can differ substantially in norm.  Unit-norm column scaling makes
        # LSMR's stopping criteria meaningful and keeps a capped solve from
        # implicitly favoring one state family over the other.
        column_norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        column_norm = np.maximum(column_norm, 1e-12)
        scaled_matrix = matrix @ diags(1.0 / column_norm)
        solution = lsmr(
            scaled_matrix,
            target,
            atol=lsmr_tolerance,
            btol=lsmr_tolerance,
            maxiter=lsmr_maxiter,
        )
        delta = solution[0] / column_norm
        delta_travel = np.clip(
            delta[:n], -weights.max_travel_step_mm, weights.max_travel_step_mm
        )
        delta_body = np.clip(
            delta[n : 4 * n].reshape(n, 3),
            -weights.max_field_step_mg,
            weights.max_field_step_mg,
        )
        delta_world = np.clip(
            delta[4 * n :].reshape(n, 3),
            -weights.max_field_step_mg,
            weights.max_field_step_mg,
        )
        travel = np.clip(
            travel + damping * delta_travel,
            weights.travel_min_mm,
            weights.travel_max_mm,
        )
        body += damping * delta_body
        world += damping * delta_world
        diagnostics.update(
            {
                "iteration": iteration + 1,
                "travel_step_rms_mm": float(np.sqrt(np.mean(delta_travel**2))),
                "travel_step_max_mm": float(np.max(np.abs(delta_travel))),
                "field_step_rms_mg": float(
                    np.sqrt(np.mean(np.square(delta_body + delta_world)))
                ),
                "lsmr_stop": int(solution[1]),
                "lsmr_iterations": int(solution[2]),
                "lsmr_condition": float(solution[6]),
            }
        )
        iteration_diagnostics.append(diagnostics)

    return JointSolverResult(
        travel=travel,
        body_field=body,
        world_field=world,
        iteration_diagnostics=iteration_diagnostics,
    )
