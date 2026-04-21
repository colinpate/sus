from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import csr_matrix, lil_matrix


def flatten_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def dense_index_mask(length: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < length)]
    mask[idx] = True
    return mask


@dataclass(frozen=True)
class SolverWeights:
    """Residual weights and gates for the travel optimizer."""

    v0: float = 5
    x0: float = 1000.0
    mag_x: float = 200.0
    mag_x_thresh: float = 500.0
    mag_off_floor: float = 0.1
    zupt_v: float = 320.0
    b: float = 1.0
    oob: float = 1000.0
    travel_max: float = 170.0


def solver_weights_for_mag_baseline(mag_baseline: float, **overrides: float) -> SolverWeights:
    return SolverWeights(mag_x_thresh=float(mag_baseline), **overrides)


@dataclass(frozen=True)
class SolverInputs:
    """Normalized solver inputs with shared clipping, dt, and ZUPT-mask handling."""

    time_s: np.ndarray
    accel_mm_s2: np.ndarray
    mag: np.ndarray
    mag_preds_mm: np.ndarray
    mag_zv_points: np.ndarray
    mag_baseline: float
    initial_dt_s: float = 0.01
    dt_s: np.ndarray = field(init=False)
    mag_zv_mask: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        time_s = flatten_1d(self.time_s)
        accel = flatten_1d(self.accel_mm_s2)
        mag = flatten_1d(self.mag)
        mag_preds = np.clip(flatten_1d(self.mag_preds_mm), 0, None)

        n = len(time_s)
        if n == 0:
            raise ValueError("solver inputs must contain at least one sample")
        for name, arr in (
            ("accel_mm_s2", accel),
            ("mag", mag),
            ("mag_preds_mm", mag_preds),
        ):
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match time_s length {n}")

        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "accel_mm_s2", accel)
        object.__setattr__(self, "mag", mag)
        object.__setattr__(self, "mag_preds_mm", mag_preds)
        object.__setattr__(self, "mag_zv_points", np.asarray(self.mag_zv_points, dtype=int).reshape(-1))
        object.__setattr__(self, "mag_baseline", float(self.mag_baseline))
        object.__setattr__(self, "dt_s", np.diff(time_s, prepend=time_s[0] - self.initial_dt_s))
        object.__setattr__(self, "mag_zv_mask", dense_index_mask(n, self.mag_zv_points))


@dataclass(frozen=True)
class SolverTerms:
    x: np.ndarray
    v: np.ndarray
    b: float
    gate: np.ndarray
    v_res: np.ndarray
    x_res: np.ndarray
    mag_res: np.ndarray
    zv_res: np.ndarray
    oob_res: np.ndarray
    stacked: np.ndarray


@dataclass(frozen=True)
class PreparedSolver:
    inputs: SolverInputs
    weights: SolverWeights
    mag_anchor_mask: np.ndarray
    mag_gate: np.ndarray
    jac_sparsity: csr_matrix


@dataclass(frozen=True)
class SolverRun:
    state0: np.ndarray
    state: np.ndarray
    init_terms: SolverTerms
    opt_terms: SolverTerms
    scipy_result: OptimizeResult

    @property
    def x(self) -> np.ndarray:
        return self.opt_terms.x


def make_initial_state(inputs: SolverInputs) -> np.ndarray:
    n = len(inputs.time_s)
    state = np.zeros(2 * n + 1)
    state[:n] = inputs.mag_preds_mm
    state[n : 2 * n] = np.cumsum(inputs.accel_mm_s2) * np.mean(inputs.dt_s)
    return state


def prepare_solver(inputs: SolverInputs, weights: SolverWeights) -> PreparedSolver:
    mag_anchor_mask = inputs.mag > weights.mag_x_thresh
    mag_gate = (mag_anchor_mask[1:].astype(float) + weights.mag_off_floor) / (1.0 + weights.mag_off_floor)
    return PreparedSolver(
        inputs=inputs,
        weights=weights,
        mag_anchor_mask=mag_anchor_mask,
        mag_gate=mag_gate,
        jac_sparsity=make_jac_sparsity(len(inputs.time_s)),
    )


def make_jac_sparsity(n: int, n_res_per_step: int = 5) -> csr_matrix:
    n_var = 2 * n + 1
    n_res = n_res_per_step * (n - 1) + 1
    jac = lil_matrix((n_res, n_var), dtype=bool)
    ix_b = 2 * n

    for i in range(1, n):
        r0 = (i - 1) * n_res_per_step

        jac[r0 + 0, n + i] = True
        jac[r0 + 0, n + i - 1] = True
        jac[r0 + 0, ix_b] = True

        jac[r0 + 1, i] = True
        jac[r0 + 1, i - 1] = True
        jac[r0 + 1, n + i - 1] = True
        jac[r0 + 1, ix_b] = True

        jac[r0 + 2, i] = True
        jac[r0 + 3, n + i] = True
        jac[r0 + 4, i] = True

    jac[n_res - 1, ix_b] = True
    return jac.tocsr()


def calculate_solver_terms(
    state: np.ndarray,
    prepared: PreparedSolver,
) -> SolverTerms:
    inputs = prepared.inputs
    weights = prepared.weights
    n = len(inputs.time_s)
    x = state[:n]
    v = state[n : 2 * n]
    b = float(state[2 * n])

    dt = inputs.dt_s[1:]
    a = inputs.accel_mm_s2[: n - 1] - b
    v_res = v[:-1] + a * dt - v[1:]
    x_res = x[:-1] + v[:-1] * dt + 0.5 * a * dt**2 - x[1:]
    mag_res = (inputs.mag_preds_mm[1:] - x[1:]) * prepared.mag_gate
    zv_res = inputs.mag_zv_mask[1:] * v[1:]
    oob_res = x[1:] * (x[1:] < 0)
    oob_res += (x[1:] - weights.travel_max) * (x[1:] > weights.travel_max)

    res = np.zeros((5, n), dtype=float)
    res[0, 1:] = v_res * weights.v0
    res[1, 1:] = x_res * weights.x0
    res[2, 1:] = mag_res * weights.mag_x
    res[3, 1:] = zv_res * weights.zupt_v
    res[4, 1:] = oob_res * weights.oob

    return SolverTerms(
        x=x,
        v=v,
        b=b,
        gate=prepared.mag_gate,
        v_res=v_res,
        x_res=x_res,
        mag_res=mag_res,
        zv_res=zv_res,
        oob_res=oob_res,
        stacked=np.concatenate((res[:, 1:].ravel(order="F"), [b * weights.b])),
    )


def solve_prepared_travel(
    prepared: PreparedSolver,
    *,
    max_nfev: int = 100,
    verbose: int = 0,
) -> SolverRun:
    inputs = prepared.inputs
    state0 = make_initial_state(inputs)

    def residuals(state: np.ndarray) -> np.ndarray:
        return calculate_solver_terms(state, prepared).stacked

    result = least_squares(
        fun=residuals,
        x0=state0,
        jac_sparsity=prepared.jac_sparsity,
        jac="2-point",
        method="trf",
        x_scale="jac",
        verbose=verbose,
        max_nfev=max_nfev,
    )
    return SolverRun(
        state0=state0,
        state=result.x,
        init_terms=calculate_solver_terms(state0, prepared),
        opt_terms=calculate_solver_terms(result.x, prepared),
        scipy_result=result,
    )


def solve_travel(
    inputs: SolverInputs,
    weights: SolverWeights,
    *,
    max_nfev: int = 100,
    verbose: int = 0,
) -> SolverRun:
    return solve_prepared_travel(
        prepare_solver(inputs, weights),
        max_nfev=max_nfev,
        verbose=verbose,
    )


def term_costs(terms: SolverTerms, weights: SolverWeights) -> dict[str, float]:
    return {
        "v_dyn": float(np.mean((terms.v_res * weights.v0) ** 2)),
        "x_dyn": float(np.mean((terms.x_res * weights.x0) ** 2)),
        "mag_anchor": float(np.mean((terms.mag_res * weights.mag_x) ** 2)),
        "zupt": float(np.mean((terms.zv_res * weights.zupt_v) ** 2)) if weights.zupt_v else 0.0,
        "oob": float(np.mean((terms.oob_res * weights.oob) ** 2)) if weights.oob else 0.0,
    }
