from dataclasses import dataclass

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step

# Create and define weights
@dataclass
class SolverWeights:
    v0: float =           2.5     # prior on v0 (m/s)
    x0: float =           5e2     # prior on x0 (m)
    mag_x: float =        200     # position anchor std (m)     
    mag_x_thresh: float = 500     # threshold for applying mag_x weight (mG)
    zupt_v: float =       320     # velocity at ZUPT std (m/s) 
    b: float =            1       # bias penalty (m/s^2)
    oob: float =          1000    # out-of-bounds penalty (m)

@dataclass
class TravelSolver(Step):
    """ Solve for travel using multiple inputs and a simple physical model """
    travel_max = 170

    def run(self, ws: Workspace) -> None:
        self.process_inputs(ws)

        N = len(self.accel)
        x0 = np.zeros(2*N + 2)  # [x(0..N-1), v(0..N-1), b]
        x0[:N] = self.mag_preds  # initial guess for x is the mag pred
        x0[N:2*N] = np.cumsum(self.accel) * np.mean(self.dt_s)  # initial guess for v is just integrating accel

        Jsp = self.make_jac_sparsity(N, include_bias=True)

        #res = least_squares(self.calculate_res, x0, jac_sparsity=Jsp, verbose=2, max_nfev=100)
        res = least_squares(
            fun=self.calculate_res,
            x0=x0, 
            jac_sparsity=Jsp, 
            jac="2-point",
            method="trf",
            x_scale="jac",
            verbose=1,
            max_nfev=100,
        )

        x_opt = res.x[:N]
        ws[self.outputs[0]] = TimeSeries(
            t=self.dt_s.cumsum(),
            x=x_opt,
            units="mm",
            frame=ws[self.inputs[0]].frame,
            meta={**ws[self.inputs[0]].meta},
        )

    def process_inputs(self, ws: Workspace) -> None:
        accel_ts: TimeSeries = ws[self.inputs[0]]
        mag: np.ndarray = ws[self.inputs[1]].x[:, 0]
        mag_preds: np.ndarray = ws[self.inputs[2]].x[:, 0]
        mag_zv_points: np.ndarray = ws[self.inputs[3]]
        mag_baseline: float = ws[self.inputs[4]][0]
        travel_ts: TimeSeries | None = ws.get(self.inputs[5])

        self.accel = accel_ts.x[:, 0] * 1000  # convert to mm/s^2
        t = accel_ts.t
        self.dt_s = np.diff(t, prepend=t[0]-0.01)
        self.mag = mag
        self.mag_preds = mag_preds
        dense_mag_zv = np.zeros_like(t)
        dense_mag_zv[mag_zv_points] = 1
        self.dense_mag_zv = dense_mag_zv
        if travel_ts is not None:
            travel = travel_ts.x[:, 0]
            self.v_gt = np.diff(travel, prepend=travel[0]) / self.dt_s

        self.weights = SolverWeights(
            mag_x_thresh = max(500, mag_baseline)
        )

    def calculate_res(self, vec):
        N = len(self.accel)
        x = vec[:N]
        v = vec[N:2*N]
        b = vec[2*N]  # <-- add bias as last variable
        #bx = vec[2*N+1]
        bx = 0

        dt = self.dt_s[1:]          # (N-1,)
        a  = self.accel[:N-1] - b           # (N-1,)

        acc_v = v[:-1] + a * dt      # predicts v[1:] from v[:-1]  -> (N-1,)
        acc_x = x[:-1] + v[:-1]*dt + 0.5 * a * dt**2   # predicts x[1:] -> (N-1,)

        v_res = acc_v - v[1:]        # (N-1,)
        x_res = acc_x - x[1:]        # (N-1,)

        mag_res_mask = ((self.mag[1:] > self.weights.mag_x_thresh) + 0.1) / 1.1
        mag_pred_res = (self.mag_preds[1:] - x[1:] - bx) * mag_res_mask  # (N-1,)

        # zv_res = v[i] if dense_mag_zv[i] else 0, for i=1..N-1
        zv_res = self.dense_mag_zv[1:] * v[1:]   # works if dense_mag_zv is bool or 0/1  -> (N-1,)

        # OOB penalty: if x[i] is outside [min_x, max_x], add penalty proportional to distance outside bounds
        oob_res = (x[1:]) * (x[1:] < 0)
        oob_res += (x[1:] - self.travel_max) * (x[1:] > self.travel_max)

        res = np.zeros((5, N), dtype=float)
        res[0, 1:] = v_res * self.weights.v0
        res[1, 1:] = x_res * self.weights.x0
        res[2, 1:] = mag_pred_res * self.weights.mag_x
        res[3, 1:] = zv_res * self.weights.zupt_v
        res[4, 1:] = oob_res * self.weights.oob
        
        r = np.concatenate((res[:, 1:].ravel(order="F"), [b * self.weights.b]))
        return r

    def make_jac_sparsity(self, N, n_res_per_step=5, include_bias=True, include_bias_penalty=True):
        # z = [x(0..N-1), v(0..N-1), b]
        n_var = 2*N + 1 + (1 if include_bias else 0)

        n_steps = N - 1
        n_res = n_res_per_step * n_steps + (1 if (include_bias and include_bias_penalty) else 0)

        J = lil_matrix((n_res, n_var), dtype=bool)

        def ix_x(i): return i
        def ix_v(i): return N + i
        ix_b = 2*N
        ix_bx = 2*N+1 # last variable

        for i in range(1, N):
            # This assumes your residual vector is packed as:
            # [step1(5 entries), step2(5 entries), ..., stepN-1(5 entries), bias_penalty]
            r0 = (i - 1) * n_res_per_step

            # v_dyn at step i depends on v[i], v[i-1], b
            J[r0 + 0, ix_v(i)] = True
            J[r0 + 0, ix_v(i - 1)] = True
            if include_bias:
                J[r0 + 0, ix_b] = True

            # x_dyn depends on x[i], x[i-1], v[i-1], b
            J[r0 + 1, ix_x(i)] = True
            J[r0 + 1, ix_x(i - 1)] = True
            J[r0 + 1, ix_v(i - 1)] = True
            if include_bias:
                J[r0 + 1, ix_b] = True
            J[r0 + 1, ix_bx] = True

            # mag anchor depends on x[i] (even if gated to 0, keep structure)
            J[r0 + 2, ix_x(i)] = True
            J[r0 + 2, ix_bx] = True

            # zupt depends on v[i] (even if gated)
            J[r0 + 3, ix_v(i)] = True

            # oob depends on x[i] (even if gated)
            J[r0 + 4, ix_x(i)] = True

        # Final residual: b*w.b depends only on b
        if include_bias and include_bias_penalty:
            J[n_res - 1, ix_b] = True

        return J.tocsr()