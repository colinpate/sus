"""Offline evaluation baselines; these outputs must not feed production estimation."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfiltfilt
from classes.step import Step
from classes.time_series import TimeSeries
from mag_calibration import fit_supervised_power_oracle
from mag_to_travel_model_core import MagToTravelModel

# Selected in experiments/imu_baseline/development-v3 (development-set tuning).
# Configuration overrides remain explicit.
DEFAULT_IMU_SETTINGS = {
    'front': {'cutoff_hz': 1.0, 'placement': 'both'},
    'rear': {'cutoff_hz': 4.0, 'placement': 'displacement'},
}


def project_bad_mask(source_t, source_bad, target_t, halo_s=0.):
    """Project contiguous invalid intervals without bridging holes between them."""
    source_t = np.asarray(source_t).reshape(-1)
    idx = np.flatnonzero(np.asarray(source_bad).reshape(-1))
    bad = np.zeros(len(target_t), dtype=bool)
    if not len(idx):
        return bad
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        bad |= (target_t >= source_t[run[0]] - halo_s) & (target_t <= source_t[run[-1]] + halo_s)
    return bad


def integrate_imu(t, accel, *, cutoff_hz, placement, bad=None, lowpass_hz=40., order=2, edge_s=1.):
    """m/s^2 -> mm, zero initial x/v per segment, zero-phase offline filtering.

    Restart at dropouts/nonfinite samples/time gaps. A fixed two-second margin
    at every segment boundary is excluded for every candidate (not a guarantee
    of complete filter settling). No reference or magnetic signal is consulted.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    a = np.asarray(accel, dtype=float).reshape(-1)
    if len(t) != len(a) or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Need aligned acceleration and strictly increasing finite timestamps')
    if placement not in ('velocity', 'displacement', 'both'):
        raise ValueError('Unknown HPF placement')
    dt = float(np.median(np.diff(t)))
    fs = 1 / dt
    if not 0 < cutoff_hz < fs / 2 or not 0 < lowpass_hz < fs / 2 or edge_s < 0:
        raise ValueError('Invalid filter cutoff or edge margin')
    valid = np.isfinite(a)
    if bad is not None:
        valid &= ~np.asarray(bad, dtype=bool).reshape(-1)
    indices = np.flatnonzero(valid)
    result = np.full(len(t), np.nan)
    if not len(indices):
        return result
    splits = np.flatnonzero((np.diff(indices) > 1) | (np.diff(t[indices]) > max(5 * dt, 0.1))) + 1
    hp = butter(order, cutoff_hz, 'highpass', fs=fs, output='sos')
    lp = butter(2, lowpass_hz, 'lowpass', fs=fs, output='sos')
    for ix in np.split(indices, splits):
        if len(ix) < max(24, int(2 * edge_s * fs) + 2):
            continue
        # Bridge isolated missing samples on a uniform grid, but never dropouts
        # or recording gaps longer than 100 ms (or five sample intervals).
        grid = np.arange(t[ix[0]], t[ix[-1]] + dt * .1, dt)
        acc = sosfiltfilt(lp, np.interp(grid, t[ix], a[ix]))
        v = cumulative_trapezoid(acc, grid, initial=0)
        if placement in ('velocity', 'both'):
            v = sosfiltfilt(hp, v)
        x = cumulative_trapezoid(v, grid, initial=0)
        if placement in ('displacement', 'both'):
            x = sosfiltfilt(hp, x)
        keep = (t[ix] >= t[ix[0]] + edge_s) & (t[ix] <= t[ix[-1]] - edge_s)
        result[ix[keep]] = np.interp(t[ix[keep]], grid, x) * 1000
    return result


def resample_valid(source_t, values, target_t):
    """Interpolate within finite segments only; never fill an invalid gap."""
    values = np.asarray(values).reshape(-1)
    result = np.full(len(target_t), np.nan)
    ix = np.flatnonzero(np.isfinite(values))
    if len(ix):
        dt = np.median(np.diff(source_t))
        splits = np.flatnonzero((np.diff(ix) > 1) | (np.diff(source_t[ix]) > max(5 * dt, 0.1))) + 1
        for run in np.split(ix, splits):
            if len(run) < 2:
                continue
            inside = (target_t >= source_t[run[0]]) & (target_t <= source_t[run[-1]])
            result[inside] = np.interp(target_t[inside], source_t[run], values[run])
    return result


@dataclass
class IMUIntegrationBaseline(Step):
    pipeline: str = 'front'

    def run(self, ws):
        accel, target = (ws[key] for key in self.inputs[:2])
        settings = {**DEFAULT_IMU_SETTINGS[self.pipeline], **self.config(ws)}
        settings.pop('enabled', None)
        bad = None
        if len(self.inputs) > 2:
            dropout = ws[self.inputs[2]]
            bad = project_bad_mask(dropout.t, dropout.x, accel.t)
        x = integrate_imu(accel.t, accel.x, bad=bad, **settings)
        # Only target timestamps are used, never its values.
        ws[self.outputs[0]] = TimeSeries(t=target.t, x=resample_valid(accel.t, x, target.t), units='mm', meta=settings)


@dataclass
class MagneticPowerOracle(Step):
    pred_soft_mg: float = 1.

    def run(self, ws):
        mag, reference = ws[self.inputs[0]], ws[self.inputs[1]]
        if not np.array_equal(mag.t, reference.t):
            raise ValueError('Oracle requires identical magnetic/reference timelines')
        valid = np.asarray(ws[self.inputs[2]], dtype=bool).reshape(-1).copy()
        for key in self.inputs[3:]:
            mask = ws[key]
            valid &= ~project_bad_mask(mask.t, mask.x, reference.t, halo_s=.08 if key == "angle/bad_mask" else 0.)
        m, y = mag.x.reshape(-1), reference.x.reshape(-1)
        valid &= np.isfinite(m) & np.isfinite(y)
        if valid.sum() < 4 or np.ptp(m[valid]) == 0:
            raise ValueError('Oracle needs at least four finite training samples with magnetic variation')
        soft = float(self.param(ws, 'pred_soft_mg'))
        coefficients, offset, rmse = fit_supervised_power_oracle(m[valid], y[valid], pred_soft_mg=soft)
        prediction = MagToTravelModel(pred_soft_mg=soft).pred_x(m, coefficients) + offset
        ws[self.outputs[0]] = TimeSeries(t=mag.t, x=prediction, units='mm', meta={'supervised': True, 'in_sample': True})
        ws[self.outputs[1]] = np.r_[coefficients, offset, rmse, valid.sum()]


def evaluation_steps(pipeline, log_config):
    """Optional diagnostics appended after production outputs; no downstream consumers."""
    from mag_to_travel_model_core import MagToTravelModelCore
    from rear_mag_model import RearMagModel
    front = pipeline == 'front'
    steps = [
        IMUIntegrationBaseline(name='imu_integration_baseline', pipeline=pipeline,
            inputs=('accel/proj', 'travel') if front else ('accel/lpf/proj', 'travel'),
            outputs=('travel/baseline/accel',)),
        MagneticPowerOracle(name='magnetic_power_oracle',
            pred_soft_mg=float(MagToTravelModelCore.pred_soft_mg if front else RearMagModel.pred_soft_mg),
            inputs=('mag/norm/corr/lpf' if front else 'mag/angle/lpf', 'travel', 'active_mask',
                    'angle/bad_mask') + (('imu_dropout_mask',) if front else ()),
            outputs=('travel/oracle/mag_power', 'oracle_power_fit')),
    ]
    return [s for s in steps if log_config.get('steps', {}).get(s.name, {}).get('enabled', True)]
