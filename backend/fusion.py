from dataclasses import dataclass
import random
from unittest import result

import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step

import matplotlib.pyplot as plt

def print_err_stats(x, gt, center=False, prefix=""):
    if center:
        x = x.copy() - np.mean(x)
        gt = gt.copy() - np.mean(gt)
    error = x - gt
    rmse = np.mean(error ** 2) ** 0.5
    mae = np.mean(abs(error))
    me = np.mean(error)
    print(f"{prefix} RMSE: {rmse:.3f}, MAE: {mae:.3f}, ME: {me:.3f}")
    return rmse, mae, me


@dataclass
class GetMagToTravelModel(Step):
    """ Train a model using least squares  """
    chunk_min_dx = 10
    chunk_max_dx = 1500
    chunk_len = 20
    fit_balance_bins = 8
    fit_balance_mode = "center_mag"
    train_with_mask: bool = False
    apply_ref_point: bool = True
    bad_thresh: float = 0.5
    pred_soft_mg: float = 50.0
    ref_zero_percentile: float = 8.0
    ref_neg_fallback_max_pct: float = 0.1
    ref_fallback_accel_quantile: float = 70.0
    power_weight: float = 1000.0

    # For re-calculating mag baseline, if desired
    still_len_s: float = 0.1
    still_a_max: float = 1000
    min_mag_relaxed_still_std_scale: float = 0.5
    min_mag_relax_min_chunks: int = 50

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        travel_ts: TimeSeries = ws[self.inputs[2]]
        mask_ts: np.ndarray = ws[self.inputs[3]]
        idxs: np.ndarray = ws[self.inputs[4]]
        ref_point: np.ndarray = ws[self.inputs[5]]
        mag_baseline: float = ws[self.inputs[6]]

        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        travel = travel_ts.x[:, 0]
        mag_proj_bad_mask = mask_ts.x.flatten().astype(bool)
        t = mag_ts.t

        baseline_min_mag = mag_baseline[0]

        if self.train_with_mask:
            print("Trainign with mask, shape of bad mask", mag_proj_bad_mask.shape, "num bad samples", np.sum(mag_proj_bad_mask))
            training_mask = mag_proj_bad_mask
        else:
            training_mask = np.zeros(mag_ts.x.shape[0], dtype=bool)

        self.min_mag = baseline_min_mag
        xs, mags, all_mags = self.get_chunks(idxs, mag, accel, t, training_mask, self.min_mag)
        mag_mins = [np.min(mag_chunk) for mag_chunk in all_mags]
        relaxed_min_mag = np.sort(mag_mins)[-self.min_mag_relax_min_chunks]

        use_relaxed_min_mag = (
            np.isfinite(relaxed_min_mag)
            and len(xs) < self.min_mag_relax_min_chunks
            and relaxed_min_mag < baseline_min_mag
        )
        if use_relaxed_min_mag:
            print(
                "Relaxing min mag from",
                baseline_min_mag,
                "to",
                relaxed_min_mag,
                "initial chunks",
                len(xs),
            )
            xs, mags, _ = self.get_chunks(idxs, mag, accel, t, training_mask, relaxed_min_mag)
        else:
            print(
                "Using raw min mag",
                baseline_min_mag,
                "chunks",
                len(xs),
            )

        input_arr = self.format_chunks_for_fit(xs, mags)
        result = self.least_squares_fit(input_arr, power_weight=self.power_weight)
        print("x0, y_scale, power:", result.x)

        x_preds = self.pred_x(mag, result.x[0], result.x[1], result.x[2])

        if self.apply_ref_point:
            ref_fallback_mask = self.build_ref_fallback_mask(accel, mag_proj_bad_mask)
            x_preds_adj = self.adjust_with_ref_point(x_preds, ref_point[0], ref_point[1], result.x, mag, ref_fallback_mask)
        else:
            x_preds_adj = x_preds

        x_preds_adj = np.clip(x_preds_adj, 0, None)
        x_preds = np.clip(x_preds, 0, None)
        ws[self.outputs[0]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )
        ws[self.outputs[1]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds_adj,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )
        scatter_points = np.array([mag, travel, x_preds_adj]).T
        ws[self.outputs[2]] = scatter_points
        ws[self.outputs[3]] = np.array([result.x[0], result.x[1], result.x[2]])

    def build_ref_fallback_mask(self, accel: np.ndarray, mag_proj_bad_mask: np.ndarray) -> np.ndarray:
        accel = np.asarray(accel, dtype=float).reshape(-1)
        mag_proj_bad_mask = np.asarray(mag_proj_bad_mask, dtype=bool).reshape(-1)
        accel_abs = np.abs(accel)
        finite_mask = np.isfinite(accel_abs)
        candidate_mask = finite_mask & ~mag_proj_bad_mask
        if not np.any(candidate_mask):
            return np.zeros_like(accel_abs, dtype=bool)

        accel_thresh = float(np.percentile(accel_abs[candidate_mask], self.ref_fallback_accel_quantile))
        motion_mask = candidate_mask & (accel_abs > accel_thresh)
        if not np.any(motion_mask):
            return np.zeros_like(accel_abs, dtype=bool)
        return motion_mask

    def adjust_with_ref_point(self, x_preds, ref_x, ref_mag, coeffs, mag=None, active_mask=None):
        x0, y_scale, power = coeffs
        ref_x_pred = self.pred_x(ref_mag, x0, y_scale, power)
        offset = - ref_x_pred + ref_x
        x_preds_ref = x_preds + offset

        if mag is not None and active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if np.any(active_mask):
                neg_pct = float(np.mean(x_preds_ref[active_mask] < 0))
                print(
                    "Ref-point fallback check: {:.1f}% of motion-mask samples have negative predicted travel".format(
                        neg_pct * 100
                    )
                )
                if neg_pct > self.ref_neg_fallback_max_pct:
                    zero_mag = float(np.percentile(mag, self.ref_zero_percentile))
                    zero_offset = -float(self.pred_x(zero_mag, x0, y_scale, power))
                    if zero_offset > offset:
                        print(
                            f"Ref-point fallback: neg_pct={neg_pct * 100:.1f}% exceeds {self.ref_neg_fallback_max_pct * 100:.1f}%, "
                            f"switching offset from {offset:.1f} to {zero_offset:.1f} using mag p{self.ref_zero_percentile:.0f}={zero_mag:.1f}"
                        )
                        offset = zero_offset
                        x_preds_ref = x_preds + offset

        return x_preds_ref

    def get_chunks(self, idxs_filt, mag, acc, t_s, mag_proj_bad_mask, min_mag):
        chunk_len = self.chunk_len
        min_dx = self.chunk_min_dx
        max_dx = self.chunk_max_dx
        print("Min mag:", min_mag)

        xs = []
        mags = []
        all_mags = []
        for idx in idxs_filt:
            if idx < chunk_len or idx + chunk_len >= len(mag):
                continue
            t_chunk = t_s[idx - chunk_len:idx + chunk_len]
            a_chunk = acc[idx - chunk_len:idx + chunk_len] * 1000
            badmask_chunk = mag_proj_bad_mask[idx - chunk_len:idx + chunk_len]
            if np.mean(badmask_chunk) > self.bad_thresh:
                continue
            v_chunk = scipy.integrate.cumulative_trapezoid(a_chunk, t_chunk, initial=0)
            v_chunk -= v_chunk[chunk_len]
            x_chunk = scipy.integrate.cumulative_trapezoid(v_chunk, t_chunk, initial=0)
            x_chunk -= x_chunk[chunk_len]
            chunk_dx = max(x_chunk) - min(x_chunk)
            if chunk_dx < min_dx or chunk_dx > max_dx:
                continue
            mag_chunk = mag[idx - chunk_len:idx + chunk_len]
            dm_chunk = np.diff(mag_chunk, prepend=mag_chunk[0])
            dm_dx = dm_chunk / (v_chunk + 1e-6)
            if np.median(dm_dx) < 0.05:
                continue
            all_mags.append(mag_chunk)
            if min(mag_chunk) < min_mag:
                continue
            xs.append(x_chunk)
            mags.append(mag_chunk)

        print("Xs:", len(xs))

        return xs, mags, all_mags

    def format_chunks_for_fit(self, xs, mags):
        # Formulate input data and residuals and threshold by min mag
        # Axis 0: chunk: n_chunks
        # Axis 1: point index: n_points
        # Axis 2: mag (absolute), x (relative to point at index 0): 
        chunk_len = self.chunk_len

        pt_idxes = [chunk_len] + list(range(0, chunk_len)) + list(range(chunk_len + 1, 2 * chunk_len))

        input_list = []
        for i, (x_i, mag_i) in enumerate(zip(xs, mags)):
            input_list.append([mag_i[pt_idxes], x_i[pt_idxes]])

        input_arr = np.array(input_list)
        print("Min mag at indices:", np.min(input_arr[:, 0, :]), "mean", np.mean(input_arr[:, 0, :]), "max", np.max(input_arr[:, 0, :]))
        print(input_arr.shape)
        return input_arr
    
    def pred_x(self, mag_i, x0, y_scale, power):
        dx = np.asarray(mag_i, dtype=float) - x0
        soft = (np.abs(dx) + self.pred_soft_mg) ** power - (self.pred_soft_mg ** power)
        return np.sign(dx) * soft * y_scale

    def get_fit_chunk_weights(self, input_arr):
        if input_arr.shape[0] == 0:
            return np.array([])

        if self.fit_balance_mode == "max_mag":
            rep_mag = np.max(input_arr[:, 0, :], axis=1)
        elif self.fit_balance_mode == "mean_mag":
            rep_mag = np.mean(input_arr[:, 0, :], axis=1)
        else:
            rep_mag = input_arr[:, 0, 0]

        rep_mag = np.asarray(rep_mag, dtype=float)
        n_bins = int(np.clip(self.fit_balance_bins, 1, len(rep_mag)))
        if n_bins <= 1 or np.allclose(rep_mag, rep_mag[0]):
            return np.ones_like(rep_mag)

        edges = np.linspace(np.min(rep_mag), np.max(rep_mag), n_bins + 1)
        bin_idx = np.digitize(rep_mag, edges[1:-1], right=False)
        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        weights = 1.0 / np.maximum(counts[bin_idx], 100)

        # Normalize so the average chunk keeps about unit weight.
        weights *= len(weights) / np.sum(weights)

        print(
            "Balanced fit chunk counts by mag bin:",
            counts.astype(int),
            "rep mag percentiles:",
            np.percentile(rep_mag, [0, 25, 50, 75, 100]),
            "edges:",
            edges,
        )
        return weights
    
    def least_squares_fit(self, input_arr, power_prior = 1/3, power_weight = 1000):
        #chunk_weights = self.get_fit_chunk_weights(input_arr)

        def calculate_res(vec):
            x0, y_scale, power = vec[0], vec[1], vec[2]

            zero_x_mags = input_arr[:, 0, 0]
            zero_x_preds = self.pred_x(zero_x_mags, x0, y_scale, power=power)
            x_acc_preds = input_arr[:, 1, 1:] + zero_x_preds[:, np.newaxis]

            mag_pts = input_arr[:, 0, 1:]
            x_mag_preds = self.pred_x(mag_pts, x0, y_scale, power=power)
            res = x_acc_preds - x_mag_preds
            #res *= np.sqrt(chunk_weights)[:, np.newaxis]

            power_res = power - power_prior

            return np.concatenate([res.flatten(), np.array([power_res]) * power_weight])

        guess_vec = [self.min_mag, 3, 1/3]
        result = least_squares(
                fun=calculate_res,
                x0=guess_vec, 
                method="trf",
                verbose=1,
                max_nfev=1000,
                #loss='huber',
            )
        
        return result
    
@dataclass
class GetErrorStats(Step):
    """ Get error stats for mag to travel model """
    gt_thresh: float = 0

    def run(self, ws: Workspace) -> None:
        preds_ts: TimeSeries = ws[self.inputs[0]]
        gt_ts: TimeSeries = ws[self.inputs[1]]
        mask_in: np.ndarray | None = None
        if len(self.inputs) > 2:
            mask_in = ws[self.inputs[2]]

        preds = preds_ts.x[:, 0]
        gt = gt_ts.x[:, 0]

        mask = gt > self.gt_thresh
        if mask_in is not None:
            mask *= mask_in.flatten()
            print(f"Calculating error stats with mask, using {np.sum(mask_in)/len(mask)*100:.1f}% samples")
        preds_masked = preds[mask]
        gt_masked = gt[mask]

        print_err_stats(preds_masked, gt_masked, prefix=f"Thresh (> {self.gt_thresh:.1f} mm) (centered)", center=True)
        print_err_stats(preds_masked, gt_masked, prefix=f"Thresh (> {self.gt_thresh:.1f} mm)")
    

@dataclass
class GetMagTravelRefPoint(Step):
    """Find chunks where we can be pretty sure about travel and use this to set up a static mag to travel reference point"""
    bump_mag_min: float = 1000 # mG
    still_a_max: float = 1000 # mm/s^2
    bump_dx_min: int = 20

    still_len_s: float = 0.1 # seconds
    bump_len_s: float = 0.3 # seconds
    stride_s: float = 0.05 # seconds
    skips: int = 3 # number of following strides to skip if we find a good one, prevents repeats

    ref_mag_range: float = 2000
    min_ref_mag: float = 2000

    debug: bool = False

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag_baseline: float = ws[self.inputs[2]][0]
        gt_x_ts: TimeSeries | None = ws.get(self.inputs[3])
        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        t = mag_ts.t
        dt_s = np.diff(t, prepend=t[0]-0.01)

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        bump_len = int(self.bump_len_s * mag_ts.meta["fs_hz"])
        stride = int(self.stride_s * mag_ts.meta["fs_hz"])

        mag_chunks, a_intint_chunks, _, gt_x_chunks = self.find_chunks(
            accel, 
            mag, 
            gt_x_ts.x if gt_x_ts is not None else None,
            dt_s, 
            still_len, 
            bump_len, 
            stride, 
            mag_baseline
        )
        if len(mag_chunks):
            mag_maxes = [np.max(mag_chunk) for mag_chunk in mag_chunks]
            print("Max mags in chunks:", np.percentile(mag_maxes, 25), np.percentile(mag_maxes, 50), np.percentile(mag_maxes, 75))
        abs_pos_ref_x, abs_pos_ref_mag = self.get_abs_pos_ref(mag_chunks, a_intint_chunks, mag_baseline, gt_x_chunks)
        print(f"Absolute position reference point: x={abs_pos_ref_x:.1f} mm, mag={abs_pos_ref_mag:.1f} mG")

        ws[self.outputs[0]] = np.array([abs_pos_ref_x, abs_pos_ref_mag])

    def find_chunks(self, accel, mag, gt_x, dt_s, still_len, bump_len, stride, still_mag_max):
        # Find the chunks
        a_mms = accel * 1000
        still_slice = slice(0, still_len)
        bump_slice = slice(still_len, still_len + bump_len)
        chunk_len = still_len + bump_len

        slices = []
        a_intint_chunks = []
        mag_chunks = []
        if gt_x is not None:
            gt_x_chunks = []
        else:
            gt_x_chunks = None
        i = 0
        skip = 0
        for i in range(0, a_mms.shape[0] - chunk_len, stride):
            if skip > 0:
                skip -= 1
                continue

            chunk_r = slice(i, i+chunk_len)
            chunk_l = slice(i+chunk_len, i, -1)
            chunks = [chunk_r, chunk_l]
            
            for chunk_i in chunks: 
                mag_still = mag[chunk_i][still_slice]
                a_still = a_mms[chunk_i][still_slice]

                a_bump = a_mms[chunk_i][bump_slice]
                dt_bump = dt_s[chunk_i][bump_slice]
                mag_bump = mag[chunk_i][bump_slice]

                mag_still_mean = np.mean(mag_still)

                if np.mean(mag_still) > still_mag_max:
                    continue
                if max(abs(a_still)) > self.still_a_max:
                    continue
                if max(mag_bump) < mag_still_mean + self.bump_mag_min:
                    continue
                
                a_int = np.cumsum(a_bump * dt_bump)
                a_intint = np.cumsum(a_int * dt_bump)

                if max(a_intint) < self.bump_dx_min:
                    continue

                skip = self.skips

                a_intint_chunks.append(a_intint)
                mag_chunks.append(mag_bump)
                slices.append(chunk_i)
                if gt_x is not None:
                    gt_x_chunks.append(gt_x[chunk_i][bump_slice])
        
        if len(a_intint_chunks) == 0:
            print("No chunks found")
        else:
            print("Calibration chunks:", len(a_intint_chunks), "chunks,", len(a_intint_chunks[0]), "samples per chunk")

        return mag_chunks, a_intint_chunks, slices, gt_x_chunks

    def get_abs_pos_ref(self, mag_chunks, a_intint_chunks, mag_baseline, gt_x_chunks=None):
        if len(mag_chunks) == 0:
            print("No calibration chunks found, cannot determine absolute position reference point, returning 0 and mag baseline + min ref mag")
            return 0, mag_baseline + self.min_ref_mag
        x_points = np.concatenate(a_intint_chunks)
        mag_points = np.concatenate(mag_chunks)
        print("Absolute position reference input points", x_points.shape[0])

        mag_center = max(mag_baseline + self.min_ref_mag, np.median(mag_points))
        center_range = self.ref_mag_range / 2
        thresh_mask = (mag_points > mag_center - center_range) & (mag_points < mag_center + center_range)
        print(f"Using {np.sum(thresh_mask)} points within mag range {mag_center - center_range} to {mag_center + center_range} for absolute position reference stats")
        abs_pos_ref_x = np.median(x_points[thresh_mask])
        abs_pos_ref_mag = np.median(mag_points[thresh_mask])
        print(f"X STD: {np.std(x_points[thresh_mask]):.1f} Mag STD: {np.std(mag_points[thresh_mask]):.1f}")

        if gt_x_chunks is not None:
            gt_x_points = np.concatenate(gt_x_chunks)
            abs_pos_ref_error = abs_pos_ref_x - np.median(gt_x_points[thresh_mask])
            #self.plot_points(x_points, mag_points, gt_x_points)
            print(f"Absolute position reference error compared to GT: {abs_pos_ref_error:.1f} mm")

        return abs_pos_ref_x, abs_pos_ref_mag
    
    def plot_points(self, x_points, mag_points, gt_x_points=None):
        plt.figure(figsize=(10, 6))
        if gt_x_points is not None:
            plt.scatter(mag_points, gt_x_points, alpha=0.5, label="GT x points")
        plt.scatter(mag_points, x_points, alpha=0.5, label="Calibration points")
        plt.xlabel("Mag (mG)")
        plt.ylabel("Integrated accel (mm)")
        plt.title("Calibration points for absolute position reference")
        plt.legend()
        plt.grid()
        plt.show()


class GetMagBaseline(Step):
    """Find the mag baseline by looking at still regions and taking the median + std"""
    still_len_s: float = 0.1 # seconds
    still_a_max: float = 1000 # mm/s^2

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        a_mms = accel * 1000
        still_mags = []
        for i in range(0, mag.shape[0] - still_len, still_len):
            mag_chunk = mag[i:i+still_len]
            a_chunk = a_mms[i:i+still_len]
            if max(abs(a_chunk)) < self.still_a_max:
                still_mags.append(mag_chunk)

        mag_baseline = np.median(still_mags) + np.std(still_mags)
        print("Mag baseline", mag_baseline, "std", np.std(still_mags))
        ws[self.outputs[0]] = np.array([mag_baseline])
