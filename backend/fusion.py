from dataclasses import dataclass
import random
from unittest import result

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
        x = x - np.mean(x)
        gt = gt - np.mean(gt)
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
    chunk_len = 20
    min_mag = 500
    train_with_mask: bool = False
    apply_ref_point: bool = True

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        travel_ts: TimeSeries = ws[self.inputs[2]]
        mask_ts: np.ndarray = ws[self.inputs[3]]
        idxs: np.ndarray = ws[self.inputs[4]]
        ref_point: np.ndarray = ws[self.inputs[5]]

        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        travel = travel_ts.x[:, 0]
        mag_proj_bad_mask = mask_ts.x.flatten().astype(bool)
        t = mag_ts.t
        dt_s = np.diff(t, prepend=t[0]-0.01)

        if self.train_with_mask:
            training_mask = mag_proj_bad_mask
        else:
            training_mask = np.zeros(mag_ts.x.shape[0], dtype=bool)

        xs, mags = self.get_chunks(idxs, mag, accel, dt_s, training_mask)
        input_arr = self.format_chunks_for_fit(xs, mags)
        result = self.least_squares_fit(input_arr)
        print("x0, y_scale, power:", result.x)

        x_preds = self.pred_x(mag, result.x[0], result.x[1], result.x[2])

        if self.apply_ref_point:
            x_preds_adj = self.adjust_with_ref_point(x_preds, ref_point[0], ref_point[1], result.x)
        else:
            x_preds_adj = x_preds

        self.calculate_rmse(x_preds, travel, x_preds_adj, thresh=0)

        ws[self.outputs[0]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds_adj,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )

        scatter_points = np.array([mag, travel, x_preds_adj]).T
        ws[self.outputs[1]] = scatter_points

    def adjust_with_ref_point(self, x_preds, ref_x, ref_mag, coeffs):
        x0, y_scale, power = coeffs
        ref_x_pred = self.pred_x(ref_mag, x0, y_scale, power)
        offset = - ref_x_pred + ref_x
        print("Adjusting predicted x by offset", offset, "to align reference point")
        x_preds_ref = x_preds + offset
        return x_preds_ref

    def get_chunks(self, idxs_filt, mag, acc, dt_s, mag_proj_bad_mask):
        chunk_len = self.chunk_len
        min_dx = self.chunk_min_dx

        xs = []
        mags = []
        for idx in idxs_filt:
            if idx < chunk_len or idx + chunk_len >= len(mag):
                continue
            dt_chunk = dt_s[idx - chunk_len:idx + chunk_len]
            a_chunk = acc[idx - chunk_len:idx + chunk_len] * 1000
            badmask_chunk = mag_proj_bad_mask[idx - chunk_len:idx + chunk_len]
            if np.mean(badmask_chunk) > 0.1:
                continue
            v_chunk = np.cumsum(a_chunk * dt_chunk)
            x_chunk = np.cumsum((v_chunk - v_chunk[chunk_len]) * dt_chunk)
            x_chunk -= x_chunk[chunk_len]
            if max(x_chunk) - min(x_chunk) < min_dx:
                continue
            mag_chunk = mag[idx - chunk_len:idx + chunk_len]
            xs.append(x_chunk)
            mags.append(mag_chunk)

        print(len(xs))

        return xs, mags

    def format_chunks_for_fit(self, xs, mags):
        # Formulate input data and residuals and threshold by min mag
        # Axis 0: chunk: n_chunks
        # Axis 1: point index: n_points
        # Axis 2: mag (absolute), x (relative to point at index 0): 
        chunk_len = self.chunk_len

        pt_idxes = [chunk_len] + list(range(0, chunk_len)) + list(range(chunk_len + 1, 2 * chunk_len))

        input_list = []
        for i, (x_i, mag_i) in enumerate(zip(xs, mags)):
            if np.count_nonzero(mag_i[pt_idxes] < self.min_mag) == 0:
                input_list.append([mag_i[pt_idxes], x_i[pt_idxes]])

        input_arr = np.array(input_list)
        print(input_arr.shape)
        return input_arr
    
    def pred_x(self, mag_i, x0, y_scale, power):
        mag_i = np.copy(mag_i)
        mag_i[(mag_i - x0) < 0] = x0
        return ((mag_i - x0) ** power) * y_scale
    
    def least_squares_fit(self, input_arr, power_prior = 1/3, power_weight = 1000):

        def calculate_res(vec):
            x0, y_scale, power = vec[0], vec[1], vec[2]

            zero_x_mags = input_arr[:, 0, 0]
            zero_x_preds = self.pred_x(zero_x_mags, x0, y_scale, power=power)
            x_acc_preds = input_arr[:, 1, 1:] + zero_x_preds[:, np.newaxis]

            mag_pts = input_arr[:, 0, 1:]
            x_mag_preds = self.pred_x(mag_pts, x0, y_scale, power=power)
            res = x_acc_preds - x_mag_preds

            power_res = power - power_prior

            return np.concatenate([res.flatten(), np.array([power_res]) * power_weight])

        guess_vec = [0, 1/3, 1/3]
        result = least_squares(
                fun=calculate_res,
                x0=guess_vec, 
                method="trf",
                verbose=1,
                max_nfev=1000,
                #loss='huber',
            )
        
        return result
    
    def calculate_rmse(self, preds, travel, preds_adj, thresh):
        # Calc RMSE
        trav_thresh_mask = travel > thresh

        print_err_stats(preds, travel, prefix=f"Mag-predicted x")
        print_err_stats(preds[trav_thresh_mask], travel[trav_thresh_mask], prefix=f"Mag-predicted x (> {thresh:.1f} mm)")
        print_err_stats(preds[trav_thresh_mask], travel[trav_thresh_mask], prefix=f"Mag-predicted x (> {thresh:.1f} mm) (centered)", center=True)
        print_err_stats(preds_adj[trav_thresh_mask], travel[trav_thresh_mask], prefix=f"Mag-predicted x adjusted with reference point (> {thresh:.1f} mm)")

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
    min_ref_mag: float = 1500

    debug: bool = False

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        gt_x_ts: TimeSeries | None = ws.get(self.inputs[2])
        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        t = mag_ts.t
        dt_s = np.diff(t, prepend=t[0]-0.01)

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        bump_len = int(self.bump_len_s * mag_ts.meta["fs_hz"])
        stride = int(self.stride_s * mag_ts.meta["fs_hz"])
        
        mag_baseline = self.get_mag_baseline(mag, accel, still_len)

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
        
        print("Calibration chunks:", len(a_intint_chunks), "chunks,", len(a_intint_chunks[0]), "samples per chunk")

        return mag_chunks, a_intint_chunks, slices, gt_x_chunks

    def get_abs_pos_ref(self, mag_chunks, a_intint_chunks, mag_baseline, gt_x_chunks=None):
        x_points = np.concatenate(a_intint_chunks)
        mag_points = np.concatenate(mag_chunks)
        print("Absolute position reference input points", x_points.shape[0])

        mag_center = max(mag_baseline + self.min_ref_mag, np.median(mag_points))
        center_range = self.ref_mag_range / 2
        thresh_mask = (mag_points > mag_center - center_range) & (mag_points < mag_center + center_range)
        print(f"Using {np.sum(thresh_mask)} points within mag range {mag_center - center_range} to {mag_center + center_range} for absolute position reference stats")
        abs_pos_ref_x = np.median(x_points[thresh_mask])
        abs_pos_ref_mag = np.median(mag_points[thresh_mask])

        if gt_x_chunks is not None:
            gt_x_points = np.concatenate(gt_x_chunks)
            abs_pos_ref_error = abs_pos_ref_x - np.median(gt_x_points[thresh_mask])
            print(f"Absolute position reference error compared to GT: {abs_pos_ref_error:.1f} mm")

        return abs_pos_ref_x, abs_pos_ref_mag
    
    def get_mag_baseline(self, mag, accel, still_len):
        a_mms = accel * 1000
        still_mags = []
        for i in range(0, mag.shape[0] - still_len, still_len):
            mag_chunk = mag[i:i+still_len]
            a_chunk = a_mms[i:i+still_len]
            if max(abs(a_chunk)) < self.still_a_max:
                still_mags.append(mag_chunk)

        mag_baseline = np.median(still_mags) + np.std(still_mags)
        print("Mag baseline", mag_baseline, "std", np.std(still_mags))
        return mag_baseline