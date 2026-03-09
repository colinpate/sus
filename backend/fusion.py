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
class GetMagBaseline(Step):
    """Get the baseline for still magnetometer data"""
    still_len_s: float = 0.1 # seconds
    still_a_max: float = 0.5 # m/s^2

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        mag_proj = mag_ts.x
        a_proj_lhp = accel_ts.x
        
        still_mags = []
        for i in range(0, mag_proj.shape[0] - still_len, still_len):
            mag_chunk = mag_proj[i:i+still_len]
            a_chunk = a_proj_lhp[i:i+still_len]
            if max(abs(a_chunk)) < self.still_a_max:
                still_mags.append(mag_chunk)

        mag_baseline = np.median(still_mags) + np.std(still_mags)
        print("Mag baseline", mag_baseline, "std", np.std(still_mags))

        ws[self.outputs[0]] = np.array(mag_baseline)

@dataclass
class GetMagToTravelModel(Step):
    """ Train a model using least squares  """
    chunk_min_dx = 10
    chunk_len = 20
    min_mag = 500
    train_with_mask: bool = False

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        travel_ts: TimeSeries = ws[self.inputs[2]]
        mask_ts: np.ndarray = ws[self.inputs[3]]
        idxs: np.ndarray = ws[self.inputs[4]]

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

        self.calculate_rmse(result, mag, travel, mag_proj_bad_mask)

        x_preds = self.pred_x(mag, result.x[0], result.x[1], result.x[2])

        ws[self.outputs[0]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )

        scatter_points = np.array([mag, travel, x_preds]).T
        ws[self.outputs[1]] = scatter_points

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
    
    def calculate_rmse(self, result, mag, travel, mag_proj_bad_mask):
        # Calc RMSE
        for mask_str, bad_mask_i in [
                ["With mask", mag_proj_bad_mask],
                ["Without mask", np.zeros_like(mag_proj_bad_mask, dtype=bool)]
            ]:
            good_mag_mask = ~bad_mask_i
            mags_flat = mag[good_mag_mask].flatten().copy() #input_arr[:, 0, :].flatten()
            x_mag_flat = self.pred_x(mags_flat, result.x[0], result.x[1], result.x[2])
            x_mag_flat -= np.mean(x_mag_flat)

            trav_flat = travel[good_mag_mask].flatten().copy() #np.array(gt_list).flatten()
            thresh = 50
            trav_thresh_mask = trav_flat > thresh
            trav_offset = np.mean(trav_flat)
            trav_flat -= trav_offset

            print_err_stats(x_mag_flat, trav_flat, prefix=f"Mag-predicted x ({mask_str})")
            print_err_stats(x_mag_flat[trav_thresh_mask], trav_flat[trav_thresh_mask], prefix=f"Mag-predicted x ({mask_str}) (> {thresh:.1f} mm)")


class GetLinearMagToTravelModel(Step):
    """Find chunks where we can be pretty sure about travel"""
    bump_mag_min: float = 1000 # mG
    still_a_max: float = 1000 # m/s^2
    bump_dx_min: int = 20

    still_len_s: float = 0.1 # seconds
    bump_len_s: float = 0.3 # seconds
    stride_s: float = 0.05 # seconds
    skips: int = 3 # number of following strides to skip if we find a good one, prevents repeats

    mag_range: float = 2000
    poly_degree: int = 1
    ransac_k: int = 10
    ransac_n_iter: int = 100

    debug = False

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag_baseline: float = ws[self.inputs[2]]
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
        chunk_len = still_len + bump_len
        still_mag_max = mag_baseline
        self.mag_lower_thresh = mag_baseline
        self.mag_upper_thresh = mag_baseline + self.mag_range
        print("Mag bounds", int(self.mag_lower_thresh), int(self.mag_upper_thresh))

        mag_chunks, a_intint_chunks, _ = self.find_chunks(
            accel, 
            mag, 
            dt_s, 
            still_len, 
            bump_len, 
            stride, 
            chunk_len, 
            still_mag_max
        )
        mag_chunks_filt, a_intint_chunks_filt = self.filter_chunks(
            mag_chunks, 
            a_intint_chunks, 
            self.mag_lower_thresh, 
            self.mag_upper_thresh
        )
        model = self.chunkwise_ransac(mag_chunks_filt, a_intint_chunks_filt)

        x_preds = self.predict(model, mag)
        if gt_x_ts is not None:
            error = self.get_error_in_bounds(model, mag, gt_x_ts)
            print(f"Final RMSE in bounds: {error}")

        ws[self.outputs[0]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )
        ws[self.outputs[1]] = np.array([self.mag_lower_thresh, self.mag_upper_thresh])
        ws[self.outputs[2]] = np.concatenate((model.coef_, [model.intercept_]))

    def get_error_in_bounds(self, model, mag, gt_x_ts):
        x_preds = self.predict(model, mag)
        gt_x = gt_x_ts.x[:, 0]
        mag_mask = (mag > self.mag_lower_thresh) * (mag < self.mag_upper_thresh)
        mag_filt = mag[mag_mask]
        preds_filt = x_preds[mag_mask]
        gt_filt = gt_x[mag_mask]
        error = np.mean((gt_filt - preds_filt) ** 2) ** 0.5

        if self.debug:
            plt.figure(figsize=(12,6))
            plt.scatter(mag_filt, gt_filt, alpha=0.1, label="Ground truth")
            plt.scatter(mag_filt, preds_filt, alpha=0.5, label="Predictions")
            plt.xlabel("Magnetometer (mG)")
            plt.ylabel("Integrated travel (mm)")
            plt.grid()
            plt.legend()
            plt.show()

        return error

    def find_chunks(self, accel, mag, dt_s, still_len, bump_len, stride, chunk_len, still_mag_max):
        # Find the chunks
        a_mms = accel * 1000
        still_slice = slice(0, still_len)
        bump_slice = slice(still_len, still_len + bump_len)

        slices = []
        a_intint_chunks = []
        mag_chunks = []
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
        
        print("Calibration chunks:", len(a_intint_chunks), "chunks,", len(a_intint_chunks[0]), "samples per chunk")

        return mag_chunks, a_intint_chunks, slices

    def filter_chunks(self, mag_chunks, a_intint_chunks, mag_lo, mag_hi):
        mag_chunks_filt = []
        aii_chunks_filt = []
        for (mag_chunk, a_intint) in zip(mag_chunks, a_intint_chunks):
            mask_i = (mag_chunk > mag_lo) * (mag_chunk < mag_hi)
            if len(a_intint[mask_i]) > 0:
                aii_chunks_filt.append(a_intint[mask_i])
                mag_chunks_filt.append(mag_chunk[mask_i])
        print("Filtered chunks:", len(aii_chunks_filt), "chunks,", len(aii_chunks_filt[0]), "samples per chunk")
        return mag_chunks_filt, aii_chunks_filt

    def mean_bin_std(self, mag_chunks_filt, a_intint_chunks_filt, num_bins=10):
        x_points = np.concatenate(a_intint_chunks_filt)
        mag_points = np.concatenate(mag_chunks_filt)
        print("Mean bin std input points", x_points.shape[0])

        mag_max = np.percentile(mag_points, 95)
        mag_min = np.percentile(mag_points, 5)
        bin_size = (mag_max - mag_min) / num_bins

        bins = []
        for i in range(num_bins):
            bin_min = mag_min + (bin_size * i)
            bin_max = mag_min + (bin_size * (i + 1))

            mask = (mag_points > bin_min) * (mag_points <= bin_max)
            x_masked = x_points[mask]
            bins.append(x_masked)

        bin_stds = np.asarray([np.std(bin) for bin in bins])
        bin_centers = np.asarray([mag_min + (bin_size * (i + 0.5)) for i in range(num_bins)])
        min_std_bin = bin_centers[np.argmin(bin_stds)]
        print("Min std bin center", int(min_std_bin), "std", np.min(bin_stds), "mean", int(np.mean(bins[np.argmin(bin_stds)])))

        mean_bin_std = np.mean(bin_stds)
        print("Mean bin std", mean_bin_std)
        return mean_bin_std

    def chunkwise_ransac(self, mag_chunks_filt, a_intint_chunks_filt):
        if len(mag_chunks_filt) < self.ransac_k:
            print("Not enough chunks for RANSAC, using all points for fit")
            mag_points = np.concatenate(mag_chunks_filt)
            x_points = np.concatenate(a_intint_chunks_filt)
            poly = PolynomialFeatures(degree=self.poly_degree)
            X_poly = poly.fit_transform(mag_points.reshape(-1, 1))
            model = LinearRegression()
            model.fit(X_poly, x_points)
            return model
        # RANSAC
        mean_bin_std = self.mean_bin_std(mag_chunks_filt, a_intint_chunks_filt)
        outlier_dist = mean_bin_std * 0.5
        print("Outlier distance threshold", outlier_dist)
        print(len(mag_chunks_filt), "chunks to RANSAC on")

        best_model = None
        best_inliers = None
        for i in range(self.ransac_n_iter):
            sampled_chunks = random.sample(list(zip(mag_chunks_filt, a_intint_chunks_filt)), self.ransac_k)
            mag_points_i = np.concatenate([chunk_i[0] for chunk_i in sampled_chunks])
            x_points_i = np.concatenate([chunk_i[1] for chunk_i in sampled_chunks])

            # Train on just the sampled chunks
            poly = PolynomialFeatures(degree=self.poly_degree)
            X_poly = poly.fit_transform(mag_points_i.reshape(-1, 1))
            model = LinearRegression()
            model.fit(X_poly, x_points_i)

            # Find inlier chunks based on distance to model
            n_inliers = 0
            for chunk_i in zip(mag_chunks_filt, a_intint_chunks_filt):
                mag_chunk = chunk_i[0]
                x_chunk = chunk_i[1]
                mag_chunk_poly = poly.fit_transform(mag_chunk.reshape(-1, 1))
                preds_chunk = model.predict(mag_chunk_poly)
                dist = np.sqrt(np.mean((preds_chunk - x_chunk) ** 2))
                if dist < outlier_dist:
                    n_inliers += 1#preds_chunk.shape[0]

            if best_inliers is None or n_inliers > best_inliers:
                best_inliers = n_inliers
                best_model = model
                
        print("Best model inliers:", best_inliers)
        print("Best model coeffs, intercept:", best_model.coef_, best_model.intercept_)
        return best_model

    def predict(self, model, mag):
        poly = PolynomialFeatures(degree=self.poly_degree)
        mag_poly = poly.fit_transform(mag.reshape(-1, 1))
        return model.predict(mag_poly)