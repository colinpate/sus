from dataclasses import dataclass

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step

@dataclass
class ProjectMag(Step):
    """Project magnet data onto mean vector"""
    mag_threshold: int = 3000  # mG

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]

        x = a.x
        
        # Threshold magnet data and project along the direction of travel
        mag_filtered_thresh = x[np.linalg.norm(x, axis=1) > self.mag_threshold]
        mean_vector = np.mean(mag_filtered_thresh, axis=0)
        # Project mag data onto mean vector to get travel along that direction
        mag_travel_vector = mean_vector / np.linalg.norm(mean_vector)
        print("Primary vector of magnet:", mag_travel_vector)
        mag_proj = x @ mag_travel_vector

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=mag_proj,
            units=a.units,
            frame=a.frame,
            meta={**a.meta},
        )


class FindBadMagProj(Step): 
    raw_norm_maxdiff: int = 2000  # mG

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        b: TimeSeries = ws[self.inputs[1]]

        x = a.x
        mag_proj = b.x.flatten()

        # Check that raw mag norm is not too different from projected mag to filter out bad data
        mag_raw_norm = np.linalg.norm(x, axis=1)
        norm_diff = np.abs(mag_raw_norm - mag_proj)
        bad_data_mask = norm_diff > self.raw_norm_maxdiff
        print(f"{np.mean(bad_data_mask)*100:.1f}% of magnet data points have raw norm differing from projected by more than {self.raw_norm_maxdiff} mG. ")

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=bad_data_mask.astype(float),  # 1 for bad data points, 0 for good
            units=a.units,
            frame=a.frame,
            meta={**a.meta},
        )


class CorrectBadMagProj(Step): 
    raw_norm_maxdiff: int = 2000  # mG

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]] # Raw mag data
        b: TimeSeries = ws[self.inputs[1]]

        mag_raw = a.x
        mag_proj = b.x.flatten()

        # Check that raw mag norm is not too different from projected mag to filter out bad data
        mag_raw_norm = np.linalg.norm(mag_raw, axis=1)
        norm_diff = np.abs(mag_raw_norm - mag_proj)
        bad_data_mask = norm_diff > self.raw_norm_maxdiff
        print(f"{np.mean(bad_data_mask)*100:.1f}% of magnet data points have raw norm differing from projected by more than {self.raw_norm_maxdiff} mG. ")

        corrected_mag = mag_proj.copy()
        corrected_mag[bad_data_mask] = mag_raw_norm[bad_data_mask]

        ws[self.outputs[0]] = TimeSeries(
                    t=a.t,
                    x=corrected_mag,
                    units=a.units,
                    frame=a.frame,
                    meta={**a.meta},
                )
        ws[self.outputs[1]] = TimeSeries(
            t=a.t,
            x=bad_data_mask.astype(float),  # 1 for bad data points, 0 for good
            units=a.units,
            frame=a.frame,
            meta={**a.meta},
        )


class FindMagZVPoints(Step):
    """Find zero-velocity points in magnetometer data"""
    min_dt = 5
    min_dm = 50

    def run(self, ws: Workspace) -> None:
        mag_proj_series: TimeSeries = ws[self.inputs[0]]
        mag_proj = mag_proj_series.x.flatten()

        local_max_indices = (np.diff(np.sign(np.diff(mag_proj))) < 0).nonzero()[0] + 1
        local_min_indices = (np.diff(np.sign(np.diff(mag_proj))) > 0).nonzero()[0] + 1
        v0_idxs = np.sort(np.concatenate((local_max_indices, local_min_indices)))

        idxs_filt = []
        for i in range(1, v0_idxs.shape[0] - 1):
            idx = v0_idxs[i]
            near_idxs = [v0_idxs[i-1], v0_idxs[i+1]]
            mag_i = mag_proj[v0_idxs[i]]
            mag_near = mag_proj[near_idxs]
            min_dm_i = min(abs(mag_near - mag_i))
            min_dt_i = min(abs(near_idxs - idx))
            if min_dt_i < self.min_dt:
                continue
            if min_dm_i < self.min_dm:
                continue
            idxs_filt.append(idx)

        idx_arr = np.array(idxs_filt)
        print("Found", idx_arr.shape[0], "ZV points in magnetometer data")
        ws[self.outputs[0]] = idx_arr


@dataclass
class MagToTravelPolyFit(Step):
    """Project magnet data onto mean vector"""
    mag_threshold: int = 1500  # mG

    def run(self, ws: Workspace) -> None:
        mag_proj_series: TimeSeries = ws[self.inputs[0]]
        travel_series: TimeSeries = ws[self.inputs[1]]

        mag_proj = mag_proj_series.x.flatten()
        travel = travel_series.x

        # Fit polynomial to get travel from projected mag

        def get_xy_by_xmin(x, y, xmin):
            x_mask = x > xmin
            xmasked = x[x_mask]
            ymasked = y[x_mask]
            return xmasked, ymasked

        mag_m, travel_m = get_xy_by_xmin(mag_proj, travel, self.mag_threshold)

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(mag_m.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, travel_m)
        mag_poly = poly.fit_transform(mag_proj.reshape(-1, 1))
        travel_pred = model.predict(mag_poly)

        print("Magnet-to-travel polynomial fit coeffs, intercept:", model.coef_, model.intercept_)

        error = np.mean((travel - travel_pred) ** 2) ** 0.5
        print(f"Magnet-to-travel polynomial fit RMSE: {error}")

        for mask_thresh in [1000, 2000, 3000]:
            min_mag_mask = mag_proj.flatten() > mask_thresh
            masked_travel = travel[min_mag_mask]
            masked_travel_pred = travel_pred[min_mag_mask]
            error_masked = np.mean((masked_travel - masked_travel_pred) ** 2) ** 0.5
            error_mean = np.mean(abs(masked_travel - masked_travel_pred))
            print(f"RMSE (points over {mask_thresh}): {error_masked}, Mean Abs Error: {error_mean}")

        ws[self.outputs[0]] = TimeSeries(
            t=mag_proj_series.t,
            x=travel_pred,
            units=travel_series.units,
            frame=travel_series.frame,
            meta={**mag_proj_series.meta, "polyfit_applied": True},
        )
        travel_vs_mag = np.concat((travel.reshape(-1, 1), mag_proj.reshape(-1, 1)), axis=1)
        travel_vs_pred = np.concat((travel.reshape(-1, 1), travel_pred.reshape(-1, 1)), axis=1)
        ws[self.outputs[1]] = travel_vs_mag
        ws[self.outputs[2]] = travel_vs_pred