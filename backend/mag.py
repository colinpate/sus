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