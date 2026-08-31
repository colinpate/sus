
import numpy as np
from dataclasses import dataclass

@dataclass
class MagSolverWeights:
    # These numbers are completely made up right now
    mag_update_threshold: float = 1500 # milli-Gauss
    world_rw: float = 100 # milli-Gauss
    body_rw: float = 100 # milli-Gauss per second
    mag_sigma: float = 1000 # milli-Gauss


def get_res(
    # Static inputs
    gyro: np.ndarray,
    mag: np.ndarray,
    travel_pred: np.ndarray,
    w: MagSolverWeights,
    dt: np.ndarray,

    # Solver changeable variables
    body_field: np.ndarray,
    world_field: np.ndarray,
):

    residual = []

    for i in range(gyro.shape[0]):
        # Get the remaining residual
        pred_mag = pred_mag_from_travel(travel_pred[i])
        mag_res = mag[i] - world_field[i] - body_field[i] - pred_mag
        mag_res *= 1 if (np.linalg.norm(pred_mag) < w.mag_update_threshold) else 0
        residual.append(mag_res / w.mag_sigma)

        if i > 0:
            angular_rate = 0.5 * (gyro[i - 1] + gyro[i])
            rotation_vector = np.deg2rad(angular_rate) * dt[i]
            delta_rotation = integrate_rotation(rotation_vector)
            expected_world_field = world_field[i-1] @ delta_rotation
            residual.append((expected_world_field - world_field[i]) / (w.world_rw * np.sqrt(dt[i])))
            residual.append((body_field[i] - body_field[i-1]) / (w.body_rw * np.sqrt(dt[i])))

    return residual