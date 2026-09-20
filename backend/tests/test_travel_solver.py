from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from travel_solver_core import (
    SolverInputs,
    SolverWeights,
    calculate_solver_terms,
    make_initial_state,
    prepare_solver,
)


class SolverInputsTests(unittest.TestCase):
    @staticmethod
    def inputs(bounds: tuple[float, float] | None) -> SolverInputs:
        return SolverInputs(
            time_s=np.arange(3, dtype=float) / 100.0,
            accel_mm_s2=np.zeros(3),
            mag=np.ones(3),
            mag_preds_mm=np.array([-20.0, 50.0, 240.0]),
            mag_zv_points=np.array([], dtype=int),
            mag_baseline=1.0,
            mag_prediction_bounds=bounds,
        )

    def test_unbounded_predictions_preserve_relative_coordinate(self):
        np.testing.assert_array_equal(
            self.inputs(None).mag_preds_mm,
            np.array([-20.0, 50.0, 240.0]),
        )

    def test_physical_solver_can_bound_predictions(self):
        np.testing.assert_array_equal(
            self.inputs((0.0, 200.0)).mag_preds_mm,
            np.array([0.0, 50.0, 200.0]),
        )


class TrapezoidalDynamicsTests(unittest.TestCase):
    @staticmethod
    def inputs() -> SolverInputs:
        return SolverInputs(
            time_s=np.array([0.0, 0.1, 0.3]),
            accel_mm_s2=np.array([0.0, 2.0, 4.0]),
            mag=None,
            mag_preds_mm=np.zeros(3),
            mag_zv_points=np.array([], dtype=int),
            mag_baseline=None,
        )

    def test_initial_velocity_uses_interval_average_acceleration(self):
        state = make_initial_state(self.inputs())

        np.testing.assert_allclose(state[3:6], np.array([0.0, 0.1, 0.7]))

    def test_dynamics_are_exact_for_trapezoidally_integrated_state(self):
        inputs = self.inputs()
        prepared = prepare_solver(inputs, SolverWeights())
        state = np.zeros(7)
        state[:3] = np.array([3.0, 3.005, 3.085])
        state[3:6] = np.array([0.0, 0.1, 0.7])

        terms = calculate_solver_terms(state, prepared)

        np.testing.assert_allclose(terms.v_res, np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(terms.x_res, np.zeros(2), atol=1e-12)

    def test_position_residual_depends_on_both_endpoint_velocities(self):
        sparsity = prepare_solver(self.inputs(), SolverWeights()).jac_sparsity

        # The second residual in each five-row interval block is x dynamics.
        self.assertTrue(sparsity[1, 3])
        self.assertTrue(sparsity[1, 4])


if __name__ == "__main__":
    unittest.main()
