from dataclasses import dataclass

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step
from travel_solver_core import SolverInputs, solve_travel, solver_weights_for_mag_baseline


@dataclass
class TravelSolver(Step):
    """ Solve for travel using multiple inputs and a simple physical model """
    max_nfev: int = 100
    verbose: int = 1

    def run(self, ws: Workspace) -> None:
        inputs = self.solver_inputs(ws)
        weights = solver_weights_for_mag_baseline(inputs.mag_baseline)
        result = solve_travel(inputs, weights, max_nfev=self.max_nfev, verbose=self.verbose)

        ws[self.outputs[0]] = TimeSeries(
            t=inputs.dt_s.cumsum(),
            x=result.x,
            units="mm",
            frame=ws[self.inputs[0]].frame,
            meta={**ws[self.inputs[0]].meta},
        )

    def solver_inputs(self, ws: Workspace) -> SolverInputs:
        accel_ts: TimeSeries = ws[self.inputs[0]]
        mag_baseline: float = ws[self.inputs[4]][0]
        return SolverInputs(
            time_s=accel_ts.t,
            accel_mm_s2=accel_ts.x[:, 0] * 1000.0,
            mag=ws[self.inputs[1]].x[:, 0],
            mag_preds_mm=ws[self.inputs[2]].x[:, 0],
            mag_zv_points=ws[self.inputs[3]],
            mag_baseline=mag_baseline,
        )
