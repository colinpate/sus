from dataclasses import dataclass

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step

@dataclass
class AngleToTravel(Step):
    """Get suspension travel from angle"""
    hypotenuse: float = 120
    top_adjacent: float = 237.5 / 2

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        
        # Get corrected angle
        top_angle = np.arccos(self.top_adjacent / self.hypotenuse)
        top_zeroangle = np.percentile(a.x, 95)
        net_angle = -1 * (a.x - top_zeroangle) + top_angle

        travel = 2 * (self.top_adjacent - (self.hypotenuse * np.cos(net_angle)))

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=travel,
            units="mm",
            frame=a.frame,
            meta={**a.meta},
        )