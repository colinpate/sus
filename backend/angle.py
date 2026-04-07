from dataclasses import dataclass

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step

@dataclass
class AngleToTravel(Step):
    """Get suspension travel from angle"""
    hypotenuse: float = 120
    top_adjacent: float = 239 / 2#237.5 / 2

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        
        # Get corrected angle
        top_angle = np.arccos(self.top_adjacent / self.hypotenuse)
        top_zeroangle = np.percentile(a.x, 99.5)
        net_angle = -1 * (a.x - top_zeroangle) + top_angle

        travel = 2 * (self.top_adjacent - (self.hypotenuse * np.cos(net_angle)))
        print("Travel min, max:", np.min(travel), np.max(travel))
        print("Travel top zero angle:", top_zeroangle)

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=travel,
            units="mm",
            frame=a.frame,
            meta={**a.meta},
        )

@dataclass
class FindBoringRegions(Step):
    """Find boring regions where travel is stable"""
    travel_delta_threshold: float = 10  # mm
    max_travel: float = 50 # mm
    min_region_len_samp: int = 100
    padding : int = 10

    def run(self, ws: Workspace) -> None:
        trav_ts: TimeSeries = ws[self.inputs[0]]
        trav = trav_ts.x[:, 0]
        print(trav.shape)

        chunk_start = 0

        chunks = []
        i = 1
        while i < len(trav):
            i += 1
            # Find end of boring region
            cond_1 = np.abs(max(trav[chunk_start:i]) - min(trav[chunk_start:i])) > self.travel_delta_threshold
            cond_2 = np.max(trav[chunk_start:i]) > self.max_travel
            if cond_1 or cond_2:
                chunk_end = i

                # Only keep boring regions that are long enough
                if (chunk_end - chunk_start) >= self.min_region_len_samp:
                    chunks.append((max(0, chunk_start + self.padding), min(len(trav), chunk_end - self.padding)))

                chunk_start = i

        print(len(chunks), "boring regions found")

        # Create mask for boring regions
        mask = np.ones(len(trav), dtype=bool)
        for start, end in chunks:
            mask[start:end] = False

        ws[self.outputs[0]] = chunks
        ws[self.outputs[1]] = mask