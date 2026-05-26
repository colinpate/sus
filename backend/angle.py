from dataclasses import dataclass

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step
from linkage_curve import RockerTravelCurve

@dataclass
class AngleToTravel(Step):
    """Get suspension travel from angle"""
    hypotenuse: float = 120
    top_adjacent: float = 239 / 2#237.5 / 2
    top_zeroangle_percentile: float = 99.5

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        hypotenuse = self.param(ws, "hypotenuse", self.hypotenuse)
        top_adjacent = self.param(ws, "top_adjacent", self.top_adjacent)
        top_zeroangle = self.param(ws, "top_zeroangle")
        top_zeroangle_percentile = self.param(ws, "top_zeroangle_percentile", self.top_zeroangle_percentile)
        
        # Get corrected angle
        top_angle = np.arccos(top_adjacent / hypotenuse)
        if top_zeroangle is None:
            top_zeroangle = np.percentile(a.x, top_zeroangle_percentile)
        net_angle = -1 * (a.x - top_zeroangle) + top_angle

        travel = 2 * (top_adjacent - (hypotenuse * np.cos(net_angle)))
        print("Travel min, max:", np.min(travel), np.max(travel))
        print("Travel top zero angle:", top_zeroangle)

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=travel,
            units="mm",
            frame=a.frame,
            meta={
                **a.meta,
                "angle_to_travel": {
                    "hypotenuse": hypotenuse,
                    "top_adjacent": top_adjacent,
                    "top_zeroangle": float(np.asarray(top_zeroangle).reshape(())),
                    "top_zeroangle_percentile": top_zeroangle_percentile,
                },
            },
        )


@dataclass
class LinkageAngleToTravel(Step):
    """Map rocker angle to wheel travel using a sampled linkage curve."""
    linkage_path: str = ""
    top_zeroangle_percentile: float = 0.5
    angle_sign: float = 1.0

    def run(self, ws: Workspace) -> None:
        if self.angle_sign == 0:
            raise ValueError("angle_sign must be non-zero")

        a: TimeSeries = ws[self.inputs[0]]
        linkage_path = self.param(ws, "linkage_path", self.linkage_path)

        angle_reference_percentile = float(
            self.param(ws, "angle_reference_percentile", self.top_zeroangle_percentile)
        )
        angle_sign = float(self.param(ws, "angle_sign", self.angle_sign))

        angle_deg_raw = np.degrees(a.x[:, 0]) * angle_sign
        sensor_reference_deg = float(np.percentile(angle_deg_raw, angle_reference_percentile))
        angle_deg = angle_deg_raw - sensor_reference_deg

        curve = RockerTravelCurve(linkage_path)
        travel_mm = curve.angle_to_travel(angle_deg)

        print("Raw angle min, max:", np.min(angle_deg_raw), np.max(angle_deg_raw))
        print("Travel angle 0 reference:", sensor_reference_deg)
        print("Travel min, max:", np.min(travel_mm), np.max(travel_mm))

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=travel_mm,
            units="mm",
            frame=a.frame,
            meta={},
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
