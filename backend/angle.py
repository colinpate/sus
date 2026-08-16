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
    angle_sign: float = 1.0

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        hypotenuse = self.param(ws, "hypotenuse", self.hypotenuse)
        top_adjacent = self.param(ws, "top_adjacent", self.top_adjacent)
        top_zeroangle = self.param(ws, "top_zeroangle")
        top_zeroangle_percentile = self.param(ws, "top_zeroangle_percentile", self.top_zeroangle_percentile)
        angle_sign = float(self.param(ws, "angle_sign", self.angle_sign))
        
        # Get corrected angle
        top_angle = np.arccos(top_adjacent / hypotenuse)
        angle_raw = a.x[:, 0] * angle_sign
        if top_zeroangle is None:
            top_zeroangle = np.percentile(angle_raw, top_zeroangle_percentile)
        net_angle = -1 * (angle_raw - top_zeroangle) + top_angle

        travel = 2 * (top_adjacent - (hypotenuse * np.cos(net_angle)))
        print("Travel min, max, median:", np.min(travel), np.max(travel), np.median(travel))
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
        trav = np.asarray(trav_ts.x[:, 0], dtype=float)
        print(trav.shape)

        chunks = []
        mask = np.ones(len(trav), dtype=bool)

        finite_trav = trav[np.isfinite(trav)]
        if len(trav) == 0:
            print("No travel samples; skipping boring region detection")
        elif len(finite_trav) == 0:
            print("No finite travel samples; skipping boring region detection")
        elif np.ptp(finite_trav) <= self.travel_delta_threshold:
            print(
                "Travel range",
                f"{np.ptp(finite_trav):.3f}",
                "is below boring-region threshold; skipping boring region detection",
            )
        else:
            chunk_start = 0
            chunk_min = np.inf
            chunk_max = -np.inf
            chunk_has_finite = False

            for i, value in enumerate(trav):
                if np.isfinite(value):
                    chunk_min = min(chunk_min, value)
                    chunk_max = max(chunk_max, value)
                    chunk_has_finite = True

                if i <= chunk_start or not chunk_has_finite:
                    continue

                # Find end of boring region
                cond_1 = (chunk_max - chunk_min) > self.travel_delta_threshold
                cond_2 = chunk_max > self.max_travel
                if cond_1 or cond_2:
                    chunk_end = i + 1

                    # Only keep boring regions that are long enough
                    if (chunk_end - chunk_start) >= self.min_region_len_samp:
                        chunks.append((max(0, chunk_start + self.padding), min(len(trav), chunk_end - self.padding)))

                    chunk_start = chunk_end
                    chunk_min = np.inf
                    chunk_max = -np.inf
                    chunk_has_finite = False

        print(len(chunks), "boring regions found")

        # Create mask for boring regions
        for start, end in chunks:
            mask[start:end] = False

        boring_percentage = 100 * (1 - np.sum(mask) / len(mask))
        print("Interesting %:", boring_percentage)

        ws[self.outputs[0]] = chunks
        if len(self.outputs) > 1:
            ws[self.outputs[1]] = mask
