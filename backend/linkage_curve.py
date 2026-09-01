from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import json

@dataclass
class RockerTravelCurve:
    source_path: str = ""

    def __post_init__(self) -> None:
        with open(self.source_path, "r") as fi:
            json_lists = json.load(fi)
        angles_i = json_lists[0]
        locs_i = json_lists[1]
        self.angle = np.asarray(angles_i)
        axle_locs = np.asarray(locs_i)
        self.travel = np.zeros_like(self.angle)
        delta = axle_locs[1:, :] - axle_locs[:-1, :]
        self.travel[1:] = np.cumsum(np.linalg.norm(delta, axis=1))

        if len(json_lists) > 2:
            ss_angles = np.asarray(json_lists[2])
            #print("Seatstay angles (degrees):", ss_angles)
            delta_angles = np.degrees(np.atan2(delta[:, 1], delta[:, 0]))
            #print("Angle deltas (degrees):", delta_angles)
            angles_in_accel_frame = delta_angles - ss_angles[:-1]
            #print("Angles in accel frame:", angles_in_accel_frame)
            #print("Travel", self.travel)



    def angle_to_travel(self, angle_series: np.ndarray):
        return np.interp(
            x=angle_series,
            xp=self.angle,
            fp=self.travel
        )
    
        
def main():
    curve = RockerTravelCurve("config/linkages/stumpjumper-rear-original.json")

if __name__ == "__main__":
    main()
