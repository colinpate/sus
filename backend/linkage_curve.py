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
        self.travel[1:] = np.cumsum(np.linalg.norm(axle_locs[1:, :] - axle_locs[:-1, :], axis=1))

    def angle_to_travel(self, angle_series: np.ndarray):
        return np.interp(
            x=angle_series,
            xp=self.angle,
            fp=self.travel
        )
    
        
def main():
    curve = RockerTravelCurve("foo.json")

if __name__ == "__main__":
    main()