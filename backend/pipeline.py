from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from classes.sensor_loader import Workspace, SensorLoader, AccelLoader
from classes.step import Step, FilterStep, ChunkStep
from accel_rotation import FilterChunkPairs, FilterColinearPairs, RotationFromPairs, RelativeAccel, AlignedAccel
from classes.time_series import TimeSeries
from classes.runner import Runner, PlotSpec

def main() -> None:
    out_dir = Path("run_artifacts")

    # Load sensors (OOP edge)
    loaders: List[SensorLoader] = [
        AccelLoader(sensor_id="lis1", path=Path("../logs/log018.csv")),
        AccelLoader(sensor_id="lis2", path=Path("../logs/log018.csv"))
    ]

    ws: Workspace = {}
    for loader in loaders:
        ws.update(loader.load())

    # Define pipeline (functional core + fusion)
    steps: List[Step] = [
        FilterStep(
            name="lowpass_lis1",
            inputs=("accel/lis1",),
            outputs=("accel_filt/lis1",),
            plot_keys=("accel/lis1", "accel_filt/lis1"),
            fc_hz=20,
        ),
        FilterStep(
            name="lowpass_lis2",
            inputs=("accel/lis2",),
            outputs=("accel_filt/lis2",),
            plot_keys=("accel/lis2", "accel_filt/lis2"),
            fc_hz=20,
        ),
        ChunkStep(
            name="chunk_lis1",
            inputs=("accel_filt/lis1",),
            outputs=("accel_chunks/lis1",),
            chunk_t_s=0.25,
        ),
        ChunkStep(
            name="chunk_lis2",
            inputs=("accel_filt/lis2",),
            outputs=("accel_chunks/lis2",),
            chunk_t_s=0.25,
        ),
        FilterChunkPairs(
            name="filter_pairs",
            inputs=("accel_chunks/lis1", "accel_chunks/lis2"),
            outputs=("filtered_pairs",)
        ),
        FilterColinearPairs(
            name="filter_colinear",
            inputs=("filtered_pairs",),
            outputs=("filtered_pairs_col",)
        ),
        RotationFromPairs(
            name="rot_from_pairs",
            inputs=("filtered_pairs_col",),
            outputs=("rotation_matrix",)
        ),
        RelativeAccel(
            name="relative_accel",
            inputs=("accel_filt/lis1", "accel_filt/lis2", "rotation_matrix"),
            outputs=("accel/lis2_in_lis1", "accel/relative"),
            plot_keys=("accel/lis2_in_lis1", "accel/relative")
        ),
        AlignedAccel(
            name="aligned_accel",
            inputs=("accel/relative",),
            outputs=("mags_vs_means","accel/projected"),
            plot_keys=(
                PlotSpec(kind="scatter", key="mags_vs_means"),
                "accel/projected"
            )
        )
    ]

    runner = Runner(out_dir=out_dir, write_cache=True, make_plots=True)
    ws = runner.run(ws, steps)

    # Example: access final result
    diff: TimeSeries = ws["accel_filt/a"]
    print("Final diff shape:", diff.x.shape)


if __name__ == "__main__":
    main()