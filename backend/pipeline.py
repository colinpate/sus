from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from classes.sensor_loader import Workspace, SensorLoader, AccelLoader, MagLoader, AngleLoader
from classes.step import Step, FilterStep, ChunkStep
from accel_rotation import FilterChunkPairs, FilterColinearPairs, RotationFromPairs, GetRelativeAccel, GetAccelTravelVector, ProjectAccel
from angle import AngleToTravel
from mag import ProjectMag, MagToTravelPolyFit
from classes.time_series import TimeSeries
from classes.runner import Runner, PlotSpec

def main() -> None:
    log_filename = "log023"
    out_dir = Path("run_artifacts") / log_filename
    log_path = Path(f"../logs/{log_filename}.csv")


    # Load sensors (OOP edge)
    loaders: List[SensorLoader] = [
        AccelLoader(sensor_id="lis1", path=log_path),
        AccelLoader(sensor_id="lis2", path=log_path),
        MagLoader(path=log_path),
        AngleLoader(path=log_path)
    ]

    ws: Workspace = {}
    for loader in loaders:
        ws.update(loader.load())

    # Define pipeline (functional core + fusion)
    steps: List[Step] = [
        # Get rotation matrix to align accelerometer data
        FilterStep(
            name="lowpass_lis1",
            inputs=("accel/lis1",),
            outputs=("accel_filt/lis1",),
            plot_keys=("accel/lis1", "accel_filt/lis1"),
            fc_hz=20,
            btype="low",
        ),
        FilterStep(
            name="lowpass_lis2",
            inputs=("accel/lis2",),
            outputs=("accel_filt/lis2",),
            plot_keys=("accel/lis2", "accel_filt/lis2"),
            fc_hz=20,
            btype="low",
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
            name="accel_rot_from_pairs",
            inputs=("filtered_pairs_col",),
            outputs=("rotation_matrix",)
        ),

        # Get net acceleration between the sensors, find vector of travel, and project
        GetRelativeAccel(
            name="get_rel_accel",
            inputs=("accel/lis1", "accel/lis2", "rotation_matrix"),
            outputs=("accel/lis2_in_lis1", "accel/relative"),
            plot_keys=("accel/lis2_in_lis1", "accel/relative")
        ),
        FilterStep(
            name="lowpass_accelrel",
            inputs=("accel/relative",),
            outputs=("accel_filt/relative",),
            fc_hz=20,
            btype="low",
        ),
        GetAccelTravelVector(
            name="get_acc_trav_vec",
            inputs=("accel_filt/relative",),
            outputs=("accel_trav_vec", "mags_vs_means",),
            plot_keys=(
                PlotSpec(kind="scatter", key="mags_vs_means"),
            )
        ),
        ProjectAccel(
            name="project_accel",
            inputs=("accel_trav_vec", "accel/relative",),
            outputs=("accel/proj",),
            plot_keys=("accel/proj",)
        ),
        FilterStep(
            name="lowpass_accelproj",
            inputs=("accel/proj",),
            outputs=("accel_filt/proj",),
            fc_hz=20,
            btype="low",
        ),
        
        # Angle data to travel
        FilterStep(
            name="lowpass_angle",
            inputs=("angle",),
            outputs=("angle/filt",),
            plot_keys=("angle","angle/filt"),
            fc_hz=20,
            btype="low",
        ),
        AngleToTravel(
            name="angle_to_travel",
            inputs=("angle/filt",),
            outputs=("travel",),
        ),

        # Magnetometer processing
        FilterStep(
            name="lowpass_mag",
            inputs=("mag",),
            outputs=("mag_filt",),
            plot_keys=("mag", "mag_filt"),
            fc_hz=20,
            btype="low",
        ),
        ProjectMag(
            name="project_mag",
            inputs=("mag_filt",),
            outputs=("mag_proj",),
            plot_keys=("mag_proj",)
        ),
        MagToTravelPolyFit(
            name="mag_to_travel_polyfit",
            inputs=("mag_proj", "travel"),
            outputs=("travel/mag_polyfit","travel_vs_mag","travel_vs_pred"),
            plot_keys=(
                PlotSpec(kind="scatter", key="travel_vs_mag"),
                PlotSpec(kind="scatter", key="travel_vs_pred"),
            )
        ),
    ]

    runner = Runner(out_dir=out_dir, write_cache=True, make_plots=True)
    ws = runner.run(ws, steps)

    # Example: access final result
    diff: TimeSeries = ws["accel_filt/a"]
    print("Final diff shape:", diff.x.shape)


if __name__ == "__main__":
    main()