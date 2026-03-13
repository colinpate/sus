from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from argparse import ArgumentParser

from classes.sensor_loader import Workspace, SensorLoader, AccelLoader, MagLoader, AngleLoader
from classes.step import Step, FilterStep, ChunkStep
from accel_rotation import FilterChunkPairs, FilterColinearPairs, RotationFromPairs, GetRelativeAccel, GetAccelTravelVector, ProjectAccel
from angle import AngleToTravel, FindBoringRegions
from mag import ProjectMag, FindMagZVPoints, CorrectBadMagProj
from fusion import GetMagTravelRefPoint, GetMagToTravelModel, GetErrorStats
from travel_solver import TravelSolver
from classes.time_series import TimeSeries
from classes.runner import Runner, PlotSpec

def main() -> None:
    log_filename = parse_args().log_filename
    out_dir = Path("run_artifacts") / log_filename
    log_path = Path(f"../logs/{log_filename}.csv")


    # Load sensors (OOP edge)
    loaders: List[SensorLoader] = [
        AccelLoader(sensor_id="lis1", path=log_path),
        AccelLoader(sensor_id="lis2", path=log_path, scale=9.81 / 1000 * 1.0),
        MagLoader(path=log_path, lag=0),
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
            outputs=("accel/lpf/lis1",),
            plot_keys=("accel/lis1", "accel/lpf/lis1"),
            fc_hz=20,
            btype="low",
        ),
        FilterStep(
            name="lowpass_lis2",
            inputs=("accel/lis2",),
            outputs=("accel/lpf/lis2",),
            plot_keys=("accel/lis2", "accel/lpf/lis2"),
            fc_hz=20,
            btype="low",
        ),
        ChunkStep(
            name="chunk_lis1",
            inputs=("accel/lpf/lis1",),
            outputs=("accel_chunks/lis1",),
            chunk_t_s=0.25,
        ),
        ChunkStep(
            name="chunk_lis2",
            inputs=("accel/lpf/lis2",),
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
            outputs=("accel/lpf/relative",),
            fc_hz=20,
            btype="low",
        ),
        GetAccelTravelVector(
            name="get_acc_trav_vec",
            inputs=("accel/lpf/relative",),
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
            outputs=("accel/lpf/proj",),
            fc_hz=20,
            btype="low",
        ),
        FilterStep(
            name="highpass_accelproj",
            inputs=("accel/lpf/proj",),
            outputs=("accel/lpfhp/proj",),
            fc_hz=1,
            btype="high",
        ),
        
        # Angle data to travel
        FilterStep(
            name="lowpass_angle",
            inputs=("angle",),
            outputs=("angle/lpf",),
            plot_keys=("angle","angle/lpf"),
            fc_hz=20,
            btype="low",
        ),
        AngleToTravel(
            name="angle_to_travel",
            inputs=("angle/lpf",),
            outputs=("travel",),
        ),
        FindBoringRegions(
            name="find_boring_regions",
            inputs=("travel",),
            outputs=("boring_regions", "boring_mask"),
        ),

        # Magnetometer processing
        ProjectMag(
            name="project_mag",
            inputs=("mag",),
            outputs=("mag/proj",),
            plot_keys=("mag/proj",)
        ),
        FilterStep(
            name="lowpass_mag/proj",
            inputs=("mag/proj",),
            outputs=("mag/proj/lpf",),
            plot_keys=("mag/proj/lpf",),
            fc_hz=20,
            btype="low",
        ),
        CorrectBadMagProj(
            name="find_bad_mag/proj",
            inputs=("mag", "mag/proj"),
            outputs=("mag/proj/corr", "mag/proj/bad_mask",)
        ),
        FilterStep(
            name="lowpass_mag/proj/corr",
            inputs=("mag/proj/corr",),
            outputs=("mag/proj/corr/lpf",),
            plot_keys=("mag/proj/corr/lpf",),
            fc_hz=20,
            btype="low",
        ),
        FindMagZVPoints(
            name="find_mag_zv_points",
            inputs=("mag/proj/lpf",),
            outputs=("mag_zv_points",)
        ),

        # Fusion steps
        GetMagTravelRefPoint(
            name="get_mag_travel_ref_point",
            inputs=("mag/proj/corr/lpf", "accel/lpfhp/proj", "travel"),
            outputs=("mag_travel_ref_point", "mag_baseline")
        ),
        GetMagToTravelModel(
            name="mag_to_travel_model",
            inputs=(
                "mag/proj/corr/lpf", 
                "accel/lpf/proj", 
                "travel", 
                "mag/proj/bad_mask", 
                "mag_zv_points",
                "mag_travel_ref_point",
                ),
            outputs=("travel/mag_model", "travel/mag_model/adj", "fusion_scatter_points"),
            plot_keys=(
                PlotSpec(kind="scatter", key="fusion_scatter_points"),
            ),
            train_with_mask=True,
        ),
        GetErrorStats(
            name="x_preds_stats",
            inputs=("travel/mag_model", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=0
        ),
        GetErrorStats(
            name="x_preds_adj_stats",
            inputs=("travel/mag_model/adj", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=0
        ),
        TravelSolver(
            name="travel_solver",
            inputs=(
                "accel/lpfhp/proj", 
                "mag/proj/corr/lpf", 
                "travel/mag_model/adj", 
                "mag_zv_points", 
                "mag_baseline",
                "travel",
            ),
            outputs=("travel/solved",),
            plot_keys=("travel/solved",)
        ),
        GetErrorStats(
            name="x_preds_solver",
            inputs=("travel/solved", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=0
        ),
    ]

    runner = Runner(out_dir=out_dir, write_cache=True, make_plots=True)
    ws = runner.run(ws, steps)

    # Example: access final result
    #print(ws.keys())
    #diff: TimeSeries = ws["accel/lpf/a"]
    #print("Final diff shape:", diff.x.shape)

def parse_args() -> Any:
    parser = ArgumentParser(description="Run suspension data processing pipeline")
    parser.add_argument("log_filename", type=str, default="log038", help="Name of log file (without .csv extension) to process")
    return parser.parse_args()

if __name__ == "__main__":
    main()