from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from argparse import ArgumentParser

from classes.sensor_loader import (
    Workspace,
    SensorLoader,
    AccelLoader,
    MagLoader,
    AngleLoader,
    LISMagLoader,
    GyroLoader,
)
from classes.step import Step, FilterStep, ChunkStep
from accel_rotation import (
    FilterChunkPairs, 
    FilterColinearPairs, 
    RotationFromPairs, 
    GetRelativeAccel, 
    GetAccelTravelVector, 
    ProjectAccel,
    GetAccelError,
    CorrectStaticOffset
)
from angle import AngleToTravel, FindBoringRegions
from mag import ProjectMag, FindMagZVPoints, CorrectBadMagProj, MagMagnitude
from mag_nuisance import (
    MagNuisanceFullRateCorrection,
    MagNuisanceTravelCorrection,
)
from fusion import GetMagTravelRefPoint, GetMagToTravelModel, GetErrorStats, GetMagBaseline
from travel_solver import TravelSolver
from classes.time_series import TimeSeries
from classes.runner import Runner, PlotSpec
from classes.log_config import attach_log_config, get_log_config_path, get_signal_config, load_log_config

DEC_FREQ = 100 # Hz, for decimating data to speed up optimization

def main() -> None:
    log_filename = parse_args().log_filename
    out_dir = Path("backend/run_artifacts") / log_filename
    log_path = Path(f"logs/{log_filename}.csv")
    log_config = load_log_config(log_path)
    if log_config:
        print(f"Loaded log config from {get_log_config_path(log_path)}")


    # Load sensors (OOP edge)
    loaders: List[SensorLoader] = [
        AccelLoader(sensor_id="lis1", path=log_path),
        AccelLoader(sensor_id="lis2", path=log_path, scale=9.81 / 1000 * 1.0),
        GyroLoader(sensor_id="gyro1", path=log_path),
        GyroLoader(sensor_id="gyro2", path=log_path),
        MagLoader(path=log_path, lag=0, signal_config=get_signal_config(log_config, "mag")),
        LISMagLoader(path=log_path, lag=0, signal_config=get_signal_config(log_config, "mag_lis")),
        AngleLoader(path=log_path, lag=-1, allow_degenerate=True),
    ]

    ws: Workspace = {}
    attach_log_config(ws, log_config)
    for loader in loaders:
        ws.update(loader.load())

    # Define pipeline (functional core + fusion)
    steps: List[Step] = [
        FilterStep(
            name="lowpass_gyro1",
            inputs=("gyro/gyro1",),
            outputs=("gyro/lpf/gyro1",),
            plot_keys=("gyro/gyro1", "gyro/lpf/gyro1"),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_gyro2",
            inputs=("gyro/gyro2",),
            outputs=("gyro/lpf/gyro2",),
            plot_keys=("gyro/gyro2", "gyro/lpf/gyro2"),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        # Get rotation matrix to align accelerometer data
        FilterStep(
            name="lowpass_lis1",
            inputs=("accel/lis1",),
            outputs=("accel/lpf/lis1",),
            plot_keys=("accel/lis1", "accel/lpf/lis1"),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_lis2",
            inputs=("accel/lis2",),
            outputs=("accel/lpf/lis2",),
            plot_keys=("accel/lis2", "accel/lpf/lis2"),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
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
            outputs=("filtered_pairs_col", "lis1_chunks_filt", "lis2_chunks_filt")
        ),
        CorrectStaticOffset(
            name="correct_lis1_offset",
            inputs=("lis1_chunks_filt", "accel/lis1"),
            outputs=("lis1_chunks_filt", "accel/lis1"),
        ),
        CorrectStaticOffset(
            name="correct_lis2_offset",
            inputs=("lis2_chunks_filt", "accel/lis2"),
            outputs=("lis2_chunks_filt", "accel/lis2"),
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
            dec_freq=DEC_FREQ,
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
            dec_freq=DEC_FREQ,
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
            dec_freq=DEC_FREQ,
        ),
        AngleToTravel(
            name="angle_to_travel",
            inputs=("angle/lpf",),
            outputs=("travel",),
        ),
        GetAccelError(
            name="accel_proj_error",
            inputs=("accel/lpf/proj", "travel"),
            outputs=(),
        ),
        FindBoringRegions(
            name="find_boring_regions",
            inputs=("travel",),
            outputs=("boring_regions", "boring_mask"),
            read_cache=True
        ),

        # Magnetometer processing
        MagMagnitude(
            name="mag_magnitude",
            inputs=("mag", "accel/proj"),
            outputs=("mag/norm",),
            plot_keys=("mag/norm",),
        ),
        FilterStep(
            name="lowpass_mag",
            inputs=("mag",),
            outputs=("mag/lpf",),
            plot_keys=("mag/lpf",),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_mag_lis",
            inputs=("mag_lis",),
            outputs=("mag_lis/lpf",),
            plot_keys=("mag_lis/lpf",),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_mag/norm",
            inputs=("mag/norm",),
            outputs=("mag/norm/lpf",),
            plot_keys=("mag/norm/lpf",),
            fc_hz=20,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        CorrectBadMagProj(
            name="find_bad_mag_proj",
            inputs=("mag/lpf", "mag/norm/lpf"),
            outputs=("mag/norm/corr/lpf", "mag/norm/bad_mask",)
        ),
        FindMagZVPoints(
            name="find_mag_zv_points",
            inputs=("mag/norm/corr/lpf",),
            outputs=("mag_zv_points",)
        ),

        # Fusion steps
        GetMagBaseline(
            name="get_mag_baseline",
            inputs=("mag/norm/corr/lpf", "accel/lpfhp/proj"),
            outputs=("mag_baseline",)
        ),
        GetMagTravelRefPoint(
            name="get_mag_travel_ref_point",
            inputs=("mag/norm/corr/lpf", "accel/lpfhp/proj", "mag_baseline", "travel"),
            outputs=("mag_travel_ref_point",)
        ),
        GetMagToTravelModel(
            name="mag_to_travel_model",
            inputs=(
                "mag/norm/corr/lpf",
                "accel/lpfhp/proj", 
                "travel", 
                "mag/norm/bad_mask",
                "mag_zv_points",
                "mag_travel_ref_point",
                "mag_baseline"
                ),
            outputs=(
                "travel/mag_model",
                "travel/mag_model/adj",
                "fusion_scatter_points",
                "mag_model_coeffs",
                "mag_model_offset_mm",
            ),
            plot_keys=(
                PlotSpec(kind="scatter", key="fusion_scatter_points"),
            ),
            train_with_mask=False,
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
                "mag/norm/corr/lpf",
                "travel/mag_model/adj", 
                "mag_zv_points", 
                "mag_baseline",
            ),
            outputs=("travel/solved",),
            plot_keys=("travel/solved",)
        ),
        # Stage one: retain the original full-rate solution and expose the
        # validated four-iteration correction on its 10 Hz state grid. A later
        # stage can turn its proposal into a confidence-weighted fusion input.
        MagNuisanceTravelCorrection(
            name="mag_nuisance_correction",
            inputs=(
                "mag/lpf",
                "gyro/lpf/gyro1",
                "mag/norm/corr/lpf",
                "mag_model_coeffs",
                "mag_model_offset_mm",
                "travel/solved",
            ),
            outputs=(
                "travel/solved/mag_nuisance/10hz",
                "travel/mag_nuisance/proposal/10hz",
                "mag/nuisance/body/10hz",
                "mag/nuisance/world/10hz",
                "mag/nuisance/correction/10hz",
                "mag/nuisance/corrected_xyz/10hz",
                "mag/nuisance/update_mask/10hz",
                "mag/nuisance/xyz_path",
                "mag/nuisance/iteration_change_mm",
                "mag/nuisance/summary",
            ),
            plot_keys=("travel/solved/mag_nuisance/10hz",),
        ),
        MagNuisanceFullRateCorrection(
            name="mag_nuisance_full_rate",
            inputs=(
                "mag/lpf",
                "gyro/lpf/gyro1",
                "travel/solved",
                "travel/mag_model/adj",
                "travel/solved/mag_nuisance/10hz",
                "mag/nuisance/body/10hz",
                "mag/nuisance/world/10hz",
                "mag/nuisance/xyz_path",
            ),
            outputs=(
                "travel/solved/mag_nuisance/delta_lifted",
                "travel/mag_nuisance/corrected",
                "mag/nuisance/corrected_xyz",
                "mag/nuisance/correction",
                "mag/nuisance/confidence",
                "mag/nuisance/full_rate_summary",
            ),
            plot_keys=(
                "travel/solved/mag_nuisance/delta_lifted",
                "travel/mag_nuisance/corrected",
            ),
        ),
        TravelSolver(
            name="travel_solver_mag_nuisance",
            inputs=(
                "accel/lpfhp/proj",
                "mag/norm/corr/lpf",
                "travel/mag_nuisance/corrected",
                "mag_zv_points",
                "mag_baseline",
            ),
            outputs=("travel/solved/mag_nuisance/fusion2",),
            plot_keys=("travel/solved/mag_nuisance/fusion2",),
        ),
        GetErrorStats(
            name="x_preds_solver_mag_nuisance_delta_lifted",
            inputs=(
                "travel/solved/mag_nuisance/delta_lifted",
                "travel",
                "boring_mask",
            ),
            outputs=(),
            gt_thresh=0,
        ),
        GetErrorStats(
            name="x_preds_solver_mag_nuisance_fusion2",
            inputs=(
                "travel/solved/mag_nuisance/fusion2",
                "travel",
                "boring_mask",
            ),
            outputs=(),
            gt_thresh=0,
        ),
        GetErrorStats(
            name="x_preds_solver",
            inputs=("travel/solved", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=0
        ),
        GetErrorStats(
            name="x_preds_solver",
            inputs=("travel/solved", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=30
        ),
    ]

    runner = Runner(out_dir=out_dir, write_cache=True, make_plots=False)
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
