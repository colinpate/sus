from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from argparse import ArgumentParser

from accel_rotation import ProjectAccel, GetAccelTravelVectorRear
from classes.sensor_loader import (
    Workspace,
    SensorLoader,
    AccelLoader,
    MagLoader,
    AngleLoader,
    LISMagLoader,
    GyroLoader,
)
from classes.step import Step, FilterStep
from angle import FindBoringRegions, LinkageAngleToTravel
from mag import ProjectMag, FindMagZVPoints, DiffMag
from fusion import GetMagTravelRefPoint, GetMagToTravelModel, GetRearMagToTravelModel, GetErrorStats, GetMagBaseline
from travel_solver import RearTravelSolver, TravelSolver
from classes.time_series import TimeSeries
from classes.runner import Runner, PlotSpec
from classes.log_config import attach_log_config, get_log_config_path, get_signal_config, load_log_config

DEC_FREQ = 200 # Hz, for decimating data to speed up optimization
LP_FREQ = 40 # Hz, for lowpass filtering accel and gyro data
MAG_LP_FREQ = 20 # Hz, for lowpass filtering magnetometer data

def main() -> None:
    log_filename = parse_args().log_filename
    out_dir = Path("backend/run_artifacts") / log_filename
    log_path = Path(f"logs/{log_filename}.csv")
    log_config = load_log_config(log_path)
    if log_config:
        print(f"Loaded log config from {get_log_config_path(log_path)}")
    angle_signal_config = get_signal_config(log_config, "angle")


    # Load sensors (OOP edge)
    loaders: List[SensorLoader] = [
        AccelLoader(sensor_id="lis1", path=log_path),
        GyroLoader(sensor_id="gyro1", path=log_path),
        AccelLoader(sensor_id="lis2", path=log_path),
        GyroLoader(sensor_id="gyro2", path=log_path),
        MagLoader(path=log_path, lag=0, signal_config=get_signal_config(log_config, "mag")),
        LISMagLoader(path=log_path, lag=0, signal_config=get_signal_config(log_config, "mag_lis")),
        AngleLoader(
            path=log_path,
            lag=int(angle_signal_config.get("lag", -1)),
            interpolate_bad=bool(angle_signal_config.get("interpolate_bad", False)),
            offset=int(angle_signal_config.get("offset", 2048)),
        ),
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
            fc_hz=LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_gyro2",
            inputs=("gyro/gyro2",),
            outputs=("gyro/lpf/gyro2",),
            plot_keys=("gyro/gyro2", "gyro/lpf/gyro2"),
            fc_hz=LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_lis1",
            inputs=("accel/lis1",),
            outputs=("accel/lpf/lis1",),
            plot_keys=("accel/lis1", "accel/lpf/lis1"),
            fc_hz=LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_lis2",
            inputs=("accel/lis2",),
            outputs=("accel/lpf/lis2",),
            plot_keys=("accel/lis2", "accel/lpf/lis2"),
            fc_hz=LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),

        FilterStep(
            name="highpass_accel",
            inputs=("accel/lpf/lis2",),
            outputs=("accel/lphp/lis2",),
            fc_hz=2,
            btype="high",
            N=2,
        ),
        GetAccelTravelVectorRear(
            name="get_acc_trav_vec",
            inputs=("accel/lphp/lis2",),
            outputs=("accel_trav_vec",),
        ),
        ProjectAccel(
            name="project_accel",
            inputs=("accel_trav_vec", "accel/lphp/lis2",),
            outputs=("accel/lphp/proj",),
            plot_keys=("accel/lphp/proj",)
        ),
        
        # Angle data to travel
        FilterStep(
            name="lowpass_angle",
            inputs=("angle",),
            outputs=("angle/lpf",),
            plot_keys=("angle","angle/lpf"),
            fc_hz=LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        LinkageAngleToTravel(
            name="linkage_angle_to_travel",
            inputs=("angle/lpf",),
            outputs=("travel",),
        ),
        # GetAccelError(
        #     name="accel_proj_error",
        #     inputs=("accel/lpf/proj", "travel"),
        #     outputs=(),
        # ),
        FindBoringRegions(
            name="find_boring_regions",
            inputs=("travel",),
            outputs=("boring_regions", "boring_mask"),
            read_cache=True
        ),

        # Magnetometer processing
        FilterStep(
            name="lowpass_mag",
            inputs=("mag",),
            outputs=("mag/lpf",),
            plot_keys=("mag/lpf",),
            fc_hz=MAG_LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        FilterStep(
            name="lowpass_mag_lis",
            inputs=("mag_lis",),
            outputs=("mag_lis/lpf",),
            plot_keys=("mag_lis/lpf",),
            fc_hz=MAG_LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        DiffMag(
            name="diff_mag",
            inputs=("mag", "mag_lis"),
            outputs=("mag_diff",),
        ),
        FilterStep(
            name="lowpass_mag_diff",
            inputs=("mag_diff",),
            outputs=("mag_diff/lpf",),
            plot_keys=("mag_diff/lpf",),
            fc_hz=MAG_LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
        ProjectMag(
            name="project_mag",
            inputs=("mag_diff",),
            outputs=("mag/proj",),
            plot_keys=("mag/proj",),
            normalize=True,
        ),

        FilterStep(
            name="lowpass_mag/proj",
            inputs=("mag/proj",),
            outputs=("mag/proj/corr/lpf",),
            plot_keys=("mag/proj/corr/lpf",),
            fc_hz=MAG_LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
        ),
    #     CorrectBadMagProj(
    #         name="find_bad_mag_proj",
    #         inputs=("mag/lpf", "mag/proj/lpf"),
    #         outputs=("mag/proj/corr/lpf", "mag/proj/bad_mask",)
    #     ),
        FindMagZVPoints(
            name="find_mag_zv_points",
            inputs=("mag/proj/corr/lpf",),
            outputs=("mag_zv_points",),
            min_dt=0,
            min_dm=0
        ),

    #     # Fusion steps
    #     GetMagBaseline(
    #         name="get_mag_baseline",
    #         inputs=("mag/proj/corr/lpf", "accel/lpfhp/proj"),
    #         outputs=("mag_baseline",)
    #     ),
    #     GetMagTravelRefPoint(
    #         name="get_mag_travel_ref_point",
    #         inputs=("mag/proj/corr/lpf", "accel/lpfhp/proj", "mag_baseline", "travel"),
    #         outputs=("mag_travel_ref_point",)
    #     ),
        GetRearMagToTravelModel(
            name="mag_to_travel_model",
            inputs=(
                "mag/proj/corr/lpf", 
                "accel/lphp/proj",
                "mag_zv_points",
                ),
            outputs=(
                "travel/mag_model",
                "travel/mag_model/adj",
                "fusion_scatter_points",
                "mag_model_coeffs"
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
            gt_thresh=None
        ),
        GetErrorStats(
            name="x_preds_adj_stats",
            inputs=("travel/mag_model/adj", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=None
        ),
        RearTravelSolver(
            name="travel_solver",
            inputs=(
                "accel/lphp/proj",
                "travel/mag_model/adj", 
                "mag_zv_points",
            ),
            outputs=("travel/solved",),
            plot_keys=("travel/solved",)
        ),
        GetErrorStats(
            name="x_preds_solver",
            inputs=("travel/solved", "travel", "boring_mask"),
            outputs=(),
            gt_thresh=None
        ),
        # GetErrorStats(
        #     name="x_preds_solver",
        #     inputs=("travel/solved", "travel", "boring_mask"),
        #     outputs=(),
        #     gt_thresh=30
        # ),
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
