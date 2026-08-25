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
from angle import FindBoringRegions, LinkageAngleToTravel, AngleToTravel
from mag import ProjectMag, FindMagZVPoints, DiffMag, MagAngle
from classes.runner import Runner, PlotSpec
from classes.log_config import attach_log_config, get_log_config_path, get_signal_config, load_log_config

DEC_FREQ = 200 # Hz, for decimating data to speed up optimization
LP_FREQ = 40 # Hz, for lowpass filtering accel and gyro data
ACCEL_HP_FREQ = 1 # Hz, for highpass filtering rear accel before projection
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
            mark_bad_samples=bool(angle_signal_config.get("mark_bad_samples", False)),
            allow_degenerate=True,
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
    ]
    if "angle_to_travel" in log_config["steps"]:
        steps += [
            AngleToTravel(
                name="angle_to_travel",
                inputs=("angle/lpf",),
                outputs=("travel",),
            ),
        ]
    else:
        steps += [
            LinkageAngleToTravel(
                name="linkage_angle_to_travel",
                inputs=("angle/lpf",),
                outputs=("travel",),
            ),
        ]
    steps += [
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
        MagAngle(
            name="mag_angle",
            inputs=("mag_diff",),
            outputs=("mag/angle",)
        ),

        FilterStep(
            name="lowpass_mag/angle",
            inputs=("mag/angle",),
            outputs=("mag/angle/lpf",),
            plot_keys=("mag/angle/lpf",),
            fc_hz=MAG_LP_FREQ,
            btype="low",
            dec_freq=DEC_FREQ,
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
