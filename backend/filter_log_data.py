from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from argparse import ArgumentParser
from classes.runner import Runner
from classes.step import FilterStep

from angle import AngleToTravel, FindBoringRegions
from classes.sensor_loader import AngleLoader
from classes.log_config import attach_log_config, get_log_config_path, load_log_config
import numpy as np

def main() -> None:
    log_filename = parse_args().log_filename
    out_dir = Path("run_artifacts") / log_filename
    log_path = Path(f"../logs/{log_filename}.csv")
    log_config = load_log_config(log_path)
    if log_config:
        print(f"Loaded log config from {get_log_config_path(log_path)}")

    angle_loader = AngleLoader(path=log_path)
    ws = {}
    attach_log_config(ws, log_config)
    ws.update(angle_loader.load())

    steps = [
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
            name="find_boring",
            inputs=("travel",),
            outputs=("boring_chunks",),
        )
    ]

    runner = Runner(out_dir=out_dir, write_cache=False, make_plots=False)
    ws = runner.run(ws, steps)

    create_filtered_csv(log_filename, log_path, ws["boring_chunks"])

def create_filtered_csv(log_filename: str, log_path: Path, boring_chunks: List[Tuple[int, int]]) -> None:
    import pandas as pd

    df = pd.read_csv(log_path)

    t = df["t_s"].values
    hacked_t = hack_time(t, boring_chunks)

    # Create mask for boring regions
    mask = np.ones(len(df), dtype=bool)
    for start, end in boring_chunks:
        mask[start:end] = False

    filtered_df = df[mask].copy()
    filtered_df["t_s"] = hacked_t[mask]
    print("Saving filtered log with", len(filtered_df), "rows (removed", len(df) - len(filtered_df), "rows)")

    out_path = Path(f"../logs/{log_filename}_filtered.csv")
    filtered_df.to_csv(out_path, index=False)
    print(f"Filtered log saved to {out_path}")

def hack_time(t: np.ndarray, chunks: List[Tuple[int, int]]) -> np.ndarray:
    hacked_t = t.copy()
    avg_dt = np.median(np.diff(t))
    for chunk in chunks:
        chunk_dt = hacked_t[chunk[1]] - hacked_t[chunk[0]]
        hacked_t[chunk[1]:] -= chunk_dt - avg_dt
    return hacked_t

def parse_args() -> Any:
    parser = ArgumentParser(description="Remove boring regions from log file")
    parser.add_argument("log_filename", type=str, default="log038", help="Name of log file (without .csv extension) to process")
    return parser.parse_args()

if __name__ == "__main__":
    main()
