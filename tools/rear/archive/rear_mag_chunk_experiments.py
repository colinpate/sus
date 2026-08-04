from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rear_mag_model import RearMagModel, load_ws  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    min_chunk_dt: float = 0.1
    max_chunk_dt: float = 0.2
    min_chunk_db: float = 500.0
    chunk_min_dx: float = 10.0
    chunk_max_dx: float = 150.0
    min_abs_b_x_corr: float | None = None
    min_db_per_dx: float | None = None
    pair_mode: str = "first_valid"


def aligned_rmse(pred_travel: np.ndarray, travel: np.ndarray) -> tuple[float, float, float]:
    best_offset = float(np.median(travel - pred_travel))
    pred_aligned = pred_travel + best_offset
    rmse = float(np.sqrt(np.mean((travel - pred_aligned) ** 2)))
    corr = float(np.corrcoef(pred_travel, travel)[0, 1])
    return rmse, corr, best_offset


def apply_config(model: RearMagModel, config: ExperimentConfig) -> None:
    model.min_chunk_dt = config.min_chunk_dt
    model.max_chunk_dt = config.max_chunk_dt
    model.min_chunk_db = config.min_chunk_db
    model.chunk_min_dx = config.chunk_min_dx
    model.chunk_max_dx = config.chunk_max_dx
    model.min_abs_b_x_corr = config.min_abs_b_x_corr
    model.min_db_per_dx = config.min_db_per_dx
    model.pair_mode = config.pair_mode


def evaluate_log(log_filename: str, config: ExperimentConfig) -> dict:
    a_hp_proj, b_proj, t, travel, v_gt, a_gt, zv_points, _roi_mask = load_ws(log_filename)

    sign_results = []
    for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
        model = RearMagModel()
        apply_config(model, config)
        chunks = model.create_chunks(zv_points, b_proj, sign * a_hp_proj, t)
        model.prepare_chunks(chunks)
        model.calc_chunks_errors(chunks, travel, v_gt, a_gt)
        chunks_filt = model.filter_chunks(chunks, model.get_filter_fns())
        if not chunks_filt:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            input_arr = model.format_chunks_for_fit(chunks_filt)
            model.fit_model(input_arr, guess_vec=[0, -1, 1 / 3])

        pred_travel = model.model.pred_x(b_proj)
        rmse, corr, offset = aligned_rmse(pred_travel, travel)
        sign_results.append(
            {
                "sign_name": sign_name,
                "rmse": rmse,
                "corr": corr,
                "offset": offset,
                "chunks": len(chunks_filt),
            }
        )

    if not sign_results:
        raise ValueError(f"No chunks survived for {log_filename} with config {config.name}")
    return min(sign_results, key=lambda x: x["rmse"])


def build_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(name="legacy_base", chunk_max_dx=1500.0),
        ExperimentConfig(name="rear_default"),
        ExperimentConfig(name="dx_130", chunk_max_dx=130.0),
        ExperimentConfig(name="dx_140", chunk_max_dx=140.0),
        ExperimentConfig(name="dx_160", chunk_max_dx=160.0),
        ExperimentConfig(name="dx_180", chunk_max_dx=180.0),
        ExperimentConfig(name="corr_0p7", min_abs_b_x_corr=0.7),
        ExperimentConfig(name="dbdx_8", min_db_per_dx=8.0),
        ExperimentConfig(
            name="pair_max_db_per_dt_legacy",
            pair_mode="max_db_per_dt",
            chunk_max_dx=1500.0,
        ),
        ExperimentConfig(
            name="pair_max_db_per_dt_dx_150",
            pair_mode="max_db_per_dt",
            chunk_max_dx=150.0,
        ),
    ]


def print_results(logs: list[str], configs: list[ExperimentConfig]) -> None:
    rows = []
    for config in configs:
        per_log = []
        for log_filename in logs:
            result = evaluate_log(log_filename, config)
            per_log.append((log_filename, result))
        mean_rmse = float(np.mean([item[1]["rmse"] for item in per_log]))
        mean_chunks = float(np.mean([item[1]["chunks"] for item in per_log]))
        rows.append((mean_rmse, mean_chunks, config, per_log))

    rows.sort(key=lambda x: x[0])
    for mean_rmse, mean_chunks, config, per_log in rows:
        details = ", ".join(
            f"{log}: rmse={result['rmse']:.3f}, chunks={result['chunks']}, sign={result['sign_name']}"
            for log, result in per_log
        )
        print(
            f"{config.name:24s} mean_rmse={mean_rmse:7.3f} "
            f"mean_chunks={mean_chunks:6.1f}  {details}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep rear mag chunk settings.")
    parser.add_argument(
        "logs",
        nargs="*",
        default=["log136_rear", "log137_rear"],
        help="Log names to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_results(args.logs, build_configs())


if __name__ == "__main__":
    main()
