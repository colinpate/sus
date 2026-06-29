from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModel, MagToTravelModelCore


RearChunkingMethod = Literal["centered_zv", "debiased_centered_zv", "paired_zv"]


def parse_args():
    parser = ArgumentParser(description="Run rear suspension mag model constructor")
    parser.add_argument("log_filename", type=str, default="log136_rear", help="Name of log file (without .csv extension) to process")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the results")
    return parser.parse_args()

@dataclass
class RearMagModel(MagToTravelModelCore):
    min_chunk_dt: float = 0.1
    max_chunk_dt: float = 0.2
    min_chunk_db: float = 500
    dm_dx_thresh: float | None = None
    pair_mode: str = "first_valid" # "max_db_per_dt" # 
    default_chunk_max_dx: float = 150.0
    max_b_x_corr: float | None = None
    min_abs_b_x_corr: float | None = None
    min_db_per_dx: float | None = None
    chunking_method: RearChunkingMethod = "centered_zv"

    allowed_chunking_methods: ClassVar[tuple[str, ...]] = (
        "centered_zv",
        "debiased_centered_zv",
        "paired_zv",
    )

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_max_dx == 1500:
            self.chunk_max_dx = self.default_chunk_max_dx

    def find_zv_pairs(self, idxs: list[int], b_proj: np.ndarray, t: np.ndarray):
        pairs = []
        idx_1 = 0
        while idx_1 < len(idxs) - 1:
            zv_1 = idxs[idx_1]
            candidates = []
            for idx_2 in range(idx_1 + 1, len(idxs)):
                zv_2 = idxs[idx_2]
                dt = t[zv_2] - t[zv_1]
                db = b_proj[zv_2] - b_proj[zv_1]
                if dt > self.max_chunk_dt:
                    break
                if dt < self.min_chunk_dt:
                    continue
                if abs(db) < self.min_chunk_db:
                    continue
                candidates.append((idx_2, zv_1, zv_2, dt, db))
                if self.pair_mode == "first_valid":
                    break

            if not candidates:
                idx_1 += 1
                continue

            pair_idx, _, pair_zv_2, _, _ = self.choose_pair(candidates)
            pairs.append((zv_1, pair_zv_2))
            idx_1 = pair_idx
        return pairs

    def choose_pair(self, candidates):
        if self.pair_mode == "max_abs_db":
            return max(candidates, key=lambda x: abs(x[4]))
        if self.pair_mode == "max_db_per_dt":
            return max(candidates, key=lambda x: abs(x[4]) / max(x[3], 1e-6))
        return candidates[0]

    def create_chunks(self, idxs, mag, acc, t_s, mag_proj_bad_mask=None):
        self.validate_chunking_method()
        if self.chunking_method in ("centered_zv", "debiased_centered_zv"):
            print("Using centered_zv chunking")
            return super().create_chunks(idxs, mag, acc, t_s, mag_proj_bad_mask)
        elif self.chunking_method == "paired_zv":
            chunks = []
            pairs = self.find_zv_pairs(idxs, mag, t_s)
            for zv_start, zv_stop in pairs:
                # Include both ZV endpoints so the bias correction can force the
                # integrated velocity back to zero across the whole chunk.
                slice_i = slice(zv_start, zv_stop + 1)
                raw_acc = acc[slice_i]
                t_chunk = t_s[slice_i]
                duration = t_chunk[-1] - t_chunk[0]
                if duration <= 0:
                    continue
                # Bias is the constant acceleration term that makes the net delta-v
                # between the two zero-velocity endpoints equal to zero.
                acc_bias = scipy.integrate.trapezoid(raw_acc, t_chunk) / duration
                chunk = MagToTravelChunk(
                    a=(raw_acc - acc_bias) * 1000,
                    t=t_chunk,
                    mag=mag[slice_i],
                    slice_i=slice_i,
                    zv_idx=0
                )
                chunks.append(chunk)
            return chunks
        else:
            raise ValueError(f"Invalid chunking method: {self.chunking_method}")

    def get_filter_fns(self):
        filter_fns = [
            self.filter_chunk_dx,
            #self.filter_chunk_abs_b_x_corr,
            #self.filter_chunk_db_per_dx,
            #self.filter_chunk_max_b_x_corr,
            #self.filter_chunk_max_dm_dx,
            self.filter_chunk_dm_dx
        ]
        return filter_fns

    def filter_chunk_max_dm_dx(self, chunk: MagToTravelChunk):
        if self.dm_dx_thresh is None:
            return True
        return chunk.metrics["dm/dx_median"] <= self.dm_dx_thresh
    
    def filter_chunk_max_b_x_corr(self, chunk: MagToTravelChunk):
        if self.max_b_x_corr is None:
            return True
        return chunk.metrics["b_x_corr"] <= self.max_b_x_corr

    def filter_chunk_abs_b_x_corr(self, chunk: MagToTravelChunk):
        if self.min_abs_b_x_corr is None:
            return True
        return chunk.metrics["abs_b_x_corr"] >= self.min_abs_b_x_corr

    def filter_chunk_db_per_dx(self, chunk: MagToTravelChunk):
        if self.min_db_per_dx is None:
            return True
        return chunk.metrics["db_per_dx"] >= self.min_db_per_dx
    
    def create_training_data(
        self,
        mag,
        accel,
        t,
        idxs,
    ):
        if self.train_with_mask:
            print("Rear mag model ignores train_mask during chunk selection")

        chunks = self.create_chunks(idxs, mag, accel, t)
        self.prepare_chunks(chunks)
        self.chunks = self.filter_chunks(chunks, self.get_filter_fns())
        print("Training chunks:", len(chunks))
        print("Filtered training chunks:", len(self.chunks))
        return self.format_chunks_for_fit(self.chunks)
    

def load_ws(log_filename):
    out_dir = f"backend/run_artifacts/{log_filename}/cache/"
    ws_file = out_dir + "/all.npz"
    ws = np.load(ws_file)
    accel_idx = "2"
    a = ws[f"accel/lphp/proj/zv__x"][:, 0]
    b_proj = ws["mag/angle/lpf__x"][:, 0]
    t = ws[f"accel/lpf/lis{accel_idx}__t"]
    zv_points = ws["mag_zv_points"]
    travel = ws["travel__x"][:, 0]
    roi_mask = ws["boring_mask"]
    salient_time = int((max(t) - min(t)) * np.mean(roi_mask))
    print("Salient mask time, proportion:", salient_time, np.mean(roi_mask))

    v_gt = np.gradient(travel, t, edge_order = 2)
    a_gt = np.gradient(v_gt, t, edge_order = 2)
    return a, b_proj, t, travel, v_gt, a_gt, zv_points, roi_mask

def get_chunk_corrs(chunks, travel):
    x_mag_corrs = []
    x_gt_corrs = []
    for chunk in chunks:
        if chunk.chunk_len < 3:
            continue
        gt_rel = travel[chunk.slice_i] - travel[chunk.slice_i][chunk.zv_idx]
        x_mag_corr = np.corrcoef(chunk.x, chunk.mag)[0, 1]
        x_gt_corr = np.corrcoef(chunk.x, gt_rel)[0, 1]
        if np.isfinite(x_mag_corr):
            x_mag_corrs.append(x_mag_corr)
        if np.isfinite(x_gt_corr):
            x_gt_corrs.append(x_gt_corr)
    return x_mag_corrs, x_gt_corrs

def fit_regression_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.ptp(x) <= 1e-9 or len(np.unique(x)) < 3:
        return np.nan
    return float(scipy.stats.linregress(x, y).slope)

def calc_binned_rmse(err: np.ndarray, gt: np.ndarray, min_bin_count: int = 50):
    edges = np.linspace(0.0, 150.0, 6)
    bin_rmses = []
    bin_mes = []
    eligible_mses = []
    for i, (bin_min, bin_max) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            bin_mask = (bin_min <= gt) & (gt <= bin_max)
        else:
            bin_mask = (bin_min <= gt) & (gt < bin_max)

        count = int(np.sum(bin_mask))
        if count:
            bin_mse = float(np.mean(err[bin_mask] ** 2))
            bin_me = float(np.mean(err[bin_mask]))
            bin_rmses.append(bin_mse ** 0.5)
            bin_mes.append(bin_me)
            if count >= min_bin_count:
                eligible_mses.append(bin_mse)
        else:
            bin_rmses.append(np.nan)
            bin_mes.append(np.nan)

    bin_rmse = float(np.sqrt(np.mean(eligible_mses))) if eligible_mses else np.nan
    return bin_rmse, bin_rmses, bin_mes

def evaluate_predictions(pred_travel, travel, label, roi_mask):
    raw_rmse = np.mean((travel - pred_travel) ** 2) ** 0.5

    # The fit only learns relative chunk motion, so a constant offset is still
    # free here. Report the best constant-offset RMSE separately.
    best_offset = np.median(travel - pred_travel)
    pred_aligned = pred_travel + best_offset
    aligned_rmse = np.mean((travel - pred_aligned) ** 2) ** 0.5
    masked_aligned_rmse = np.mean(((travel - pred_aligned)[roi_mask]) ** 2) ** 0.5
    masked_rms_travel = np.mean((travel[roi_mask]) ** 2) ** 0.5

    pred_centered = pred_travel - np.mean(pred_travel)
    travel_centered = travel - np.mean(travel)
    centered_rmse = np.mean((travel_centered - pred_centered) ** 2) ** 0.5

    masked_pred = pred_travel[roi_mask]
    masked_travel = travel[roi_mask]

    #masked_pred_offset = np.percentile(masked_pred, 1)# - np.percentile(masked_travel, 1)
    masked_pred_offset = np.mean(masked_pred - masked_travel)
    masked_centered_err = (
        masked_pred - masked_pred_offset
    ) - (
        masked_travel
    )
    bin_rmse, bin_rmses, bin_mes = calc_binned_rmse(masked_centered_err, masked_travel)
    corr = np.corrcoef(pred_travel, travel)[0, 1]

    print(
        f"{label} raw RMSE: {raw_rmse:.3f} mm "
        f"(includes arbitrary absolute offset of {best_offset:.3f} mm)"
    )
    print(
        f"{label} best-offset RMSE: {aligned_rmse:.3f} mm, "
        f"centered RMSE: {centered_rmse:.3f} mm, corr: {corr:.4f}, "
        f"masked aligned RMSE: {masked_aligned_rmse:.3f} mm"
    )
    print(
        f"{label} masked centered bin RMSE: {bin_rmse:.3f} mm "
        f"bins: {[round(x, 3) if np.isfinite(x) else np.nan for x in bin_rmses]}, "
        f"bin MEs: {[round(x, 3) if np.isfinite(x) else np.nan for x in bin_mes]}"
    )
    print(
        f"RMS travel: {masked_rms_travel}"
    )
    return {
        "raw_rmse": raw_rmse,
        "aligned_rmse": aligned_rmse,
        "centered_rmse": centered_rmse,
        "corr": corr,
        "offset": best_offset,
        "masked_aligned_rmse": masked_aligned_rmse,
        "bin_rmse": bin_rmse,
        "bin_rmses": bin_rmses,
        "preds": pred_travel,
        "gt": travel,
        "masked_pred_offset": masked_pred_offset
    }

def fit_oracle_model(mag, travel, guess_vec, pred_soft_mg, power_weight, power_prior=1 / 3):
    oracle_model = MagToTravelModel(pred_soft_mg=pred_soft_mg)

    def calculate_res(vec):
        pred_travel = oracle_model.pred_x(mag, vec[:3]) + vec[3]
        power_res = (vec[2] - power_prior) * power_weight
        return np.concatenate([pred_travel - travel, np.array([power_res])])

    offset_guess = np.median(travel - oracle_model.pred_x(mag, guess_vec))
    result = scipy.optimize.least_squares(
        fun=calculate_res,
        x0=np.concatenate([guess_vec, np.array([offset_guess])]),
        method="trf",
        verbose=0,
        max_nfev=1000,
    )
    oracle_model.set_coeffs(result.x[:3])
    return oracle_model, float(result.x[3]), result

def run_case(case_name, b_proj, accel_proj, t, travel, v_gt, a_gt, zv_points, roi_mask, params):
    model = RearMagModel(
        chunking_method="centered_zv",
        **params
        )
    
    input_arr = model.create_training_data(
        mag=b_proj,
        accel=accel_proj,
        t=t,
        idxs=zv_points
    )
    chunks_filt = model.chunks
    #chunks = model.create_chunks(zv_points, b_proj, accel_proj, t)
    #model.prepare_chunks(chunks)

    model.calc_chunks_errors(chunks_filt, travel, v_gt, a_gt)

    #chunks_filt = model.filter_chunks(chunks, model.get_filter_fns())
    print(f"{case_name} training chunks: {len(chunks_filt)}")

    chunk_corrs = [chunk.metrics["b_x_corr"] for chunk in chunks_filt]
    chunk_abs_errs = [np.median(np.abs(chunk.errors["x"])) for chunk in chunks_filt]
    corr_err_corr = scipy.stats.spearmanr(chunk_corrs, chunk_abs_errs).correlation
    print("b-x corr to err correlation", corr_err_corr)
    x_mag_corrs, x_gt_corrs = get_chunk_corrs(chunks_filt, travel)
    if x_mag_corrs:
        print(
            f"{case_name} median corr(x, mag): {np.median(x_mag_corrs):.4f}, "
            f"median corr(x, gt): {np.median(x_gt_corrs):.4f}"
        )

    #input_arr = model.format_chunks_for_fit(chunks_filt)
    result = model.train(input_arr, guess_vec=[0, 1, 1])
    pred_travel = model.model.pred_x(b_proj)
    metrics = evaluate_predictions(pred_travel, travel, case_name, roi_mask)
    oracle_model, oracle_offset, oracle_result = fit_oracle_model(
        mag=b_proj[roi_mask],
        travel=travel[roi_mask],
        guess_vec=result.x.copy(),
        pred_soft_mg=model.pred_soft_mg,
        power_weight=model.power_weight,
    )
    oracle_pred_travel = oracle_model.pred_x(b_proj) + oracle_offset
    oracle_metrics = evaluate_predictions(
        oracle_pred_travel,
        travel,
        f"{case_name} oracle_gt_fit",
        roi_mask,
    )
    metrics["oracle_masked_aligned_rmse"] = oracle_metrics["masked_aligned_rmse"]
    metrics["oracle_preds"] = oracle_metrics["preds"]
    print(f"{case_name} coeffs: {result.x}")
    print(f"{case_name} oracle coeffs: {oracle_result.x[:3]}, oracle offset: {oracle_offset}")
    return result, metrics, chunks_filt


def get_chunk_slopes(chunks: list[MagToTravelChunk], mag_all, travel, roi_mask, plot):
    if not chunks:
        print("No chunks available for slope summary")
        return

    chunk_med_mags = np.asarray([np.median(chunk.mag) for chunk in chunks])
    chunk_proxy_raw = np.asarray([chunk.metrics["dm/dx_median"] for chunk in chunks])
    chunk_proxy_scaled = []
    chunk_reg_slopes = []
    for chunk in chunks:
        dt_sample = np.median(np.diff(chunk.t)) if len(chunk.t) > 1 else np.nan
        chunk_proxy_scaled.append(chunk.metrics["dm/dx_median"] / max(dt_sample, 1e-9))
        chunk_reg_slopes.append(fit_regression_slope(chunk.x, chunk.mag))
    chunk_proxy_scaled = np.asarray(chunk_proxy_scaled)
    chunk_reg_slopes = np.asarray(chunk_reg_slopes)

    roi_mask = np.asarray(roi_mask, dtype=bool)
    mag_roi = np.asarray(mag_all, dtype=float)[roi_mask]
    travel_roi = np.asarray(travel, dtype=float)[roi_mask]

    n_bins = 5
    hist_min = np.percentile(chunk_med_mags, 1)
    hist_max = np.percentile(chunk_med_mags, 99)
    bin_size = (hist_max - hist_min) / n_bins

    bin_centers = []
    proxy_raw_bins = []
    proxy_scaled_bins = []
    chunk_reg_bins = []
    gt_reg_bins = []
    chunk_counts = []
    gt_counts = []
    for i in range(n_bins):
        bin_min = hist_min + (i * bin_size)
        bin_max = bin_min + bin_size
        bin_mask = (bin_min <= chunk_med_mags) & (chunk_med_mags < bin_max)
        gt_bin_mask = (bin_min <= mag_roi) & (mag_roi < bin_max)

        center = bin_min + 0.5 * bin_size
        bin_centers.append(center)
        chunk_counts.append(int(np.sum(bin_mask)))
        gt_counts.append(int(np.sum(gt_bin_mask)))
        proxy_raw_bins.append(float(np.nanmedian(chunk_proxy_raw[bin_mask])) if np.any(bin_mask) else np.nan)
        proxy_scaled_bins.append(float(np.nanmedian(chunk_proxy_scaled[bin_mask])) if np.any(bin_mask) else np.nan)
        chunk_reg_bins.append(float(np.nanmedian(chunk_reg_slopes[bin_mask])) if np.any(bin_mask) else np.nan)
        gt_reg_bins.append(fit_regression_slope(travel_roi[gt_bin_mask], mag_roi[gt_bin_mask]))

    print()
    print("Mag-binned slope summary")
    print("proxy_raw is the current metric dm/v; proxy_scaled and slopes are approx dmag/dx (mG/mm)")
    print(
        f"{'bin':>3} {'center':>9} {'n_chunk':>8} {'n_gt':>6} "
        f"{'proxy_raw':>11} {'proxy_scaled':>13} {'chunk_reg':>11} {'gt_reg':>11}"
    )
    for i, (center, n_chunk, n_gt, proxy_raw, proxy_scaled, chunk_reg, gt_reg) in enumerate(
        zip(
            bin_centers,
            chunk_counts,
            gt_counts,
            proxy_raw_bins,
            proxy_scaled_bins,
            chunk_reg_bins,
            gt_reg_bins,
        )
    ):
        print(
            f"{i:>3} {center:>9.1f} {n_chunk:>8} {n_gt:>6} "
            f"{proxy_raw:>11.4f} {proxy_scaled:>13.3f} {chunk_reg:>11.3f} {gt_reg:>11.3f}"
        )

    chunk_reg_bins = np.asarray(chunk_reg_bins)
    gt_reg_bins = np.asarray(gt_reg_bins)
    proxy_scaled_bins = np.asarray(proxy_scaled_bins)
    valid = np.isfinite(chunk_reg_bins) & np.isfinite(gt_reg_bins)
    proxy_valid = np.isfinite(proxy_scaled_bins) & np.isfinite(gt_reg_bins)
    if np.sum(valid) >= 2:
        chunk_gt_corr = scipy.stats.spearmanr(chunk_reg_bins[valid], gt_reg_bins[valid]).correlation
        print(f"Bin Spearman chunk_reg vs gt_reg: {chunk_gt_corr:.3f}")
    if np.sum(proxy_valid) >= 2:
        proxy_gt_corr = scipy.stats.spearmanr(proxy_scaled_bins[proxy_valid], gt_reg_bins[proxy_valid]).correlation
        print(f"Bin Spearman proxy_scaled vs gt_reg: {proxy_gt_corr:.3f}")

    first_last_valid = np.isfinite(chunk_reg_bins[[0, -1]]) & np.isfinite(gt_reg_bins[[0, -1]])
    if np.all(first_last_valid):
        chunk_ratio = abs(chunk_reg_bins[-1]) / max(abs(chunk_reg_bins[0]), 1e-9)
        gt_ratio = abs(gt_reg_bins[-1]) / max(abs(gt_reg_bins[0]), 1e-9)
        print(f"First->last |slope| ratio, chunk_reg: {chunk_ratio:.3f}, gt_reg: {gt_ratio:.3f}")

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(range(n_bins), proxy_scaled_bins, marker="o", label="proxy_scaled")
        plt.plot(range(n_bins), chunk_reg_bins, marker="o", label="chunk_reg")
        plt.plot(range(n_bins), gt_reg_bins, marker="o", label="gt_reg")
        plt.grid()
        plt.legend()
        plt.xlabel("Mag bin")
        plt.ylabel("Slope (mG/mm)")
        plt.title("Mag-binned chunk and GT slopes")
        plt.show()


def main():
    args = parse_args()
    log_filename = args.log_filename
    a_hp_proj, b_proj, t, travel, v_gt, a_gt, zv_points, roi_mask = load_ws(log_filename)
    results = []
    for case_name, params in (
        ("dm_dx_thresh 0", {"dm_dx_thresh": 0}),
        ("dm_dx_thresh 0.05", {"dm_dx_thresh": 0.05}),
        #("x0 1", 1),
#       ("negative_accel_sign", -a_hp_proj),
    ):
        _, metrics, chunks_filt = run_case(case_name, b_proj, a_hp_proj, t, travel, v_gt, a_gt, zv_points, roi_mask, params=params)
        results.append((case_name, metrics))

    get_chunk_slopes(chunks_filt, b_proj, travel, roi_mask, args.plot)

    if args.plot:
        plt.figure(figsize=(12, 6))
        plt.scatter(b_proj, travel, label="travel", alpha=0.1)
        for case_name, metrics in results:
            preds = metrics["preds"]
            preds_centered = preds - metrics["masked_pred_offset"]
            plt.scatter(b_proj, preds_centered, label=case_name)
            oracle_preds = metrics.get("oracle_preds")
            if oracle_preds is not None:
                plt.scatter(b_proj, oracle_preds)
        plt.legend()
        plt.title("Predicted travel vs mag_proj")
        plt.xlabel("mag_proj")
        plt.ylabel("Travel (mm)")
        plt.show()

    best_case = min(results, key=lambda x: x[1]["masked_aligned_rmse"])
    print()
    print(
        "Best sign by masked aligned RMSE:",
        best_case[0],
        f"({best_case[1]['masked_aligned_rmse']:.3f} mm best-offset RMSE)",
    )


if __name__ == "__main__":
    main()
