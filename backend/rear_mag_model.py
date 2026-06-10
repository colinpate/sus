import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModel, MagToTravelModelCore

def parse_args():
    parser = ArgumentParser(description="Run rear suspension mag model constructor")
    parser.add_argument("log_filename", type=str, default="log136_rear", help="Name of log file (without .csv extension) to process")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the results")
    return parser.parse_args()

class RearMagModel(MagToTravelModelCore):
    min_chunk_dt: float = 0.1
    max_chunk_dt: float = 0.2
    min_chunk_db: float = 500
    pair_mode: str = "first_valid"
    default_chunk_max_dx: float = 150.0
    max_b_x_corr: float | None = None
    min_abs_b_x_corr: float | None = None
    min_db_per_dx: float | None = None

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

    def create_chunks(self, idxs, mag, acc, t_s):
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

    def get_filter_fns(self):
        filter_fns = [
            self.filter_chunk_dx,
            self.filter_chunk_abs_b_x_corr,
            self.filter_chunk_db_per_dx,
            self.filter_chunk_max_b_x_corr
        ]
        return filter_fns

    def filter_chunk_max_dm_dx(self, chunk: MagToTravelChunk):
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
    
def project_accel(a):
    # Highpass accel and project
    hp_fc_hz=2
    sos_hp = butter(N=2, Wn=hp_fc_hz, btype="high", fs=200, output="sos")
    a_hp = sosfiltfilt(sos_hp, a, axis=0)
    a_hp_norm = np.linalg.norm(a_hp, axis=1)

    mask = (a_hp_norm > 10)
    # dilate mask
    mask = np.convolve(mask, np.ones(200), mode="same") > 0

    # Find main axis of acceleration
    a_hp_m = a_hp[mask]
    accel_axis = np.mean(a_hp_m[a_hp_m[:, 1] < 0], axis=0)
    accel_axis /= np.linalg.norm(accel_axis)
    print("Main accel axis:", accel_axis)
    a_hp_proj = a_hp @ accel_axis
    a_proj = a @ accel_axis
    return a_hp_proj, a_proj

def load_ws(log_filename):
    out_dir = f"backend/run_artifacts/{log_filename}/cache/"
    ws_file = out_dir + "/all.npz"
    ws = np.load(ws_file)
    accel_idx = "2"
    a = ws[f"accel/lphp/proj__x"][:, 0]
    b_proj = ws["mag/proj/lpf__x"][:, 0]
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
        f"RMS travel: {masked_rms_travel}"
    )
    return {
        "raw_rmse": raw_rmse,
        "aligned_rmse": aligned_rmse,
        "centered_rmse": centered_rmse,
        "corr": corr,
        "offset": best_offset,
        "masked_aligned_rmse": masked_aligned_rmse,
        "preds": pred_travel,
        "gt": travel,
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

def run_case(case_name, b_proj, accel_proj, t, travel, v_gt, a_gt, zv_points, roi_mask, x0_weight):
    model = RearMagModel(x0_weight=x0_weight, dm_dx_thresh=None)
    chunks = model.create_chunks(zv_points, b_proj, accel_proj, t)
    model.prepare_chunks(chunks)

    model.calc_chunks_errors(chunks, travel, v_gt, a_gt)

    chunks_filt = model.filter_chunks(chunks, model.get_filter_fns())
    print(f"{case_name} training chunks: {len(chunks_filt)}")

    chunk_corrs = [chunk.metrics["b_x_corr"] for chunk in chunks]
    chunk_abs_errs = [np.median(np.abs(chunk.errors["x"])) for chunk in chunks]
    corr_err_corr = scipy.stats.spearmanr(chunk_corrs, chunk_abs_errs).correlation
    print("b-x corr to err correlation", corr_err_corr)
    x_mag_corrs, x_gt_corrs = get_chunk_corrs(chunks_filt, travel)
    if x_mag_corrs:
        print(
            f"{case_name} median corr(x, mag): {np.median(x_mag_corrs):.4f}, "
            f"median corr(x, gt): {np.median(x_gt_corrs):.4f}"
        )

    input_arr = model.format_chunks_for_fit(chunks_filt)
    result = model.fit_model(input_arr, guess_vec=[0, -1, 1 / 3])
    pred_travel = model.model.pred_x(b_proj)
    metrics = evaluate_predictions(pred_travel, travel, case_name, roi_mask)
    # oracle_model, oracle_offset, oracle_result = fit_oracle_model(
    #     mag=b_proj[roi_mask],
    #     travel=travel[roi_mask],
    #     guess_vec=result.x.copy(),
    #     pred_soft_mg=model.pred_soft_mg,
    #     power_weight=model.power_weight,
    # )
    # oracle_pred_travel = oracle_model.pred_x(b_proj) + oracle_offset
    # oracle_metrics = evaluate_predictions(
    #     oracle_pred_travel,
    #     travel,
    #     f"{case_name} oracle_gt_fit",
    #     roi_mask,
    # )
    # metrics["oracle_masked_aligned_rmse"] = oracle_metrics["masked_aligned_rmse"]
    print(f"{case_name} coeffs: {result.x}")
    # print(f"{case_name} oracle coeffs: {oracle_result.x[:3]}, oracle offset: {oracle_offset}")
    return result, metrics, chunks_filt


def get_chunk_slopes(chunks: list[MagToTravelChunk], plot):
    chunk_med_mags = np.asarray([np.median(chunk.mag) for chunk in chunks])
    chunk_med_slopes = np.asarray([chunk.metrics["dm/dx_median"] for chunk in chunks])
    n_bins = 5
    hist_min = np.percentile(chunk_med_mags, 1)
    hist_max = np.percentile(chunk_med_mags, 99)
    bin_size = (hist_max - hist_min) / n_bins

    bin_slopes = []
    bin_pts = []
    for i in range(n_bins):
        bin_min = hist_min + (i * bin_size)
        bin_max = bin_min + bin_size
        bin_mask = (bin_min <= chunk_med_mags) & (chunk_med_mags < bin_max)
        med_slopes = chunk_med_slopes[bin_mask]
        bin_slope = np.median(med_slopes)
        bin_slopes.append(bin_slope)
        bin_pts.append(int(np.sum(bin_mask)))

    print("Bin points:", bin_pts)
    print("Bin centers:", hist_min + (np.arange(n_bins) + 0.5) * bin_size)
    print("Bin slopes", bin_slopes)
    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(range(n_bins), bin_slopes)
        plt.grid()
        plt.show()


def main():
    args = parse_args()
    log_filename = args.log_filename
    a_hp_proj, b_proj, t, travel, v_gt, a_gt, zv_points, roi_mask = load_ws(log_filename)
    results = []
    for case_name, x0_weight in (
        ("x0 0", 0),
        ("x0 1", 1),
#       ("negative_accel_sign", -a_hp_proj),
    ):
        _, metrics, chunks_filt = run_case(case_name, b_proj, a_hp_proj, t, travel, v_gt, a_gt, zv_points, roi_mask, x0_weight)
        results.append((case_name, metrics))

    get_chunk_slopes(chunks_filt, args.plot)

    if args.plot:
        plt.figure(figsize=(12, 6))
        plt.scatter(b_proj, travel - np.mean(travel), label="travel", alpha=0.1)
        for case_name, metrics in results:
            preds = metrics["preds"]
            preds_centered = preds - np.mean(preds)
            plt.scatter(b_proj, preds_centered, label=case_name)
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
