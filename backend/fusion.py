from dataclasses import dataclass

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step
from mag_to_travel_model_core import MagToTravelModelCore

import matplotlib.pyplot as plt

def print_err_stats(x, gt, center=False, prefix=""):
    if center:
        x = x.copy() - np.mean(x)
        gt = gt.copy() - np.mean(gt)
    error = x - gt
    rmse = np.mean(error ** 2) ** 0.5
    mae = np.mean(abs(error))
    me = np.mean(error)
    print(f"{prefix} RMSE: {rmse:.3f}, MAE: {mae:.3f}, ME: {me:.3f}")
    return rmse, mae, me


@dataclass
class GetMagToTravelModel(Step, MagToTravelModelCore):
    ref_zero_percentile: float = 8.0
    ref_neg_fallback_max_pct: float = 0.08
    ref_fallback_accel_quantile: float = 70.0
    apply_ref_point: bool = True

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        travel_ts: TimeSeries = ws[self.inputs[2]]
        mask_ts: np.ndarray = ws[self.inputs[3]]
        idxs: np.ndarray = ws[self.inputs[4]]
        ref_point: np.ndarray = ws[self.inputs[5]]
        mag_baseline: float = ws[self.inputs[6]]

        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        travel = travel_ts.x[:, 0]
        mag_proj_bad_mask = mask_ts.x.flatten().astype(bool)
        t = mag_ts.t
        baseline_min_mag = mag_baseline[0]

        training_data = self.create_training_data(
            mag=mag,
            accel=accel,
            train_mask=mag_proj_bad_mask,
            t=t,
            baseline_min_mag=baseline_min_mag,
            idxs=idxs
        )

        result = self.train(training_data)
        x0, y_scale, power = result.x[0], result.x[1], result.x[2]

        x_preds = self.model.pred_x(mag)
        
        if self.apply_ref_point:
            ref_fallback_mask = self.build_ref_fallback_mask(accel, mag_proj_bad_mask)
            x_preds_adj = self.adjust_with_ref_point(
                x_preds, 
                ref_point[0], 
                ref_point[1], 
                mag, 
                ref_fallback_mask
            )
        else:
            x_preds_adj = x_preds

        ws[self.outputs[0]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )
        ws[self.outputs[1]] = TimeSeries(
            t=accel_ts.t,
            x=x_preds_adj,
            units="mm",
            frame=accel_ts.frame,
            meta={**accel_ts.meta},
        )
        scatter_points = np.array([mag, travel, x_preds_adj]).T
        ws[self.outputs[2]] = scatter_points
        ws[self.outputs[3]] = np.array([x0, y_scale, power])

    def build_ref_fallback_mask(self, accel: np.ndarray, mag_proj_bad_mask: np.ndarray) -> np.ndarray:
        accel = np.asarray(accel, dtype=float).reshape(-1)
        mag_proj_bad_mask = np.asarray(mag_proj_bad_mask, dtype=bool).reshape(-1)
        accel_abs = np.abs(accel)
        finite_mask = np.isfinite(accel_abs)
        candidate_mask = finite_mask & ~mag_proj_bad_mask
        if not np.any(candidate_mask):
            return np.zeros_like(accel_abs, dtype=bool)

        accel_thresh = float(np.percentile(accel_abs[candidate_mask], self.ref_fallback_accel_quantile))
        motion_mask = candidate_mask & (accel_abs > accel_thresh)
        if not np.any(motion_mask):
            return np.zeros_like(accel_abs, dtype=bool)
        return motion_mask

    def adjust_with_ref_point(self, x_preds, ref_x, ref_mag, mag=None, active_mask=None):
        ref_x_pred = self.model.pred_x(ref_mag)
        offset = - ref_x_pred + ref_x
        x_preds_ref = x_preds + offset

        if mag is not None and active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if np.any(active_mask):
                neg_pct = float(np.mean(x_preds_ref[active_mask] < 0))
                print(
                    "Ref-point fallback check: {:.1f}% of motion-mask samples have negative predicted travel".format(
                        neg_pct * 100
                    )
                )
                if neg_pct > self.ref_neg_fallback_max_pct:
                    zero_mag = float(np.percentile(mag, self.ref_zero_percentile))
                    zero_offset = -float(self.model.pred_x(zero_mag))
                    if zero_offset > offset:
                        print(
                            f"Ref-point fallback: neg_pct={neg_pct * 100:.1f}% exceeds {self.ref_neg_fallback_max_pct * 100:.1f}%, "
                            f"switching offset from {offset:.1f} to {zero_offset:.1f} using mag p{self.ref_zero_percentile:.0f}={zero_mag:.1f}"
                        )
                        offset = zero_offset
                        x_preds_ref = x_preds + offset

        return x_preds_ref

    def get_fit_chunk_weights(self, input_arr):
        if input_arr.shape[0] == 0:
            return np.array([])

        if self.fit_balance_mode == "max_mag":
            rep_mag = np.max(input_arr[:, 0, :], axis=1)
        elif self.fit_balance_mode == "mean_mag":
            rep_mag = np.mean(input_arr[:, 0, :], axis=1)
        else:
            rep_mag = input_arr[:, 0, 0]

        rep_mag = np.asarray(rep_mag, dtype=float)
        n_bins = int(np.clip(self.fit_balance_bins, 1, len(rep_mag)))
        if n_bins <= 1 or np.allclose(rep_mag, rep_mag[0]):
            return np.ones_like(rep_mag)

        edges = np.linspace(np.min(rep_mag), np.max(rep_mag), n_bins + 1)
        bin_idx = np.digitize(rep_mag, edges[1:-1], right=False)
        counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        weights = 1.0 / np.maximum(counts[bin_idx], 100)

        # Normalize so the average chunk keeps about unit weight.
        weights *= len(weights) / np.sum(weights)

        print(
            "Balanced fit chunk counts by mag bin:",
            counts.astype(int),
            "rep mag percentiles:",
            np.percentile(rep_mag, [0, 25, 50, 75, 100]),
            "edges:",
            edges,
        )
        return weights


@dataclass
class GetErrorStats(Step):
    """ Get error stats for mag to travel model """
    gt_thresh: float = 0

    def run(self, ws: Workspace) -> None:
        preds_ts: TimeSeries = ws[self.inputs[0]]
        gt_ts: TimeSeries = ws[self.inputs[1]]
        mask_in: np.ndarray | None = None
        if len(self.inputs) > 2:
            mask_in = ws[self.inputs[2]]

        preds = preds_ts.x[:, 0]
        gt = gt_ts.x[:, 0]

        mask = gt > self.gt_thresh
        if mask_in is not None:
            mask *= mask_in.flatten()
            print(f"Calculating error stats with mask, using {np.sum(mask_in)/len(mask)*100:.1f}% samples")
        preds_masked = preds[mask]
        gt_masked = gt[mask]

        print_err_stats(preds_masked, gt_masked, prefix=f"Thresh (> {self.gt_thresh:.1f} mm) (centered)", center=True)
        print_err_stats(preds_masked, gt_masked, prefix=f"Thresh (> {self.gt_thresh:.1f} mm)")
    

@dataclass
class GetMagTravelRefPoint(Step):
    """Find chunks where we can be pretty sure about travel and use this to set up a static mag to travel reference point"""
    bump_mag_min: float = 1000 # mG
    still_a_max: float = 1000 # mm/s^2
    bump_dx_min: int = 20

    still_len_s: float = 0.1 # seconds
    bump_len_s: float = 0.3 # seconds
    stride_s: float = 0.05 # seconds
    skips: int = 3 # number of following strides to skip if we find a good one, prevents repeats

    ref_mag_range: float = 2000
    min_ref_mag: float = 2000

    debug: bool = False

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag_baseline: float = ws[self.inputs[2]][0]
        gt_x_ts: TimeSeries | None = ws.get(self.inputs[3])
        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        t = mag_ts.t
        dt_s = np.diff(t, prepend=t[0]-0.01)

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        bump_len = int(self.bump_len_s * mag_ts.meta["fs_hz"])
        stride = int(self.stride_s * mag_ts.meta["fs_hz"])

        mag_chunks, a_intint_chunks, _, gt_x_chunks = self.find_chunks(
            accel, 
            mag, 
            gt_x_ts.x if gt_x_ts is not None else None,
            dt_s, 
            still_len, 
            bump_len, 
            stride, 
            mag_baseline
        )
        if len(mag_chunks):
            mag_maxes = [np.max(mag_chunk) for mag_chunk in mag_chunks]
            print("Max mags in chunks:", np.percentile(mag_maxes, 25), np.percentile(mag_maxes, 50), np.percentile(mag_maxes, 75))
        abs_pos_ref_x, abs_pos_ref_mag = self.get_abs_pos_ref(mag_chunks, a_intint_chunks, mag_baseline, gt_x_chunks)
        print(f"Absolute position reference point: x={abs_pos_ref_x:.1f} mm, mag={abs_pos_ref_mag:.1f} mG")

        ws[self.outputs[0]] = np.array([abs_pos_ref_x, abs_pos_ref_mag])

    def find_chunks(self, accel, mag, gt_x, dt_s, still_len, bump_len, stride, still_mag_max):
        # Find the chunks
        a_mms = accel * 1000
        still_slice = slice(0, still_len)
        bump_slice = slice(still_len, still_len + bump_len)
        chunk_len = still_len + bump_len

        slices = []
        a_intint_chunks = []
        mag_chunks = []
        if gt_x is not None:
            gt_x_chunks = []
        else:
            gt_x_chunks = None
        i = 0
        skip = 0
        for i in range(0, a_mms.shape[0] - chunk_len, stride):
            if skip > 0:
                skip -= 1
                continue

            chunk_r = slice(i, i+chunk_len)
            chunk_l = slice(i+chunk_len, i, -1)
            chunks = [chunk_r, chunk_l]
            
            for chunk_i in chunks: 
                mag_still = mag[chunk_i][still_slice]
                a_still = a_mms[chunk_i][still_slice]

                a_bump = a_mms[chunk_i][bump_slice]
                dt_bump = dt_s[chunk_i][bump_slice]
                mag_bump = mag[chunk_i][bump_slice]

                mag_still_mean = np.mean(mag_still)

                if np.mean(mag_still) > still_mag_max:
                    continue
                if max(abs(a_still)) > self.still_a_max:
                    continue
                if max(mag_bump) < mag_still_mean + self.bump_mag_min:
                    continue
                
                a_int = np.cumsum(a_bump * dt_bump)
                a_intint = np.cumsum(a_int * dt_bump)

                if max(a_intint) < self.bump_dx_min:
                    continue

                skip = self.skips

                a_intint_chunks.append(a_intint)
                mag_chunks.append(mag_bump)
                slices.append(chunk_i)
                if gt_x is not None:
                    gt_x_chunks.append(gt_x[chunk_i][bump_slice])
        
        if len(a_intint_chunks) == 0:
            print("No chunks found")
        else:
            print("Calibration chunks:", len(a_intint_chunks), "chunks,", len(a_intint_chunks[0]), "samples per chunk")

        return mag_chunks, a_intint_chunks, slices, gt_x_chunks

    def get_abs_pos_ref(self, mag_chunks, a_intint_chunks, mag_baseline, gt_x_chunks=None):
        if len(mag_chunks) == 0:
            print("No calibration chunks found, cannot determine absolute position reference point, returning 0 and mag baseline + min ref mag")
            return 0, mag_baseline + self.min_ref_mag
        x_points = np.concatenate(a_intint_chunks)
        mag_points = np.concatenate(mag_chunks)
        print("Absolute position reference input points", x_points.shape[0])

        mag_center = max(mag_baseline + self.min_ref_mag, np.median(mag_points))
        center_range = self.ref_mag_range / 2
        thresh_mask = (mag_points > mag_center - center_range) & (mag_points < mag_center + center_range)
        print(f"Using {np.sum(thresh_mask)} points within mag range {mag_center - center_range} to {mag_center + center_range} for absolute position reference stats")
        abs_pos_ref_x = np.median(x_points[thresh_mask])
        abs_pos_ref_mag = np.median(mag_points[thresh_mask])
        print(f"X STD: {np.std(x_points[thresh_mask]):.1f} Mag STD: {np.std(mag_points[thresh_mask]):.1f}")

        if gt_x_chunks is not None:
            gt_x_points = np.concatenate(gt_x_chunks)
            abs_pos_ref_error = abs_pos_ref_x - np.median(gt_x_points[thresh_mask])
            #self.plot_points(x_points, mag_points, gt_x_points)
            print(f"Absolute position reference error compared to GT: {abs_pos_ref_error:.1f} mm")

        return abs_pos_ref_x, abs_pos_ref_mag
    
    def plot_points(self, x_points, mag_points, gt_x_points=None):
        plt.figure(figsize=(10, 6))
        if gt_x_points is not None:
            plt.scatter(mag_points, gt_x_points, alpha=0.5, label="GT x points")
        plt.scatter(mag_points, x_points, alpha=0.5, label="Calibration points")
        plt.xlabel("Mag (mG)")
        plt.ylabel("Integrated accel (mm)")
        plt.title("Calibration points for absolute position reference")
        plt.legend()
        plt.grid()
        plt.show()


class GetMagBaseline(Step):
    """Find the mag baseline by looking at still regions and taking the median + std"""
    still_len_s: float = 0.1 # seconds
    still_a_max: float = 1000 # mm/s^2

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = int(self.still_len_s * mag_ts.meta["fs_hz"])
        a_mms = accel * 1000
        still_mags = []
        for i in range(0, mag.shape[0] - still_len, still_len):
            mag_chunk = mag[i:i+still_len]
            a_chunk = a_mms[i:i+still_len]
            if max(abs(a_chunk)) < self.still_a_max:
                still_mags.append(mag_chunk)

        mag_baseline = np.median(still_mags) + np.std(still_mags)
        print("Mag baseline", mag_baseline, "std", np.std(still_mags))
        ws[self.outputs[0]] = np.array([mag_baseline])
