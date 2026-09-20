from dataclasses import dataclass

import numpy as np

from angle_corruption import project_mask_to_timeline
from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries
from classes.step import Step
from mag_to_travel_model_core import MagToTravelModel, MagToTravelModelCore
from rear_mag_model import RearMagModel

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
    ref_max_offset_delta_mm: float | None = None
    ref_max_out_of_range_pct: float = 0.08
    ref_min_travel_mm: float = 0.0
    ref_max_travel_mm: float = 200.0

    def run(self, ws: Workspace) -> None:
        mag_baseline = ws[self.inputs[4]]

        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mask_ts: np.ndarray = ws[self.inputs[2]]
        idxs: np.ndarray = ws[self.inputs[3]]

        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
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
        ws[self.outputs[2]] = np.array([x0, y_scale, power])
        if len(self.outputs) > 3:
            # The reference adjustment is an additive constant. Expose it so
            # downstream models do not have to reconstruct calibration state
            # from two complete prediction arrays.
            scalar_offset_mm = float(np.median(x_preds_adj - x_preds))
            ws[self.outputs[3]] = np.array([scalar_offset_mm])

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

    def adjust_with_ref_point(
        self,
        x_preds,
        ref_x,
        ref_mag,
        mag=None,
        active_mask=None,
        *,
        max_offset_delta_mm: float | None = None,
    ):
        x_preds = np.asarray(x_preds, dtype=float)
        finite_mag = None if mag is None else np.asarray(mag, dtype=float)
        if finite_mag is not None:
            finite_mag = finite_mag[np.isfinite(finite_mag)]

        zero_mag = None
        zero_offset = None
        if finite_mag is not None and finite_mag.size:
            zero_mag = float(np.percentile(finite_mag, self.ref_zero_percentile))
            zero_offset = -float(self.model.pred_x(zero_mag))

        ref_values = np.asarray([ref_x, ref_mag], dtype=float)
        if not np.all(np.isfinite(ref_values)):
            if zero_offset is None:
                raise ValueError(
                    "Absolute travel reference is non-finite and no finite magnetic samples "
                    "are available for a fallback."
                )
            print("Ref-point fallback: absolute reference contains non-finite values")
            return x_preds + zero_offset

        ref_x_pred = float(self.model.pred_x(ref_mag))
        offset = -ref_x_pred + float(ref_x)
        if not np.isfinite(offset):
            if zero_offset is None:
                raise ValueError("Absolute travel reference produced a non-finite offset")
            print("Ref-point fallback: absolute reference produced a non-finite offset")
            return x_preds + zero_offset

        fallback_reasons = []
        if (
            zero_offset is not None
            and max_offset_delta_mm is not None
            and abs(offset - zero_offset) > max_offset_delta_mm
        ):
            fallback_reasons.append(
                f"reference offset differs from the data-driven zero by "
                f"{abs(offset - zero_offset):.1f} mm"
            )

        x_preds_ref = x_preds + offset
        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if active_mask.shape != x_preds.shape:
                raise ValueError(
                    f"Reference fallback mask shape differs from predictions: "
                    f"{active_mask.shape} vs {x_preds.shape}"
                )
            valid_active = active_mask & np.isfinite(x_preds_ref)
            if np.any(valid_active):
                active_preds = x_preds_ref[valid_active]
                neg_pct = float(np.mean(active_preds < self.ref_min_travel_mm))
                high_pct = float(np.mean(active_preds > self.ref_max_travel_mm))
                print(
                    "Ref-point fallback check: {:.1f}% below {:.0f} mm, {:.1f}% above {:.0f} mm "
                    "in motion-mask samples".format(
                        neg_pct * 100,
                        self.ref_min_travel_mm,
                        high_pct * 100,
                        self.ref_max_travel_mm,
                    )
                )
                if neg_pct > self.ref_neg_fallback_max_pct:
                    fallback_reasons.append(
                        f"{neg_pct * 100:.1f}% of motion-mask predictions are below "
                        f"{self.ref_min_travel_mm:.0f} mm"
                    )
                if high_pct > self.ref_max_out_of_range_pct:
                    fallback_reasons.append(
                        f"{high_pct * 100:.1f}% of motion-mask predictions are above "
                        f"{self.ref_max_travel_mm:.0f} mm"
                    )

        if fallback_reasons and zero_offset is not None:
            print(
                "Ref-point fallback: "
                + "; ".join(fallback_reasons)
                + f"; switching offset from {offset:.1f} to {zero_offset:.1f} "
                + f"using mag p{self.ref_zero_percentile:.0f}={zero_mag:.1f}"
            )
            return x_preds + zero_offset

        return x_preds_ref


@dataclass
class ApplyMagTravelRefPoint(Step):
    """Apply an absolute reference to an already inferred magnetic travel signal."""

    ref_zero_percentile: float = 8.0
    ref_neg_fallback_max_pct: float = 0.08
    ref_fallback_accel_quantile: float = 70.0
    ref_max_offset_delta_mm: float | None = None
    ref_max_out_of_range_pct: float = 0.08
    ref_min_travel_mm: float = 0.0
    ref_max_travel_mm: float = 200.0
    pred_soft_mg: float = 50.0

    def run(self, ws: Workspace) -> None:
        if len(self.inputs) != 6:
            raise ValueError(
                "ApplyMagTravelRefPoint expects magnetic travel, scalar mag, "
                "accel, bad-mag mask, reference point, and model coefficients"
            )
        if len(self.outputs) != 1:
            raise ValueError("ApplyMagTravelRefPoint requires one adjusted-travel output")

        travel_ts: TimeSeries = ws[self.inputs[0]]
        mag_ts: TimeSeries = ws[self.inputs[1]]
        accel_ts: TimeSeries = ws[self.inputs[2]]
        bad_mask_ts: TimeSeries = ws[self.inputs[3]]
        ref_point = np.asarray(ws[self.inputs[4]], dtype=float).reshape(-1)
        coefficients = np.asarray(ws[self.inputs[5]], dtype=float).reshape(-1)

        if ref_point.shape != (2,):
            raise ValueError("Magnetic travel reference must contain [travel_mm, magnitude_mG]")
        if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Magnetic travel model coefficients must contain three finite values")

        lengths = {
            len(travel_ts.t),
            len(mag_ts.t),
            len(accel_ts.t),
            len(bad_mask_ts.t),
        }
        if len(lengths) != 1:
            raise ValueError("Reference-application inputs must be index-aligned")

        # Reuse the established reference/fallback policy, but apply the
        # resulting constant only after nuisance correction. The scalar model
        # remains the source of the mag-to-travel coordinate at the reference.
        adjuster = GetMagToTravelModel(
            name=self.name,
            inputs=(),
            outputs=(),
            ref_zero_percentile=self.ref_zero_percentile,
            ref_neg_fallback_max_pct=self.ref_neg_fallback_max_pct,
            ref_fallback_accel_quantile=self.ref_fallback_accel_quantile,
            ref_max_offset_delta_mm=self.ref_max_offset_delta_mm,
            ref_max_out_of_range_pct=self.ref_max_out_of_range_pct,
            ref_min_travel_mm=self.ref_min_travel_mm,
            ref_max_travel_mm=self.ref_max_travel_mm,
            pred_soft_mg=self.pred_soft_mg,
        )
        adjuster.model = MagToTravelModel(
            pred_soft_mg=self.pred_soft_mg,
            coeffs=coefficients,
        )

        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        bad_mask = bad_mask_ts.x[:, 0].astype(bool)
        fallback_mask = adjuster.build_ref_fallback_mask(accel, bad_mask)
        adjusted = adjuster.adjust_with_ref_point(
            travel_ts.x[:, 0],
            ref_point[0],
            ref_point[1],
            mag,
            fallback_mask,
            max_offset_delta_mm=self.param(
                ws,
                "ref_max_offset_delta_mm",
                self.ref_max_offset_delta_mm,
            ),
        )

        ws[self.outputs[0]] = TimeSeries(
            t=travel_ts.t,
            x=adjusted,
            units=travel_ts.units,
            frame=travel_ts.frame,
            meta={**travel_ts.meta},
        )


@dataclass
class GetRearMagToTravelModel(Step, RearMagModel):
    min_chunk_dt: float = RearMagModel.min_chunk_dt
    max_chunk_dt: float = RearMagModel.max_chunk_dt
    min_chunk_db: float = RearMagModel.min_chunk_db
    pair_mode: str = RearMagModel.pair_mode
    default_chunk_max_dx: float = RearMagModel.default_chunk_max_dx
    max_b_x_corr: float | None = RearMagModel.max_b_x_corr
    min_abs_b_x_corr: float | None = RearMagModel.min_abs_b_x_corr
    min_db_per_dx: float | None = RearMagModel.min_db_per_dx
    zero_travel_percentile: float = 8

    def run(self, ws: Workspace) -> None:
        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        idxs: np.ndarray = ws[self.inputs[2]]

        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]
        t = mag_ts.t

        training_data = self.create_training_data(
            mag=mag,
            accel=accel,
            t=t,
            idxs=idxs
        )

        result = self.train(training_data, guess_vec=[0.1, 250, 1 / 3])
        x0, y_scale, power = result.x[0], result.x[1], result.x[2]
        print(f"Mag to travel model coefficients: {x0:.2f}, {y_scale:.2f}, {power:.3f}")

        x_preds = self.model.pred_x(mag)

        x_preds_adj = x_preds - np.percentile(x_preds, self.zero_travel_percentile)

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
        scatter_points = np.array([mag, x_preds_adj]).T
        ws[self.outputs[2]] = scatter_points
        ws[self.outputs[3]] = np.array([x0, y_scale, power])


@dataclass
class GetErrorStats(Step):
    """ Get error stats for mag to travel model """
    gt_thresh: float | None = None

    def run(self, ws: Workspace) -> None:
        preds_ts: TimeSeries = ws[self.inputs[0]]
        gt_ts: TimeSeries = ws[self.inputs[1]]
        angle_bad_mask_ts = ws.get("angle/bad_mask")
        mask_in: np.ndarray | None = None
        if len(self.inputs) > 2:
            mask_in = ws[self.inputs[2]]

        preds = preds_ts.x[:, 0]
        gt = gt_ts.x[:, 0]
        if preds.shape != gt.shape:
            raise ValueError(f"Prediction and ground-truth shapes differ: {preds.shape} vs {gt.shape}")

        mask = np.isfinite(preds) & np.isfinite(gt)
        if self.gt_thresh is not None:
            mask &= gt > self.gt_thresh
            mask_text = f"Thresh (> {self.gt_thresh:.1f} mm)"
        else:
            mask_text = ""
        if mask_in is not None:
            input_mask = np.asarray(mask_in, dtype=bool).reshape(-1)
            if input_mask.shape != mask.shape:
                raise ValueError(f"Error-stat mask shape differs from ground truth: {input_mask.shape} vs {mask.shape}")
            mask &= input_mask

        if isinstance(angle_bad_mask_ts, TimeSeries):
            angle_bad_mask = project_mask_to_timeline(
                angle_bad_mask_ts.t,
                angle_bad_mask_ts.x[:, 0].astype(bool),
                gt_ts.t,
            )
            excluded_count = int(np.sum(mask & angle_bad_mask))
            if excluded_count:
                candidate_count = int(np.sum(mask))
                mask &= ~angle_bad_mask
                print(
                    "Excluding",
                    f"{excluded_count / candidate_count * 100:.2f}%",
                    "of candidate error samples due to corrupted angle data",
                )

        print(f"Calculating error stats with mask, using {np.mean(mask) * 100:.1f}% samples")
        preds_masked = preds[mask]
        gt_masked = gt[mask]

        print_err_stats(preds_masked, gt_masked, prefix=f"{mask_text} (centered)", center=True)
        print_err_stats(preds_masked, gt_masked, prefix=mask_text)
    

@dataclass
class GetMagTravelRefPoint(Step):
    """Find chunks where we can be pretty sure about travel and use this to set up a static mag to travel reference point"""
    bump_mag_min: float = 1000 # mG
    still_a_max: float = 1000 # mm/s^2
    bump_dx_min: int = 20

    still_len_s: float = 0.1 # seconds
    bump_len_s: float = 0.2 # seconds
    stride_s: float = 0.05 # seconds
    skips: int = 3 # number of following strides to skip if we find a good one, prevents repeats

    ref_mag_range: float = 2000
    min_ref_mag: float = 2000
    min_ref_points: int = 1
    ref_zero_percentile: float = 8.0

    debug: bool = False

    def run(self, ws: Workspace) -> None:
        fixed_reference = self.param(ws, "fixed_reference", None)
        if fixed_reference is not None:
            reference = np.asarray(fixed_reference, dtype=float).reshape(-1)
            if reference.shape != (2,) or not np.all(np.isfinite(reference)):
                raise ValueError("fixed_reference must contain finite [travel_mm, magnitude_mG]")
            print(
                "Using fixed absolute position reference point: "
                f"x={reference[0]:.1f} mm, mag={reference[1]:.1f} mG"
            )
            ws[self.outputs[0]] = reference
            return

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
        still_len = max(
            1,
            int(float(self.param(ws, "still_len_s", self.still_len_s)) * mag_ts.meta["fs_hz"]),
        )
        bump_len = max(
            1,
            int(float(self.param(ws, "bump_len_s", self.bump_len_s)) * mag_ts.meta["fs_hz"]),
        )
        stride = max(
            1,
            int(float(self.param(ws, "stride_s", self.stride_s)) * mag_ts.meta["fs_hz"]),
        )

        mag_chunks, a_intint_chunks, _, gt_x_chunks = self.find_chunks(
            accel, 
            mag, 
            gt_x_ts.x if gt_x_ts is not None else None,
            dt_s, 
            still_len, 
            bump_len, 
            stride, 
            mag_baseline,
            still_a_max=float(self.param(ws, "still_a_max", self.still_a_max)),
            bump_mag_min=float(self.param(ws, "bump_mag_min", self.bump_mag_min)),
            bump_dx_min=float(self.param(ws, "bump_dx_min", self.bump_dx_min)),
            skips=int(self.param(ws, "skips", self.skips)),
        )
        #if len(mag_chunks):
        #    mag_maxes = [np.max(mag_chunk) for mag_chunk in mag_chunks]
        #    print("Max mags in chunks:", np.percentile(mag_maxes, 25), np.percentile(mag_maxes, 50), np.percentile(mag_maxes, 75))
        finite_mag = mag[np.isfinite(mag)]
        if finite_mag.size:
            fallback_ref_mag = float(
                np.percentile(finite_mag, self.param(ws, "ref_zero_percentile", self.ref_zero_percentile))
            )
        else:
            fallback_ref_mag = float(
                mag_baseline + float(self.param(ws, "min_ref_mag", self.min_ref_mag))
            )
        abs_pos_ref_x, abs_pos_ref_mag = self.get_abs_pos_ref(
            mag_chunks,
            a_intint_chunks,
            mag_baseline,
            gt_x_chunks,
            min_ref_points=int(self.param(ws, "min_ref_points", self.min_ref_points)),
            fallback_ref_mag=fallback_ref_mag,
            min_ref_mag=float(self.param(ws, "min_ref_mag", self.min_ref_mag)),
            ref_mag_range=float(self.param(ws, "ref_mag_range", self.ref_mag_range)),
        )
        print(f"Absolute position reference point: x={abs_pos_ref_x:.1f} mm, mag={abs_pos_ref_mag:.1f} mG")

        ws[self.outputs[0]] = np.array([abs_pos_ref_x, abs_pos_ref_mag])

    def find_chunks(
        self,
        accel,
        mag,
        gt_x,
        dt_s,
        still_len,
        bump_len,
        stride,
        still_mag_max,
        *,
        still_a_max: float | None = None,
        bump_mag_min: float | None = None,
        bump_dx_min: float | None = None,
        skips: int | None = None,
    ):
        # Find the chunks
        still_a_max = self.still_a_max if still_a_max is None else still_a_max
        bump_mag_min = self.bump_mag_min if bump_mag_min is None else bump_mag_min
        bump_dx_min = self.bump_dx_min if bump_dx_min is None else bump_dx_min
        skips = self.skips if skips is None else skips
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
                if max(abs(a_still)) > still_a_max:
                    continue
                if max(mag_bump) < mag_still_mean + bump_mag_min:
                    continue
                
                a_int = np.cumsum(a_bump * dt_bump)
                a_intint = np.cumsum(a_int * dt_bump)

                if max(a_intint) < bump_dx_min:
                    continue

                skip = skips

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

    def get_abs_pos_ref(
        self,
        mag_chunks,
        a_intint_chunks,
        mag_baseline,
        gt_x_chunks=None,
        *,
        min_ref_points: int | None = None,
        fallback_ref_mag: float | None = None,
        min_ref_mag: float | None = None,
        ref_mag_range: float | None = None,
    ):
        if min_ref_points is None:
            min_ref_points = self.min_ref_points
        if min_ref_mag is None:
            min_ref_mag = self.min_ref_mag
        if ref_mag_range is None:
            ref_mag_range = self.ref_mag_range
        if fallback_ref_mag is None:
            fallback_ref_mag = float(mag_baseline + min_ref_mag)

        if len(mag_chunks) == 0:
            print(
                "No calibration chunks found; using the data-driven zero-travel "
                "magnetic reference"
            )
            return 0.0, fallback_ref_mag
        x_points = np.concatenate(a_intint_chunks)
        mag_points = np.concatenate(mag_chunks)
        print("Absolute position reference input points", x_points.shape[0])

        mag_center = max(mag_baseline + min_ref_mag, np.median(mag_points))
        center_range = ref_mag_range / 2
        thresh_mask = (
            np.isfinite(x_points)
            & np.isfinite(mag_points)
            & (mag_points > mag_center - center_range)
            & (mag_points < mag_center + center_range)
            #& (x_points > self.bump_dx_min)
        )
        selected_count = int(np.sum(thresh_mask))
        print(f"Using {selected_count} points within mag range {mag_center - center_range} to {mag_center + center_range} for absolute position reference stats")
        if selected_count < min_ref_points:
            print(
                "Absolute position reference is under-supported: "
                f"found {selected_count} valid points, need at least {min_ref_points}; "
                "using the data-driven zero-travel magnetic reference"
            )
            return 0.0, fallback_ref_mag

        abs_pos_ref_x = np.median(x_points[thresh_mask])
        abs_pos_ref_mag = np.median(mag_points[thresh_mask])
        if not np.isfinite(abs_pos_ref_x) or not np.isfinite(abs_pos_ref_mag):
            print(
                "Absolute position reference is non-finite; using the data-driven "
                "zero-travel magnetic reference"
            )
            return 0.0, fallback_ref_mag
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


@dataclass
class GetMagBaseline(Step):
    """Find the mag baseline by looking at still regions and taking the median + std"""
    still_len_s: float = 0.1 # seconds
    still_a_max: float = 1000 # mm/s^2
    fallback_percentile: float = 5.0

    def run(self, ws: Workspace) -> None:
        fixed_baseline = self.param(ws, "fixed_baseline_mG", None)
        if fixed_baseline is not None:
            baseline = float(fixed_baseline)
            if not np.isfinite(baseline):
                raise ValueError("fixed_baseline_mG must be finite")
            print("Using fixed mag baseline", baseline)
            ws[self.outputs[0]] = np.array([baseline])
            return

        mag_ts: TimeSeries = ws[self.inputs[0]]
        accel_ts: TimeSeries = ws[self.inputs[1]]
        mag = mag_ts.x[:, 0]
        accel = accel_ts.x[:, 0]

        assert mag_ts.units == "milli-Gauss"
        assert accel_ts.units == "m/s^2"
        still_len = max(
            1,
            int(float(self.param(ws, "still_len_s", self.still_len_s)) * mag_ts.meta["fs_hz"]),
        )
        still_a_max = float(self.param(ws, "still_a_max", self.still_a_max))
        fallback_percentile = float(
            self.param(ws, "fallback_percentile", self.fallback_percentile)
        )
        a_mms = accel * 1000
        still_mags = []
        for i in range(0, mag.shape[0] - still_len, still_len):
            mag_chunk = mag[i:i+still_len]
            a_chunk = a_mms[i:i+still_len]
            if max(abs(a_chunk)) < still_a_max:
                still_mags.append(mag_chunk)

        finite_mag = mag[np.isfinite(mag)]
        if finite_mag.size == 0:
            raise ValueError("Cannot estimate magnetic baseline without finite samples")
        fallback = float(np.percentile(finite_mag, fallback_percentile))
        if still_mags:
            still_values = np.concatenate(still_mags)
            still_median = float(np.median(still_values))
            still_std = float(np.std(still_values))
            mag_baseline = min(still_median, fallback) + still_std
        else:
            still_std = 0.0
            mag_baseline = fallback
            print("No stationary magnetic windows found; using percentile fallback")
        print("Fallback", fallback_percentile)
        print("Mag baseline", mag_baseline, "std", still_std)
        ws[self.outputs[0]] = np.array([mag_baseline])
