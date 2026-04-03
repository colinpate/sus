from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple
from scipy.optimize import least_squares

import numpy as np

from classes.sensor_loader import Workspace
from classes.time_series import TimeSeries, ChunkedTimeSeries
from classes.step import Step


def normalize_rows(V, eps=1e-9):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n


@dataclass
class FilterChunkPairs(Step):
    diff_thresh: float = 2
    conf_thresh: float = 0.98
    grav_diff_thresh: float = 1.0

    """Example 'fusion': difference between two aligned signals."""
    def run(self, ws: Workspace) -> None:
        a: ChunkedTimeSeries = ws[self.inputs[0]]
        b: ChunkedTimeSeries = ws[self.inputs[1]]

        jitter_rejects = 0
        diff_rejects = 0

        still_pairs = []
        for i, (chunk_a, chunk_b) in enumerate(zip(a.iter_chunks(), b.iter_chunks())):
            # Calculate the acceleration jitter
            mean_accels = []
            for chunk_i in [chunk_a, chunk_b]:
                chunk_i = chunk_i.x  # (N, 3)
                norm_samples = chunk_i / np.linalg.norm(chunk_i, axis=1, keepdims=True)
                mean_vector = np.mean(norm_samples, axis=0)
                conf = np.linalg.norm(mean_vector)
                if conf < self.conf_thresh:
                    jitter_rejects += 1
                    break
                mean_accel = np.mean(np.linalg.norm(chunk_i, axis=1))
                mean_accels.append(mean_accel)
            else:
                if (max(mean_accels) - min(mean_accels)) > self.diff_thresh:
                    diff_rejects += 1
                    continue
                if max(abs(np.array(mean_accels) - 9.81)) > self.grav_diff_thresh:
                    diff_rejects += 1
                    continue
                still_pairs.append([chunk_a, chunk_b])

        self.get_pair_stats(still_pairs)

        print(f"Accel pose alignment: Rejected {diff_rejects} diff, {jitter_rejects} jitter, remaining {len(still_pairs)}")
        ws[self.outputs[0]] = still_pairs

    def get_pair_stats(self, pairs):
        pair_array = np.asarray([[a.x, b.x] for (a, b) in pairs]) # N_pairs, chunk_idx, chunk_len, xyz
        print(pair_array.shape)
        accel_means = np.mean(pair_array, axis=2)
        print(accel_means.shape)
        accel_means_mags = np.linalg.norm(accel_means, axis=2)
        print(accel_means_mags.shape)
        s1_mags = accel_means_mags[:, 0]
        s2_mags = accel_means_mags[:, 1]
        print(np.mean(s1_mags), np.mean(s2_mags))
        s2_scales = s2_mags / s1_mags
        s2_scale_mean = np.mean(s2_scales, axis=0)
        s2_scale_std = np.std(s2_scales, axis=0)
        print(f"Chunk stats: s2 to s1 scale mean {s2_scale_mean} std {s2_scale_std}")


@dataclass
class FilterColinearPairs(Step):
    colinear_thresh_deg = 10

    """Example 'fusion': difference between two aligned signals."""
    def run(self, ws: Workspace) -> None:
        pairs: List = ws[self.inputs[0]]
        
        # Normalize chunks to get unit direction vectors
        chunks_u = []
        for (chunk_a, _) in pairs:
            a_mean = np.mean(chunk_a.x, axis=0) # Mean across time
            chunks_u.append(a_mean / np.linalg.norm(a_mean)) # convert to unit vector

        a_u = np.asarray(chunks_u) # (N_chunks, 3)

        mean_dir = normalize_rows(np.mean(a_u, axis=0, keepdims=True)).T
        cosang = np.clip(a_u @ mean_dir, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cosang))
        keep = ang_deg > self.colinear_thresh_deg  # tune threshold (e.g. 5–15°)

        filt_pairs = []
        for i in range(len(pairs)):
            if keep[i]:
                filt_pairs.append(pairs[i])

        print(f"Colinear pair removal: Kept {len(filt_pairs)} out of {len(pairs)}")
        
        ws[self.outputs[0]] = filt_pairs
        if len(self.outputs) == 3:
            pair_array = np.asarray([[pair[0].x, pair[1].x] for pair in pairs])
            ws[self.outputs[1]] = pair_array[:, 0, :, :]
            ws[self.outputs[2]] = pair_array[:, 1, :, :]


@dataclass
class RotationFromPairs(Step):
    """Example 'fusion': difference between two aligned signals."""
    def run(self, ws: Workspace) -> None:
        pairs: List = ws[self.inputs[0]]

        A_u = []
        B_u = []
        A_means = []
        B_means = []
        for (chunk_a, chunk_b) in pairs:
            a_mean = np.mean(chunk_a.x, axis=0) # Mean across time
            b_mean = np.mean(chunk_b.x, axis=0)
            A_u.append(a_mean / np.linalg.norm(a_mean)) # convert to unit vector
            B_u.append(b_mean / np.linalg.norm(b_mean))
            A_means.append(a_mean)
            B_means.append(b_mean)

        A_u = np.asarray(A_u)
        B_u = np.asarray(B_u)
        A_means = np.asarray(A_means)
        B_means = np.asarray(B_means)

        weights = self.get_weights(pairs)

        # Get the rotation vector and save to file
        #R_2_to_1 = self.kabsch_rotation(B_u, A_u, weights=weights)
        R_2_to_1 = self.iter_train(A_u, B_u, weights=weights, n_iter=2, percentile=90)

        # Apply to raw sensor2 samples (example for one chunk):
        # s2_in_s1 = (R_2_to_1 @ s2[k].T).T

        #pred = (R_2_to_1 @ B_u.T).T
        #coserr = np.sum(pred * A_u, axis=1)
        #coserr = np.clip(coserr, -1.0, 1.0)
        err_deg = self.get_errors(A_u, B_u, R_2_to_1)
        #err_deg = np.degrees(np.arccos(coserr))
        print("Rotation array", R_2_to_1)
        print("median err (deg):", np.median(err_deg))
        print("p90 err (deg):", np.percentile(err_deg, 90))

        # Save the rotation matrix for later use
        ws[self.outputs[0]] = R_2_to_1
        #self.plot_means(A_means, B_means, R_2_to_1)

    def plot_means(self, A_means, B_means, R_2_to_1):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(18, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(A_means[:, 0], A_means[:, 1], A_means[:, 2], color='r', label='Sensor 1 Means')
        # ax.scatter(B_means[:, 0], B_means[:, 1], B_means[:, 2], color='b', label='Sensor 2 Means')
        B_rotated = B_means#(R_2_to_1 @ B_means.T).T
        # ax.scatter(B_rotated[:, 0], B_rotated[:, 1], B_rotated[:, 2], color='g', label='Sensor 2 Rotated Means')
        # ax.legend()
        plt.subplot(1, 3, 1)
        plt.scatter(B_rotated[:, 0], A_means[:, 0], color='k')
        trend = np.poly1d(np.polyfit(B_rotated[:, 0], A_means[:, 0], 1))
        plt.plot(B_rotated[:, 0], trend(B_rotated[:, 0]), color='r')
        plt.title(trend.coef)
        plt.grid()
        plt.subplot(1, 3, 2)
        plt.scatter(B_rotated[:, 1], A_means[:, 1], color='k')
        trend = np.poly1d(np.polyfit(B_rotated[:, 1], A_means[:, 1], 1))
        plt.plot(B_rotated[:, 1], trend(B_rotated[:, 1]), color='r')
        plt.title(trend.coef)
        plt.grid()
        plt.subplot(1, 3, 3)
        plt.scatter(B_rotated[:, 2], A_means[:, 2], color='k')
        trend = np.poly1d(np.polyfit(B_rotated[:, 2], A_means[:, 2], 1))
        plt.plot(B_rotated[:, 2], trend(B_rotated[:, 2]), color='r')
        plt.title(trend.coef)
        plt.grid()
        plt.show()


    def get_errors(self, A_u, B_u, R):
        errs = []
        for (a_u, b_u) in zip(A_u, B_u):
            b_in_a = R @ b_u
            coserr = np.clip(np.dot(a_u, b_in_a), -1.0, 1.0)
            err_deg = np.degrees(np.arccos(coserr))
            errs.append(err_deg)
        return errs

    def iter_train(self, A_u, B_u, weights, n_iter=10, percentile=95):
        mask = np.ones_like(weights, dtype=bool)
        for i in range(n_iter):
            R = self.kabsch_rotation(B_u[mask], A_u[mask], weights=weights[mask])
            errs = self.get_errors(A_u[mask], B_u[mask], R)
            err_thresh = np.percentile(errs, percentile)
            mask[mask] = np.array(errs) < err_thresh
            print(f"Iter {i}: keeping {np.sum(mask)} pairs with error < {err_thresh:.2f} deg")
            print("median err (deg):", np.median(self.get_errors(A_u, B_u, R)))
        return self.kabsch_rotation(B_u[mask], A_u[mask], weights=weights[mask])

    def get_weights(self, pairs):
        weights = []
        for (chunk_a, chunk_b) in pairs:
            normeda = np.linalg.norm(chunk_a.x, axis=1)  # (N,25)
            normedb = np.linalg.norm(chunk_b.x, axis=1)  # (N,25)
            stda = np.std(normeda) 
            stdb = np.std(normedb)
            stillness = stda + stdb
            weights.append(1.0 / (stillness + 1e-6))
        return np.asarray(weights)

    def kabsch_rotation(self, B, A, weights=None):
        """
        Solve R that maps B -> A (i.e., A ≈ R @ B), using Kabsch/Wahba.
        A, B: (K,3) arrays of corresponding vectors
        weights: (K,) optional nonnegative weights
        Returns: (3,3) rotation matrix with det=+1
        """
        if weights is None:
            w = np.ones((B.shape[0],), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            w = np.maximum(w, 0)

        # Weighted correlation matrix H = sum_k w_k * A_k * B_k^T
        H = (A * w[:, None]).T @ B

        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        # Enforce proper rotation (det=+1), fix reflection if needed
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R
    

@dataclass
class GetRelativeAccel(Step):
    """Rotate accel and calculate difference."""

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        b: TimeSeries = ws[self.inputs[1]]
        rot: np.array = ws[self.inputs[2]]
        
        b_in_a = (rot @ b.x.T).T
        diff = a.x - b_in_a

        ws[self.outputs[0]] = TimeSeries(
            t=b.t,
            x=b_in_a,
            units=b.units,
            frame=a.frame,
            meta={**b.meta, "rotation_applied": True},
        )
        ws[self.outputs[1]] = TimeSeries(
            t=b.t,
            x=diff,
            units=b.units,
            frame=a.frame,
            meta={**b.meta, "rotation_applied": True},
        )


@dataclass
class GetAccelTravelVector(Step):
    chunk_size: int = 10
    accel_threshold = 4.5 # m/s^2
    """Get the primary travel vector"""

    def run(self, ws: Workspace) -> None:
        a: TimeSeries = ws[self.inputs[0]]
        
        # First get rid of DC offset
        offset = np.mean(a.x, axis=0)
        print("Relative acceleration dc offset", offset)
        x = a.x - offset

        # Chunkify it
        chunks = []
        for i in range(0, x.shape[0], self.chunk_size):
            chunk = x[i : i + self.chunk_size, :]
            if chunk.shape[0] == self.chunk_size:  # Only keep full chunks
                chunks.append(chunk)
        
        # Filter chunks by overall magnitude (keep only those with significant motion)
        good_chunks = []
        for chunk in chunks:
            chunk_net_magnitude = np.linalg.norm(np.mean(chunk, axis=0))
            if chunk_net_magnitude > self.accel_threshold:
                if np.mean(chunk, axis=0)[0] > 0: # Only keep chunks with net positive X acceleration
                    good_chunks.append(chunk)
        print("Accel travel vector:", len(good_chunks), "interesting chunks found")
        good_chunks = np.array(good_chunks)

        chunk_means = np.mean(good_chunks, axis=1)
        chunk_mags = np.linalg.norm(chunk_means, axis=1)

        travel_vector = np.mean(chunk_means, axis=0) / np.linalg.norm(np.mean(chunk_means, axis=0))
        print("Accel travel vector:", travel_vector)
        
        travel_unit_vector = travel_vector / np.linalg.norm(travel_vector)
        ws[self.outputs[0]] = travel_unit_vector
        
        scatter_points = np.concat((np.reshape(chunk_mags, (-1, 1)), chunk_means), axis=1)
        ws[self.outputs[1]] = scatter_points

@dataclass
class ProjectAccel(Step):
    """Project accel onto the travel vector"""

    def run(self, ws: Workspace) -> None:
        travel_vector = ws[self.inputs[0]]
        a = ws[self.inputs[1]]

        # Project accel onto travel vector
        a_proj = a.x @ travel_vector

        print("Mean projected acceleration:", np.mean(a_proj))

        ws[self.outputs[0]] = TimeSeries(
            t=a.t,
            x=a_proj,
            units=a.units,
            frame=a.frame,
            meta={**a.meta, "travel_vector": travel_vector},
        )


class CorrectStaticOffset(Step):
    """Calculate and remove static offset from accel signal"""
    only_x = True

    def run(self, ws: Workspace) -> None:
        chunks: np.ndarray = ws[self.inputs[0]]
        accel: TimeSeries = ws[self.inputs[1]]
        g = 9.81

        samples = np.mean(chunks, axis=1) # Convert to N, 3

        samples = np.asarray(samples, dtype=float)

        def residuals(b):
            return np.linalg.norm(samples - b, axis=1) - g

        result = least_squares(residuals, x0=np.zeros(3))

        # A = np.hstack([2 * samples, np.ones((len(samples), 1))])
        # y = np.sum(samples**2, axis=1)

        # x, *_ = np.linalg.lstsq(A, y, rcond=None)
        # bias = x[:3]

        if self.only_x:
            bias = np.array([result.x[0], 0, 0])
        else:
            bias = result.x

        corrected_accel = accel.x - bias
        corrected_chunks = chunks - bias

        print(f"Correcting accel offset by {bias}, residual error {np.mean(np.linalg.norm(samples - bias, axis=1) - g):.2f} m/s^2")

        ws[self.outputs[0]] = corrected_chunks
        ws[self.outputs[1]] = TimeSeries(
            t=accel.t,
            x=corrected_accel,
            units=accel.units,
            frame=accel.frame,
            meta={**accel.meta},
        )


@dataclass
class GetAccelError(Step):
    """Calculate the error between accel and the travel"""
    threshold = 0.5

    def run(self, ws: Workspace) -> None:
        a_proj_ts = ws[self.inputs[0]]
        travel_ts = ws[self.inputs[1]]

        travel = travel_ts.x[:, 0]
        a_proj = a_proj_ts.x[:, 0]
        t = travel_ts.t

        dt_s = np.diff(t, prepend=t[0]-0.01)
        v = np.gradient(travel, t, edge_order = 2)
        a_gt = np.gradient(v, t, edge_order = 2) / 1000.0

        error = a_proj - a_gt
        ratio_error = error / (a_gt + 1e-6)
        error = error[abs(a_gt) > self.threshold]  # Only evaluate on parts where we have significant travel acceleration
        ratio_error = ratio_error[abs(a_gt) > self.threshold]

        print(f"RMSE error (m/s^2): {np.sqrt(np.mean(error**2)):.2f}, MAE error (m/s^2): {np.mean(np.abs(error)):.2f}, Mean error (m/s^2): {np.mean(error):.2f}")
        print(f"RMSE ratio error: {np.sqrt(np.mean(ratio_error**2)):.2f}, MAE ratio error: {np.mean(np.abs(ratio_error)):.2f}, Mean ratio error: {np.mean(ratio_error):.2f}")
