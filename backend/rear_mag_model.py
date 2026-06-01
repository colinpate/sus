import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt

from mag_to_travel_model_core import MagToTravelChunk, MagToTravelModelCore

class RearMagModel(MagToTravelModelCore):
    min_chunk_dt: float = 0.1
    max_chunk_dt: float = 0.2
    min_chunk_db: float = 500

    def find_zv_pairs(self, idxs: list[int], b_proj: np.ndarray, t: np.ndarray):
        pairs = []
        rejects = {"min_dt": 0, "max_dt": 0, "min_db": 0}
        idx_1 = 0
        while idx_1 < len(idxs) - 1:
            zv_1 = idxs[idx_1]
            for idx_2 in range(idx_1 + 1, len(idxs)):
                zv_2 = idxs[idx_2]
                dt = t[zv_2] - t[zv_1]
                db = b_proj[zv_2] - b_proj[zv_1]
                if dt > self.max_chunk_dt:
                    rejects["max_dt"] += 1
                    idx_1 += 1
                    break
                if dt < self.min_chunk_dt:
                    rejects["min_dt"] += 1
                    continue
                if abs(db) < self.min_chunk_db:
                    rejects["min_db"] += 1
                    continue
                pairs.append((zv_1, zv_2))
                idx_1 = idx_2
                break
            else:
                idx_1 += 1
        return pairs

    def create_chunks(self, idxs, mag, acc, t_s):
        chunks = []
        pairs = self.find_zv_pairs(idxs, mag, t_s)
        for pair in pairs:
            slice_i = slice(pair[0], pair[1])
            raw_acc = acc[slice_i]
            t_chunk = t_s[slice_i]
            # Accel should integrate to zero cuz pairs are zero-v points
            #acc_bias = np.mean(raw_acc)
            acc_bias = scipy.integrate.trapezoid(raw_acc, t_chunk) / t_chunk[-1]
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
            self.filter_chunk_dx
        ]
        return filter_fns
    
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
    a = ws[f"accel/lpf/lis{accel_idx}__x"]
    b_proj = ws["mag/proj/lpf__x"][:, 0]
    t = ws[f"accel/lpf/lis{accel_idx}__t"]
    zv_points = ws["mag_zv_points"]
    travel = ws["travel__x"][:, 0]

    v_gt = np.gradient(travel, t, edge_order = 2)
    a_gt = np.gradient(v_gt, t, edge_order = 2)
    return a, b_proj, t, travel, v_gt, a_gt, zv_points

def main():
    log_filename = "log136_rear"
    a_raw, b_proj, t, travel, v_gt, a_gt, zv_points = load_ws(log_filename)
    a_hp_proj, a_proj = project_accel(a_raw)
    model = RearMagModel()
    chunks = model.create_chunks(zv_points, b_proj, -a_hp_proj, t)
    model.prepare_chunks(chunks)
    model.calc_chunks_errors(chunks, travel, v_gt, a_gt)
    filters = model.get_filter_fns()
    chunks_filt = model.filter_chunks(chunks, filters)
    print("Training chunks:", len(chunks_filt))
    input_arr = model.format_chunks_for_fit(chunks_filt)
    coeffs = model.fit_model(input_arr, guess_vec=[0, -1, 1/3])
    print(coeffs)
    pred_travel = model.model.pred_x(b_proj)
    travel_error = np.mean((travel - pred_travel) ** 2) ** 0.5
    print(f"Predicted travel RMSE: {travel_error}")


if __name__ == "__main__":
    main()