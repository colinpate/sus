from dataclasses import dataclass
import numpy as np
import scipy
from scipy.optimize import least_squares


@dataclass
class MagToTravelChunk:
    v: np.ndarray
    x: np.ndarray
    mag: np.ndarray
    idx: int
    chunk_len: int

    def __post_init__(self):
        self.slice = slice(self.idx-self.chunk_len, self.idx+self.chunk_len)


@dataclass
class MagToTravelModel:
    pred_soft_mg: float
    coeffs: np.ndarray | None = None

    def set_coeffs(self, coeffs: np.ndarray):
        assert coeffs.shape == (3,)
        self.coeffs = coeffs
    
    def pred_x(self, mag_i: np.ndarray | float, coeffs: np.ndarray | None = None):
        if coeffs is None:
            x0, y_scale, power = self.coeffs[0], self.coeffs[1], self.coeffs[2]
        else:
            x0, y_scale, power = coeffs[0], coeffs[1], coeffs[2]
        dx = np.asarray(mag_i, dtype=float) - x0
        soft = (np.abs(dx) + self.pred_soft_mg) ** power - (self.pred_soft_mg ** power)
        return np.sign(dx) * soft * y_scale


@dataclass
class MagToTravelModelCore:
    """ Train a model using least squares  """
    chunk_min_dx: float = 10
    chunk_max_dx: float = 1500
    chunk_len: int = 20
    train_with_mask: bool = False
    bad_thresh: float = 0.5
    pred_soft_mg: float = 50.0
    power_weight: float = 1000.0
    min_mag_relax_min_chunks: int = 50
    retrain_drop_worst_chunk_frac: float = 0.0
    retrain_drop_worst_chunk_min_count: int = 1
    retrain_drop_worst_chunk_min_remaining: int = 25
    model: MagToTravelModel | None = None

    def __post_init__(self):
        self.chunks: list[MagToTravelChunk] = []
        self.stats: dict = {}

    def create_training_data(
            self, 
            mag, 
            accel,
            train_mask,
            t,
            baseline_min_mag,
            idxs
        ):
        if self.train_with_mask:
            print("Training with mask, shape of bad mask", train_mask.shape, "num bad samples", np.sum(train_mask))
            training_mask = train_mask
        else:
            training_mask = np.zeros(mag.shape[0], dtype=bool)

        self.min_mag = baseline_min_mag
        chunks, all_mags = self.get_chunks(idxs, mag, accel, t, training_mask, self.min_mag)
        mag_mins = [np.min(mag_chunk) for mag_chunk in all_mags]
        if mag_mins:
            relax_rank = min(len(mag_mins), self.min_mag_relax_min_chunks)
            relaxed_min_mag = np.sort(mag_mins)[-relax_rank]
        else:
            relaxed_min_mag = np.nan

        use_relaxed_min_mag = (
            np.isfinite(relaxed_min_mag)
            and len(chunks) < self.min_mag_relax_min_chunks
            and relaxed_min_mag < baseline_min_mag
        )
        if use_relaxed_min_mag:
            print(
                "Relaxing min mag from",
                baseline_min_mag,
                "to",
                relaxed_min_mag,
                "initial chunks",
                len(chunks),
            )
            chunks, _ = self.get_chunks(idxs, mag, accel, t, training_mask, relaxed_min_mag)
        else:
            print(
                "Using raw min mag",
                baseline_min_mag,
                "chunks",
                len(chunks),
            )
        self.chunks = chunks

        return self.format_chunks_for_fit(chunks)

    def get_chunks(self, idxs_filt, mag, acc, t_s, mag_proj_bad_mask, min_mag):
        chunk_len = self.chunk_len
        min_dx = self.chunk_min_dx
        max_dx = self.chunk_max_dx
        filt_stats = {"mask": 0, "dx": 0, "dm/dx": 0, "mag": 0}

        chunks = []
        all_mags = []
        for idx in idxs_filt:
            if idx < chunk_len or idx + chunk_len >= len(mag):
                continue
            t_chunk = t_s[idx - chunk_len:idx + chunk_len]
            a_chunk = acc[idx - chunk_len:idx + chunk_len] * 1000
            badmask_chunk = mag_proj_bad_mask[idx - chunk_len:idx + chunk_len]
            if np.mean(badmask_chunk) > self.bad_thresh:
                filt_stats["mask"] += 1
                continue
            v_chunk = scipy.integrate.cumulative_trapezoid(a_chunk, t_chunk, initial=0)
            v_chunk -= v_chunk[chunk_len]
            x_chunk = scipy.integrate.cumulative_trapezoid(v_chunk, t_chunk, initial=0)
            x_chunk -= x_chunk[chunk_len]
            chunk_dx = max(x_chunk) - min(x_chunk)
            if chunk_dx < min_dx or chunk_dx > max_dx:
                filt_stats["dx"] += 1
                continue
            mag_chunk = mag[idx - chunk_len:idx + chunk_len]
            dm_chunk = np.diff(mag_chunk, prepend=mag_chunk[0])
            dm_dx = dm_chunk / (v_chunk + 1e-6)
            if np.median(dm_dx) < 0.05:
                filt_stats["dm/dx"] += 1
                continue
            all_mags.append(mag_chunk)
            if min(mag_chunk) < min_mag:
                filt_stats["mag"] += 1
                continue
            chunk_i = MagToTravelChunk(
                v=v_chunk,
                x=x_chunk,
                mag=mag_chunk,
                idx=idx,
                chunk_len=chunk_len
            )
            chunks.append(chunk_i)

        self.stats["chunks_filtered"] = filt_stats

        print("Training chunks:", len(chunks))

        return chunks, all_mags

    def format_chunks_for_fit(self, chunks: list[MagToTravelChunk]):
        # Formulate input data and residuals and threshold by min mag
        # Axis 0: chunk: n_chunks
        # Axis 1: point index: n_points
        # Axis 2: mag (absolute), x (relative to point at index 0): 
        chunk_len = self.chunk_len

        pt_idxes = [chunk_len] + list(range(0, chunk_len)) + list(range(chunk_len + 1, 2 * chunk_len))

        input_list = []
        for chunk_i in chunks:
            input_list.append([chunk_i.mag[pt_idxes], chunk_i.x[pt_idxes]])

        if not input_list:
            return np.empty((0, 2, 2 * chunk_len), dtype=float)

        input_arr = np.array(input_list, dtype=float)
        print("Min mag at indices:", np.min(input_arr[:, 0, :]), "mean", np.mean(input_arr[:, 0, :]), "max", np.max(input_arr[:, 0, :]))
        print(input_arr.shape)
        return input_arr

    def make_residual_fn(self, model: MagToTravelModel, input_arr, power_prior: float):
        def calculate_res(vec):
            x0, y_scale, power = vec[0], vec[1], vec[2]

            zero_x_mags = input_arr[:, 0, 0]
            zero_x_preds = model.pred_x(zero_x_mags, np.array([x0, y_scale, power]))
            x_acc_preds = input_arr[:, 1, 1:] + zero_x_preds[:, np.newaxis]

            mag_pts = input_arr[:, 0, 1:]
            x_mag_preds = model.pred_x(mag_pts, np.array([x0, y_scale, power]))
            res = x_acc_preds - x_mag_preds
            #res *= np.sqrt(chunk_weights)[:, np.newaxis]

            power_res = power - power_prior

            return np.concatenate([res.flatten(), np.array([power_res]) * self.power_weight])

        return calculate_res

    def fit_model(self, input_arr, power_prior=1 / 3, guess_vec=None):
        #chunk_weights = self.get_fit_chunk_weights(input_arr)
        model = MagToTravelModel(pred_soft_mg=self.pred_soft_mg)
        calculate_res = self.make_residual_fn(model, input_arr, power_prior)
        if guess_vec is None:
            guess_vec = [self.min_mag, 3, 1 / 3]
        result = least_squares(
                fun=calculate_res,
                x0=guess_vec, 
                method="trf",
                verbose=1,
                max_nfev=1000,
                #loss='huber',
            )

        model.set_coeffs(result.x)
        self.model = model

        return result

    def maybe_filter_worst_chunks(self, input_arr, result):
        frac = float(self.retrain_drop_worst_chunk_frac)
        if frac <= 0 or input_arr.shape[0] == 0:
            return input_arr, self.chunks, False

        n_chunks = input_arr.shape[0]
        if n_chunks <= 1:
            print("Skipping retrain chunk pruning: need more than one chunk,", n_chunks, "available")
            return input_arr, self.chunks, False

        chunk_res_len = input_arr.shape[2] - 1
        fit_residuals = result.fun[: n_chunks * chunk_res_len]
        chunk_residuals = fit_residuals.reshape(n_chunks, chunk_res_len)
        chunk_scores = np.mean(np.abs(chunk_residuals), axis=1)

        remove_count = max(int(n_chunks * frac), self.retrain_drop_worst_chunk_min_count)
        max_remove_count = n_chunks - self.retrain_drop_worst_chunk_min_remaining
        if max_remove_count <= 0:
            print(
                "Skipping retrain chunk pruning: min remaining",
                self.retrain_drop_worst_chunk_min_remaining,
                "would leave no room to prune from",
                n_chunks,
                "chunks",
            )
            return input_arr, self.chunks, False

        remove_count = min(remove_count, max_remove_count)
        if remove_count <= 0:
            print("Skipping retrain chunk pruning: computed remove count is", remove_count)
            return input_arr, self.chunks, False

        worst_chunks = np.argsort(chunk_scores)[-remove_count:]
        keep_mask = np.ones(n_chunks, dtype=bool)
        keep_mask[worst_chunks] = False

        chunk_centers = [int(self.chunks[i].idx) for i in worst_chunks]
        print(
            "Retraining after pruning",
            remove_count,
            "worst chunks out of",
            n_chunks,
            "score pct",
            np.percentile(chunk_scores, [0, 50, 90, 100]),
            "chunk centers",
            chunk_centers,
        )

        self.stats["retrain_pruned_chunks"] = {
            "removed_count": int(remove_count),
            "removed_positions": worst_chunks.tolist(),
            "removed_centers": chunk_centers,
            "score_percentiles": np.percentile(chunk_scores, [0, 50, 90, 100]).tolist(),
        }

        filtered_chunks = [chunk for i, chunk in enumerate(self.chunks) if keep_mask[i]]
        return input_arr[keep_mask], filtered_chunks, True

    def train(self, input_arr, power_prior = 1/3):
        if input_arr.shape[0] == 0:
            raise ValueError("No training chunks available for mag-to-travel fit")

        result = self.fit_model(input_arr, power_prior=power_prior)

        filtered_input_arr, filtered_chunks, filtered = self.maybe_filter_worst_chunks(input_arr, result)
        if filtered:
            self.chunks = filtered_chunks
            result = self.fit_model(filtered_input_arr, power_prior=power_prior, guess_vec=result.x.copy())

        return result
