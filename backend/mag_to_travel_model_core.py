from dataclasses import dataclass, field
import numpy as np
import scipy
from scipy.optimize import least_squares


@dataclass
class MagToTravelChunk:
    a: np.ndarray
    t: np.ndarray
    mag: np.ndarray
    slice_i: slice
    zv_idx: int
    badmask: np.ndarray | None = None
    v: np.ndarray | None = None
    x: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    errors: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.chunk_len = self.slice_i.stop - self.slice_i.start


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
    chunk_rad: int = 20
    train_with_mask: bool = False
    bad_thresh: float = 0.5
    dm_dx_thresh: float | None = 0.05
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

    def integrate_chunk(self, chunk: MagToTravelChunk):
        v_chunk = scipy.integrate.cumulative_trapezoid(chunk.a, chunk.t, initial=0)
        v_chunk -= v_chunk[chunk.zv_idx]
        x_chunk = scipy.integrate.cumulative_trapezoid(v_chunk, chunk.t, initial=0)
        x_chunk -= x_chunk[chunk.zv_idx]
        chunk.v = v_chunk
        chunk.x = x_chunk

    def calc_chunk_metrics(self, chunk: MagToTravelChunk):
        chunk.metrics["dx"] = max(chunk.x) - min(chunk.x)
        chunk.metrics["mag_min"] = min(chunk.mag)
        chunk.metrics["dm/dx"] = np.diff(chunk.mag, prepend=chunk.mag[0]) / (chunk.v + 1e-6)
        chunk.metrics["dm/dx_median"] = np.median(chunk.metrics["dm/dx"])
        chunk.metrics["b_x_corr"] = scipy.stats.spearmanr(chunk.mag, chunk.x).correlation
        if chunk.badmask is not None:
            chunk.metrics["badmask_mean"] = np.mean(chunk.badmask)

    def calc_chunk_errors(
        self, 
        chunk: MagToTravelChunk, 
        travel_gt: np.ndarray, 
        v_gt: np.ndarray | None,
        a_gt: np.ndarray | None
    ):
        trav_rel = travel_gt[chunk.slice_i]
        trav_rel -= trav_rel[chunk.zv_idx]
        chunk.errors["x"] = chunk.x - trav_rel
        if v_gt is not None:
            chunk.errors["v"] = chunk.v - v_gt[chunk.slice_i]
        if a_gt is not None:
            chunk.errors["a"] = chunk.a - a_gt[chunk.slice_i]

    def calc_chunks_errors(
        self, 
        chunks: list[MagToTravelChunk],
        travel_gt: np.ndarray, 
        v_gt: np.ndarray | None,
        a_gt: np.ndarray | None
    ):
        for chunk in chunks:
            self.calc_chunk_errors(chunk, travel_gt, v_gt, a_gt)

    def filter_chunk_dx(self, chunk: MagToTravelChunk):
        return self.chunk_min_dx <= chunk.metrics["dx"] <= self.chunk_max_dx
    
    def filter_chunk_dm_dx(self, chunk: MagToTravelChunk):
        return chunk.metrics["dm/dx_median"] >= self.dm_dx_thresh
    
    def filter_chunk_badmask(self, chunk: MagToTravelChunk):
        return chunk.metrics["badmask_mean"] <= self.bad_thresh
    
    def filter_chunk_minmag(self, chunk: MagToTravelChunk, min_mag: float):
        return chunk.metrics["mag_min"] >= min_mag

    def get_filter_fns(self, min_mag: float):
        filter_fns = [
            self.filter_chunk_badmask,
            self.filter_chunk_dm_dx,
            self.filter_chunk_dx,
            lambda x: self.filter_chunk_minmag(x, min_mag),
        ]
        return filter_fns

    def create_chunks(self, idxs_filt, mag, acc, t_s, mag_proj_bad_mask):
        chunks = []
        chunk_rad = self.chunk_rad
        for idx in idxs_filt:
            if idx < chunk_rad or idx + chunk_rad >= len(mag):
                continue
            slice_i = slice(idx - chunk_rad, idx + chunk_rad)
            chunk = MagToTravelChunk(
                a=acc[slice_i] * 1000,
                t=t_s[slice_i],
                mag=mag[slice_i],
                badmask=mag_proj_bad_mask[slice_i],
                slice_i=slice_i,
                zv_idx=chunk_rad
            )
            chunks.append(chunk)
        return chunks
    
    def prepare_chunks(self, chunks: list[MagToTravelChunk]):
        for chunk in chunks:
            self.integrate_chunk(chunk)
            self.calc_chunk_metrics(chunk)
    
    def filter_chunks(self, chunks: list[MagToTravelChunk], filters: list[callable]):
        chunks_filt = []
        for chunk in chunks:
            for filter in filters:
                if not filter(chunk):
                    break
            else:
                chunks_filt.append(chunk)
        return chunks_filt
    
    def get_chunks(self, idxs_filt, mag, acc, t_s, mag_proj_bad_mask, min_mag):
        chunks = self.create_chunks(idxs_filt, mag, acc, t_s, mag_proj_bad_mask)
        all_mags = [chunk.mag for chunk in chunks]
        self.prepare_chunks(chunks)
        filters = self.get_filter_fns(min_mag=min_mag)
        chunks_filt = self.filter_chunks(chunks, filters)
        print("Training chunks:", len(chunks))
        return chunks_filt, all_mags

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

    def format_chunks_for_fit(self, chunks: list[MagToTravelChunk]):
        # Formulate input data and residuals and threshold by min mag
        # Axis 0: chunk: n_chunks
        # Axis 1: point index: n_points
        # Axis 2: mag (absolute), x (relative to point at index 0): 
        if not chunks:
            return np.empty((0, 3, 0), dtype=float)
        max_chunk_len = max([chunk.chunk_len for chunk in chunks])

        input_arr = np.zeros((len(chunks), 3, max_chunk_len))

        for i, chunk in enumerate(chunks):
            all_idxes = np.arange(chunk.chunk_len)
            pt_idxes = [chunk.zv_idx] + all_idxes[all_idxes != chunk.zv_idx].tolist()

            input_arr[i, 0, :chunk.chunk_len] = chunk.mag[pt_idxes]
            input_arr[i, 1, :chunk.chunk_len] = chunk.x[pt_idxes]
            input_arr[i, 2, :chunk.chunk_len] = 1

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
            mask = input_arr[:, 2, 1:]
            res = (x_acc_preds - x_mag_preds) * mask
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

        chunk_centers = [int(self.chunks[i].slice_i.start) for i in worst_chunks]
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
