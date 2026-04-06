import matplotlib.pyplot as plt
import numpy as np
from stats_aggregator import DEFAULT_LOGS
import argparse
from scipy.optimize import least_squares


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pipeline cache durations and error statistics.")
    parser.add_argument(
        "logs",
        nargs="*",
        default=DEFAULT_LOGS,
        help="Log names to summarize. Defaults to the logs used by refine_mag_proj.py.",
    )
    parser.add_argument(
        "--top-adjacent",
        type=float,
        default=237.5 / 2,
    )
    parser.add_argument(
        "--n-close",
        type=int,
        default=10,
    )
    return parser.parse_args()


HYPOTENUSE = 120


def get_accel_mask(ws, raw_accel_max=150, proj_accel_min=1):
    accel_raw = ws["accel/lis2__x"][:, 0]
    raw_mask = np.abs(accel_raw) < raw_accel_max
    accel_proj = ws["accel/proj__x"][:, 0]
    proj_mask = np.abs(accel_proj) > proj_accel_min
    return raw_mask & proj_mask


def get_travel(angle, top_adjacent, top_zeroangle):
    top_angle = np.arccos(top_adjacent / HYPOTENUSE)
    #top_zeroangle = np.percentile(angle, 99.9)
    net_angle = -1 * (angle - top_zeroangle) + top_angle

    travel = 2 * (top_adjacent - (HYPOTENUSE * np.cos(net_angle)))
    return travel


def get_travel_accel(travel, t):
    dt_s = np.diff(t, prepend=t[0] - 0.01)
    v = np.gradient(travel, t, edge_order = 2)
    a_gt = np.gradient(v, t, edge_order = 2) / 1000.0
    return a_gt


def learn_params(
        angle, 
        accel, 
        t, 
        accel_mask, 
        guess_ta,
        guess_za,
        max_travel=165, 
        oob_weight=1,
    ):

    def get_res(x):
        top_adjacent, top_zeroangle = x[0], x[1]
        travel = get_travel(angle, top_adjacent, top_zeroangle)
        travel_accel = get_travel_accel(travel, t)

        oob_res = (travel > max_travel) * (travel - max_travel) * oob_weight
        accel_res = travel_accel[accel_mask] - accel[accel_mask]

        return np.concat([oob_res, accel_res], axis=0)

    guess_vec = [guess_ta, guess_za]

    gt_result = least_squares(
                fun=get_res,
                x0=guess_vec, 
                method="trf",
                verbose=1,
                max_nfev=1000,
                #loss='huber',
            )    
    
    return gt_result.x


def main():
    args = parse_args()
    for log in args.logs:
        out_dir = f"backend/run_artifacts/{log}/cache/"
        ws_file = out_dir + "/all.npz"
        ws = np.load(ws_file)

        angle = ws["angle__x"][:, 0]
        t = ws["angle__t"]

        accel_mask = get_accel_mask(ws)

        # Calculate travel the old fashioned way
        angle_sorted = np.sort(angle)
        top_zeroangle = angle_sorted[-args.n_close]
        og_travel = get_travel(angle, args.top_adjacent, top_zeroangle)

        learned_ta, learned_za = learn_params(
            angle=angle,
            accel=ws["accel/proj__x"][:, 0],
            t=t,
            accel_mask=accel_mask,
            guess_ta=args.top_adjacent,
            guess_za=top_zeroangle,
        )
        learned_travel = get_travel(angle, learned_ta, learned_za)

        print(log)
        print(f"Learned top adjacent: {learned_ta:.3f}")
        print(f"Learned top zero angle: {learned_za:.3f}")
        for travel in [og_travel, learned_travel]:
            travel_sorted = np.sort(travel)
            travel_floor = travel_sorted[args.n_close]
            travel_ceil = travel_sorted[-args.n_close]
            print(f"Travel min, max: {np.min(travel):.2f}, {np.max(travel):.2f}")
            print(f"Travel floor, ceil: {travel_floor:.2f}, {travel_ceil:.2f}")
            print(f"Travel top zero angle: {top_zeroangle:.3f}")

if __name__ == "__main__":
    main()