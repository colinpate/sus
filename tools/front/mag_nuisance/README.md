# Front magnetic-nuisance tools

Run these entry points from the repository root with `venv/bin/python`.
Production-candidate experiments are encoder-blind during fitting and prediction;
encoder travel is loaded afterward only for metrics unless a tool is explicitly
listed as a supervised diagnostic.

## Reusable solvers

| File | Purpose |
| --- | --- |
| `../../../backend/mag_nuisance_core.py` | Production gyro-aware Kalman/RTS smoother, encoder-blind XYZ-path fit, iterative corrected-XYZ inversion, and covariance helpers. |
| `../../../backend/mag_nuisance.py` | Pipeline wrappers for the 10 Hz four-iteration state solve, full-rate field transport, delta lifting, and corrected magnetic travel observation. |
| `mag_correction_solver.py` | Compatibility import for older notebooks and scripts; new code imports the production core directly. |
| `joint_mag_accel_solver.py` | Sparse joint latent-travel/body-field/world-field optimizer used by the joint-solver experiment. |

## Current encoder-blind experiments

| File | Purpose |
| --- | --- |
| `experiment_unsupervised_mag_xyz.py` | Builds an XYZ path from the existing scalar model without encoder labels and tests iterative body/world correction. |
| `experiment_mag_nuisance_observability.py` | Compares anchors, curve-normal observations, and acceleration-derived tangents. |
| `experiment_iterative_tangent_ablation.py` | Separates outer-iteration count from fixed tangent weighting. |
| `experiment_slope_derived_covariance.py` | Propagates assumed travel uncertainty through local XYZ slope and tests high-slope normal inflation. |
| `experiment_multirate_observations.py` | Separates source-rate gyro integration from point, low-pass, mean, and median residual observations on the 10 Hz nuisance-state grid. |
| `experiment_joint_mag_accel.py` | Tests joint latent travel and nuisance fields, including direct acceleration-factor ablations. |
| `evaluate_correction_standard_metrics.py` | Recomputes standard overall and travel-bin metrics on the validated 10 Hz state samples. |
| `evaluate_full_rate_correction.py` | Compares the pipeline baseline, full-rate delta lift, corrected magnetic observation, and second fusion pass, including a high-frequency preservation check. |

## Supervised diagnostics

| File | Purpose |
| --- | --- |
| `test_ambient_field.py` | Windowed encoder-anchored test of body-fixed, world-fixed, and combined residual stationarity. |
| `experiment_mag_correction.py` | Alternating-block held-out test with encoder-calibrated XYZ/scalar models. Useful as an upper-bound and model-validity diagnostic, not a deployable training path. |
| `analyze_low_end_curve_error.py` | Decomposes centered low-travel error into stable curve disagreement, block drift, and within-bin scatter. |

Superseded snapshots and one-off hacks are retained in `archive/`. The complete
experiment narrative and recommended starting point are in
`reports/front_mag_nuisance/README.md`.
