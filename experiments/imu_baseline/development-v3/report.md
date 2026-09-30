# IMU integration baseline filter sweep

Development-set tuning; zero-phase filters; zero initial velocity/displacement; fixed common scoring samples per recording across candidates.

Mean RMSE within parent recording, then mean within setup, then equal mean across setups; separately front/rear.

Completed 50 of 50 recordings; 0 failures.

Input: front raw projected acceleration; rear 40 Hz low-pass projected acceleration before magnetic ZV correction. The baseline applies a 40 Hz second-order low-pass in both cases.

Two seconds at each valid segment boundary are excluded for every candidate. This is a common edge policy, not a claim that every cutoff fully settles within two seconds.

| Pipeline | HPF location | Cutoff Hz | Setup-balanced centered RMSE mm |
|---|---|---:|---:|
| front | both | 1.0 | 17.651 |
| rear | displacement | 4.0 | 15.938 |

Selection uses the evaluation cohort and is not held-out validation. Scores include only the common finite active samples passing the existing reference/IMU quality masks. Source cache hashes and fingerprints are frozen in the manifest; these are historical preprocessing inputs, not a claim that every cache matches the later edited backend.

Failures: []
