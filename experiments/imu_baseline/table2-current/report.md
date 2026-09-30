# Table 2: method comparison

Centered RMSE in mm, median [Q1, Q3] across independent parent recordings. Derived chunks are averaged within parent first. Every method is scored on identical valid samples within each recording.

| Setup | Entries / parents | Method | Centered RMSE mm |
|---|---:|---|---:|
| Stumpjumper / pod v2 | 11 / 11 | IMU integration | 17.68 [15.71, 20.43] |
| Stumpjumper / pod v2 | 11 / 11 | Supervised magnetic oracle (in-sample) | 3.05 [2.59, 3.53] |
| Stumpjumper / pod v2 | 11 / 11 | Self-supervised magnetic estimate | 4.09 [3.74, 4.59] |
| Stumpjumper / pod v2 | 11 / 11 | First fusion | 3.77 [3.00, 4.34] |
| Stumpjumper / pod v2 | 11 / 11 | Complete pipeline | 2.73 [2.53, 3.05] |
| TR11 / pod v2 (Jamaal) | 6 / 6 | IMU integration | 19.31 [15.26, 21.81] |
| TR11 / pod v2 (Jamaal) | 6 / 6 | Supervised magnetic oracle (in-sample) | 7.84 [6.92, 8.38] |
| TR11 / pod v2 (Jamaal) | 6 / 6 | Self-supervised magnetic estimate | 11.36 [10.08, 12.81] |
| TR11 / pod v2 (Jamaal) | 6 / 6 | First fusion | 11.06 [9.67, 12.21] |
| TR11 / pod v2 (Jamaal) | 6 / 6 | Complete pipeline | 5.53 [5.45, 7.69] |
| TR11 2025 / pod v2 (Harry) | 14 / 14 | IMU integration | 25.81 [22.79, 27.65] |
| TR11 2025 / pod v2 (Harry) | 14 / 14 | Supervised magnetic oracle (in-sample) | 5.29 [4.95, 5.71] |
| TR11 2025 / pod v2 (Harry) | 14 / 14 | Self-supervised magnetic estimate | 7.56 [6.74, 8.10] |
| TR11 2025 / pod v2 (Harry) | 14 / 14 | First fusion | 7.03 [6.28, 7.47] |
| TR11 2025 / pod v2 (Harry) | 14 / 14 | Complete pipeline | 5.50 [4.76, 6.04] |
| Slayer / pod v2 | 7 / 5 | IMU integration | 8.92 [5.90, 11.29] |
| Slayer / pod v2 | 7 / 5 | Supervised magnetic oracle (in-sample) | 2.23 [1.75, 2.52] |
| Slayer / pod v2 | 7 / 5 | Self-supervised magnetic estimate | 3.48 [2.71, 4.58] |
| Slayer / pod v2 | 7 / 5 | First fusion | 3.43 [2.56, 4.02] |
| Slayer / pod v2 | 7 / 5 | Complete pipeline | 3.89 [2.54, 5.04] |
| Specialized Stumpjumper rear / pod v1 | 12 / 9 | IMU integration | 17.03 [15.18, 17.75] |
| Specialized Stumpjumper rear / pod v1 | 12 / 9 | Supervised magnetic oracle (in-sample) | 1.66 [1.56, 1.74] |
| Specialized Stumpjumper rear / pod v1 | 12 / 9 | Self-supervised magnetic estimate | 2.62 [2.20, 3.22] |
| Specialized Stumpjumper rear / pod v1 | 12 / 9 | Complete pipeline | 2.48 [2.15, 3.25] |

The self-supervised magnetic estimate has lower centered RMSE than IMU-only integration in every setup. First fusion improves on the self-supervised magnetic estimate in 4/4 front setups. The complete pipeline improves on the self-supervised magnetic estimate in 4/5 setups; the exception is the Slayer, where first fusion has the lowest median among the two fused outputs.

The oracle is trained and evaluated on the same recording with reference labels. It is a diagnostic comparator, not an independent validation result. IMU filter settings were selected on this development cohort. First fusion to complete pipeline combines nuisance correction, anchoring, and a second solve; rear has only one fusion stage.

The CSV summary also includes absolute MAE and centered travel-bin RMSE. The IMU baseline has no absolute zero reference. Edge exclusions make these results differ from historical full-support statistics. No setup-stratified generalization claim beyond these development recordings is implied.

Source experiments:
- experiments/imu_baseline/table2-current/source_experiments/20260927T170342Z-table2-method-comparison-front
- experiments/imu_baseline/table2-current/source_experiments/20260927T170338Z-table2-method-comparison-rear
