# Rear Mag Curve Plots

Plots overlay the learned mag-to-travel curve on `mag/proj/lpf` vs GT travel samples.
Gray points are boring-mask GT samples; red points are GT travel >=120 mm.
The curve offset is matched to the cached `travel/mag_model/adj` signal, and the curve range spans the full boring-mask mag range so it includes the high-travel tail.

## Selected Logs

| Group | Log | 120-150 RMSE | Samples | Filtered chunks |
|---|---|---:|---:|---:|
| Worst | `log144_rear` | 19.16 | 59 | 6 |
| Worst | `log141_rear` | 18.40 | 200 | 22 |
| Worst | `log140_rear` | 16.96 | 514 | 42 |
| Best | `log151_rear` | 7.51 | 267 | 23 |
| Best | `log152_rear` | 7.75 | 140 | 17 |
| Best | `log150_rear` | 11.07 | 164 | 14 |