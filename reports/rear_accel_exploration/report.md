# Rear IMU Travel-Acceleration Exploration

Logs used:

- `log136_rear`
- `log137_rear`

Ground truth:

- `travel__x` from the existing rear pipeline cache
- `a_gt = d²(travel)/dt² / 1000`

Runtime constraints for the estimator:

- no linkage-curve geometry at runtime
- no travel-GT-trained weights for the production candidate
- offline analysis is allowed
- hinge axis can be treated as sensor `z`
- sensor is at the axle in the travel plane

Generated artifacts:

- `summary.json`
- `log136_rear_comparison.png`
- `log137_rear_comparison.png`

## Key Result

`mag/proj` changes the problem a lot.

On both logs, the low-passed projected magnetic signal is almost a direct travel surrogate:

- `log136_rear`: corr(`mag/proj/lpf`, `travel`) = `-0.9849`
- `log137_rear`: corr(`mag/proj/lpf`, `travel`) = `-0.9857`

So the best use of magnetics is not as a small extra feature on top of the old IMU model. It works much better as the main **travel state estimate**, with acceleration obtained by differentiating that travel estimate.

## Best Overall Upper Bound

If I allow travel GT only to calibrate a monotonic `mag -> travel` map, the magnetic signal becomes the best model in the repo so far.

Model:

1. Fit a monotonic isotonic regression `travel_mm = f(mag/proj/lpf)`.
2. Transfer that mapping across logs.
3. Differentiate the predicted travel with a 7-sample Savitzky-Golay second derivative.

Leave-one-log-out results:

| Train | Test | Travel RMSE (mm) | Accel RMSE | Accel MAE | Accel Corr |
|---|---|---:|---:|---:|---:|
| `log136_rear` | `log137_rear` | 4.339 | 6.206 | 3.332 | 0.756 |
| `log137_rear` | `log136_rear` | 4.331 | 6.796 | 3.895 | 0.763 |

Global active-sample metrics (`|a_gt| >= 0.5 m/s²`):

| Model | RMSE | MAE | Corr |
|---|---:|---:|---:|
| Gravity-basis IMU model (old best GT-trained IMU-only) | 7.340 | 4.175 | 0.616 |
| **Mag isotonic upper bound** | **6.504** | **3.610** | **0.760** |

So if a bike-specific mag-to-travel calibration is available, the magnetic signal is now clearly the strongest path.

## Best Production Candidate Without Travel GT

The best no-GT model I found is a **self-calibrated magnetic travel model** built from the ride itself.

Model:

1. Use `mag/proj/lpf` as the position-like signal.
2. Find local extrema of `mag/proj/lpf` as zero-velocity points.
3. Use the existing `MagToTravelModelCore` chunk fitter to learn a monotonic mag-to-travel curve from:
   - `mag/proj/lpf`
   - `HP(lis2_z, 1.0 Hz)` as the local motion cue
   - a fixed compression sign convention
4. Differentiate the fitted travel with a 7-sample Savitzky-Golay second derivative.

Best global no-GT metrics:

| Model | RMSE | MAE | Corr |
|---|---:|---:|---:|
| Zero predictor | 9.317 | 5.465 | n/a |
| Fixed `lis2_z` high-pass baseline | 10.070 | 6.303 | 0.183 |
| **Mag self-calibrated** | **8.160** | **4.639** | **0.724** |

Per-log no-GT self-calibrated metrics:

| Log | Travel RMSE, centered (mm) | Travel Corr | Accel RMSE | Accel MAE | Accel Corr |
|---|---:|---:|---:|---:|---:|
| `log136_rear` | 10.490 | 0.985 | 8.264 | 4.807 | 0.759 |
| `log137_rear` | 13.133 | 0.923 | 8.057 | 4.475 | 0.721 |

This is a meaningful improvement over the old no-GT IMU-only baseline:

- RMSE improves by `1.91 m/s²`
- correlation jumps from `0.183` to `0.724`

That correlation jump is the biggest change here. The model is following the right acceleration shape much more reliably once mag is used as the travel backbone.

## Recommendation

If the requirement is **no travel-GT-trained weights**, I would now use the magnetic path, not the IMU-only high-pass path:

1. Build `mag/proj/lpf`.
2. Find mag extrema / ZV points.
3. Self-calibrate a monotonic `mag -> travel` curve per ride with `MagToTravelModelCore`.
4. Differentiate that travel estimate to get axle acceleration.
5. Keep a one-time sign convention so “compression-positive” stays consistent.

This is the best match to the transfer requirements so far because it does not depend on:

- linkage dimensions
- GT-trained regression weights
- a fixed travel-vs-IMU coefficient set

and it should tolerate yaw remounting around the `z` axis reasonably well because:

- the axle acceleration proxy is only using sensor `z`
- `mag/proj` is derived from a reprojected magnetic vector, not a fixed raw axis

## Practical Meaning Of The Sign Convention

The self-calibrated fit still has one unavoidable ambiguity: sign.

From unlabeled data alone, the fitter can recover “this magnet signal is monotonic with suspension motion,” but it cannot know which direction should count as positive travel. In these logs, the correct convention was:

- `travel_sign = -1`

I do **not** think of that as a learned magic weight. It is just a setup convention. In production, this can be fixed once with a short manual compression calibration.

## Caveats

- The self-calibrated model still has a noticeable scale gap to the supervised mag upper bound, especially on `log137_rear`. That is where most of the remaining RMSE seems to live.
- Absolute travel offset is not solved here, but that does not matter for acceleration.
- This is still only two logs. The next important validation step is more varied rides to see whether the self-calibration remains stable when whole-bike motion gets wilder.

## Bottom Line

Before `mag/proj`, the best no-GT answer was basically “high-pass `lis2_z` and accept that it is weak.”

After `mag/proj`, the best no-GT answer is much better:

- treat mag as the primary travel sensor
- self-calibrate its monotonic travel curve from the ride
- differentiate that travel estimate for acceleration

That is the direction I would build on next.
