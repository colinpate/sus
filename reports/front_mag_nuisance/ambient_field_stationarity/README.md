# Ambient-field stationarity diagnostics

These encoder-supervised window tests established the initial evidence for the
body/world nuisance model. They subtract an encoder-binned expected XYZ curve,
integrate gyro1, and compare held-out residual stationarity in body and local
world frames.

Across `log109`, `log110`, and `log-0078-valid`:

| Model | Median held-out residual improvement | Windows improved |
| --- | ---: | ---: |
| World-fixed only | 2–10% | 70–80% |
| Body offset + world field | 22–44% | 88–97% |

The world-fixed hypothesis is directionally correct but incomplete. A combined
model consistently explains much more residual structure. The individual body
and world vectors should not be interpreted as unique physical decompositions;
their sum is the useful observation.

The pod-v1 and pod-v2 strong-magnet rotation recordings support the same signed
axis mapping:

```text
gyro_xyz = [[0, 0, 1],
            [0,-1, 0],
            [1, 0, 0]] @ recorded_mmc_xyz
```

The secondary LIS3MDL tests are retained here too. They showed railing and weak
axis-level agreement after orientation correction, so the later production
experiments intentionally use only MMC5603 and gyro1.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/test_ambient_field.py LOG_NAME
```

Each log directory contains `summary.json`, fitted `field_curves.csv`, and the
available residual/stationarity plots.
