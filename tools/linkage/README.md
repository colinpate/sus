# Linkage Tools

`export_horst_linkage_curve.py` samples the Horst linkage and writes a CSV with
`rocker_angle_deg` and `wheel_travel_mm`, which `backend.angle.LinkageAngleToTravel`
can interpolate.

Example:

```bash
venv/bin/python tools/linkage/export_horst_linkage_curve.py \
  --output linkage_curve.csv \
  --rocker-angle-stop-deg 35 \
  --axle-offset-x -35 \
  --axle-offset-y -20
```

Use `horst_linkage_example.py` when you want a geometry plot or axle-path inspection.
