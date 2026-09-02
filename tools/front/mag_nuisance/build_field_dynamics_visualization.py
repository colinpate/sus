#!/usr/bin/env python3
"""Build the interactive field-dynamics visualization from report CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_or_none(value: float, digits: int) -> float | None:
    return round(float(value), digits) if np.isfinite(value) else None


def summary_payload(row: pd.Series) -> dict:
    return {
        "name": row["log"],
        "fork": row["fork"],
        "duration": round(float(row["duration_s"]), 1),
        "update": round(float(row["update_fraction"]), 3),
        "body": {
            "p10": round(float(row["body_magnitude_mg_p10"]), 1),
            "median": round(float(row["body_magnitude_mg_median"]), 1),
            "p90": round(float(row["body_magnitude_mg_p90"]), 1),
            "rateMedian": round(float(row["body_direction_rate_dps_median"]), 2),
            "rateP90": round(float(row["body_direction_rate_dps_p90"]), 2),
        },
        "world": {
            "p10": round(float(row["world_magnitude_mg_p10"]), 1),
            "median": round(float(row["world_magnitude_mg_median"]), 1),
            "p90": round(float(row["world_magnitude_mg_p90"]), 1),
            "rateMedian": round(
                float(row["world_initial_direction_rate_dps_median"]), 2
            ),
            "rateP90": round(
                float(row["world_initial_direction_rate_dps_p90"]), 2
            ),
            "bodyFrameRateMedian": round(
                float(row["world_body_direction_rate_dps_median"]), 2
            ),
            "bodyFrameRateP90": round(
                float(row["world_body_direction_rate_dps_p90"]), 2
            ),
        },
        "total": {
            "p10": round(float(row["total_magnitude_mg_p10"]), 1),
            "median": round(float(row["total_magnitude_mg_median"]), 1),
            "p90": round(float(row["total_magnitude_mg_p90"]), 1),
        },
        "angle": round(float(row["body_world_angle_median_deg"]), 1),
    }


def series_payload(frame: pd.DataFrame, stride: int) -> dict[str, list[list]]:
    payload = {}
    columns = [
        "time_s",
        "body_magnitude_mg",
        "world_magnitude_mg",
        "total_magnitude_mg",
        "body_direction_rate_dps",
        "world_initial_direction_rate_dps",
        "world_body_direction_rate_dps",
        "update_active",
    ]
    digits = [1, 1, 1, 1, 2, 2, 2, 0]
    for name, log_frame in frame.groupby("log", sort=False):
        sampled = log_frame.iloc[::stride]
        if sampled.index[-1] != log_frame.index[-1]:
            sampled = pd.concat([sampled, log_frame.iloc[[-1]]])
        rows = []
        for values in sampled[columns].itertuples(index=False, name=None):
            rows.append(
                [finite_or_none(value, digit) for value, digit in zip(values, digits)]
            )
        payload[name] = rows
    return payload


TEMPLATE = r'''<div id="nfd-viz" class="nfd-shell">
  <style>
    #nfd-viz {
      --nfd-fg: var(--color-text-primary, #182230);
      --nfd-muted: var(--color-text-secondary, #637083);
      --nfd-border: var(--color-border, rgba(99,112,131,.24));
      --nfd-panel: var(--color-background-secondary, rgba(127,127,127,.055));
      --nfd-grid: var(--color-border, rgba(99,112,131,.16));
      color: var(--nfd-fg);
      font-family: var(--font-sans, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
      max-width: 1180px;
      margin: 0 auto;
      padding: 10px 2px 18px;
    }
    #nfd-viz * { box-sizing: border-box; }
    #nfd-viz .nfd-head {
      display: flex; align-items: end; justify-content: space-between;
      gap: 18px; margin-bottom: 14px;
    }
    #nfd-viz h2 { margin: 0 0 5px; font-size: 22px; line-height: 1.2; }
    #nfd-viz .nfd-sub { margin: 0; color: var(--nfd-muted); font-size: 13px; line-height: 1.45; max-width: 790px; }
    #nfd-viz label { color: var(--nfd-muted); font-size: 12px; white-space: nowrap; }
    #nfd-viz select {
      display: block; min-width: 170px; margin-top: 4px; padding: 7px 28px 7px 9px;
      color: var(--nfd-fg); background: var(--nfd-panel); border: 1px solid var(--nfd-border);
      border-radius: 7px; font: inherit;
    }
    #nfd-viz .nfd-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
    #nfd-viz .nfd-stat { padding: 11px 12px; border: 1px solid var(--nfd-border); border-radius: 9px; background: var(--nfd-panel); }
    #nfd-viz .nfd-stat-label { color: var(--nfd-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .055em; }
    #nfd-viz .nfd-stat strong { display: block; margin: 4px 0 2px; font-size: 22px; font-variant-numeric: tabular-nums; }
    #nfd-viz .nfd-stat small { color: var(--nfd-muted); font-size: 12px; line-height: 1.35; }
    #nfd-viz .nfd-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    #nfd-viz .nfd-panel { min-width: 0; border: 1px solid var(--nfd-border); border-radius: 9px; padding: 11px 10px 8px; background: var(--nfd-panel); }
    #nfd-viz .nfd-panel-wide { grid-column: 1 / -1; }
    #nfd-viz .nfd-panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 0 3px 3px; }
    #nfd-viz h3 { margin: 0; font-size: 14px; font-weight: 650; }
    #nfd-viz .nfd-legend { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px 12px; color: var(--nfd-muted); font-size: 11px; }
    #nfd-viz .nfd-key { display: inline-flex; align-items: center; gap: 5px; }
    #nfd-viz .nfd-swatch { width: 15px; height: 3px; border-radius: 2px; }
    #nfd-viz svg { display: block; width: 100%; height: 300px; overflow: visible; }
    #nfd-viz .nfd-panel-wide svg { height: 220px; }
    #nfd-viz .domain, #nfd-viz .tick line { stroke: var(--nfd-grid); }
    #nfd-viz .tick text, #nfd-viz .nfd-axis-label { fill: var(--nfd-muted); font-size: 11px; }
    #nfd-viz .nfd-footer { color: var(--nfd-muted); font-size: 11.5px; line-height: 1.45; margin: 10px 3px 0; }
    #nfd-viz .nfd-tooltip {
      position: fixed; z-index: 20; pointer-events: none; opacity: 0;
      min-width: 190px; padding: 8px 10px; border-radius: 7px;
      color: var(--nfd-fg); background: var(--color-background-primary, rgba(255,255,255,.97));
      border: 1px solid var(--nfd-border); box-shadow: 0 6px 24px rgba(0,0,0,.14);
      font-size: 12px; line-height: 1.45; font-variant-numeric: tabular-nums;
    }
    #nfd-viz .nfd-tooltip b { font-weight: 650; }
    @media (max-width: 720px) {
      #nfd-viz .nfd-head { align-items: stretch; flex-direction: column; }
      #nfd-viz select { width: 100%; }
      #nfd-viz .nfd-stats { grid-template-columns: 1fr; }
      #nfd-viz .nfd-grid { grid-template-columns: 1fr; }
      #nfd-viz .nfd-panel-wide { grid-column: auto; }
      #nfd-viz .nfd-panel-head { align-items: flex-start; flex-direction: column; }
      #nfd-viz .nfd-legend { justify-content: flex-start; }
      #nfd-viz svg, #nfd-viz .nfd-panel-wide svg { height: 270px; }
    }
  </style>
  <div class="nfd-head">
    <div>
      <h2>Four-iteration nuisance-field dynamics</h2>
      <p class="nfd-sub" id="nfd-cohort"></p>
    </div>
    <label>Time-series log<select id="nfd-log"></select></label>
  </div>
  <div class="nfd-stats">
    <div class="nfd-stat"><span class="nfd-stat-label">Body state</span><strong id="nfd-body-stat"></strong><small id="nfd-body-detail"></small></div>
    <div class="nfd-stat"><span class="nfd-stat-label">World state</span><strong id="nfd-world-stat"></strong><small id="nfd-world-detail"></small></div>
    <div class="nfd-stat"><span class="nfd-stat-label">Applied sum</span><strong id="nfd-total-stat"></strong><small id="nfd-total-detail"></small></div>
  </div>
  <div class="nfd-grid">
    <section class="nfd-panel">
      <div class="nfd-panel-head"><h3>Magnitude over time</h3><div class="nfd-legend"><span class="nfd-key"><i class="nfd-swatch" style="background:#4e79a7"></i>body</span><span class="nfd-key"><i class="nfd-swatch" style="background:#f28e2b"></i>world</span><span class="nfd-key"><i class="nfd-swatch" style="background:#59a14f"></i>sum</span></div></div>
      <svg id="nfd-mag" role="img" aria-label="Body, world, and summed nuisance field magnitude over time"></svg>
    </section>
    <section class="nfd-panel">
      <div class="nfd-panel-head"><h3>Intrinsic direction rate</h3><div class="nfd-legend"><span class="nfd-key"><i class="nfd-swatch" style="background:#4e79a7"></i>body / body frame</span><span class="nfd-key"><i class="nfd-swatch" style="background:#f28e2b"></i>world / initial frame</span></div></div>
      <svg id="nfd-rate" role="img" aria-label="Body and world nuisance field direction rate over time"></svg>
    </section>
    <section class="nfd-panel nfd-panel-wide">
      <div class="nfd-panel-head"><h3>World field seen in the rotating body frame</h3><div class="nfd-legend"><span class="nfd-key"><i class="nfd-swatch" style="background:#9c6ade"></i>includes bike rotation</span></div></div>
      <svg id="nfd-frame" role="img" aria-label="World field direction rate in the body frame over time"></svg>
    </section>
  </div>
  <p class="nfd-footer">Direction rate is net angular displacement over a 1 s lag. The world-frame trace rotates the solver output back through integrated gyro; samples below 40 mG are excluded. Lines are plotted at 0.5 Hz from 10 Hz solver states. The body/world split is regularized and weakly identifiable; their sum is the reliable correction.</p>
  <div class="nfd-tooltip" id="nfd-tip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {
    const summaries = __SUMMARY_JSON__;
    const cohort = __COHORT_JSON__;
    const series = __SERIES_JSON__;
    const root = document.getElementById('nfd-viz');
    const select = root.querySelector('#nfd-log');
    const tip = root.querySelector('#nfd-tip');
    const colors = {body:'#4e79a7', world:'#f28e2b', total:'#59a14f', frame:'#9c6ade'};
    const byName = new Map(summaries.map(d => [d.name, d]));
    const defaultLog = byName.has('log-0078-valid') ? 'log-0078-valid' : summaries[0].name;
    summaries.forEach(d => select.append(new Option(`${d.name} · ${d.fork}`, d.name)));
    select.value = defaultLog;
    root.querySelector('#nfd-cohort').textContent = `Across ${cohort.logs} logs, median per-log body / world / summed magnitude is ${cohort.body} / ${cohort.world} / ${cohort.total} mG; intrinsic direction rate is ${cohort.bodyRate} / ${cohort.worldRate}°/s.`;

    function q(values, p) {
      const finite = values.filter(Number.isFinite).sort((a,b) => a-b);
      return finite.length ? d3.quantileSorted(finite, p) : 1;
    }
    function updateStats(s) {
      root.querySelector('#nfd-body-stat').textContent = `${s.body.median.toFixed(0)} mG`;
      root.querySelector('#nfd-body-detail').textContent = `${s.body.p10.toFixed(0)}–${s.body.p90.toFixed(0)} mG P10–P90 · ${s.body.rateMedian.toFixed(2)}°/s median, ${s.body.rateP90.toFixed(2)} P90`;
      root.querySelector('#nfd-world-stat').textContent = `${s.world.median.toFixed(0)} mG`;
      root.querySelector('#nfd-world-detail').textContent = `${s.world.p10.toFixed(0)}–${s.world.p90.toFixed(0)} mG P10–P90 · ${s.world.rateMedian.toFixed(2)}°/s median, ${s.world.rateP90.toFixed(2)} P90`;
      root.querySelector('#nfd-total-stat').textContent = `${s.total.median.toFixed(0)} mG`;
      root.querySelector('#nfd-total-detail').textContent = `${s.total.p10.toFixed(0)}–${s.total.p90.toFixed(0)} mG P10–P90 · ${s.angle.toFixed(0)}° median body/world angle`;
    }
    function drawChart(svgNode, rows, specs, yLabel, tooltipRows) {
      const svg = d3.select(svgNode);
      svg.selectAll('*').remove();
      const bounds = svgNode.getBoundingClientRect();
      const width = Math.max(300, bounds.width || 500);
      const height = Math.max(210, bounds.height || 280);
      const margin = {top:12, right:16, bottom:38, left:54};
      svg.attr('viewBox', `0 0 ${width} ${height}`);
      const x = d3.scaleLinear().domain([0, d3.max(rows, d => d[0]) || 1]).nice().range([margin.left, width-margin.right]);
      const allY = specs.flatMap(s => rows.map(d => d[s.index])).filter(Number.isFinite);
      const cap = Math.max(1, q(allY, .995) * 1.08);
      const y = d3.scaleLinear().domain([0, cap]).nice().range([height-margin.bottom, margin.top]);
      svg.append('g').attr('transform',`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(width < 430 ? 4 : 6));
      svg.append('g').attr('transform',`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5));
      svg.append('text').attr('class','nfd-axis-label').attr('x',(margin.left+width-margin.right)/2).attr('y',height-5).attr('text-anchor','middle').text('Time (s)');
      svg.append('text').attr('class','nfd-axis-label').attr('transform',`translate(13 ${(margin.top+height-margin.bottom)/2}) rotate(-90)`).attr('text-anchor','middle').text(yLabel);
      specs.forEach(spec => {
        const line = d3.line().defined(d => Number.isFinite(d[spec.index])).x(d => x(d[0])).y(d => y(Math.min(d[spec.index], cap))).curve(d3.curveLinear);
        svg.append('path').datum(rows).attr('fill','none').attr('stroke',spec.color).attr('stroke-width',1.55).attr('stroke-linejoin','round').attr('stroke-linecap','round').attr('d',line);
      });
      const cross = svg.append('line').attr('y1',margin.top).attr('y2',height-margin.bottom).attr('stroke','var(--nfd-muted)').attr('stroke-width',1).attr('stroke-dasharray','3,3').style('opacity',0);
      const bisect = d3.bisector(d => d[0]).center;
      svg.append('rect').attr('x',margin.left).attr('y',margin.top).attr('width',width-margin.left-margin.right).attr('height',height-margin.top-margin.bottom).attr('fill','transparent').style('cursor','crosshair')
        .on('pointermove', event => {
          const [mx] = d3.pointer(event);
          const i = Math.max(0, Math.min(rows.length-1, bisect(rows, x.invert(mx))));
          const d = rows[i]; cross.attr('x1',x(d[0])).attr('x2',x(d[0])).style('opacity',.65);
          const lines = tooltipRows.map(t => `<span style="color:${t.color}">●</span> ${t.label}: <b>${Number.isFinite(d[t.index]) ? d[t.index].toFixed(t.digits) : '—'} ${t.unit}</b>`).join('<br>');
          tip.innerHTML = `<b>${select.value}</b> · ${d[0].toFixed(1)} s<br>${lines}<br>field update: <b>${d[7] ? 'active' : 'propagating'}</b>`;
          tip.style.opacity = 1;
          const pad = 12, tw = tip.offsetWidth, th = tip.offsetHeight;
          tip.style.left = `${Math.min(window.innerWidth-tw-pad, event.clientX+14)}px`;
          tip.style.top = `${Math.min(window.innerHeight-th-pad, event.clientY+14)}px`;
        })
        .on('pointerleave', () => { cross.style('opacity',0); tip.style.opacity = 0; });
    }
    function render() {
      const name = select.value, s = byName.get(name), rows = series[name];
      updateStats(s);
      drawChart(root.querySelector('#nfd-mag'), rows,
        [{index:1,color:colors.body},{index:2,color:colors.world},{index:3,color:colors.total}], 'Magnitude (mG)',
        [{index:1,color:colors.body,label:'body',digits:1,unit:'mG'},{index:2,color:colors.world,label:'world',digits:1,unit:'mG'},{index:3,color:colors.total,label:'sum',digits:1,unit:'mG'}]);
      drawChart(root.querySelector('#nfd-rate'), rows,
        [{index:4,color:colors.body},{index:5,color:colors.world}], 'Direction rate (deg/s)',
        [{index:4,color:colors.body,label:'body / body',digits:2,unit:'°/s'},{index:5,color:colors.world,label:'world / initial',digits:2,unit:'°/s'}]);
      drawChart(root.querySelector('#nfd-frame'), rows,
        [{index:6,color:colors.frame}], 'Direction rate (deg/s)',
        [{index:6,color:colors.frame,label:'world / body',digits:2,unit:'°/s'}]);
    }
    select.addEventListener('change', render);
    let resizeTimer;
    new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 80); }).observe(root);
    render();
  })();
  </script>
</div>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stride", type=int, default=2,
        help="Subsample the report's 1 Hz series; 2 produces a 0.5 Hz plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("stride must be at least one")
    summaries_frame = pd.read_csv(args.report_dir / "per_log_summary.csv")
    cohort_frame = pd.read_csv(args.report_dir / "cohort_summary.csv")
    timeseries_frame = pd.read_csv(args.report_dir / "timeseries_1hz.csv")

    summaries = [summary_payload(row) for _, row in summaries_frame.iterrows()]
    all_cohort = cohort_frame[cohort_frame["fork"] == "all"].iloc[0]
    cohort = {
        "logs": int(all_cohort["logs"]),
        "body": round(float(all_cohort["body_magnitude_mg_median"])),
        "world": round(float(all_cohort["world_magnitude_mg_median"])),
        "total": round(float(all_cohort["total_magnitude_mg_median"])),
        "bodyRate": round(float(all_cohort["body_direction_rate_dps_median"]), 2),
        "worldRate": round(
            float(all_cohort["world_initial_direction_rate_dps_median"]), 2
        ),
    }
    series = series_payload(timeseries_frame, args.stride)
    html = (
        TEMPLATE.replace("__SUMMARY_JSON__", json.dumps(summaries, separators=(",", ":")))
        .replace("__COHORT_JSON__", json.dumps(cohort, separators=(",", ":")))
        .replace("__SERIES_JSON__", json.dumps(series, separators=(",", ":")))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
