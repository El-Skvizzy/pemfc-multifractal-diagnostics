"""
prepare_phm2014.py -- load the IEEE PHM 2014 (FCLAB) ageing logs, characterise their
(non-uniform) time base, resample onto a uniform grid and cache everything to .npz.

Data source used here: public GitHub mirror of the official dataUBFC archive
(doi:10.25666/DATAUBFC-2021-07-19, CC BY-NC-SA 3.0), folder `data/FC1_FC2_Excel`.
FC1 = steady state at 0.7 A/cm^2, FC2 = same stack design but with a 5 kHz triangular
current ripple.  Columns (positional, the published headers are mis-encoded):
   0 Time (h) | 1..5 U1..U5 (V) | 6 Utot (V) | 7 J (A/cm2) | 8 I (A)
   9..14 temperatures | 15..18 pressures | 19..23 flows | 24 RH_air (%)
All electrical channels are logged at 1 Hz in the raw logs; the published Excel export is
decimated to ~30 s, which is verified (not assumed) in the diagnostics section below.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time as _time

import numpy as np
import openpyxl

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DATA = os.path.join(PROJECT_ROOT, "data", "phm2014")
OUT = os.path.join(PROJECT_ROOT, "data", "cache")
os.makedirs(OUT, exist_ok=True)

COLS = (["t_h"] + [f"u{i}" for i in range(1, 6)] + ["utot", "j", "i"] +
        ["tin_h2", "tout_h2", "tin_air", "tout_air", "tin_wat", "tout_wat",
         "pin_air", "pout_air", "pout_h2", "pin_h2",
         "din_h2", "dout_h2", "din_air", "dout_air", "dwat", "rh_air"])

FILES = {
    "FC1": [
        ("xlsx", "data__FC1_FC2_Excel__FC1_Without_Ripples_Excel__FC1_Ageing_part1.xlsx"),
        ("xlsx", "data__FC1_FC2_Excel__FC1_Without_Ripples_Excel__FC1_Ageing_part2.xlsx"),
        ("xlsx", "data__FC1_FC2_Excel__FC1_Without_Ripples_Excel__FC1_Ageing_part3.xlsx"),
    ],
    "FC2": [
        ("xlsx", "data__FC1_FC2_Excel__FC2_With_Ripples_Excel__FC2_Ageing_part1.xlsx"),
        ("csv", "data__FC1_FC2_Excel__FC2_With_Ripples_Excel__FC2_Ageing_part2.csv"),
    ],
}


def _f(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_xlsx(path: str) -> np.ndarray:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r is None or r[0] is None:
            continue
        rows.append([_f(v) for v in r[:len(COLS)]])
    wb.close()
    return np.asarray(rows, dtype=float)


def read_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f, delimiter=";")
        next(rd)
        for row in rd:
            if not row or row[0].strip() == "":
                continue
            rows.append([_f(v) for v in row[:len(COLS)]])
    return np.asarray(rows, dtype=float)


def load_stack(name: str) -> np.ndarray:
    parts = []
    for kind, fn in FILES[name]:
        p = os.path.join(DATA, fn)
        t0 = _time.time()
        a = read_xlsx(p) if kind == "xlsx" else read_csv(p)
        print(f"  read {fn[:70]} -> {a.shape} in {_time.time()-t0:.1f}s", flush=True)
        parts.append(a)
    a = np.vstack(parts)
    # sort by time, drop duplicated / non-increasing timestamps
    o = np.argsort(a[:, 0], kind="stable")
    a = a[o]
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    print(f"  {name}: rows={a.shape[0]} after dedup={int(keep.sum())} "
          f"t=[{a[0,0]:.3f}, {a[-1,0]:.3f}] h", flush=True)
    return a[keep]


def main():
    report = {}
    for name in ("FC1", "FC2"):
        print(f"=== {name} ===", flush=True)
        a = load_stack(name)
        t_h = a[:, 0]
        t_s = t_h * 3600.0
        dt = np.diff(t_s)

        # --- diagnostics of the time base -------------------------------------------
        frac = {f"dt<{thr}s": float(np.mean(dt < thr)) for thr in (0.5, 1.5, 5, 15, 45, 75)}
        diag = dict(
            n=int(t_s.size), t_start_h=float(t_h[0]), t_end_h=float(t_h[-1]),
            dt_median_s=float(np.median(dt)),
            dt_p01=float(np.percentile(dt, 1)), dt_p99=float(np.percentile(dt, 99)),
            dt_min=float(dt.min()), dt_max=float(dt.max()),
            frac_small_dt=frac,
            n_gaps_gt_60s=int(np.sum(dt > 60)), gap_total_h=float(dt[dt > 60].sum() / 3600),
        )
        # effective uniform step = mode of the dt histogram
        hist, edges = np.histogram(dt, bins=400)
        i = int(np.argmax(hist))
        step = float(0.5 * (edges[i] + edges[i + 1]))
        diag["mode_dt_s"] = step
        diag["frac_at_mode"] = float(hist[i] / dt.size)
        print("  diag:", json.dumps(diag, indent=None)[:400], flush=True)

        # --- resample on a uniform grid ---------------------------------------------
        # knot spacing ~ the modal sampling interval, but never finer than 30 s
        gstep = max(30.0, round(step))
        grid = np.arange(0.0, t_s[-1], gstep)
        out = {"t_s": grid, "t_h": grid / 3600.0, "step_s": np.array([gstep])}
        for ci, cn in enumerate(COLS):
            if cn in ("t_h",):
                continue
            v = a[:, ci]
            good = np.isfinite(v)
            out[cn] = np.interp(grid, t_s[good], v[good])
        # data-coverage indicator: distance to the nearest real sample
        idx = np.searchsorted(t_s, grid)
        idx = np.clip(idx, 0, t_s.size - 1)
        dist = np.abs(t_s[idx] - grid)
        out["age_s"] = dist
        np.savez_compressed(os.path.join(OUT, f"phm2014_{name}.npz"), **out)
        diag["grid_step_s"] = gstep
        diag["n_grid"] = int(grid.size)
        diag["grid_frac_interp_gt_60s"] = float(np.mean(dist > 60))
        diag["stack"] = name
        report[name] = diag
        print(f"  cached -> phm2014_{name}.npz  grid={grid.size} step={gstep}s", flush=True)

    with open(os.path.join(OUT, "phm2014_timing.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
