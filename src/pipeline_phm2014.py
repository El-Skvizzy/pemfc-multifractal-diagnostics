"""
run_phm2014.py -- multifractal analysis of the IEEE PHM 2014 (FCLAB) PEMFC ageing data.

Pipeline
--------
1. Load the cached uniform 30 s grid of the FC1 (steady state) and FC2 (with current
   ripple) ageing logs: stack voltage Utot + the five individual cell voltages U1..U5.
2. Two signal variants per record:
      raw   : as published (contains the slow degradation drift and the operating-point
              variations),
      hp8h  : high-pass = x - moving average over 960 samples (8 h), which removes the
              degradation drift and keeps the two-phase-flow / water-management
              intermittency that is the physical object of interest.
3. Full-record multifractal analysis with
      * MFDFA (this work, cross-validated against the MFDFA package and theory),
      * WLMA  (this work, cross-validated against pymultifracs and theory),
      * MFDCCA between selected channel pairs (cross-correlation exponents).
4. Sliding-window analysis (L = 2^14 samples = 136 h, stride 2^12 = 34 h) giving the
   time evolution of the multifractal metrics over the whole life of both stacks.
5. Numerical null references: (i) monofractal fGn with the empirical H at the same N
   (noise floor of the estimators), (ii) IAAFT surrogates of a 2^16 sub-record
   (multifractality from the PDF vs from the correlations).

Outputs: results/phm2014_*.csv, results/phm2014_full.json
"""
from __future__ import annotations

import json
import os
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifractal_methods as mf  # noqa: E402

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DATA = os.path.join(PROJECT_ROOT, "data", "cache")
RESULTS = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

QS = np.linspace(-6.0, 6.0, 25)   # uniform grid: required by the discrete Legendre transform
CHANNELS = ["utot", "u1", "u2", "u3", "u4", "u5"]
HP_W = 960          # 8 h high-pass window at 30 s sampling
WIN_L = 2 ** 14     # 16384 samples = 136.5 h
WIN_STRIDE = 2 ** 12


def highpass(x, w=HP_W):
    """x minus a centred moving average of length w (edge-safe)."""
    x = np.asarray(x, float)
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    half = w // 2
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half, 0, n)
    ma = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return x - ma


def metrics_full(x, tag, smax=None):
    """MFDFA + WLMA descriptors of one record."""
    n = x.size
    r = mf.mfdfa(x, qs=QS, smin=40, smax=smax, nscale=30, m=1)
    w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=12, qs=QS)
    out = dict(tag=tag, n=int(n),
               mf_h0=r["h0"], mf_dh=r["dh"], mf_dh_half=r["dh_half"], mf_asym=r["asym"],
               mf_dh_q=r["dh_q"], mf_h2=r["h_plus"], mf_hm2=r["h_minus"],
               mf_h0q=r["h0_q"], mf_r2_mean=float(np.nanmean(r["r2"])),
               mf_smin=r["smin"], mf_smax=r["smax"],
               mf_dh5=r["dh_5"], mf_dh5_half=r["dh_half_5"], mf_h05=r["h0_5"],
               mf_closed=r["closed"], mf_closed5=r["closed_5"],
               wl_c1=w["c1"], wl_c2=w["c2"], wl_c3=w["c3"],
               wl_h0=w["h0_cum"], wl_dh=w["dh_cum"], wl_dh_half=w["dh_half_cum"],
               wl_asym=w["asym_cum"], wl_dh_q=w["dh_q"],
               wl_dh5=w["dh_cum_5"], wl_h05=w["h0_cum_5"], wl_closed=w["closed_cum"],
               wl_dh_leg=w["dh"], wl_r2_mean=float(np.nanmean(w["r2"])),
               wl_j1=w["j1"], wl_j2=w["j2"])
    return out, r, w


def null_monofractal(n, H, seed, smax=None):
    """MFDFA + WLMA of a monofractal fGn with the same H and N (estimator noise floor)."""
    rng = np.random.default_rng(seed)
    x = mf.fgn_local(n, H, rng) if hasattr(mf, "fgn_local") else None
    if x is None:
        # simple spectral synthesis of fGn (see validate_methods.fgn)
        from validate_methods import fgn as _fgn
        x = _fgn(n, H, rng)
    r = mf.mfdfa(x, qs=QS, smin=40, smax=smax, nscale=30, m=1)
    w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=12, qs=QS)
    return dict(H=H, mf_dh=r["dh"], mf_dh_half=r["dh_half"], wl_dh=w["dh_cum"],
                wl_c2=w["c2"], mf_h0=r["h0"], wl_c1=w["c1"])


def main():
    t0 = _time.time()
    report = {}
    full_rows = []
    win_rows = []

    for stack in ("FC1", "FC2"):
        d = np.load(os.path.join(DATA, f"phm2014_{stack}.npz"))
        t_h = d["t_h"]
        print(f"=== {stack}: n={t_h.size} span {t_h[0]:.1f}-{t_h[-1]:.1f} h ===", flush=True)
        base_utot = d["utot"]

        for var, fn in (("raw", lambda v: np.asarray(v, float)),
                        ("hp8h", lambda v: highpass(np.asarray(v, float), HP_W))):
            for ch in CHANNELS:
                x = fn(d[ch])
                row, _, _ = metrics_full(x, f"{stack}|{ch}|{var}", smax=4096)
                row.update(stack=stack, channel=ch, variant=var)
                full_rows.append(row)
                print(f"  [{var}] {ch}: MFDFA dh={row['mf_dh']:.3f} h0={row['mf_h0']:.3f} "
                      f"h(2)={row['mf_h2']:.3f} | WLMA c1={row['wl_c1']:.4f} "
                      f"c2={row['wl_c2']:.4f} dh={row['wl_dh']:.3f}  ({_time.time()-t0:.0f}s)",
                      flush=True)

        # ---- monofractal nulls at the full-record length ---------------------------
        nulls = []
        for ch in ("utot", "u1", "u5"):
            x = highpass(np.asarray(d[ch], float), HP_W)
            H = mf.mfdfa(x, qs=np.array([2.0]), smin=40, smax=4096, nscale=30)["h"][0]
            for seed in (1, 2, 3):
                nulls.append(dict(channel=ch, **null_monofractal(x.size, float(H), seed,
                                                                 smax=4096)))
        report[f"{stack}_nulls"] = nulls
        print(f"  nulls: MFDFA dh={np.mean([q['mf_dh'] for q in nulls]):.3f} "
              f"WLMA dh={np.mean([q['wl_dh'] for q in nulls]):.3f}", flush=True)

        # ---- sensitivity to the scaling range (hp8h, Utot) ------------------------
        sens = []
        x = highpass(np.asarray(d["utot"], float), HP_W)
        for smax in (2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15):
            r = mf.mfdfa(x, qs=QS, smin=40, smax=smax, nscale=24, m=1)
            w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=int(np.log2(smax)),
                        qs=QS)
            sens.append(dict(stack=stack, smax=smax, smax_hours=smax * 30 / 3600,
                             mf_dh=float(r["dh"]), mf_h0=float(r["h0"]),
                             mf_h2=float(r["h_plus"]), mf_hm2=float(r["h_minus"]),
                             mf_dh_q=float(r["dh_q"]),
                             wl_c1=float(w["c1"]), wl_c2=float(w["c2"]),
                             wl_dh=float(w["dh_cum"])))
            print(f"  range smax=2^{int(np.log2(smax))} ({smax*30/3600:.0f} h): "
                  f"dh={r['dh']:.3f} h0={r['h0']:.3f} h(2)={r['h_plus']:.3f} "
                  f"h(-2)={r['h_minus']:.3f} | WLMA c1={w['c1']:.3f} c2={w['c2']:.4f}",
                  flush=True)
        report[f"{stack}_scale_range_sensitivity"] = sens

        # ---- sliding windows ------------------------------------------------------
        for var, fn in (("raw", lambda v: np.asarray(v, float)),
                        ("hp8h", lambda v: highpass(np.asarray(v, float), HP_W))):
            starts = list(range(0, t_h.size - WIN_L + 1, WIN_STRIDE))
            for ch in CHANNELS:
                x = fn(d[ch])
                for s0 in starts:
                    seg = x[s0:s0 + WIN_L]
                    row, _, _ = metrics_full(seg, f"{stack}|{ch}|{var}", smax=WIN_L // 4)
                    row.update(stack=stack, channel=ch, variant=var,
                               t_start_h=float(t_h[s0]),
                               t_center_h=float(t_h[s0 + WIN_L // 2]),
                               t_end_h=float(t_h[min(s0 + WIN_L - 1, t_h.size - 1)]),
                               mean_V=float(np.mean(seg)), std_V=float(np.std(seg)))
                    win_rows.append(row)
            print(f"  windows [{var}] done ({len(starts)} starts)  {_time.time()-t0:.0f}s",
                  flush=True)

        # ---- MFDCCA on selected pairs (BoL / EoL halves) --------------------------
        hs = highpass(np.asarray(d["utot"], float), HP_W)
        h1 = highpass(np.asarray(d["u1"], float), HP_W)
        h5 = highpass(np.asarray(d["u5"], float), HP_W)
        hI = highpass(np.asarray(d["i"], float), HP_W)
        half = t_h.size // 2
        dcca = []
        for seg_name, sl in (("BoL_half", slice(0, 2 ** 16)),
                             ("EoL_half", slice(t_h.size - 2 ** 16, t_h.size))):
            for name, a, b in (("u1_vs_u5", h1, h5), ("utot_vs_i", hs, hI),
                               ("utot_vs_u1", hs, h1)):
                res = mf.mfdcca(a[sl], b[sl], qs=QS, smin=40, smax=4096, nscale=20)
                dcca.append(dict(stack=stack, segment=seg_name, pair=name,
                                 dh=float(res["dh"]), h0=float(res["h0"]),
                                 dh_q=float(res["dh_q"]),
                                 h_plus=float(np.interp(2.0, QS, res["hxy"])),
                                 h_minus=float(np.interp(-2.0, QS, res["hxy"])),
                                 rho_dcca_s256=float(mf.dcca_coefficient(a[sl], b[sl], 256)),
                                 rho_dcca_s1024=float(mf.dcca_coefficient(a[sl], b[sl], 1024))))
            print(f"  MFDCCA {seg_name} done", flush=True)
        report[f"{stack}_mfdcca"] = dcca

    Full = pd.DataFrame(full_rows)
    Win = pd.DataFrame(win_rows)
    Full.to_csv(os.path.join(RESULTS, "phm2014_full.csv"), index=False, sep=";",
                decimal=",")
    Win.to_csv(os.path.join(RESULTS, "phm2014_windows.csv"), index=False, sep=";",
               decimal=",")
    report["windows"] = Win.to_dict(orient="records")
    report["full"] = Full.to_dict(orient="records")
    report["runtime_s"] = _time.time() - t0
    with open(os.path.join(RESULTS, "phm2014_full.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=float)
    print("saved results/phm2014_full.csv, phm2014_windows.csv, phm2014_full.json")
    print("runtime %.1f s" % report["runtime_s"])


if __name__ == "__main__":
    sys.exit(main())
