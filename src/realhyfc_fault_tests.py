"""
realhyfc_extra.py -- two additions that complete the fault-diagnosis and per-cell EIS parts
of the study on the RealHyFC / ZSW dataset (Zenodo 20715542, CC BY 4.0):

A. SENSITIVITY TESTS  (PartialSensitivityTest_Humidity_*, PartialSensitivityTest_Stoic_*)
   These files hold stepwise variations of the cathode dew-point (i.e. of the relative
   humidity -> flooding/drying-like conditions) and of the cathode stoichiometry
   (-> starvation-like conditions) at begin-of-life and after 2300 h.  Each file is streamed
   once; rows are grouped into blocks of constant set point; every block long enough for a
   multifractal estimate (>= 8192 samples = 2.3 h at 1 Hz) is analysed with MFDFA + WLMA.
   This gives the *fault-condition* response of the multifractal metrics, which is what a
   diagnosis method needs.

B. PER-CELL EIS -> PER-CELL MULTIFRACTALITY
   EIS.zip contains cell-resolved impedance spectra at BoL, 1150 h and 2300 h (files
   01..10 = cells 1..10, file 00 = reference cell).  Each spectrum is fitted with the
   R-L-CPE and R-L-CPE2 models; the per-cell descriptors are then correlated (n = 10 cells)
   with the per-cell multifractal metrics of the voltage-deviation signal obtained from the
   load-cycle windows at the same stages.

Outputs: results/realhyfc_sensitivity.csv, results/realhyfc_cell_eis.csv,
         results/realhyfc_cell_correlations.csv, results/realhyfc_extra.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time as _time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifractal_methods as mf  # noqa: E402
import eis_fitting as eismod  # noqa: E402

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DATA = os.path.join(PROJECT_ROOT, "data")
PHEIS = os.path.join(PROJECT_ROOT, "data", "realhyfc_EIS", "EIS")
RESULTS = os.path.join(PROJECT_ROOT, "results")
QS = np.linspace(-6.0, 6.0, 25)
MIN_BLOCK = 8192


def num(b) -> float:
    if isinstance(b, bytes):
        s = b.strip()
        if not s:
            return np.nan
        if b"," in s and b"." in s:
            s = s.replace(b",", b"")
        elif b"," in s:
            s = s.replace(b",", b".")
    else:
        s = str(b).strip()
        if not s:
            return np.nan
        if "," in s and "." in s:
            s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------------------
# A. sensitivity tests
# --------------------------------------------------------------------------------------
def highpass(x, w=3600):
    x = np.asarray(x, float)
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    half = w // 2
    i = np.arange(n)
    lo = np.clip(i - half, 0, n)
    hi = np.clip(i + half, 0, n)
    return x - (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def metrics(x):
    r = mf.mfdfa(x, qs=QS, smin=40, smax=4096, nscale=30, m=1)
    w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=12, qs=QS)
    return dict(mf_h0=r["h0"], mf_dh=r["dh"], mf_dh5=r["dh_5"], mf_half=r["dh_half"],
                mf_h2=r["h_plus"], mf_hm2=r["h_minus"], mf_dh_q=r["dh_q"],
                wl_c1=w["c1"], wl_c2=w["c2"], wl_c3=w["c3"], wl_dh5=w["dh_cum_5"])


NEED = ["OpHrs [h]", "I.S.act [A]", "j.S.act [A/cm2]", "Stoic.S.C [-]", "Stoic.S.C.set [-]",
        "U.S.AveCell [V]", "MIN(`U.S.MinCell [V]`)"] + [f"U{i} [V]" for i in range(1, 11)] + \
       ["DPT.Si.A.set [°C]", "DPT.Si.C.set [°C]"]
SHORT = ["OpHrs", "I_act", "j", "StoicC", "StoicC_set", "Uave", "Umin"] + \
        [f"U{i}" for i in range(1, 11)] + ["DPT_A_set", "DPT_C_set"]
WIN_S = 2048        # 34 min at 1 Hz
HOP_S = 1024


def read_levels(path: str, kind: str):
    """Return {level: (OpHrs, Uave, Ud_2d)} keeping only rows whose set point is stable."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n").split(";")
        try:
            idx = [header.index(n) for n in NEED]
        except ValueError as e:
            print("   [warn] column missing:", e, flush=True)
            return {}
        kmax = max(idx)
        rows = []
        for line in f:
            p = line.rstrip(b"\r\n").split(b";")
            if len(p) <= kmax:
                continue
            v = [num(p[i]) for i in idx]
            if not np.isfinite(v).all():
                continue
            rows.append(v)
    if not rows:
        return {}
    A = np.asarray(rows, float)
    if kind == "humidity":
        key = np.round(A[:, SHORT.index("DPT_C_set")])
    else:
        key = np.round(A[:, SHORT.index("StoicC_set")] * 10) / 10.0
    # hysteresis: a level change is accepted only after HYS consecutive identical samples
    HYS = 5
    stable = np.empty(key.size, bool)
    cur = key[0]
    run = 0
    cand = key[0]
    for i, k in enumerate(key):
        if k == cur:
            run = 0
        else:
            if k == cand:
                run += 1
            else:
                cand, run = k, 1
            if run >= HYS:
                cur = k
                run = 0
            else:
                k = cur
        stable[i] = (k == cur)
        key[i] = cur
    A, key = A[stable], key[stable]
    ix = SHORT.index("Uave")
    uix = [SHORT.index(f"U{i}") for i in range(1, 11)]
    out = {}
    for lev in np.unique(key):
        m = key == lev
        if m.sum() < WIN_S:
            continue
        Uave = A[m, ix]
        Ud = A[np.ix_(m, uix)]
        Ud = Ud - Ud.mean(axis=1, keepdims=True)
        out[float(lev)] = (A[m, SHORT.index("OpHrs")], Uave, Ud)
    return out


def metrics_short(x):
    n = x.size
    smax = int(min(512, max(64, n // 4)))
    r = mf.mfdfa(x, qs=QS, smin=20, smax=smax, nscale=16, m=1)
    j2 = int(max(4, min(9, np.floor(np.log2(n)) - 2)))
    w = mf.wlma(x, integrate=True, wavelet="db3", j1=2, j2=j2, qs=QS)
    return dict(mf_h0=r["h0"], mf_dh=r["dh"], mf_dh5=r["dh_5"], mf_half=r["dh_half"],
                mf_h2=r["h_plus"], mf_hm2=r["h_minus"], mf_r2=float(np.nanmean(r["r2"])),
                wl_c1=w["c1"], wl_c2=w["c2"], wl_c3=w["c3"], wl_dh5=w["dh_cum_5"],
                smax=smax, j1=2, j2=j2)


def sensitivity_file(path: str, kind: str, verbose=False):
    """Sliding-window multifractal analysis inside each constant-setpoint level."""
    lv = read_levels(path, kind)
    out = []
    for lev, (t, Uave, Ud) in sorted(lv.items()):
        recs = []
        n = Uave.size
        for s0 in range(0, n - WIN_S + 1, HOP_S):
            x = Uave[s0:s0 + WIN_S]
            x = x - np.convolve(x, np.ones(3600 // 4) / (3600 // 4), mode="same")
            d = metrics_short(x)
            per_cell = [metrics_short(Ud[s0:s0 + WIN_S, i] - 0.0) for i in range(Ud.shape[1])]
            for k in per_cell[0]:
                d["dU_" + k] = float(np.mean([q[k] for q in per_cell]))
            d["Ucell_mean"] = float(Uave[s0:s0 + WIN_S].mean())
            d["t_h"] = float(t[s0])
            recs.append(d)
        if not recs:
            continue
        R = pd.DataFrame(recs)
        agg = dict(file=os.path.basename(path), kind=kind, level=lev,
                   n_windows=int(R.shape[0]), duration_h=float(n / 3600.0))
        for c in R.columns:
            if c == "t_h":
                continue
            agg[c + "_mean"] = float(R[c].mean())
            agg[c + "_sd"] = float(R[c].std())
        out.append(agg)
        if verbose:
            print(f"   level {lev:8.2f}: {R.shape[0]:3d} windows, "
                  f"Δh5={agg['mf_dh5_mean']:.3f}±{agg['mf_dh5_sd']:.3f}, "
                  f"h0={agg['mf_h0_mean']:.3f}, U={agg['Ucell_mean_mean']:.4f} V", flush=True)
    return out


def _block_result(buf, names, kind, key, path):
    A = {n: np.asarray(v, float) for n, v in buf.items()}
    ucols = [f"U{i}" for i in range(1, 11)]
    Ud = np.column_stack([A[c] for c in ucols])
    Umean = Ud.mean(axis=1)
    res = dict(file=os.path.basename(path), kind=kind, setpoint=key,
               t_start_h=float(A["OpHrs"][0]), t_end_h=float(A["OpHrs"][-1]),
               n=int(A["OpHrs"].size), duration_h=float(A["OpHrs"][-1] - A["OpHrs"][0]),
               I_mean=float(A["I_act"].mean()), j_mean=float(A["j"].mean()),
               Uave_mean=float(A["Uave"].mean()), Umin_mean=float(A["Umin"].mean()),
               Ucell_mean=float(Umean.mean()), Ucell_std_across_cells=float(Ud.mean(axis=0).std()),
               DPT_A_set=float(A["DPT_A_set"].mean()), DPT_C_set=float(A["DPT_C_set"].mean()),
               stoic_C_set=float(A["StoicC_set"].mean()))
    # stack-average cell voltage, 1 h high-passed
    res.update({("Uave_" + k): v for k, v in metrics(highpass(Umean, 3600)).items()})
    # mean over the ten cells of the deviation-from-mean metrics
    per_cell = [metrics(a - Umean) for a in (Ud[:, i] for i in range(Ud.shape[1]))]
    for k in per_cell[0]:
        res["dU_" + k] = float(np.mean([q[k] for q in per_cell]))
    return res


def part_a():
    files = [("humidity", os.path.join(DATA, "realhyfc_PartialSensitivityTest_Humidity_BoL.csv")),
             ("humidity", os.path.join(DATA, "realhyfc_PartialSensitivityTest_Humidity_2300h.csv")),
             ("stoic", os.path.join(DATA, "realhyfc_PartialSensitivityTest_Stoic_BoL.csv")),
             ("stoic", os.path.join(DATA, "realhyfc_PartialSensitivityTest_Stoic_2300h.csv"))]
    rows = []
    for kind, p in files:
        if not os.path.exists(p):
            print("  missing", p, flush=True)
            continue
        t0 = _time.time()
        r = sensitivity_file(p, kind, verbose=True)
        rows += r
        print(f"  {os.path.basename(p)}: {len(r)} blocks in {_time.time()-t0:.0f}s", flush=True)
    W = pd.DataFrame(rows)
    W.to_csv(os.path.join(RESULTS, "realhyfc_sensitivity.csv"), index=False, sep=";",
             decimal=",")
    return W


# --------------------------------------------------------------------------------------
# B. per-cell EIS
# --------------------------------------------------------------------------------------
def read_cell_eis(path):
    """RealHyFC EIS file: (Number, f/Hz, |Z|/Ohm, Phase/deg, Significance, Time, U, I)."""
    rows = []
    started = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not started:
                if "Frequency" in line:
                    started = True
                continue
            p = line.split()
            if len(p) < 4:
                continue
            try:
                rows.append([float(p[1]), float(p[2]), float(p[3])])
            except ValueError:
                continue
    a = np.asarray(rows, float)
    if a.size == 0:
        raise ValueError("no data")
    a = a[np.argsort(-a[:, 0])]
    f, zabs, ph = a[:, 0], a[:, 1], np.deg2rad(a[:, 2])
    re = zabs * np.cos(ph)
    im3 = -zabs * np.sin(ph)          # = -Im Z  (positive for a capacitive arc)
    return f, re, im3


def part_b():
    rows = []
    for stage in ("BoL", "1150h", "2300h"):
        for folder in sorted(glob.glob(os.path.join(PHEIS, stage, "*"))):
            if not os.path.isdir(folder):
                continue
            for p in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                m = re.search(r"tst(\d+)\.txt$", os.path.basename(p).lower())
                if not m:
                    continue
                cell = int(m.group(1))
                if cell == 0:
                    continue                      # 00 = reference cell (per ReadMe)
                try:
                    f, rez, im3 = read_cell_eis(p)
                    one = eismod.fit_single_arc(f, rez, im3)
                    two = eismod.fit_two_arc(f, rez, im3)
                except Exception as e:
                    print("   [skip]", os.path.basename(p), type(e).__name__, str(e)[:70])
                    continue
                cap = float(im3.max())
                f_apex = float(f[int(np.argmax(im3))])
                rows.append(dict(stage=stage, folder=os.path.basename(folder), cell=cell,
                                 I_nominal_A=float(re.search(r"^(\d+)A", os.path.basename(folder)).group(1))
                                 if re.match(r"^\d+A", os.path.basename(folder)) else np.nan,
                                 n_points=int(f.size), cap_max=cap, f_apex=f_apex,
                                 r_lf=float(rez[-1]), r_min=float(rez.min()),
                                 r_p_naive=float(rez[-1] - rez.min()), **one, **two))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(RESULTS, "realhyfc_cell_eis.csv"), index=False, sep=";",
             decimal=",")
    return E


# --------------------------------------------------------------------------------------
# C. per-cell correlation
# --------------------------------------------------------------------------------------
def part_c(E: pd.DataFrame):
    W = pd.read_csv(os.path.join(RESULTS, "realhyfc_windows.csv"), sep=";", decimal=",")
    W = W[(W.variant == "dU") & (W.channel.str.match(r"^U\d+$"))]
    W["cell"] = W.channel.str.replace("U", "").astype(int)
    stage_map = {"early": "BoL", "mid": "1150h", "late": "2300h"}
    rows = []
    mf_metrics = ["mf_dh5", "mf_h0", "mf_h2", "mf_hm2", "wl_c1", "wl_c2", "wl_dh5"]
    eis_metrics = ["r_ohm_fit", "r_p_fit", "alpha_fit", "r_ohm_2", "rp_tot", "a1", "a2",
                   "cap_max", "f_apex", "r_lf"]
    for wstage, estate in stage_map.items():
        ww = W[W.stage == wstage]
        if ww.empty:
            continue
        for folder, ee in E[E.stage == estate].groupby("folder"):
            ee = ee[ee.cell.isin(ww.cell.unique())]
            if ee.shape[0] < 8:
                continue
            m = ww.merge(ee, on="cell", suffixes=("", "_eis"))
            for a in mf_metrics:
                for b in eis_metrics:
                    if a not in m or b not in m:
                        continue
                    d = m[[a, b]].dropna()
                    if d.shape[0] < 8:
                        continue
                    rho, p = stats.spearmanr(d[a], d[b])
                    r, pp = stats.pearsonr(d[a], d[b])
                    rows.append(dict(window_stage=wstage, eis_stage=estate, folder=folder,
                                     mf_metric=a, eis_metric=b, n=int(d.shape[0]),
                                     spearman_rho=rho, spearman_p=p, pearson_r=r,
                                     pearson_p=pp))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(RESULTS, "realhyfc_cell_correlations.csv"), index=False,
             sep=";", decimal=",")
    return C


def main():
    t0 = _time.time()
    print("=== A. sensitivity tests (fault-like conditions) ===", flush=True)
    S = part_a()
    print(f"  blocks: {len(S)}", flush=True)
    if len(S):
        cols = [c for c in ["kind", "file", "level", "n_windows", "duration_h",
                            "Ucell_mean_mean", "mf_dh5_mean", "mf_dh5_sd", "mf_h0_mean",
                            "mf_h2_mean", "mf_hm2_mean", "wl_c1_mean", "wl_c2_mean",
                            "dU_mf_dh5_mean", "dU_wl_c1_mean", "dU_wl_c2_mean"]
                if c in S.columns]
        print(S[cols].round(4).to_string(index=False))
    print("=== B. per-cell EIS ===", flush=True)
    E = part_b()
    print(f"  spectra: {len(E)}", flush=True)
    if len(E):
        print(E.groupby(["stage", "folder"]).size().to_string())
    print("=== C. per-cell correlations ===", flush=True)
    C = part_c(E) if len(E) else pd.DataFrame()
    if len(C):
        CC = C.copy()
        CC["absrho"] = CC.spearman_rho.abs()
        top = CC.sort_values("absrho", ascending=False).head(20)
        print(top[["window_stage", "folder", "mf_metric", "eis_metric", "n",
                   "spearman_rho", "spearman_p"]].round(3).to_string(index=False))
    meta = dict(sensitivity_records=S.to_dict(orient="records") if len(S) else [],
                cell_eis_records=E.to_dict(orient="records") if len(E) else [],
                runtime_s=_time.time() - t0)
    with open(os.path.join(RESULTS, "realhyfc_extra.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=float)
    print("done in %.0f s" % meta["runtime_s"])


if __name__ == "__main__":
    sys.exit(main())
