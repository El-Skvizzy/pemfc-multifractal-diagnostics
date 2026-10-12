"""
analyze_results.py -- ageing trends of the multifractal metrics, their correlation with the
EIS descriptors, and export of every table to Excel.

Inputs : results/phm2014_windows.csv, results/phm2014_full.csv, results/eis_descriptors.json
Outputs: results/trends.csv, results/eis_vs_mf.csv, results/correlations.csv,
         results/phm2014_analysis.xlsx
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
RESULTS = os.path.join(PROJECT_ROOT, "results")
XLSX = os.path.join(RESULTS, "phm2014_analysis.xlsx")

MF_METRICS = ["mf_dh", "mf_dh_half", "mf_h0", "mf_h2", "mf_hm2", "mf_dh_q", "mf_asym",
              "wl_c1", "wl_c2", "wl_dh", "wl_dh_leg"]
EIS_METRICS = ["r_ohm_cross", "r_min", "r_lf", "r_p_naive", "cap_max", "f_apex", "tau_apex",
               "r_ohm_fit", "l_fit", "r_p_fit", "tau_fit", "alpha_fit", "chi2",
               "r_ohm_2", "rp1", "tau1", "a1", "rp2", "tau2", "a2", "rp_tot", "chi2_2"]


def read_csv_semicolon(path):
    return pd.read_csv(path, sep=";", decimal=",")


def trends(win: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stack, ch, var), g in win.groupby(["stack", "channel", "variant"]):
        g = g.sort_values("t_center_h")
        t = g["t_center_h"].to_numpy(float)
        for m in MF_METRICS:
            if m not in g:
                continue
            v = g[m].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(t)
            if ok.sum() < 5:
                continue
            rho, p = stats.spearmanr(t[ok], v[ok])
            sl, ic, r, pp, se = stats.linregress(t[ok], v[ok])
            # early vs late life (first / last 25 % of the windows)
            k = max(2, ok.sum() // 4)
            idx = np.argsort(t[ok])
            vs = v[ok][idx]
            early, late = float(np.mean(vs[:k])), float(np.mean(vs[-k:]))
            rows.append(dict(stack=stack, channel=ch, variant=var, metric=m, n=int(ok.sum()),
                             spearman_rho=rho, spearman_p=p, slope_per_100h=sl * 100,
                             early_mean=early, late_mean=late, delta_late_early=late - early,
                             rel_change_pct=100 * (late - early) / early if early != 0 else np.nan))
    return pd.DataFrame(rows)


def stage_table(win: pd.DataFrame, eis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eis = eis[eis["kind"] == "postpola"].copy()
    for _, e in eis.iterrows():
        stack, t_h, cur = e["stack"], float(e["t_h"]), float(e["current_A"])
        for var in win["variant"].unique():
            sub = win[(win["stack"] == stack) & (win["variant"] == var)]
            for ch in sorted(sub["channel"].unique()):
                g = sub[sub["channel"] == ch]
                if g.empty:
                    continue
                j = (g["t_center_h"] - t_h).abs().idxmin()
                w = g.loc[j]
                dt = float(abs(w["t_center_h"] - t_h))
                if dt > 90:
                    continue
                r = dict(stack=stack, channel=ch, variant=var, current_A=cur, t_stage_h=t_h,
                         t_win_center_h=float(w["t_center_h"]), dt_h=dt,
                         mean_V=float(w["mean_V"]), std_V=float(w["std_V"]))
                for m in MF_METRICS:
                    r[m] = float(w[m]) if m in w else np.nan
                for m in EIS_METRICS:
                    r[m] = float(e[m]) if m in e and pd.notna(e[m]) else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def correlations(stage: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman/Pearson correlations between multifractal and EIS descriptors.

    Scopes (all reporting n and two-sided p):
      stage_ch : per EIS stage, metrics averaged over the three characterisation currents,
                 computed separately for each channel -> n = 8 (conservative, independent
                 stages; this is the scope quoted as the main result),
      current  : per stage and per current, per channel -> n = 8, repeated for 3 currents,
      pooled   : all stage x current x channel combinations -> n = 72 (pseudoreplicated,
                 reported only as a consistency check, its p-values are NOT meaningful).
    """
    eis_phys = ["r_ohm_cross", "r_ohm_fit", "r_ohm_2", "r_p_naive", "r_p_fit", "rp_tot",
                "alpha_fit", "a1", "a2", "f_apex", "tau_apex", "cap_max"]
    recs = {"stage_ch": [], "stage_x_current": [], "pooled": []}
    for (stack, var), g in stage.groupby(["stack", "variant"]):
        for ch in sorted(g["channel"].unique()):
            gg = g[g["channel"] == ch]
            agg = gg.groupby("t_stage_h")[MF_METRICS + eis_phys].mean().reset_index()
            for _, r in agg.iterrows():
                d = r.to_dict(); d.update(stack=stack, variant=var, channel=ch)
                recs["stage_ch"].append(d)
            for _, r in gg.iterrows():
                d = r.to_dict(); d.update(stack=stack, variant=var, channel=ch)
                recs["stage_x_current"].append(d)
            for _, r in gg.iterrows():
                d = r.to_dict(); d.update(stack=stack, variant=var, channel="pooled")
                recs["pooled"].append(d)

    rows = []
    for scope, rr in recs.items():
        D = pd.DataFrame(rr)
        if D.empty:
            continue
        keys = ["stack", "variant", "channel"]
        for (stack, var, ch), dd in D.groupby(keys):
            for mf in MF_METRICS:
                for em in eis_phys:
                    d = dd[[mf, em]].dropna()
                    if d.shape[0] < 6:
                        continue
                    rho, p = stats.spearmanr(d[mf], d[em])
                    r, pp = stats.pearsonr(d[mf], d[em])
                    rows.append(dict(scope=scope, stack=stack, variant=var, channel=ch,
                                     mf_metric=mf, eis_metric=em, n=int(d.shape[0]),
                                     spearman_rho=rho, spearman_p=p,
                                     pearson_r=r, pearson_p=pp))
    return pd.DataFrame(rows)


def main():
    win = read_csv_semicolon(os.path.join(RESULTS, "phm2014_windows.csv"))
    full = read_csv_semicolon(os.path.join(RESULTS, "phm2014_full.csv"))
    with open(os.path.join(RESULTS, "eis_descriptors.json"), encoding="utf-8") as f:
        eis = pd.DataFrame(json.load(f))

    T = trends(win)
    S = stage_table(win, eis)
    C = correlations(S)

    T.to_csv(os.path.join(RESULTS, "trends.csv"), index=False, sep=";", decimal=",")
    S.to_csv(os.path.join(RESULTS, "eis_vs_mf.csv"), index=False, sep=";", decimal=",")
    C.to_csv(os.path.join(RESULTS, "correlations.csv"), index=False, sep=";", decimal=",")

    # ---------------- Excel workbook -------------------------------------------------
    eis_out = eis.copy()
    with pd.ExcelWriter(XLSX, engine="xlsxwriter") as xw:
        full.to_excel(xw, sheet_name="full_record", index=False)
        win.to_excel(xw, sheet_name="windows", index=False)
        T.to_excel(xw, sheet_name="trends", index=False)
        eis_out.to_excel(xw, sheet_name="EIS_descriptors", index=False)
        S.to_excel(xw, sheet_name="EIS_vs_multifractal", index=False)
        C.to_excel(xw, sheet_name="correlations", index=False)
        try:
            v = pd.read_csv(os.path.join(RESULTS, "validation.md"), sep="\n", header=None)
            v.to_excel(xw, sheet_name="validation_md", index=False, header=False)
        except Exception:
            pass
        wb = xw.book
        for sh in ("full_record", "windows", "trends", "EIS_descriptors",
                   "EIS_vs_multifractal", "correlations"):
            ws = xw.sheets[sh]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 40, 16)

    # ---------------- console summary -------------------------------------------------
    print("=== ageing trends (hp8h, utot) ===")
    q = T[(T.variant == "hp8h") & (T.channel == "utot")][
        ["stack", "metric", "spearman_rho", "spearman_p", "early_mean", "late_mean",
         "rel_change_pct"]]
    print(q.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    print("\n=== correlations MF vs EIS, conservative scope (per-stage mean, n=8) ===")
    cc = C[(C.scope == "stage_ch") & (C.n >= 8)].copy()
    cc["absrho"] = cc.spearman_rho.abs()
    cc = cc.sort_values("absrho", ascending=False).head(30)
    print(cc[["scope", "stack", "variant", "channel", "mf_metric", "eis_metric", "n",
              "spearman_rho", "spearman_p"]].to_string(index=False,
                                                       float_format=lambda x: f"{x:8.3f}"))

    print("\n=== same, per stage x current (n=20..24, pseudoreplicated) ===")
    cd = C[(C.scope == "stage_x_current") & (C.n >= 8)].copy()
    cd["absrho"] = cd.spearman_rho.abs()
    cd = cd[cd.absrho >= 0.70].sort_values("absrho", ascending=False).head(12)
    print(cd[["stack", "variant", "channel", "mf_metric", "eis_metric", "n",
              "spearman_rho", "spearman_p"]].to_string(index=False,
                                                       float_format=lambda x: f"{x:8.3f}"))

    # ---------------- focused tables for the report --------------------------------
    KEY_MF = ["mf_dh", "mf_dh_half", "mf_h0", "mf_h2", "mf_hm2", "wl_c1", "wl_c2", "wl_dh"]
    KEY_EIS = ["r_ohm_2", "r_p_fit", "rp_tot", "alpha_fit", "a1", "a2", "cap_max", "f_apex"]
    K = C[(C.scope == "stage_ch") & (C.n >= 6)]
    K = K[K.mf_metric.isin(KEY_MF) & K.eis_metric.isin(KEY_EIS)]
    K = K[K.channel.isin(["utot", "u1", "u5"])]
    K = K.sort_values(["stack", "variant", "channel", "mf_metric", "eis_metric"])
    K.to_csv(os.path.join(RESULTS, "key_correlations.csv"), index=False, sep=";",
             decimal=",")

    print("\n=== key correlations, utot, hp8h (n=8) ===")
    kk = K[(K.channel == "utot") & (K.variant == "hp8h")]
    print(kk[["stack", "mf_metric", "eis_metric", "n", "spearman_rho", "spearman_p",
              "pearson_r"]].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== EIS ageing trends (post-pola spectra, mean over channels n/a) ===")
    et = []
    for (stack, cur), g in eis[eis.kind == "postpola"].groupby(["stack", "current_A"]):
        g = g.sort_values("t_h")
        if g.shape[0] < 5:
            continue
        for m in ("r_ohm_2", "r_ohm_fit", "r_p_fit", "rp_tot", "alpha_fit", "a1", "a2",
                  "f_apex", "cap_max"):
            v = g[m].to_numpy(float)
            rho, p = stats.spearmanr(g["t_h"], v)
            et.append(dict(stack=stack, current_A=cur, metric=m,
                           first=float(v[0]), last=float(v[-1]),
                           rel_change_pct=100 * (v[-1] - v[0]) / v[0] if v[0] else np.nan,
                           spearman_rho=rho, spearman_p=p, n=int(g.shape[0])))
    ET = pd.DataFrame(et)
    ET.to_csv(os.path.join(RESULTS, "eis_trends.csv"), index=False, sep=";", decimal=",")
    print(ET[ET.metric.isin(["r_ohm_2", "r_p_fit", "alpha_fit", "a1", "cap_max"])]
          .to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    with pd.ExcelWriter(XLSX, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as xw:
        ET.to_excel(xw, sheet_name="EIS_trends", index=False)
        K.to_excel(xw, sheet_name="key_correlations", index=False)
    print("\nworkbook ->", XLSX)


if __name__ == "__main__":
    sys.exit(main())
