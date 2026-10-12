"""
figures.py -- all figures of the study (PNG, 200 dpi) into results/figures/.

Fig 01  degradation of the stack and cell voltages (PHM 2014 FC1/FC2)
Fig 02  singularity spectra D(h) at begin-of-life vs end-of-life (FC1/FC2)
Fig 03  h(q) and tau(q) for FC1 / FC2, and the exact reference curves
Fig 04  ageing trajectory of the multifractal width vs the EIS resistances
Fig 05  EIS Nyquist plots with CPE fits at three ageing stages
Fig 06  surrogate test (shuffle / IAAFT) for a real voltage window
Fig 07  method validation: spurious multifractality vs record length; c1 vs H
Fig 08  RealHyFC per-cell multifractal widths at three ageing stages
"""
from __future__ import annotations

import ast
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifractal_methods as mf  # noqa: E402
import eis_fitting as eismod  # noqa: E402

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
R = os.path.join(PROJECT_ROOT, "results")
DATA = os.path.join(PROJECT_ROOT, "data")
CACHE = os.path.join(DATA, "cache")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
QS = np.array([-5., -4., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.])
WIN = 2 ** 14
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def hp(x, w=960):
    x = np.asarray(x, float)
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    half = w // 2
    i = np.arange(n)
    lo = np.clip(i - half, 0, n); hi = np.clip(i + half, 0, n)
    return x - (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def series(stack):
    return np.load(os.path.join(CACHE, f"phm2014_{stack}.npz"))


def fig01():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, stack in enumerate(("FC1", "FC2")):
        d = series(stack)
        t = d["t_h"]
        step = max(1, t.size // 4000)
        for c in [f"u{i}" for i in range(1, 6)]:
            ax[k].plot(t[::step], d[c][::step], lw=0.5, alpha=0.75, label=c.upper())
        ax[k].plot(t[::step], d["utot"][::step], lw=0.8, color="k", label="Utot")
        ax[k].set_title(f"{stack}  ({'steady state' if stack=='FC1' else '5 kHz current ripple'})")
        ax[k].set_xlabel("time [h]"); ax[k].set_ylabel("voltage [V]")
        ax[k].legend(ncol=2, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig01_degradation.png")); plt.close(fig)


def spectra_for(x, tag):
    r = mf.mfdfa(x, qs=QS, smin=40, smax=WIN // 4, nscale=24, m=1)
    w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=None, qs=QS)
    return r, w


def fig02():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, stack in enumerate(("FC1", "FC2")):
        d = series(stack)
        n = d["t_h"].size
        xmin, xmax = 9e9, -9e9
        for ch, ls in (("utot", "-"), ("u1", "--"), ("u3", ":"), ("u5", "-.")):
            for lab, sl, col in (("BoL", slice(0, WIN), "tab:blue"),
                                 ("EoL", slice(n - WIN, n), "tab:red")):
                x = hp(d[ch][sl], 960)
                r, w = spectra_for(x, f"{stack}{lab}{ch}")
                axes[k].plot(r["alpha"], r["f"], color=col, ls=ls, lw=1.2,
                             label=f"{ch.upper()} {lab} ($\\Delta h$={r['dh']:.3f})")
                xmin = min(xmin, np.nanmin(r["alpha"])); xmax = max(xmax, np.nanmax(r["alpha"]))
        axes[k].axhline(0, color="gray", lw=0.6)
        axes[k].set_title(f"{stack}: singularity spectrum D(h), 8 h high-pass "
                          f"({'steady state' if stack == 'FC1' else '5 kHz ripple'})")
        axes[k].set_xlabel(r"$h$"); axes[k].set_ylabel(r"$D(h)$")
        axes[k].set_xlim(xmin - 0.05, xmax + 0.05)
        axes[k].set_ylim(-0.25, 1.2)
        axes[k].legend(fontsize=6, ncol=2)
        axes[k].annotate(f"$\\Delta h$ range: {xmax - xmin:.2f}", xy=(0.02, 0.03),
                         xycoords="axes fraction", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig02_spectra_BoL_EoL.png")); plt.close(fig)


def fig03():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for stack, col in (("FC1", "tab:blue"), ("FC2", "tab:red")):
        d = series(stack)
        x = hp(d["utot"][:WIN], 960)
        r, w = spectra_for(x, stack)
        ax[0].plot(r["qs"], r["h"], col, lw=1.4,
                   label=f"{stack} MFDFA $h(q)$ ($\\Delta h$={r['dh']:.3f})")
        ax[0].plot(w["qs"], w["zeta_cum"], col, ls=":", lw=1.2,
                   label=f"{stack} WLMA $\\zeta(q)$")
        ax[1].plot(r["qs"], r["tau"], col, lw=1.4, label=f"{stack} MFDFA $\\tau(q)$")
        ax[1].plot(w["qs"], w["tau_cum"], col, ls=":", lw=1.2, label=f"{stack} WLMA $\\tau(q)$")
    ax[0].set_xlabel("q"); ax[0].set_ylabel("h(q) / zeta(q)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("q"); ax[1].set_ylabel(r"$\tau(q)$"); ax[1].legend(fontsize=7)
    ax[1].plot([-5, 5], [-5, 5], "k--", lw=0.6)
    ax[1].plot([-5, 5], [-6, 4], color="gray", lw=0.8, label="tau=qH-1 (H=1)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig03_hq_tauq.png")); plt.close(fig)


def fig04():
    W = pd.read_csv(os.path.join(R, "phm2014_windows.csv"), sep=";", decimal=",")
    eis = pd.DataFrame(json.load(open(os.path.join(R, "eis_descriptors.json"),
                                      encoding="utf-8")))
    meta = json.load(open(os.path.join(R, "phm2014_full.json"), encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex=False)
    for k, stack in enumerate(("FC1", "FC2")):
        null = float(np.mean([q["mf_dh"] for q in meta.get(f"{stack}_nulls", [])])) \
            if meta.get(f"{stack}_nulls") else np.nan
        for v, var in enumerate(("hp8h", "raw")):
            ax = axes[k, v]
            g = W[(W["stack"] == stack) & (W.channel == "utot") & (W.variant == var)]
            g = g.sort_values("t_center_h")
            y = g.mf_dh5 if "mf_dh5" in g else g.mf_dh
            ax.plot(g.t_center_h, y, "o-", ms=3, lw=1.0, color="tab:blue",
                    label=r"MFDFA $\Delta h_5$ ($|q|\leq 5$)")
            if np.isfinite(null):
                ax.axhline(null, color="tab:gray", ls=":", lw=1.2,
                           label=f"monofractal null = {null:.3f}")
            ax.plot(g.t_center_h, g.mf_dh5.rolling(5, center=True, min_periods=2).mean(),
                    "-", color="k", lw=1.6, alpha=0.6, label="5-window mean")
            ax.set_ylabel(r"$\Delta h$")
            ax.set_title(f"{stack}, {var}")
            if k == 1:
                ax.set_xlabel("time [h]")
            ax2 = ax.twinx()
            e = eis[(eis["stack"] == stack) & (eis.kind == "postpola") &
                    (eis.current_A == 70)].sort_values("t_h")
            ax2.plot(e.t_h, e.r_p_fit * 1e3, "^-", color="tab:red", ms=5,
                     label=r"$R_p$ (EIS, 70 A) [m$\Omega$]")
            ax2.grid(False)
            ax2.set_ylabel(r"EIS $R_p$ [m$\Omega$]", color="tab:red")
            ax2.tick_params(axis="y", colors="tab:red")
            if k == 0 and v == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig04_dh_trajectory_vs_EIS.png"))
    plt.close(fig)


def fig05():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    d = os.path.join(DATA, "phm2014")
    names = {"FC1": "data__FC1_FC2_Excel__FC1_Without_Ripples_Excel__FC1_EIS70A_postpola_T{}.xlsx",
             "FC2": "data__FC1_FC2_Excel__FC2_With_Ripples_Excel__FC2_EIS70A_postpola_T{}.xlsx"}
    stages = {"FC1": ["000", "515", "991"], "FC2": ["000", "515", "1016"]}
    for k, stack in enumerate(("FC1", "FC2")):
        ax = axes[k]
        for t, col in zip(stages[stack], ("tab:blue", "tab:green", "tab:red")):
            p = os.path.join(d, names[stack].format(t))
            if not os.path.exists(p):
                continue
            f, re, im3 = eismod.read_spectrum(p)
            dsc = eismod.descriptors(p)
            ax.plot(re * 1e3, im3 * 1e3, "o", ms=2.5, color=col, label=f"t={t} h")
            w = 2 * np.pi * f
            z = eismod.model_double(w, [dsc["r_ohm_2"], dsc["l_2"], dsc["rp1"], dsc["tau1"],
                                        dsc["a1"], dsc["rp2"], dsc["tau2"], dsc["a2"]])
            ax.plot(z.real * 1e3, -z.imag * 1e3, "-", lw=1.0, color=col, alpha=0.8)
        ax.set_title(f"{stack}: EIS at 70 A, CPE-2 fit")
        ax.set_xlabel(r"Re $Z$ [m$\Omega$]"); ax.set_ylabel(r"$-$Im $Z$ [m$\Omega$]")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 25); ax.set_ylim(-1, 8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig05_eis_nyquist_fits.png")); plt.close(fig)


def fig06():
    d = series("FC1")
    x = hp(d["utot"][:WIN], 960)
    r0 = mf.mfdfa(x, qs=QS, smin=40, smax=WIN // 4, nscale=24)
    rng = np.random.default_rng(3)
    sh = [mf.mfdfa(mf.shuffle_surrogate(x, rng), qs=QS, smin=40, smax=WIN // 4,
                   nscale=24)["dh"] for _ in range(15)]
    ia = [mf.mfdfa(mf.iaaft_surrogate(x, rng, n_iter=20), qs=QS, smin=40, smax=WIN // 4,
                   nscale=24)["dh"] for _ in range(15)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sh, bins=8, alpha=0.65, label="shuffled (15)", color="tab:gray")
    ax.hist(ia, bins=8, alpha=0.65, label="IAAFT (15)", color="tab:orange")
    ax.axvline(r0["dh"], color="tab:red", lw=2, label=f"measured $\\Delta h$={r0['dh']:.3f}")
    ax.set_xlabel(r"$\Delta h$"); ax.set_ylabel("count")
    ax.set_title("FC1 Utot, begin of life: origin of the multifractality")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig06_surrogates.png")); plt.close(fig)


def fig07():
    val = json.load(open(os.path.join(R, "validation.json"), encoding="utf-8"))
    D = pd.DataFrame(val["experiments"]["D_finite_size"])
    A = pd.DataFrame(val["experiments"]["A_monofractal_fgn"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for H, g in D.groupby("H"):
        ax[0].errorbar(g.N, g.dh_mfdfa, yerr=g.dh_mfdfa_sd, marker="o", capsize=3,
                       label=f"MFDFA H={H}")
        ax[0].errorbar(g.N, g.dh_wlma, yerr=g.dh_wlma_sd, marker="s", ls="--", capsize=3,
                       label=f"WLMA H={H}")
    ax[0].set_xscale("log", base=2); ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("record length N"); ax[0].set_ylabel(r"$\Delta h$")
    ax[0].set_title("Spurious multifractality of a MONOfractal signal")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].scatter(A.H, A.h_mine, label="MFDFA h(2)", marker="o")
    ax[1].scatter(A.H, A.c1_mine, label="WLMA c1", marker="^")
    ax[1].scatter(A.H, A.get("c1_pm"), label="pymultifracs c1", marker="x")
    ax[1].plot([0.2, 0.8], [0.2, 0.8], "k--", lw=0.8, label="h = H")
    ax[1].set_xlabel("H (input)"); ax[1].set_ylabel("estimate")
    ax[1].set_title("Absolute calibration on fGn, N = 2^16")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig07_validation.png")); plt.close(fig)


def fig08():
    p = os.path.join(R, "realhyfc_windows.csv")
    if not os.path.exists(p):
        print("realhyfc_windows.csv not available yet - skipping fig08")
        return
    W = pd.read_csv(p, sep=";", decimal=",")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v, var in enumerate(("hp1h", "dU")):
        ax = axes[v]
        g = W[(W.variant == var) & (W.channel != "Ustack") & (W.channel != "U6")]
        for stage, col in zip(["early", "mid", "late"], ("tab:blue", "tab:green", "tab:red")):
            gg = g[g.stage == stage].sort_values("channel")
            if gg.empty:
                continue
            ax.plot(range(1, gg.shape[0] + 1), gg.mf_dh, "o-", color=col, label=f"{stage}")
            ax.plot(range(1, gg.shape[0] + 1), gg.wl_dh, "s--", color=col, alpha=0.6)
        ax.set_xlabel("cell index"); ax.set_ylabel(r"$\Delta h$")
        ax.set_title(f"RealHyFC 10-cell stack, {var}  "
                     f"(solid = MFDFA $\\Delta h$, dashed = WLMA $\\Delta h$)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig08_realhyfc_cells.png")); plt.close(fig)


if __name__ == "__main__":
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        try:
            f()
            print("ok", f.__name__, flush=True)
        except Exception as e:
            print("FAIL", f.__name__, type(e).__name__, str(e)[:200], flush=True)
