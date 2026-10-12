"""
validate_methods.py -- validation of the multifractal toolbox (src/multifractal_methods.py) on synthetic
signals whose multifractal properties are known exactly, and cross-validation against
two independent published implementations:

  * MFDFA 0.4.3 (LRydin/MFDFA, MIT)            -> independent MFDFA fluctuation function
  * pymultifracs 0.3.1 (neurospin, MIT)        -> independent wavelet-leader formalism

Experiments
-----------
A. fGn/fBm with known Hurst exponent H: MFDFA must return h(q) = H (independent of q) and
   a degenerate spectrum (Delta_h -> 0).  This is the absolute-reference test.
B. Binomial multiplicative cascade (p-model): truly multifractal, exact mass exponent
   tau_m(q) = -log2(p^q + (1-p)^q).  Used to (i) fix the sign/convention of the MFDFA
   correspondence and (ii) measure the agreement of both methods with the exact spectrum.
C. Cross-method agreement: tau(q) from MFDFA vs tau(q) from WLMA on the same series.
D. Finite-size / spurious-multifractality study: Delta_h vs record length N for a
   MONOfractal signal (0.12 < H < 0.9) -- quantifies the artefact documented by
   Lopez & Contreras, Phys. Rev. E 87, 022918 (2013) (MFDFA reliable only for N >~ 2^16,
   |q| < 5), and the same test for WLMA.
E. Surrogate tests: shuffle (destroys correlations) vs IAAFT (keeps both amplitude
   distribution and power spectrum) -- separates "multifractality from broad PDF" from
   "multifractality from long-range correlations".

Outputs: results/validation.json, results/validation.md
"""
from __future__ import annotations

import json
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifractal_methods as mf  # noqa: E402

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
RESULTS = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

QS = np.linspace(-5.0, 5.0, 21)   # uniform q grid
# the MFDFA package silently drops |q| <= 0.1 (q = 0 does not converge in its formulation)
QS_PKG = QS[np.abs(QS) > 0.1]

try:
    import MFDFA as mfdfa_pkg
except Exception as e:  # pragma: no cover
    mfdfa_pkg = None
    print("MFDFA package unavailable:", e)

try:
    from pymultifracs import wavelet_analysis as pm_wavelet_analysis
    from pymultifracs.mf_analysis import mfa as pm_mfa
except Exception as e:  # pragma: no cover
    pm_wavelet_analysis = None
    pm_mfa = None
    print("pymultifracs unavailable:", e)


# --------------------------------------------------------------------------------------
# synthetic signals
# --------------------------------------------------------------------------------------
def fgn(N, H, rng):
    """Fractional Gaussian noise with Hurst exponent H (Davies-Harte via spectral method)."""
    if mfdfa_pkg is not None:
        # the package ships a fgn generator; make it reproducible
        st = np.random.get_state()
        np.random.seed(rng.integers(0, 2 ** 31 - 1))
        try:
            x = np.asarray(mfdfa_pkg.fgn(N, H), dtype=float).ravel()
        finally:
            np.random.set_state(st)
        if x.size == N and np.all(np.isfinite(x)):
            return x - x.mean()
    # fallback: circulant embedding
    k = np.arange(0, N)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    lam = np.fft.fft(np.concatenate([g, g[-2:0:-1]])).real
    lam = np.maximum(lam, 0.0)
    z = rng.normal(size=2 * N) + 1j * rng.normal(size=2 * N)
    w = np.fft.fft(z) * np.sqrt(lam / (2 * N))
    y = np.fft.ifft(w).real[:N]
    return y - y.mean()


def binom_cascade(nlevel, p, rng=None):
    """Binomial multiplicative cascade (p-model) measure weights, length 2**nlevel."""
    w = np.ones(1)
    for _ in range(nlevel):
        w = np.concatenate([w * p, w * (1.0 - p)])
    return w


def tau_measure(q, p):
    """Exact mass exponent of the binomial measure: Z_n(q) = (p^q+(1-p)^q)^n."""
    q = np.asarray(q, float)
    return -np.log2(p ** q + (1.0 - p) ** q)


def alpha_of_q(q, p):
    dq = 1e-5
    return (tau_measure(q + dq, p) - tau_measure(q - dq, p)) / (2 * dq)


def spectrum_exact(p, nq=4001):
    q = np.linspace(-40, 40, nq)
    a = alpha_of_q(q, p)
    t = tau_measure(q, p)
    f = q * a - t
    o = np.argsort(a)
    return a[o], f[o]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def slopes_from_F(lags, F, qs):
    """h(q) from an externally computed fluctuation matrix F (nlag, nq)."""
    out = np.full(qs.size, np.nan)
    for j in range(qs.size):
        with np.errstate(divide="ignore", invalid="ignore"):
            ly = np.log(F[:, j])
        ok = np.isfinite(ly)
        out[j] = np.polyfit(np.log(np.asarray(lags, float)[ok]), ly[ok], 1)[0]
    return out


def spectrum_from_h(qs, h):
    tau = qs * h - 1.0
    return mf.spectrum_from_tau(qs, tau)


def my_mfdfa(x, **kw):
    return mf.mfdfa(x, qs=QS, smin=kw.pop("smin", 40), smax=kw.pop("smax", None),
                    nscale=kw.pop("nscale", 30), m=kw.pop("m", 1), **kw)


def pkg_mfdfa(x, scales, q):
    lags, F = mfdfa_pkg.MFDFA(np.asarray(x, float), np.asarray(scales), 1, np.asarray(q, float))
    return np.asarray(lags), np.asarray(F)


def pm_wlma(x, j1=3, j2=None):
    """Independent WLMA (wavelet leaders) via pymultifracs, same integrated semantics."""
    wt = pm_wavelet_analysis(np.asarray(x, float), wt_name="db3", j2=j2)
    mrq = wt.get_leaders(p_exp=np.inf)
    res = pm_mfa(mrq, scaling_ranges=[(j1, j2 if j2 is not None else (int(np.log2(x.size)) - 3))],
                 n_cumul=3, estimates="auto", check_regularity=False)
    c = res.cumulants
    c3 = getattr(c, "c3", None)
    return dict(c1=float(np.asarray(c.c1)), c2=float(np.asarray(c.c2)),
                c3=float(np.asarray(c3)) if c3 is not None else np.nan)


# --------------------------------------------------------------------------------------
def main():
    t00 = _time.time()
    out = {"experiments": {}, "packages": {
        "MFDFA": None if mfdfa_pkg is None else getattr(mfdfa_pkg, "__version__", "0.4.3"),
        "pymultifracs": None if pm_mfa is None else "0.3.1"}}
    md = ["# Validation of the multifractal toolbox\n"]

    # ---------------- A. monofractal fGn: h(q) = H ------------------------------------
    print("=== A. fGn monofractal ===", flush=True)
    A = []
    for H in (0.2, 0.5, 0.8):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            x = fgn(2 ** 16, H, rng)
            r1 = my_mfdfa(x)
            scales = mf.log_scales(x.size, smin=40, nscale=30)
            lags, F = pkg_mfdfa(x, scales, QS_PKG)
            h_pkg = slopes_from_F(lags, F, QS_PKG)
            # WLMA on the profile (= cumsum), both my implementation and pymultifracs
            w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=None, qs=QS)
            A.append(dict(H=H, seed=seed,
                          h_mine=float(np.interp(2.0, QS, r1["h"])),
                          h_pkg=float(np.interp(2.0, QS_PKG, h_pkg)),
                          h_std_mine=float(np.std(r1["h"][np.abs(QS) >= 1])),
                          dh_mine=float(r1["dh"]), dh_half_mine=float(r1["dh_half"]),
                          c1_mine=float(w["c1"]), c2_mine=float(w["c2"])))
            if pm_mfa is not None:
                try:
                    pm = pm_wlma(x, j1=3, j2=int(np.log2(x.size)) - 3)
                    A[-1].update(c1_pm=pm["c1"], c2_pm=pm["c2"])
                except Exception as e:
                    A[-1]["c1_pm"] = None; A[-1]["c1_pm_err"] = str(e)[:120]
            print(f"  H={H} seed={seed}: h_mine={A[-1]['h_mine']:.3f} h_pkg={A[-1]['h_pkg']:.3f} "
                  f"dh={A[-1]['dh_mine']:.3f} c1={A[-1]['c1_mine']:.3f} "
                  f"c2={A[-1]['c2_mine']:.4f} c1_pm={A[-1].get('c1_pm')} "
                  f"err_pm={A[-1].get('c1_pm_err','')}", flush=True)
    out["experiments"]["A_monofractal_fgn"] = A
    err_mine = [abs(a["h_mine"] - a["H"]) for a in A]
    err_pkg = [abs(a["h_pkg"] - a["H"]) for a in A]
    err_c1 = [abs(a["c1_mine"] - a["H"]) for a in A]
    md += ["## A. Monofractal fGn, N = 2^16, MFDFA h(q=2) vs known H\n",
           f"- |h(2)_mine - H|: mean {np.mean(err_mine):.4f}, max {np.max(err_mine):.4f}",
           f"- |h(2)_MFDFA-package - H|: mean {np.mean(err_pkg):.4f}, max {np.max(err_pkg):.4f}",
           f"- |c1_WLMA - H|: mean {np.mean(err_c1):.4f}, max {np.max(err_c1):.4f}",
           f"- mean Delta_h (spectrum width, must be ~0 for a monofractal): "
           f"{np.mean([a['dh_mine'] for a in A]):.3f} +- {np.std([a['dh_mine'] for a in A]):.3f}",
           f"- mean c2 (must be ~0): {np.mean([a['c2_mine'] for a in A]):.3e} "
           f"+- {np.std([a['c2_mine'] for a in A]):.3e}\n"]

    # ---------------- B. binomial cascade ---------------------------------------------
    print("=== B. binomial cascade ===", flush=True)
    B = []
    for p in (0.6, 0.7, 0.75):
        x = binom_cascade(16, p)
        r1 = my_mfdfa(x)
        scales = mf.log_scales(x.size, smin=40, nscale=30)
        lags, F = pkg_mfdfa(x, scales, QS_PKG)
        h_pkg = slopes_from_F(lags, F, QS_PKG)
        w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=13, qs=QS)
        # exact spectrum and candidate correspondences
        a_ex, f_ex = spectrum_exact(p)
        mm = f_ex >= 0
        dh_exact = float(a_ex[mm].max() - a_ex[mm].min())
        h0_exact = float(a_ex[np.argmax(f_ex)])
        cand = {}
        for sign, name in ((+1.0, "h=(1+tau(+q))/q"), (-1.0, "h=(1+tau(-q))/q")):
            qq = sign * QS
            with np.errstate(divide="ignore", invalid="ignore"):
                h_th = np.where(np.abs(QS) < 1e-12, np.nan,
                                (1.0 + tau_measure(qq, p)) / np.where(np.abs(QS) < 1e-12, np.nan, QS))
            err = np.nanmax(np.abs(h_th - r1["h"]))
            cand[name] = float(err)
        tau_mine = r1["tau"]
        tau_pkg = QS_PKG * h_pkg - 1.0
        tau_wlma = w["tau"]
        ok = np.abs(QS) >= 1
        h_mine_on_pkg = np.interp(QS_PKG, QS, r1["h"])
        tau_mine_on_pkg = QS_PKG * h_mine_on_pkg - 1.0
        B.append(dict(p=p, dh_exact=dh_exact, h0_exact=h0_exact,
                      dh_mfdfa=float(r1["dh"]), h0_mfdfa=float(r1["h0"]),
                      dh_wlma=float(w["dh"]), h0_wlma=float(w["h0"]),
                      max_abs_dtau_mine_vs_pkg=float(np.max(np.abs(tau_mine_on_pkg - tau_pkg))),
                      max_abs_dtau_mine_vs_wlma=float(np.max(np.abs(tau_mine[ok] - tau_wlma[ok]))),
                      c1=float(w["c1"]), c2=float(w["c2"]), c3=float(w["c3"]),
                      cand_err=cand))
        if pm_mfa is not None:
            try:
                pmv = pm_wlma(x, j1=3, j2=13)
                B[-1].update(c1_pm=pmv["c1"], c2_pm=pmv["c2"])
            except Exception as e:
                B[-1]["c1_pm"] = None
        print(f"  p={p}: dh_exact={dh_exact:.3f} dh_mfdfa={B[-1]['dh_mfdfa']:.3f} "
              f"dh_wlma={B[-1]['dh_wlma']:.3f} max|dtau_mine-pkg|={B[-1]['max_abs_dtau_mine_vs_pkg']:.3f} "
              f"max|dtau_mine-wlma|={B[-1]['max_abs_dtau_mine_vs_wlma']:.3f} "
              f"c1={B[-1]['c1']:.3f} c2={B[-1]['c2']:.4f} cand={ {k: round(v,3) for k,v in cand.items()} }", flush=True)
    out["experiments"]["B_binomial_cascade"] = B
    md += ["## B. Binomial cascade (exact tau_m(q) = -log2(p^q+(1-p)^q))\n",
           "| p | Delta_h exact | Delta_h MFDFA | Delta_h WLMA | max\\|d tau\\| mine vs pkg | max\\|d tau\\| MFDFA vs WLMA |",
           "|---|---|---|---|---|---|"]
    for b in B:
        md.append(f"| {b['p']} | {b['dh_exact']:.3f} | {b['dh_mfdfa']:.3f} | {b['dh_wlma']:.3f} | "
                  f"{b['max_abs_dtau_mine_vs_pkg']:.4f} | {b['max_abs_dtau_mine_vs_wlma']:.4f} |")
    md.append("")
    md.append("Candidate MFDFA<->measure correspondence, max |h_theory - h_estimated|:")
    for b in B:
        md.append(f"- p={b['p']}: " + ", ".join(f"{k}: {v:.3f}" for k, v in b["cand_err"].items()))
    md.append("")

    # ---------------- D. finite-size spurious multifractality -------------------------
    print("=== D. finite-size study ===", flush=True)
    D = []
    for H in (0.2, 0.5, 0.8):
        for nlev in (12, 14, 16):
            N = 2 ** nlev
            vals = []
            for seed in (11, 12, 13):
                rng = np.random.default_rng(seed)
                x = fgn(N, H, rng)
                s_min = max(8, min(40, N // 32))
                r1 = mf.mfdfa(x, qs=QS, smin=s_min, smax=N // 4, nscale=24)
                w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=nlev - 3, qs=QS)
                vals.append((r1["dh"], r1["dh_half"], w["dh"], w["dh_half"], w["c2"], r1["h_minus"], r1["h_plus"]))
            v = np.array(vals)
            D.append(dict(H=H, N=N, nlev=nlev, smin=s_min,
                          dh_mfdfa=float(v[:, 0].mean()), dh_mfdfa_sd=float(v[:, 0].std()),
                          dh_mfdfa_half=float(v[:, 1].mean()),
                          dh_wlma=float(v[:, 2].mean()), dh_wlma_sd=float(v[:, 3].std()),
                          dh_wlma_half=float(v[:, 3].mean()),
                          c2_wlma=float(v[:, 4].mean()),
                          h_minus=float(v[:, 5].mean()), h_plus=float(v[:, 6].mean())))
            print(f"  H={H} N=2^{nlev}: dh_mfdfa={D[-1]['dh_mfdfa']:.3f}+-{D[-1]['dh_mfdfa_sd']:.3f} "
                  f"dh_wlma={D[-1]['dh_wlma']:.3f}+-{D[-1]['dh_wlma_sd']:.3f} c2={D[-1]['c2_wlma']:.4f}",
                  flush=True)
    out["experiments"]["D_finite_size"] = D
    md += ["## D. Spurious multifractality of a MONOfractal signal vs record length N\n",
           "| H | N | Delta_h MFDFA | sd | Delta_h MFDFA (f>=0.5) | Delta_h WLMA | sd | c2 WLMA |",
           "|---|---|---|---|---|---|---|---|"]
    for d in D:
        md.append(f"| {d['H']} | 2^{d['nlev']} | {d['dh_mfdfa']:.3f} | {d['dh_mfdfa_sd']:.3f} | "
                  f"{d['dh_mfdfa_half']:.3f} | {d['dh_wlma']:.3f} | {d['dh_wlma_sd']:.3f} | {d['c2_wlma']:.4f} |")
    md.append("")

    # ---------------- E. surrogates ---------------------------------------------------
    print("=== E. surrogates ===", flush=True)
    E = []
    rng = np.random.default_rng(7)
    for name, x in (("binomial p=0.75", binom_cascade(16, 0.75)),
                    ("fGn H=0.8", fgn(2 ** 16, 0.8, rng))):
        r0 = my_mfdfa(x)
        shuf, iaaft = [], []
        for k in range(10):
            r = np.random.default_rng(100 + k)
            shuf.append(my_mfdfa(mf.shuffle_surrogate(x, r))["dh"])
            iaaft.append(my_mfdfa(mf.iaaft_surrogate(x[:2 ** 14], r, n_iter=20))["dh"])
        E.append(dict(signal=name, dh_orig=float(r0["dh"]),
                      dh_shuffle_mean=float(np.mean(shuf)), dh_shuffle_sd=float(np.std(shuf)),
                      dh_iaaft_mean=float(np.mean(iaaft)), dh_iaaft_sd=float(np.std(iaaft))))
        print(f"  {name}: orig={E[-1]['dh_orig']:.3f} shuffle={E[-1]['dh_shuffle_mean']:.3f}"
              f"+-{E[-1]['dh_shuffle_sd']:.3f} iaaft={E[-1]['dh_iaaft_mean']:.3f}+-{E[-1]['dh_iaaft_sd']:.3f}",
              flush=True)
    out["experiments"]["E_surrogates"] = E
    md += ["## E. Surrogate test (origin of multifractality)\n",
           "| signal | Delta_h original | Delta_h shuffle (10) | Delta_h IAAFT (10) |",
           "|---|---|---|---|"]
    for e in E:
        md.append(f"| {e['signal']} | {e['dh_orig']:.3f} | {e['dh_shuffle_mean']:.3f} +- "
                  f"{e['dh_shuffle_sd']:.3f} | {e['dh_iaaft_mean']:.3f} +- {e['dh_iaaft_sd']:.3f} |")
    md.append("")

    out["runtime_s"] = _time.time() - t00
    with open(os.path.join(RESULTS, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS, "validation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("saved results/validation.json and results/validation.md; runtime %.1fs" % out["runtime_s"])


if __name__ == "__main__":
    sys.exit(main())
