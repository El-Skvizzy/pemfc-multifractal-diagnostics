"""
mf.py -- Multifractal analysis toolbox for PEMFC voltage signals.

Implements, from scratch and with a single consistent convention:

  1. MFDFA  -- Multifractal Detrended Fluctuation Analysis (Kantelhardt et al., 2002)
               applied to a *stationary increment series* x(t).
  2. WLMA   -- Wavelet Leader Multifractal Analysis (Jaffard, Lashermes, Abry, Wendt)
               applied to the *profile* Y(t) = cumsum(x - mean(x)), so that the object
               analysed is the same function as in MFDFA (fair cross-comparison).
  3. MFDCCA -- Multifractal Detrended Cross-Correlation Analysis (Zhou & Leung 2007;
               Podobnik & Stanley 2008) between two increment series.
  4. Surrogates (IAAFT / shuffle) for testing the origin of multifractality.
  5. Spectrum metrics: h0 (=alpha_0), Delta_h, asymmetry, h(q) range.

Conventions used everywhere in this file
----------------------------------------
For a series x of length N we build the profile Y(i) = sum_{k<=i} (x_k - mean(x)).
MFDFA gives F_q(s) ~ s^{h(q)}, hence

        tau(q) = q*h(q) - 1 ,      alpha = dtau/dq ,     f(alpha) = q*alpha - tau(q).

For a monofractal (fBm profile with Hurst H, i.e. fGn increments) h(q) = H, tau(q) = qH - 1
and f(alpha) degenerates to a point at alpha = H with f = 1 (support dimension 1).
WLMA is made consistent with this by using the wavelet-leader scaling exponents
zeta(q) (S(j,q) ~ 2^{j*zeta(q)}) and tau(q) = zeta(q) - 1.
For a fBm with Hurst H the leaders scale as L ~ 2^{jH}, hence zeta(q) = qH, c1 = H.
Because both methods are calibrated on the same synthetic references (validate.py),
no magic constants are hard-coded: the Legendre transform is always done numerically.

Author: research pipeline for the PEMFC multifractal study.
Python 3.12, numpy/scipy/pywt only.
"""

from __future__ import annotations

import numpy as np

try:  # PyWavelets is used by WLMA only
    import pywt
except Exception:  # pragma: no cover
    pywt = None


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def log_scales(n: int, smin: int = 8, smax: int | None = None, nscale: int = 24):
    """Logarithmically spaced integer detrending window sizes."""
    if smax is None:
        smax = n // 4
    smax = int(min(smax, n // 4))
    smin = int(max(smin, 4))
    if smax <= smin:
        raise ValueError(f"scale range empty: smin={smin} smax={smax} n={n}")
    s = np.unique(np.round(np.logspace(np.log10(smin), np.log10(smax), nscale)).astype(int))
    return s


# --------------------------------------------------------------------------------------
# 1. MFDFA
# --------------------------------------------------------------------------------------
def _segment_variances(y: np.ndarray, s: int, m: int) -> np.ndarray:
    """
    Variances F^2(v,s) of the locally detrended profile for all 2*N_s segments
    (forward and backward segmentation, as in Kantelhardt et al. 2002).
    """
    n = y.size
    ns = n // s
    if ns < 2:
        raise ValueError("window too short for this scale")
    idx = np.arange(s, dtype=float)

    # forward
    seg = y[: ns * s].reshape(ns, s).T                     # (s, ns)
    coef = np.polyfit(idx, seg, m)                         # (m+1, ns)
    fit = np.polyval(coef, idx[:, None])                   # (s, ns)
    var_f = np.mean((seg - fit) ** 2, axis=0)              # (ns,)

    # backward
    seg = y[n - ns * s:][::-1].reshape(ns, s).T
    coef = np.polyfit(idx, seg, m)
    fit = np.polyval(coef, idx[:, None])
    var_b = np.mean((seg - fit) ** 2, axis=0)

    return np.concatenate([var_f, var_b])


def mfdfa(
    x: np.ndarray,
    qs=None,
    scales=None,
    m: int = 1,
    smin: int = 8,
    smax: int | None = None,
    nscale: int = 24,
    qmax: float = 10.0,
    nq: int = 41,
    return_fluct: bool = False,
):
    """
    Multifractal DFA of the increment series x.

    Returns dict with:
      qs, scales, Fq (nq x nscale), h (nq,), se_h (nq,), r2 (nq,),
      q, tau, alpha, f, h0, dh, dh_half, asym, dh_q (h(-qmax)-h(+qmax)),
      h_minus (h(-2)), h_plus (h(2)), h0_q (h(0))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    y = np.cumsum(x)
    n = y.size

    if qs is None:
        qs = np.unique(np.concatenate([-np.logspace(np.log10(qmax), 0, nq // 2 + 1),
                                       np.array([0.0]), np.logspace(0, np.log10(qmax), nq // 2 + 1)]))
    qs = np.asarray(qs, dtype=float)
    if scales is None:
        scales = log_scales(n, smin=smin, smax=smax, nscale=nscale)
    scales = np.asarray(scales)

    Fq = np.empty((qs.size, scales.size))
    for i, s in enumerate(scales):
        var = _segment_variances(y, int(s), m)
        var = np.maximum(var, 1e-300)
        for j, q in enumerate(qs):
            if abs(q) < 1e-12:
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(var)))
            else:
                Fq[j, i] = np.mean(var ** (q / 2.0)) ** (1.0 / q)

    lx = np.log(scales.astype(float))
    h = np.empty(qs.size)
    se = np.empty(qs.size)
    r2 = np.empty(qs.size)
    for j in range(qs.size):
        ly = np.log(Fq[j])
        A = np.vstack([lx, np.ones_like(lx)]).T
        coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        h[j] = coef[0]
        if ly.size > 2:
            se[j] = np.sqrt(ss_res / (ly.size - 2) / np.sum((lx - lx.mean()) ** 2))
        else:
            se[j] = np.nan

    tau = qs * h - 1.0
    alpha, f, h0, dh, dh_half, asym, closed = spectrum_from_tau(qs, tau)
    # conservative width restricted to the MFDFA validation range |q| <= 5
    m5 = np.abs(qs) <= 5.0 + 1e-9
    if m5.sum() >= 5:
        _, _, h0_5, dh5, dh5_half, _, closed5 = spectrum_from_tau(qs[m5], tau[m5])
    else:
        h0_5 = dh5 = dh5_half = np.nan
        closed5 = False
    out = dict(
        qs=qs, scales=scales, Fq=Fq, h=h, se_h=se, r2=r2,
        tau=tau, alpha=alpha, f=f,
        h0=h0, dh=dh, dh_half=dh_half, asym=asym, closed=closed,
        h0_5=h0_5, dh_5=dh5, dh_half_5=dh5_half, closed_5=closed5,
        dh_q=float(h[0] - h[-1]),
        h_minus=float(np.interp(-2.0, qs, h)),
        h_plus=float(np.interp(2.0, qs, h)),
        h0_q=float(np.interp(0.0, qs, h)),
        n=int(n), m=int(m), smin=float(scales.min()), smax=float(scales.max()),
    )
    if return_fluct:
        out["Fq"] = Fq
    return out


# --------------------------------------------------------------------------------------
# Legendre transform helpers
# --------------------------------------------------------------------------------------
def spectrum_from_tau(qs, tau, qgrid=None, smooth=0.002):
    """
    Numerical Legendre transform f(alpha) = q*alpha - tau(q), alpha = dtau/dq.

    tau(q) is first smoothed with a cubic smoothing spline (scipy UnivariateSpline) with a
    *small* smoothing factor, because a raw finite-difference derivative of a noisy
    tau(q) produces a ragged, multivalued alpha(q) and a broken D(h).  The spectrum is then
    evaluated parametrically on a dense q grid, which keeps f(alpha) single-valued and is
    the numerically stable way to build D(h) from an empirical mass exponent.
    With smooth=0 or without scipy the plain linear-interpolation + gradient path is used.

    Returns (alpha, f, h0, dh, dh_half, asym).
        h0      : alpha at the maximum of f(alpha) (position of the spectrum maximum)
        dh      : support width of D(h) measured where f >= 0 (standard "spectrum width")
        dh_half : width measured at f = 0.5 (robust against the noisy tails)
        asym    : (alpha_max - h0) / (h0 - alpha_min) -- right/left asymmetry of D(h)
    """
    qs = np.asarray(qs, float)
    tau = np.asarray(tau, float)
    ok = np.isfinite(tau)
    qs, tau = qs[ok], tau[ok]
    if qs.size < 3:                      # a spectrum cannot be built from <3 q values
        return (np.array([np.nan]), np.array([np.nan]), np.nan, np.nan, np.nan, np.nan,
                False)

    if qgrid is None:
        qgrid = np.linspace(qs.min(), qs.max(), 2001)
    qgrid = np.asarray(qgrid, float)

    # ---- (1) widths from the DISCRETE mass exponent ---------------------------------
    # Central differences on the measured q grid: no interpolation and no extrapolation, so
    # the accessible alpha range is exactly alpha(q_min) .. alpha(q_max).  The envelope
    # (cumulative maximum) is applied ONLY when alpha(q) is non-monotone, i.e. when the raw
    # Legendre transform folds back on itself; for the usual monotone case it would clip the
    # descending branch of D(h).
    alpha_d = np.gradient(tau, qs)
    f_d = qs * alpha_d - tau
    d_alpha = np.diff(alpha_d)
    monotone = bool(np.all(d_alpha >= 0) or np.all(d_alpha <= 0))
    o = np.argsort(alpha_d, kind="stable")
    al_d, ff_d = alpha_d[o], f_d[o]
    if not monotone:
        ff_d = np.maximum.accumulate(np.where(np.isfinite(ff_d), ff_d, -np.inf))

    def width(level, al, ff):
        m = np.isfinite(ff) & (ff >= level)
        if m.sum() < 2:
            return np.nan
        return float(al[m].max() - al[m].min())

    dh = width(0.0, al_d, ff_d)
    dh_half = width(0.5, al_d, ff_d)
    i0 = int(np.nanargmax(ff_d)) if np.any(np.isfinite(ff_d)) else -1
    h0 = float(al_d[i0]) if i0 >= 0 else np.nan
    mm = np.isfinite(ff_d) & (ff_d >= 0.0)
    if mm.sum() >= 2:
        amin, amax = float(al_d[mm].min()), float(al_d[mm].max())
        asym = (amax - h0) / (h0 - amin) if (h0 - amin) > 0 else np.nan
    else:
        asym = np.nan
    # does the spectrum actually close inside the analysed q range?
    closed = bool(np.isfinite(ff_d[0]) and np.isfinite(ff_d[-1])
                  and min(ff_d[0], ff_d[-1]) <= 0.1)

    # ---- (2) smooth parametric curve for plots --------------------------------------
    al_s, ff_s = al_d, ff_d
    if smooth and qs.size >= 6:
        try:
            from scipy.interpolate import UnivariateSpline
            var = float(np.var(tau))
            s = smooth * qs.size * var if var > 0 else 0.0
            spl = UnivariateSpline(qs, tau, k=3, s=s)
            t = spl(qgrid)
            a = spl.derivative()(qgrid)
            al_s = a
            ff_s = qgrid * a - t
        except Exception:
            pass

    return al_s, ff_s, h0, dh, dh_half, asym, closed


# --------------------------------------------------------------------------------------
# 2. WLMA
# --------------------------------------------------------------------------------------
def _leaders_from_dwt(d, j1, j2):
    """
    Wavelet leaders from DWT coefficients.

    Definition (Jaffard; Lashermes, Jaffard & Abry 2005):
        L(j,k) = sup_{lambda' subset 3*lambda_{j,k}} |d_{lambda'}|
    with the L1 normalisation that makes the leaders of a fractional Brownian motion
    scale as L(j,.) ~ 2^{jH}.  Implemented by the standard recursion used in the
    wavelet-leader literature (identical structure to pymultifracs' compute_leaders):

        L(j,k) = max( max |d_{j,k'}| over the 3 same-scale neighbours,
                      0.5 * max(L(j-1, 2k), L(j-1, 2k+1)) )

    The factor 0.5 converts the L2-normalised DWT coefficients of the finer level to the
    common (L1) normalisation.  Entries at the array edges where the 3-point window is
    incomplete are left as NaN and excluded from the structure functions.
    """
    pleader = {}
    for scale in sorted(d.keys()):
        if scale < j1:
            continue
        coefs = np.abs(np.asarray(d[scale], dtype=float))
        L = coefs.size
        win = np.full(L, np.nan)
        if L >= 3:
            win[1:-1] = np.maximum(np.maximum(coefs[:-2], coefs[1:-1]), coefs[2:])
        elif L > 0:
            win[:] = coefs
        if scale == j1 or (scale - 1) not in pleader:
            pleader[scale] = win
            continue
        prev = pleader[scale - 1]
        lo = prev[::2][:L]
        hi = prev[1::2][:L]
        n = min(L, lo.size, hi.size)
        child = np.full(L, np.nan)
        if n > 0:
            child[:n] = 0.5 * np.maximum(lo[:n], hi[:n])
        a = np.where(np.isnan(win), -np.inf, win)
        b = np.where(np.isnan(child), -np.inf, child)
        comb = np.maximum(a, b)
        comb[~np.isfinite(comb)] = np.nan
        pleader[scale] = comb
    return pleader


def wlma(
    x: np.ndarray,
    integrate: bool = True,
    wavelet: str = "db3",
    j1: int = 3,
    j2: int | None = None,
    qmax: float = 8.0,
    nq: int = 33,
    qs=None,
    normalization: str = "L1",
    return_structure: bool = False,
):
    """
    Wavelet leader multifractal analysis.

    integrate=True  -> analyse the profile Y = cumsum(x - mean(x))  (comparable to MFDFA on x)
    integrate=False -> analyse the raw series x as a function (Benouioua-style usage)

    normalization='L1' (default): the orthonormal (L2) DWT coefficients returned by
    PyWavelets scale as |d_j| ~ 2^{j(H+1/2)} for a process of Hurst exponent H; they are
    multiplied by 2^{-j/2} to obtain the L1 normalisation assumed by the wavelet-leader
    formalism, for which the leaders of a fractional Brownian motion obey L(j,.) ~ 2^{jH}
    and hence c1 = H, c2 = 0.  Set normalization='L2' to keep the raw convention
    (then c1 = H + 1/2, which is *not* the convention used in the WLMA literature).

    Returns dict with zeta(q) from the empirical structure-function slopes, the
    log-cumulant based zeta(q), c1, c2, c3 and the spectrum metrics.
    """
    if pywt is None:
        raise RuntimeError("PyWavelets is required for WLMA")
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if integrate:
        x = x - x.mean()
        y = np.cumsum(x)
    else:
        y = x - x.mean()
    n = y.size

    # --- DWT (periodisation keeps dyadic bookkeeping simple and exact for 2^J lengths)
    wave = pywt.Wavelet(wavelet)
    J = int(min(np.floor(np.log2(n)), pywt.dwt_max_level(n, wave.dec_len)))
    J = max(J, 3)
    if j2 is None:
        j2 = J - 3
    coeffs = pywt.wavedec(y, wave, mode="periodization", level=J)
    # coeffs[0] = approximation at scale 2^J; coeffs[1] = detail at scale 2^J ... coeffs[J] = scale 2^1
    d = {}
    for i, c in enumerate(coeffs[1:]):
        scale_j = J - i
        v = np.asarray(c, dtype=float)
        if normalization == "L1":
            v = v * 2.0 ** (-0.5 * scale_j)
        d[scale_j] = v
    j1 = max(j1, 1)
    j2 = min(j2, J - 2)
    if j2 <= j1 + 1:
        raise ValueError("series too short for WLMA")

    leaders = _leaders_from_dwt(d, j1, j2)
    js = np.array(sorted(leaders.keys()))
    nj = np.array([leaders[j].size for j in js], dtype=float)

    if qs is None:
        qs = np.unique(np.concatenate([-np.logspace(np.log10(qmax), 0, nq // 2 + 1),
                                       np.array([0.0]), np.logspace(0, np.log10(qmax), nq // 2 + 1)]))
    qs = np.asarray(qs, float)

    # structure functions S(j,q) (edge NaNs of the leader field are excluded)
    S = np.empty((qs.size, js.size))
    nvalid = np.empty(js.size)
    for i, j in enumerate(js):
        L = leaders[int(j)]
        ok = np.isfinite(L) & (L > 0)
        Lv = L[ok]
        nvalid[i] = Lv.size
        Lv = np.maximum(Lv, 1e-300)
        for k, q in enumerate(qs):
            if abs(q) < 1e-12:
                S[k, i] = np.exp(np.mean(np.log(Lv)))
            else:
                S[k, i] = np.mean(Lv ** q)

    lnj = js * np.log(2.0)
    zeta = np.empty(qs.size)
    r2 = np.empty(qs.size)
    for k in range(qs.size):
        ly = np.log(S[k])
        A = np.vstack([lnj, np.ones_like(lnj)]).T
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2[k] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        zeta[k] = coef[0]

    # log-cumulants C_p(j) of ln L, linear regression against ln(2^j)
    pmax = 3
    C = np.empty((pmax, js.size))
    for i, j in enumerate(js):
        l = leaders[int(j)]
        l = l[np.isfinite(l) & (l > 0)]
        l = np.log(np.maximum(l, 1e-300))
        C[0, i] = l.mean()
        C[1, i] = l.var(ddof=0)
        C[2, i] = np.mean((l - l.mean()) ** 3) if l.size > 2 else np.nan
    cp = np.empty(pmax)
    cp0 = np.empty(pmax)
    for p in range(pmax):
        yv = C[p]
        ok = np.isfinite(yv)
        A = np.vstack([lnj[ok], np.ones(ok.sum())]).T
        coef, *_ = np.linalg.lstsq(A, yv[ok], rcond=None)
        cp[p] = coef[0]
        cp0[p] = coef[1]

    c1, c2, c3 = float(cp[0]), float(cp[1]), float(cp[2])
    # zeta from log-cumulants expanded around q = 0 (leading behaviour)
    zeta_cum = qs * c1 + 0.5 * c2 * qs ** 2 + (1.0 / 6.0) * c3 * qs ** 3

    for tag, z in (("", zeta), ("_cum", zeta_cum)):
        pass

    tau = zeta - 1.0
    alpha, f, h0, dh, dh_half, asym, closed = spectrum_from_tau(qs, tau)
    tau_cum = zeta_cum - 1.0
    alpha_c, f_c, h0_c, dh_c, dh_half_c, asym_c, closed_c = spectrum_from_tau(qs, tau_cum)
    m5 = np.abs(qs) <= 5.0 + 1e-9
    if m5.sum() >= 5:
        _, _, h0_c5, dh_c5, dh_half_c5, _, closed_c5 = spectrum_from_tau(qs[m5], tau_cum[m5])
    else:
        h0_c5 = dh_c5 = dh_half_c5 = np.nan
        closed_c5 = False

    out = dict(
        qs=qs, js=js, nj=nj, S=S,
        zeta=zeta, r2=r2, tau=tau, alpha=alpha, f=f,
        h0=h0, dh=dh, dh_half=dh_half, asym=asym, closed=closed,
        c1=c1, c2=c2, c3=c3, cp=cp, cp0=cp0, nvalid=nvalid,
        zeta_cum=zeta_cum, tau_cum=tau_cum, alpha_cum=alpha_c, f_cum=f_c,
        h0_cum=h0_c, dh_cum=dh_c, dh_half_cum=dh_half_c, asym_cum=asym_c,
        closed_cum=closed_c,
        dh_cum_5=dh_c5, h0_cum_5=h0_c5, dh_half_cum_5=dh_half_c5, closed_cum_5=closed_c5,
        dh_q=float(zeta[0] - zeta[-1]),
        h0_q_equiv=float(np.interp(0.0, qs, zeta)),
        n=int(n), wavelet=wavelet, j1=int(j1), j2=int(j2), integrate=bool(integrate),
    )
    if not return_structure:
        out.pop("S", None)
    return out


# --------------------------------------------------------------------------------------
# 3. MFDCCA
# --------------------------------------------------------------------------------------
def mfdcca(x, y, qs=None, scales=None, m: int = 1, smin: int = 8, smax: int | None = None,
           nscale: int = 22, qmax: float = 8.0, nq: int = 33):
    """
    Multifractal detrended cross-correlation analysis between increment series x and y.
    Negative local covariances are handled by the signed-moment convention
    F_q(s) = { 1/(2Ns) sum sign(f2)|f2|^{q/2} }^{1/q}.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    X = np.cumsum(x - x.mean()); Y = np.cumsum(y - y.mean())
    if qs is None:
        qs = np.unique(np.concatenate([-np.logspace(np.log10(qmax), 0, nq // 2 + 1), np.array([0.0]), np.logspace(0, np.log10(qmax), nq // 2 + 1)]))
    qs = np.asarray(qs, float)
    if scales is None:
        scales = log_scales(n, smin=smin, smax=smax, nscale=nscale)
    species = np.asarray(scales)

    Fq = np.empty((qs.size, species.size))
    for i, s in enumerate(species):
        s = int(s)
        ns = n // s
        if ns < 2:
            raise ValueError("window too short")
        idx = np.arange(s, dtype=float)
        f2 = np.empty(2 * ns)
        for rev in (0, 1):
            if rev == 0:
                sx = X[: ns * s].reshape(ns, s).T
                sy = Y[: ns * s].reshape(ns, s).T
            else:
                sx = X[n - ns * s:][::-1].reshape(ns, s).T
                sy = Y[n - ns * s:][::-1].reshape(ns, s).T
            cx = np.polyfit(idx, sx, m); fx = np.polyval(cx, idx[:, None])
            cy = np.polyfit(idx, sy, m); fy = np.polyval(cy, idx[:, None])
            f2[rev * ns:(rev + 1) * ns] = np.mean((sx - fx) * (sy - fy), axis=0)
        eps = np.finfo(float).tiny
        a = np.sign(f2) * np.maximum(np.abs(f2), eps)
        for j, q in enumerate(qs):
            if abs(q) < 1e-12:
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(np.abs(f2) + eps)))
            else:
                Fq[j, i] = np.sign(np.mean(a ** (q / 2.0))) * \
                           np.abs(np.mean(a ** (q / 2.0))) ** (1.0 / q)

    lx = np.log(species.astype(float))
    hxy = np.empty(qs.size); r2 = np.empty(qs.size)
    for j in range(qs.size):
        ly = np.log(np.abs(Fq[j]) + np.finfo(float).tiny)
        A = np.vstack([lx, np.ones_like(lx)]).T
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((ly - pred) ** 2)); ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2[j] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        hxy[j] = coef[0]
    tau = qs * hxy - 1.0
    alpha, f, h0, dh, dh_half, asym, closed = spectrum_from_tau(qs, tau)
    return dict(qs=qs, scales=species, Fq=Fq, hxy=hxy, r2=r2, tau=tau, alpha=alpha, f=f,
                h0=h0, dh=dh, dh_half=dh_half, asym=asym, closed=closed,
                dh_q=float(hxy[0] - hxy[-1]), n=int(n), m=int(m))


def dcca_coefficient(x, y, s: int, m: int = 1):
    """Podobnik DCCA cross-correlation coefficient rho_DCCA at scale s."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(x.size, y.size)
    X = np.cumsum(x[:n] - x[:n].mean()); Y = np.cumsum(y[:n] - y[:n].mean())
    ns = n // s
    idx = np.arange(s, dtype=float)
    f2xy = []; f2x = []; f2y = []
    for rev in (0, 1):
        if rev == 0:
            sx = X[: ns * s].reshape(ns, s).T; sy = Y[: ns * s].reshape(ns, s).T
        else:
            sx = X[n - ns * s:][::-1].reshape(ns, s).T; sy = Y[n - ns * s:][::-1].reshape(ns, s).T
        cx = np.polyfit(idx, sx, m); fx = np.polyval(cx, idx[:, None])
        cy = np.polyfit(idx, sy, m); fy = np.polyval(cy, idx[:, None])
        rx, ry = sx - fx, sy - fy
        f2xy.append(np.mean(rx * ry, axis=0)); f2x.append(np.mean(rx * rx, axis=0))
        f2y.append(np.mean(ry * ry, axis=0))
    f2xy = np.concatenate(f2xy); f2x = np.concatenate(f2x); f2y = np.concatenate(f2y)
    den = np.sqrt(f2x.mean() * f2y.mean())
    return float(f2xy.mean() / den) if den > 0 else np.nan


# --------------------------------------------------------------------------------------
# 4. surrogates
# --------------------------------------------------------------------------------------
def shuffle_surrogate(x, rng: np.random.Generator):
    """Random permutation: destroys temporal correlations, keeps the value distribution."""
    return rng.permutation(np.asarray(x, float))


def iaaft_surrogate(x, rng: np.random.Generator, n_iter: int = 30):
    """
    Iterative Amplitude Adjusted Fourier Transform surrogate (Schreiber & Schmitz 1996):
    preserves both the amplitude distribution and the power spectrum (=> tests whether
    multifractality comes from the *distribution* or from the *correlations*).
    """
    x = np.asarray(x, float); n = x.size
    amp = np.abs(np.fft.rfft(x))
    xs = np.sort(x)
    y = rng.permutation(x).astype(float)
    for _ in range(n_iter):
        phase = np.angle(np.fft.rfft(y))
        Y = amp * np.exp(1j * phase)
        y = np.fft.irfft(Y, n=n)
        order = np.argsort(y)
        y[order] = xs
    return y


def fit_heavy_tail(x):
    """
    Crude heavy-tail indicator: Hill index of the |x| upper tail and excess kurtosis.
    Used to interpret whether the multifractal width is distribution-driven.
    """
    x = np.asarray(x, float)
    a = np.abs(x - np.median(x))
    a = a[a > 0]
    n = a.size
    if n < 100:
        return np.nan, np.nan
    k = max(10, int(0.05 * n))
    top = np.sort(a)[-k:]
    hill = 1.0 / np.mean(np.log(top / top.min()))
    m = x.mean(); s = x.std()
    kurt = float(np.mean(((x - m) / s) ** 4) - 3.0) if s > 0 else np.nan
    return float(hill), kurt
