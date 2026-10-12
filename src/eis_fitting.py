"""
eis.py -- parsing and analysis of the electrochemical impedance spectra (EIS) of the
IEEE PHM 2014 dataset, and extraction of descriptors to be correlated with the
multifractal metrics.

Two published data layouts are handled:
  * FC1 (steady state): xlsx with columns [f/Hz, Re(Z)/Ohm, -Im?] -> actually i/Ohm (signed)
  * FC2 (with ripple): csv, semicolon separated, same three columns.

Descriptors extracted per spectrum:
  R_hf   : high-frequency resistance (ohmic), Re(Z) at the highest frequencies
  R_lf   : Re(Z) at the lowest frequency
  R_p    : polarisation resistance, R_lf - R_hf
  f_apex : frequency of the maximum of -Im(Z)
  tau    : 1/(2*pi*f_apex)
  single-arc CPE fit:  Z = R_hf + R_p / (1 + (j w tau)^alpha)   -> R_p, tau, alpha, chi2
  two-arc CPE fit   :  Z = R_hf + Rp1/(1+(j w tau1)^a1) + Rp2/(1+(j w tau2)^a2)
The CPE exponent alpha is the electrochemically meaningful "fractal" descriptor: it
measures the dispersion of the impedance arc and is classically linked to the fractal
geometry / inhomogeneity of the porous electrode.
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np
import openpyxl
from scipy.optimize import least_squares
PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def read_spectrum(path: str):
    """Return (freq_Hz, Re_Ohm, Im_Ohm) sorted by descending frequency."""
    if path.lower().endswith(".xlsx"):
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = []
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r is None or r[0] is None:
                continue
            try:
                rows.append([float(str(v).replace(",", ".")) for v in r[:3]])
            except (TypeError, ValueError):
                continue
        wb.close()
        a = np.asarray(rows, float)
    else:
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            rd = csv.reader(f, delimiter=";")
            for i, row in enumerate(rd):
                if i == 0 or not row or row[0].strip() == "":
                    continue
                try:
                    rows.append([float(str(v).replace(",", ".")) for v in row[:3]])
                except (TypeError, ValueError):
                    continue
        a = np.asarray(rows, float)
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"no numeric EIS rows parsed from {path}")
    a = a[np.argsort(-a[:, 0])]
    return a[:, 0], a[:, 1], a[:, 2]


def model_arc(w, rp, tau, alpha):
    """CPE (Cole-Cole) arc: rp / (1 + (j w tau)^alpha)."""
    return rp / (1.0 + (1j * w * tau) ** alpha)


# The published files store (f, Re Z, -Im Z) -- the usual electrochemistry convention in
# which a capacitive arc has a POSITIVE third column and cable/stack inductance makes it
# negative at high frequency.  Internally we always use Z = Re + j Im, i.e.
#     Z_meas = re - 1j * col3
# and the model includes a series inductance:
#     Z(w) = R_ohm + j w L + sum_k R_k / (1 + (j w tau_k)^{alpha_k})
def model_single(w, p):
    r, l, rp, tau, a = p
    return r + 1j * w * l + model_arc(w, rp, tau, a)


def model_double(w, p):
    r, l, rp1, tau1, a1, rp2, tau2, a2 = p
    return (r + 1j * w * l + model_arc(w, rp1, tau1, a1) + model_arc(w, rp2, tau2, a2))


def _prep(f, re, im_col3):
    w = 2 * np.pi * f
    z = re - 1j * im_col3          # Z = Re + j Im
    return w, z


def _fit(w, z, model, p0, lo, hi):
    def resid(p):
        r = model(w, p) - z
        s = np.abs(z) + 1e-12      # proportional (modulus) weighting
        return np.concatenate([r.real / s, r.imag / s])
    res = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13,
                        x_scale="jac", max_nfev=40000)
    chi2 = float(np.sum(res.fun ** 2) / res.fun.size)
    return res.x, chi2


def fit_single_arc(f, re, im_col3):
    w, z = _prep(f, re, im_col3)
    r_hf0 = float(re.min())
    r_p0 = float(max(re.max() - r_hf0, 1e-5))
    # induction estimate from the highest-frequency point
    l0 = float(max(im_col3[0], 1e-12)) / (2 * np.pi * f[0])
    fpeak = float(f[int(np.argmax(im_col3))])
    p0 = [r_hf0, max(l0, 1e-10), r_p0, 1.0 / (2 * np.pi * max(fpeak, 1e-3)), 0.85]
    lo = [0.0, 0.0, 0.0, 1e-8, 0.30]
    hi = [np.inf, 1e-4, np.inf, 1e8, 1.0]
    try:
        p, chi2 = _fit(w, z, model_single, p0, lo, hi)
        return dict(r_ohm_fit=float(p[0]), l_fit=float(p[1]), r_p_fit=float(p[2]),
                    tau_fit=float(p[3]), alpha_fit=float(p[4]), chi2=chi2, ok=True)
    except Exception as e:  # pragma: no cover
        return dict(ok=False, err=str(e)[:120])


def fit_two_arc(f, re, im_col3):
    w, z = _prep(f, re, im_col3)
    r_hf0 = float(re.min())
    fpeak = float(f[int(np.argmax(im_col3))])
    r_p0 = float(max(re.max() - r_hf0, 1e-5))
    l0 = float(max(im_col3[0], 1e-12)) / (2 * np.pi * f[0])
    p0 = [r_hf0, max(l0, 1e-10), 0.5 * r_p0, 1.0 / (2 * np.pi * fpeak * 30), 0.85,
          0.5 * r_p0, 1.0 / (2 * np.pi * fpeak / 3), 0.85]
    lo = [0.0, 0.0, 0.0, 1e-8, 0.30, 0.0, 1e-8, 0.30]
    hi = [np.inf, 1e-4, np.inf, 1e8, 1.0, np.inf, 1e8, 1.0]
    try:
        p, chi2 = _fit(w, z, model_double, p0, lo, hi)
        return dict(r_ohm_2=float(p[0]), l_2=float(p[1]), rp1=float(p[2]), tau1=float(p[3]),
                    a1=float(p[4]), rp2=float(p[5]), tau2=float(p[6]), a2=float(p[7]),
                    rp_tot=float(p[2] + p[5]), chi2_2=chi2, ok2=True)
    except Exception as e:  # pragma: no cover
        return dict(ok2=False, err2=str(e)[:120])


def descriptors(path: str):
    """Full descriptor set for one EIS file."""
    f, re, im3 = read_spectrum(path)
    # Z = Re + j Im  with  Im = -col3
    im = -im3
    out = dict(file=os.path.basename(path), n_points=int(f.size),
               f_min=float(f.min()), f_max=float(f.max()))
    # inductive crossover: f is sorted descending, so the FIRST sign change of Im(Z) is the
    # highest-frequency crossing between the inductive (cable) tail and the capacitive arc
    cross = np.where(np.diff(np.sign(im)) != 0)[0]
    out["f_cross"] = float(f[cross[0]]) if cross.size else np.nan
    out["r_ohm_cross"] = float(re[cross[0]]) if cross.size else float(re.min())
    out["r_min"] = float(re.min())
    out["r_lf"] = float(re[-1])
    out["r_p_naive"] = float(re[-1] - (out["r_ohm_cross"] if cross.size else re.min()))
    out["cap_max"] = float(im3.max())          # max of -Im Z = height of the capacitive arc
    out["f_apex"] = float(f[int(np.argmax(im3))])
    out["tau_apex"] = 1.0 / (2 * np.pi * out["f_apex"]) if out["f_apex"] > 0 else np.nan
    out.update(fit_single_arc(f, re, im3))
    out.update(fit_two_arc(f, re, im3))
    return out


def parse_tag(name: str):
    """Extract (stack, current_A, stage_h) from an EIS file name."""
    m = re.search(r"(FC\d)_EIS(\d+)A_(\w+)_T(\d+)", os.path.basename(name))
    if not m:
        return None
    return dict(stack=m.group(1), current_A=float(m.group(2)), kind=m.group(3),
                t_h=float(m.group(4)))


def collect(data_dir: str, pattern: str = "data__FC1_FC2_Excel__*EIS*"):
    rows = []
    for p in sorted(glob.glob(os.path.join(data_dir, pattern))):
        tag = parse_tag(p)
        if tag is None:
            continue
        try:
            d = descriptors(p)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {type(e).__name__}: {str(e)[:90]}")
            continue
        if d.get("n_points", 0) < 8:
            print(f"  [skip] {os.path.basename(p)}: only {d.get('n_points')} points")
            continue
        d.update(tag)
        rows.append(d)
    return rows


if __name__ == "__main__":
    import json
    d = os.path.join(PROJECT_ROOT, "data", "phm2014")
    rows = collect(d)
    print("spectra:", len(rows))
    for r in rows[:4]:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(PROJECT_ROOT, "results", "eis_descriptors.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=float)
