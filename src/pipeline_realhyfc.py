"""
realhyfc.py -- RealHyFC / ZSW heavy-duty PEMFC stack durability dataset
(Zenodo 20715542, CC BY 4.0; Fensterle et al., Int. J. Hydrogen Energy 2026).

Loadcycle_data.zip contains 17 semicolon-separated CSV files, 1 Hz, one 10-cell 280 cm2
stack, 66 channels, operation hours 0 .. ~2678 h.  The archive is 5.8 GB uncompressed, so
the file is streamed twice and never fully materialised:

  pass 1 : read only the `OpHrs [h]` column -> exact time coverage and stage locations
  pass 2 : keep only the rows belonging to the windows of interest (3 stages x 40 h)

For each stage a 2^17-sample (36.4 h) window ending at the stage time is analysed with
MFDFA and WLMA, for the stack voltage and for each of the ten individual cell voltages,
in two variants: raw and 1 h high-pass (which removes the slow load-cycle envelope and
leaves the water-management / two-phase-flow intermittency).
"""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multifractal_methods as mf  # noqa: E402

PROJECT_ROOT = os.environ.get("PEMFC_PROJECT_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DATA = os.path.join(PROJECT_ROOT, "data")
RESULTS = os.path.join(PROJECT_ROOT, "results")
ZIP = os.path.join(DATA, "realhyfc_Loadcycle_data.zip")

QS = np.linspace(-6.0, 6.0, 25)   # uniform grid: required by the discrete Legendre transform
WIN = 2 ** 17
STAGES = [(110.0, "early"), (1150.0, "mid"), (2300.0, "late")]
UCOLS = [f"U{i} [V]" for i in range(1, 12)]
COLS = ["OpHrs [h]", "I.S.act [A]"] + UCOLS


def num(s) -> float:
    """European formatting: '2,678.46' or '1,5'; accepts str or bytes."""
    if isinstance(s, bytes):
        s = s.strip()
        if not s:
            return np.nan
        if b"," in s and b"." in s:
            s = s.replace(b",", b"")
        elif b"," in s:
            s = s.replace(b",", b".")
        try:
            return float(s)
        except ValueError:
            return np.nan
    s = s.strip()
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


def members(zf):
    return sorted([n for n in zf.namelist() if n.endswith(".csv")],
                  key=lambda s: int(re.search(r"Loadcycle_(\d+)", s).group(1)))


def pass1_hours(zf, ms, chunks=300000):
    """'OpHrs [h]' column only, in blocks.

    The column is located BY NAME: the column order is not stable across members
    (Loadcycle_51-70.csv and Loadcycle_134-150.csv store a 'CalHrs [h]' column first).
    Every physical line is counted (NaN for unreadable ones) so that the returned
    index space matches pass2_selected's global line numbering.
    """
    out = []
    for m in ms:
        with zf.open(m) as fh:
            header = fh.readline().split(b";")
            ci = header.index(b"OpHrs [h]")
            buf = []
            for line in fh:
                p = line.split(b";", ci + 2)
                if len(p) <= ci:
                    buf.append(np.nan)
                else:
                    buf.append(num(p[ci]))
                if len(buf) >= chunks:
                    out.append(np.asarray(buf, float)); buf = []
            if buf:
                out.append(np.asarray(buf, float))
    return np.concatenate(out)


def pass2_selected(zf, ms, wanted):
    """
    wanted: list of (global_lo, global_hi, tag).  Returns dict tag -> array (rows, ncols)
    for the requested columns.
    """
    res = {t: [] for _, _, t in wanted}
    g0 = 0
    for m in ms:
        with zf.open(m) as fh:
            header = fh.readline().decode("utf-8-sig", errors="replace").split(";")
            idx = [header.index(c) for c in COLS]
            kmax = max(idx)
            gi = g0
            for line in fh:
                p = line.rstrip(b"\r\n").split(b";")
                if len(p) <= kmax:
                    gi += 1
                    continue
                for lo, hi, tag in wanted:
                    if lo <= gi < hi:
                        res[tag].append([num(p[k]) for k in idx])
                        break
                gi += 1
        g0 = gi
    return {t: (np.asarray(v, float) if v else None) for t, v in res.items()}


def highpass(x, w=3600):
    x = np.asarray(x, float)
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    half = w // 2
    i = np.arange(n)
    lo = np.clip(i - half, 0, n); hi = np.clip(i + half, 0, n)
    return x - (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def metrics(x, tag):
    r = mf.mfdfa(x, qs=QS, smin=40, smax=4096, nscale=30, m=1)
    w = mf.wlma(x, integrate=True, wavelet="db3", j1=3, j2=12, qs=QS)
    return dict(tag=tag, n=int(x.size), mean_V=float(np.mean(x)), std_V=float(np.std(x)),
                mf_h0=r["h0"], mf_dh=r["dh"], mf_dh_half=r["dh_half"], mf_asym=r["asym"],
                mf_h2=r["h_plus"], mf_hm2=r["h_minus"], mf_dh_q=r["dh_q"],
                mf_dh5=r["dh_5"], mf_dh5_half=r["dh_half_5"], mf_h05=r["h0_5"],
                wl_c1=w["c1"], wl_c2=w["c2"], wl_c3=w["c3"],
                wl_dh=w["dh_cum"], wl_dh_half=w["dh_half_cum"], wl_dh_leg=w["dh"],
                wl_dh5=w["dh_cum_5"])


def main():
    zf = zipfile.ZipFile(ZIP)
    ms = members(zf)
    print("members:", len(ms), flush=True)
    t = pass1_hours(zf, ms)
    print("rows=%d  OpHrs %.1f .. %.1f h" % (t.size, np.nanmin(t), np.nanmax(t)), flush=True)
    # a few rows carry implausible OpHrs values (logger artefacts): the documented test
    # duration is ~2678 h, so values above 5000 h are excluded from the stage search
    ok = np.isfinite(t) & (t >= 0.0) & (t <= 5000.0)
    print("  plausible OpHrs rows: %d (%.4f %% excluded)"
          % (int(ok.sum()), 100.0 * (1 - ok.sum() / t.size)), flush=True)
    # stage index: first row whose OpHrs exceeds the stage target
    wanted = []
    chosen = []
    for target, name in STAGES:
        j = int(np.searchsorted(t[ok], target))
        j = min(j, int(ok.sum()) - 1)
        if j < WIN:
            j = WIN
        lo, hi = j - WIN, j
        wanted.append((lo, hi, name))
        chosen.append(dict(stage_target_h=target, name=name,
                           t_start_h=float(t[ok][lo]), t_end_h=float(t[ok][hi - 1]),
                           n=hi - lo))
        print("  stage %-5s target %.0f h -> window %.1f .. %.1f h (%d rows)"
              % (name, target, t[ok][lo], t[ok][hi - 1], hi - lo), flush=True)
    del t
    data = pass2_selected(zf, ms, wanted)
    # the stage-search indices above are computed on the plausibility-filtered OpHrs
    # array; the actual extracted rows may sit elsewhere (gaps / short lines), so the
    # authoritative window times are those of the extracted rows themselves
    for c in chosen:
        A = data.get(c["name"])
        if A is not None and A.shape[0]:
            c["t_start_h"] = float(A[0, 0])
            c["t_end_h"] = float(A[-1, 0])
    rows = []
    for name, _ in [(w[2], None) for w in wanted]:
        A = data[name]
        print(f"  extracted {name}: {None if A is None else A.shape}", flush=True)
        if A is None or A.shape[0] < WIN // 2:
            continue
        n = A.shape[0]
        uidx = [COLS.index(f"U{k} [V]") for k in range(1, 12)]
        Ud = A[:, uidx]
        Umean = np.nanmean(Ud, axis=1)                      # common-mode (load-cycle) part
        for var, fn in (("hp1h", lambda v: highpass(v, 3600)),
                        ("dU", lambda v: np.asarray(v, float) - Umean)):
            for k in range(1, 12):
                ci = COLS.index(f"U{k} [V]")
                x = fn(A[:, ci])
                x = x[np.isfinite(x)]
                if x.size < WIN // 2:
                    continue
                d = metrics(x, f"{var}|U{k}")
                d.update(variant=var, channel=f"U{k}", stage=name,
                         t_start_h=float(A[0, 0]), t_end_h=float(A[-1, 0]))
                rows.append(d)
            xs = fn(np.nansum(Ud, axis=1))
            d = metrics(xs, f"{var}|Ustack")
            d.update(variant=var, channel="Ustack", stage=name,
                     t_start_h=float(A[0, 0]), t_end_h=float(A[-1, 0]))
            rows.append(d)
        print(f"    metrics done for {name}", flush=True)

    W = pd.DataFrame(rows)
    W.to_csv(os.path.join(RESULTS, "realhyfc_windows.csv"), index=False, sep=";",
             decimal=",")

    # null reference at the same N and the same signal type (cell deviation from cell mean)
    from validate_methods import fgn
    A = data["late"] if data.get("late") is not None else data["mid"]
    uidx = [COLS.index(f"U{k} [V]") for k in range(1, 12)]
    Ud = A[:, uidx]
    xi = Ud[:, 0] - np.nanmean(Ud, axis=1)
    H0 = float(mf.mfdfa(xi, qs=np.array([2.0]), smin=40, smax=4096, nscale=30)["h"][0])
    nulls = [metrics(fgn(WIN, H0, np.random.default_rng(s)), "null") for s in (1, 2, 3)]
    meta = dict(stages=chosen, null_H=H0,
                null_mf_dh=float(np.mean([q["mf_dh"] for q in nulls])),
                null_mf_dh5=float(np.mean([q["mf_dh5"] for q in nulls])),
                null_wl_dh=float(np.mean([q["wl_dh"] for q in nulls])),
                null_wl_dh5=float(np.mean([q["wl_dh5"] for q in nulls])),
                windows=W.to_dict(orient="records"))
    with open(os.path.join(RESULTS, "realhyfc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=float)
    print("null H=%.3f MFDFA dh=%.3f (|q|<=5: %.3f) WLMA dh=%.3f (|q|<=5: %.3f)"
          % (H0, meta["null_mf_dh"], meta["null_mf_dh5"], meta["null_wl_dh"],
             meta["null_wl_dh5"]))
    print("saved results/realhyfc_windows.csv, realhyfc_meta.json")


if __name__ == "__main__":
    sys.exit(main())
