#!/usr/bin/env python3
"""Hunt a mixed-portal nested benchmark with f_comp → 1, then plot its GW.

Needs ΔT_n ≲ few GeV (χ_nest ≲ 0.3).  Restores μ_mix via φ_off targets.

Example
-------
    PYTHONPATH=. python -u analysis/find_fcomp_benchmark.py \\
        --out data/portal/fcomp_benchmark --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.portalBounce import (  # noqa: E402
    SET8_BOSON_PHI,
    SET8_FERMION_PHI,
    SET8_V,
    SET8_V0,
)
from analysis.portalEpsScan import (  # noqa: E402
    DEFAULT_T_REH,
    _Tee,
    _qs_ref_from_row,
    _total_spectrum,
    log,
    run_one,
    s0_from_mS,
)
from analysis.run_mixed_portal_scan import mu2_for_phi_off  # noqa: E402
from analysis.twoStepGW import (  # noqa: E402
    chi_nest,
    f_comp_from_chi,
    spectrum,
)


def _parse(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def annotate_fcomp(row: dict, vw1=0.6, vw2=1.0) -> dict:
    dTn = float(row.get("delta_T_n_S", float("nan")))
    Tn1 = float(row.get("T_n_0_to_S", row.get("T_n_step1", float("nan"))))
    b1 = float(row.get("betaH_step1", float("nan")))
    if not (np.isfinite(dTn) and dTn >= 0 and np.isfinite(Tn1) and Tn1 > 0
            and np.isfinite(b1) and b1 > 0):
        # Regime II immediate / missing ΔT → treat as fully composite
        if "nested" in str(row.get("regime", "")).lower() or "Regime II" in str(
                row.get("sequence", "")):
            row["chi_nest"] = 0.0
            row["f_comp"] = 1.0
        else:
            row["chi_nest"] = float("nan")
            row["f_comp"] = 0.0
        return row
    chi = chi_nest(b1, dTn, Tn1, vw1=vw1, vw2=vw2)
    row["chi_nest"] = chi
    row["f_comp"] = f_comp_from_chi(chi)
    row["vw_outer"] = vw1
    row["vw_inner"] = vw2
    return row


def plot_benchmark_gw(row: dict, out_dir: str, V0: float = SET8_V0,
                      T_reh: float = DEFAULT_T_REH) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.portalEpsScan import recompute_gw_secIV

    # rebuild Sec.III.C peaks into the row
    rows = recompute_gw_secIV([dict(row)], V0=V0, T_reh=T_reh)
    r = rows[0]
    qs, ref = _qs_ref_from_row(r)

    f = np.geomspace(1e-4, 1e4, 3000)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    try:
        from analysis.published_plisc import add_detector_curves
        add_detector_curves(ax)
    except Exception:
        pass

    styles = {
        "step1": ("#7570b3", "--", r"unmerged $0\to S$"),
        "step2": ("#d95f02", "-.", r"unmerged $S\to v$"),
        "composite": ("#1b9e77", "-", r"composite $O\to B$"),
        "cross": ("#e7298a", ":", r"catch-up cross"),
    }
    tot = np.zeros_like(f)
    for q in qs:
        if not (q.get("h2Omega_peak", 0) > 0 and q.get("f_peak [Hz]", 0) > 0):
            continue
        lab = q["label"]
        col, ls, nice = styles.get(lab, ("0.4", "-", lab))
        s = spectrum(f, q)
        tot = tot + s
        ax.loglog(f, s, ls=ls, color=col, lw=1.6,
                  label=rf"{nice}: $f_p$={q['f_peak [Hz]']:.3g} Hz, "
                        rf"$h^2\Omega_p$={q['h2Omega_peak']:.3g}")
    if tot.max() > 0:
        ax.loglog(f, tot, color="0.1", lw=2.2, label="nested total")
    if ref and ref.get("h2Omega_peak", 0) > 0:
        ax.loglog(f, spectrum(f, ref), ":", color="0.35", lw=1.5,
                  label=rf"$0\to v$ ref: $h^2\Omega_p$={ref['h2Omega_peak']:.3g}")

    ax.set_ylim(1e-25, 1e-5)
    ax.set_xlim(1e-4, 1e4)
    ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{\rm GW}$")
    eps = float(r.get("eps", float("nan")))
    kap = float(r.get("kappa", float("nan")))
    mS = float(r.get("m_S", float("nan")))
    po = float(r.get("phi_off_target", r.get("phi_off_tgt", 0.0)) or 0.0)
    fc = float(r.get("f_comp", float("nan")))
    chi = float(r.get("chi_nest", float("nan")))
    dTn = float(r.get("delta_T_n_S", float("nan")))
    amp = float(r.get("nested_amp_factor", float("nan")))
    ax.set_title(
        rf"Sec.~III.C nested benchmark  $f_{{\rm comp}}={fc:.3g}$"
        rf"  ($\chi_{{\rm nest}}={chi:.3g}$, $\Delta T_n={dTn:.3g}$ GeV)",
        fontsize=11)
    ax.text(
        0.98, 0.02,
        rf"$\varepsilon={eps:.2g}$, $\kappa={kap:g}$, $m_S={mS:g}$ GeV, "
        rf"$\phi_{{\rm off}}^{{\rm tgt}}={po:g}$ GeV"
        + (rf", amp$\times={amp:.2g}$" if np.isfinite(amp) else ""),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.75", alpha=0.9))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, "gw_fcomp_benchmark.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    log(f"wrote {path}")

    with open(os.path.join(out_dir, "benchmark.json"), "w") as fh:
        json.dump({"row": r, "qs": qs, "ref": ref}, fh, indent=2, default=float)
    return path


def main():
    from analysis import portalEpsScan as _pes

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/portal/fcomp_benchmark")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--nT", type=int, default=22)
    p.add_argument("--a2v_n_fine", type=int, default=28)
    p.add_argument("--eps", type=str, default="1e-20")
    p.add_argument("--kappa", type=str, default="0.01,0.02,0.05,0.1")
    p.add_argument("--m_S", type=str, default="470,475,478,480,485,490")
    p.add_argument("--phi_off", type=str, default="0,100,200,400,600")
    p.add_argument("--vw_outer", type=float, default=0.6)
    p.add_argument("--vw_inner", type=float, default=1.0)
    p.add_argument("--fcomp_min", type=float, default=0.9,
                   help="stop early once a point with f_comp >= this is found")
    p.add_argument("--from_csv", default=None,
                   help="skip scan; pick best f_comp from existing CSV and plot")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.from_csv:
        import pandas as pd
        df = pd.read_csv(a.from_csv)
        rows = [annotate_fcomp(r, a.vw_outer, a.vw_inner)
                for r in df.to_dict("records")]
        rows = sorted(rows, key=lambda r: -float(r.get("f_comp", 0) or 0))
        best = rows[0]
        log(f"best from CSV: f_comp={best.get('f_comp')}  ΔT_n={best.get('delta_T_n_S')}  "
            f"m_S={best.get('m_S')} κ={best.get('kappa')} φ_off={best.get('phi_off_target')}")
        plot_benchmark_gw(best, a.out)
        return

    log_path = os.path.join(a.out, "progress.log")
    _pes._LOG = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _pes._LOG)
    sys.stderr = _Tee(sys.__stderr__, _pes._LOG)

    epss = _parse(a.eps)
    kappas = _parse(a.kappa)
    mSs = _parse(a.m_S)
    phis = _parse(a.phi_off)
    # order: small m_S, small κ, then raise φ_off (mixing shrinks ΔT_n)
    jobs = [(e, k, m, p)
            for m in mSs for k in kappas for p in phis for e in epss]
    log(f"f_comp benchmark hunt → {a.out}")
    log(f"jobs={len(jobs)}  target f_comp≥{a.fcomp_min:g}")
    log(f"vw=({a.vw_outer:g},{a.vw_inner:g})")

    rows = []
    best = None
    t0 = time.time()
    for i, (eps, kappa, m_S, phi_off) in enumerate(jobs, 1):
        s0 = s0_from_mS(eps, SET8_V0, m_S)
        mu2 = mu2_for_phi_off(
            SET8_V0, SET8_V, s0, eps, kappa,
            SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)],
            100.0, "set8", float(phi_off))
        args = SimpleNamespace(
            V0=SET8_V0, v=SET8_V, m_S=m_S, s0=s0,
            boson_phi=SET8_BOSON_PHI, fermion_phi=SET8_FERMION_PHI,
            boson_S=[(6.0, 1.0)], g_star=100.0, phi_thermal="set8",
            Tmin=None, Tmax=None, nT=a.nT, refine=8, workers=a.workers,
            a2v_n_fine=a.a2v_n_fine, a2v_T_hi_fac=1.25, a2v_T_lo_fac=0.85,
            a2v_S_trigger=2000.0, T_reh=DEFAULT_T_REH,
            rmix=None, mu2_mix=mu2,
        )
        log(f"\n======== [{i}/{len(jobs)}] m_S={m_S:g} κ={kappa:g} "
            f"ε={eps:.3g} φ_off={phi_off:g} μ²={mu2:.4g} ========")
        try:
            row = run_one(eps, kappa, args, i, len(jobs))
        except Exception as exc:
            log(f"  ERROR: {exc}")
            row = {"eps": eps, "kappa": kappa, "m_S": m_S,
                   "phi_off_target": phi_off, "mu2_mix": mu2,
                   "regime": f"error: {exc}", "f_comp": 0.0}
        row["phi_off_target"] = phi_off
        row = annotate_fcomp(row, a.vw_outer, a.vw_inner)
        rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "hunt.csv"), index=False)
        log(f"  → regime={row.get('regime')}  ΔT_n={row.get('delta_T_n_S')}  "
            f"χ={row.get('chi_nest')}  f_comp={row.get('f_comp'):.4g}  "
            f"elapsed={time.time()-t0:.0f}s")
        if best is None or float(row.get("f_comp", 0) or 0) > float(
                best.get("f_comp", 0) or 0):
            best = dict(row)
        if float(row.get("f_comp", 0) or 0) >= a.fcomp_min:
            log(f"  HIT f_comp≥{a.fcomp_min:g} — stopping early")
            break

    with open(os.path.join(a.out, "hunt.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    if best is None:
        log("no usable points")
        return
    log(f"\nBEST: f_comp={best.get('f_comp'):.4g}  χ={best.get('chi_nest')}  "
        f"ΔT_n={best.get('delta_T_n_S')}  m_S={best.get('m_S')}  "
        f"κ={best.get('kappa')}  φ_off={best.get('phi_off_target')}")
    plot_benchmark_gw(best, a.out)
    # also dump top-5
    top = sorted(rows, key=lambda r: -float(r.get("f_comp", 0) or 0))[:5]
    log("top-5 f_comp:")
    for r in top:
        log(f"  f_comp={r.get('f_comp'):.4g}  ΔT_n={r.get('delta_T_n_S')}  "
            f"m_S={r.get('m_S')} κ={r.get('kappa')} φ_off={r.get('phi_off_target')}  "
            f"regime={r.get('regime')}")


if __name__ == "__main__":
    main()
