#!/usr/bin/env python3
"""Scan ε ∈ [0.01, 0.1] at m_S values that can still give two-step separated PTs.

Writes live progress to <out>/progress.log and CSV after each point.
Sec. IV GW: amp ∝ (ΔV/V0)² D_inf R_md — compare total vs 0→v reference.
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
    _LOG,
    log,
    plot_scan,
    run_one,
)


def main():
    global _LOG
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/portal/eps_scan_large_eps")
    p.add_argument("--V0", type=float, default=SET8_V0)
    p.add_argument("--v", type=float, default=SET8_V)
    p.add_argument("--T_reh", type=float, default=DEFAULT_T_REH)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nT", type=int, default=22)
    p.add_argument("--refine", type=int, default=8)
    p.add_argument("--a2v_n_fine", type=int, default=24)
    p.add_argument("--probe_only", action="store_true",
                   help="only run the coarse (m_S,κ,ε) probe grid")
    args_ns = p.parse_args()

    os.makedirs(args_ns.out, exist_ok=True)
    log_path = os.path.join(args_ns.out, "progress.log")
    _LOG = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)

    # Probe grid: find (m_S, κ) where two-step survives at large ε
    probe = [
        (1000.0, 0.05, 0.05), (1000.0, 0.1, 0.05), (1000.0, 0.3, 0.05),
        (1500.0, 0.1, 0.05), (1500.0, 0.2, 0.05), (1500.0, 0.1, 0.1),
        (2000.0, 0.1, 0.05), (2000.0, 0.2, 0.1), (2000.0, 0.3, 0.05),
        (1800.0, 0.15, 0.05), (1800.0, 0.15, 0.1), (1200.0, 0.1, 0.05),
        (2500.0, 0.2, 0.1), (800.0, 0.05, 0.05),
    ]

    log(f"large-ε scan → {args_ns.out}")
    log(f"progress log: {os.path.abspath(log_path)}")
    log(f"V0={args_ns.V0:.4e}  T_reh={args_ns.T_reh:g} GeV  workers={args_ns.workers}")
    log(f"PROBE: {len(probe)} points")

    base = SimpleNamespace(
        V0=args_ns.V0, v=args_ns.v, m_S=1000.0, s0=None,
        boson_phi=SET8_BOSON_PHI, fermion_phi=SET8_FERMION_PHI,
        boson_S=[(6.0, 1.0)], g_star=100.0, phi_thermal="set8",
        Tmin=None, Tmax=None, nT=args_ns.nT, refine=args_ns.refine,
        workers=args_ns.workers,
        a2v_n_fine=args_ns.a2v_n_fine, a2v_T_hi_fac=1.25, a2v_T_lo_fac=0.85,
        a2v_S_trigger=2000.0, T_reh=args_ns.T_reh,
    )

    import pandas as pd
    probe_rows = []
    t_all = time.time()
    for i, (m_S, kappa, eps) in enumerate(probe, 1):
        args = SimpleNamespace(**{**base.__dict__, "m_S": m_S})
        log(f"\n======== PROBE [{i}/{len(probe)}] m_S={m_S:g} κ={kappa:g} ε={eps:g} ========")
        try:
            row = run_one(eps, kappa, args, i, len(probe))
        except Exception as exc:
            log(f"FAILED: {exc}")
            row = {"eps": eps, "kappa": kappa, "m_S": m_S, "regime": f"error: {exc}",
                   "h2Omega_peak_tot": float("nan"), "h2Omega_peak_0v": float("nan")}
        a0 = row.get("h2Omega_peak_0v", float("nan"))
        at = row.get("h2Omega_peak_tot", float("nan"))
        a2 = row.get("h2Omega_peak_step2", float("nan"))
        ratio = (at / a0) if (isinstance(a0, (int, float)) and a0 > 0
                              and isinstance(at, (int, float)) and math.isfinite(at)) else float("nan")
        row["Omega_tot_over_0v"] = ratio
        probe_rows.append(row)
        pd.DataFrame(probe_rows).to_csv(
            os.path.join(args_ns.out, "probe.csv"), index=False)
        log(f"  → regime={row.get('regime')}  Ω_tot/Ω_0v={ratio:.4g}  "
            f"Ω2={a2:.4g}  Ω0v={a0:.4g}")

    hits = [r for r in probe_rows
            if "two-step" in str(r.get("regime", "")) or "nested" in str(r.get("regime", ""))]
    log(f"\nPROBE done in {time.time()-t_all:.0f}s: {len(hits)}/{len(probe)} two-step hits")
    with open(os.path.join(args_ns.out, "probe.json"), "w") as fh:
        json.dump(probe_rows, fh, indent=2, default=float)

    if args_ns.probe_only:
        log("probe_only → stop")
        _LOG.close()
        return

    if not hits:
        log("No two-step hit in probe — skipping dense ε scan. Check probe.csv.")
        _LOG.close()
        return

    # Pick best (m_S, κ) by preferring separated + largest Ω_tot/Ω_0v among hits
    def score(r):
        reg = str(r.get("regime", ""))
        sep = 2.0 if "separated" in reg else (1.0 if "nested" in reg else 0.0)
        ratio = r.get("Omega_tot_over_0v", 0.0)
        if not (isinstance(ratio, (int, float)) and math.isfinite(ratio)):
            ratio = 0.0
        return (sep, ratio)

    hits_sorted = sorted(hits, key=score, reverse=True)
    best = hits_sorted[0]
    m_S_best = float(best["m_S"])
    kappa_best = float(best["kappa"])
    log(f"\nBest probe hit: m_S={m_S_best:g} κ={kappa_best:g} "
        f"(regime={best.get('regime')}, Ω_tot/Ω_0v={best.get('Omega_tot_over_0v')})")

    # (m_S, κ) pairs that gave two-step in the probe (keep all distinct)
    pairs = []
    seen = set()
    for r in hits_sorted:
        key = (round(float(r["m_S"]), 3), round(float(r["kappa"]), 5))
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    # Dense ε ∈ [0.01, 0.1] for top probe (m_S,κ) hits
    epss = list(np.geomspace(0.01, 0.1, 6))
    pairs_eps = pairs[:5]
    # Dense m_S at large ε for κ from best hits (+ 0.1)
    m_S_grid = list(np.unique(np.round(np.linspace(800.0, 2500.0, 9), 1)))
    kappas_mS = sorted({float(k) for _, k in pairs[:4]} | {0.1})
    eps_mS = [0.05, 0.1]

    jobs = []
    # 1) ε scan at top probe-hit (m_S,κ)
    for m, k in pairs_eps:
        for e in epss:
            jobs.append((e, k, m, "eps"))
    # 2) m_S scan at large ε
    for e in eps_mS:
        for k in kappas_mS:
            for m in m_S_grid:
                jobs.append((e, k, m, "m_S"))
    # de-duplicate (eps, kappa, m_S)
    uniq, jobs_u = set(), []
    for e, k, m, tag in jobs:
        key = (round(float(e), 8), round(float(k), 5), round(float(m), 3))
        if key in uniq:
            continue
        uniq.add(key)
        jobs_u.append((e, k, m, tag))
    jobs = jobs_u

    log(f"(m_S,κ) from probe: {pairs}")
    log(f"ε grid: {epss}")
    log(f"m_S grid: {m_S_grid}")
    log(f"κ for m_S scan: {kappas_mS}  ε for m_S scan: {eps_mS}")
    log(f"DENSE jobs: {len(jobs)} (ε-series + m_S-series, deduped)")

    rows = []
    for i, (eps, kap, m, tag) in enumerate(jobs, 1):
        args = SimpleNamespace(**{**base.__dict__, "m_S": m})
        log(f"\n======== SCAN [{i}/{len(jobs)}] ({tag}) m_S={m:g} κ={kap:g} ε={eps:.4g} ========")
        try:
            row = run_one(eps, kap, args, i, len(jobs))
        except Exception as exc:
            log(f"FAILED: {exc}")
            row = {"eps": eps, "kappa": kap, "m_S": m, "regime": f"error: {exc}",
                   "h2Omega_peak_tot": float("nan"), "h2Omega_peak_0v": float("nan")}
        a0 = row.get("h2Omega_peak_0v", float("nan"))
        at = row.get("h2Omega_peak_tot", float("nan"))
        ratio = (at / a0) if (isinstance(a0, (int, float)) and a0 > 0
                              and isinstance(at, (int, float)) and math.isfinite(at)) else float("nan")
        row["Omega_tot_over_0v"] = ratio
        row["scan_tag"] = tag
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(args_ns.out, "eps_scan.csv"), index=False)
        log(f"  → regime={row.get('regime')}  Ω_tot/Ω_0v={ratio:.4g}")

    with open(os.path.join(args_ns.out, "eps_scan.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    try:
        plot_scan(rows, args_ns.out, args_ns.V0, m_S_best)
        log(f"figures → {args_ns.out}/gw_spectra_kappa_*.png")
    except Exception as exc:
        log(f"plot_scan failed: {exc}")

    # Amplification summary
    log("\n=== Amplification vs 0→v (black dashed) ===")
    for r in rows:
        log(f"  m_S={r.get('m_S')} κ={r.get('kappa')} ε={r.get('eps'):.4g}  "
            f"{r.get('regime')}  Ω_tot/Ω_0v={r.get('Omega_tot_over_0v')}")
    amp = [r for r in rows
           if isinstance(r.get("Omega_tot_over_0v"), (int, float))
           and r["Omega_tot_over_0v"] > 1.0]
    log(f"Points with Ω_tot > Ω_0v: {len(amp)} / {len(rows)}")
    log(f"Done in {time.time()-t_all:.0f}s → {args_ns.out}")
    _LOG.close()


if __name__ == "__main__":
    main()
