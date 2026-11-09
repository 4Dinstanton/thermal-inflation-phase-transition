#!/usr/bin/env python3
"""Portal GW scan with λ_φS |φ|²|S|² only — no μ_mix / φ*S bilinear.

Parameters: (ε, κ, m_S).  Uses PortalPotential (unmixed).

Example
-------
    PYTHONPATH=. python -u analysis/run_portal_scan.py \\
        --out data/portal/portal_scan --workers 6
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
    log,
    plot_scan,
    recompute_gw_secIV,
    run_one,
    s0_from_mS,
)


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/portal/portal_scan")
    p.add_argument("--V0", type=float, default=SET8_V0)
    p.add_argument("--boson_S", type=str, default="6,1", help="S-bath bosons as n,g (default 6,1)")
    p.add_argument("--S_thermal", choices=("plain", "set8"), default="plain", help="set8 = S bath identical to the phi bath")
    p.add_argument("--gS_bath", type=float, default=0.0, help="shared-bath coupling gS (phi-bath masses ⊃ 1/2 gS^2 S^2)")
    p.add_argument("--v", type=float, default=SET8_V)
    p.add_argument("--T_reh", type=float, default=DEFAULT_T_REH)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--nT", type=int, default=20)
    p.add_argument("--refine", type=int, default=8)
    p.add_argument("--a2v_n_fine", type=int, default=20)
    p.add_argument("--eps", type=str,
                   default="1e-20,1e-12,1e-6,1e-3,0.01,0.05,0.1")
    p.add_argument("--kappa", type=str, default="0.05,0.1,0.2,0.3")
    p.add_argument("--m_S", type=str, default="500,800,1000,1200,1600,2000")
    p.add_argument("--phi_thermal", choices=("paper", "set8"), default="set8")
    p.add_argument("--quick", action="store_true",
                   help="tiny grid for smoke test")
    p.add_argument("--replot", action="store_true",
                   help="recompute GW + regime maps from existing CSV")
    args_ns = p.parse_args()

    os.makedirs(args_ns.out, exist_ok=True)
    log_path = os.path.join(args_ns.out, "progress.log")

    if args_ns.replot:
        import pandas as pd
        csv_path = os.path.join(args_ns.out, "portal_scan.csv")
        df = pd.read_csv(csv_path)
        rows = recompute_gw_secIV(df.to_dict("records"), V0=args_ns.V0,
                                  T_reh=args_ns.T_reh)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        with open(os.path.join(args_ns.out, "portal_scan.json"), "w") as fh:
            json.dump(rows, fh, indent=2, default=float)
        m_S = float(df["m_S"].dropna().iloc[0]) if "m_S" in df.columns else 1000.0
        plot_scan(rows, args_ns.out, args_ns.V0, m_S)
        # regime maps
        from analysis.plot_regime_maps import run_one_csv
        run_one_csv(csv_path, os.path.join(args_ns.out, "regime_maps"),
                    tag="portal_scan")
        print(f"replot done → {args_ns.out}")
        return

    _logf = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _logf)
    sys.stderr = _Tee(sys.__stderr__, _logf)

    epss = _parse_floats(args_ns.eps)
    kappas = _parse_floats(args_ns.kappa)
    mSs = _parse_floats(args_ns.m_S)
    if args_ns.quick:
        epss = epss[:3] or [1e-20]
        kappas = kappas[:2] or [0.1]
        mSs = mSs[:2] or [800.0]

    jobs = [(eps, kap, mS) for mS in mSs for kap in kappas for eps in epss]
    log(f"portal GW scan (no μ_mix) → {args_ns.out}")
    log(f"progress log: {os.path.abspath(log_path)}")
    log(f"V0={args_ns.V0:.4e}  T_reh={args_ns.T_reh:g}  workers={args_ns.workers}")
    log(f"κ={kappas}")
    log(f"ε={epss}")
    log(f"m_S={mSs}")
    log(f"jobs: {len(jobs)}")

    rows = []
    t_all = time.time()
    for i, (eps, kappa, m_S) in enumerate(jobs, 1):
        s0 = s0_from_mS(eps, args_ns.V0, m_S)
        args = SimpleNamespace(
            V0=args_ns.V0, v=args_ns.v, m_S=m_S, s0=s0,
            boson_phi=SET8_BOSON_PHI, fermion_phi=SET8_FERMION_PHI,
            boson_S=[tuple(float(x) for x in args_ns.boson_S.split(','))], g_star=100.0,
            phi_thermal=args_ns.phi_thermal,
            Tmin=None, Tmax=None, nT=args_ns.nT, refine=args_ns.refine,
            workers=args_ns.workers,
            a2v_n_fine=args_ns.a2v_n_fine, a2v_T_hi_fac=1.25, a2v_T_lo_fac=0.85,
            a2v_S_trigger=2000.0, T_reh=args_ns.T_reh,
            rmix=None, mu2_mix=None, gS_bath=args_ns.gS_bath, S_thermal=args_ns.S_thermal,
        )
        log(f"\n======== [{i}/{len(jobs)}] m_S={m_S:g} κ={kappa:g} ε={eps:.3g} "
            f"s0={s0:.4e} ========")
        try:
            row = run_one(eps, kappa, args, i, len(jobs))
        except Exception as exc:
            log(f"  ERROR: {exc}")
            row = {
                "eps": eps, "kappa": kappa, "m_S": m_S, "s0": s0,
                "regime": f"error: {exc}",
                "h2Omega_peak_tot": float("nan"),
                "h2Omega_peak_0v": float("nan"),
            }
        a0 = row.get("h2Omega_peak_0v", float("nan"))
        at = row.get("h2Omega_peak_tot", float("nan"))
        ratio = (at / a0) if (isinstance(a0, (int, float)) and isinstance(at, (int, float))
                              and math.isfinite(a0) and math.isfinite(at) and a0 > 0) else float("nan")
        row["Omega_tot_over_0v"] = ratio
        rows.append(row)

        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(args_ns.out, "portal_scan.csv"),
                                  index=False)
        log(f"  → regime={row.get('regime')}  Ω_tot/Ω_0v={ratio:.4g}  "
            f"ΔT_n={row.get('delta_T_n_S')}  "
            f"elapsed={time.time()-t_all:.0f}s")

    with open(os.path.join(args_ns.out, "portal_scan.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    hits = [r for r in rows
            if "two-step" in str(r.get("regime", ""))
            or "nested" in str(r.get("regime", ""))]
    stuck = [r for r in rows if "stuck" in str(r.get("regime", ""))]
    log(f"\nDone. two-step/nested={len(hits)}/{len(rows)}  stuck={len(stuck)}/{len(rows)}")
    for r in sorted(hits, key=lambda x: float(x.get("delta_T_n_S", 1e99)
                                              if np.isfinite(x.get("delta_T_n_S", float("nan")))
                                              else 1e99))[:12]:
        log(f"  {r.get('regime')}  ε={r['eps']:.3g}  m_S={r['m_S']:.4g}  "
            f"κ={r['kappa']:.3g}  ΔT_n={r.get('delta_T_n_S')}  "
            f"Ω_tot/Ω_0v={r.get('Omega_tot_over_0v')}")

    try:
        plot_scan(rows, args_ns.out, args_ns.V0, mSs[0])
    except Exception as exc:
        log(f"plot_scan skipped: {exc}")
    try:
        from analysis.plot_regime_maps import run_one_csv
        run_one_csv(os.path.join(args_ns.out, "portal_scan.csv"),
                    os.path.join(args_ns.out, "regime_maps"), tag="portal_scan")
    except Exception as exc:
        log(f"regime maps skipped: {exc}")
    log(f"total wall time {time.time()-t_all:.0f}s → {args_ns.out}")


if __name__ == "__main__":
    main()
