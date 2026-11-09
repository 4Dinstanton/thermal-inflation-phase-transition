#!/usr/bin/env python3
"""GW scan for the mixed portal potential V ⊃ −μ²_mix φ S (paper Oct 2026).

Scans ε (small→large), m_S (small→large), and μ via target φ_off (GeV).
μ²_mix is set so that the analytic offset

    φ_off ≃ μ²_mix s_T / M²_φ,A

matches the requested target at a reference T (avoids fixed rmix being either
useless at small ε or destroying A′ at large ε).

Uses PortalMixedPotential + the A′=(φ_off, s_T)-aware bounce in portalBounce.

Live progress → <out>/progress.log; CSV updated after each point.

Example
-------
    PYTHONPATH=. python analysis/run_mixed_portal_scan.py \\
        --out data/portal/mixed_scan --workers 8
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
    build_pot,
    log,
    plot_scan,
    run_one,
    s0_from_mS,
)


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def mu2_for_phi_off(V0, v, s0, eps, kappa, boson_phi, fermion_phi, boson_S,
                    g_star, phi_thermal, phi_off_target: float) -> float:
    """Choose μ²_mix so analytic φ_off ≲ phi_off_target over the A-stable window.

    Uses the *smallest* positive M²_φ,A in the unmixed A window so φ_off does
    not overshoot (and destroy A′) at lower T where the thermal mass is smaller.
    """
    if phi_off_target <= 0:
        return 0.0
    pot0 = build_pot(V0, v, s0, eps, kappa, boson_phi, fermion_phi, boson_S,
                     g_star, phi_thermal=phi_thermal)
    T_hi = max(1.3 * pot0.T0_S(), 100.0)
    T_lo = max(0.15 * min(pot0.T0_S(), pot0.T0_phi()), 20.0)
    Ts = np.geomspace(T_hi, T_lo, 24)
    samples = []  # (M2, mu2_for_exact_target)
    for T_try in Ts:
        vac = pot0.vacua(float(T_try))
        if vac.get("A") is None:
            continue
        s_T = float(vac["A"][1])
        M2 = float(pot0.mass2_phi(0.0, s_T, float(T_try)))
        if M2 > 0 and s_T > 0:
            samples.append((M2, phi_off_target * M2 / s_T))
    if samples:
        M2_max = max(m for m, _ in samples)
        # stay away from the spinodal (M2→0) where φ_off would blow up
        ok = [mu for m, mu in samples if m >= 0.2 * M2_max]
        if not ok:
            ok = [mu for _, mu in samples]
        # conservative: 25th percentile so max_T φ_off stays near target
        return float(np.percentile(ok, 25))
    # fallback: tree estimate at s0
    T_ref = float(max(0.5 * pot0.T0_S(), 100.0))
    M2 = float(pot0.mass2_phi(0.0, pot0.s0, T_ref))
    if not (M2 > 0):
        M2 = float(pot0.m_S2)
    return float(phi_off_target * M2 / max(pot0.s0, 1.0))


def main():
    from analysis import portalEpsScan as _pes

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/portal/mixed_scan")
    p.add_argument("--V0", type=float, default=SET8_V0)
    p.add_argument("--S_thermal", choices=("plain", "set8"), default="plain", help="set8 = S bath identical to the phi bath")
    p.add_argument("--gS_bath", type=float, default=0.0, help="shared-bath coupling gS (phi-bath masses ⊃ 1/2 gS^2 S^2)")
    p.add_argument("--v", type=float, default=SET8_V)
    p.add_argument("--T_reh", type=float, default=DEFAULT_T_REH)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nT", type=int, default=22)
    p.add_argument("--refine", type=int, default=8)
    p.add_argument("--a2v_n_fine", type=int, default=24)
    p.add_argument("--kappa", type=str, default="0.1,0.3",
                   help="comma-separated κ values")
    p.add_argument("--eps", type=str,
                   default="1e-20,1e-12,1e-6,1e-3,0.01,0.05,0.1",
                   help="comma-separated ε grid (small→large)")
    p.add_argument("--m_S", type=str, default="800,1200,1600,2000",
                   help="comma-separated m_S [GeV] (small→large)")
    p.add_argument("--phi_off", type=str, default="0,100,400,800",
                   help="comma-separated target φ_off [GeV] (μ knob; 0=unmixed)")
    p.add_argument("--rmix", type=str, default="",
                   help="optional absolute rmix=μ²/m_S² grid (overrides --phi_off)")
    p.add_argument("--mu2", type=str, default="",
                   help="optional absolute μ²_mix [GeV²] grid (overrides --phi_off)")
    p.add_argument("--probe_only", action="store_true",
                   help="short probe covering small/large ε, m_S, μ")
    args_ns = p.parse_args()

    os.makedirs(args_ns.out, exist_ok=True)
    log_path = os.path.join(args_ns.out, "progress.log")
    _pes._LOG = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _pes._LOG)
    sys.stderr = _Tee(sys.__stderr__, _pes._LOG)

    kappas = _parse_floats(args_ns.kappa)
    epss = _parse_floats(args_ns.eps)
    m_Ss = _parse_floats(args_ns.m_S)
    phi_offs = _parse_floats(args_ns.phi_off) if args_ns.phi_off else [0.0]
    rmixes = _parse_floats(args_ns.rmix) if args_ns.rmix else None
    mu2s = _parse_floats(args_ns.mu2) if args_ns.mu2 else None

    mode = "phi_off"
    if mu2s is not None:
        mode = "mu2"
    elif rmixes is not None:
        mode = "rmix"

    if args_ns.probe_only:
        kappas = kappas[:1] or [0.1]
        m_use = (m_Ss[0], m_Ss[-1]) if len(m_Ss) >= 2 else tuple(m_Ss)
        e_use = (epss[0], epss[len(epss) // 2], epss[-1])
        if mode == "phi_off":
            mu_use = (phi_offs[0], phi_offs[len(phi_offs) // 2], phi_offs[-1])
        elif mode == "rmix":
            mu_use = (rmixes[0], rmixes[-1])
        else:
            mu_use = (mu2s[0], mu2s[-1])
    else:
        m_use, e_use = m_Ss, epss
        mu_use = phi_offs if mode == "phi_off" else (rmixes if mode == "rmix" else mu2s)

    jobs = []
    for kappa in kappas:
        for m_S in m_use:
            for eps in e_use:
                for mu_val in mu_use:
                    jobs.append((eps, kappa, m_S, mu_val, mode))

    uniq, jobs_u = set(), []
    for e, k, m, mu_val, md in jobs:
        key = (round(float(e), 12), round(float(k), 6),
               round(float(m), 3), round(float(mu_val), 8), md)
        if key in uniq:
            continue
        uniq.add(key)
        jobs_u.append((e, k, m, mu_val, md))
    jobs = jobs_u

    log(f"mixed-portal GW scan → {args_ns.out}")
    log(f"progress log: {os.path.abspath(log_path)}")
    log(f"V0={args_ns.V0:.4e}  T_reh={args_ns.T_reh:g} GeV  workers={args_ns.workers}")
    log(f"κ={kappas}")
    log(f"ε={list(e_use)}")
    log(f"m_S={list(m_use)}")
    log(f"μ mode={mode}  values={list(mu_use)}")
    log(f"jobs: {len(jobs)}")

    base = SimpleNamespace(
        V0=args_ns.V0, v=args_ns.v, m_S=1000.0, s0=None,
        boson_phi=SET8_BOSON_PHI, fermion_phi=SET8_FERMION_PHI,
        boson_S=[(6.0, 1.0)], g_star=100.0, phi_thermal="set8",
        Tmin=None, Tmax=None, nT=args_ns.nT, refine=args_ns.refine,
        workers=args_ns.workers,
        a2v_n_fine=args_ns.a2v_n_fine, a2v_T_hi_fac=1.25, a2v_T_lo_fac=0.85,
        a2v_S_trigger=2000.0, T_reh=args_ns.T_reh,
        rmix=None, mu2_mix=None, gS_bath=args_ns.gS_bath, S_thermal=args_ns.S_thermal,
    )

    import pandas as pd
    rows = []
    t_all = time.time()
    for i, (eps, kappa, m_S, mu_val, md) in enumerate(jobs, 1):
        s0 = s0_from_mS(eps, args_ns.V0, m_S)
        if md == "phi_off":
            mu2 = mu2_for_phi_off(
                args_ns.V0, args_ns.v, s0, eps, kappa,
                SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)],
                100.0, "set8", float(mu_val))
            rmix = mu2 / (m_S ** 2) if m_S > 0 else 0.0
            phi_off_tgt = float(mu_val)
        elif md == "rmix":
            rmix = float(mu_val)
            mu2 = rmix * m_S ** 2
            phi_off_tgt = float("nan")
        else:
            mu2 = float(mu_val)
            rmix = mu2 / (m_S ** 2) if m_S > 0 else 0.0
            phi_off_tgt = float("nan")

        args = SimpleNamespace(**{**base.__dict__, "m_S": m_S,
                                  "rmix": None, "mu2_mix": mu2})
        log(f"\n======== [{i}/{len(jobs)}] m_S={m_S:g} κ={kappa:g} "
            f"ε={eps:.4g} φ_off_tgt={phi_off_tgt:g} rmix={rmix:.4g} "
            f"μ²={mu2:.4g} ========")
        try:
            row = run_one(eps, kappa, args, i, len(jobs))
        except Exception as exc:
            log(f"FAILED: {exc}")
            row = {
                "eps": eps, "kappa": kappa, "m_S": m_S, "rmix": rmix,
                "mu2_mix": mu2, "regime": f"error: {exc}",
                "h2Omega_peak_tot": float("nan"),
                "h2Omega_peak_0v": float("nan"),
            }
        row["phi_off_target"] = phi_off_tgt
        row["mu_mode"] = md
        a0 = row.get("h2Omega_peak_0v", float("nan"))
        at = row.get("h2Omega_peak_tot", float("nan"))
        ratio = (at / a0) if (isinstance(a0, (int, float)) and a0 > 0
                              and isinstance(at, (int, float))
                              and math.isfinite(at)) else float("nan")
        row["Omega_tot_over_0v"] = ratio
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(args_ns.out, "mixed_scan.csv"), index=False)
        log(f"  → regime={row.get('regime')}  Ω_tot/Ω_0v={ratio:.4g}  "
            f"Ω_tot={at:.4g}  Ω_0v={a0:.4g}  eps_eff={row.get('eps_eff', float('nan'))}")

    with open(os.path.join(args_ns.out, "mixed_scan.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    try:
        plot_scan(rows, args_ns.out, args_ns.V0, float("nan"))
    except Exception as exc:
        log(f"plot_scan skipped: {exc}")

    log(f"\nDONE in {time.time() - t_all:.0f}s  → {args_ns.out}")
    hits = [r for r in rows
            if "two-step" in str(r.get("regime", ""))
            or "nested" in str(r.get("regime", ""))]
    log(f"two-step/nested hits: {len(hits)}/{len(rows)}")
    amp = [r for r in rows
           if isinstance(r.get("Omega_tot_over_0v"), (int, float))
           and math.isfinite(r["Omega_tot_over_0v"])
           and r["Omega_tot_over_0v"] > 1.05]
    log(f"Ω_tot/Ω_0v > 1.05: {len(amp)}")
    for r in sorted(amp, key=lambda x: -x["Omega_tot_over_0v"])[:12]:
        log(f"  amp={r['Omega_tot_over_0v']:.3g}  ε={r['eps']:.3g}  "
            f"m_S={r['m_S']:.4g}  κ={r['kappa']:.3g}  "
            f"φ_off_tgt={r.get('phi_off_target', float('nan')):g}  "
            f"rmix={r.get('rmix', 0):.3g}  regime={r.get('regime')}")

    _pes._LOG.close()


if __name__ == "__main__":
    main()
