#!/usr/bin/env python3
"""Scan epsilon = Delta V_S / V0 for the portal two-step model.

Default: keep m_S fixed (1 TeV) and set
    s0(eps) = sqrt(2 eps V0) / m_S
so lam_S = m_S^4 / (4 eps V0). Larger eps → larger s0 → smaller lam_S
(stronger first-order S transition). Tree-level 0->v is eps-independent
for the cross-term potential; eps changes whether S nucleates first.

Live progress is written to <out>/progress.log (also mirrored to stdout).

Observability vs LISA / DECIGO / BBO: the GW spectrum h^2 Omega_GW(f) is
compared to each detector's sensitivity curve over that detector's frequency
band. ``meets_<DET>`` is True if Omega_GW(f) >= Omega_sens(f) anywhere in
band (the curves cross / the signal rises above the floor). ``max_ratio_<DET>``
is max_f Omega_GW(f)/Omega_sens(f) in band (nan if the spectrum is empty).

Example
-------
    PYTHONPATH=. python analysis/portalEpsScan.py \\
        --kappa 0.5 --kappa 2 --m_S 1000 --boson_S 6,1 \\
        --eps_min 1e-20 --eps_max 1e-4 --n_eps 12 --workers 8 \\
        --out data/portal/eps_scan
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.portalBounce import (  # noqa: E402
    SET8_BOSON_PHI,
    SET8_FERMION_PHI,
    SET8_V,
    SET8_V0,
    analyze,
    bounce_2d,
    scan,
)
from analysis.twoStepGW import (  # noqa: E402
    DEFAULT_T_REH,
    NESTED_DTN_MAX_GEV,
    apply_two_step_gw,
    make_step,
    spectrum,
    step_quantities,
)
from potential.portalPotential import PortalPotential, make_species  # noqa: E402

# global log handle set in main()
_LOG = None


def log(msg: str = ""):
    """Structured progress line (timestamped). Goes to stdout; with Tee also to progress.log."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


class _Tee:
    """Mirror stdout to a log file so bounce T= lines are watchable."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def _parse_ng(items, default):
    if not items:
        return list(default)
    return [tuple(float(x) for x in s.split(",")) for s in items]


def s0_from_mS(eps: float, V0: float, m_S: float) -> float:
    """s0 such that m_S^2 = 2 lam_S s0^2 with lam_S = eps V0 / s0^4."""
    if eps <= 0 or m_S <= 0 or V0 <= 0:
        raise ValueError("need eps, m_S, V0 > 0")
    return math.sqrt(2.0 * eps * V0) / m_S


def build_pot(V0, v, s0, eps, kappa, boson_phi, fermion_phi, boson_S, g_star,
              phi_thermal="paper", rmix=None, mu2_mix=None, gS_bath=0.0, S_thermal="plain"):
    """phi_thermal: 'paper' (m=g|phi|) or 'set8' (V_correct Debye+CW).

    rmix / mu2_mix: if either is set and nonzero, return PortalMixedPotential
    with V ⊃ −μ²_mix φ S (μ²_mix = rmix · m_S² when rmix is given).
    """
    sp_phi = make_species(boson_phi) + make_species(fermion_phi, fermion=True)
    sp_S = make_species(boson_S)
    use_mix = ((rmix is not None and float(rmix) > 0.0)
               or (mu2_mix is not None and float(mu2_mix) > 0.0))
    if use_mix:
        from potential.portalMixedPotential import PortalMixedPotential
        return PortalMixedPotential(
            V0=V0, v=v, s0=s0, eps=eps, kappa=kappa,
            species_phi=sp_phi, species_S=sp_S, g_star=g_star,
            phi_thermal=phi_thermal,
            rmix=float(rmix) if rmix is not None else None,
            mu2_mix=float(mu2_mix) if mu2_mix is not None else 0.0,
            gS_bath=float(gS_bath or 0.0), S_thermal=S_thermal,
        )
    return PortalPotential(
        V0=V0, v=v, s0=s0, eps=eps, kappa=kappa,
        species_phi=sp_phi, species_S=sp_S, g_star=g_star,
        phi_thermal=phi_thermal, gS_bath=float(gS_bath or 0.0), S_thermal=S_thermal,
    )


DETECTORS = (
    ("LISA", 1e-5, 0.5),
    ("DECIGO", 1e-3, 100.0),
    ("BBO", 1e-3, 100.0),
)


def _qs_ref_from_row(row: dict):
    """Rebuild minimal peak dicts for spectrum / sensitivity from a scan row."""
    qs = []
    for key in ("step1", "step2", "composite", "cross"):
        fp = row.get(f"f_peak_{key} [Hz]", float("nan"))
        amp = row.get(f"h2Omega_peak_{key}", float("nan"))
        if np.isfinite(fp) and np.isfinite(amp) and amp > 0 and fp > 0:
            qs.append({"label": key, "f_peak [Hz]": float(fp), "h2Omega_peak": float(amp)})
    ref = None
    fp0 = row.get("f_peak_0v [Hz]", float("nan"))
    amp0 = row.get("h2Omega_peak_0v", float("nan"))
    if np.isfinite(fp0) and np.isfinite(amp0) and amp0 > 0 and fp0 > 0:
        ref = {"label": "0->v", "f_peak [Hz]": float(fp0), "h2Omega_peak": float(amp0)}
    return qs, ref


def _pack_gw_row(r: dict, qs: list[dict], ref: dict | None, meta: dict | None = None) -> dict:
    """Write peak / nested diagnostics from qs into a scan row + total peak."""
    meta = meta or {}

    def _peak(label, key):
        return next((q[key] for q in qs if q["label"] == label), float("nan"))

    r["gw_formula"] = meta.get("gw_formula", "Sec.IV sequential")
    r["f_comp"] = meta.get("f_comp", 0.0)
    r["chi_nest"] = meta.get("chi_nest", float("nan"))
    r["beta_eff_H"] = meta.get("beta_eff_H", float("nan"))
    r["kappa_comp"] = meta.get("kappa_comp", float("nan"))
    r["nested_amp_factor"] = meta.get("nested_amp_factor", float("nan"))

    r["f_peak_step1_undiluted [Hz]"] = next(
        (q.get("f_peak_undiluted [Hz]", float("nan")) for q in qs if q["label"] == "step1"),
        float("nan"))
    r["h2Omega_peak_step1_undiluted"] = next(
        (q.get("h2Omega_peak_undiluted", float("nan")) for q in qs if q["label"] == "step1"),
        float("nan"))
    for lab in ("step1", "step2", "composite", "cross"):
        r[f"f_peak_{lab} [Hz]"] = _peak(lab, "f_peak [Hz]")
        r[f"h2Omega_peak_{lab}"] = _peak(lab, "h2Omega_peak")
    r["N_e_inflate"] = next(
        (q.get("N_e_inflate", 0.0) for q in qs if q["label"] == "step1"), 0.0)
    r["f_peak_0v [Hz]"] = ref["f_peak [Hz]"] if ref else float("nan")
    r["h2Omega_peak_0v"] = ref["h2Omega_peak"] if ref else float("nan")

    qs_row, ref_row = _qs_ref_from_row(r)
    fps = [q["f_peak [Hz]"] for q in qs_row]
    if ref_row and not qs_row:
        fps.append(ref_row["f_peak [Hz]"])
    if fps:
        f = np.geomspace(min(fps) * 1e-3, max(fps) * 1e3, 2000)
        tot = _total_spectrum(qs_row, None if qs_row else ref_row, f)
        if tot.max() > 0:
            i = int(np.argmax(tot))
            r["f_peak_tot [Hz]"] = float(f[i])
            r["h2Omega_peak_tot"] = float(tot[i])
        else:
            r["f_peak_tot [Hz]"] = float("nan")
            r["h2Omega_peak_tot"] = float("nan")
    else:
        r["f_peak_tot [Hz]"] = float("nan")
        r["h2Omega_peak_tot"] = float("nan")
    r.update(_meets_sensitivity(qs_row, ref_row))
    return r


def recompute_gw_secIV(rows: list[dict], V0: float, T_reh: float = DEFAULT_T_REH,
                       g_star: float = 100.0) -> list[dict]:
    """Rebuild GW peaks from (eps, T_n, beta/H) in a scan CSV.

    Separated (ΔT_n > 0.05 TeV): Sec. II/IV envelope with D_inf dilution.
    Nested (ΔT_n ≤ 0.05 TeV or regime tag): Sec. III.C multi-boundary spectrum.
    """
    out = []
    for r0 in rows:
        r = dict(r0)
        eps = float(r.get("eps_eff", r.get("eps", float("nan"))))
        if not np.isfinite(eps):
            eps = float(r.get("eps", float("nan")))
        T1 = float(r.get("T_n_step1", r.get("T_n_0_to_S", float("nan"))))
        T2 = float(r.get("T_n_step2", r.get("T_n_S_to_v", float("nan"))))
        b1 = float(r.get("betaH_step1", float("nan")))
        b2 = float(r.get("betaH_step2", float("nan")))
        b0 = float(r.get("betaH_phi", float("nan")))
        T0 = float(r.get("T_n_0_to_v", r.get("T_n_phi", float("nan"))))
        dTn = float(r.get("delta_T_n_S", float("nan")))
        if not np.isfinite(dTn) and np.isfinite(T1) and np.isfinite(T2):
            dTn = T1 - T2
        stuck = "stuck" in str(r.get("regime", "")) or "stuck" in str(r.get("sequence", ""))
        # Sec. III.A/C: apply nested spectrum only for ΔT_n ≤ 0.05 TeV (or
        # Regime II with unknown ΔT_n → treat as fully composite).
        force_nested = (
            (np.isfinite(dTn) and 0 <= dTn <= NESTED_DTN_MAX_GEV)
            or (
                not np.isfinite(dTn)
                and ("Regime II" in str(r.get("sequence", ""))
                     or "nested" in str(r.get("regime", "")).lower())
            )
        )

        qs_raw = []
        if np.isfinite(eps) and eps > 0 and np.isfinite(T1) and np.isfinite(b1) and b1 > 0:
            qs_raw.append(step_quantities(make_step(
                "step1", eps * V0, T1, b1, V0, T_reh, g_star=g_star)))
        if (not stuck and np.isfinite(eps) and 0 < eps < 1
                and np.isfinite(T2) and np.isfinite(b2) and b2 > 0):
            qs_raw.append(step_quantities(make_step(
                "step2", (1.0 - eps) * V0, T2, b2, V0, T_reh, g_star=g_star)))
        elif (not stuck and np.isfinite(eps) and 0 < eps < 1
              and np.isfinite(T2) and np.isfinite(b1) and b1 > 0
              and not (np.isfinite(b2) and b2 > 0)):
            qs_raw.append(step_quantities(make_step(
                "step2", (1.0 - eps) * V0, T2, b1, V0, T_reh, g_star=g_star)))

        qs, meta = apply_two_step_gw(
            qs_raw, eps=eps if np.isfinite(eps) else 0.0, stuck=stuck,
            dTn=dTn, force_nested=force_nested and not stuck)
        ref = None
        if np.isfinite(T0) and np.isfinite(b0) and b0 > 0:
            ref = step_quantities(make_step(
                "0->v", V0, T0, b0, V0, T_reh, g_star=g_star))
        out.append(_pack_gw_row(r, qs, ref, meta))
    return out


# Back-compat name
def reapply_inter_step_dilution(rows, V0=None, T_reh=DEFAULT_T_REH, g_star=100.0):
    if V0 is None:
        raise ValueError("recompute_gw_secIV requires V0")
    return recompute_gw_secIV(rows, V0=V0, T_reh=T_reh, g_star=g_star)


def _total_spectrum(qs, ref, f):
    """Sum envelope spectra of nucleation steps; fall back to 0->v reference."""
    tot = np.zeros_like(f, dtype=float)
    for q in qs:
        if q.get("h2Omega_peak", 0) > 0 and q.get("f_peak [Hz]", 0) > 0:
            tot = tot + spectrum(f, q)
    if tot.max() <= 0 and ref is not None:
        tot = spectrum(f, ref)
    return tot


def _meets_sensitivity(qs, ref):
    """Check whether h^2 Omega_GW(f) rises above each detector sensitivity curve.

    Returns dict with:
      meets_<DET>     : True if Omega_GW(f) >= Omega_sens(f) for some f in band
      max_ratio_<DET> : max_f Omega_GW/Omega_sens in band
      f_cross_<DET>   : frequency of that maximum ratio (where the signal is
                        most above / least below the curve)
    """
    out = {}
    try:
        from analysis import gwSpectrum as G
        sens = {
            "LISA": G.sensitivity_LISA,
            "DECIGO": G.sensitivity_DECIGO,
            "BBO": G.sensitivity_BBO,
        }
    except Exception as exc:
        log(f"  detector curves unavailable: {exc}")
        for name, _, _ in DETECTORS:
            out[f"meets_{name}"] = False
            out[f"max_ratio_{name}"] = float("nan")
            out[f"f_cross_{name}"] = float("nan")
        return out

    if not qs and ref is None:
        for name, _, _ in DETECTORS:
            out[f"meets_{name}"] = False
            out[f"max_ratio_{name}"] = float("nan")
            out[f"f_cross_{name}"] = float("nan")
        return out

    for name, lo, hi in DETECTORS:
        f = np.geomspace(lo, hi, 800)
        sig = _total_spectrum(qs, ref, f)
        noise = np.asarray(sens[name](f), dtype=float)
        ok = np.isfinite(sig) & np.isfinite(noise) & (noise > 0) & (sig > 0)
        if not ok.any():
            out[f"meets_{name}"] = False
            out[f"max_ratio_{name}"] = float("nan")
            out[f"f_cross_{name}"] = float("nan")
            continue
        ratio = np.full_like(sig, np.nan)
        ratio[ok] = sig[ok] / noise[ok]
        i = int(np.nanargmax(ratio))
        rmax = float(ratio[i])
        out[f"meets_{name}"] = bool(np.isfinite(rmax) and rmax >= 1.0)
        out[f"max_ratio_{name}"] = rmax
        out[f"f_cross_{name}"] = float(f[i])
    return out


def _gw_from_info(pot, info, T_reh: float = DEFAULT_T_REH):
    """GW from nucleation info: Sec. II/IV sequential or Sec. III.C nested."""
    qs_raw, ref = [], None
    beta1 = info.get("step1", {}).get("beta/H", np.nan)
    for key in ("step1", "step2"):
        n = info.get(key, {})
        T_n, bH = n.get("T_n", np.nan), n.get("beta/H", np.nan)
        if key == "step2" and np.isfinite(T_n) and not (np.isfinite(bH) and bH > 0) \
                and "immediate" in n.get("note", ""):
            bH = beta1
        if np.isfinite(T_n) and np.isfinite(bH) and bH > 0 and np.isfinite(n.get("delV", np.nan)):
            qs_raw.append(step_quantities(make_step(
                key, n["delV"], T_n, bH, pot.V0, T_reh, g_star=pot.g_star)))
    stuck = "stuck" in info.get("sequence", "")
    seq = info.get("sequence", "")
    eps = float(getattr(pot, "eps_eff", getattr(pot, "eps", 0.0)) or 0.0)
    if not (0 < eps < 1) and len(qs_raw) >= 1 and pot.V0 > 0:
        eps = float(qs_raw[0].get("frac_delV_V0", 0.0))
    T1 = info.get("step1", {}).get("T_n", float("nan"))
    T2 = info.get("step2", {}).get("T_n", float("nan"))
    dTn = (T1 - T2) if (np.isfinite(T1) and np.isfinite(T2)) else float("nan")
    # Nested spectrum only inside ΔT_n ≤ 0.05 TeV; Regime II without ΔT_n → composite
    force_nested = (
        (not np.isfinite(dTn))
        and ("Regime II" in seq or "nested" in seq.lower())
        and not stuck
    )
    qs, meta = apply_two_step_gw(
        qs_raw, eps=eps, stuck=stuck, dTn=dTn, force_nested=force_nested)
    n = info.get("phi direct (0->v, single-step reference)", {})
    if np.isfinite(n.get("T_n", np.nan)) and np.isfinite(n.get("beta/H", np.nan)):
        ref = step_quantities(make_step(
            "0->v", pot.V0, n["T_n"], n["beta/H"], pot.V0, T_reh, g_star=pot.g_star))
    fps = [q["f_peak [Hz]"] for q in qs
           if q.get("h2Omega_peak", 0) > 0 and q.get("f_peak [Hz]", 0) > 0]
    if ref and not fps:
        fps.append(ref["f_peak [Hz]"])
    if not fps:
        return qs, ref, float("nan"), float("nan"), meta
    f = np.geomspace(min(fps) * 1e-3, max(fps) * 1e3, 2000)
    tot = _total_spectrum(qs, ref if not qs else None, f)
    if tot.max() <= 0:
        return qs, ref, float("nan"), float("nan"), meta
    i = int(np.argmax(tot))
    return qs, ref, float(f[i]), float(tot[i]), meta


def _regime_tag(info: dict) -> str:
    seq = info.get("sequence", "")
    if "single step" in seq:
        return "0->v wins"
    if "S never" in seq:
        return "0->v wins"
    if "stuck" in seq or "fails" in seq:
        return "stuck in A"
    if "Regime II" in seq or "nested" in seq:
        return "two-step nested"
    if "Regime I" in seq:
        return "two-step separated"
    return seq[:40] if seq else "unknown"


def run_one(eps, kappa, args, i_pt: int, n_pt: int) -> dict:
    s0 = s0_from_mS(eps, args.V0, args.m_S)
    rmix = getattr(args, "rmix", None)
    mu2_mix = getattr(args, "mu2_mix", None)
    pot = build_pot(args.V0, args.v, s0, eps, kappa,
                    args.boson_phi, args.fermion_phi, args.boson_S, args.g_star,
                    phi_thermal=getattr(args, "phi_thermal", "paper"), gS_bath=getattr(args, "gS_bath", 0.0), S_thermal=getattr(args, "S_thermal", "plain"),
                    rmix=rmix, mu2_mix=mu2_mix)
    m_S = math.sqrt(pot.m_S2)
    mu2 = float(getattr(pot, "mu2_mix", 0.0) or 0.0)
    rmix_eff = (mu2 / pot.m_S2) if pot.m_S2 > 0 else 0.0
    log(f"--- [{i_pt}/{n_pt}] kappa={kappa:g}  eps={eps:.4e}  s0={s0:.4e}  "
        f"m_S={m_S:.4g} (target {args.m_S:g})  lam_S={pot.lam_S:.4g}  "
        f"rmix={rmix_eff:.4g}  μ²_mix={mu2:.4g}  "
        f"T0_S={pot.T0_S():.4g}  T0_phi={pot.T0_phi():.4g}")
    if abs(m_S - args.m_S) / args.m_S > 1e-6:
        log(f"  WARNING: m_S mismatch {m_S} vs {args.m_S}")

    # 0→S nucleates near ∼1.5 T0_S (S3/T∼140); 1.3× clipped T_n low and left
    # high-T NaNs in the 0→a column.  Use 2.5× so the S3/T curve is resolved.
    Tmax = args.Tmax or 2.5 * max(pot.T0_S(), pot.T0_phi())
    Tmin = args.Tmin or 0.02 * min(pot.T0_S(), pot.T0_phi())
    Tmin = max(Tmin, 5.0)
    Ts = np.geomspace(Tmax, Tmin, args.nT)
    log(f"  T scan [{Tmin:.4g}, {Tmax:.4g}]  (T0_S={pot.T0_S():.4g}, T0_phi={pot.T0_phi():.4g})")

    # Pass 1: 1D channels only (fast) to decide if S can win the race.
    # T_n(S->v) needs the 2D bounce → only available after pass2.
    t0 = time.time()
    log(f"  pass1: 1D only [T_n(0->S), T_n(0->v)]; T_n(S->v) deferred to pass2")
    res1 = scan(pot, Ts, refine=args.refine, verbose=False, workers=args.workers, do_2d=False)
    info1 = analyze(pot, res1, b4=None)
    TnS = info1.get("S step (0->a)", {}).get("T_n", float("nan"))
    TnPhi = info1.get("phi direct (0->v, single-step reference)", {}).get("T_n", float("nan"))
    s_first = np.isfinite(TnS) and (not np.isfinite(TnPhi) or TnS > TnPhi)
    log(f"  pass1 done in {time.time()-t0:.1f}s: T_n(0->S)={TnS:.4g}  "
        f"T_n(0->v)={TnPhi:.4g}  T_n(S->v)=pending  "
        f"({'S-first → run pass2' if s_first else '0->v wins → skip pass2'})")

    info = info1
    b4 = None
    if s_first and pot.cross:
        t1 = time.time()
        log("  pass2: 2D bounce for T_n(S->v) = A->B  (this is the slow step)")
        if kappa > 1:
            try:
                out4 = bounce_2d(pot, 0.0, d=4)
                b4 = out4[0] if out4 else None
                log(f"  S4(A->B)={b4.S:.5g}" if b4 else "  S4(A->B)=none")
            except Exception as exc:
                log(f"  S4 failed: {exc}")
                b4 = None
        # verbose=True streams each T into progress.log so the run is watchable.
        # a2v fine pass: once S3/T(A->B) < trigger, densify T in that window
        # (avoids a uniformly fine grid from the start).
        res2 = scan(pot, Ts, refine=args.refine, verbose=True, workers=args.workers,
                    do_2d=True,
                    a2v_S_trigger=getattr(args, "a2v_S_trigger", 2000.0),
                    a2v_n_fine=getattr(args, "a2v_n_fine", 64),
                    a2v_T_hi_fac=getattr(args, "a2v_T_hi_fac", 1.25),
                    a2v_T_lo_fac=getattr(args, "a2v_T_lo_fac", 0.85))
        info = analyze(pot, res2, b4=b4)
        TnSv = info.get("step2", {}).get("T_n", float("nan"))
        log(f"  pass2 done in {time.time()-t1:.1f}s: T_n(0->S)={TnS:.4g}  "
            f"T_n(S->v)={TnSv:.4g}  ΔT_n={(TnS - TnSv) if np.isfinite(TnS) and np.isfinite(TnSv) else float('nan'):.4g}  "
            f"seq={info.get('sequence', '')}")
    else:
        log("  skip pass2 → T_n(S->v)=nan (no intermediate S phase)")

    # Mixed: refresh ε_eff / ΔV at nucleation T of A′→B (or T=0 fallback)
    if hasattr(pot, "_refresh_delV_eff") and getattr(pot, "mixed", False):
        Tn2 = info.get("step2", {}).get("T_n", float("nan"))
        T_ref = float(Tn2) if np.isfinite(Tn2) else 0.0
        pot._refresh_delV_eff(T_ref)
        # keep analyze-side delV in sync for Sec. IV amp
        if "step1" in info:
            info["step1"]["delV"] = pot.delV1
        if "step2" in info:
            info["step2"]["delV"] = pot.delV2

    T_reh = float(getattr(args, "T_reh", DEFAULT_T_REH))
    qs, ref, f_tot, h2_tot, gw_meta = _gw_from_info(pot, info, T_reh=T_reh)
    det = _meets_sensitivity(qs, ref)
    s1, s2 = info.get("step1", {}), info.get("step2", {})
    phi = info.get("phi direct (0->v, single-step reference)", {})
    s_ch = info.get("S step (0->a)", {})
    row = {
        "eps": eps,
        "kappa": kappa,
        "s0": s0,
        "m_S": m_S,
        "rmix": rmix_eff,
        "mu2_mix": mu2,
        "eps_eff": float(getattr(pot, "eps_eff", eps)),
        "lam_S": pot.lam_S,
        "lam_phiS": pot.lam_phiS,
        "lam_eff": (pot.lam_phiS**2 / (4 * pot.lam_S) - pot.lam_phi) if pot.cross else float("nan"),
        "T0_S": pot.T0_S(),
        "T0_phi": pot.T0_phi(),
        "sequence": info.get("sequence", ""),
        # nucleation temperatures (GeV)
        "T_n_0_to_S": s_ch.get("T_n", float("nan")),   # O -> A along S
        "T_n_S_to_v": s2.get("T_n", float("nan")) if pot.cross else float("nan"),  # A -> B
        "T_n_0_to_v": phi.get("T_n", float("nan")),    # direct O -> (v,0) reference
        # legacy aliases
        "T_n_S": s_ch.get("T_n", float("nan")),
        "T_n_phi": phi.get("T_n", float("nan")),
        "T_n_a2v": s2.get("T_n", float("nan")) if pot.cross else float("nan"),
        "T_n_step1": s1.get("T_n", float("nan")),
        "T_n_step2": s2.get("T_n", float("nan")),
        "delta_T_n_S": (
            (s_ch.get("T_n", float("nan")) - s2.get("T_n", float("nan")))
            if pot.cross else float("nan")
        ),  # T_n(0->S) - T_n(S->v); larger ⇒ more separated steps
        "betaH_step1": s1.get("beta/H", float("nan")),
        "betaH_step2": s2.get("beta/H", float("nan")),
        "betaH_phi": phi.get("beta/H", float("nan")),
        "S4": s2.get("S4 (T=0)", float("nan")),
        "regime": _regime_tag(info),
        "f_peak_tot [Hz]": f_tot,
        "h2Omega_peak_tot": h2_tot,
        **det,
    }
    row = _pack_gw_row(row, qs, ref, gw_meta)
    # _pack_gw_row overwrites tot/det; restore the already-computed totals if finite
    if np.isfinite(f_tot):
        row["f_peak_tot [Hz]"] = f_tot
        row["h2Omega_peak_tot"] = h2_tot
    row.update(det)
    log(f"  RESULT regime={row['regime']}")
    log(f"  T_n(0->S)={row['T_n_0_to_S']:.4g}  T_n(S->v)={row['T_n_S_to_v']:.4g}  "
        f"ΔT_n={row['delta_T_n_S']:.4g}  T_n(0->v ref)={row['T_n_0_to_v']:.4g}")
    Ne = row.get("N_e_inflate", 0.0)
    ne_s = f"  N_e(1→2)={Ne:.4g}" if (isinstance(Ne, (int, float)) and Ne > 0) else ""
    nest_s = ""
    if gw_meta.get("nested"):
        nest_s = (f"  [{gw_meta.get('gw_formula')}] f_comp={gw_meta.get('f_comp', 0):.3g}"
                  f" χ={gw_meta.get('chi_nest', float('nan')):.3g}"
                  f" amp×={gw_meta.get('nested_amp_factor', float('nan')):.3g}")
    log(f"  GW: f_peak={f_tot:.3g} Hz  h2Omega_peak={h2_tot:.3g}{ne_s}{nest_s}")
    meets = ", ".join(
        f"{name}={'YES' if det[f'meets_{name}'] else 'no'}"
        f"(maxΩ/sens={det[f'max_ratio_{name}']:.3g} @ {det[f'f_cross_{name}']:.3g} Hz)"
        for name, _, _ in DETECTORS
    )
    log(f"  spectrum vs sensitivity (crosses curve?): {meets}")
    return row


def plot_scan(rows, out_dir, V0, m_S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis import gwSpectrum as G
    from analysis.twoStepGW import shape

    os.makedirs(out_dir, exist_ok=True)
    kappas = sorted({r["kappa"] for r in rows})
    cmap = {0.5: "C0", 1.5: "C1", 2.0: "C3", 2: "C3"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in kappas:
        rs = sorted([r for r in rows if r["kappa"] == k], key=lambda r: r["eps"])
        eps = np.array([r["eps"] for r in rs])
        col = cmap.get(k, "C2")
        ax.loglog(eps, [r.get("T_n_0_to_v", r.get("T_n_phi")) for r in rs], "^--", color=col, ms=7,
                  label=rf"$\kappa={k:g}$: $T_n(0\to v)$")
        ax.loglog(eps, [r.get("T_n_0_to_S", r.get("T_n_S")) for r in rs], "o-", color=col, ms=6,
                  label=rf"$\kappa={k:g}$: $T_n(0\to S)$")
        ax.loglog(eps, [r.get("T_n_S_to_v", r.get("T_n_a2v")) for r in rs], "s-", color=col, ms=5,
                  alpha=0.8, label=rf"$\kappa={k:g}$: $T_n(S\to v)$")
        win = [r["regime"] == "0->v wins" for r in rs]
        if any(win):
            e_win = eps[np.array(win)]
            ax.axvspan(e_win.min() / 1.05, e_win.max() * 1.05, color=col, alpha=0.08)
    ax.set(xlabel=r"$\epsilon=\Delta V_S/V_0$  ($m_S$ fixed, $s_0\propto\sqrt{\epsilon}$)",
           ylabel=r"$T_n$ [GeV]",
           title=rf"Nucleation vs $\epsilon$ (set8 $\phi$, $m_S={m_S:g}$ GeV, $V_0={V0:.2e}$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Tn_vs_eps.png"), dpi=150)
    plt.close(fig)

    # separation between step-1 and step-2 nucleation
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in kappas:
        rs = sorted([r for r in rows if r["kappa"] == k], key=lambda r: r["eps"])
        eps = np.array([r["eps"] for r in rs])
        dT = np.array([r.get("delta_T_n_S", float("nan")) for r in rs], dtype=float)
        ax.semilogx(eps, dT, "o-", color=cmap.get(k, "C2"), ms=7,
                    label=rf"$\kappa={k:g}$: $T_n(0\to S)-T_n(S\to v)$")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(xlabel=r"$\epsilon$", ylabel=r"$\Delta T_n$ [GeV]",
           title=r"Step separation: $T_n(0\to S) - T_n(S\to v)$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_Tn_vs_eps.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.4]})
    regime_y = {"0->v wins": 0, "stuck in A": 1, "two-step nested": 2, "two-step separated": 3}
    for k in kappas:
        rs = sorted([r for r in rows if r["kappa"] == k], key=lambda r: r["eps"])
        axes[0].semilogx([r["eps"] for r in rs],
                         [regime_y.get(r["regime"], -1) for r in rs],
                         "o-", color=cmap.get(k, "C2"), ms=8, label=rf"$\kappa={k:g}$")
    axes[0].set_yticks(list(regime_y.values()))
    axes[0].set_yticklabels(list(regime_y.keys()))
    axes[0].set_ylabel("regime")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title(r"Exit regime vs $\epsilon$")

    for k in kappas:
        rs = sorted([r for r in rows if r["kappa"] == k], key=lambda r: r["eps"])
        eps = np.array([r["eps"] for r in rs])
        col = cmap.get(k, "C2")
        for det, ls in (("LISA", "-"), ("DECIGO", "--"), ("BBO", ":")):
            ratio = np.array([r.get(f"max_ratio_{det}", float("nan")) for r in rs], dtype=float)
            axes[1].loglog(eps, ratio, ls, color=col, marker="o", ms=4,
                           label=rf"$\kappa={k:g}$ / {det}")
    axes[1].axhline(1.0, color="k", lw=1.2, label=r"$\Omega_{\rm GW}=\Omega_{\rm sens}$ (crosses)")
    axes[1].set(xlabel=r"$\epsilon$",
                ylabel=r"$\max_f\, h^2\Omega_{\rm GW}(f)/\Omega_{\rm sens}(f)$",
                title="Spectrum vs sensitivity: >1 ⇒ GW curve rises above detector floor in-band")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regime_and_sensitivity_vs_eps.png"), dpi=150)
    plt.close(fig)

    from analysis.published_plisc import add_detector_curves

    for k in kappas:
        rs = sorted([r for r in rows if r["kappa"] == k], key=lambda r: r["eps"])
        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        add_detector_curves(ax)
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(rs)))
        f = np.geomspace(1e-5, 1e5, 3000)
        # 0->v single-step reference (eps-independent for cross-term tree): plot once
        y0v = None
        for r in rs:
            fp0 = r.get("f_peak_0v [Hz]", float("nan"))
            amp0 = r.get("h2Omega_peak_0v", float("nan"))
            if np.isfinite(fp0) and np.isfinite(amp0) and amp0 > 0 and fp0 > 0:
                y0v = amp0 * shape(f / fp0)
                break
        if y0v is not None and y0v.max() > 0:
            ax.loglog(f, y0v, color="k", lw=2.2, ls="--", alpha=0.85,
                      label=r"$0\to v$ (single-step ref.)", zorder=5)
        for r, col in zip(rs, colors):
            y = np.zeros_like(f)
            used = False
            for key in ("step1", "step2"):
                fp = r.get(f"f_peak_{key} [Hz]", float("nan"))
                amp = r.get(f"h2Omega_peak_{key}", float("nan"))
                if np.isfinite(fp) and np.isfinite(amp) and amp > 0 and fp > 0:
                    y = y + amp * shape(f / fp)
                    used = True
            # if no two-step steps, show that eps's own 0->v (still list it)
            if not used:
                fp, amp = r.get("f_peak_0v [Hz]", float("nan")), r.get("h2Omega_peak_0v", float("nan"))
                if np.isfinite(fp) and np.isfinite(amp) and amp > 0 and fp > 0:
                    y = amp * shape(f / fp)
                    used = True
            if not used or y.max() <= 0:
                continue
            ax.loglog(f, y, color=col, lw=1.6, zorder=5,
                      label=rf"$\epsilon={r['eps']:.2e}$ ({r['regime']})")
        ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{\rm GW}$",
               title=rf"SGWB vs $\epsilon$, $\kappa={k:g}$, $m_S={m_S:g}$ GeV")
        ax.set_ylim(1e-25, 1e-5)
        ax.set_xlim(1e-5, 1e5)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gw_spectra_kappa_{k:g}.png"), dpi=160)
        plt.close(fig)

        # Per-stage GW: 0->S, S->v, and 0->v reference separately
        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        add_detector_curves(ax)
        # one 0->v reference (eps-independent for cross term)
        plotted_0v = False
        for r in rs:
            fp0 = r.get("f_peak_0v [Hz]", float("nan"))
            amp0 = r.get("h2Omega_peak_0v", float("nan"))
            if np.isfinite(fp0) and np.isfinite(amp0) and amp0 > 0 and fp0 > 0:
                ax.loglog(f, amp0 * shape(f / fp0), "k--", lw=2.2, alpha=0.85,
                          label=r"$0\to v$ (single-step ref.)", zorder=5)
                plotted_0v = True
                break
        stage_style = (
            ("step1", ":", 1.5, r"$0\to S$"),
            ("step2", "-", 1.8, r"$S\to v$"),
        )
        for r, col in zip(rs, colors):
            for key, ls, lw, lab in stage_style:
                fp = r.get(f"f_peak_{key} [Hz]", float("nan"))
                amp = r.get(f"h2Omega_peak_{key}", float("nan"))
                if not (np.isfinite(fp) and np.isfinite(amp) and amp > 0 and fp > 0):
                    continue
                ax.loglog(f, amp * shape(f / fp), color=col, ls=ls, lw=lw, zorder=5,
                          label=rf"$\epsilon={r['eps']:.2e}$: {lab}")
        ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{\rm GW}$",
               title=rf"SGWB by stage, $\kappa={k:g}$ "
                     rf"(dotted $0\to S$ diluted by $e^{{-4N_e}}$, solid $S\to v$"
                     + (r", dashed $0\to v$)" if plotted_0v else ")"))
        ax.set_ylim(1e-25, 1e-5)
        ax.set_xlim(1e-5, 1e5)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gw_stages_kappa_{k:g}.png"), dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in kappas:
        rs = sorted([r for r in rows if r["kappa"] == k], key=lambda r: r["eps"])
        eps = [r["eps"] for r in rs]
        col = cmap.get(k, "C2")
        ax.loglog(eps, [r["h2Omega_peak_tot"] for r in rs], "o-",
                  color=col, label=rf"$\kappa={k:g}$ total")
        ax.loglog(eps, [r.get("h2Omega_peak_step1", float("nan")) for r in rs], "s:",
                  color=col, alpha=0.85, label=rf"$\kappa={k:g}$ $0\to S$")
        ax.loglog(eps, [r.get("h2Omega_peak_step2", float("nan")) for r in rs], "D-",
                  color=col, alpha=0.85, label=rf"$\kappa={k:g}$ $S\to v$")
        ax.loglog(eps, [r["h2Omega_peak_0v"] for r in rs], "^--",
                  color=col, alpha=0.65, label=rf"$\kappa={k:g}$ $0\to v$ ref")
    ax.set(xlabel=r"$\epsilon$", ylabel=r"$h^2\Omega_{\rm peak}$",
           title=rf"Peak SGWB by stage vs eps ($m_S={m_S:g}$ GeV fixed)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h2Omega_peak_vs_eps.png"), dpi=150)
    plt.close(fig)

    # s0(eps) diagnostic
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rs0 = sorted({(r["eps"], r["s0"]) for r in rows})
    ax.loglog([e for e, _ in rs0], [s for _, s in rs0], "ko-")
    ax.set(xlabel=r"$\epsilon$", ylabel=r"$s_0$ [GeV]",
           title=rf"$s_0=\sqrt{{2\epsilon V_0}}/m_S$ with $m_S={m_S:g}$ GeV")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "s0_vs_eps.png"), dpi=150)
    plt.close(fig)


def main():
    global _LOG
    p = argparse.ArgumentParser(description="Portal model: epsilon scan (fixed m_S)")
    p.add_argument("--V0", type=float, default=SET8_V0)
    p.add_argument("--v", type=float, default=SET8_V)
    p.add_argument("--m_S", type=float, default=1000.0,
                   help="fixed S tree mass [GeV]; s0 = sqrt(2 eps V0)/m_S (default 1 TeV)")
    p.add_argument("--s0", type=float, default=None,
                   help="if set, fix s0 instead of m_S (legacy mode)")
    p.add_argument("--kappa", type=float, action="append", default=None)
    p.add_argument("--eps_min", type=float, default=1e-20)
    p.add_argument("--eps_max", type=float, default=1e-4)
    p.add_argument("--n_eps", type=int, default=12)
    p.add_argument("--boson_phi", action="append", default=None,
                   help="n,g bosons on phi (default 20,1.09). Overrides --g_phi.")
    p.add_argument("--fermion_phi", action="append", default=None,
                   help="n,g fermions on phi (default 20,1.09)")
    p.add_argument("--boson_S", action="append", default=None,
                   help="n,g bosons on S (default 6,1). Overrides --g_S.")
    p.add_argument("--g_phi", type=float, default=None,
                   help="shorthand: bosons+fermions on phi with this g (n=20 each)")
    p.add_argument("--g_S", type=float, default=None,
                   help="shorthand: n=6 bosons on S with this g")
    p.add_argument("--g_star", type=float, default=100.0)
    p.add_argument("--gS_bath", type=float, default=0.0,
                   help="shared-bath coupling: phi-bath masses get 1/2 gS^2 S^2 (0 = separate sectors)")
    p.add_argument("--T_reh", type=float, default=DEFAULT_T_REH,
                   help="flaton decay / reheating T [GeV] for Sec. IV R_md "
                        f"(default {DEFAULT_T_REH:g} = 0.1 TeV)")
    p.add_argument("--phi_thermal", choices=("paper", "set8"), default="paper",
                   help="paper = m=g|phi|; set8 = V_correct Debye+mb2_0+CW on phi")
    p.add_argument("--Tmin", type=float, default=None)
    p.add_argument("--Tmax", type=float, default=None)
    p.add_argument("--nT", type=int, default=28)
    p.add_argument("--refine", type=int, default=12)
    p.add_argument("--a2v_n_fine", type=int, default=64,
                   help="triggered fine T points for A->B near T0_phi (0=off)")
    p.add_argument("--a2v_T_hi_fac", type=float, default=1.25,
                   help="fine window upper = fac * T0_phi (~1.4 TeV for set8)")
    p.add_argument("--a2v_T_lo_fac", type=float, default=0.85,
                   help="fine window lower = fac * T0_phi (~1.0 TeV for set8)")
    p.add_argument("--a2v_S_trigger", type=float, default=2000.0,
                   help="also widen fine window where S3/T(a->v) < this")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="data/portal/eps_scan")
    p.add_argument("--replot", action="store_true",
                   help="reload <out>/*_scan.csv, recompute GW "
                        "(Sec.II/IV sequential or Sec.III.C nested), rewrite plots")
    args = p.parse_args()

    if args.replot:
        import pandas as pd
        csv_path = os.path.join(args.out, "eps_scan.csv")
        df = pd.read_csv(csv_path)
        rows = recompute_gw_secIV(df.to_dict("records"), V0=args.V0,
                                  T_reh=args.T_reh, g_star=args.g_star)
        df2 = pd.DataFrame(rows)
        front = [c for c in ("eps", "kappa", "s0", "m_S", "regime",
                             "T_n_0_to_S", "T_n_S_to_v", "delta_T_n_S", "T_n_0_to_v",
                             "N_e_inflate")
                 if c in df2.columns]
        df2 = df2[front + [c for c in df2.columns if c not in front]]
        df2.to_csv(csv_path, index=False)
        with open(os.path.join(args.out, "eps_scan.json"), "w") as fh:
            json.dump(rows, fh, indent=2, default=float)
        m_S = float(df2["m_S"].dropna().iloc[0]) if "m_S" in df2.columns else args.m_S
        V0 = args.V0
        plot_scan(rows, args.out, V0, m_S)
        print(f"replot Sec.IV GW → {args.out}  (T_reh={args.T_reh:g} GeV, V0={V0:.3e})")
        cols = [c for c in ("eps", "kappa", "regime", "N_e_inflate",
                            "h2Omega_peak_step1", "h2Omega_peak_step2",
                            "h2Omega_peak_tot", "h2Omega_peak_0v",
                            "meets_LISA", "meets_DECIGO", "meets_BBO")
                if c in df2.columns]
        print(df2[cols].to_string(index=False))
        return

    if args.g_phi is not None and args.boson_phi is None:
        args.boson_phi = [f"20,{args.g_phi}"]
        if args.fermion_phi is None:
            args.fermion_phi = [f"20,{args.g_phi}"]
    if args.g_S is not None and args.boson_S is None:
        args.boson_S = [f"6,{args.g_S}"]
    args.boson_phi = _parse_ng(args.boson_phi, SET8_BOSON_PHI)
    args.fermion_phi = _parse_ng(args.fermion_phi, SET8_FERMION_PHI)
    args.boson_S = _parse_ng(args.boson_S, [(6.0, 1.0)])
    kappas = args.kappa if args.kappa else [0.5, 2.0]
    epss = np.geomspace(args.eps_min, args.eps_max, args.n_eps)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "progress.log")
    _LOG = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)
    log(f"portalEpsScan start → {args.out}")
    log(f"tree scale (set8-like): V0={args.V0:.4e}  v={args.v:.4e}")
    log(f"phi thermal = {args.phi_thermal}"
        + (" (V_correct Debye+mb2_0+CW)" if args.phi_thermal == "set8"
           else " (m = g_phi |phi|)"))
    if args.s0 is not None:
        log(f"LEGACY mode: fixed s0={args.s0:g} (m_S varies with eps)")
    else:
        log(f"fixed m_S={args.m_S:g} GeV  →  s0(eps)=sqrt(2 eps V0)/m_S")
        s0_lo = s0_from_mS(args.eps_min, args.V0, args.m_S)
        s0_hi = s0_from_mS(args.eps_max, args.V0, args.m_S)
        log(f"eps in [{args.eps_min:.3e}, {args.eps_max:.3e}]  "
            f"⇒ s0 in [{s0_lo:.4e}, {s0_hi:.4e}] GeV")
    log(f"kappa={kappas}  n_eps={args.n_eps}  workers={args.workers}  "
        f"nT={args.nT}  refine={args.refine}")
    log(f"a->v fine trigger: n={args.a2v_n_fine}  "
        f"T in [{args.a2v_T_lo_fac:g}, {args.a2v_T_hi_fac:g}]*T0_phi  "
        f"(+ widen if S3/T < {args.a2v_S_trigger:g})")
    log(f"bath phi: bosons={args.boson_phi} fermions={args.fermion_phi}")
    log(f"bath S:   bosons={args.boson_S}")
    log("observability: meets_DET if Omega_GW(f) >= Omega_sens(f) anywhere in DET band")
    log(f"live log file: {log_path}")

    rows = []
    jobs = [(float(k), float(e)) for k in kappas for e in epss]
    for i, (kappa, eps) in enumerate(jobs, 1):
        try:
            if args.s0 is not None:
                row = run_one_fixed_s0(eps, kappa, args.s0, args, i, len(jobs))
            else:
                row = run_one(eps, kappa, args, i, len(jobs))
            rows.append(row)
            # incremental CSV so progress is visible on disk
            import pandas as pd
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "eps_scan.csv"), index=False)
        except Exception as exc:
            log(f"FAILED kappa={kappa} eps={eps:.3e}: {exc}")
            rows.append({"eps": eps, "kappa": kappa, "s0": float("nan"),
                         "sequence": f"ERROR: {exc}", "regime": "error",
                         "T_n_0_to_S": float("nan"), "T_n_S_to_v": float("nan"),
                         "T_n_0_to_v": float("nan"), "delta_T_n_S": float("nan"),
                         "T_n_S": float("nan"), "T_n_phi": float("nan"),
                         "T_n_a2v": float("nan"), "f_peak_tot [Hz]": float("nan"),
                         "h2Omega_peak_tot": float("nan"),
                         "h2Omega_peak_0v": float("nan"),
                         "f_peak_0v [Hz]": float("nan"),
                         "meets_LISA": False, "meets_DECIGO": False, "meets_BBO": False,
                         "max_ratio_LISA": float("nan"), "max_ratio_DECIGO": float("nan"),
                         "max_ratio_BBO": float("nan")})

    import pandas as pd
    df = pd.DataFrame(rows)
    # put the nucleation temps near the front for readability
    front = [c for c in ("eps", "kappa", "s0", "m_S", "regime",
                         "T_n_0_to_S", "T_n_S_to_v", "delta_T_n_S", "T_n_0_to_v")
             if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(os.path.join(args.out, "eps_scan.csv"), index=False)
    with open(os.path.join(args.out, "eps_scan.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    plot_scan(rows, args.out, args.V0, args.m_S if args.s0 is None else float("nan"))
    log(f"Wrote {args.out}")
    cols = [c for c in ("eps", "kappa", "s0", "m_S", "regime",
                        "T_n_0_to_S", "T_n_S_to_v", "delta_T_n_S", "T_n_0_to_v",
                        "h2Omega_peak_tot",
                        "meets_LISA", "meets_DECIGO", "meets_BBO",
                        "max_ratio_LISA", "max_ratio_DECIGO", "max_ratio_BBO")
            if c in df.columns]
    log("\n" + df[cols].to_string(index=False))
    _LOG.close()


def run_one_fixed_s0(eps, kappa, s0, args, i_pt, n_pt):
    """Legacy path: fixed s0, m_S = sqrt(2 eps V0)/s0 varies."""
    # temporarily set args.m_S to the implied value and use s0_from by overriding
    old = args.m_S
    args.m_S = math.sqrt(2.0 * eps * args.V0) / s0
    # force s0 by calling build path differently
    pot = build_pot(args.V0, args.v, s0, eps, kappa,
                    args.boson_phi, args.fermion_phi, args.boson_S, args.g_star,
                    phi_thermal=getattr(args, "phi_thermal", "paper"), gS_bath=getattr(args, "gS_bath", 0.0), S_thermal=getattr(args, "S_thermal", "plain"))
    # reuse run_one logic by setting m_S so s0_from matches
    args.m_S = math.sqrt(pot.m_S2)
    row = run_one(eps, kappa, args, i_pt, n_pt)
    args.m_S = old
    return row


if __name__ == "__main__":
    main()
