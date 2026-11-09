#!/usr/bin/env python3
"""Double-peak SGWB from a two-step (0 -> a -> v) exit of thermal inflation.

Separated sequential regime — paper Sec. II / envelope (Eqs. 11–16)
-------------------------------------------------------------------
Each step i is a scalar-driven envelope source in the weak-friction limit
(v_w → 1):

    Ω_φ,0^(i)(f) h² = 1.67e-5 (H_*/β_i)² (κ_φ,i ΔV_i / ρ_tot,*)² (100/g_*,d)^{1/3}
                      × 0.11 v_w³/(0.42+v_w²) × D_inf,i × R_md × S_φ(f/f0)

with ρ_tot,* ≃ V0, R_md = (ρ_rad(T_reh)/V0)^{1/3}, D_inf,1 = e^{-4 ΔN_e},
D_inf,2 = 1, ΔN_e = ln[T_n(0→S)/T_n(S→v)].  Step-1 is suppressed by ε² and
by D_inf,1; stuck in A ⇒ D_inf,1 → 0.

Nested / proximate regime — paper Sec. III.C (Eqs. 25–35)
---------------------------------------------------------
When ΔT_n = T_n(0→S) − T_n(S→v) ≤ 0.05 TeV, use the multi-boundary spectrum:

    χ_nest ∝ (β1/H) (ΔT_n / T_n,1),   f_comp = exp(−χ_nest³),

    Ω_nested h² ≃ (1−f_comp)² [Ω₁ + Ω₂] + f_comp² Ω_comp
                  + 2 f_comp(1−f_comp) Ω_cross,

with Ω_comp carrying full V0 on the parent scale β1/H (κ_comp from Eq. 29)
and Ω_cross peaking at f_comp,0 / χ_nest.  D_inf,1 → 1 in this regime.

Examples
--------
    python analysis/twoStepGW.py --V0 1e28 --T_reh 100 \\
        --step 0.2,1200 --step 0.8,250 --reference 1000 \\
        --out data/twoStep/gw_secIV
    python analysis/twoStepGW.py --from_json data/twoStep/toy/nucleation.json \\
        --T_reh 100 --out data/twoStep/toy
    python analysis/twoStepGW.py --nested_check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

M_PL = 2.435e18
T_CMB_GEV = 2.7255 * 8.6173e-14
G_S0 = 3.91
GEV_TO_HZ = 1.0 / 6.5822e-25
# Paper Sec. IV default reheating temperature (0.1 TeV)
DEFAULT_T_REH = 100.0
# Paper Sec. III.A proximate / nested onset criterion (0.05 TeV)
NESTED_DTN_MAX_GEV = 50.0
# Default wall velocities for catch-up kinematics (v_w,2 > v_w,1 required)
DEFAULT_VW_OUTER = 0.6
DEFAULT_VW_INNER = 1.0
DEFAULT_ETA_SH = 0.5
DEFAULT_KAPPA_V1 = 0.0


@dataclass
class Step:
    label: str
    delV: float  # latent heat released in this step [GeV^4]
    T_n: float  # nucleation temperature [GeV] (for ΔN_e only)
    beta_H: float  # β/H_* with H_*^2 ≃ V0/(3 M_Pl^2)
    V0: float  # total false-vacuum energy [GeV^4]; ρ_tot,* ≃ V0
    T_reh: float = DEFAULT_T_REH  # flaton decay / reheating T [GeV]
    v_w: float = 1.0
    kappa: float = 1.0  # κ_φ: wall-gradient fraction
    g_star: float = 100.0
    g_s: float | None = None

    # Back-compat alias used by older callers that passed T_d ≡ T_n
    @property
    def T_d(self) -> float:
        return self.T_n


def rho_rad(T, g_star):
    return math.pi**2 / 30.0 * g_star * T**4


def step_quantities(s: Step, hstar: str = "paper") -> dict:
    """Present-day peak (undiluted by inter-step D_inf) for one step.

    Sec. IV: amplitude ∝ (κ ΔV/V0)² R_md(V0, T_reh).  ``hstar`` is kept for
    CLI compatibility; H_* always follows V0 (paper convention).
    """
    del hstar  # H_* is fixed to V0 in Sec. IV
    if not (s.V0 > 0 and s.delV > 0 and s.T_reh > 0 and s.beta_H > 0):
        raise ValueError(f"invalid Step for GW: {s}")
    rr = rho_rad(s.T_reh, s.g_star)
    # latent-heat fraction vs total vacuum (NOT vs radiation)
    frac = s.delV / s.V0
    # α_* = ΔV / ρ_rad(T_reh) — diagnostic only
    alpha = s.delV / rr
    R_md = min(1.0, (rr / s.V0) ** (1.0 / 3.0))
    H = math.sqrt(s.V0 / 3.0) / M_PL
    g_s = s.g_s or s.g_star
    f_H0 = H * GEV_TO_HZ * R_md * (G_S0 / g_s) ** (1.0 / 3.0) * T_CMB_GEV / s.T_reh
    f0 = 0.62 / (1.8 - 0.1 * s.v_w + s.v_w**2) * s.beta_H * f_H0
    # Eq. (11): (κ ΔV/ρ_tot)² with ρ_tot ≃ V0 → (κ frac)²
    amp = (1.67e-5 / s.beta_H**2 * (s.kappa * frac) ** 2
           * (100.0 / s.g_star) ** (1.0 / 3.0)
           * 0.11 * s.v_w**3 / (0.42 + s.v_w**2) * R_md)
    return {
        "label": s.label,
        "alpha": alpha,
        "frac_delV_V0": frac,
        "R_md": R_md,
        "H*": H,
        "T_reh": s.T_reh,
        "f_H0 [Hz]": f_H0,
        "f_peak [Hz]": f0,
        "h2Omega_peak": amp,
        "N_e_inflate": 0.0,
        "f_peak_undiluted [Hz]": f0,
        "h2Omega_peak_undiluted": amp,
        **asdict(s),
    }


def inter_step_Ne(T1, T2) -> float:
    """ΔN_e = ln[T_n(0→S)/T_n(S→v)] of vacuum-dominated expansion between steps."""
    if not (math.isfinite(T1) and math.isfinite(T2) and T1 > T2 > 0):
        return 0.0
    return math.log(T1 / T2)


def dilute_by_Ne(q: dict, N_e: float) -> dict:
    """Apply D_inf: f → f e^{-N}, Ω → Ω e^{-4N} (Sec. IV Eqs. 12, 14)."""
    out = dict(q)
    out.setdefault("f_peak_undiluted [Hz]", out["f_peak [Hz]"])
    out.setdefault("h2Omega_peak_undiluted", out["h2Omega_peak"])
    if math.isinf(N_e) and N_e > 0:
        out["N_e_inflate"] = float("inf")
        out["f_peak [Hz]"] = out["f_peak_undiluted [Hz]"]
        out["h2Omega_peak"] = 0.0
        out["D_inf"] = 0.0
        return out
    if not math.isfinite(N_e) or N_e <= 0:
        out["N_e_inflate"] = 0.0
        out["f_peak [Hz]"] = out["f_peak_undiluted [Hz]"]
        out["h2Omega_peak"] = out["h2Omega_peak_undiluted"]
        out["D_inf"] = 1.0
        return out
    fac = math.exp(-N_e)
    out["N_e_inflate"] = N_e
    out["D_inf"] = fac**4
    out["f_peak [Hz]"] = out["f_peak_undiluted [Hz]"] * fac
    out["h2Omega_peak"] = out["h2Omega_peak_undiluted"] * fac**4
    return out


def apply_inter_step_dilution(qs: list[dict], stuck: bool = False) -> list[dict]:
    """Dilute step 1 by D_inf (or remove it if stuck in A / eternal inflation)."""
    if not qs:
        return qs
    out = [dict(q) for q in qs]
    T1 = out[0].get("T_n", out[0].get("T_d", float("nan")))
    if len(out) >= 2:
        T2 = out[1].get("T_n", out[1].get("T_d", float("nan")))
        Ne = inter_step_Ne(T1, T2)
        out[0] = dilute_by_Ne(out[0], Ne)
        out[1] = dilute_by_Ne(out[1], 0.0)  # D_inf,2 = 1
        return out
    if stuck:
        out[0] = dilute_by_Ne(out[0], float("inf"))
    return out


def shape(x):
    return 3.8 * x**2.8 / (1.0 + 2.8 * x**3.8)


def spectrum(f, q: dict):
    return q["h2Omega_peak"] * shape(np.asarray(f) / q["f_peak [Hz]"])


def make_step(label, delV, T_n, beta_H, V0, T_reh=DEFAULT_T_REH,
              v_w=1.0, kappa=1.0, g_star=100.0) -> Step:
    """Helper so callers need not remember field order after the Sec. IV update."""
    return Step(label, float(delV), float(T_n), float(beta_H), float(V0),
                float(T_reh), float(v_w), float(kappa), float(g_star))


# --------------------------------------------------------------------------
# Paper Sec. III.C — nested / composite-wall multi-boundary spectrum
# --------------------------------------------------------------------------
def chi_nest(beta1_H: float, dTn: float, Tn1: float,
             vw1: float = DEFAULT_VW_OUTER,
             vw2: float = DEFAULT_VW_INNER) -> float:
    """Dimensionless catch-up parameter χ_nest (Eq. 25).

    χ_nest ≪ 1 ⇒ inner walls overtake parents before outer collisions (f_comp∼1).
    χ_nest ≫ 1 ⇒ sequential envelope limit (f_comp→0).
    """
    if not (math.isfinite(beta1_H) and beta1_H > 0
            and math.isfinite(Tn1) and Tn1 > 0
            and math.isfinite(dTn) and dTn >= 0
            and math.isfinite(vw1) and math.isfinite(vw2) and vw2 > vw1 > 0):
        return float("inf")
    pref = ((4.0 * vw2 - 3.0 * vw1)
            / (4.0 * (8.0 * math.pi) ** (1.0 / 3.0) * (vw2 - vw1)))
    return pref * beta1_H * (dTn / Tn1)


def f_comp_from_chi(chi: float) -> float:
    """Composite-boundary fraction f_comp = exp(−χ_nest³) (Eq. 26)."""
    if not math.isfinite(chi) or chi < 0:
        return 0.0
    x3 = chi ** 3
    if x3 > 700.0:  # underflow guard
        return 0.0
    return math.exp(-x3)


def beta_eff_over_H(f_comp: float, beta1_H: float, beta2_H: float) -> float:
    """Effective inverse duration β_eff/H_* (Eq. 26)."""
    if not (math.isfinite(f_comp) and 0.0 <= f_comp <= 1.0):
        f_comp = 0.0
    terms = []
    if f_comp > 0 and math.isfinite(beta1_H) and beta1_H > 0:
        terms.append(f_comp / beta1_H ** 2)
    if (1.0 - f_comp) > 0 and math.isfinite(beta2_H) and beta2_H > 0:
        terms.append((1.0 - f_comp) / beta2_H ** 2)
    if not terms:
        return float("nan")
    return 1.0 / math.sqrt(sum(terms))


def kappa_comp(eps: float, kappa_phi1: float = 1.0, kappa_phi2: float = 1.0,
               kappa_v1: float = DEFAULT_KAPPA_V1,
               eta_sh: float = DEFAULT_ETA_SH) -> float:
    """Composite-wall conversion efficiency κ_comp (Eq. 29)."""
    eps = float(np.clip(eps, 0.0, 1.0))
    return (kappa_phi2 * (1.0 - eps)
            + (kappa_phi1 + eta_sh * kappa_v1) * eps)


def _envelope_amp(beta_H: float, stress: float, V0: float, T_reh: float,
                  g_star: float = 100.0, v_w: float = 1.0) -> float:
    """Envelope peak amp with arbitrary stress factor replacing (κ ΔV/V0)²."""
    if not (beta_H > 0 and stress >= 0 and V0 > 0 and T_reh > 0):
        return 0.0
    rr = rho_rad(T_reh, g_star)
    R_md = min(1.0, (rr / V0) ** (1.0 / 3.0))
    return (1.67e-5 / beta_H ** 2 * stress
            * (100.0 / g_star) ** (1.0 / 3.0)
            * 0.11 * v_w ** 3 / (0.42 + v_w ** 2) * R_md)


def _envelope_fpeak(beta_H: float, V0: float, T_reh: float,
                    g_star: float = 100.0, v_w: float = 1.0) -> float:
    if not (beta_H > 0 and V0 > 0 and T_reh > 0):
        return float("nan")
    rr = rho_rad(T_reh, g_star)
    R_md = min(1.0, (rr / V0) ** (1.0 / 3.0))
    H = math.sqrt(V0 / 3.0) / M_PL
    f_H0 = H * GEV_TO_HZ * R_md * (G_S0 / g_star) ** (1.0 / 3.0) * T_CMB_GEV / T_reh
    return 0.62 / (1.8 - 0.1 * v_w + v_w ** 2) * beta_H * f_H0


def _scale_amp(q: dict, weight: float) -> dict:
    out = dict(q)
    w = float(weight) if math.isfinite(weight) and weight > 0 else 0.0
    out["spectrum_weight"] = w
    out["h2Omega_peak"] = out.get("h2Omega_peak_undiluted", out["h2Omega_peak"]) * w
    out["f_peak [Hz]"] = out.get("f_peak_undiluted [Hz]", out["f_peak [Hz]"])
    out["D_inf"] = 1.0
    out["N_e_inflate"] = 0.0
    return out


def is_nested_regime(dTn: float, Tn1: float | None = None,
                     dTn_max: float = NESTED_DTN_MAX_GEV,
                     force_nested: bool = False) -> bool:
    """True if proximate-temperature nested criterion applies (ΔT_n ≤ 0.05 TeV)."""
    if force_nested:
        return True
    if not (math.isfinite(dTn) and dTn >= 0 and dTn <= dTn_max):
        return False
    if Tn1 is not None and math.isfinite(Tn1) and Tn1 > 0 and dTn > Tn1:
        return False
    return True


def nested_amplification(kappa_c: float, kappa_phi2: float, eps: float,
                         beta1_H: float, beta2_H: float) -> float:
    """Ω_comp^peak / Ω₂^peak amplification factor (Eq. 35)."""
    den = kappa_phi2 * max(1.0 - eps, 1e-30)
    if not (den > 0 and math.isfinite(beta1_H) and beta1_H > 0
            and math.isfinite(beta2_H) and beta2_H > 0):
        return float("nan")
    return (kappa_c / den) ** 2 * (beta2_H / beta1_H) ** 2


def build_nested_spectrum(
    q1: dict,
    q2: dict,
    eps: float,
    dTn: float | None = None,
    vw1: float = DEFAULT_VW_OUTER,
    vw2: float = DEFAULT_VW_INNER,
    kappa_phi1: float = 1.0,
    kappa_phi2: float = 1.0,
    kappa_v1: float = DEFAULT_KAPPA_V1,
    eta_sh: float = DEFAULT_ETA_SH,
    force_nested: bool = False,
) -> tuple[list[dict], dict]:
    """Build Sec. III.C multi-boundary components from undiluted step peaks.

    Returns
    -------
    qs : weighted components [step1_unmerged, step2_unmerged, composite, cross]
         (entries with vanishing weight are omitted)
    meta : diagnostics (χ_nest, f_comp, κ_comp, β_eff/H, amp factor, …)
    """
    beta1 = float(q1.get("beta_H", float("nan")))
    beta2 = float(q2.get("beta_H", float("nan")))
    Tn1 = float(q1.get("T_n", q1.get("T_d", float("nan"))))
    Tn2 = float(q2.get("T_n", q2.get("T_d", float("nan"))))
    V0 = float(q1.get("V0", q2.get("V0", float("nan"))))
    T_reh = float(q1.get("T_reh", DEFAULT_T_REH))
    g_star = float(q1.get("g_star", 100.0))
    if dTn is None:
        dTn = (Tn1 - Tn2) if (math.isfinite(Tn1) and math.isfinite(Tn2)) else float("nan")
    dTn = float(dTn)

    nested = is_nested_regime(dTn, Tn1, force_nested=force_nested)
    # Immediate / Regime-II with no usable ΔT_n → treat as fully composite
    if force_nested and not (math.isfinite(dTn) and dTn >= 0):
        dTn = 0.0
        nested = True

    meta = {
        "gw_formula": "Sec.III.C nested" if nested else "Sec.IV sequential",
        "delta_T_n": dTn,
        "nested": bool(nested),
        "chi_nest": float("nan"),
        "f_comp": 0.0,
        "beta_eff_H": float("nan"),
        "kappa_comp": float("nan"),
        "nested_amp_factor": float("nan"),
        "vw1": vw1,
        "vw2": vw2,
    }
    if not nested:
        # Caller should use sequential dilution; return empty to signal passthrough
        return [], meta

    chi = chi_nest(beta1, max(dTn, 0.0), Tn1, vw1=vw1, vw2=vw2)
    fc = f_comp_from_chi(chi)
    kcomp = kappa_comp(eps, kappa_phi1, kappa_phi2, kappa_v1, eta_sh)
    beff = beta_eff_over_H(fc, beta1, beta2)
    amp_fac = nested_amplification(kcomp, kappa_phi2, eps, beta1, beta2)
    meta.update({
        "chi_nest": chi,
        "f_comp": fc,
        "beta_eff_H": beff,
        "kappa_comp": kcomp,
        "nested_amp_factor": amp_fac,
    })

    qs: list[dict] = []
    w_u = (1.0 - fc) ** 2
    if w_u > 0:
        qs.append(_scale_amp({**q1, "label": "step1"}, w_u))
        qs.append(_scale_amp({**q2, "label": "step2"}, w_u))

    w_c = fc ** 2
    if w_c > 0 and math.isfinite(beta1) and beta1 > 0 and V0 > 0:
        f_comp0 = _envelope_fpeak(beta1, V0, T_reh, g_star=g_star, v_w=vw2)
        amp_c = _envelope_amp(beta1, kcomp ** 2, V0, T_reh, g_star=g_star, v_w=vw2)
        qs.append({
            "label": "composite",
            "frac_delV_V0": 1.0,
            "R_md": min(1.0, (rho_rad(T_reh, g_star) / V0) ** (1.0 / 3.0)),
            "T_reh": T_reh,
            "T_n": Tn1,
            "beta_H": beta1,
            "V0": V0,
            "v_w": vw2,
            "kappa": kcomp,
            "g_star": g_star,
            "f_peak [Hz]": f_comp0,
            "f_peak_undiluted [Hz]": f_comp0,
            "h2Omega_peak": amp_c * w_c,
            "h2Omega_peak_undiluted": amp_c,
            "spectrum_weight": w_c,
            "D_inf": 1.0,
            "N_e_inflate": 0.0,
        })

    w_x = 2.0 * fc * (1.0 - fc)
    if (w_x > 0 and math.isfinite(chi) and chi > 0
            and math.isfinite(beta1) and beta1 > 0 and V0 > 0):
        stress_x = kappa_phi1 * kappa_phi2 * eps * (1.0 - eps) * chi ** 2
        f_comp0 = _envelope_fpeak(beta1, V0, T_reh, g_star=g_star, v_w=vw2)
        f_cross = f_comp0 / chi  # Eq. (34)
        amp_x = _envelope_amp(beta1, stress_x, V0, T_reh, g_star=g_star, v_w=vw2)
        qs.append({
            "label": "cross",
            "frac_delV_V0": math.sqrt(max(eps * (1.0 - eps), 0.0)),
            "R_md": min(1.0, (rho_rad(T_reh, g_star) / V0) ** (1.0 / 3.0)),
            "T_reh": T_reh,
            "T_n": Tn1,
            "beta_H": beta1,
            "V0": V0,
            "v_w": vw2,
            "kappa": math.sqrt(max(stress_x, 0.0)),
            "g_star": g_star,
            "f_peak [Hz]": f_cross,
            "f_peak_undiluted [Hz]": f_cross,
            "h2Omega_peak": amp_x * w_x,
            "h2Omega_peak_undiluted": amp_x,
            "spectrum_weight": w_x,
            "D_inf": 1.0,
            "N_e_inflate": 0.0,
            "chi_nest": chi,
        })
    return qs, meta


def apply_two_step_gw(
    qs_raw: list[dict],
    eps: float,
    stuck: bool = False,
    dTn: float | None = None,
    force_nested: bool = False,
    **nested_kw,
) -> tuple[list[dict], dict]:
    """Choose Sec. IV sequential dilution or Sec. III.C nested spectrum.

    ``qs_raw`` must contain undiluted step_quantities for step1 and (optionally)
    step2, in that order.
    """
    meta = {"gw_formula": "Sec.IV sequential", "nested": False, "f_comp": 0.0}
    if stuck or len(qs_raw) < 2:
        return apply_inter_step_dilution(qs_raw, stuck=stuck), meta

    q1, q2 = qs_raw[0], qs_raw[1]
    Tn1 = float(q1.get("T_n", float("nan")))
    Tn2 = float(q2.get("T_n", float("nan")))
    if dTn is None and math.isfinite(Tn1) and math.isfinite(Tn2):
        dTn = Tn1 - Tn2

    nested_qs, nested_meta = build_nested_spectrum(
        q1, q2, eps=eps, dTn=dTn, force_nested=force_nested, **nested_kw)
    if nested_meta.get("nested") and nested_qs:
        return nested_qs, nested_meta
    return apply_inter_step_dilution(qs_raw, stuck=False), meta


# --------------------------------------------------------------------------
# Paper Sec. IV reference numbers (single-step + ε-scaling)
# --------------------------------------------------------------------------
def secIV_check(V0=1e28, T_reh=100.0, g_star=100.0):
    """Reproduce Eq. (11) single-step peak ≈ 8.93e-19 at paper fiducials."""
    q = step_quantities(make_step("0->v", V0, T_reh, 1000.0, V0, T_reh, g_star=g_star))
    print(f"Sec.IV single-step: f_peak={q['f_peak [Hz]']:.4g} Hz (paper ~4.62), "
          f"h2Omega={q['h2Omega_peak']:.4g} (paper 8.93e-19)")
    for eps in (1e-2, 1e-4, 1e-20):
        q1 = step_quantities(make_step("step1", eps * V0, T_reh, 1000.0, V0, T_reh,
                                       g_star=g_star))
        ratio = q1["h2Omega_peak"] / q["h2Omega_peak"]
        print(f"  eps={eps:.0e}: amp_1/amp_0v = {ratio:.4g}  (expect eps^2={eps**2:.4g})")
    return q


def nested_check(V0=1e28, T_reh=100.0, g_star=100.0):
    """Sanity: f_comp→0 (separated) and f_comp→1 (fully nested) limits."""
    eps, b1, b2 = 1e-4, 500.0, 2000.0
    q1 = step_quantities(make_step("step1", eps * V0, 1000.0, b1, V0, T_reh, g_star=g_star))
    q2 = step_quantities(make_step("step2", (1 - eps) * V0, 999.0, b2, V0, T_reh, g_star=g_star))
    # Fully nested: ΔT_n = 0
    qs0, m0 = build_nested_spectrum(q1, q2, eps=eps, dTn=0.0, force_nested=True)
    # Separated within nested window but χ≫1
    qs_sep, m_sep = build_nested_spectrum(
        q1, {**q2, "T_n": 950.0}, eps=eps, dTn=50.0, force_nested=True)
    amp0 = next(q["h2Omega_peak"] for q in qs0 if q["label"] == "composite")
    q2_only = step_quantities(make_step("step2", (1 - eps) * V0, 999.0, b2, V0, T_reh,
                                        g_star=g_star))
    print(f"nested ΔT=0: f_comp={m0['f_comp']:.4g}  κ_comp={m0['kappa_comp']:.4g}  "
          f"amp_comp={amp0:.4g}  amp_factor={m0['nested_amp_factor']:.4g}  "
          f"(expect ~(β2/β1)^2={(b2/b1)**2:.4g})")
    print(f"nested ΔT=50 GeV: f_comp={m_sep['f_comp']:.4g}  χ={m_sep['chi_nest']:.4g}  "
          f"n_components={len(qs_sep)}")
    print(f"step2-only (no nest): amp={q2_only['h2Omega_peak']:.4g}")
    return m0, m_sep


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot(qs: list[dict], ref: dict | None, path: str, title: str = "", detectors: bool = True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.geomspace(1e-5, 1e5, 2500)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    if detectors:
        try:
            from analysis.published_plisc import add_detector_curves

            add_detector_curves(ax)
        except Exception as exc:
            print(f"detector curves skipped: {exc}")
    tot = np.zeros_like(f)
    for q, col in zip(qs, ("C0", "C3", "C1", "C4")):
        if not (q.get("h2Omega_peak", 0) > 0 and q.get("f_peak [Hz]", 0) > 0):
            continue
        s = spectrum(f, q)
        tot += s
        Ne = q.get("N_e_inflate", 0.0)
        ne_tag = (f", $D_{{\\rm inf}}=e^{{-4N_e}}$, $N_e$={Ne:.3g}"
                  if (isinstance(Ne, (int, float)) and Ne > 0 and math.isfinite(Ne)) else "")
        ax.loglog(f, s, "--", color=col, lw=1.5, zorder=5,
                  label=f"{q['label']}: $f_p$={q['f_peak [Hz]']:.3g} Hz, "
                        f"$h^2\\Omega_p$={q['h2Omega_peak']:.3g}{ne_tag}")
    ax.loglog(f, tot, "k", lw=2.2, zorder=6, label="two-step total")
    if ref:
        ax.loglog(f, spectrum(f, ref), ":", color="C2", lw=2, zorder=5,
                  label=f"{ref['label']}: $f_p$={ref['f_peak [Hz]']:.3g} Hz, "
                        f"$h^2\\Omega_p$={ref['h2Omega_peak']:.3g}")
    ax.set_ylim(1e-25, 1e-5)
    ax.set_xlim(1e-5, 1e5)
    ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{\rm GW}$",
           title=title or r"Two-step SGWB (Sec.~II / III.C)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def steps_from_json(path, v_w, kappa, g_star, T_reh):
    meta = json.load(open(path))
    pot, nuc = meta["potential"], meta["nucleation"]
    V0, d1 = pot["V0"], pot["delV1"]
    steps, ref = [], None
    beta1 = nuc.get("step1", {}).get("beta/H", np.nan)
    for key, label, dV in (("step1", "step 1 (0->a)", d1),
                           ("step2", "step 2 (a->v)", V0 - d1)):
        n = nuc.get(key, {})
        T_n, bH = n.get("T_n", np.nan), n.get("beta/H", np.nan)
        if key == "step2" and np.isfinite(T_n) and "immediate" in n.get("note", "") \
                and not (np.isfinite(bH) and bH > 0):
            bH = beta1
            label += " [nested, beta from step 1]"
        if not (np.isfinite(T_n) and np.isfinite(bH) and bH > 0):
            print(f"{label}: no usable T_n / beta/H in {path} -> skipped "
                  f"({n.get('note', 'did not nucleate')})")
            continue
        steps.append(make_step(label, dV, T_n, bH, V0, T_reh, v_w, kappa, g_star))
    n = nuc.get("single-step reference", {})
    if np.isfinite(n.get("T_n", np.nan)) and np.isfinite(n.get("beta/H", np.nan)):
        ref = make_step("single-step 0->v", V0, n["T_n"], n["beta/H"], V0, T_reh,
                        v_w, kappa, g_star)
    return steps, ref


def main(argv=None):
    p = argparse.ArgumentParser(description="Two-step SGWB (paper Sec. II / III.C)")
    p.add_argument("--secIV_check", action="store_true",
                   help="print Sec. IV single-step / ε² scaling check and exit")
    p.add_argument("--nested_check", action="store_true",
                   help="print Sec. III.C nested limit checks and exit")
    p.add_argument("--from_json", help="nucleation.json from analysis/twoStepBounce.py")
    p.add_argument("--V0", type=float, default=1e28, help="total vacuum energy [GeV^4]")
    p.add_argument("--T_reh", type=float, default=DEFAULT_T_REH,
                   help="flaton decay / reheating temperature [GeV] (default 100 = 0.1 TeV)")
    p.add_argument("--step", action="append",
                   help="manual step: delV/V0,beta/H[,T_n GeV] (T_n default=T_reh; repeat)")
    p.add_argument("--reference", help="single-step reference: beta/H[,T_n]")
    p.add_argument("--v_w", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--g_star", type=float, default=100.0)
    p.add_argument("--hstar", choices=("paper", "false_vacuum"), default="paper")
    p.add_argument("--force_nested", action="store_true",
                   help="force Sec. III.C nested spectrum (ignore ΔT_n cut)")
    p.add_argument("--vw_outer", type=float, default=DEFAULT_VW_OUTER)
    p.add_argument("--vw_inner", type=float, default=DEFAULT_VW_INNER)
    p.add_argument("--no_detectors", action="store_true")
    p.add_argument("--out", help="output directory (gw_spectrum.png, gw_summary.json)")
    a = p.parse_args(argv)

    if a.secIV_check:
        secIV_check(a.V0, a.T_reh, a.g_star)
        return
    if a.nested_check:
        nested_check(a.V0, a.T_reh, a.g_star)
        return
    if a.from_json:
        steps, ref = steps_from_json(a.from_json, a.v_w, a.kappa, a.g_star, a.T_reh)
    else:
        steps = []
        used = 0.0
        for i, s in enumerate(a.step or [], 1):
            parts = [float(x) for x in s.split(",")]
            if len(parts) == 2:
                fr, bH = parts
                T_n = a.T_reh
            elif len(parts) == 3:
                fr, bH, T_n = parts
            else:
                p.error("--step expects delV/V0,beta/H or delV/V0,beta/H,T_n")
            used += fr
            steps.append(make_step(f"step {i}", fr * a.V0, T_n, bH, a.V0, a.T_reh,
                                   a.v_w, a.kappa, a.g_star))
        ref = None
        if a.reference:
            parts = [float(x) for x in a.reference.split(",")]
            bH = parts[0]
            T_n = parts[1] if len(parts) > 1 else a.T_reh
            ref = make_step("single-step 0->v", a.V0, T_n, bH, a.V0, a.T_reh,
                            a.v_w, a.kappa, a.g_star)
    if not steps and ref is None:
        p.error("nothing to compute: give --from_json, --step, --secIV_check or --nested_check")
    qs_raw = [step_quantities(s, a.hstar) for s in steps]
    eps = 0.0
    if len(steps) >= 1 and a.V0 > 0:
        eps = float(steps[0].delV / a.V0)
    qs, meta = apply_two_step_gw(
        qs_raw, eps=eps, force_nested=a.force_nested,
        vw1=a.vw_outer, vw2=a.vw_inner,
        kappa_phi1=a.kappa, kappa_phi2=a.kappa)
    rq = step_quantities(ref, a.hstar) if ref else None
    if meta.get("nested"):
        print(f"GW formula: {meta['gw_formula']}  f_comp={meta['f_comp']:.4g}  "
              f"χ_nest={meta['chi_nest']:.4g}  κ_comp={meta['kappa_comp']:.4g}  "
              f"amp_factor={meta['nested_amp_factor']:.4g}")
    for q in qs + ([rq] if rq else []):
        Ne = q.get("N_e_inflate", 0.0)
        ne_s = f"  N_e={Ne:.4g}  D_inf={q.get('D_inf', 1):.3g}" if (
            isinstance(Ne, (int, float)) and Ne > 0) else ""
        w = q.get("spectrum_weight", None)
        w_s = f"  w={w:.3g}" if isinstance(w, (int, float)) else ""
        print(f"{q['label']:20s} delV/V0={q.get('frac_delV_V0', float('nan')):.3e}  "
              f"T_n={q.get('T_n', float('nan')):.4g} GeV  "
              f"beta/H={q.get('beta_H', float('nan')):.4g}  R_md={q.get('R_md', float('nan')):.3e}  "
              f"f_peak={q['f_peak [Hz]']:.4g} Hz  h2Omega_peak={q['h2Omega_peak']:.4g}"
              f"{ne_s}{w_s}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        ttl = (rf"Nested SGWB ($f_{{\rm comp}}$={meta.get('f_comp', 0):.3g})"
               if meta.get("nested") else "")
        plot(qs, rq, os.path.join(a.out, "gw_spectrum.png"),
             title=ttl, detectors=not a.no_detectors)
        with open(os.path.join(a.out, "gw_summary.json"), "w") as fh:
            json.dump({"formula": meta.get("gw_formula", "Sec.IV"),
                       "nested_meta": meta, "T_reh": a.T_reh, "V0": a.V0,
                       "steps": qs, "reference": rq}, fh, indent=2, default=float)
        print(f"wrote {a.out}/gw_spectrum.png, gw_summary.json")


if __name__ == "__main__":
    main()
