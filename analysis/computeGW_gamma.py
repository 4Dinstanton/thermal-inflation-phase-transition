#!/usr/bin/env python3
"""GW spectrum as a function of γ = φ₀ / M_Pl  with fixed m_φ = 1 TeV.

Physics:
  m is fixed  →  λ = m²/φ₀²  changes with γ
  S₃/T is invariant (depends on potential shape, not scale)
  V₀ = m² φ₀² / 4 = m² γ² M_Pl² / 4   scales as γ²
  T_n(γ) from  S₃/T = 4 ln(2√3 T/(mγ))

Uses the TI bubble-collision formula from gwSpectrum.py.
Thermal-integral and S₃/T infrastructure duplicated from computeTn.py.
"""

import math
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwSpectrum import (
    gw_thermal_inflation,
    gw_soundwave_thermal_inflation,
    gw_turbulence_thermal_inflation,
    kappa_v,
    sensitivity_LISA,
    sensitivity_DECIGO,
    sensitivity_BBO,
    sensitivity_ET,
    sensitivity_aLIGO,
    _annotate_detector,
    M_PL,
    G_STAR_DEFAULT,
)

# ═══════════════════════════════════════════════════════════════════
#  Physical parameters  (must match computeTn.py exactly)
# ═══════════════════════════════════════════════════════════════════
m = 1000.0  # flaton mass [GeV]
y = 1.09  # Yukawa coupling
g_c = 1.05  # gauge coupling
n_f = 20  # fermion DOF
delta2 = g_c**2 / 6.0  # δ² ≈ 0.1838

QX = np.array([0.0320224, 0.339406])
QW = np.array([0.900848, 0.0991516])

# ═══════════════════════════════════════════════════════════════════
#  Fermionic thermal integral J'_F(u²)  (spline-cached)
# ═══════════════════════════════════════════════════════════════════
print("Building J'_F spline … ", end="", flush=True)


def _jfp_integrand(x, u2):
    s = math.sqrt(u2 + x * x)
    if s > 500:
        return 0.0
    return x * x / (s * (math.exp(s) + 1.0))


def _jfp_exact(u2):
    val, _ = integrate.quad(
        _jfp_integrand,
        0,
        200,
        args=(u2,),
        limit=500,
        epsabs=1e-15,
        epsrel=1e-15,
    )
    return val / (4.0 * math.pi**2)


_U2_GRID = np.concatenate([np.linspace(0, 5, 3000), np.linspace(5.01, 60, 3000)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _JFP_VALS = np.array([_jfp_exact(u2) for u2 in _U2_GRID])
_SP = CubicSpline(_U2_GRID, _JFP_VALS)
print("done.")


def Jp(u2):
    if u2 > 60:
        return 0.0
    return float(_SP(u2))


def Jpp(u2):
    if u2 > 60:
        return 0.0
    return float(_SP(u2, 1))


# ═══════════════════════════════════════════════════════════════════
#  T_c2, α(T) solver, S₃/T  — replicated from computeTn.py
# ═══════════════════════════════════════════════════════════════════
T_c2 = m / (y * math.sqrt(n_f * Jp(delta2)))
print(f"T_c2 = {T_c2:.4f} GeV")


def _Gi(alpha, i):
    z = delta2 + QX[i] * alpha
    return Jp(z) - QX[i] * alpha / 2.0 * Jpp(z)


def _f_sum(alpha):
    return QW[0] * _Gi(alpha, 0) + QW[1] * _Gi(alpha, 1)


_F0 = n_f * _f_sum(0.0)


def solve_alpha(T):
    lhs = m**2 / (y**2 * T**2)
    if lhs >= _F0:
        return None

    def residual(a):
        return lhs - n_f * _f_sum(a)

    hi = 1.0
    while residual(hi) < 0 and hi < 500:
        hi *= 2
    if residual(hi) < 0:
        return None
    return optimize.brentq(residual, 0.0, hi, xtol=1e-12)


def S3_over_T(alpha):
    denom = sum(QW[i] * QX[i] * Jpp(delta2 + QX[i] * alpha) for i in range(2))
    arg = -2.0 * alpha / (3.0 * y**2 * n_f * denom)
    if arg <= 0:
        return None
    return 2.0 * math.pi / (3.0 * y**2) * math.sqrt(arg)


# ═══════════════════════════════════════════════════════════════════
#  Pre-compute S₃/T on a fine temperature grid
# ═══════════════════════════════════════════════════════════════════
_T_LO = T_c2 + 0.5
_T_HI = T_c2 + 1000.0
_N_T = 6000
_T_GRID = np.linspace(_T_LO, _T_HI, _N_T)
_A_GRID = np.full(_N_T, np.nan)
_S_GRID = np.full(_N_T, np.nan)

print(f"Pre-computing S₃/T on [{_T_LO:.1f}, {_T_HI:.1f}] GeV … ", end="", flush=True)
for k, T_val in enumerate(_T_GRID):
    a = solve_alpha(T_val)
    if a is not None:
        _A_GRID[k] = a
        s = S3_over_T(a)
        if s is not None:
            _S_GRID[k] = s
_valid = ~np.isnan(_S_GRID)
_S3T_spl = CubicSpline(_T_GRID[_valid], _S_GRID[_valid])
print(f"done ({np.sum(_valid)} valid).")


def find_Tn(gamma):
    T_v = _T_GRID[_valid]
    S_v = _S_GRID[_valid]
    rhs = 4.0 * np.log(2 * np.sqrt(3) * T_v / (m * gamma))
    diff = S_v - rhs
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return np.nan
    idx = sc[0]

    def _res(T):
        return float(_S3T_spl(T)) - 4.0 * math.log(2 * math.sqrt(3) * T / (m * gamma))

    return optimize.brentq(_res, T_v[idx], T_v[idx + 1], xtol=1e-3)


def V0_of_gamma(gamma):
    """V₀ = m² φ₀² / 4 = m² γ² M_Pl² / 4."""
    return 0.25 * m**2 * (gamma * M_PL) ** 2


def DV_wall_of(Tn):
    r"""Energy released at the bubble wall ≈ (m² − δ²T_n²) T_n² / 2."""
    return 0.5 * (m**2 - delta2 * Tn**2) * Tn**2


def P_BM_of(Tn):
    r"""Bödeker–Moore saturation pressure.  P_BM = n_f y² T⁴ / 48."""
    return n_f * y**2 * Tn**4 / 48.0


def terminal_vw(Tn):
    r"""Terminal wall velocity from simplified friction model."""
    DV = DV_wall_of(Tn)
    P = P_BM_of(Tn)
    if P <= 0 or DV <= 0:
        return 1.0
    r = DV / P
    if r >= 1.0:
        return 1.0
    u2 = r / (1.0 - r)
    uy = math.sqrt(u2) / y
    return uy / math.sqrt(1.0 + uy**2)


# ═══════════════════════════════════════════════════════════════════
#  GW spectra for γ = 10⁻², 10⁻³, 10⁻⁴
# ═══════════════════════════════════════════════════════════════════
GAMMA_LIST = [1e-2, 1e-3, 1e-4]
T_D_LIST = [100.0, 1000.0]
g_star = G_STAR_DEFAULT
freq = np.logspace(-5, 5, 3000)
GAMMA_CUT = 1e-4

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "gw_gamma")
os.makedirs(OUT, exist_ok=True)

EPS_TURB_LO = 0.05
EPS_TURB_HI = 0.10

# Pre-compute γ-dependent quantities (independent of T_d)
gamma_info = []
for gv in GAMMA_LIST:
    phi0 = gv * M_PL
    V0 = V0_of_gamma(gv)
    Tn = find_Tn(gv)
    if np.isnan(Tn):
        continue
    beta_H = float(Tn * _S3T_spl(Tn, 1))
    rho_rad = math.pi**2 / 30.0 * g_star * Tn**4
    alpha_gw = V0 / rho_rad
    H_TI = math.sqrt(V0 / (3.0 * M_PL**2))

    DV = DV_wall_of(Tn)
    vwt = terminal_vw(Tn)

    gamma_info.append(
        dict(
            gamma=gv, phi0=phi0, V0=V0, lam=m**2 / phi0**2,
            Tn=Tn, beta_H=beta_H, alpha=alpha_gw, H_TI=H_TI,
            DV_wall=DV, vw_terminal=vwt,
        )
    )

# Wider scan (γ-dependent, T_d-independent)
print("\nScanning γ ∈ [10⁻¹⁰, 10⁻¹] … ", end="", flush=True)
gamma_scan = np.logspace(-10, -1, 300)
scan_base = []
for gv in gamma_scan:
    Tn = find_Tn(gv)
    if np.isnan(Tn):
        continue
    V0 = V0_of_gamma(gv)
    beta_H = float(Tn * _S3T_spl(Tn, 1))
    DV = DV_wall_of(Tn)
    vwt = terminal_vw(Tn)
    scan_base.append((gv, Tn, beta_H, V0, DV, vwt))
scan_base = np.array(scan_base)
print(f"done ({len(scan_base)} points).")


# ═══════════════════════════════════════════════════════════════════
#  Loop over T_d values — generate all plots for each
# ═══════════════════════════════════════════════════════════════════
for T_D in T_D_LIST:
    td_tag = f"Td{int(T_D)}"
    print(f"\n{'═' * 80}")
    print(f"  T_d = {T_D:.0f} GeV")
    print(f"{'═' * 80}")

    # ──────────────────────────────────────────────────────────────
    #  A) Bubble collision — both spectral shapes
    # ──────────────────────────────────────────────────────────────
    SHAPE_DEFS = [
        ("envelope", "envelope",
         "Envelope (HK 2008)", "env"),
        ("jt2016", "jt2016",
         "JT 2016", "jt"),
    ]
    sig_coll = {}  # keyed by shape tag
    for sh, eff, label, tag in SHAPE_DEFS:
        sig_coll[tag] = []
        for gi in gamma_info:
            gv = gi["gamma"]
            V0 = gi["V0"]
            R_md = (math.pi**2 * g_star * T_D**4 / (30.0 * V0)) ** (1.0 / 3.0)
            omega, fpk, h2pk = gw_thermal_inflation(
                freq, gi["beta_H"], V0, T_D,
                g_star=g_star, v_w=1.0, kappa_phi=1.0,
                shape=sh, efficiency=eff,
            )
            sig_coll[tag].append({**gi, "T_D": T_D, "R_md": R_md,
                                  "omega": omega, "fpk": fpk, "h2pk": h2pk})

            exp = int(round(math.log10(gv)))
            print(
                f"  [{tag}] γ = 10^{exp}:  f_pk = {fpk:.2e} Hz,"
                f"  h²Ω = {h2pk:.2e}"
            )

    # ──────────────────────────────────────────────────────────────
    #  B) Turbulence (ε_turb × V₀ energy budget)
    # ──────────────────────────────────────────────────────────────
    sig_turb_lo = []
    sig_turb_hi = []
    for gi in gamma_info:
        gv = gi["gamma"]
        V0 = gi["V0"]
        for eps_t, sig_list in [
            (EPS_TURB_LO, sig_turb_lo),
            (EPS_TURB_HI, sig_turb_hi),
        ]:
            omega, fpk, h2pk = gw_turbulence_thermal_inflation(
                freq, gi["beta_H"], V0, T_D,
                eps_turb=eps_t, g_star=g_star, v_w=1.0,
            )
            sig_list.append({**gi, "T_D": T_D, "omega": omega,
                             "fpk": fpk, "h2pk": h2pk, "eps_turb": eps_t})

        exp = int(round(math.log10(gv)))
        print(
            f"  [Turbulence] γ = 10^{exp}:  "
            f"ε=5%: h²Ω = {sig_turb_lo[-1]['h2pk']:.2e},  "
            f"ε=10%: h²Ω = {sig_turb_hi[-1]['h2pk']:.2e}"
        )

    # ──────────────────────────────────────────────────────────────
    #  C) Sound wave (ΔV_wall energy budget) — kept for reference
    # ──────────────────────────────────────────────────────────────
    sig_sw = []
    for gi in gamma_info:
        gv = gi["gamma"]
        V0 = gi["V0"]
        vwt = gi["vw_terminal"]
        omega, fpk, h2pk = gw_soundwave_thermal_inflation(
            freq, gi["beta_H"], V0, gi["DV_wall"],
            gi["Tn"], T_D, g_star=g_star, v_w=vwt,
        )
        sig_sw.append({**gi, "T_D": T_D, "omega": omega,
                       "fpk": fpk, "h2pk": h2pk})

        exp = int(round(math.log10(gv)))
        alpha_w = gi["DV_wall"] / (math.pi**2 / 30.0 * g_star * gi["Tn"]**4)
        print(
            f"  [Sound wave] γ = 10^{exp}:  "
            f"α_wall = {alpha_w:.4f}, "
            f"h²Ω = {h2pk:.2e}"
        )

    # ──────────────────────────────────────────────────────────────
    #  Wider scan for peak values
    # ──────────────────────────────────────────────────────────────
    n_scan = len(scan_base)
    scan_fpk = {}
    scan_h2pk = {}
    for _, eff, _, tag in SHAPE_DEFS:
        scan_fpk[tag] = np.empty(n_scan)
        scan_h2pk[tag] = np.empty(n_scan)

    scan_h2pk_turb_lo = np.empty(n_scan)
    scan_h2pk_turb_hi = np.empty(n_scan)
    scan_h2pk_sw = np.empty(n_scan)
    for j in range(n_scan):
        gv_j, Tn_j, bh_j, v0_j, dv_j, vwt_j = scan_base[j]

        for sh, eff, _, tag in SHAPE_DEFS:
            _, fp, hp = gw_thermal_inflation(
                np.array([1.0]), bh_j, v0_j, T_D,
                g_star=g_star, v_w=1.0, kappa_phi=1.0,
                shape=sh, efficiency=eff,
            )
            scan_fpk[tag][j] = fp
            scan_h2pk[tag][j] = hp

        _, _, hp_tlo = gw_turbulence_thermal_inflation(
            np.array([1.0]), bh_j, v0_j, T_D,
            eps_turb=EPS_TURB_LO, g_star=g_star, v_w=1.0,
        )
        scan_h2pk_turb_lo[j] = hp_tlo

        _, _, hp_thi = gw_turbulence_thermal_inflation(
            np.array([1.0]), bh_j, v0_j, T_D,
            eps_turb=EPS_TURB_HI, g_star=g_star, v_w=1.0,
        )
        scan_h2pk_turb_hi[j] = hp_thi

        _, _, hp_sw = gw_soundwave_thermal_inflation(
            np.array([1.0]), bh_j, v0_j, dv_j,
            Tn_j, T_D, g_star=g_star, v_w=vwt_j,
        )
        scan_h2pk_sw[j] = hp_sw

    g_sc = scan_base[:, 0]
    Tn_sc = scan_base[:, 1]
    beta_sc = scan_base[:, 2]
    m_ok = g_sc >= GAMMA_CUT
    m_ex = g_sc < GAMMA_CUT

    # ─── Plot 1: Collision + Turbulence spectrum — one per shape ─
    f_lisa = np.logspace(-5, -0.5, 500)
    f_decigo = np.logspace(-3, 2, 500)
    f_bbo = np.logspace(-3, 2, 500)
    f_et = np.logspace(0.3, 3.5, 500)
    f_ligo = np.logspace(0.7, 3.7, 500)
    det_list = [
        (f_lisa, sensitivity_LISA, "purple", "LISA", 3e-3),
        (f_decigo, sensitivity_DECIGO, "orange", "DECIGO", 0.2),
        (f_bbo, sensitivity_BBO, "cyan", "BBO", 0.05),
        (f_et, sensitivity_ET, "brown", "ET", 5.0),
        (f_ligo, sensitivity_aLIGO, "green", "aLIGO", 50.0),
    ]
    SIG_COLORS = ["navy", "crimson", "forestgreen"]

    for _, _, shape_label, tag in SHAPE_DEFS:
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (sc, stlo, sthi) in enumerate(
            zip(sig_coll[tag], sig_turb_lo, sig_turb_hi)
        ):
            gv = sc["gamma"]
            exp = int(round(math.log10(gv)))
            c = SIG_COLORS[i]
            total_lo = sc["omega"] + stlo["omega"]
            total_hi = sc["omega"] + sthi["omega"]

            ax.loglog(freq, sc["omega"], color=c, lw=2.0, ls="-",
                      label=rf"$\gamma = 10^{{{exp}}}$ — collision",
                      zorder=5 + i)
            ax.loglog(freq, sthi["omega"], color=c, lw=1.2, ls=":",
                      zorder=4 + i)
            ax.fill_between(freq, total_lo, total_hi,
                            color=c, alpha=0.10, zorder=2)
            ax.loglog(freq, total_hi, color=c, lw=2.5, ls="--",
                      label=rf"$\gamma = 10^{{{exp}}}$ — coll+turb",
                      zorder=6 + i)

            ax.annotate(
                rf"$\gamma=10^{{{exp}}}$",
                xy=(sc["fpk"], sc["h2pk"]),
                xytext=(sc["fpk"] * 3, sc["h2pk"] * 5),
                fontsize=10, color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
                zorder=10,
            )

        for fv, sens_fn, col, name, ft in det_list:
            sv = sens_fn(fv)
            ax.loglog(fv, sv, color=col, lw=1.5, alpha=0.5)
            _annotate_detector(ax, fv, sv, name, col, ft)

        ax.set_xlabel(r"$f$ [Hz]", fontsize=14)
        ax.set_ylabel(r"$h^2 \Omega_{\mathrm{GW}}$", fontsize=14)
        ax.set_xlim(1e-5, 1e5)
        ax.set_ylim(1e-25, 1e-5)
        ax.legend(fontsize=9, loc="upper right", ncol=2)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(
            rf"GW Spectrum — {shape_label} + Turbulence"
            rf"  ($m = 1$ TeV, $T_d = {T_D:.0f}$ GeV)",
            fontsize=11,
        )
        fig.tight_layout()
        p1 = f"{OUT}/gw_spectrum_{tag}_{td_tag}.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        print(f"  Saved: {p1}")

    # ─── Plot 2: h²Ω_peak vs γ (both shapes) ──────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 8))

    def _split(a, g, yy, color, fn, **kw):
        fn(g[m_ok], yy[m_ok], color=color, ls=kw.get("ls", "-"),
           lw=kw.get("lw", 2), zorder=kw.get("z", 5),
           label=kw.get("label", None))
        fn(g[m_ex], yy[m_ex], color=color, ls="--",
           lw=max(kw.get("lw", 2) - 0.5, 1), alpha=0.4, zorder=4)

    _split(ax3, g_sc, scan_h2pk["env"], "royalblue", ax3.loglog, lw=2.0,
           label=r"Collision — envelope (HK 2008)")
    _split(ax3, g_sc, scan_h2pk["jt"], "b", ax3.loglog, lw=2.5,
           label=r"Collision — JT 2016")
    _split(ax3, g_sc, scan_h2pk["jt"] + scan_h2pk_turb_hi, "r",
           ax3.loglog, lw=2.5, z=6,
           label=r"JT + Turb ($\epsilon=10\%$)")
    ax3.fill_between(g_sc[m_ok],
                     (scan_h2pk["jt"] + scan_h2pk_turb_lo)[m_ok],
                     (scan_h2pk["jt"] + scan_h2pk_turb_hi)[m_ok],
                     color="red", alpha=0.10, zorder=3,
                     label=r"Turb band ($\epsilon=5$–$10\%$)")
    _split(ax3, g_sc, scan_h2pk_sw, "g", ax3.loglog, lw=1.5,
           label=r"Sound wave (ref.)")

    for name, sens_fn, frange, col in [
        ("LISA", sensitivity_LISA, (-5, -0.5), "purple"),
        ("DECIGO", sensitivity_DECIGO, (-3, 2), "orange"),
        ("BBO", sensitivity_BBO, (-3, 2), "cyan"),
    ]:
        fv = np.logspace(*frange, 500)
        min_s = np.min(sens_fn(fv))
        ax3.axhline(min_s, color=col, ls=":", lw=1.2, alpha=0.6)
        ax3.text(1e-1 * 0.6, min_s * 2, f"{name} floor",
                 fontsize=9, color=col, ha="right")

    ax3.axvline(GAMMA_CUT, color="gray", ls=":", lw=1, alpha=0.7)
    ax3.axvspan(g_sc[0] * 0.5, GAMMA_CUT, color="gray", alpha=0.05, zorder=0)

    ax3.set_xlabel(r"$\gamma = \phi_0 / M_{\rm Pl}$", fontsize=14)
    ax3.set_ylabel(r"$h^2 \Omega_{\rm peak}$", fontsize=14)
    ax3.set_title(
        r"Peak GW amplitude vs $\gamma$"
        rf"  ($m = 1$ TeV, $T_d = {T_D:.0f}$ GeV)",
        fontsize=12,
    )
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(True, which="both", alpha=0.25)
    fig3.tight_layout()
    p3 = f"{OUT}/gw_comparison_{td_tag}.png"
    fig3.savefig(p3, dpi=200)
    plt.close(fig3)
    print(f"  Saved: {p3}")

    # ─── Plot 3: parameter panels ────────────────────────────────
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_fpk, ax_opk), (ax_beta, ax_tn) = axes

    g_sel = np.array([s["gamma"] for s in sig_coll["jt"]])
    fpk_sel = np.array([s["fpk"] for s in sig_coll["jt"]])
    h2pk_sel = np.array([s["h2pk"] for s in sig_coll["jt"]])
    beta_sel = np.array([s["beta_H"] for s in sig_coll["jt"]])
    Tn_sel = np.array([s["Tn"] for s in sig_coll["jt"]])

    def _plot_split(a, g, yy, color, fn, **kw):
        fn(g[m_ok], yy[m_ok], color=color, ls=kw.get("ls", "-"),
           lw=kw.get("lw", 2), zorder=3, label=kw.get("label", None))
        fn(g[m_ex], yy[m_ex], color=color, ls="--",
           lw=kw.get("lw", 1.5), alpha=0.45, zorder=2)
        a.axvline(GAMMA_CUT, color="gray", ls=":", lw=1, alpha=0.7, zorder=1)

    def _shade(a):
        a.axvspan(g_sc[0] * 0.5, GAMMA_CUT, color="gray", alpha=0.07, zorder=0)
        yl = a.get_ylim()
        if a.get_yscale() == "log":
            ytxt = math.exp((math.log(yl[0]) + math.log(yl[1])) / 2)
        else:
            ytxt = (yl[0] + yl[1]) / 2
        a.text(
            GAMMA_CUT * 0.012, ytxt,
            r"excluded ($N_{\rm TI} \leq 10$)",
            fontsize=8.5, color="gray", rotation=90, va="center", ha="center",
        )

    _plot_split(ax_fpk, g_sc, scan_fpk["jt"], "b", ax_fpk.loglog)
    ax_fpk.loglog(g_sel, fpk_sel, "ro", ms=8, zorder=5)
    ax_fpk.set_xlabel(r"$\gamma$", fontsize=13)
    ax_fpk.set_ylabel(r"$f_{\rm peak}$ [Hz]", fontsize=13)
    ax_fpk.set_title("Peak frequency", fontsize=13)
    ax_fpk.grid(True, which="both", alpha=0.3)
    _shade(ax_fpk)

    _plot_split(ax_opk, g_sc, scan_h2pk["env"], "royalblue", ax_opk.loglog,
                lw=1.5, label="Env (HK 2008)")
    _plot_split(ax_opk, g_sc, scan_h2pk["jt"], "b", ax_opk.loglog,
                label="JT 2016")
    _plot_split(ax_opk, g_sc, scan_h2pk["jt"] + scan_h2pk_turb_hi, "r",
                ax_opk.loglog, lw=2.0,
                label=r"JT+Turb ($\epsilon=10\%$)")
    ax_opk.loglog(g_sel, h2pk_sel, "ko", ms=8, zorder=5)
    for name, sens_fn, frange, col in [
        ("LISA", sensitivity_LISA, (-5, -0.5), "purple"),
        ("DECIGO", sensitivity_DECIGO, (-3, 2), "orange"),
        ("BBO", sensitivity_BBO, (-3, 2), "cyan"),
    ]:
        fv = np.logspace(*frange, 500)
        min_s = np.min(sens_fn(fv))
        ax_opk.axhline(min_s, color=col, ls="--", lw=1, alpha=0.6)
        ax_opk.text(g_sc[-1] * 0.6, min_s * 2.5,
                    f"{name} floor", fontsize=8, color=col, ha="right")
    ax_opk.set_xlabel(r"$\gamma$", fontsize=13)
    ax_opk.set_ylabel(r"$h^2 \Omega_{\rm peak}$", fontsize=13)
    ax_opk.set_title("Peak amplitude", fontsize=13)
    ax_opk.grid(True, which="both", alpha=0.3)
    ax_opk.legend(fontsize=8, loc="lower right")
    _shade(ax_opk)

    _plot_split(ax_beta, g_sc, beta_sc, "g", ax_beta.loglog)
    ax_beta.loglog(g_sel, beta_sel, "ko", ms=8, zorder=5)
    ax_beta.set_xlabel(r"$\gamma$", fontsize=13)
    ax_beta.set_ylabel(r"$\beta / H$", fontsize=13)
    ax_beta.set_title(r"$\beta/H$ at nucleation", fontsize=13)
    ax_beta.grid(True, which="both", alpha=0.3)
    _shade(ax_beta)

    _plot_split(ax_tn, g_sc, Tn_sc / 1e3, "m", ax_tn.semilogx)
    ax_tn.semilogx(g_sel, Tn_sel / 1e3, "ko", ms=8, zorder=5)
    ax_tn.axhline(T_c2 / 1e3, color="gray", ls=":", lw=1, alpha=0.6)
    ax_tn.text(g_sc[-1], T_c2 / 1e3 + 0.002, r"$T_{c2}$", fontsize=9, color="gray")
    ax_tn.set_xlabel(r"$\gamma$", fontsize=13)
    ax_tn.set_ylabel(r"$T_n$ [TeV]", fontsize=13)
    ax_tn.set_title("Nucleation temperature", fontsize=13)
    ax_tn.grid(True, which="both", alpha=0.3)
    _shade(ax_tn)

    fig2.suptitle(
        r"GW parameters vs $\gamma$" rf"  ($m = 1$ TeV, $T_d = {T_D:.0f}$ GeV)",
        fontsize=14,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = f"{OUT}/gw_peak_vs_gamma_{td_tag}.png"
    fig2.savefig(p2, dpi=200)
    plt.close(fig2)
    print(f"  Saved: {p2}")

# ═══════════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 120)
print("  SUMMARY: Collision (envelope vs JT 2016) + Turbulence")
print("═" * 120)
print(f"{'γ':>10}  {'Td':>6}  {'h²Ω(env)':>14}  {'h²Ω(JT)':>14}  "
      f"{'h²Ω(turb 10%)':>15}  {'h²Ω(JT+turb)':>16}")
print("─" * 120)
for i, gi in enumerate(gamma_info):
    exp = int(round(math.log10(gi["gamma"])))
    for T_D in T_D_LIST:
        _, _, h_env = gw_thermal_inflation(
            np.array([1.0]), gi["beta_H"], gi["V0"], T_D,
            g_star=g_star, v_w=1.0, kappa_phi=1.0,
            shape="envelope", efficiency="envelope",
        )
        _, _, h_jt = gw_thermal_inflation(
            np.array([1.0]), gi["beta_H"], gi["V0"], T_D,
            g_star=g_star, v_w=1.0, kappa_phi=1.0,
        )
        _, _, h_thi = gw_turbulence_thermal_inflation(
            np.array([1.0]), gi["beta_H"], gi["V0"], T_D,
            eps_turb=EPS_TURB_HI, g_star=g_star, v_w=1.0,
        )
        print(f"  10^{exp:+d}  {T_D:5.0f}  {h_env:14.2e}  {h_jt:14.2e}  "
              f"{h_thi:15.2e}  {h_jt + h_thi:16.2e}")

print()
print(f"  Turbulence range: ε_turb = {EPS_TURB_LO:.0%} (lower) to {EPS_TURB_HI:.0%} (upper)")
print()
print("Done.")
