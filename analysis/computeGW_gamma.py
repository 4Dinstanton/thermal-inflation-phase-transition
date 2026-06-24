#!/usr/bin/env python3
"""GW spectra vs γ from TIPT_4 (Section V, Eqs. 94–113, 99–101).

Implements the thermal-inflation benchmark used in the paper:
  • Plasma-coupled:  h²Ω_sw,0 + h²Ω_turb,0  (sound waves + MHD turbulence)
  • Weak-friction upper envelope:  h²Ω_φ,0  (scalar bubble collisions, κ_φ = 1)

Fixed illustration parameters (Eq. 87):  β/H* = 1000, v_w = 1, g_* = g_*s = 100,
m_φ = 1 TeV, T_d = 0.1 TeV.  Only γ (hence V_0) is scanned.

Detector curves use published PLISC tables (Schmitz 2020, Zenodo 3689582).

Usage:
    python analysis/computeGW_gamma.py --plot
    python analysis/computeGW_gamma.py --csv figs/gw_gamma_spectra.csv
    python analysis/computeGW_gamma.py --validate
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from gwSpectrum import GEV_TO_HZ, G_S0, M_PL, T_CMB_GEV
from published_plisc import (
    DETECTORS,
    load_all as load_published_plisc,
    place_detector_label,
)

# ═══════════════════════════════════════════════════════════════════
#  Paper defaults (TIPT_4, Eq. 86–87, Section V)
# ═══════════════════════════════════════════════════════════════════
M_PHI_GEV = 1000.0  # m_φ = 1 TeV
T_D_GEV = 100.0  # T_d = 0.1 TeV
G_STAR_D = 100.0
G_STAR_S_D = 100.0
BETA_OVER_H = 1000.0
V_W = 1.0
U_F = math.sqrt(3.0 / 4.0)  # Ū_f for strong-transition benchmark (Eq. 105)
EPSILON_TURB = 0.05  # ε_turb in Eq. (112)
KAPPA_V = 1.0

DEFAULT_GAMMAS = [1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3]


@dataclass(frozen=True)
class TIPT4GWParams:
    """Parameters for TIPT_4 Section V GW estimates."""

    m_phi: float = M_PHI_GEV
    M_pl: float = M_PL
    T_d: float = T_D_GEV
    g_star: float = G_STAR_D
    g_star_s_d: float = G_STAR_S_D
    g_star_s_0: float = G_S0
    T_cmb: float = T_CMB_GEV
    beta_over_H: float = BETA_OVER_H
    v_w: float = V_W
    epsilon_turb: float = EPSILON_TURB
    kappa_v: float = KAPPA_V

    # Reference scales in Eq. (108), (110), (99) for dimensionless prefactors
    T_d_ref: float = 100.0  # 0.1 TeV
    V04_ref: float = 1.0e7  # 10^4 TeV in GeV
    beta_ref: float = 1000.0


# ═══════════════════════════════════════════════════════════════════
#  Spectral shapes (Eqs. 100, 103, 111)
# ═══════════════════════════════════════════════════════════════════
def S_sw(x: np.ndarray) -> np.ndarray:
    """Sound-wave template S_sw(x), Eq. (103)."""
    x = np.asarray(x, dtype=float)
    return x**3 * (7.0 / (4.0 + 3.0 * x**2)) ** 3.5


def S_turb(f: np.ndarray, f_turb: float, f_H: float) -> np.ndarray:
    """Turbulence template S_turb, Eq. (111)."""
    f = np.asarray(f, dtype=float)
    x = f / f_turb
    return x**3 / ((1.0 + x) ** (11.0 / 3.0) * (1.0 + 8.0 * math.pi * f / f_H))


def S_phi(x: np.ndarray) -> np.ndarray:
    """Scalar-collision envelope S_φ(x), Eq. (100)."""
    x = np.asarray(x, dtype=float)
    return 3.8 * x**2.8 / (1.0 + 2.8 * x**3.8)


# ═══════════════════════════════════════════════════════════════════
#  Cosmology and amplitudes (Eqs. 94–98, 106–108, 110, 99)
# ═══════════════════════════════════════════════════════════════════
def V0_of_gamma(gamma: float, params: TIPT4GWParams | None = None) -> float:
    """V_0 = m² φ₀²/4 with φ₀ = γ M_Pl, Eq. (90) context."""
    p = params or TIPT4GWParams()
    phi0 = gamma * p.M_pl
    return 0.25 * p.m_phi**2 * phi0**2


def V04_of_gamma(gamma: float, params: TIPT4GWParams | None = None) -> float:
    """V_0^(1/4) in GeV: (γ m_φ M_Pl / 2)^(1/2)."""
    p = params or TIPT4GWParams()
    return math.sqrt(gamma * p.m_phi * p.M_pl / 2.0)


def R_md(gamma: float, params: TIPT4GWParams | None = None) -> float:
    """Matter-domination dilution factor R_md, Eq. (94)."""
    p = params or TIPT4GWParams()
    V0 = V0_of_gamma(gamma, p)
    return (math.pi**2 * p.g_star * p.T_d**4 / (30.0 * V0)) ** (1.0 / 3.0)


def H_star(gamma: float, params: TIPT4GWParams | None = None) -> float:
    """H_* ≃ H_TI at GW production [GeV], Section V."""
    p = params or TIPT4GWParams()
    return gamma * p.m_phi / (2.0 * math.sqrt(3.0))


def a_d_over_a0(params: TIPT4GWParams | None = None) -> float:
    """Scale-factor ratio a_d/a_0, Eq. (96)."""
    p = params or TIPT4GWParams()
    return (p.g_star_s_0 / p.g_star_s_d) ** (1.0 / 3.0) * p.T_cmb / p.T_d


def f_H0(gamma: float, params: TIPT4GWParams | None = None) -> float:
    """Redshifted Hubble frequency f_{H,0}, Eq. (97)–(98) [Hz]."""
    p = params or TIPT4GWParams()
    return H_star(gamma, p) * GEV_TO_HZ * R_md(gamma, p) * a_d_over_a0(p)


def Upsilon_sw(gamma: float, params: TIPT4GWParams | None = None) -> float:
    """Acoustic lifetime factor Υ_sw, Eqs. (105)–(106)."""
    p = params or TIPT4GWParams()
    H_Rstar = (8.0 * math.pi) ** (1.0 / 3.0) * p.v_w / p.beta_over_H
    return min(1.0, H_Rstar / U_F)


@dataclass
class GWBenchmark:
    """Per-γ GW quantities and spectra."""

    gamma: float
    V0: float
    V04: float
    R_md: float
    f_H: float
    f_sw: float
    f_turb: float
    f_phi: float
    Upsilon_sw: float
    h2Omega_sw_peak: float
    h2Omega_turb_peak: float
    h2Omega_phi_peak: float


def model_quantities(gamma: float, params: TIPT4GWParams | None = None) -> GWBenchmark:
    """All frequencies and peak amplitudes for one γ."""
    p = params or TIPT4GWParams()
    V0 = V0_of_gamma(gamma, p)
    V04 = V04_of_gamma(gamma, p)
    R = R_md(gamma, p)
    fH = f_H0(gamma, p)
    ups = Upsilon_sw(gamma, p)

    f_sw = 1.16 * p.beta_over_H * fH / p.v_w
    f_turb = 1.64 * p.beta_over_H * fH / p.v_w
    f_phi = 0.62 / (1.8 - 0.1 * p.v_w + p.v_w**2) * p.beta_over_H * fH

    # Eq. (108) simplified amplitude (includes Υ_sw)
    h2_sw = (
        8.21e-16
        * (p.T_d / p.T_d_ref) ** (4.0 / 3.0)
        * (V04 / p.V04_ref) ** (-4.0 / 3.0)
        * (p.beta_over_H / p.beta_ref) ** (-1.0)
        * ups
    )

    # Eq. (110) with κ_turb = ε_turb κ_v, α* ≫ 1
    h2_turb = (
        3.35e-4
        * (1.0 / p.beta_over_H)
        * (p.epsilon_turb * p.kappa_v) ** 1.5
        * (100.0 / p.g_star) ** (1.0 / 3.0)
        * p.v_w
        * R
    )

    # Eq. (99) optimistic upper envelope (κ_φ = 1, α* ≫ 1)
    h2_phi = (
        1.67e-5
        * (1.0 / p.beta_over_H) ** 2
        * (100.0 / p.g_star) ** (1.0 / 3.0)
        * (0.11 * p.v_w**3 / (0.42 + p.v_w**2))
        * R
    )

    return GWBenchmark(
        gamma=gamma,
        V0=V0,
        V04=V04,
        R_md=R,
        f_H=fH,
        f_sw=f_sw,
        f_turb=f_turb,
        f_phi=f_phi,
        Upsilon_sw=ups,
        h2Omega_sw_peak=h2_sw,
        h2Omega_turb_peak=h2_turb,
        h2Omega_phi_peak=h2_phi,
    )


def h2Omega_sw(
    f: np.ndarray, gamma: float, params: TIPT4GWParams | None = None
) -> np.ndarray:
    """h²Ω_sw,0(f), Eq. (108) with shape Eq. (103)."""
    q = model_quantities(gamma, params)
    x = np.asarray(f, dtype=float) / q.f_sw
    return q.h2Omega_sw_peak * S_sw(x)


def h2Omega_turb(
    f: np.ndarray, gamma: float, params: TIPT4GWParams | None = None
) -> np.ndarray:
    """h²Ω_turb,0(f), Eq. (110) with shape Eq. (111)."""
    q = model_quantities(gamma, params)
    return q.h2Omega_turb_peak * S_turb(f, q.f_turb, q.f_H)


def h2Omega_phi(
    f: np.ndarray, gamma: float, params: TIPT4GWParams | None = None
) -> np.ndarray:
    """h²Ω_φ,0(f) weak-friction upper envelope, Eq. (99)–(100)."""
    q = model_quantities(gamma, params)
    x = np.asarray(f, dtype=float) / q.f_phi
    return q.h2Omega_phi_peak * S_phi(x)


def h2Omega_plasma(
    f: np.ndarray, gamma: float, params: TIPT4GWParams | None = None
) -> np.ndarray:
    """Plasma-coupled benchmark: sound waves + turbulence."""
    return h2Omega_sw(f, gamma, params) + h2Omega_turb(f, gamma, params)


# ═══════════════════════════════════════════════════════════════════
#  Validation against paper numerics (Eqs. 98, 99, 108, 109)
# ═══════════════════════════════════════════════════════════════════
def validate_benchmark(params: TIPT4GWParams | None = None) -> None:
    """Check γ with V_0^(1/4) = 10^4 TeV against quoted paper values."""
    p = params or TIPT4GWParams()
    gamma = 2.0 * p.V04_ref**2 / (p.m_phi * p.M_pl)
    q = model_quantities(gamma, p)

    checks = [
        ("f_H,0 [Hz]", q.f_H, 2.01e-2, 0.02),
        ("f_sw,0 [Hz]", q.f_sw, 2.33e1, 0.02),
        ("h²Ω_sw peak", q.h2Omega_sw_peak, 2.76e-18, 0.03),
        ("h²Ω_φ peak", q.h2Omega_phi_peak, 8.93e-19, 0.02),
        ("Υ_sw", q.Upsilon_sw, 3.38e-3, 0.05),
    ]
    print(f"Benchmark γ = {gamma:.4e}  (V_0^(1/4) = {q.V04/1e3:.0f} TeV)")
    print(f"{'Quantity':<18} {'computed':>14} {'paper':>14} {'rel.err':>10}")
    print("-" * 60)
    for name, val, ref, tol in checks:
        rel = abs(val - ref) / ref
        ok = "OK" if rel <= tol else "FAIL"
        print(f"{name:<18} {val:14.4e} {ref:14.4e} {rel:10.4f}  {ok}")


# ═══════════════════════════════════════════════════════════════════
#  Plotting (Fig. 7 style, published PLISC)
# ═══════════════════════════════════════════════════════════════════
_GAMMA_COLORS = [
    "teal",
    "darkviolet",
    "darkorange",
    "forestgreen",
    "crimson",
    "navy",
]


def _gamma_label(gamma: float) -> str:
    return rf"$\gamma=10^{{{int(round(math.log10(gamma)))}}}$"


def _add_detectors(ax, plisc: dict | None = None) -> None:
    """Overlay Schmitz (2020) published PLISC curves (Zenodo 3689582)."""
    curves = plisc or load_published_plisc()
    for name, (_, color, f_target) in DETECTORS.items():
        fv, sv = curves[name]
        ax.loglog(fv, sv, color=color, ls="-.", lw=1.5, alpha=0.85, zorder=2)
        place_detector_label(ax, name, fv, sv, f_target)


def _annotate_gamma_on_curve(
    ax,
    freq: np.ndarray,
    spectrum: np.ndarray,
    gamma: float,
    color: str,
    *,
    y_scale: float = 3.0,
) -> None:
    """Place γ label above the peak of a benchmark spectrum."""
    idx = int(np.argmax(spectrum))
    f_pk, o_pk = float(freq[idx]), float(spectrum[idx])
    ax.text(
        f_pk,
        o_pk * y_scale,
        _gamma_label(gamma),
        fontsize=11,
        fontweight="bold",
        color=color,
        ha="center",
        va="bottom",
        zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            alpha=0.82,
            edgecolor="none",
        ),
    )


def _add_benchmark_legend(
    ax,
    *,
    lw: float = 2.2,
    show_collision: bool = False,
) -> None:
    """Legend on the right: plasma total and optional bubble-collision envelope."""
    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=lw,
            label="Plasma-coupled benchmark",
        ),
    ]
    if show_collision:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                ls="--",
                lw=1.55,
                label="Bubble collision upper envelope",
            )
        )
    ax.legend(
        handles=handles,
        loc="right",
        fontsize=10,
        framealpha=0.92,
    )


def _style_axes(
    ax,
    ylabel: str,
    title: str,
    *,
    ylo: float = 1e-25,
    yhi: float = 1e-4,
) -> None:
    ax.set_xlabel(r"$f$ [Hz]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(1e-5, 1e5)
    ax.set_ylim(ylo, yhi)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", alpha=0.25)


def plot_figures(
    gammas: Iterable[float],
    freq: np.ndarray,
    out_dir: str,
    params: TIPT4GWParams | None = None,
) -> None:
    """Write plasma, collision, and combined figures (TIPT_4 Fig. 7)."""
    os.makedirs(out_dir, exist_ok=True)
    p = params or TIPT4GWParams()
    gammas = list(gammas)
    plisc = load_published_plisc()

    # --- Plasma-coupled (Turb + SW, multi-γ style) ---
    fig, ax = plt.subplots(figsize=(11, 7.5))
    _add_detectors(ax, plisc)
    for i, gv in enumerate(gammas):
        c = _GAMMA_COLORS[i % len(_GAMMA_COLORS)]
        total = h2Omega_plasma(freq, gv, p)
        ax.loglog(freq, total, color=c, lw=2.5, zorder=5)
        _annotate_gamma_on_curve(ax, freq, total, gv, c, y_scale=3.5)
    _style_axes(
        ax,
        r"$h^2 \Omega_{\mathrm{GW}}$",
        r"GW Spectrum — Turb + SW ($m_\phi = 1$ TeV, $T_d = 0.1$ TeV)",
    )
    _add_benchmark_legend(ax, lw=2.5, show_collision=False)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/gw_plasma_TIPT4.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{out_dir}/gw_plasma_TIPT4.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Scalar collision upper envelope ---
    fig, ax = plt.subplots(figsize=(11, 7.5))
    _add_detectors(ax, plisc)
    for i, gv in enumerate(gammas):
        c = _GAMMA_COLORS[i % len(_GAMMA_COLORS)]
        phi = h2Omega_phi(freq, gv, p)
        ax.loglog(freq, phi, color=c, lw=2.5, zorder=5)
        _annotate_gamma_on_curve(ax, freq, phi, gv, c, y_scale=3.0)
    _style_axes(
        ax,
        r"$h^2 \Omega_{\phi}$",
        r"GW Spectrum — scalar collision ($\kappa_\phi=1$, $m_\phi=1$ TeV)",
        yhi=1e-5,
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/gw_collision_TIPT4.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{out_dir}/gw_collision_TIPT4.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Combined (Fig. 7) ---
    fig, ax = plt.subplots(figsize=(11.6, 7.5))
    _add_detectors(ax, plisc)
    for i, gv in enumerate(gammas):
        c = _GAMMA_COLORS[i % len(_GAMMA_COLORS)]
        total = h2Omega_plasma(freq, gv, p)
        phi = h2Omega_phi(freq, gv, p)
        ax.loglog(freq, total, color=c, lw=2.2, zorder=5)
        ax.loglog(freq, phi, color=c, ls="--", lw=1.55, alpha=0.9, zorder=4)
        _annotate_gamma_on_curve(ax, freq, total, gv, c, y_scale=3.0)
    _style_axes(
        ax,
        r"$h^2 \Omega_{\mathrm{GW}}$",
        r"Thermal-inflation GW benchmarks and published PLISC curves",
        yhi=1e-5,
    )
    _add_benchmark_legend(ax, lw=2.2, show_collision=True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/gw_combined_TIPT4.png", dpi=260, bbox_inches="tight")
    fig.savefig(f"{out_dir}/gw_combined_TIPT4.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures under {out_dir}/")


def export_summary_csv(
    gammas: Iterable[float],
    csv_path: str,
    params: TIPT4GWParams | None = None,
) -> None:
    """Write per-γ peak frequencies and amplitudes."""
    import csv

    p = params or TIPT4GWParams()
    rows = []
    for gv in gammas:
        q = model_quantities(gv, p)
        rows.append(
            {
                "gamma": gv,
                "V0_GeV4": q.V0,
                "V04_GeV": q.V04,
                "R_md": q.R_md,
                "f_H0_Hz": q.f_H,
                "f_sw0_Hz": q.f_sw,
                "f_turb0_Hz": q.f_turb,
                "f_phi0_Hz": q.f_phi,
                "Upsilon_sw": q.Upsilon_sw,
                "h2Omega_sw_peak": q.h2Omega_sw_peak,
                "h2Omega_turb_peak": q.h2Omega_turb_peak,
                "h2Omega_phi_peak": q.h2Omega_phi_peak,
                "h2Omega_plasma_peak": q.h2Omega_sw_peak + q.h2Omega_turb_peak,
            }
        )
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved summary: {csv_path}")


def print_summary_table(
    gammas: Iterable[float], params: TIPT4GWParams | None = None
) -> None:
    p = params or TIPT4GWParams()
    print("\n" + "═" * 90)
    print("  TIPT_4 GW benchmarks vs γ  (β/H* = 1000, T_d = 0.1 TeV, v_w = 1)")
    print("═" * 90)
    print(
        f"{'γ':>12}  {'V0^1/4 [TeV]':>14}  {'f_sw [Hz]':>12}  "
        f"{'h²Ω_sw':>12}  {'h²Ω_turb':>12}  {'h²Ω_φ':>12}  {'Υ_sw':>10}"
    )
    print("-" * 90)
    for gv in gammas:
        q = model_quantities(gv, p)
        exp = int(round(math.log10(gv)))
        print(
            f"  10^{exp:+3d}  {q.V04/1e3:14.3f}  {q.f_sw:12.3e}  "
            f"{q.h2Omega_sw_peak:12.3e}  {q.h2Omega_turb_peak:12.3e}  "
            f"{q.h2Omega_phi_peak:12.3e}  {q.Upsilon_sw:10.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TIPT_4 Section V GW spectrum vs γ")
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=None,
        help="γ values to evaluate (default: 10^-13 … 10^-3)",
    )
    parser.add_argument(
        "--Td",
        type=float,
        default=T_D_GEV,
        help="Flaton decay temperature [GeV] (default: 100 = 0.1 TeV)",
    )
    parser.add_argument(
        "--beta-over-H",
        type=float,
        default=BETA_OVER_H,
        help="β/H* (default: 1000)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate Fig. 7-style PNG/PDF plots",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write per-γ summary CSV to this path",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print benchmark check vs paper numerics (V0^1/4 = 10^4 TeV)",
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=2200,
        help="Number of frequency points",
    )
    args = parser.parse_args()

    gammas = args.gammas if args.gammas is not None else DEFAULT_GAMMAS
    params = TIPT4GWParams(T_d=args.Td, beta_over_H=args.beta_over_H)

    if args.validate:
        validate_benchmark(params)

    print_summary_table(gammas, params)

    freq = np.logspace(-5, 5, args.n_freq)

    if args.csv:
        export_summary_csv(gammas, args.csv, params)

    if args.plot:
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "figs",
            "gw_TIPT4",
        )
        plot_figures(gammas, freq, out_dir, params)


if __name__ == "__main__":
    main()
