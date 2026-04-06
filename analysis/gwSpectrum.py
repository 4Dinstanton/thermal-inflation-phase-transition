#!/usr/bin/env python3
"""
Gravitational wave spectrum from a first-order cosmological phase transition
during thermal inflation.

Reads tunneling CSV data (T, S3/T), computes nucleation parameters
(T_n, T_RH, beta/H, HR_*, alpha), and plots the GW power spectrum with
detector sensitivity curves overlaid.

Formula sets via --formula:

  TI    – Thermal Inflation (recommended, default)
          First-principles calculation for vacuum-dominated transitions:
            * Only bubble collisions — no sound waves / turbulence
              (physically absent in vacuum-dominated era)
            * Proper redshift through flaton matter-dominated era
            * Spectral shapes: --shape envelope | jt2016
            * Efficiency models: --efficiency envelope | jt2016

  DJS   – Dutka, Jung, Shin (2024) [arXiv:2412.15864]
          Sound waves (Hindmarsh et al. 2017), turbulence, envelope.
          Suppression factor from Ellis, Lewicki, No (2019).
          NOTE: sound-wave formulas assume radiation domination —
          physically inconsistent for vacuum-dominated transitions.

  EGLPS – Easther, Giblin, Lim, Park, Stewart (2008) [arXiv:0801.4197]
          Legacy bubble-collision envelope for vacuum-dominated transitions.
          Kept for backward comparison.

Key references (vacuum-dominated / thermal inflation GW):
  [1] Kosowsky, Turner, Watkins (1992)       arXiv:astro-ph/9211004
  [2] Kamionkowski, Kosowsky, Turner (1994)   arXiv:astro-ph/9311023
  [3] Huber & Konstandin (2008)               arXiv:0806.1828
  [4] Easther, Giblin, Lim, Park, Stewart (2008) arXiv:0801.4197
  [5] Espinosa, Konstandin, No, Servant (2010)   arXiv:1004.4187
  [6] Caprini et al. (2016) [LISA CWG]       arXiv:1512.06239
  [7] Jinno & Takimoto (2017)                 arXiv:1707.03111
  [8] Cutting, Hindmarsh, Huber, Konstandin (2018) arXiv:1802.04276
  [9] Caprini et al. (2020) [LISA CWG update] arXiv:1910.13125
  [10] Cutting, Escartin, Hindmarsh, Weir (2020)  arXiv:2005.13537
  [11] Lewicki & Vaskonen (2020)              arXiv:2007.04967
  [12] Dutka, Jung, Shin (2024)               arXiv:2412.15864

Usage:
    python analysis/gwSpectrum.py <csv> --formula TI --T_d 100
    python analysis/gwSpectrum.py <csv> --formula TI --T_d 100 --shape jt2016 --efficiency jt2016
    python analysis/gwSpectrum.py <csv> --formula DJS
    python analysis/gwSpectrum.py <csv> --formula EGLPS --T_d 100
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
M_PL = 2.4e18  # reduced Planck mass (GeV)
G_STAR_DEFAULT = 106.75  # SM relativistic DOF
CHI_G2 = 30.0 / (math.pi**2 * G_STAR_DEFAULT)
DEL_V_DEFAULT = 1.0e28 / 4  # vacuum energy difference (GeV^4)
H_PARAM = 0.674  # h = H0 / (100 km/s/Mpc)
T_CMB_GEV = 2.7255 * 8.6173e-14  # CMB temperature in GeV
G_S0 = 3.91  # entropy DOF today (photons + neutrinos)
HBAR_GEV_S = 6.5822e-25  # ℏ in GeV·s
GEV_TO_HZ = 1.0 / HBAR_GEV_S  # convert natural-unit freq (GeV) → Hz


# ---------------------------------------------------------------------------
# Hubble parameter
# ---------------------------------------------------------------------------
def hubble(T, del_V=DEL_V_DEFAULT):
    """H(T) in GeV, radiation + vacuum energy."""
    T = np.asarray(T, dtype=float)
    return np.sqrt((T**4 / CHI_G2 + del_V) / (3.0 * M_PL**2))


# ---------------------------------------------------------------------------
# Reheating temperature: rho_rad(T_RH) = delV
# ---------------------------------------------------------------------------
def compute_T_RH(del_V, g_star):
    """T_RH = (30 delV / (pi^2 g_*))^{1/4}."""
    return (30.0 * del_V / (math.pi**2 * g_star)) ** 0.25


def compute_T_RH_model(Ms, gX, mu_star, g_star):
    """T_RH from model parameters. Eq. (4.9) of arXiv:2412.15864."""
    return (
        1.6e6  # 1.6 PeV in GeV
        * (100.0 / g_star) ** 0.25
        * (math.sqrt(gX) * Ms / 1.0e4) ** 0.5
        * (mu_star / 1.0e10) ** 0.5
    )


def compute_delV_model(Ms, gX, mu_star):
    """delV from model parameters. Eq. (4.7) of arXiv:2412.15864."""
    return math.e**0.5 / (8.0 * math.pi**2) * gX * Ms**2 * mu_star**2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tunneling_data(csv_path):
    """Return (T, S3_over_T) arrays sorted by ascending T."""
    df = pd.read_csv(csv_path)
    if "S3/T" not in df.columns or "T" not in df.columns:
        raise ValueError(
            f"CSV must contain 'T' and 'S3/T' columns. Found: {list(df.columns)}"
        )
    df = df.dropna(subset=["T", "S3/T"])
    df = df[df["S3/T"] > 0].copy()
    df.sort_values("T", inplace=True)
    return df["T"].values, df["S3/T"].values


# ---------------------------------------------------------------------------
# Nucleation rate  ln(Gamma) = -S3/T + 4 ln T + (3/2) ln(S3/(2piT))
# ---------------------------------------------------------------------------
def ln_gamma(T, S3_T):
    T = np.asarray(T, dtype=float)
    S3_T = np.asarray(S3_T, dtype=float)
    return -S3_T + 4.0 * np.log(T) + 1.5 * np.log(S3_T / (2.0 * math.pi))


# ---------------------------------------------------------------------------
# Find nucleation temperature  Gamma / H^4 = 1
# ---------------------------------------------------------------------------
def find_nucleation_temp(T, S3_T, del_V=DEL_V_DEFAULT):
    lg = ln_gamma(T, S3_T)
    H = hubble(T, del_V)
    log_ratio = lg - 4.0 * np.log(H)

    crossings = np.where(np.diff(np.sign(log_ratio)))[0]
    if len(crossings) == 0:
        max_ratio = np.max(log_ratio)
        idx_max = np.argmax(log_ratio)
        print(
            f"  WARNING: Gamma/H^4 never reaches 1.  "
            f"Max ln(Gamma/H^4) = {max_ratio:.2f} at T = {T[idx_max]:.1f} GeV"
        )
        if max_ratio > -5:
            print("  Using temperature of maximum Gamma/H^4 as approximate T_n")
            T_n = T[idx_max]
        else:
            raise RuntimeError("Nucleation condition Gamma/H^4 >= 1 never met.")
    else:
        idx = crossings[0]
        t0 = log_ratio[idx] / (log_ratio[idx] - log_ratio[idx + 1])
        T_n = T[idx] + t0 * (T[idx + 1] - T[idx])
    return T_n


# ---------------------------------------------------------------------------
# beta/H = -d ln(Gamma) / d ln(T)  at T_n
# ---------------------------------------------------------------------------
def compute_beta_over_H(T, S3_T, T_n):
    lg = ln_gamma(T, S3_T)
    ln_T = np.log(T)
    cs = CubicSpline(ln_T, lg)
    dln_Gamma_dln_T = cs(np.log(T_n), 1)
    return float(-dln_Gamma_dln_T)


# ---------------------------------------------------------------------------
# Transition strength  alpha = delV / rho_rad(T_n)
# ---------------------------------------------------------------------------
def compute_alpha(T_n, del_V, g_star):
    rho_rad = (math.pi**2 / 30.0) * g_star * T_n**4
    return del_V / rho_rad


# ---------------------------------------------------------------------------
# Mean bubble separation  HR_* = (8pi)^{1/3} / (beta/H)
# ---------------------------------------------------------------------------
def compute_HR_star(beta_H):
    return (8.0 * math.pi) ** (1.0 / 3.0) / beta_H


# ---------------------------------------------------------------------------
# RMS fluid velocity  Uf^2 = (3/4) kappa_v alpha / (1 + alpha)
# ---------------------------------------------------------------------------
def compute_Uf(kv, alpha_val):
    return math.sqrt(0.75 * kv * alpha_val / (1.0 + alpha_val))


# ---------------------------------------------------------------------------
# Efficiency factors  —  Espinosa, Konstandin, No & Servant (2010)
#   [arXiv:1004.4187], Appendix A
# ---------------------------------------------------------------------------
def kappa_v_jouguet(alpha):
    """κ_D — Jouguet detonation (v_w = v_J⁺). Eq. (A.4)."""
    return alpha / (0.73 + 0.083 * math.sqrt(alpha) + alpha)


def kappa_v_deflagration(alpha, v_w):
    """κ_B — subsonic deflagration. Eq. (A.2).

    Valid for v_w < v_J(α).  For v_w → 0 the efficiency vanishes as v_w^{6/5}.
    """
    return v_w ** (6.0 / 5.0) * 6.9 * alpha / (1.36 - 0.037 * math.sqrt(alpha) + alpha)


def kappa_v(alpha, v_w=None):
    """General efficiency factor.

    If *v_w* is given and subsonic (< 1/√3), uses the deflagration fit κ_B.
    Otherwise falls back to the Jouguet detonation fit κ_D.
    """
    c_s = 1.0 / math.sqrt(3.0)
    if v_w is not None and v_w < c_s:
        return kappa_v_deflagration(alpha, v_w)
    return kappa_v_jouguet(alpha)


def kappa_turb_frac():
    return 0.05


# ---------------------------------------------------------------------------
# Peak frequencies (Hz today)
# ---------------------------------------------------------------------------
def f_peak_sw(HR_star, T_RH, g_star, z_p=10.0):
    """Sound wave peak. Eq. (4.33) of arXiv:2412.15864."""
    return (
        8.9
        * (8.0 * math.pi) ** (1.0 / 3.0)
        * 1.0e-2
        / HR_star
        * (z_p / 10.0)
        * (T_RH / 1.0e6)
        * (g_star / 100.0) ** (1.0 / 6.0)
    )


def f_peak_env(beta_H, T_RH, g_star, v_w):
    """Envelope peak. Caprini et al. (2016)."""
    return (
        1.65e-5
        * (T_RH / 100.0)
        * (g_star / 100.0) ** (1.0 / 6.0)
        * beta_H
        * 0.62
        / (1.8 - 0.1 * v_w + v_w**2)
    )


def f_peak_turb(beta_H, T_RH, g_star, v_w):
    """Turbulence peak. Caprini et al. (2016)."""
    return (
        2.7e-5 * (1.0 / v_w) * beta_H * (T_RH / 100.0) * (g_star / 100.0) ** (1.0 / 6.0)
    )


# ---------------------------------------------------------------------------
# GW spectrum: sound waves
#   Eq. (4.31) of arXiv:2412.15864  (from Hindmarsh et al. 2017)
# ---------------------------------------------------------------------------
def gw_sound_wave(f, HR_star, Uf, g_star, fp):
    """h^2 Omega_GW from sound waves."""
    f = np.asarray(f, dtype=float)
    F_GW0 = 3.5e-5 * (100.0 / g_star) ** (1.0 / 3.0)
    Gamma = 4.0 / 3.0
    Omega_tilde = 1.0e-2
    x = f / fp
    S_sw = x**3 * (7.0 / (4.0 + 3.0 * x**2)) ** 3.5
    return 2.061 * F_GW0 * H_PARAM**2 * Gamma**2 * Uf**4 * HR_star * Omega_tilde * S_sw


# ---------------------------------------------------------------------------
# GW spectrum: bubble collision (envelope)  -- Caprini et al. (2016)
# ---------------------------------------------------------------------------
def gw_envelope(f, alpha_val, beta_H, g_star, kphi, fp):
    f = np.asarray(f, dtype=float)
    Sf = 3.8 * (f / fp) ** 2.8 / (1.0 + 2.8 * (f / fp) ** 3.8)
    return (
        1.67e-5
        * (1.0 / beta_H) ** 2
        * (kphi * alpha_val / (1.0 + alpha_val)) ** 2
        * (100.0 / g_star) ** (1.0 / 3.0)
        * Sf
    )


# ---------------------------------------------------------------------------
# GW spectrum: MHD turbulence  -- Caprini et al. (2009)
# ---------------------------------------------------------------------------
def gw_turbulence(f, alpha_val, beta_H, T_RH, g_star, v_w, kturb, fp):
    f = np.asarray(f, dtype=float)
    h_star = 1.65e-5 * (T_RH / 100.0) * (g_star / 100.0) ** (1.0 / 6.0)
    Sf = (f / fp) ** 3 / (
        (1.0 + f / fp) ** (11.0 / 3.0) * (1.0 + 8.0 * math.pi * f / h_star)
    )
    return (
        3.35e-4
        * (1.0 / beta_H)
        * (kturb * alpha_val / (1.0 + alpha_val)) ** 1.5
        * (100.0 / g_star) ** (1.0 / 3.0)
        * v_w
        * Sf
    )


# ---------------------------------------------------------------------------
# GW spectrum: EGLPS – Easther, Giblin, Lim, Park, Stewart (2008)
#   arXiv:0801.4197, Eqs. (27), (29), (31)
#   Bubble-collision envelope for vacuum-dominated (alpha >> 1) transitions
#   with flaton matter domination after nucleation.
# ---------------------------------------------------------------------------
def eglps_peak_freq(beta_H, del_V, T_d):
    """Peak frequency today [Hz].  Eq. (29)."""
    V14 = del_V**0.25
    return (
        0.7
        * (beta_H / 1000.0)
        * (V14 / 1.0e6) ** (2.0 / 3.0)
        * (T_d / 100.0) ** (1.0 / 3.0)
    )


def eglps_peak_amplitude(beta_H, del_V, T_d):
    """Peak h^2 Omega_GW.  Eq. (31)."""
    V14 = del_V**0.25
    return (
        5.0e-18
        * (beta_H / 1000.0) ** (-2)
        * (V14 / 1.0e6) ** (-4.0 / 3.0)
        * (T_d / 100.0) ** (4.0 / 3.0)
    )


def gw_eglps(f, beta_H, del_V, T_d):
    """Legacy EGLPS bubble-collision spectrum (arXiv:0801.4197).

    Kept for backward comparison.  Prefer gw_thermal_inflation() for
    updated spectral shapes and efficiency factors.

    Returns (omega_gw, f_peak, omega_peak).
    Spectral shape: Kamionkowski, Kosowsky & Turner (1994)
    envelope S(x) = 3.8 x^{2.8} / (1 + 2.8 x^{3.8}),  S(1)=1.
    """
    f = np.asarray(f, dtype=float)
    f_peak = eglps_peak_freq(beta_H, del_V, T_d)
    omega_peak = eglps_peak_amplitude(beta_H, del_V, T_d)
    x = f / f_peak
    Sf = 3.8 * x**2.8 / (1.0 + 2.8 * x**3.8)
    return omega_peak * Sf, f_peak, omega_peak


# ---------------------------------------------------------------------------
# Spectral shapes for bubble-collision GW
# ---------------------------------------------------------------------------
def spectral_shape_envelope(x):
    r"""Envelope approximation spectral shape.

    S(x) = 3.8 x^{2.8} / (1 + 2.8 x^{3.8})
    Low-f: ~ f^{2.8},  High-f: ~ f^{-1.0}

    From Eq.(19) with fitting values a=2.8, b=1.0 for v_w ~ 1.

    Ref: Huber & Konstandin (2008) [arXiv:0806.1828], Eq. (19)
    """
    x = np.asarray(x, dtype=float)
    return 3.8 * x**2.8 / (1.0 + 2.8 * x**3.8)


def spectral_shape_jt2016(x, c_l=0.064, c_h=0.48):
    r"""Spectral shape from Jinno & Takimoto (2016) beyond-envelope calculation.

    S(x) = (c_l + c_h) / (c_l x^{-3} + c_h x)

    Normalized so that S(1) = 1.
    Low-f: ~ f^3 (causal),  High-f: ~ f^{-1}

    Parameters (c_l, c_h) = (0.064, 0.48) for v_w = 1 (runaway).

    Ref: Jinno & Takimoto (2016) [arXiv:1605.01403], Eq. (67)
    """
    x = np.asarray(x, dtype=float)
    return (c_l + c_h) / (c_l * x ** (-3.0) + c_h * x)


# ---------------------------------------------------------------------------
# GW production efficiency  Delta(v_w)
# ---------------------------------------------------------------------------
def efficiency_envelope(v_w):
    r"""Envelope approximation efficiency.

    Delta = 0.11 v_w^3 / (0.42 + v_w^2)
    For v_w = 1: Delta ~ 0.077

    Ref: Huber & Konstandin (2008) [arXiv:0806.1828], Eq. (22)
    """
    return 0.11 * v_w**3 / (0.42 + v_w**2)


def efficiency_jt2016(v_w):
    r"""Beyond-envelope efficiency from analytic derivation.

    Delta = 0.48 v_w^3 / (1 + 5.3 v_w^2 + 5 v_w^4)
    For v_w = 1: Delta ~ 0.042

    Ref: Jinno & Takimoto (2016) [arXiv:1605.01403], Eq. (75)
    """
    return 0.48 * v_w**3 / (1.0 + 5.3 * v_w**2 + 5.0 * v_w**4)


# ---------------------------------------------------------------------------
# Thermal Inflation GW: first-principles calculation
# ---------------------------------------------------------------------------
def gw_thermal_inflation(
    freq,
    beta_H,
    V_TI,
    T_d,
    g_star=G_STAR_DEFAULT,
    v_w=1.0,
    kappa_phi=1.0,
    shape="jt2016",
    efficiency="jt2016",
):
    r"""GW spectrum from bubble collisions during thermal inflation.

    First-principles calculation accounting for non-standard cosmology:
      vacuum domination → nucleation → bubble collision →
      flaton matter domination → flaton decay at T_d → radiation domination

    Only the bubble-collision source is included; sound waves and turbulence
    are physically absent in a vacuum-dominated transition (alpha >> 1).

    Derivation
    ----------
    1. Peak frequency at production (Huber & Konstandin 2008):
         f_peak,* = C_v × β   with  C_v = 0.62/(1.8 - 0.1 v_w + v_w^2)

    2. GW energy fraction at production:
         Ω_GW,* = Δ(v_w) × κ_φ^2 × (H/β)^2
       For vacuum transitions: κ_φ ≈ 1,  α/(1+α) ≈ 1.

    3. Redshift through matter-dominated era (* → d):
         R_md = [π^2 g_* T_d^4 / (30 V_TI)]^{1/3}
         Ω_GW,d = Ω_GW,* × R_md   (extra a^{-1} dilution vs background)

    4. Standard redshift from T_d to today:
         f_peak,0 = f_peak,* × R_md × (a_d/a_0)
         h^2 Ω_0  = F_GW × Ω_GW,d
       where F_GW = 1.67e-5 × (100/g_*)^{1/3}  [Caprini+ 2016]

    Parameters
    ----------
    freq : array_like
        Frequency array in Hz.
    beta_H : float
        β/H at nucleation.
    V_TI : float
        Vacuum energy during thermal inflation in GeV^4.
    T_d : float
        Flaton decay (reheating) temperature in GeV.
    g_star : float
        Relativistic DOF at T_d.
    v_w : float
        Bubble wall velocity (1.0 for vacuum/runaway).
    kappa_phi : float
        Fraction of vacuum energy in bubble walls (~1 for vacuum transitions).
    shape : str
        "envelope" (HK 2008) or "jt2016" (Jinno & Takimoto 2016).
    efficiency : str
        "envelope" (HK 2008) or "jt2016" (JT 2016).

    Returns
    -------
    omega_gw : ndarray
        h^2 Omega_GW(f)
    f_peak : float
        Peak frequency [Hz]
    h2_omega_peak : float
        Peak amplitude h^2 Omega_peak

    References
    ----------
    Huber & Konstandin (2008) [arXiv:0806.1828]  — envelope shape Eq.(19),
                                                    efficiency Eq.(22),
                                                    peak frequency Eq.(23)
    Jinno & Takimoto (2016) [arXiv:1605.01403]   — JT shape Eq.(67),
                                                    efficiency Eq.(75),
                                                    peak frequency Eq.(74)
    Easther et al. (2008) [arXiv:0801.4197]       — thermal inflation redshift
    """
    freq = np.asarray(freq, dtype=float)

    # Hubble rate during thermal inflation (vacuum dominated)
    H_TI = math.sqrt(V_TI / (3.0 * M_PL**2))

    # Matter-domination dilution factor  R_md < 1
    R_md = (math.pi**2 * g_star * T_d**4 / (30.0 * V_TI)) ** (1.0 / 3.0)

    # Scale factor ratio from decay to today (standard radiation era)
    a_d_over_a0 = (G_S0 / g_star) ** (1.0 / 3.0) * T_CMB_GEV / T_d

    # --- Peak frequency [Hz] ---
    # HK 2008 Eq.(23) vs JT 2016 Eq.(74)
    if shape == "jt2016":
        C_v = 0.35 / (1.0 + 0.069 * v_w + 0.69 * v_w**4)
    else:
        C_v = 0.62 / (1.8 - 0.1 * v_w + v_w**2)
    f_peak = C_v * beta_H * H_TI * R_md * a_d_over_a0 * GEV_TO_HZ

    # --- Peak amplitude ---
    F_GW = 1.67e-5 * (100.0 / g_star) ** (1.0 / 3.0)

    if efficiency == "jt2016":
        Delta = efficiency_jt2016(v_w)
    else:
        Delta = efficiency_envelope(v_w)

    Omega_GW_star = Delta * kappa_phi**2 * (1.0 / beta_H) ** 2
    h2_omega_peak = F_GW * Omega_GW_star * R_md

    # --- Spectral shape ---
    x = freq / f_peak
    if shape == "jt2016":
        S = spectral_shape_jt2016(x)
    else:
        S = spectral_shape_envelope(x)

    omega_gw = h2_omega_peak * S
    return omega_gw, f_peak, h2_omega_peak


# ---------------------------------------------------------------------------
# GW spectrum: MHD turbulence during thermal inflation
# ---------------------------------------------------------------------------
def gw_soundwave_thermal_inflation(
    freq,
    beta_H,
    V_TI,
    DV_wall,
    T_n,
    T_d,
    g_star=G_STAR_DEFAULT,
    v_w=0.5,
):
    r"""GW spectrum from sound waves during thermal inflation.

    Accounts for the non-standard cosmological history:
      vacuum domination → flaton matter domination → radiation at T_d.

    The available energy for sound waves is DV_wall (energy released at the
    bubble wall), NOT V_TI.  The field only crosses the thermal barrier at the
    wall (φ: 0 → φ_escape ~ T), and the bulk of V_TI is released later
    through coherent field rolling (matter-dominated era).

    Parameters
    ----------
    freq : array_like
        Frequency array [Hz].
    beta_H : float
        β/H at nucleation.
    V_TI : float
        Total vacuum energy [GeV⁴] — sets H_TI and R_md.
    DV_wall : float
        Energy released AT THE WALL [GeV⁴] — sets GW source amplitude.
    T_n : float
        Nucleation temperature [GeV].
    T_d : float
        Flaton decay temperature [GeV].
    g_star : float
        Relativistic DOF.
    v_w : float
        Bubble wall velocity (terminal).

    Returns
    -------
    omega_gw : ndarray
        h² Ω_GW(f)
    f_peak : float
        Peak frequency [Hz]
    h2_omega_peak : float
        Peak amplitude
    """
    freq = np.asarray(freq, dtype=float)

    H_TI = math.sqrt(V_TI / (3.0 * M_PL**2))
    R_md = (math.pi**2 * g_star * T_d**4 / (30.0 * V_TI)) ** (1.0 / 3.0)
    a_d_over_a0 = (G_S0 / g_star) ** (1.0 / 3.0) * T_CMB_GEV / T_d

    rho_rad = math.pi**2 / 30.0 * g_star * T_n**4
    alpha_wall = DV_wall / rho_rad
    kv = kappa_v(alpha_wall, v_w)

    K = kv * alpha_wall / (1.0 + alpha_wall)
    Uf = math.sqrt(max(0.75 * K, 1e-300))

    c_s = 1.0 / math.sqrt(3.0)
    z_p = 10.0
    HR_star = (8.0 * math.pi) ** (1.0 / 3.0) / beta_H
    f_peak_star = z_p * beta_H * H_TI / ((8.0 * math.pi) ** (1.0 / 3.0) * c_s)
    f_peak = f_peak_star * R_md * a_d_over_a0 * GEV_TO_HZ

    Gamma = 4.0 / 3.0
    Omega_tilde = 0.01
    Omega_sw_star = 3.0 * Gamma**2 * K**2 * (1.0 / beta_H) * Omega_tilde

    Upsilon = min(1.0, HR_star / Uf) if Uf > 0 else 1.0
    Omega_sw_star *= Upsilon

    F_GW = 1.67e-5 * (100.0 / g_star) ** (1.0 / 3.0)
    h2_omega_peak = F_GW * Omega_sw_star * R_md

    x = freq / f_peak
    S_sw = x**3 * (7.0 / (4.0 + 3.0 * x**2)) ** 3.5
    omega_gw = h2_omega_peak * S_sw

    return omega_gw, f_peak, h2_omega_peak


def gw_turbulence_thermal_inflation(
    freq,
    beta_H,
    V_TI,
    T_d,
    eps_turb=0.05,
    g_star=G_STAR_DEFAULT,
    v_w=1.0,
):
    r"""MHD turbulence contribution to GW during thermal inflation.

    Adapts the Caprini et al. (2016) [arXiv:1512.06239] Eq. (3.9)
    turbulence formula for the non-standard thermal-inflation cosmology
    (vacuum domination → matter-dominated flaton oscillation → radiation
    domination at T_d).

    eps_turb parameterises the fraction of vacuum energy converted to
    MHD turbulence after bubble collisions.  Physically:
      * Standard radiation-dominated: κ_turb ≈ 0.05 κ_v
      * Vacuum-dominated (runaway):   model-dependent, 0.01–0.10

    Parameters
    ----------
    freq : array_like
        Frequency array [Hz].
    beta_H : float
        β/H at nucleation.
    V_TI : float
        Vacuum energy during thermal inflation [GeV⁴].
    T_d : float
        Flaton decay temperature [GeV].
    eps_turb : float
        Fraction of vacuum energy converted to MHD turbulence.
    g_star : float
        Relativistic DOF at T_d.
    v_w : float
        Bubble wall velocity.

    Returns
    -------
    omega_turb : ndarray
        h² Ω_GW(f)
    f_peak : float
        Peak frequency [Hz]
    h2_omega_peak : float
        Peak amplitude
    """
    freq = np.asarray(freq, dtype=float)

    H_TI = math.sqrt(V_TI / (3.0 * M_PL**2))
    R_md = (math.pi**2 * g_star * T_d**4 / (30.0 * V_TI)) ** (1.0 / 3.0)
    a_d_over_a0 = (G_S0 / g_star) ** (1.0 / 3.0) * T_CMB_GEV / T_d

    C_turb = 1.6
    f_peak_est = C_turb / v_w * beta_H * H_TI * R_md * a_d_over_a0 * GEV_TO_HZ
    h_star = H_TI * R_md * a_d_over_a0 * GEV_TO_HZ

    prefactor = (
        3.35e-4
        * (1.0 / beta_H)
        * eps_turb**1.5
        * (100.0 / g_star) ** (1.0 / 3.0)
        * v_w
        * R_md
    )

    x = freq / f_peak_est
    S_turb = x**3 / ((1.0 + x) ** (11.0 / 3.0) * (1.0 + 8.0 * math.pi * freq / h_star))

    omega_turb = prefactor * S_turb

    idx_pk = np.argmax(omega_turb)
    return omega_turb, freq[idx_pk], omega_turb[idx_pk]


# ---------------------------------------------------------------------------
# Detector sensitivity curves  h^2 Omega_sens(f)
# ---------------------------------------------------------------------------
def _h2omega_from_Sh(f, Sh):
    H0 = 100.0 * H_PARAM * 1.0e3 / 3.086e22  # Hz
    return (2.0 * math.pi**2 / 3.0) * f**3 * Sh / H0**2 * H_PARAM**2


def sensitivity_LISA(f):
    f = np.asarray(f, dtype=float)
    L = 2.5e9
    f_star = 3.0e8 / (2.0 * math.pi * L)
    P_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / f) ** 4)
    P_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8.0e-3) ** 4)
    Sc = (
        10.0
        / (3.0 * L**2)
        * (
            P_oms
            + 2.0 * (1.0 + np.cos(f / f_star) ** 2) * P_acc / (2.0 * math.pi * f) ** 4
        )
        * (1.0 + 0.6 * (f / f_star) ** 2)
    )
    return _h2omega_from_Sh(f, Sc)


def sensitivity_DECIGO(f):
    f = np.asarray(f, dtype=float)
    Sh = (
        7.05e-48 * (1.0 + (f / 7.36) ** 2)
        + 4.8e-51 * f ** (-4) / (1.0 + (f / 7.36) ** 2)
        + 5.33e-52 * f ** (-4)
    )
    return _h2omega_from_Sh(f, Sh)


def sensitivity_BBO(f):
    return sensitivity_DECIGO(f) / 100.0


def sensitivity_ET(f):
    f = np.asarray(f, dtype=float)
    x = f / 100.0
    Sh = (
        2.39e-27 * x ** (-15.64)
        + 0.349e-27 * x ** (-2.145)
        + 1.76e-27 * (1.0 + 0.12 * (x ** (-3.01)))
    ) ** 2
    return _h2omega_from_Sh(f, Sh)


def sensitivity_aLIGO(f):
    f = np.asarray(f, dtype=float)
    f0 = 215.0
    x = f / f0
    Sh = 1.0e-49 * (
        x ** (-4.14)
        - 5.0 * x ** (-2)
        + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )
    Sh = np.abs(Sh)
    return _h2omega_from_Sh(f, Sh)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_gw_spectrum(
    freq,
    omega_sw,
    omega_sw_supp,
    omega_turb,
    omega_env,
    alpha_val,
    beta_H,
    HR_star,
    T_n,
    T_RH,
    Uf,
    Upsilon,
    v_w,
    g_star,
    output_path,
    model_params=None,
):
    """Plot h^2 Omega_GW(f) with uncertainty band and detector curves."""
    omega_total = omega_sw + omega_turb + omega_env
    omega_total_supp = omega_sw_supp + omega_turb + omega_env

    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

    # Uncertainty band for total signal
    ax.fill_between(
        freq, omega_total_supp, omega_total, color="royalblue", alpha=0.25, zorder=3
    )
    ax.loglog(freq, omega_total, color="navy", lw=2.5, zorder=5)
    ax.loglog(freq, omega_total_supp, color="navy", lw=1.0, ls=":", alpha=0.6, zorder=5)

    ax.loglog(freq, omega_turb, "r--", lw=1.2, alpha=0.6, zorder=4)
    if np.any(omega_env > 0):
        ax.loglog(freq, omega_env, "g--", lw=1.2, alpha=0.6, zorder=4)

    # Detector sensitivity curves
    f_lisa = np.logspace(-5, -0.5, 500)
    f_decigo = np.logspace(-3, 2, 500)
    f_bbo = np.logspace(-3, 2, 500)
    f_et = np.logspace(0.3, 3.5, 500)
    f_ligo = np.logspace(0.7, 3.7, 500)

    det_curves = [
        (f_lisa, sensitivity_LISA(f_lisa), "purple", "LISA"),
        (f_decigo, sensitivity_DECIGO(f_decigo), "orange", "DECIGO"),
        (f_bbo, sensitivity_BBO(f_bbo), "cyan", "BBO"),
        (f_et, sensitivity_ET(f_et), "brown", "ET"),
        (f_ligo, sensitivity_aLIGO(f_ligo), "green", "aLIGO"),
    ]
    for fv, sv, color, name in det_curves:
        ax.loglog(fv, sv, color=color, lw=1.5, alpha=0.55)

    # --- Annotations ---
    peak_idx = np.argmax(omega_total)
    f_pk = freq[peak_idx]
    o_pk = omega_total[peak_idx]
    ax.annotate(
        "TI PT GW",
        xy=(f_pk, o_pk),
        xytext=(f_pk * 0.04, o_pk * 10),
        fontsize=13,
        fontweight="bold",
        color="navy",
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
    )

    turb_peak_idx = np.argmax(omega_turb)
    f_tl = freq[max(turb_peak_idx - 400, 0)]
    o_tl = omega_turb[max(turb_peak_idx - 400, 0)]
    if o_tl > 1e-25:
        ax.text(
            f_tl,
            o_tl * 3,
            "Turbulence",
            fontsize=9,
            color="red",
            alpha=0.8,
            rotation=30,
        )

    _annotate_detector(ax, f_lisa, sensitivity_LISA(f_lisa), "LISA", "purple", 3e-3)
    _annotate_detector(
        ax, f_decigo, sensitivity_DECIGO(f_decigo), "DECIGO", "orange", 0.2
    )
    _annotate_detector(ax, f_bbo, sensitivity_BBO(f_bbo), "BBO", "cyan", 0.05)
    _annotate_detector(ax, f_et, sensitivity_ET(f_et), "ET", "brown", 5.0)
    _annotate_detector(ax, f_ligo, sensitivity_aLIGO(f_ligo), "aLIGO", "green", 50.0)

    ax.set_xlabel(r"$f$ [Hz]", fontsize=14)
    ax.set_ylabel(r"$h^2 \Omega_{\mathrm{GW}}$", fontsize=14)

    if T_RH >= 1e6:
        trh_str = rf"$T_{{\rm RH}} = {T_RH / 1e6:.2f}$ PeV"
    elif T_RH >= 1e3:
        trh_str = rf"$T_{{\rm RH}} = {T_RH / 1e3:.1f}$ TeV"
    else:
        trh_str = rf"$T_{{\rm RH}} = {T_RH:.0f}$ GeV"

    param_text = (
        rf"$T_n = {T_n:.0f}$ GeV"
        + "\n"
        + trh_str
        + "\n"
        + rf"$\beta/H = {beta_H:.0f}$"
        + "\n"
        + rf"$HR_* = {HR_star:.2e}$"
        + "\n"
        + rf"$\alpha = {alpha_val:.2e}$"
        + "\n"
        + rf"$\Upsilon = {Upsilon:.2e}$"
    )
    if model_params:
        param_text += (
            "\n"
            + rf"$M_S = {model_params['Ms']:.0f}$ GeV"
            + rf", $g_X = {model_params['gX']}$"
            + "\n"
            + rf"$\mu_* = {model_params['mu_star']:.0f}$ GeV"
        )
    ax.text(
        0.02,
        0.97,
        param_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
    )

    ax.set_xlim(1e-5, 1e5)
    ax.set_ylim(1e-20, 1e-5)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("GW Spectrum — DJS (2024) [arXiv:2412.15864]", fontsize=13)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"  Saved figure: {output_path}")
    plt.close(fig)


def _annotate_detector(ax, fv, sv, name, color, f_target):
    idx = np.argmin(np.abs(fv - f_target))
    y_val = sv[idx]
    if y_val < 1e-20 or y_val > 1e-5:
        idx = np.argmin(sv)
        y_val = sv[idx]
    ax.annotate(
        name,
        xy=(fv[idx], y_val),
        xytext=(fv[idx], y_val * 0.15),
        fontsize=10,
        fontweight="bold",
        color=color,
        alpha=0.8,
        ha="center",
    )


# ---------------------------------------------------------------------------
# Plot – EGLPS formula
# ---------------------------------------------------------------------------
def plot_gw_spectrum_eglps(
    freq,
    omega_coll,
    f_peak,
    omega_peak,
    alpha_val,
    beta_H,
    T_n,
    T_d,
    del_V,
    v_w,
    g_star,
    output_path,
    model_params=None,
    title_override=None,
    signal_label=None,
):
    """Plot h^2 Omega_GW(f) using the EGLPS (2008) bubble-collision formula."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

    ax.loglog(freq, omega_coll, color="navy", lw=2.5, zorder=5)

    # Detector sensitivity curves
    f_lisa = np.logspace(-5, -0.5, 500)
    f_decigo = np.logspace(-3, 2, 500)
    f_bbo = np.logspace(-3, 2, 500)
    f_et = np.logspace(0.3, 3.5, 500)
    f_ligo = np.logspace(0.7, 3.7, 500)

    det_curves = [
        (f_lisa, sensitivity_LISA(f_lisa), "purple", "LISA"),
        (f_decigo, sensitivity_DECIGO(f_decigo), "orange", "DECIGO"),
        (f_bbo, sensitivity_BBO(f_bbo), "cyan", "BBO"),
        (f_et, sensitivity_ET(f_et), "brown", "ET"),
        (f_ligo, sensitivity_aLIGO(f_ligo), "green", "aLIGO"),
    ]
    for fv, sv, color, name in det_curves:
        ax.loglog(fv, sv, color=color, lw=1.5, alpha=0.55)

    _sig_label = signal_label or "TI PT GW"
    ax.annotate(
        _sig_label,
        xy=(f_peak, omega_peak),
        xytext=(f_peak * 0.04, omega_peak * 10),
        fontsize=13,
        fontweight="bold",
        color="navy",
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
    )

    _annotate_detector(ax, f_lisa, sensitivity_LISA(f_lisa), "LISA", "purple", 3e-3)
    _annotate_detector(
        ax, f_decigo, sensitivity_DECIGO(f_decigo), "DECIGO", "orange", 0.2
    )
    _annotate_detector(ax, f_bbo, sensitivity_BBO(f_bbo), "BBO", "cyan", 0.05)
    _annotate_detector(ax, f_et, sensitivity_ET(f_et), "ET", "brown", 5.0)
    _annotate_detector(ax, f_ligo, sensitivity_aLIGO(f_ligo), "aLIGO", "green", 50.0)

    ax.set_xlabel(r"$f$ [Hz]", fontsize=14)
    ax.set_ylabel(r"$h^2 \Omega_{\mathrm{GW}}$", fontsize=14)

    V14 = del_V**0.25
    if V14 >= 1e6:
        V14_str = rf"$V_{{\rm TI}}^{{1/4}} = {V14/1e6:.2f}$ PeV"
    elif V14 >= 1e3:
        V14_str = rf"$V_{{\rm TI}}^{{1/4}} = {V14/1e3:.1f}$ TeV"
    else:
        V14_str = rf"$V_{{\rm TI}}^{{1/4}} = {V14:.0f}$ GeV"

    if T_d >= 1e3:
        td_str = rf"$T_d = {T_d/1e3:.1f}$ TeV"
    else:
        td_str = rf"$T_d = {T_d:.0f}$ GeV"

    param_text = (
        rf"$T_n = {T_n:.0f}$ GeV"
        + "\n"
        + V14_str
        + "\n"
        + td_str
        + "\n"
        + rf"$\beta/H = {beta_H:.0f}$"
        + "\n"
        + rf"$\alpha = {alpha_val:.2e}$"
        + "\n"
        + rf"$f_{{\rm peak}} = {f_peak:.2e}$ Hz"
        + "\n"
        + rf"$\Omega_{{\rm peak}} h^2 = {omega_peak:.2e}$"
    )
    if model_params:
        param_text += (
            "\n"
            + rf"$M_S = {model_params['Ms']:.0f}$ GeV"
            + rf", $g_X = {model_params['gX']}$"
            + "\n"
            + rf"$\mu_* = {model_params['mu_star']:.0f}$ GeV"
        )
    ax.text(
        0.02,
        0.97,
        param_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
    )

    ax.set_xlim(1e-5, 1e5)
    ax.set_ylim(1e-25, 1e-5)
    ax.grid(True, which="both", alpha=0.25)
    _title = (
        title_override
        or "GW Spectrum — EGLPS (2008) Bubble Collision [arXiv:0801.4197]"
    )
    ax.set_title(_title, fontsize=13)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"  Saved figure: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-signal overlay plot
# ---------------------------------------------------------------------------
_SIGNAL_COLORS = [
    "navy", "crimson", "forestgreen", "darkorange", "darkviolet",
    "teal", "saddlebrown", "deeppink",
]


def _format_val(val, unit="GeV"):
    """Human-readable value with PeV / TeV / GeV auto-scaling."""
    if unit == "GeV":
        if abs(val) >= 1e6:
            return rf"{val/1e6:.2f}\ \mathrm{{PeV}}"
        if abs(val) >= 1e3:
            return rf"{val/1e3:.1f}\ \mathrm{{TeV}}"
        return rf"{val:.0f}\ \mathrm{{GeV}}"
    return f"{val:.2e}"


def plot_gw_multi(signals, output_path, title=None):
    """Overlay multiple GW spectra on one plot with detector curves.

    Parameters
    ----------
    signals : list of dict
        Each dict must contain:
            freq        : array, Hz
            omega       : array, h^2 Omega_GW
            f_peak      : float, Hz
            omega_peak  : float
            label       : str,  short legend label
            params      : dict  with keys used for the annotation box
                          (T_n, T_d, beta_H, alpha, delV, shape, efficiency, …)
        Optional:
            color       : str  (auto-assigned if omitted)
    output_path : str
    title : str or None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # --- Detector sensitivity curves (drawn first, behind signals) ---
    f_lisa = np.logspace(-5, -0.5, 500)
    f_decigo = np.logspace(-3, 2, 500)
    f_bbo = np.logspace(-3, 2, 500)
    f_et = np.logspace(0.3, 3.5, 500)
    f_ligo = np.logspace(0.7, 3.7, 500)

    det_curves = [
        (f_lisa, sensitivity_LISA(f_lisa), "purple", "LISA"),
        (f_decigo, sensitivity_DECIGO(f_decigo), "orange", "DECIGO"),
        (f_bbo, sensitivity_BBO(f_bbo), "cyan", "BBO"),
        (f_et, sensitivity_ET(f_et), "brown", "ET"),
        (f_ligo, sensitivity_aLIGO(f_ligo), "green", "aLIGO"),
    ]
    for fv, sv, color, name in det_curves:
        ax.loglog(fv, sv, color=color, lw=1.5, alpha=0.45, zorder=2)

    _annotate_detector(ax, f_lisa, sensitivity_LISA(f_lisa), "LISA", "purple", 3e-3)
    _annotate_detector(ax, f_decigo, sensitivity_DECIGO(f_decigo), "DECIGO", "orange", 0.2)
    _annotate_detector(ax, f_bbo, sensitivity_BBO(f_bbo), "BBO", "cyan", 0.05)
    _annotate_detector(ax, f_et, sensitivity_ET(f_et), "ET", "brown", 5.0)
    _annotate_detector(ax, f_ligo, sensitivity_aLIGO(f_ligo), "aLIGO", "green", 50.0)

    # --- Plot each signal ---
    box_y = 0.97
    for i, sig in enumerate(signals):
        c = sig.get("color", _SIGNAL_COLORS[i % len(_SIGNAL_COLORS)])
        freq = sig["freq"]
        omega = sig["omega"]
        label = sig["label"]
        f_pk = sig["f_peak"]
        o_pk = sig["omega_peak"]
        p = sig.get("params", {})

        ax.loglog(freq, omega, color=c, lw=2.2, zorder=5 + i, label=label)

        # Arrow annotation at peak
        ax.annotate(
            label,
            xy=(f_pk, o_pk),
            xytext=(f_pk * 0.04, o_pk * 8),
            fontsize=10,
            fontweight="bold",
            color=c,
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
        )

        # Parameter box
        lines = []
        if "T_n" in p:
            lines.append(rf"$T_n = {_format_val(p['T_n'])}$")
        if "delV" in p:
            V14 = p["delV"] ** 0.25
            lines.append(rf"$V_{{TI}}^{{1/4}} = {_format_val(V14)}$")
        if "T_d" in p:
            lines.append(rf"$T_d = {_format_val(p['T_d'])}$")
        if "beta_H" in p:
            lines.append(rf"$\beta/H = {p['beta_H']:.0f}$")
        if "alpha" in p:
            lines.append(rf"$\alpha = {p['alpha']:.2e}$")
        lines.append(rf"$f_{{pk}} = {f_pk:.2e}$ Hz")
        lines.append(rf"$\Omega_{{pk}} h^2 = {o_pk:.2e}$")
        if "shape" in p:
            lines.append(f"shape: {p['shape']}")
        if "efficiency" in p:
            lines.append(f"eff: {p['efficiency']}")

        param_text = "\n".join(lines)
        ax.text(
            0.02,
            box_y,
            param_text,
            transform=ax.transAxes,
            fontsize=7.5,
            verticalalignment="top",
            color=c,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=c, alpha=0.8),
        )
        n_lines = len(lines)
        box_y -= 0.025 * n_lines + 0.03

    ax.set_xlabel(r"$f$ [Hz]", fontsize=14)
    ax.set_ylabel(r"$h^2 \Omega_{\mathrm{GW}}$", fontsize=14)
    ax.set_xlim(1e-5, 1e5)
    ax.set_ylim(1e-25, 1e-5)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        title or "GW Spectrum — Thermal Inflation (multi-configuration)", fontsize=13
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"  Saved figure: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute GW spectrum from tunneling CSV data."
    )
    parser.add_argument("csv_path", help="Path to CSV with T and S3/T columns")
    parser.add_argument(
        "--formula",
        type=str,
        default="TI",
        choices=["TI", "DJS", "EGLPS"],
        help=(
            "GW formula set: "
            "TI = Thermal Inflation first-principles (bubble collision, default), "
            "DJS = Dutka-Jung-Shin (2024, sw+turb+env), "
            "EGLPS = Easther+ (2008, legacy envelope)"
        ),
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="jt2016",
        choices=["envelope", "jt2016"],
        help=(
            "Spectral shape (TI formula): "
            "envelope = HK (2008) Eq.(19), "
            "jt2016 = Jinno & Takimoto (2016) Eq.(67), default"
        ),
    )
    parser.add_argument(
        "--efficiency",
        type=str,
        default="jt2016",
        choices=["envelope", "jt2016"],
        help=(
            "GW production efficiency (TI formula): "
            "envelope = HK (2008) Eq.(22), "
            "jt2016 = JT (2016) Eq.(75), default"
        ),
    )
    parser.add_argument(
        "--vw", type=float, default=1.0, help="Bubble wall velocity (default: 1.0)"
    )
    parser.add_argument(
        "--g_star",
        type=float,
        default=G_STAR_DEFAULT,
        help="Relativistic DOF (default: 106.75)",
    )
    parser.add_argument(
        "--delV",
        type=float,
        default=DEL_V_DEFAULT,
        help="Vacuum energy difference in GeV^4 (default: 2.5e27)",
    )
    parser.add_argument(
        "--T_d",
        type=float,
        default=None,
        help="Flaton decay temperature in GeV (TI/EGLPS formula, default: T_RH)",
    )
    parser.add_argument(
        "--T_RH",
        type=float,
        default=None,
        help="Reheating temperature in GeV (overrides delV-based computation)",
    )
    parser.add_argument(
        "--Ms",
        type=float,
        default=None,
        help="Soft SUSY breaking scale M_S in GeV (used with --gX, --mu_star)",
    )
    parser.add_argument(
        "--gX",
        type=float,
        default=None,
        help="Coupling g_X (used with --Ms, --mu_star)",
    )
    parser.add_argument(
        "--mu_star",
        type=float,
        default=None,
        help="Renormalization scale mu_* in GeV (used with --Ms, --gX)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument(
        "--csv_output", type=str, default=None, help="Output CSV path for GW parameters"
    )

    args = parser.parse_args()

    # Resolve model parameters if given
    has_model_params = (
        args.Ms is not None and args.gX is not None and args.mu_star is not None
    )
    if has_model_params:
        delV = compute_delV_model(args.Ms, args.gX, args.mu_star)
    else:
        delV = args.delV

    formula = args.formula

    print("=" * 70)
    print("GW Spectrum from Thermal Inflation Phase Transition")
    print(f"  Formula: {formula}")
    if formula == "TI":
        print("  Source: bubble collisions only (vacuum-dominated, no sw/turb)")
        print(f"  Spectral shape:  {args.shape}")
        print(f"  Efficiency:      {args.efficiency}")
        print("  Redshift: matter-dominated (flaton oscillation) + radiation")
    elif formula == "DJS":
        print("  Sound waves: Hindmarsh et al. (2017), arXiv:2412.15864 eq.(4.31)")
        print("  Suppression: Ellis, Lewicki, No (2019)")
    else:
        print("  Bubble collision: Easther et al. (2008), arXiv:0801.4197 (legacy)")
        print("  Envelope approx: Kamionkowski, Kosowsky & Turner (1994)")
    print("=" * 70)
    print(f"  Input CSV:  {args.csv_path}")
    print(f"  v_w = {args.vw},  g* = {args.g_star},  delV = {delV:.2e} GeV^4")
    if has_model_params:
        print(
            f"  Model: M_S = {args.Ms:.1f} GeV,  g_X = {args.gX},  "
            f"mu_* = {args.mu_star:.1f} GeV"
        )
    print()

    # ---- Load tunneling data ----
    T, S3_T = load_tunneling_data(args.csv_path)
    print(f"  Loaded {len(T)} data points")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] GeV")
    print(f"  S3/T range: [{S3_T.min():.2f}, {S3_T.max():.2f}]")
    print()

    # ---- Reheating temperature (priority: --T_RH > model params > delV) ----
    if args.T_RH is not None:
        T_RH = args.T_RH
        trh_source = "user-specified"
    elif has_model_params:
        T_RH = compute_T_RH_model(args.Ms, args.gX, args.mu_star, args.g_star)
        trh_source = "model params eq.(4.9)"
    else:
        T_RH = compute_T_RH(delV, args.g_star)
        trh_source = "delV"
    print("--- Reheating temperature ---")
    print(f"  T_RH = {T_RH:.4e} GeV  ({T_RH / 1e6:.4f} PeV = {T_RH / 1e3:.2f} TeV)")
    print(f"  source: {trh_source}")
    print()

    # ---- Nucleation temperature ----
    print("--- Nucleation ---")
    T_n = find_nucleation_temp(T, S3_T, delV)
    print(f"  T_n = {T_n:.2f} GeV  ({T_n / 1e3:.4f} TeV)")
    print(f"  T_RH / T_n = {T_RH / T_n:.1f}")

    # ---- beta / H ----
    beta_H = compute_beta_over_H(T, S3_T, T_n)
    print(f"  beta/H = {beta_H:.2f}  (-d ln Gamma / d ln T)")

    cs_s = CubicSpline(T, S3_T)
    dS3T_dT_at_Tn = float(cs_s(T_n, 1))
    beta_H_approx = T_n * dS3T_dT_at_Tn
    print(f"  beta/H = {beta_H_approx:.2f}  (T d(S3/T)/dT)")

    # HR_*
    HR_star = compute_HR_star(beta_H)
    print(f"  HR_* = (8pi)^{{1/3}} / (beta/H) = {HR_star:.4e}")

    # alpha
    alpha_val = compute_alpha(T_n, delV, args.g_star)
    print(f"  alpha = {alpha_val:.4e}")
    print()

    model_params = None
    if has_model_params:
        model_params = {"Ms": args.Ms, "gX": args.gX, "mu_star": args.mu_star}

    csv_name = os.path.splitext(os.path.basename(args.csv_path))[0]

    # ==================================================================
    #  TI formula  (first-principles thermal inflation)
    # ==================================================================
    if formula == "TI":
        T_d = args.T_d if args.T_d is not None else T_RH

        print("--- Thermal Inflation GW parameters ---")
        print(f"  T_d (flaton decay) = {T_d:.2f} GeV  ({T_d / 1e3:.4f} TeV)")
        print(f"  V_TI^{{1/4}}        = {delV**0.25:.4e} GeV")

        H_TI = math.sqrt(delV / (3.0 * M_PL**2))
        R_md = (math.pi**2 * args.g_star * T_d**4 / (30.0 * delV)) ** (1.0 / 3.0)
        print(f"  H_TI (vacuum)     = {H_TI:.4e} GeV")
        print(f"  R_md (dilution)   = {R_md:.4e}")
        print(f"  Shape: {args.shape},  Efficiency: {args.efficiency}")
        print()

        freq = np.logspace(-5, 5, 3000)
        omega_coll, f_peak, omega_peak = gw_thermal_inflation(
            freq,
            beta_H,
            delV,
            T_d,
            g_star=args.g_star,
            v_w=args.vw,
            shape=args.shape,
            efficiency=args.efficiency,
        )

        print("--- TI GW peak ---")
        print(f"  f_peak    = {f_peak:.4e} Hz")
        print(f"  h^2 Omega = {omega_peak:.4e}")
        print()

        # Cross-check with EGLPS legacy
        f_eglps = eglps_peak_freq(beta_H, delV, T_d)
        o_eglps = eglps_peak_amplitude(beta_H, delV, T_d)
        print("--- Cross-check: EGLPS (2008) legacy ---")
        print(f"  f_peak (EGLPS)    = {f_eglps:.4e} Hz  (ratio: {f_peak/f_eglps:.2f})")
        print(f"  h^2 Omega (EGLPS) = {o_eglps:.4e}  (ratio: {omega_peak/o_eglps:.2f})")
        print()

        delV_sci = "%e" % float(delV)
        fig_path = (
            args.output
            or f"figs/gw_spectrum/gw_spectrum_TI_{csv_name}_T_d_{T_d:.0f}_{delV_sci}.png"
        )

        title = (
            f"GW Spectrum — Thermal Inflation (bubble collision)\n"
            f"Shape: {args.shape}, Efficiency: {args.efficiency}"
        )

        plot_gw_spectrum_eglps(
            freq,
            omega_coll,
            f_peak,
            omega_peak,
            alpha_val,
            beta_H,
            T_n,
            T_d,
            delV,
            args.vw,
            args.g_star,
            fig_path,
            model_params,
            title_override=title,
            signal_label="TI bubble collision",
        )

        csv_out = args.csv_output or os.path.join(
            os.path.dirname(args.csv_path), f"gw_parameters_TI.csv"
        )
        params_df = pd.DataFrame(
            [
                {
                    "formula": "TI",
                    "shape": args.shape,
                    "efficiency": args.efficiency,
                    "T_n_GeV": T_n,
                    "T_RH_GeV": T_RH,
                    "T_d_GeV": T_d,
                    "beta_over_H_general": beta_H,
                    "beta_over_H_approx": beta_H_approx,
                    "HR_star": HR_star,
                    "alpha": alpha_val,
                    "v_w": args.vw,
                    "g_star": args.g_star,
                    "delV_GeV4": delV,
                    "V_TI_14_GeV": delV**0.25,
                    "H_TI_GeV": H_TI,
                    "R_md": R_md,
                    "f_peak_Hz": f_peak,
                    "h2_Omega_peak": omega_peak,
                    "f_peak_EGLPS_Hz": f_eglps,
                    "h2_Omega_EGLPS": o_eglps,
                }
            ]
        )
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        params_df.to_csv(csv_out, index=False)
        print(f"  Saved parameters: {csv_out}")

    # ==================================================================
    #  EGLPS formula  (arXiv:0801.4197) — legacy
    # ==================================================================
    elif formula == "EGLPS":
        T_d = args.T_d if args.T_d is not None else T_RH

        print("--- EGLPS parameters ---")
        print(f"  T_d (flaton decay) = {T_d:.2f} GeV")
        print(f"  V_TI^{{1/4}}        = {delV**0.25:.4e} GeV")
        print()

        freq = np.logspace(-5, 5, 3000)
        omega_coll, f_peak, omega_peak = gw_eglps(freq, beta_H, delV, T_d)

        print("--- EGLPS peak ---")
        print(f"  f_peak   = {f_peak:.4e} Hz")
        print(f"  Omega h^2 = {omega_peak:.4e}")
        print()

        delV_sci = "%e" % float(delV)
        fig_path = (
            args.output
            or f"figs/gw_spectrum/gw_spectrum_EGLPS_{csv_name}_T_d_{args.T_d}_{delV_sci}.png"
        )

        plot_gw_spectrum_eglps(
            freq,
            omega_coll,
            f_peak,
            omega_peak,
            alpha_val,
            beta_H,
            T_n,
            T_d,
            delV,
            args.vw,
            args.g_star,
            fig_path,
            model_params,
        )

        # Save parameters CSV
        csv_out = args.csv_output or os.path.join(
            os.path.dirname(args.csv_path), f"gw_parameters_EGLPS.csv"
        )
        params_df = pd.DataFrame(
            [
                {
                    "formula": "EGLPS",
                    "T_n_GeV": T_n,
                    "T_RH_GeV": T_RH,
                    "T_d_GeV": T_d,
                    "beta_over_H_general": beta_H,
                    "beta_over_H_approx": beta_H_approx,
                    "HR_star": HR_star,
                    "alpha": alpha_val,
                    "v_w": args.vw,
                    "g_star": args.g_star,
                    "delV_GeV4": delV,
                    "V_TI_14_GeV": delV**0.25,
                    "f_peak_Hz": f_peak,
                    "h2_Omega_peak": omega_peak,
                }
            ]
        )
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        params_df.to_csv(csv_out, index=False)
        print(f"  Saved parameters: {csv_out}")

    # ==================================================================
    #  DJS formula  (arXiv:2412.15864) – default
    # ==================================================================
    else:
        kv = kappa_v(alpha_val)
        kt = kappa_turb_frac() * kv
        kphi = 0.0
        if alpha_val > 10:
            kphi = max(0.0, 1.0 - kv)

        Uf = compute_Uf(kv, alpha_val)
        Upsilon = min(1.0, HR_star / Uf)

        print("--- Efficiency & fluid ---")
        print(f"  kappa_v    = {kv:.4f}")
        print(f"  kappa_turb = {kt:.4f}")
        print(f"  kappa_phi  = {kphi:.4f}")
        print(f"  Uf (RMS)   = {Uf:.4f}")
        print(f"  Upsilon    = {Upsilon:.4e}  (suppression factor)")
        print()

        fp_sw = f_peak_sw(HR_star, T_RH, args.g_star)
        fp_env = f_peak_env(beta_H, T_RH, args.g_star, args.vw)
        fp_turb = f_peak_turb(beta_H, T_RH, args.g_star, args.vw)

        print("--- Peak frequencies (T_RH redshift) ---")
        print(f"  f_peak (sound wave)  = {fp_sw:.4e} Hz")
        print(f"  f_peak (turbulence)  = {fp_turb:.4e} Hz")
        print(f"  f_peak (envelope)    = {fp_env:.4e} Hz")
        print()

        freq = np.logspace(-5, 5, 3000)

        omega_sw = gw_sound_wave(freq, HR_star, Uf, args.g_star, fp_sw)
        omega_sw_supp = omega_sw * Upsilon

        omega_env = gw_envelope(freq, alpha_val, beta_H, args.g_star, kphi, fp_env)
        omega_turb = gw_turbulence(
            freq, alpha_val, beta_H, T_RH, args.g_star, args.vw, kt, fp_turb
        )

        omega_total = omega_sw + omega_turb + omega_env
        omega_total_supp = omega_sw_supp + omega_turb + omega_env

        peak_total = np.max(omega_total)
        f_at_peak = freq[np.argmax(omega_total)]
        peak_supp = np.max(omega_total_supp)
        f_at_peak_supp = freq[np.argmax(omega_total_supp)]

        print("--- Peak amplitude ---")
        print(
            f"  h^2 Omega (nominal)    = {peak_total:.4e}  at  f = {f_at_peak:.4e} Hz"
        )
        print(
            f"  h^2 Omega (suppressed) = {peak_supp:.4e}  at  f = {f_at_peak_supp:.4e} Hz"
        )
        print()

        fig_path = args.output or f"figs/gw_spectrum/gw_spectrum_DJS_{csv_name}.png"

        plot_gw_spectrum(
            freq,
            omega_sw,
            omega_sw_supp,
            omega_turb,
            omega_env,
            alpha_val,
            beta_H,
            HR_star,
            T_n,
            T_RH,
            Uf,
            Upsilon,
            args.vw,
            args.g_star,
            fig_path,
            model_params,
        )

        # Save parameters CSV
        csv_out = args.csv_output or os.path.join(
            os.path.dirname(args.csv_path), f"gw_parameters_DJS.csv"
        )
        params_df = pd.DataFrame(
            [
                {
                    "formula": "DJS",
                    "T_n_GeV": T_n,
                    "T_RH_GeV": T_RH,
                    "beta_over_H_general": beta_H,
                    "beta_over_H_approx": beta_H_approx,
                    "HR_star": HR_star,
                    "alpha": alpha_val,
                    "Uf": Uf,
                    "Upsilon": Upsilon,
                    "v_w": args.vw,
                    "g_star": args.g_star,
                    "delV_GeV4": delV,
                    "kappa_v": kv,
                    "kappa_turb": kt,
                    "kappa_phi": kphi,
                    "f_peak_sw_Hz": fp_sw,
                    "f_peak_turb_Hz": fp_turb,
                    "f_peak_env_Hz": fp_env,
                    "h2_Omega_peak_nominal": peak_total,
                    "h2_Omega_peak_suppressed": peak_supp,
                    "f_at_peak_Hz": f_at_peak,
                }
            ]
        )
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        params_df.to_csv(csv_out, index=False)
        print(f"  Saved parameters: {csv_out}")

    print()
    print("=" * 70)
    print("Done.")
    print("=" * 70)


def _compute_one_signal(csv_path, delV, T_d, g_star, v_w, shape, efficiency):
    """Load CSV, compute nucleation params, return signal dict for plot_gw_multi."""
    T, S3_T = load_tunneling_data(csv_path)
    T_RH = compute_T_RH(delV, g_star)
    T_n = find_nucleation_temp(T, S3_T, delV)
    beta_H = compute_beta_over_H(T, S3_T, T_n)
    alpha_val = compute_alpha(T_n, delV, g_star)
    freq = np.logspace(-5, 5, 3000)
    omega, f_pk, h2_pk = gw_thermal_inflation(
        freq, beta_H, delV, T_d if T_d is not None else T_RH,
        g_star=g_star, v_w=v_w, shape=shape, efficiency=efficiency,
    )
    return dict(
        freq=freq, omega=omega, f_peak=f_pk, omega_peak=h2_pk,
        params=dict(
            T_n=T_n, T_d=T_d if T_d is not None else T_RH,
            beta_H=beta_H, alpha=alpha_val, delV=delV,
            shape=shape, efficiency=efficiency,
        ),
    )


def main_compare():
    """Overlay multiple GW spectra on one plot.

    Usage (JSON config file):
        python analysis/gwSpectrum.py compare config.json

    JSON format::

        {
            "output": "figs/gw_multi.png",
            "title":  "optional title",
            "signals": [
                {
                    "csv":        "data/tunneling/set7/T-S_....csv",
                    "delV":       2.5e27,
                    "T_d":        1000,
                    "label":      "set 7",
                    "color":      "navy",
                    "g_star":     106.75,
                    "v_w":        1.0,
                    "shape":      "jt2016",
                    "efficiency": "jt2016"
                },
                { ... },
                { ... }
            ]
        }

    Alternatively, specify signals directly on the command line:
        python analysis/gwSpectrum.py compare \\
            --csv data/set7.csv --delV 2.5e27 --T_d 1000 --label "set 7" \\
            --csv data/set8.csv --delV 2.5e35 --T_d 1000 --label "set 8" \\
            --output figs/gw_multi.png
    """
    import json as _json

    parser = argparse.ArgumentParser(
        description="Overlay multiple TI GW spectra on one plot."
    )
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Path to JSON config file (alternative to --csv flags)",
    )
    parser.add_argument("--csv", action="append", default=[], help="CSV path (repeatable)")
    parser.add_argument("--delV", action="append", type=float, default=[], help="delV (repeatable)")
    parser.add_argument("--T_d", action="append", type=float, default=[], help="T_d (repeatable)")
    parser.add_argument("--label", action="append", default=[], help="Label (repeatable)")
    parser.add_argument("--color", action="append", default=[], help="Color (repeatable, optional)")
    parser.add_argument("--shape", type=str, default="jt2016")
    parser.add_argument("--efficiency", type=str, default="jt2016")
    parser.add_argument("--g_star", type=float, default=G_STAR_DEFAULT)
    parser.add_argument("--vw", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="figs/gw_spectrum/gw_multi.png")
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    signals = []

    if args.config is not None:
        with open(args.config) as fh:
            cfg = _json.load(fh)
        output_path = cfg.get("output", args.output)
        title = cfg.get("title", args.title)
        for s in cfg["signals"]:
            sig = _compute_one_signal(
                csv_path=s["csv"],
                delV=s.get("delV", DEL_V_DEFAULT),
                T_d=s.get("T_d", None),
                g_star=s.get("g_star", G_STAR_DEFAULT),
                v_w=s.get("v_w", 1.0),
                shape=s.get("shape", "jt2016"),
                efficiency=s.get("efficiency", "jt2016"),
            )
            sig["label"] = s.get("label", os.path.basename(s["csv"]))
            if "color" in s:
                sig["color"] = s["color"]
            signals.append(sig)
    else:
        output_path = args.output
        title = args.title
        n = len(args.csv)
        if n == 0:
            parser.error("Provide either a JSON config or --csv flags.")
        labels = args.label if len(args.label) == n else [f"signal {i+1}" for i in range(n)]
        delVs = args.delV if len(args.delV) == n else [DEL_V_DEFAULT] * n
        T_ds = args.T_d if len(args.T_d) == n else [None] * n
        colors = args.color if len(args.color) == n else []

        for i in range(n):
            sig = _compute_one_signal(
                csv_path=args.csv[i],
                delV=delVs[i],
                T_d=T_ds[i],
                g_star=args.g_star,
                v_w=args.vw,
                shape=args.shape,
                efficiency=args.efficiency,
            )
            sig["label"] = labels[i]
            if i < len(colors):
                sig["color"] = colors[i]
            signals.append(sig)

    print(f"Plotting {len(signals)} signals → {output_path}")
    plot_gw_multi(signals, output_path, title=title)
    print("Done.")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "compare":
        _sys.argv.pop(1)
        main_compare()
    else:
        main()
