#!/usr/bin/env python3
"""
computeTn.py  —  Nucleation temperature T_n as a function of γ = φ₀/M_Pl.

Uses the Gaussian-ansatz bounce action (Eqs. 42-55 of tiend_3_6.pdf)
and the nucleation condition:

    C(T, γ) = S₃/T − 4 ln(2√3 T/(m γ)) = 0

Corrections applied to the PDF:
    • H = m γ/(2√3)       [Eq. 64 corrected]
    • C(T_n) = 0           [Eq. 66 corrected]
    • ε replaces γ in Eq. 63 to avoid symbol clash with Eq. 77

Key insight:  S₃/T depends only on {m, y, n_f, g, δ²} — independent of γ.
T_n(γ) is determined solely by where S₃/T(T) crosses 4 ln(2√3 T/(m γ)).

Also implements the semi-analytical linearisation from Eqs. 67-71 / C14-C18.
"""

import math
import os
import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
#  Physical constants & model parameters
# ═══════════════════════════════════════════════════════════════════
M_PL = 2.4e18  # reduced Planck mass [GeV]
m = 1000.0  # flaton mass [GeV]
y = 1.09  # Yukawa coupling
g = 1.05  # gauge coupling
n_f = 20  # fermion d.o.f.
delta2 = g**2 / 6.0  # δ² ≈ 0.1838

# Gaussian-quadrature nodes & weights (Appendix A of PDF)
QX = np.array([0.0320224, 0.339406])
QW = np.array([0.900848, 0.0991516])

GAMMA_REF = 1.0e11 / M_PL  # reference γ ≈ 4.1667e-8

# ═══════════════════════════════════════════════════════════════════
#  Fermionic thermal integral J'_F(u²) via cubic-spline cache
# ═══════════════════════════════════════════════════════════════════
def _jfp_integrand(x, u2):
    """Integrand for J'_F(u²) = 1/(4π²) ∫ dx x² / (√(u²+x²)(e^{√(…)}+1))."""
    s = math.sqrt(u2 + x * x)
    if s > 500:
        return 0.0
    return x * x / (s * (math.exp(s) + 1.0))


def _jfp_exact(u2):
    val, _ = integrate.quad(
        _jfp_integrand, 0, 200, args=(u2,),
        limit=500, epsabs=1e-15, epsrel=1e-15
    )
    return val / (4.0 * math.pi**2)


print("Building J'_F spline on [0, 60] …", end=" ", flush=True)
_U2_GRID = np.concatenate(
    [np.linspace(0, 5, 3000), np.linspace(5.01, 60, 3000)]
)
_JFP_VALS = np.array([_jfp_exact(u2) for u2 in _U2_GRID])
_SP = CubicSpline(_U2_GRID, _JFP_VALS)
print("done.")


def Jp(u2):
    """J'_F(u²) via spline."""
    if u2 > 60:
        return 0.0
    return float(_SP(u2))


def Jpp(u2):
    """J''_F(u²) = d/d(u²) J'_F."""
    if u2 > 60:
        return 0.0
    return float(_SP(u2, 1))


def Jppp(u2):
    """J'''_F(u²) = d²/d(u²)² J'_F."""
    if u2 > 60:
        return 0.0
    return float(_SP(u2, 2))


# ═══════════════════════════════════════════════════════════════════
#  Spinodal temperature T_c2 (Eq. 76)
# ═══════════════════════════════════════════════════════════════════
T_c2 = m / (y * math.sqrt(n_f * Jp(delta2)))
print(f"T_c2 = {T_c2:.4f} GeV  ({T_c2 / 1e3:.6f} TeV)")

# ═══════════════════════════════════════════════════════════════════
#  α(T) from Eq. (52)  and  S₃/T from Eq. (55)
# ═══════════════════════════════════════════════════════════════════
def _Gi(alpha, i):
    """G_i(α) ≡ J'_F(z) − (x_i α / 2) J''_F(z),  z = δ² + x_i α.  [Eq. C7]"""
    z = delta2 + QX[i] * alpha
    return Jp(z) - QX[i] * alpha / 2.0 * Jpp(z)


def _f_sum(alpha):
    """Σ_i w_i G_i(α)   (without n_f factor)."""
    return QW[0] * _Gi(alpha, 0) + QW[1] * _Gi(alpha, 1)


_F0 = n_f * _f_sum(0.0)  # = n_f J'_F(δ²), threshold for T > T_c2


def solve_alpha(T):
    """Solve  m²/(y²T²) = n_f Σ w_i G_i(α)  for α ≥ 0.  Returns α or None."""
    lhs = m**2 / (y**2 * T**2)
    if lhs >= _F0:
        return None  # T ≤ T_c2

    def residual(a):
        return lhs - n_f * _f_sum(a)

    hi = 1.0
    while residual(hi) < 0 and hi < 500:
        hi *= 2
    if residual(hi) < 0:
        return None
    return optimize.brentq(residual, 0.0, hi, xtol=1e-12)


def S3_over_T(alpha):
    """S₃/T from Eq. (55)."""
    denom = sum(QW[i] * QX[i] * Jpp(delta2 + QX[i] * alpha) for i in range(2))
    arg = -2.0 * alpha / (3.0 * y**2 * n_f * denom)
    if arg <= 0:
        return None
    return 2.0 * math.pi / (3.0 * y**2) * math.sqrt(arg)


# ═══════════════════════════════════════════════════════════════════
#  Pre-compute S₃/T(T) on a fine temperature grid
# ═══════════════════════════════════════════════════════════════════
_T_LO = T_c2 + 0.5
_T_HI = T_c2 + 1000.0
_N_T = 6000
_T_GRID = np.linspace(_T_LO, _T_HI, _N_T)
_A_GRID = np.full(_N_T, np.nan)
_S_GRID = np.full(_N_T, np.nan)

print(
    f"Pre-computing α(T) and S₃/T on "
    f"[{_T_LO:.1f}, {_T_HI:.1f}] GeV …",
    end=" ", flush=True,
)
for k, T_val in enumerate(_T_GRID):
    a = solve_alpha(T_val)
    if a is not None:
        _A_GRID[k] = a
        s = S3_over_T(a)
        if s is not None:
            _S_GRID[k] = s
_valid = ~np.isnan(_S_GRID)
_S3T_spl = CubicSpline(_T_GRID[_valid], _S_GRID[_valid])
_ALPHA_spl = CubicSpline(_T_GRID[_valid], _A_GRID[_valid])
print(f"done ({np.sum(_valid)} valid points).")

# ═══════════════════════════════════════════════════════════════════
#  Validate against Table II
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─' * 80}")
print(
    f"  Validation vs Table II  "
    f"(temperatures in TeV, γ_ref = {GAMMA_REF:.6e})"
)
print(f"{'─' * 80}")
print(
    f"  {'T':>5}  {'α':>10}  {'S₃/T':>10}  "
    f"{'4ln(2√3T/(mγ))':>16}  {'C(T)':>10}"
)
print(f"{'─' * 80}")
for T_TeV in np.arange(1.50, 1.66, 0.01):
    T_GeV = T_TeV * 1e3
    a = solve_alpha(T_GeV)
    if a is None:
        print(
            f"  {T_TeV:5.2f}  {'—':>10}  {'—':>10}"
            f"  {'—':>16}  {'—':>10}"
        )
        continue
    s3t = S3_over_T(a)
    log4 = 4.0 * math.log(
        2 * math.sqrt(3) * T_GeV / (m * GAMMA_REF)
    )
    C_val = s3t - log4 if s3t else None
    s3_str = f"{s3t:.5f}" if s3t else "—"
    c_str = f"{C_val:.5f}" if C_val is not None else "—"
    print(f"  {T_TeV:5.2f}  {a:10.5f}  {s3_str:>10}  {log4:16.5f}  {c_str:>10}")

# ═══════════════════════════════════════════════════════════════════
#  Fast T_n finder using pre-computed S₃/T spline
# ═══════════════════════════════════════════════════════════════════
def find_Tn(gamma):
    """Brute-force:  find T_n such that S₃/T = 4 ln(2√3 T/(m γ))."""
    T_v = _T_GRID[_valid]
    S_v = _S_GRID[_valid]
    rhs = 4.0 * np.log(2 * np.sqrt(3) * T_v / (m * gamma))
    diff = S_v - rhs
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return np.nan
    idx = sc[0]

    def _res(T):
        return float(_S3T_spl(T)) - 4.0 * math.log(
            2 * math.sqrt(3) * T / (m * gamma)
        )

    return optimize.brentq(_res, T_v[idx], T_v[idx + 1], xtol=1e-3)


# ═══════════════════════════════════════════════════════════════════
#  Semi-analytical approach (Eqs. 67-71 / C14-C18)
# ═══════════════════════════════════════════════════════════════════
def linearisation_coeffs(alpha_ref):
    """c₀, c₁, s₀, s₁ from Eqs. (68-71) evaluated at α_ref."""
    c0 = c1 = s0 = s1 = 0.0
    for i in range(2):
        z = delta2 + QX[i] * alpha_ref
        jp = Jp(z)
        jpp = Jpp(z)
        jppp = Jppp(z)
        x = QX[i]
        w = QW[i]
        c0 += w * (jp - x * alpha_ref * jpp + x**2 * alpha_ref**2 / 2 * jppp)
        c1 += w * (x / 2 * jpp - x**2 * alpha_ref / 2 * jppp)
        s0 += w * (x * jpp - alpha_ref * x**2 * jppp)
        s1 += w * x**2 * jppp
    return c0, c1, s0, s1


def find_Tn_semi(gamma, c0, c1, s0, s1):
    """Solve the linearised nucleation equation (C18) for T_n."""

    def residual(T):
        alpha = (m**2 / (n_f * y**2 * T**2) - c0) / c1
        if alpha <= 0:
            return -1e10
        denom = s0 + s1 * alpha
        inside = -2 * alpha / (3 * y**2 * n_f * denom)
        if inside <= 0:
            return -1e10
        s3t = 2 * math.pi / (3 * y**2) * math.sqrt(inside)
        return s3t - 4 * math.log(2 * math.sqrt(3) * T / (m * gamma))

    T_arr = np.linspace(_T_LO, _T_HI, 4000)
    res = np.array([residual(T) for T in T_arr])
    sc = np.where(np.diff(np.sign(res)))[0]
    if len(sc) == 0:
        return np.nan
    idx = sc[0]
    return optimize.brentq(residual, T_arr[idx], T_arr[idx + 1], xtol=1e-3)


# ─── Reference point ─────────────────────────────────────────────
Tn_ref = find_Tn(GAMMA_REF)
alpha_ref = float(_ALPHA_spl(Tn_ref))
c0, c1, s0, s1 = linearisation_coeffs(alpha_ref)

print(f"\n{'─' * 80}")
print(
    f"  Reference:  γ_ref = {GAMMA_REF:.6e}"
    f"   (φ₀ = {GAMMA_REF * M_PL:.2e} GeV)"
)
print(f"  T_n(γ_ref) = {Tn_ref:.4f} GeV   ({Tn_ref / 1e3:.6f} TeV)")
print(f"  α_n        = {alpha_ref:.5f}")
print(f"  c₀ = {c0:.10e}   c₁ = {c1:.10e}")
print(f"  s₀ = {s0:.10e}   s₁ = {s1:.10e}")
print(f"{'─' * 80}")

# ═══════════════════════════════════════════════════════════════════
#  Perturbative approach:  γ → εγ₀,  T_n → T_n⁰ + δT
#
#  Expand C(T_n⁰+δT, εγ₀) = 0 to 1st and 2nd order in δT.
#  Define F(T) ≡ S₃/T.  Taylor-expanding both sides and cancelling
#  the reference condition F(T_n⁰) = 4 ln(2√3 T_n⁰/(mγ₀)) yields:
#
#    a (δT)² + b (δT) + 4 ln ε = 0
#
#  with  b = F' − 4/T_n⁰,   a = F''/2 + 2/(T_n⁰)².
# ═══════════════════════════════════════════════════════════════════
Fp = float(_S3T_spl(Tn_ref, 1))    # dF/dT
Fpp = float(_S3T_spl(Tn_ref, 2))   # d²F/dT²
Fppp = float(_S3T_spl(Tn_ref, 3))  # d³F/dT³
lng0 = math.log(GAMMA_REF)

# Coefficients of the full cubic:
#   d(δT)³ + a(δT)² + b(δT) + 4 ln ε = 0
pt_b = Fp - 4.0 / Tn_ref
pt_a = 0.5 * Fpp + 2.0 / Tn_ref**2
pt_d = Fppp / 6.0 - 4.0 / (3.0 * Tn_ref**3)

# ── Semi-analytical polynomial coefficients in ln γ ──────────
# Perturbative inversion: δT = Σ δT_k c^k, c = 4 ln(γ/γ₀)
#   δT₁ = -1/b
#   δT₂ = -a/b³
#   δT₃ = -(2a² - bd)/b⁵
# Then T_n = T_n⁰ + δT₁·4(L-L₀) + δT₂·16(L-L₀)² + δT₃·64(L-L₀)³
# where L = lnγ, L₀ = lnγ₀.

_c1 = -4.0 / pt_b                              # coeff of (L-L₀)
_c2 = -16.0 * pt_a / pt_b**3                   # coeff of (L-L₀)²
_c3 = -64.0 * (2 * pt_a**2 - pt_b * pt_d) / pt_b**5  # coeff of (L-L₀)³

# Expand (L-L₀)^n in powers of L to get T_n = A₀+A₁L+A₂L²+A₃L³
pt_A3 = _c3
pt_A2 = _c2 - 3 * _c3 * lng0
pt_A1 = _c1 - 2 * _c2 * lng0 + 3 * _c3 * lng0**2
pt_A0 = Tn_ref - _c1 * lng0 + _c2 * lng0**2 - _c3 * lng0**3

print(f"\n{'─' * 80}")
print("  Perturbative expansion around (γ₀, T_n⁰):")
print(f"  F'   = {Fp:.10e}   F'' = {Fpp:.10e}   F''' = {Fppp:.10e}")
print(f"  b = {pt_b:.10e}   a = {pt_a:.10e}   d = {pt_d:.10e}")
print(f"\n  Semi-analytical polynomial  (3rd order):")
print(
    f"  T_n ≈ {pt_A0:.4f} + ({pt_A1:.6f}) lnγ"
    f" + ({pt_A2:.8f}) (lnγ)²"
    f" + ({pt_A3:.10f}) (lnγ)³   [GeV]"
)
print(f"{'─' * 80}")


def Tn_pert_1st(gamma):
    """1st-order:  δT = −4 ln(γ/γ₀) / b."""
    ln_eps = math.log(gamma) - lng0
    return Tn_ref + _c1 * ln_eps


def Tn_pert_2nd(gamma):
    """2nd-order:  quadratic equation for δT."""
    ln_eps = math.log(gamma) - lng0
    c_val = 4.0 * ln_eps
    disc = pt_b**2 - 4.0 * pt_a * c_val
    if disc < 0:
        return np.nan
    return Tn_ref - 2.0 * c_val / (pt_b + math.sqrt(disc))


def Tn_pert_3rd_poly(gamma):
    """3rd-order expanded polynomial in lnγ."""
    L = math.log(gamma)
    return pt_A0 + pt_A1 * L + pt_A2 * L**2 + pt_A3 * L**3


# ═══════════════════════════════════════════════════════════════════
#  Scan γ ∈ [10⁻¹⁰, 10⁻⁴]  →  T_n(γ)
# ═══════════════════════════════════════════════════════════════════
N_GAMMA = 300
gamma_arr = np.logspace(-10, -4, N_GAMMA)
print(f"\nScanning {N_GAMMA} values of γ ∈ [10⁻¹⁰, 10⁻⁴] …", end=" ", flush=True)
Tn_bf = np.array([find_Tn(gv) for gv in gamma_arr])
Tn_sa = np.array([find_Tn_semi(gv, c0, c1, s0, s1) for gv in gamma_arr])
Tn_p1 = np.array([Tn_pert_1st(gv) for gv in gamma_arr])
Tn_p2 = np.array([Tn_pert_2nd(gv) for gv in gamma_arr])
Tn_p3 = np.array([Tn_pert_3rd_poly(gv) for gv in gamma_arr])
print("done.")

# ═══════════════════════════════════════════════════════════════════
#  Numerical fit of perturbative quadratic form:
#  Fit  a_eff, b_eff  in  a(T_n-T_n⁰)² + b(T_n-T_n⁰) + 4ln(γ/γ₀) = 0
#  so that the implicit solution best matches brute-force data.
# ═══════════════════════════════════════════════════════════════════
def _pert_quad_model(lng_arr, b_eff, a_eff):
    """T_n from the perturbative quadratic with free (b, a)."""
    ln_eps = lng_arr - lng0
    c_arr = 4.0 * ln_eps
    disc = b_eff**2 - 4.0 * a_eff * c_arr
    disc = np.maximum(disc, 0)
    return Tn_ref - 2.0 * c_arr / (b_eff + np.sqrt(disc))


_mask_bf = ~np.isnan(Tn_bf)
_popt_nq, _ = curve_fit(
    _pert_quad_model,
    np.log(gamma_arr[_mask_bf]),
    Tn_bf[_mask_bf],
    p0=[pt_b, pt_a],
)
b_num, a_num = _popt_nq
Tn_nq = _pert_quad_model(np.log(gamma_arr), b_num, a_num)

# Also derive the implied polynomial coefficients from fitted a, b
nq_A2 = -16.0 * a_num / b_num**3
nq_A1 = -4.0 / b_num + 32.0 * a_num * lng0 / b_num**3
nq_A0 = (
    Tn_ref + 4.0 / b_num * lng0
    - 16.0 * a_num / b_num**3 * lng0**2
)

print(f"\n{'─' * 80}")
print("  Numerical fit of perturbative form:")
print(f"  b_eff = {b_num:.10e}   (analytic: {pt_b:.10e})")
print(f"  a_eff = {a_num:.10e}   (analytic: {pt_a:.10e})")
print(
    f"  Implied polynomial: {nq_A0:.4f}"
    f" + ({nq_A1:.6f}) lnγ + ({nq_A2:.8f}) (lnγ)²"
)
print(f"{'─' * 80}")

# ═══════════════════════════════════════════════════════════════════
#  Fit T_n(γ) to analytical forms
# ═══════════════════════════════════════════════════════════════════
mask = ~np.isnan(Tn_bf)
lng = np.log(gamma_arr[mask])
Tv = Tn_bf[mask]
ss_tot = np.sum((Tv - Tv.mean()) ** 2)

# --- Fit 1: quadratic in lnγ ---
p2 = np.polyfit(lng, Tv, 2)
Tv_f2 = np.polyval(p2, lng)
r2_f2 = 1 - np.sum((Tv - Tv_f2) ** 2) / ss_tot

# --- Fit 2: cubic in lnγ ---
p3 = np.polyfit(lng, Tv, 3)
Tv_f3 = np.polyval(p3, lng)
r2_f3 = 1 - np.sum((Tv - Tv_f3) ** 2) / ss_tot

# --- Fit 3: power law  T_n = T_c2 + A (−lnγ)^B ---
pw_ok = False
try:
    def _pwl(lngn, A, B):
        return T_c2 + A * lngn**B

    popt_pw, _ = curve_fit(_pwl, -lng, Tv, p0=[10.0, 0.5], maxfev=20000)
    Tv_pw = _pwl(-lng, *popt_pw)
    r2_pw = 1 - np.sum((Tv - Tv_pw) ** 2) / ss_tot
    pw_ok = True
except Exception:
    pass

# --- Fit 4: T_n ≈ T_c2 + A lnγ + B  (linear in lnγ shifted by T_c2) ---
dT = Tv - T_c2
p1_dT = np.polyfit(lng, dT, 1)
dT_f1 = np.polyval(p1_dT, lng)
r2_lin = 1 - np.sum((dT - dT_f1) ** 2) / np.sum((dT - dT.mean()) ** 2)

print(f"\n{'═' * 80}")
print(f"  Fitting results   (T_n in GeV,  T_c2 = {T_c2:.4f} GeV)")
print(f"{'═' * 80}")
print(
    f"  Linear:      T_n = {T_c2 + p1_dT[1]:.4f}  +  ({p1_dT[0]:.6f}) ln γ"
    f"                       R² = {r2_lin:.10f}"
)
print(
    f"  Quadratic:   T_n = {p2[2]:.4f}  +  ({p2[1]:.6f}) ln γ"
    f"  +  ({p2[0]:.8f}) (ln γ)²"
    f"       R² = {r2_f2:.10f}"
)
print(
    f"  Cubic:       T_n = {p3[3]:.4f}  +  ({p3[2]:.6f}) ln γ"
    f"  +  ({p3[1]:.8f}) (ln γ)²"
    f"  +  ({p3[0]:.10f}) (ln γ)³"
)
print(f"               R² = {r2_f3:.10f}")
if pw_ok:
    print(
        f"  Power law:   T_n = T_c2 + {popt_pw[0]:.6f} × (−ln γ)^{popt_pw[1]:.6f}"
        f"               R² = {r2_pw:.10f}"
    )

# R² for perturbative / numerical-perturbative approaches
Tv_p1 = Tn_p1[mask]
Tv_p2 = Tn_p2[mask]
Tv_p3 = Tn_p3[mask]
Tv_nq = Tn_nq[mask]
r2_p1 = 1 - np.sum((Tv - Tv_p1) ** 2) / ss_tot
r2_p2 = 1 - np.sum((Tv - Tv_p2) ** 2) / ss_tot
r2_p3 = 1 - np.sum((Tv - Tv_p3) ** 2) / ss_tot
r2_nq = 1 - np.sum((Tv - Tv_nq) ** 2) / ss_tot
print(
    f"  Pert. 1st:   δT = −(4/b) lnε"
    f"                  R² = {r2_p1:.10f}"
)
print(
    f"  Pert. 2nd:   a δT² + b δT + 4lnε = 0"
    f"         R² = {r2_p2:.10f}"
)
print(
    f"  Pert. 3rd:   semi-analytic cubic poly"
    f"         R² = {r2_p3:.10f}"
)
print(
    f"  Num. pert.:  fit a_eff, b_eff"
    f"                     R² = {r2_nq:.10f}"
)
print(f"{'═' * 80}")

# Polynomial coefficient comparison table
print("\n  Polynomial coefficients  (T_n = A₀ + A₁ lnγ + A₂ (lnγ)² + A₃ (lnγ)³):")
hdr = f"  {'Method':22} {'A₀':>12} {'A₁':>12} {'A₂':>12} {'A₃':>14}"
print(hdr)
print(f"  {'─' * len(hdr)}")
print(f"  {'Pert. 2nd (analytic)':22} {pt_A0:12.4f} {pt_A1:12.6f} {_c2:12.8f} {'—':>14}")
print(f"  {'Pert. 3rd (analytic)':22} {pt_A0:12.4f} {pt_A1:12.6f} {pt_A2:12.8f} {pt_A3:14.10f}")
print(f"  {'Num. pert. (fitted)':22} {nq_A0:12.4f} {nq_A1:12.6f} {nq_A2:12.8f} {'—':>14}")
print(f"  {'Brute-force quad.':22} {p2[2]:12.4f} {p2[1]:12.6f} {p2[0]:12.8f} {'—':>14}")
print(f"  {'Brute-force cubic':22} {p3[3]:12.4f} {p3[2]:12.6f} {p3[1]:12.8f} {p3[0]:14.10f}")

# ═══════════════════════════════════════════════════════════════════
#  Detailed table at selected γ values
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'─' * 150}")
print(
    f"  {'γ':>14}  {'T_n BF':>10}  {'T_n SA':>10}  "
    f"{'P2 anal.':>10}  {'P3 anal.':>10}  "
    f"{'Num. pert.':>10}  {'BF cubic':>10}  "
    f"{'ΔP3 [GeV]':>10}  {'ΔNP [GeV]':>10}"
)
print(f"{'─' * 150}")
for ge in np.arange(-10, -3.5, 0.5):
    gv = 10**ge
    tb = find_Tn(gv)
    ts = find_Tn_semi(gv, c0, c1, s0, s1)
    tp2 = Tn_pert_2nd(gv)
    tp3 = Tn_pert_3rd_poly(gv)
    tnq = _pert_quad_model(np.array([math.log(gv)]), b_num, a_num)[0]
    tcf = np.polyval(p3, math.log(gv))
    d_p3 = tb - tp3 if not np.isnan(tb) else np.nan
    d_nq = tb - tnq if not np.isnan(tb) else np.nan
    tb_s = f"{tb:.2f}" if not np.isnan(tb) else "—"
    ts_s = f"{ts:.2f}" if not np.isnan(ts) else "—"
    tp2_s = f"{tp2:.2f}" if not np.isnan(tp2) else "—"
    print(
        f"  10^{ge:5.1f}  "
        f"{tb_s:>10}  {ts_s:>10}  "
        f"{tp2_s:>10}  {tp3:10.2f}  "
        f"{tnq:10.2f}  {tcf:10.2f}  "
        f"{d_p3:10.4f}  {d_nq:10.4f}"
    )

# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════
OUT = "figs/Tn_gamma"
os.makedirs(OUT, exist_ok=True)

# ─── Plot 1: T_n vs γ ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

m_bf = ~np.isnan(Tn_bf)
m_sa = ~np.isnan(Tn_sa)

ax1.plot(gamma_arr[m_bf], Tn_bf[m_bf] / 1e3, "b-", lw=2, label="Brute force")
ax1.plot(
    gamma_arr[m_sa],
    Tn_sa[m_sa] / 1e3,
    "r--",
    lw=1.5,
    label="Semi-analytical (Eqs. 67-71)",
)
m_p2 = ~np.isnan(Tn_p2)
ax1.plot(
    gamma_arr[m_p2], Tn_p2[m_p2] / 1e3, "c--", lw=1.2,
    label=rf"Pert. 2nd analytic ($R^2$={r2_p2:.6f})",
)
ax1.plot(
    gamma_arr[m_p2], Tn_p3[m_p2] / 1e3, "m-", lw=1.2,
    label=rf"Pert. 3rd analytic ($R^2$={r2_p3:.6f})",
)
ax1.plot(
    gamma_arr, Tn_nq / 1e3, color="orange", ls="-.", lw=1.5,
    label=rf"Num. pert. fit ($R^2$={r2_nq:.6f})",
)
ax1.plot(
    gamma_arr[mask],
    np.polyval(p3, np.log(gamma_arr[mask])) / 1e3,
    "g:", lw=1.8,
    label=f"BF cubic fit ($R^2$={r2_f3:.6f})",
)
ax1.axvline(GAMMA_REF, color="gray", ls=":", lw=0.8, alpha=0.5)
ax1.annotate(
    r"$\gamma_{\mathrm{ref}}$",
    xy=(GAMMA_REF, ax1.get_ylim()[0]),
    fontsize=9,
    color="gray",
    ha="center",
)
ax1.set_xscale("log")
ax1.set_xlabel(r"$\gamma = \varphi_0 / M_{\mathrm{Pl}}$", fontsize=13)
ax1.set_ylabel(r"$T_n$ [TeV]", fontsize=13)
ax1.set_title(r"$T_n(\gamma)$  with $m = 1$ TeV", fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# T_n vs log10(γ) — compare semi-analytic vs numerical fit
ax2.plot(
    np.log10(gamma_arr[m_bf]), Tn_bf[m_bf] / 1e3,
    "b-", lw=2, label="Brute force",
)
ax2.plot(
    np.log10(gamma_arr[m_p2]), Tn_p3[m_p2] / 1e3,
    "m-", lw=1.2, label="Pert. 3rd (analytic)",
)
ax2.plot(
    np.log10(gamma_arr), Tn_nq / 1e3,
    color="orange", ls="-.", lw=1.5, label="Num. pert. (fitted a,b)",
)
ax2.plot(
    np.log10(gamma_arr[mask]),
    np.polyval(p3, np.log(gamma_arr[mask])) / 1e3,
    "g--", lw=1.5, label="BF cubic fit",
)
ax2.set_xlabel(r"$\log_{10}\gamma$", fontsize=13)
ax2.set_ylabel(r"$T_n$ [TeV]", fontsize=13)
ax2.set_title(r"$T_n$ vs $\log_{10}\gamma$", fontsize=14)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/Tn_vs_gamma.png", dpi=200)
plt.close(fig)
print(f"\nSaved: {OUT}/Tn_vs_gamma.png")

# ─── Plot 2: S₃/T and nucleation lines for several γ ────────────
fig2, ax3 = plt.subplots(figsize=(10, 7))
T_plot = _T_GRID[_valid]
S_plot = _S_GRID[_valid]

ax3.plot(T_plot / 1e3, S_plot, "k-", lw=2.5, label=r"$S_3/T$")

cmap = plt.cm.coolwarm
gamma_show = [-10, -9, -8, -7, -6, -5, -4]
for j, ge in enumerate(gamma_show):
    gv = 10**ge
    nuc_line = 4 * np.log(2 * np.sqrt(3) * T_plot / (m * gv))
    colour = cmap(j / (len(gamma_show) - 1))
    ax3.plot(
        T_plot / 1e3,
        nuc_line,
        "--",
        color=colour,
        lw=1.2,
        label=rf"$\gamma=10^{{{ge}}}$",
    )
    tn = find_Tn(gv)
    if not np.isnan(tn):
        s3tn = float(_S3T_spl(tn))
        ax3.plot(tn / 1e3, s3tn, "o", color=colour, ms=7, zorder=5)

ax3.set_xlabel(r"$T$ [TeV]", fontsize=13)
ax3.set_ylabel(r"$S_3/T$", fontsize=13)
ax3.set_title(
    r"Nucleation: $S_3/T ="
    r" 4\ln\!\left(\frac{2\sqrt{3}\,T}{m\gamma}\right)$",
    fontsize=14,
)
ax3.set_xlim([T_c2 / 1e3 - 0.01, T_c2 / 1e3 + 0.5])
ax3.set_ylim([0, 200])
ax3.legend(fontsize=8.5, ncol=2, loc="upper left")
ax3.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{OUT}/nucleation_condition.png", dpi=200)
plt.close(fig2)
print(f"Saved: {OUT}/nucleation_condition.png")

# ─── Plot 3: fit residuals ───────────────────────────────────────
fig3, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(
    np.log10(gamma_arr[mask]),
    Tv - Tv_f2,
    "b.-",
    ms=3,
    lw=0.8,
    label="Quadratic",
)
ax4.plot(
    np.log10(gamma_arr[mask]),
    Tv - Tv_f3,
    "r.-",
    ms=3,
    lw=0.8,
    label="Cubic",
)
ax4.plot(
    np.log10(gamma_arr[mask]),
    Tv - Tv_p1,
    "c.-",
    ms=3,
    lw=0.8,
    label="Pert. 1st",
)
ax4.plot(
    np.log10(gamma_arr[mask]),
    Tv - Tv_p2,
    color="orange",
    marker=".",
    ms=3,
    lw=0.8,
    label="Pert. 2nd",
)
if pw_ok:
    ax4.plot(
        np.log10(gamma_arr[mask]),
        Tv - Tv_pw,
        "g.-",
        ms=3,
        lw=0.8,
        label="Power law",
    )
ax4.axhline(0, color="k", lw=0.5)
ax4.set_xlabel(r"$\log_{10}\gamma$", fontsize=13)
ax4.set_ylabel(r"$T_n^{\rm brute} - T_n^{\rm fit}$ [GeV]", fontsize=13)
ax4.set_title("Fit residuals", fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(f"{OUT}/fit_residuals.png", dpi=200)
plt.close(fig3)
print(f"Saved: {OUT}/fit_residuals.png")

# ─── Plot 4: brute-force vs semi-analytical comparison ───────────
fig4, ax5 = plt.subplots(figsize=(10, 5))
both_ok = m_bf & m_sa
if np.any(both_ok):
    ax5.plot(
        np.log10(gamma_arr[both_ok]),
        (Tn_bf[both_ok] - Tn_sa[both_ok]),
        "b.-",
        ms=3,
        lw=0.8,
    )
    ax5.axhline(0, color="k", lw=0.5)
    ax5.set_xlabel(r"$\log_{10}\gamma$", fontsize=13)
    ax5.set_ylabel(r"$T_n^{\rm BF} - T_n^{\rm SA}$ [GeV]", fontsize=13)
    ax5.set_title(
        "Brute force − Semi-analytical   "
        rf"(expansion around $\alpha_n$ = {alpha_ref:.2f})",
        fontsize=12,
    )
    ax5.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(f"{OUT}/bf_vs_semi.png", dpi=200)
plt.close(fig4)
print(f"Saved: {OUT}/bf_vs_semi.png")

# ─── Plot 5: relative error of perturbative vs brute force ───────
fig5, (ax6, ax7) = plt.subplots(1, 2, figsize=(15, 5.5))

lg10 = np.log10(gamma_arr[m_bf])
Tn_valid = Tn_bf[m_bf]

# Errors for each method
_lg_bf = np.log(gamma_arr[m_bf])
err_p1 = np.abs(Tn_p1[m_bf] - Tn_valid)
err_p2 = np.abs(Tn_p2[m_bf] - Tn_valid)
err_p3 = np.abs(Tn_p3[m_bf] - Tn_valid)
err_nq = np.abs(Tn_nq[m_bf] - Tn_valid)
err_sa = np.abs(Tn_sa[m_bf] - Tn_valid)
err_qf = np.abs(np.polyval(p2, _lg_bf) - Tn_valid)
err_cf = np.abs(np.polyval(p3, _lg_bf) - Tn_valid)

_methods = [
    (err_p1, "c-",      1.5, "Pert. 1st (analytic)"),
    (err_p2, "c--",     1.2, "Pert. 2nd (analytic)"),
    (err_p3, "m-",      1.5, "Pert. 3rd (analytic)"),
    (err_nq, "orange",  1.8, "Num. pert. (fitted a,b)"),
    (err_qf, "b-",      1.5, "Brute-force quad. fit"),
    (err_cf, "g-.",     1.2, "Brute-force cubic fit"),
]

# Left: absolute error
for err, style, lw, lab in _methods:
    if isinstance(style, str) and len(style) <= 3:
        ax6.semilogy(lg10, err, style, lw=lw, label=lab)
    else:
        ax6.semilogy(lg10, err, color=style, lw=lw, label=lab)
ax6.set_xlabel(r"$\log_{10}\gamma$", fontsize=13)
ax6.set_ylabel(
    r"$|T_n^{\rm method} - T_n^{\rm BF}|$ [GeV]", fontsize=13
)
ax6.set_title("Absolute error vs brute force", fontsize=14)
ax6.legend(fontsize=8.5, loc="upper left")
ax6.grid(True, alpha=0.3, which="both")

# Right: relative error (%)
for err, style, lw, lab in _methods:
    rel = err / Tn_valid * 100
    if isinstance(style, str) and len(style) <= 3:
        ax7.semilogy(lg10, rel, style, lw=lw, label=lab)
    else:
        ax7.semilogy(lg10, rel, color=style, lw=lw, label=lab)
ax7.set_xlabel(r"$\log_{10}\gamma$", fontsize=13)
ax7.set_ylabel("Relative error [%]", fontsize=13)
ax7.set_title("Relative error vs brute force", fontsize=14)
ax7.legend(fontsize=8.5, loc="upper left")
ax7.grid(True, alpha=0.3, which="both")

fig5.tight_layout()
fig5.savefig(f"{OUT}/perturbative_error.png", dpi=200)
plt.close(fig5)
print(f"Saved: {OUT}/perturbative_error.png")

# ─── Plot 6: numerical quadratic fit  with annotated coefficients ─────
fig6, (ax8, ax9) = plt.subplots(1, 2, figsize=(15, 6))

lg10_v = np.log10(gamma_arr[mask])
lng_v = lng                         # ln γ  (natural log, same as used for fits)

# Three quadratic descriptions to compare:
#   1) Pert. 2nd analytic:  A₀ + A₁ lnγ + A₂ (lnγ)²   coefficients from F', F''
#   2) Num. pert. (fitted): implied polynomial from fitted a_eff, b_eff
#   3) BF quadratic fit:   np.polyfit degree 2

_c2_ana = -16.0 * pt_a / pt_b**3          # analytic A₂ (2nd order, no A₃)
_c1_ana = -4.0 / pt_b + 32.0 * pt_a * lng0 / pt_b**3
_c0_ana = Tn_ref + 4.0 / pt_b * lng0 - 16.0 * pt_a / pt_b**3 * lng0**2
Tn_ana_q = _c0_ana + _c1_ana * lng_v + _c2_ana * lng_v**2

quads = [
    ("Pert. 2nd (analytic)", _c0_ana, _c1_ana, _c2_ana, Tn_ana_q, "c", "--"),
    ("Num. pert. (fitted)", nq_A0, nq_A1, nq_A2, Tv_nq, "orange", "-"),
    ("BF quadratic fit", p2[2], p2[1], p2[0], Tv_f2, "b", ":"),
]

# Left panel: T_n vs lnγ with all three quadratics
ax8.plot(lng_v, Tv / 1e3, "k-", lw=2.5, label="Brute force", zorder=5)
for lab, c0, c1, c2, Tn_q, col, ls in quads:
    ax8.plot(lng_v, Tn_q / 1e3, color=col, ls=ls, lw=1.8, label=lab)
ax8.set_xlabel(r"$\ln\gamma$", fontsize=13)
ax8.set_ylabel(r"$T_n$ [TeV]", fontsize=13)
ax8.set_title(
    r"Quadratic fits:  $T_n = A_0 + A_1\,\ln\gamma + A_2\,(\ln\gamma)^2$",
    fontsize=13,
)
ax8.legend(fontsize=10, loc="upper right")
ax8.grid(True, alpha=0.3)

# Text box with coefficient values
_lines = []
for lab, c0, c1, c2, _, _, _ in quads:
    _lines.append(f"{lab}")
    _lines.append(
        f"  $A_0$ = {c0:.2f},  $A_1$ = {c1:.4f},  $A_2$ = {c2:.6f}"
    )
txt = "\n".join(_lines)
ax8.text(
    0.03, 0.03, txt, transform=ax8.transAxes, fontsize=8.5,
    verticalalignment="bottom", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.85),
)

# Right panel: residuals of each quadratic vs brute force
for lab, c0, c1, c2, Tn_q, col, ls in quads:
    ax9.plot(
        lng_v, (Tn_q - Tv), color=col, ls=ls, lw=1.8, label=lab,
    )
ax9.axhline(0, color="k", lw=0.5)
ax9.set_xlabel(r"$\ln\gamma$", fontsize=13)
ax9.set_ylabel(r"$T_n^{\rm fit} - T_n^{\rm BF}$  [GeV]", fontsize=13)
ax9.set_title("Quadratic fit residuals", fontsize=13)
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)

# Annotate R² on residual panel
_r2_ana_q = 1 - np.sum((Tv - Tn_ana_q) ** 2) / ss_tot
ax9.text(
    0.03, 0.97,
    f"$R^2$  Pert. 2nd analytic = {_r2_ana_q:.10f}\n"
    f"$R^2$  Num. pert. fitted  = {r2_nq:.10f}\n"
    f"$R^2$  BF quadratic fit   = {r2_f2:.10f}",
    transform=ax9.transAxes, fontsize=9,
    verticalalignment="top", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.85),
)

fig6.tight_layout()
fig6.savefig(f"{OUT}/quadratic_fits.png", dpi=200)
plt.close(fig6)
print(f"Saved: {OUT}/quadratic_fits.png")

print("\nAll done.")
