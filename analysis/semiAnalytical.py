#!/usr/bin/env python3
"""Semi-analytical bounce action & nucleation temperature (TABLE I).

Supports both fermionic and bosonic thermal contributions.
  --nf, --nb  : species multiplicities
  --yf, --yb  : Yukawa couplings
  --gf, --gb  : gauge couplings
  --gamma     : phi_0 / M_Pl
  --Tmin/--Tmax/--dT : temperature scan range [TeV]
"""

import argparse
import numpy as np
from math import pi, sqrt, log
from scipy.integrate import quad
from scipy.optimize import brentq

_p = argparse.ArgumentParser(description="Semi-analytical nucleation table")
_p.add_argument("--m", type=float, default=1.0, help="flaton mass [TeV] (default 1)")
_p.add_argument("--nf", type=float, default=20.0, help="fermion multiplicity")
_p.add_argument("--nb", type=float, default=20.0, help="boson multiplicity")
_p.add_argument("--yf", type=float, default=1.09, help="fermion Yukawa coupling")
_p.add_argument("--yb", type=float, default=1.09, help="boson Yukawa coupling")
_p.add_argument("--gf", type=float, default=1.05, help="fermion gauge coupling")
_p.add_argument("--gb", type=float, default=1.05, help="boson gauge coupling")
_p.add_argument("--gamma", type=float, default=4.1667e-8, help="phi0/Mpl")
_p.add_argument("--Tmin", type=float, default=1.13, help="scan start [TeV]")
_p.add_argument("--Tmax", type=float, default=1.28, help="scan end [TeV]")
_p.add_argument("--dT", type=float, default=0.01, help="scan step [TeV]")
_p.add_argument("--Tn_lo", type=float, default=None, help="T_n bracket low [TeV]")
_p.add_argument("--Tn_hi", type=float, default=None, help="T_n bracket high [TeV]")
args = _p.parse_args()

m = args.m
yf = args.yf
yb = args.yb
gf = args.gf
gb = args.gb
nf = args.nf
nb = args.nb
gamma = args.gamma

# Eq (13): delta2_f = gf^2/6
delta2_f = gf**2 / 6.0
# Eq (14): delta2_b(T) = mb^2/T^2 + (1/4)yb^2 + (2/3)gb^2  (Eq. 27)
delta2_b_const = 0.25 * yb**2 + (2.0 / 3.0) * gb**2
mb = m  # boson soft mass = flaton mass (Eq. 16)

# Gaussian quadrature nodes/weights for weight fn rho(w) = sqrt(-ln w) (Appendix A)
x_nodes = np.array([0.1535900739, 0.6908657079])
w_nodes = np.array([0.5563908709, 0.3298360546])

# phi0 = gamma * Mpl, lambda = m^2/phi0^2
Mpl_TeV = 2.4e15  # reduced Planck mass in TeV
phi0 = gamma * Mpl_TeV
lam = m**2 / phi0**2


def delta2_b(T):
    """Eq. (27): delta2_b(T) = mb^2/T^2 + (1/4)yb^2 + (2/3)gb^2"""
    return mb**2 / T**2 + delta2_b_const


# ═══════════════════════════════════════════════════════════════════
#  Thermal integrals  (sign conventions match paper Eqs. 1–5)
# ═══════════════════════════════════════════════════════════════════


def _safe_fermi(E):
    if E > 700:
        return 0.0
    return 1.0 / (np.exp(E) + 1.0)


def _safe_bose(E):
    if E > 700:
        return 0.0
    return 1.0 / (np.exp(E) - 1.0)


def JF(z):
    """J_F(u^2) = -(1/2pi^2) int dx x^2 ln(1+exp(-sqrt(x^2+u^2)))  [Eq. 2]"""

    def integrand(x):
        E = sqrt(x * x + z)
        if E > 500:
            return 0.0
        return x * x * np.log(1.0 + np.exp(-E))

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)
    return -val / (2.0 * pi**2)


def JB(z):
    """J_B(u^2) = +(1/2pi^2) int dx x^2 ln(1-exp(-sqrt(x^2+u^2)))  [Eq. 4]"""

    def integrand(x):
        E = sqrt(x * x + z)
        if E > 500:
            return 0.0
        arg = 1.0 - np.exp(-E)
        if arg <= 0:
            return 0.0
        return x * x * np.log(arg)

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)
    return val / (2.0 * pi**2)


def JFp(z):
    """J'_F(u^2) = +(1/4pi^2) int dx x^2 f(E)/E,  f = Fermi dist"""

    def integrand(x):
        E = sqrt(x * x + z)
        return x * x * _safe_fermi(E) / E

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=300)
    return val / (4.0 * pi**2)


def JBp(z):
    """J'_B(u^2) = +(1/4pi^2) int dx x^2 n(E)/E,  n = Bose dist"""

    def integrand(x):
        E = sqrt(x * x + z)
        return x * x * _safe_bose(E) / E

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=300)
    return val / (4.0 * pi**2)


def JFpp(z):
    """J''_F(u^2) < 0"""

    def integrand(x):
        E = sqrt(x * x + z)
        f = _safe_fermi(E)
        return x * x * (f * (1.0 - f) / E**2 + f / E**3)

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=300)
    return -val / (8.0 * pi**2)


def JBpp(z):
    """J''_B(u^2) < 0"""

    def integrand(x):
        E = sqrt(x * x + z)
        n = _safe_bose(E)
        return x * x * (n / E**3 + n * (1.0 + n) / E**2)

    val, _ = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-10, limit=300)
    return -val / (8.0 * pi**2)


# ═══════════════════════════════════════════════════════════════════
#  Alpha equation  (Eq. 50)
#
#  m^2/(yf^2 T^2) = lam_term
#    + (nf/sqrt(pi)) sum_i wi [3/af * (JF(df+xi*af)-JF(df))/xi - JFp(df+xi*af)]
#    + (nb/sqrt(pi)) sum_i wi [3/af * (JB(db+xi*ab)-JB(db))/xi - yb2/yf2 * JBp(db+xi*ab)]
#
#  where af = alpha_f,  ab = (yb/yf)^2 * af
# ═══════════════════════════════════════════════════════════════════


def alpha_equation(alpha_f, T):
    lhs = m**2 / (yf**2 * T**2)
    d2b = delta2_b(T)
    yb2_over_yf2 = yb**2 / yf**2

    rhs = sqrt(2.0) * lam * alpha_f / (8.0 * yf**4)  # negligible

    inv_sqrt_pi = 1.0 / sqrt(pi)
    for xi, wi in zip(x_nodes, w_nodes):
        zf = delta2_f + xi * alpha_f
        rhs += (
            inv_sqrt_pi
            * nf
            * wi
            * (3.0 / alpha_f * (JF(zf) - JF(delta2_f)) / xi - JFp(zf))
        )
        alpha_b_xi = xi * yb2_over_yf2 * alpha_f
        zb = d2b + alpha_b_xi
        rhs += (
            inv_sqrt_pi
            * nb
            * wi
            * (3.0 / alpha_f * (JB(zb) - JB(d2b)) / xi - yb2_over_yf2 * JBp(zb))
        )

    return rhs - lhs


def solve_alpha(T):
    a_min, a_max = 1e-6, 1.0
    try:
        f_min = alpha_equation(a_min, T)
    except Exception:
        return None
    f_max = alpha_equation(a_max, T)

    while f_min * f_max > 0 and a_max < 500:
        a_max *= 2.0
        f_max = alpha_equation(a_max, T)

    if f_min * f_max > 0:
        return None
    return brentq(lambda a: alpha_equation(a, T), a_min, a_max, xtol=1e-11, rtol=1e-11)


# ═══════════════════════════════════════════════════════════════════
#  F_F, F_B and their derivatives (Eqs. 40–45)
# ═══════════════════════════════════════════════════════════════════


def FF(alpha_f):
    """Eq. (42): F_F(alpha_f)"""
    s = 0.0
    for xi, wi in zip(x_nodes, w_nodes):
        zf = delta2_f + xi * alpha_f
        s += wi * (JF(zf) - JF(delta2_f)) / xi
    return (pi / sqrt(2.0)) * s


def FB(alpha_b, T):
    """Eq. (42): F_B(alpha_b)"""
    d2b = delta2_b(T)
    s = 0.0
    for xi, wi in zip(x_nodes, w_nodes):
        zb = d2b + xi * alpha_b
        s += wi * (JB(zb) - JB(d2b)) / xi
    return (pi / sqrt(2.0)) * s


def FFp(alpha_f):
    """Eq. (45): F'_F(alpha_f)"""
    s = 0.0
    for xi, wi in zip(x_nodes, w_nodes):
        zf = delta2_f + xi * alpha_f
        s += wi * JFp(zf)
    return (pi / sqrt(2.0)) * s


def FBp(alpha_b, T):
    """Eq. (45): F'_B(alpha_b)"""
    d2b = delta2_b(T)
    s = 0.0
    for xi, wi in zip(x_nodes, w_nodes):
        zb = d2b + xi * alpha_b
        s += wi * JBp(zb)
    return (pi / sqrt(2.0)) * s


# ═══════════════════════════════════════════════════════════════════
#  S3/T  (Eq. 53)
# ═══════════════════════════════════════════════════════════════════


def S3_over_T(T):
    alpha_f = solve_alpha(T)
    if alpha_f is None:
        return None, None, None

    yb2_over_yf2 = yb**2 / yf**2
    alpha_b = yb2_over_yf2 * alpha_f

    # Eq. (52) denominator:  nf*yf^2*(F'F - FF/af) + nb*yb^2*(F'B - FB/ab)
    term_f = nf * yf**2 * (FFp(alpha_f) - FF(alpha_f) / alpha_f)
    term_b = nb * yb**2 * (FBp(alpha_b, T) - FB(alpha_b, T) / alpha_b)
    denom = term_f + term_b

    if denom >= 0:
        return alpha_f, alpha_b, None

    # Eq. (53)
    prefactor = (pi**1.5 / sqrt(2.0)) * alpha_f / yf**2
    inside = -(pi**1.5 / sqrt(2.0)) / denom
    s3t = prefactor * sqrt(inside)

    return alpha_f, alpha_b, s3t


def RHS_nucleation(T):
    """4 ln(2 sqrt(3) T / (gamma * m))  [Eq. 73]"""
    return 4.0 * log(2.0 * sqrt(3.0) * T / (gamma * m))


def F_nucleation(T):
    """C(T) = S3/T - 4 ln(...)  [Eq. 73]"""
    _, _, s3t = S3_over_T(T)
    if s3t is None:
        return None
    return s3t - RHS_nucleation(T) + 3 / 2 * np.log(s3t / (2 * np.pi))


# ═══════════════════════════════════════════════════════════════════
#  Print table
# ═══════════════════════════════════════════════════════════════════

print(
    f"\nParameters: m={m} TeV, nf={nf}, nb={nb}, yf={yf}, yb={yb}, "
    f"gf={gf}, gb={gb}, gamma={gamma:.4e}"
)
print(f"  lambda   = {lam:.4e}")
print(f"  delta2_f = {delta2_f:.6f}")
print(f"  delta2_b(T→∞) = {delta2_b_const:.6f}")
print()

Ts = np.arange(args.Tmin, args.Tmax + 0.5 * args.dT, args.dT)

hdr = (
    f"{'T [TeV]':>10}  {'alpha_f':>12}  {'alpha_b':>12}  "
    f"{'S3/T':>14}  {'4ln(...)':>14}  {'C(T)':>14}"
)
print(hdr)
print("-" * len(hdr))

rows = []
c_list = []
for T in Ts:
    alpha_f, alpha_b, s3t = S3_over_T(T)
    rhs = RHS_nucleation(T)

    if alpha_f is None or s3t is None:
        print(f"{T:10.2f}  {'—':>12}  {'—':>12}  {'—':>14}  {rhs:14.5f}  {'—':>14}")
        rows.append(dict(T=T, alpha_f=None, alpha_b=None, S3T=None, RHS=rhs, C=None))
    else:
        C = s3t - rhs + 3 / 2 * np.log(s3t / (2 * np.pi))
        print(
            f"{T:10.2f}  {alpha_f:12.5f}  {alpha_b:12.5f}  "
            f"{s3t:14.5f}  {rhs:14.5f}  {C:14.5f}"
        )
        c_list.append(round(C, 5))
        rows.append(dict(T=T, alpha_f=alpha_f, alpha_b=alpha_b, S3T=s3t, RHS=rhs, C=C))

print(c_list)
# ═══════════════════════════════════════════════════════════════════
#  Find T_n
# ═══════════════════════════════════════════════════════════════════

valid = [r for r in rows if r["C"] is not None]
sign_changes = []
for i in range(len(valid) - 1):
    c1, c2 = valid[i]["C"], valid[i + 1]["C"]
    if c1 * c2 < 0:
        sign_changes.append((valid[i]["T"], valid[i + 1]["T"]))

Tn_lo = args.Tn_lo
Tn_hi = args.Tn_hi
if Tn_lo is None and sign_changes:
    Tn_lo, Tn_hi = sign_changes[0]

print(Tn_lo, Tn_hi)

if Tn_lo is not None and Tn_hi is not None:
    try:
        Tn = brentq(lambda T: F_nucleation(T), Tn_lo, Tn_hi, xtol=1e-12, rtol=1e-12)
        af_n, ab_n, s3t_n = S3_over_T(Tn)

        print(f"\nNucleation result (C(T_n) = 0)")
        print("-" * 40)
        print(f"  T_n       = {Tn:.12f} TeV")
        print(f"  alpha_f   = {af_n:.12f}")
        print(f"  alpha_b   = {ab_n:.12f}")
        print(f"  S3/T      = {s3t_n:.12f}")
        print(f"  RHS       = {RHS_nucleation(Tn):.12f}")
        print(f"  C(T_n)    = {F_nucleation(Tn):.3e}")
    except Exception as e:
        print(f"\nCould not find T_n in [{Tn_lo}, {Tn_hi}]: {e}")
else:
    print(
        "\nNo sign change in C(T) found in scan range — "
        "try adjusting --Tmin/--Tmax or use --Tn_lo/--Tn_hi"
    )
