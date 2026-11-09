"""Numba kernels for the two-step potential (shared by bounce solver and lattice).

The thermal functions are tabulated in u = m/T = g*phi/T (uniform grid, cubic
Hermite with exact du-derivatives), so V, V' cost O(n_species) per site with no
Python overhead. Beyond u_max the J's are exponentially small and set to zero.
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit

from potential.twoStepPotential import TwoStepPotential

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_PATH = os.path.join(REPO, "data", "thermal_splines", "J_u_table.npz")
U_MAX = 40.0
N_U = 40001


def build_J_table(path: str = TABLE_PATH, u_max: float = U_MAX, n: int = N_U) -> dict:
    """Tabulate J_B, J_F and dJ/du on u in [0, u_max]; cached to ``path``."""
    if os.path.isfile(path):
        d = np.load(path)
        if float(d["u_max"]) == u_max and int(d["n"]) == n:
            return {k: d[k] for k in d.files}
    from cosmoTransitions import finiteT as CTFT

    u = np.linspace(0.0, u_max, n)
    th = u * u
    tab = {
        "u_max": np.float64(u_max),
        "n": np.int64(n),
        "Jb": CTFT.Jb_spline(th),
        "dJb": 2.0 * u * CTFT.Jb_spline(th, n=1),
        "Jf": CTFT.Jf_spline(th),
        "dJf": 2.0 * u * CTFT.Jf_spline(th, n=1),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **tab)
    return tab


def pack(pot: TwoStepPotential, tab: dict | None = None):
    """Flatten a TwoStepPotential into arrays for the numba kernels.

    Returns (coef[4], sp_n, sp_g, sp_f, J_tab[4, N], du):
      coef = (c2, c3, c4, c6); sp_f = 1.0 for fermions;
      J_tab rows = (Jb, dJb/du, Jf, dJf/du).
    """
    tab = tab or build_J_table()
    coef = np.array([pot.c2, pot.c3, pot.c4, pot.c6], dtype=np.float64)
    sp_n = np.array([s.n for s in pot.species], dtype=np.float64)
    sp_g = np.array([s.g for s in pot.species], dtype=np.float64)
    sp_f = np.array([1.0 if s.fermion else 0.0 for s in pot.species], dtype=np.float64)
    J_tab = np.vstack([tab["Jb"], tab["dJb"], tab["Jf"], tab["dJf"]]).astype(np.float64)
    du = float(tab["u_max"]) / (int(tab["n"]) - 1)
    return coef, sp_n, sp_g, sp_f, J_tab, du


@njit(cache=True, inline="always")
def _hermite(u, J, dJ, du):
    """Cubic Hermite interpolation of (J, dJ/du) at u; returns (J, dJ/du)."""
    n = J.shape[0]
    s = u / du
    i = int(s)
    if i >= n - 1:
        return 0.0, 0.0
    t = s - i
    y0, y1 = J[i], J[i + 1]
    m0, m1 = dJ[i] * du, dJ[i + 1] * du
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    val = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
    d00 = 6 * t2 - 6 * t
    d10 = 3 * t2 - 4 * t + 1
    d01 = -6 * t2 + 6 * t
    d11 = 3 * t2 - 2 * t
    der = (d00 * y0 + d10 * m0 + d01 * y1 + d11 * m1) / du
    return val, der


@njit(cache=True)
def V_rel_nb(phi, T, coef, sp_n, sp_g, sp_f, J_tab, du):
    """V(phi, T) - V(0, T), with the odd term as c3 |phi|^3."""
    p2 = phi * phi
    v = p2 * (coef[0] + abs(phi) * coef[1] + p2 * (coef[2] + p2 * coef[3]))
    if T > 0.0:
        pref = T**4 / (2.0 * np.pi**2)
        ap = abs(phi)
        for k in range(sp_n.shape[0]):
            row = 2 if sp_f[k] > 0.5 else 0
            j, _ = _hermite(sp_g[k] * ap / T, J_tab[row], J_tab[row + 1], du)
            v += pref * sp_n[k] * (j - J_tab[row, 0])
    return v


@njit(cache=True)
def dV_nb(phi, T, coef, sp_n, sp_g, sp_f, J_tab, du):
    """dV/dphi."""
    p2 = phi * phi
    d = phi * (2 * coef[0] + 3 * coef[1] * abs(phi) + 4 * coef[2] * p2 + 6 * coef[3] * p2 * p2)
    if T > 0.0:
        pref = T**3 / (2.0 * np.pi**2)
        ap = abs(phi)
        sgn = 1.0 if phi >= 0 else -1.0
        for k in range(sp_n.shape[0]):
            row = 2 if sp_f[k] > 0.5 else 0
            _, dj = _hermite(sp_g[k] * ap / T, J_tab[row], J_tab[row + 1], du)
            d += sgn * pref * sp_n[k] * sp_g[k] * dj
    return d


@njit(cache=True, inline="always")
def _u_set8(phi, T, g, gg, mb2_0, boson):
    """Dimensionless mass u = m/T for set8 Debye thermal."""
    p2 = phi * phi
    if boson:
        m2 = mb2_0 + 0.5 * g * g * p2 + (0.25 * g * g + 2.0 / 3.0 * gg * gg) * T * T
    else:
        m2 = 0.5 * g * g * p2 + (1.0 / 6.0) * gg * gg * T * T
    if m2 < 0.0:
        m2 = 0.0
    return math.sqrt(m2) / T


@njit(cache=True)
def dV_set8_nb(phi, T, coef, nb, nf, gB, gF, gg, mb2_0, lam8, include_cw, J_tab, du):
    """dV/dphi with set8 / V_correct thermal (Debye + optional CW).

    Tree from coef (c2,c3,c4,c6). Thermal matches thermal_tables.hpp chain rule.
    """
    p2 = phi * phi
    d = phi * (2 * coef[0] + 3 * coef[1] * abs(phi) + 4 * coef[2] * p2 + 6 * coef[3] * p2 * p2)
    if T > 0.0:
        pref = T**4 / (2.0 * np.pi**2)
        # bosons
        if nb != 0.0:
            ub = _u_set8(phi, T, gB, gg, mb2_0, True)
            ubs = ub if ub > 1e-30 else 1e-30
            _, dJb = _hermite(ubs, J_tab[0], J_tab[1], du)
            dub = 0.5 * gB * gB * phi / (T * T * ubs)
            d += pref * nb * dJb * dub
        # fermions
        if nf != 0.0:
            uf = _u_set8(phi, T, gF, gg, 0.0, False)
            ufs = uf if uf > 1e-30 else 1e-30
            _, dJf = _hermite(ufs, J_tab[2], J_tab[3], du)
            duf = 0.5 * gF * gF * phi / (T * T * ufs)
            d += pref * nf * dJf * duf
        if include_cw > 0.5 and lam8 != 0.0:
            m2 = 3.0 * lam8 * p2
            if m2 > 0.0:
                dm2 = 6.0 * lam8 * phi
                d += dm2 * m2 / (64.0 * np.pi**2) * 2.0 * (math.log(m2 / T) - 1.0)
    return d


def pack_set8(pot: TwoStepPotential, bath, tab: dict | None = None):
    """Pack tree coef + set8 bath for dV_set8_nb."""
    tab = tab or build_J_table()
    coef = np.array([pot.c2, pot.c3, pot.c4, pot.c6], dtype=np.float64)
    J_tab = np.vstack([tab["Jb"], tab["dJb"], tab["Jf"], tab["dJf"]]).astype(np.float64)
    du = float(tab["u_max"]) / (int(tab["n"]) - 1)
    params = np.array([
        float(bath.nb), float(bath.nf), float(bath.gB), float(bath.gF),
        float(bath.g_gauge), float(bath.mb2_0), float(bath.lam_set8),
        1.0 if bath.include_cw else 0.0,
    ], dtype=np.float64)
    return coef, params, J_tab, du
