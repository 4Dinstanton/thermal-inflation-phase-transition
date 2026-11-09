#!/usr/bin/env python3
"""Bounce actions and nucleation for the two-step (0 -> a -> v) potential.

Channels, for every temperature T of the scan:

    0->a        O(3) thermal bounce, thermal false vacuum -> intermediate minimum
    a->v        O(3) thermal bounce, intermediate -> global minimum
    0->v        O(3) direct bounce in the same potential. Only defined while
                V(a, T) > V(0, T); once a is deeper than 0, any 1D path 0 -> v
                passes through the a-well, so a direct bounce does not exist.
    0->v (1-step)  O(3) bounce of the single-step reference potential
                V0 - m^2 phi^2/2 + phi^6/(6 Lambda^2) with the same V0, v, bath.

plus the T = 0 O(4) vacuum bounce a -> v (relevant when the a -> v barrier
sits outside the thermal core and S3 is T independent).

Rates: Gamma_3 = T^4 (S3/2piT)^{3/2} e^{-S3/T},  Gamma_4 = (S4/2pi)^2 R^-4 e^{-S4}.
Nucleation: Gamma = H^4, H^2 = (rho_rad + V_false)/(3 M_Pl^2).
Percolation of step 1: I_1(T) = 0.34 (paper Eq. 9 written in T, radiation-era
time-temperature relation dt = -dT/(H T)).

The bounce is found by overshoot/undershoot shooting in scaled units
x = (phi - phi_f)/Phi, rho = r sqrt(E)/Phi, starting from the analytic
linearised solution about the true minimum for thin walls; the action uses the
Derrick identity S_d = (Omega_d/d) int r^{d-1} phi'^2 dr.

Example
-------
    python analysis/twoStepBounce.py --V0 1e28 --a 3e9 --v 1e11 --delV1_frac 1e-6 \
        --boson 10,1 --Tmin 100 --Tmax 3e5 --nT 40 --logT --out data/twoStep/demo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
from numba import njit
from scipy.interpolate import PchipInterpolator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from potential.twoStepNumba import dV_nb, pack  # noqa: E402
from potential.twoStepPotential import SingleStepPotential, TwoStepPotential  # noqa: E402

PERC_I = 0.34
CHANNELS = ("0->a", "a->v", "0->v", "0->v (1-step)")


# --------------------------------------------------------------------------
# Shooting (numba)
# --------------------------------------------------------------------------
@njit(cache=True)
def _lin(z, d):
    """f(z) = Gamma(nu+1) (2/z)^nu I_nu(z), nu = d/2 - 1, and f'(z).

    delta(rho) = delta0 f(m rho) solves the linearised d-dim bounce equation.
    """
    if d == 3:
        if z < 1e-4:
            return 1.0 + z * z / 6.0, z / 3.0
        return math.sinh(z) / z, (z * math.cosh(z) - math.sinh(z)) / (z * z)
    # d = 4: f = sum (z/2)^{2k} / (k! (k+1)!)
    q = 0.25 * z * z
    term = 1.0
    f = 1.0
    fp = 0.0
    k = 0
    while True:
        k += 1
        term *= q / (k * (k + 1.0))
        f += term
        fp += 2.0 * k * term / z if z > 0 else 0.0
        if term < 1e-17 * f and k > 2:
            break
    return f, fp


@njit(cache=True)
def _rhs(rho, x, xp, alpha, sfac, phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du):
    dv = dV_nb(phi_f + Phi * x, T, coef, sp_n, sp_g, sp_f, J_tab, du) * Phi / E
    return xp, dv - alpha * xp / rho, sfac * rho**alpha * xp * xp


@njit(cache=True)
def _shoot(x0, xt, m2, rho_max, d, phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du,
           lin_frac, rtol):
    """Integrate from x(0) = x0 toward the false vacuum x = 0.

    Returns (code, S_tilde, rho_end): +1 overshoot (crossed x = 0),
    -1 undershoot (turned back), 0 neither by rho_max.
    """
    alpha = d - 1.0
    sfac = 4.0 * np.pi / 3.0 if d == 3 else np.pi**2 / 2.0
    delta0 = x0 - xt
    if delta0 == 0.0:
        return 0, 0.0, 0.0
    target = lin_frac * xt
    S = 0.0
    if m2 > 0.0 and abs(delta0) < target:
        m = math.sqrt(m2)
        ratio = target / abs(delta0)
        lo, hi = 0.0, 1.0
        while _lin(hi, d)[0] < ratio:
            hi *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _lin(mid, d)[0] < ratio:
                lo = mid
            else:
                hi = mid
        z = 0.5 * (lo + hi)
        f, fp = _lin(z, d)
        rho = z / m
        x = xt + delta0 * f
        xp = delta0 * m * fp
        # action of the linear core, int rho^alpha xp^2 (trapezoid on 400 points)
        n = 400
        prev = 0.0
        for j in range(1, n + 1):
            zj = z * j / n
            fj, fpj = _lin(zj, d)
            cur = sfac * (zj / m) ** alpha * (delta0 * m * fpj) ** 2
            S += 0.5 * (prev + cur) * (z / n / m)
            prev = cur
    else:
        rho = 1e-6
        a0 = dV_nb(phi_f + Phi * x0, T, coef, sp_n, sp_g, sp_f, J_tab, du) * Phi / E
        x = x0 + a0 * rho * rho / (2.0 * d)
        xp = a0 * rho / d

    h = max(rho * 1e-3, 1e-6)
    c2, c3, c4, c5 = 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9
    a21 = 1.0 / 5
    a31, a32 = 3.0 / 40, 9.0 / 40
    a41, a42, a43 = 44.0 / 45, -56.0 / 15, 32.0 / 9
    a51, a52, a53, a54 = 19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729
    a61, a62, a63, a64, a65 = 9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656
    b1, b3, b4, b5, b6 = 35.0 / 384, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84
    e1, e3, e4, e5, e6, e7 = (71.0 / 57600, -71.0 / 16695, 71.0 / 1920,
                              -17253.0 / 339200, 22.0 / 525, -1.0 / 40)
    for _ in range(5000000):
        if rho >= rho_max:
            return 0, S, rho
        k1x, k1p, k1s = _rhs(rho, x, xp, alpha, sfac, phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        k2x, k2p, k2s = _rhs(rho + c2 * h, x + h * a21 * k1x, xp + h * a21 * k1p, alpha, sfac,
                             phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        k3x, k3p, k3s = _rhs(rho + c3 * h, x + h * (a31 * k1x + a32 * k2x),
                             xp + h * (a31 * k1p + a32 * k2p), alpha, sfac,
                             phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        k4x, k4p, k4s = _rhs(rho + c4 * h, x + h * (a41 * k1x + a42 * k2x + a43 * k3x),
                             xp + h * (a41 * k1p + a42 * k2p + a43 * k3p), alpha, sfac,
                             phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        k5x, k5p, k5s = _rhs(rho + c5 * h,
                             x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x),
                             xp + h * (a51 * k1p + a52 * k2p + a53 * k3p + a54 * k4p), alpha, sfac,
                             phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        k6x, k6p, k6s = _rhs(rho + h,
                             x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x),
                             xp + h * (a61 * k1p + a62 * k2p + a63 * k3p + a64 * k4p + a65 * k5p),
                             alpha, sfac, phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        xn = x + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
        pn = xp + h * (b1 * k1p + b3 * k3p + b4 * k4p + b5 * k5p + b6 * k6p)
        sn = S + h * (b1 * k1s + b3 * k3s + b4 * k4s + b5 * k5s + b6 * k6s)
        k7x, k7p, _ = _rhs(rho + h, xn, pn, alpha, sfac, phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f, J_tab, du)
        ex = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x)
        ep = h * (e1 * k1p + e3 * k3p + e4 * k4p + e5 * k5p + e6 * k6p + e7 * k7p)
        sc_x = rtol * max(abs(x), abs(xn), 1e-3) + 1e-14
        sc_p = rtol * max(abs(xp), abs(pn), 1e-6) + 1e-14
        err = math.sqrt(0.5 * ((ex / sc_x) ** 2 + (ep / sc_p) ** 2))
        if err <= 1.0 or h < 1e-12:
            if xn < 0.0:
                return 1, sn, rho + h
            if pn > 0.0:
                return -1, sn, rho + h
            rho += h
            x, xp, S = xn, pn, sn
        fac = 0.9 * err ** (-0.2) if err > 0 else 5.0
        h *= min(5.0, max(0.2, fac))
    return 0, S, rho


# --------------------------------------------------------------------------
# Python driver
# --------------------------------------------------------------------------
@dataclass
class Bounce:
    S: float  # S3 (d=3, GeV) or S4 (d=4, dimensionless)
    d: int
    phi_f: float
    phi_t: float
    phi_center: float
    radius: float  # turning radius of the converged profile (GeV^-1)
    converged: bool
    method: str = "shooting"  # or "thin-wall" when the wall is too thin for doubles


def bounce_action(pot: TwoStepPotential, T: float, phi_f: float, phi_t: float, d: int = 3,
                  packed=None, n_bisect: int = 200, lin_frac: float = 1e-3,
                  rtol: float = 1e-9) -> Bounce | None:
    """O(d) bounce from the minimum phi_f to the deeper minimum phi_t at temperature T.

    Returns None if phi_t is not deeper or no barrier separates the two.
    """
    packed = packed or pack(pot)
    V_f = float(pot.V_rel(phi_f, T))
    V_t = float(pot.V_rel(phi_t, T))
    if not V_t < V_f:
        return None
    span = phi_t - phi_f
    s = np.unique(np.concatenate([np.geomspace(1e-14, 0.5, 6000),
                                  1.0 - np.geomspace(1e-14, 0.5, 6000)]))
    grid = phi_f + span * s
    Vg = pot.V_rel(grid, T) - V_f
    if Vg.max() <= 0.0:
        return None
    E = float(Vg.max())
    above = np.nonzero(Vg >= 0.0)[0]
    j = int(above[-1])
    g0, g1, v0, v1 = grid[j], grid[j + 1], Vg[j], Vg[j + 1]
    phi_bar = g0 + (g1 - g0) * v0 / (v0 - v1)
    Phi = phi_bar - phi_f
    xt = span / Phi
    hstep = 1e-6 * abs(span)
    m2 = ((float(pot.dV(phi_t + hstep, T)) - float(pot.dV(phi_t - hstep, T))) / (2 * hstep)
          * Phi**2 / E)
    coef, sp_n, sp_g, sp_f, J_tab, du = packed
    rho_max = 1e4 * max(1.0, 1.0 / math.sqrt(m2) if m2 > 0 else 1.0) * max(1.0, xt)

    def x_of(y):
        return 1.0 + (xt - 1.0) * (-math.expm1(-y))

    def run(y):
        return _shoot(x_of(y), xt, m2, rho_max, d, phi_f, Phi, E, T, coef, sp_n, sp_g, sp_f,
                      J_tab, du, lin_frac, rtol)

    y_lo, y_hi = 0.0, 1.0
    while run(y_hi)[0] != 1 and y_hi < 700:
        y_lo, y_hi = y_hi, 2.0 * y_hi
    if run(y_hi)[0] != 1:
        # Start point indistinguishable from phi_t: thin-wall limit.
        eps = V_f - V_t
        sigma = float(np.trapz(np.sqrt(2.0 * np.clip(Vg[: j + 2], 0.0, None)), grid[: j + 2]))
        R = (d - 1) * sigma / eps
        S_tw = (16 * np.pi / 3 * sigma**3 / eps**2 if d == 3
                else 27 * np.pi**2 / 2 * sigma**4 / eps**3)
        return Bounce(S_tw, d, phi_f, phi_t, phi_t, R, True, "thin-wall")
    best = None
    for _ in range(n_bisect):
        y_mid = 0.5 * (y_lo + y_hi)
        code, S, rho_end = run(y_mid)
        if code == 1:
            y_hi = y_mid
        else:
            y_lo = y_mid
            best = (y_mid, S, rho_end)
        if y_hi - y_lo <= 1e-15 * max(1.0, y_hi):
            break
    if best is None:
        code, S, rho_end = run(y_hi)
        best = (y_hi, S, rho_end)
    y, S_tilde, rho_end = best
    L = Phi / math.sqrt(E)
    S_phys = S_tilde * Phi**d * E ** ((2 - d) / 2.0)
    return Bounce(S_phys, d, phi_f, phi_t, phi_f + Phi * x_of(y), rho_end * L, True)


# --------------------------------------------------------------------------
# Temperature scan
# --------------------------------------------------------------------------
def _point(pot, ref, T, packed, packed_ref):
    vac = pot.vacua(T)
    f, i, t = vac["false"], vac["inter"], vac["true"]
    row = {"T": T, "false_is_min": f is not None,
           "phi_inter": i.phi if i else np.nan, "phi_true": t.phi if t else np.nan,
           "V_inter": i.V_rel if i else np.nan, "V_true": t.V_rel if t else np.nan}
    direct_ok = f is not None and t is not None and (i is None or i.V_rel >= 0.0)
    pairs = {"0->a": (f, i) if (f and i) else None,
             "a->v": (i, t) if (i and t) else None,
             "0->v": (f, t) if direct_ok else None}
    for c, pr in pairs.items():
        b = bounce_action(pot, T, pr[0].phi, pr[1].phi, packed=packed) if pr else None
        row[f"S3/T {c}"] = b.S / T if (b and b.converged) else np.nan
        row[f"phi_center {c}"] = b.phi_center if b else np.nan
    c = "0->v (1-step)"
    row[f"S3/T {c}"], row[f"phi_center {c}"] = np.nan, np.nan
    if ref is not None:
        rv = ref.vacua(T)
        if rv["false"] and rv["true"]:
            b = bounce_action(ref, T, 0.0, rv["true"].phi, packed=packed_ref)
            if b and b.converged:
                row[f"S3/T {c}"], row[f"phi_center {c}"] = b.S / T, b.phi_center
    return row


def scan(pot: TwoStepPotential, Ts, ref: SingleStepPotential | None = None,
         refine: int = 30, verbose: bool = True) -> dict:
    """Coarse scan over Ts, then ``refine`` extra points where each S3/T crosses
    the nucleation threshold ~ 4 ln(T/H)."""
    packed = pack(pot)
    packed_ref = pack(ref) if ref is not None else None
    rows = []

    def add(T):
        r = _point(pot, ref, float(T), packed, packed_ref)
        rows.append(r)
        if verbose:
            print(f"T={T:11.5g}  " + "  ".join(f"{c}: {r[f'S3/T {c}']:10.4g}" for c in CHANNELS),
                  flush=True)

    for T in Ts:
        add(T)
    if refine:
        coarse = sorted(rows, key=lambda r: r["T"])
        Tc = np.array([r["T"] for r in coarse])
        for c in CHANNELS:
            V_false = pot.V0 - (pot.delV1 if c == "a->v" else 0.0)
            S = np.array([r[f"S3/T {c}"] for r in coarse])
            thr = 4 * np.log(Tc / pot.hubble(Tc, V_false))
            g = S - thr
            for k in range(len(Tc) - 1):
                a_ok, b_ok = np.isfinite(g[k]), np.isfinite(g[k + 1])
                cross = (a_ok and b_ok and g[k] * g[k + 1] <= 0) or (a_ok != b_ok and (
                    (a_ok and g[k] < 3 * thr[k]) or (b_ok and g[k + 1] < 3 * thr[k + 1])))
                if cross:
                    lo = Tc[max(k - 1, 0)]
                    hi = Tc[min(k + 2, len(Tc) - 1)]
                    for T in np.linspace(lo, hi, refine + 2)[1:-1]:
                        add(T)
    rows.sort(key=lambda r: -r["T"])
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


# --------------------------------------------------------------------------
# Rates, nucleation, percolation
# --------------------------------------------------------------------------
def log_gamma3(T, S_over_T):
    return 4 * np.log(T) + 1.5 * np.log(S_over_T / (2 * np.pi)) - S_over_T


def log_gamma4(b: Bounce | None) -> float:
    if b is None or not b.converged or not np.isfinite(b.S):
        return -np.inf
    return 2 * np.log(b.S / (2 * np.pi)) - 4 * np.log(b.radius) - b.S


def _sorted_valid(res, channel):
    T = res["T"]
    S = res[f"S3/T {channel}"]
    ok = np.isfinite(S) & (S > 0)
    Tu, idx = np.unique(T[ok], return_index=True)
    return Tu, S[ok][idx]


def nucleation(pot, res, channel, V_false, lnG4=-np.inf, T_max=np.inf) -> dict:
    """Largest T <= T_max with Gamma(T) >= H(T)^4; Gamma = Gamma_3 (+ Gamma_4 if given).

    T_max is the temperature at which the false phase of this channel first
    exists (T_n1 for a -> v, since the a-phase only appears after step 1).
    """
    Tg, Sg = _sorted_valid(res, channel)
    out = {"channel": channel, "T_n": float("nan")}
    if len(Tg) < 2 and not np.isfinite(lnG4):
        return out
    # Ultra-thin-wall / spinodal (S3/T ≪ 1): the T^4 (S3/T)^{3/2} e^{-S3/T}
    # prefactor underestimates the rate (→0 as S→0).  Treat as immediate decay
    # at T_max (or the highest T where the false phase exists).
    if len(Tg) >= 2 and float(np.median(Sg)) < 1.0:
        T_all = np.unique(res["T"][np.isfinite(res["phi_inter" if channel == "a->v" else "T"])])
        if np.isfinite(T_max):
            T_n = float(min(np.nanmax(T_all) if len(T_all) else T_max, T_max))
        else:
            T_n = float(np.nanmax(Tg))
        out["T_n"] = T_n
        out["note"] = ("Gamma >= H^4 as soon as the false phase exists (immediate decay)")
        out["H(T_n)"] = float(pot.hubble(T_n, V_false))
        out["rho_rad(T_n)"] = float(pot.rho_rad(T_n))
        out["dominant"] = "O(3) thermal"
        out["S3/T(T_n)"] = float(np.median(Sg))
        out["beta/H"] = float("nan")
        return out
    T_all = np.unique(res["T"][np.isfinite(res["phi_inter" if channel == "a->v" else "T"])])
    if np.isfinite(T_max):
        T_all = np.unique(np.append(T_all[T_all < T_max], T_max))
    lnG3 = np.full(len(T_all), -np.inf)
    if len(Tg) >= 2:
        spl = PchipInterpolator(Tg, np.log(Sg), extrapolate=False)
        s = np.exp(spl(T_all))
        ok = np.isfinite(s)
        lnG3[ok] = log_gamma3(T_all[ok], s[ok])
    lnG = np.logaddexp(lnG3, lnG4)
    f = lnG - 4 * np.log(pot.hubble(T_all, V_false))
    idx = np.nonzero(f >= 0)[0]
    if len(idx) == 0:
        out["max ln(Gamma/H^4)"] = float(np.max(f)) if len(f) else float("nan")
        return out
    k = int(idx[-1])
    if k + 1 < len(T_all) and np.isfinite(f[k + 1]):
        T_n = T_all[k] + (T_all[k + 1] - T_all[k]) * f[k] / (f[k] - f[k + 1])
        out["T_n"] = float(T_n)
    else:
        out["T_n"] = float(T_all[k])
        out["note"] = ("Gamma >= H^4 as soon as the false phase exists (immediate decay)"
                       if np.isfinite(T_max) and T_all[k] == T_max else
                       "Gamma >= H^4 already at the highest scanned T")
    T_n = out["T_n"]
    out["H(T_n)"] = float(pot.hubble(T_n, V_false))
    out["rho_rad(T_n)"] = float(pot.rho_rad(T_n))
    dom3 = len(Tg) >= 2 and Tg[0] <= T_n <= Tg[-1] and \
        log_gamma3(T_n, float(np.exp(spl(T_n)))) >= lnG4
    out["dominant"] = "O(3) thermal" if dom3 else "O(4) vacuum"
    if dom3:
        out["S3/T(T_n)"] = float(np.exp(spl(T_n)))
        out["beta/H"] = float(T_n * np.exp(spl(T_n)) * spl.derivative()(T_n))
    else:
        out["beta/H"] = float("nan")
        out["note"] = ("T-independent O(4) rate: beta/H = T d(S/T)/dT does not apply; "
                       "the transition completes within ~1 Hubble time once Gamma_4 > H^4")
    return out


def percolation_T(pot, res, channel, V_false, v_w=1.0, n=3000) -> float:
    """Largest T with I(T) >= 0.34 on a fine grid where S3/T < 1e3."""
    Tg, Sg = _sorted_valid(res, channel)
    keep = Sg < 1e3
    if keep.sum() < 2:
        return float("nan")
    i0 = max(np.nonzero(keep)[0][0] - 1, 0)
    i1 = min(np.nonzero(keep)[0][-1] + 1, len(Tg) - 1)
    Tg, Sg = Tg[i0:i1 + 1], Sg[i0:i1 + 1]
    spl = PchipInterpolator(Tg, np.log(Sg))
    Tf = np.linspace(Tg[0], Tg[-1], n)
    S_f = np.exp(spl(Tf))
    H_f = pot.hubble(Tf, V_false)
    w = np.exp(log_gamma3(Tf, S_f)) / (H_f * Tf**4)
    invH = 1.0 / H_f
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (invH[1:] + invH[:-1]) * np.diff(Tf))])
    I = np.zeros(n)
    for k in range(n - 1):
        r = v_w * (cum[k:] - cum[k])
        I[k] = 4 * np.pi / 3 * np.trapz(w[k:] * r**3, Tf[k:])
    above = np.nonzero(I >= PERC_I)[0]
    if len(above) == 0:
        return float("nan")
    k = int(above[-1])
    if k + 1 < n:
        return float(Tf[k] + (Tf[k + 1] - Tf[k]) * (I[k] - PERC_I) / (I[k] - I[k + 1]))
    return float(Tf[k])


def analyze(pot, res, ref=None, b4=None) -> dict:
    V1, V2 = pot.V0, pot.V0 - pot.delV1
    lnG4 = log_gamma4(b4)
    step1 = nucleation(pot, res, "0->a", V1)
    step1["T_perc"] = percolation_T(pot, res, "0->a", V1)
    step2 = nucleation(pot, res, "a->v", V2, lnG4=lnG4, T_max=step1["T_n"])
    direct = nucleation(pot, res, "0->v", V1)
    single = nucleation(ref or pot, res, "0->v (1-step)", V1) if ref is not None else {}
    for d_, dv in ((step1, pot.delV1), (step2, pot.delV2), (direct, pot.V0), (single, pot.V0)):
        if d_:
            d_["delV"] = dv
            if np.isfinite(d_.get("T_n", np.nan)):
                d_["alpha"] = dv / d_["rho_rad(T_n)"]
    if b4 is not None:
        step2["S4 (T=0)"] = b4.S
        step2["R4 (T=0) [1/GeV]"] = b4.radius
        step2["ln(Gamma4/H^4) at V_false=V0-delV1"] = float(lnG4 - 4 * np.log(pot.hubble(0.0, V2)))
    T_n1, T_p1, T_n2 = step1["T_n"], step1["T_perc"], step2["T_n"]
    if not np.isfinite(T_n1):
        regime = "step 1 never nucleates in the scanned T range"
    elif not np.isfinite(T_n2):
        regime = "step 2 (a->v) never nucleates: universe stuck in the intermediate vacuum"
    elif "immediate" in step2.get("note", ""):
        regime = ("Regime II (nested bubbles): a->v already faster than H^4 when a-bubbles form "
                  "(T_n2 = T_n1)")
    elif np.isfinite(T_p1) and T_p1 > T_n2:
        regime = "Regime I (well separated): T_n1 > T_perc,1 > T_n2"
    else:
        regime = "Regime II (nested bubbles): T_n2 >~ T_perc,1"
    return {"step1": step1, "step2": step2, "direct (same potential)": direct,
            "single-step reference": single, "regime": regime}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_potential(pot: TwoStepPotential, Ts, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
    phi = np.linspace(0, 1.15 * pot.v, 4000)
    axes[0].plot(phi / pot.v, pot.V_tree(phi) / pot.V0, "k", lw=1.5)
    axes[0].axvline(pot.a / pot.v, ls=":", c="C1", label=r"$a$")
    axes[0].set(xlabel=r"$\phi/v$", ylabel=r"$V_{\rm tree}/V_0$", title="Tree potential")
    axes[0].legend()
    b = pot.validate_tree()["b"]
    phi2 = np.linspace(0, min(2.0 * (b if np.isfinite(b) else pot.a), pot.v), 4000)
    axes[1].plot(phi2 / pot.a, pot.V_tree_rel(phi2) / pot.delV1, "k", lw=1.5)
    axes[1].axhline(-1, ls=":", c="gray")
    axes[1].set(xlabel=r"$\phi/a$", ylabel=r"$(V_{\rm tree}-V_0)/\Delta V_1$",
                title=r"Tree potential near $a$ (barrier $b$ to $v$)")
    cmap = plt.cm.viridis
    gmax = max(s.g for s in pot.species)
    for k, T in enumerate(Ts):
        ph = np.geomspace(1e-3 * T / gmax, 3 * pot.a, 4000)
        axes[2].semilogx(ph, pot.V_rel(ph, T) / T**4, color=cmap(k / max(1, len(Ts) - 1)),
                         label=f"T = {T:.4g} GeV")
    axes[2].axhline(0, c="gray", lw=0.8)
    axes[2].set(xlabel=r"$\phi$ [GeV]", ylabel=r"$[V(\phi,T)-V(0,T)]/T^4$",
                title="Finite-T potential: thermal barrier and the a-well")
    axes[2].set_ylim(-3, 3)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actions(pot, res, info, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    T = res["T"]
    o = np.argsort(T)
    T = T[o]
    style = {"0->a": ("C0", "o-"), "a->v": ("C3", "s-"), "0->v": ("k", "^-"),
             "0->v (1-step)": ("C2", "d--")}
    for c in CHANNELS:
        col, ls = style[c]
        y = res[f"S3/T {c}"][o]
        if np.isfinite(y).any():
            ax[0].loglog(T, y, ls, ms=2.5, color=col, label=c)
    ax[0].loglog(T, 4 * np.log(T / pot.hubble(T, pot.V0)), ":", c="gray", lw=1.2,
                 label=r"$4\ln(T/H)$ (nucleation threshold)")
    s4 = info["step2"].get("S4 (T=0)")
    if s4 is not None and np.isfinite(s4):
        ax[0].loglog(T, np.full_like(T, s4), "-.", c="C3", lw=1, label=r"$S_4$ (a->v, T=0)")
    for key, col in (("step1", "C0"), ("step2", "C3"), ("single-step reference", "C2")):
        Tn = info.get(key, {}).get("T_n", np.nan)
        if np.isfinite(Tn):
            ax[0].axvline(Tn, ls=":", color=col, lw=1.2)
    Tp = info["step1"].get("T_perc", np.nan)
    if np.isfinite(Tp):
        ax[0].axvline(Tp, ls="-.", color="C0", lw=1, label=r"$T_{\rm perc,1}$")
    ax[0].set(xlabel="T [GeV]", ylabel=r"$S_3/T$  (or $S_4$)", title="Bounce action vs temperature")
    ax[0].set_ylim(1, None)
    ax[0].legend(fontsize=8)
    ax[1].loglog(T, res["phi_inter"][o], "C0-", label=r"$\phi_{\rm inter}(T)$")
    ax[1].loglog(T, res["phi_true"][o], "C3-", label=r"$\phi_{\rm true}(T)$")
    for c in CHANNELS:
        col, _ = style[c]
        y = res[f"phi_center {c}"][o]
        if np.isfinite(y).any():
            ax[1].loglog(T, y, ":", color=col, label=f"bounce centre {c}")
    ax[1].set(xlabel="T [GeV]", ylabel=r"$\phi$ [GeV]", title="Minima and bounce-centre field")
    ax[1].legend(fontsize=8)
    fig.suptitle(info["regime"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _species(items):
    return [tuple(float(x) for x in s.split(",")) for s in (items or [])]


def build_parser():
    p = argparse.ArgumentParser(description="Two-step potential: bounce actions and nucleation")
    p.add_argument("--V0", type=float, required=True, help="V(0) at T=0 [GeV^4]")
    p.add_argument("--a", type=float, required=True, help="intermediate minimum [GeV]")
    p.add_argument("--v", type=float, required=True, help="global minimum [GeV]")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--delV1", type=float, help="V0 - V(a) [GeV^4]")
    g.add_argument("--delV1_frac", type=float, help="(V0 - V(a)) / V0")
    p.add_argument("--boson", action="append", help="n,g: boson with m = g phi (repeatable; default 10,1)")
    p.add_argument("--fermion", action="append", help="n,g: fermion with m = g phi (repeatable)")
    p.add_argument("--g_star", type=float, default=100.0)
    p.add_argument("--Tmin", type=float, required=True)
    p.add_argument("--Tmax", type=float, required=True)
    p.add_argument("--nT", type=int, default=50)
    p.add_argument("--logT", action="store_true", help="log-spaced temperature grid")
    p.add_argument("--refine", type=int, default=30, help="extra points around each threshold crossing")
    p.add_argument("--no_reference", action="store_true", help="skip the single-step reference model")
    p.add_argument("--out", required=True, help="output directory")
    return p


def run(args) -> dict:
    bosons = _species(args.boson) or [(10.0, 1.0)]
    fermions = _species(args.fermion)
    pot = TwoStepPotential.from_vacua(V0=args.V0, a=args.a, v=args.v, delV1=args.delV1,
                                      delV1_frac=args.delV1_frac, bosons=bosons,
                                      fermions=fermions, g_star=args.g_star)
    ref = None if args.no_reference else SingleStepPotential.from_vacuum(
        args.V0, args.v, bosons=bosons, fermions=fermions, g_star=args.g_star)
    print(pot.summary())
    if not pot.validate_tree()["ok"]:
        print("WARNING: tree potential lacks the max(0) < min(a) < max(b) < min(v) shape; "
              "increase delV1 or decrease a/v.")
    os.makedirs(args.out, exist_ok=True)
    b4 = bounce_action(pot, 0.0, pot.a, pot.v, d=4)
    print(f"O(4) a->v at T=0: S4 = {b4.S:.6g}, R = {b4.radius:.4g} /GeV" if b4 else "O(4): none")
    Ts = (np.geomspace if args.logT else np.linspace)(args.Tmax, args.Tmin, args.nT)
    res = scan(pot, Ts, ref=ref, refine=args.refine)
    info = analyze(pot, res, ref=ref, b4=b4)

    import pandas as pd

    pd.DataFrame(res).to_csv(os.path.join(args.out, "bounce_scan.csv"), index=False)
    val = pot.validate_tree()
    meta = {
        "potential": {"V0": pot.V0, "a": pot.a, "v": pot.v, "delV1": pot.delV1,
                      "c2": pot.c2, "c3": pot.c3, "c4": pot.c4, "c6": pot.c6,
                      "m0_sq": pot.m0_sq, "mu3": pot.mu3, "lambda": pot.lam, "Lambda": pot.Lambda,
                      "T0_spinodal": pot.T0_spinodal,
                      "species": [asdict(s) for s in pot.species], "g_star": pot.g_star,
                      "tree_ok": bool(val["ok"]), "b": float(val["b"])},
        "nucleation": info,
    }
    with open(os.path.join(args.out, "nucleation.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    lo = info["step1"].get("T_n", np.nan)
    Tsel = np.geomspace(args.Tmax, args.Tmin, 6)
    if np.isfinite(lo):
        Tsel = np.sort(np.concatenate([Tsel[[0, 2, 5]], [1.3 * lo, lo, 0.9 * lo]]))[::-1]
    plot_potential(pot, Tsel, os.path.join(args.out, "potential.png"))
    plot_actions(pot, res, info, os.path.join(args.out, "bounce_actions.png"))
    print(json.dumps(info, indent=2, default=float))
    print(f"wrote {args.out}/bounce_scan.csv, nucleation.json, potential.png, bounce_actions.png")
    return meta


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
