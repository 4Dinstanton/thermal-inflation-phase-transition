#!/usr/bin/env python3
"""Two-field portal model: potential shape, bounce actions, nucleation and SGWB.

Vacua (potential/portalPotential.py): O = (0, 0), A = (0, s0), B = (v, 0) with a
cross term (kappa) or B = (v, s0) without one.

Channels at each temperature T:

    0->a   O -> A along the S axis (exactly 1D: phi = 0 is a symmetry line)
    0->v   O -> (v, 0) along the phi axis (direct flaton transition, single-step reference)
    a->v   A -> B. With the cross term this is a genuine 2D bounce (cosmoTransitions path
           deformation). Without it the fields decouple and A -> B is the same phi-axis
           bounce as 0->v (the S value is irrelevant).

plus the T = 0 O(4) bounce A -> B.

1D bounces use the validated numba shooting of analysis/twoStepBounce.py. The 2D
bounce runs in units F = s_T (field) and E = eps V0 (energy), where S_d = F^d E^(1-d/2)
S_tilde: (i) overshoot/undershoot shooting along the valley path (S minimising V at each
phi; it only has to reach the bounce centre, so B = (v, 0), ~v/s0 field units away, is
never needed), then (ii) Newton refinement of the full 2D equations with solve_bvp.
``--check_2d`` prints the valley-path action next to the refined one and checks the
dependence on the path length.

Example
-------
    # phi sector defaults = set8 (V0=2.5e35, v=1e15, nb=nf=20, g=1.09)
    python analysis/portalBounce.py --s0 2e3 --eps 8e-24 --kappa 2 --workers 8 \\
        --boson_S 4,1 --out data/portal/cross_k2
    python analysis/portalBounce.py ... --no_cross --out data/portal/nocross
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
from scipy.optimize import brentq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.twoStepBounce import (Bounce, bounce_action, log_gamma4, nucleation,  # noqa: E402
                                    percolation_T)
from potential.portalPotential import (  # noqa: E402
    PhiAxisSlice,
    PortalPotential,
    make_species,
)

# set8 (run_cosmolattice / gamma_sets): γ=4.1667e-4, m_φ=1000 GeV
#   φ₀ = γ M_Pl ≈ 1e15 GeV,  V₀ = m_φ² φ₀² / 4 ≈ 2.5e35 GeV⁴
#   bath: nb=nf=20, gB=gF=1.09
SET8_V0 = 2.5e35
SET8_V = 1.0e15
SET8_BOSON_PHI = [(20.0, 1.09)]
SET8_FERMION_PHI = [(20.0, 1.09)]

# Process-pool worker state (set once per child via initializer).
_WORKER_POT: PortalPotential | None = None
_WORKER_DO_2D: bool = True


def _init_worker(pot: PortalPotential, do_2d: bool = True):
    global _WORKER_POT, _WORKER_DO_2D
    _WORKER_POT = pot
    _WORKER_DO_2D = do_2d


def _point_worker(T: float) -> dict:
    return _point(_WORKER_POT, float(T), do_2d=_WORKER_DO_2D)

warnings.filterwarnings("ignore", category=RuntimeWarning)
CHANNELS = ("0->a", "a->v", "0->v")


def bounce_action_set8(pot: PhiAxisSlice, T: float, phi_f: float, phi_t: float, d: int = 3,
                       n_bisect: int = 200, lin_frac: float = 1e-3, rtol: float = 1e-9):
    """O(d) bounce for set8 / V_correct phi thermal (numba Debye+CW force)."""
    from numba import njit

    from potential.twoStepNumba import dV_set8_nb, pack_set8

    # reuse the same RK45 shooter as twoStepBounce, with dV_set8_nb
    from analysis.twoStepBounce import _lin  # numba

    V_f = float(pot.V_rel(phi_f, T))
    V_t = float(pot.V_rel(phi_t, T))
    if not V_t < V_f:
        return None
    span = phi_t - phi_f
    lo = max((T if T > 0 else 1.0) * 1e-4, 1.0)
    s = np.unique(np.concatenate([
        [0.0],
        np.geomspace(lo / abs(span), 0.5, 4000),
        1.0 - np.geomspace(1e-14, 0.5, 4000),
    ]))
    grid = phi_f + span * s
    Vg = np.asarray(pot.V_rel(grid, T), float) - V_f
    if not np.isfinite(Vg).all() or Vg.max() <= 0.0:
        return None
    E = float(Vg.max())
    above = np.nonzero(Vg >= 0.0)[0]
    j = int(above[-1])
    if j + 1 >= len(grid):
        return None
    g0, g1, v0, v1 = grid[j], grid[j + 1], float(Vg[j]), float(Vg[j + 1])
    phi_bar = g0 + (g1 - g0) * v0 / (v0 - v1) if v0 != v1 else g0
    Phi = phi_bar - phi_f
    if not (Phi > 0 and E > 0):
        return None
    xt = span / Phi
    bath = getattr(pot, "bath", None) or pot.portal.set8
    coef, params, J_tab, du = pack_set8(pot, bath)
    nb, nf, gB, gF, gg, mb20, lam8, icw = params

    hstep = 1e-6 * abs(span)
    m2 = ((dV_set8_nb(phi_t + hstep, T, coef, nb, nf, gB, gF, gg, mb20, lam8, icw, J_tab, du)
           - dV_set8_nb(phi_t - hstep, T, coef, nb, nf, gB, gF, gg, mb20, lam8, icw, J_tab, du))
          / (2 * hstep) * Phi**2 / E)
    rho_max = 1e4 * max(1.0, 1.0 / math.sqrt(m2) if m2 > 0 else 1.0) * max(1.0, xt)

    @njit(cache=True)
    def _rhs(rho, x, xp, alpha, sfac, phi_f_, Phi_, E_, T_,
             coef_, nb_, nf_, gB_, gF_, gg_, mb20_, lam8_, icw_, J_tab_, du_):
        dv = dV_set8_nb(phi_f_ + Phi_ * x, T_, coef_, nb_, nf_, gB_, gF_, gg_,
                        mb20_, lam8_, icw_, J_tab_, du_) * Phi_ / E_
        return xp, dv - alpha * xp / rho, sfac * rho**alpha * xp * xp

    @njit(cache=True)
    def _shoot_s8(x0, xt_, m2_, rho_max_, d_, phi_f_, Phi_, E_, T_,
                  coef_, nb_, nf_, gB_, gF_, gg_, mb20_, lam8_, icw_, J_tab_, du_,
                  lin_frac_, rtol_):
        alpha = d_ - 1.0
        sfac = 4.0 * np.pi / 3.0 if d_ == 3 else np.pi**2 / 2.0
        delta0 = x0 - xt_
        if delta0 == 0.0:
            return 0, 0.0, 0.0
        target = lin_frac_ * xt_
        S = 0.0
        if m2_ > 0.0 and abs(delta0) < target:
            m = math.sqrt(m2_)
            ratio = target / abs(delta0)
            lo_, hi_ = 0.0, 1.0
            while _lin(hi_, d_)[0] < ratio:
                hi_ *= 2.0
            for _ in range(100):
                mid = 0.5 * (lo_ + hi_)
                if _lin(mid, d_)[0] < ratio:
                    lo_ = mid
                else:
                    hi_ = mid
            z = 0.5 * (lo_ + hi_)
            f, fp = _lin(z, d_)
            rho = z / m
            x = xt_ + delta0 * f
            xp = delta0 * m * fp
            n = 400
            prev = 0.0
            for jj in range(1, n + 1):
                zj = z * jj / n
                fj, fpj = _lin(zj, d_)
                cur = sfac * (zj / m) ** alpha * (delta0 * m * fpj) ** 2
                S += 0.5 * (prev + cur) * (z / n / m)
                prev = cur
        else:
            rho = 1e-6
            a0 = dV_set8_nb(phi_f_ + Phi_ * x0, T_, coef_, nb_, nf_, gB_, gF_, gg_,
                            mb20_, lam8_, icw_, J_tab_, du_) * Phi_ / E_
            x = x0 + a0 * rho * rho / (2.0 * d_)
            xp = a0 * rho / d_

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
            if rho >= rho_max_:
                return 0, S, rho
            k1x, k1p, k1s = _rhs(rho, x, xp, alpha, sfac, phi_f_, Phi_, E_, T_,
                                 coef_, nb_, nf_, gB_, gF_, gg_, mb20_, lam8_, icw_, J_tab_, du_)
            k2x, k2p, k2s = _rhs(rho + c2 * h, x + h * a21 * k1x, xp + h * a21 * k1p, alpha, sfac,
                                 phi_f_, Phi_, E_, T_, coef_, nb_, nf_, gB_, gF_, gg_, mb20_,
                                 lam8_, icw_, J_tab_, du_)
            k3x, k3p, k3s = _rhs(rho + c3 * h, x + h * (a31 * k1x + a32 * k2x),
                                 xp + h * (a31 * k1p + a32 * k2p), alpha, sfac,
                                 phi_f_, Phi_, E_, T_, coef_, nb_, nf_, gB_, gF_, gg_, mb20_,
                                 lam8_, icw_, J_tab_, du_)
            k4x, k4p, k4s = _rhs(rho + c4 * h, x + h * (a41 * k1x + a42 * k2x + a43 * k3x),
                                 xp + h * (a41 * k1p + a42 * k2p + a43 * k3p), alpha, sfac,
                                 phi_f_, Phi_, E_, T_, coef_, nb_, nf_, gB_, gF_, gg_, mb20_,
                                 lam8_, icw_, J_tab_, du_)
            k5x, k5p, k5s = _rhs(rho + c5 * h,
                                 x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x),
                                 xp + h * (a51 * k1p + a52 * k2p + a53 * k3p + a54 * k4p),
                                 alpha, sfac, phi_f_, Phi_, E_, T_, coef_, nb_, nf_, gB_, gF_,
                                 gg_, mb20_, lam8_, icw_, J_tab_, du_)
            k6x, k6p, k6s = _rhs(rho + h,
                                 x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x),
                                 xp + h * (a61 * k1p + a62 * k2p + a63 * k3p + a64 * k4p + a65 * k5p),
                                 alpha, sfac, phi_f_, Phi_, E_, T_, coef_, nb_, nf_, gB_, gF_,
                                 gg_, mb20_, lam8_, icw_, J_tab_, du_)
            xn = x + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x)
            pn = xp + h * (b1 * k1p + b3 * k3p + b4 * k4p + b5 * k5p + b6 * k6p)
            sn = S + h * (b1 * k1s + b3 * k3s + b4 * k4s + b5 * k5s + b6 * k6s)
            k7x, k7p, _ = _rhs(rho + h, xn, pn, alpha, sfac, phi_f_, Phi_, E_, T_,
                               coef_, nb_, nf_, gB_, gF_, gg_, mb20_, lam8_, icw_, J_tab_, du_)
            ex = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x)
            ep = h * (e1 * k1p + e3 * k3p + e4 * k4p + e5 * k5p + e6 * k6p + e7 * k7p)
            sc_x = rtol_ * max(abs(x), abs(xn), 1e-3) + 1e-14
            sc_p = rtol_ * max(abs(xp), abs(pn), 1e-6) + 1e-14
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

    def x_of(y):
        return 1.0 + (xt - 1.0) * (-math.expm1(-y))

    def run(y):
        return _shoot_s8(x_of(y), xt, m2, rho_max, d, phi_f, Phi, E, T,
                         coef, nb, nf, gB, gF, gg, mb20, lam8, icw, J_tab, du,
                         lin_frac, rtol)

    y_lo, y_hi = 0.0, 1.0
    while run(y_hi)[0] != 1 and y_hi < 700:
        y_lo, y_hi = y_hi, 2.0 * y_hi
    if run(y_hi)[0] != 1:
        eps = V_f - V_t
        sigma = float(np.trapz(np.sqrt(2.0 * np.clip(Vg[: j + 2], 0.0, None)), grid[: j + 2]))
        if eps <= 0 or sigma <= 0:
            return None
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
    return Bounce(S_phys, d, phi_f, phi_t, phi_f + Phi * x_of(y), rho_end * L, True, "set8-numba")


def bounce_action_auto(pot, T: float, phi_f: float, phi_t: float, d: int = 3):
    """Numba paper thermal, or set8 Debye+CW numba for PhiAxisSlice."""
    if isinstance(pot, PhiAxisSlice) and pot.portal is not None and pot.portal.phi_thermal == "set8":
        return bounce_action_set8(pot, T, phi_f, phi_t, d=d)
    if getattr(pot, "bath", None) is not None and pot.portal is not None:
        # S axis with a Debye-resummed (set8-type) S bath
        return bounce_action_set8(pot, T, phi_f, phi_t, d=d)
    return bounce_action(pot, T, phi_f, phi_t, d=d)


# --------------------------------------------------------------------------
# 2D bounce A -> B: 1D shooting along the valley path, then 2D BVP refinement
# --------------------------------------------------------------------------
def _valley(pot: PortalPotential, T: float, phis, sA: float):
    """S minimising V(phi, S, T) at each phi (continuation from S = s_A at phi = 0)."""
    from scipy.optimize import minimize_scalar

    if not pot.cross:
        return np.full_like(phis, sA)
    out = np.empty_like(phis)
    s_prev = sA
    for k, p in enumerate(phis):
        hi = max(1.5 * s_prev, 1e-6 * sA)
        r = minimize_scalar(lambda s: float(pot.V_rel(p, s, T)), bounds=(0.0, hi), method="bounded",
                            options={"xatol": 1e-10 * sA})
        s = r.x
        if s < 1e-6 * sA and float(pot.grad(p, 0.0, T)[1]) == 0.0:
            s = 0.0
        if s < 1e-3 * sA and pot.mass2_S(p, 0.0, T) > 0:
            s = 0.0
        out[k] = s
        s_prev = max(s, 1e-6 * sA)
    return out


def _shoot_path(Vx, dVx, x_exit, x_end, alpha, rtol=1e-10, n_bisect=80):
    """1D O(alpha+1) bounce x(r) for potential Vx along a path, false vacuum at x = 0."""
    from scipy.integrate import solve_ivp

    def rhs(r, y):
        return [y[1], dVx(y[0]) - alpha * y[1] / r]

    def over(r, y):
        return y[0]
    over.terminal, over.direction = True, -1

    def under(r, y):
        return y[1]
    under.terminal, under.direction = True, 1

    def run(x0, dense=False):
        r0 = 1e-6
        a0 = dVx(x0) / (alpha + 1)
        y0 = [x0 + 0.5 * a0 * r0**2, a0 * r0]
        sol = solve_ivp(rhs, (r0, 1e6), y0, method="DOP853", rtol=rtol, atol=1e-14,
                        events=(over, under), dense_output=dense)
        if sol.t_events[0].size:
            return 1, sol
        return -1, sol

    lo, hi = x_exit, x_end
    if run(hi)[0] != 1:
        return None
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        code, _ = run(mid)
        if code == 1:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14 * hi:
            break
    code, sol = run(lo, dense=True)
    return lo, sol


def _bounce_a2v_cosmotransitions(pot: PortalPotential, T: float, d: int = 3,
                                 return_path: bool = False):
    """A→B via CosmoTransitions with the thermal-escape pattern from scanCouplingTemp.

    Do **not** aim the path at the tree-level VEV (v ∼ 10¹⁵): that triggers
    ``extend_to_minima`` blow-ups / ``brentq`` NaN at x∼1e307.  Instead, matching
    ``analysis/scanCouplingTemp.py`` / ``getTunneling.py``:

      * ``extend_to_minima=False`` (via ``analysis.tunneling_utils.fullTunneling``)
      * path endpoint ``tv = max(10 T, 5e4)`` GeV
      * quadratic wall beyond ``phi_cutoff = max(15 T, 5e4)``
      * 1D along S = s_A (barrier tip; then classical roll toward B)

    Returns (Bounce, info) or None.
    """
    try:
        import cosmoTransitions.pathDeformation as CTPD
        from analysis.tunneling_utils import fullTunneling
    except ImportError:
        return None

    vac = pot.vacua(T)
    if vac["A"] is None:
        return None
    phi_A, sA, V_A = float(vac["A"][0]), float(vac["A"][1]), float(vac["A"][2])
    if not (sA > 0 and pot.mass2_phi(phi_A, sA, T) > 0):
        return None

    T_use = max(float(T), 0.0)
    # Escape past the thermal barrier beyond A/A′.  Mixed A′ is deeper, so the
    # first φ with V < V_A′ at fixed S sits at ∼10⁶ GeV (not 5×10⁴); we find it
    # explicitly and run CT in a rescaled field y = φ / φ_unit.
    wall_k = 100.0 * max(float(pot.m_phi2), 1.0)
    E = pot.eps * pot.V0

    def _V_raw(phi):
        phi = np.asarray(phi, float)
        svals = np.full_like(phi, sA, dtype=float)
        if hasattr(pot, "V_delta"):
            return np.asarray(
                pot.V_delta(phi, svals, T_use, sA, phi_A=phi_A), float)
        return np.asarray(pot.V_rel(phi, svals, T_use) - V_A, float)

    # Thermal-clip escape endpoint.  Mixed A′ can be deep enough that the
    # physical V < V_A′ only at φ ∼ 10⁶ — CT fails on that plateau.  Keep the
    # local escape tv ∼ 5×10⁴ and, if needed, paste a linear downhill past the
    # barrier peak so V(tv) < V(φ_A) while leaving the barrier tip intact.
    V_false = float(np.asarray(_V_raw(np.array([phi_A]))).ravel()[0])
    tv = max(phi_A + 10.0 * max(T_use, 1.0), 50000.0)
    phi_cutoff = max(15.0 * max(T_use, 1.0), float(tv))
    Vt_raw = float(np.asarray(_V_raw(np.array([tv]))).ravel()[0])
    phi_peak = None
    V_peak = None
    if not (Vt_raw < V_false):
        phi_s = np.linspace(phi_A, tv, 400)
        Vs = _V_raw(phi_s)
        j = int(np.argmax(Vs))
        if not (Vs[j] > V_false):
            return None
        # If still climbing at tv, put the paste-on peak at 70% of the way
        # so the downhill segment is nonzero.
        if j >= len(phi_s) - 3:
            j = int(0.7 * (len(phi_s) - 1))
        phi_peak = float(phi_s[j])
        V_peak = float(max(Vs[j], Vs.max() * 0.5, 1.0))

    def _V_clip(phi):
        """V_delta, with optional post-peak linear drop to −V_peak at tv."""
        phi = np.asarray(phi, float)
        out = _V_raw(phi)
        if phi_peak is None:
            return out
        out = np.array(out, dtype=float, copy=True)
        # V(tv) = −V_peak < V_false; continuous at phi_peak
        slope = (-V_peak - V_peak) / max(tv - phi_peak, 1.0)
        mask = phi > phi_peak
        if np.any(mask):
            out[mask] = V_peak + slope * (phi[mask] - phi_peak)
        return out

    def _phi_of(X):
        arr = np.asarray(X, float)
        return arr[:, 0] if arr.ndim == 2 else arr.ravel()

    def V(X):
        phi = _phi_of(X)
        out = _V_clip(phi)
        excess = np.abs(phi) - phi_cutoff
        mask = excess > 0
        if np.any(mask):
            out = np.array(out, dtype=float, copy=True)
            out[mask] += 0.5 * wall_k * excess[mask] ** 2
        return out

    def dV(X):
        phi = _phi_of(X)
        if phi_peak is None:
            gp, _ = pot.grad(phi, np.full_like(phi, sA), T_use)
            gp = np.asarray(gp, dtype=float).ravel()
        else:
            # numerical dV on the clipped potential (barrier tip uses true grad)
            gp, _ = pot.grad(phi, np.full_like(phi, sA), T_use)
            gp = np.asarray(gp, dtype=float).ravel().copy()
            slope = (-V_peak - V_peak) / max(tv - phi_peak, 1.0)
            mask = phi > phi_peak
            if np.any(mask):
                gp[mask] = slope
        excess = np.abs(phi) - phi_cutoff
        mask = excess > 0
        if np.any(mask):
            gp = gp.copy()
            gp[mask] += wall_k * excess[mask] * np.sign(phi[mask])
        return gp.reshape(-1, 1)

    Vt = float(np.asarray(V([[tv]])).ravel()[0])
    if not (Vt < V_false):
        return None

    alpha = float(d - 1)
    # CosmoTransitions prints "Path deformation converged..." to stdout; silence it.
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = fullTunneling(
                path_pts=np.array([[tv], [phi_A]]),
                V=V,
                dV=dV,
                maxiter=1,
                V_spline_samples=200,
                verbose=False,
                tunneling_init_params=dict(alpha=alpha),
                tunneling_findProfile_params=dict(
                    xtol=1e-5, phitol=1e-5, rmin=1e-5, npoints=200
                ),
                deformation_class=CTPD.Deformation_Spline,
                extend_to_minima=False,
            )
    except Exception:
        return None

    S3 = float(res.action)
    if not (np.isfinite(S3) and S3 > 0):
        return None
    R = np.asarray(res.profile1D.R, float)
    Phi = np.asarray(res.profile1D.Phi, float)
    if R.size < 2 or Phi.size < 2:
        return None
    phi_mid = 0.5 * (Phi[0] + Phi[-1])
    if Phi[0] > Phi[-1]:
        r_c = float(np.interp(phi_mid, Phi[::-1], R[::-1]))
    else:
        r_c = float(np.interp(phi_mid, Phi, R))

    phi_lo = max(phi_A, 1e-3, 1e-6 * max(T_use, 1.0))
    phi_b = np.geomspace(phi_lo, min(1.0e6, max(phi_cutoff, tv)), 2000)
    Vb = _V_clip(phi_b)
    u_end = float(Vt / E) if E > 0 else float(Vt - V_false)
    u_barrier = float(np.max(Vb) / E) if E > 0 else float(np.max(Vb))

    b = Bounce(S3, d, 0.0, 0.0, float(Phi[0]), r_c, True,
               "cosmoTransitions (thermal clip)")
    info = {
        "centre_phi": float(Phi[0]), "centre_S": float(sA), "phi_eq": float(tv),
        "S_path": S3, "bvp_ok": False, "x0_over_reach": float("nan"),
        "path_reach": None, "u_end": u_end, "u_barrier": u_barrier,
        "method": "cosmoTransitions (thermal clip)", "immediate": False,
        "tv_thermal": float(tv), "phi_cutoff": float(phi_cutoff),
        "phi_A": float(phi_A),
        "barrier_clip": phi_peak is not None,
        "phi_peak": float(phi_peak) if phi_peak is not None else float("nan"),
        "fRatio": float(getattr(res, "fRatio", float("nan"))),
    }
    if return_path:
        info["path"] = np.stack([Phi, np.full_like(Phi, sA)], axis=-1)
    return b, info


def _thin_wall_a2v(pot: PortalPotential, T: float, d: int, sA: float, V_A: float,
                   v_T: float, E: float, u_end: float = None, u_barrier: float = None,
                   path_reach: float = None, phis=None, svals=None, return_path: bool = False):
    """Last-resort thin-wall A→B if CosmoTransitions thermal-clip path fails.

    σ is ∫√(2V) dφ over the *thermal* barrier along S=s_A (φ ≲ 10⁶ GeV), not
    out to φ_eq∼s₀ where the portal cross term dominates.
    """
    phi_hi = min(1.0e6, max(1.0e4, 50.0 * max(T, 1.0)))
    phi_b = np.geomspace(max(1e-3, 1e-6 * max(T, 1.0)), phi_hi, 4000)
    if hasattr(pot, "V_delta"):
        Vb = np.maximum(pot.V_delta(phi_b, np.full_like(phi_b, sA), T, sA), 0.0)
        dV_B = -float(pot.V_delta(v_T, 0.0, T, sA))
    else:
        Vb = np.maximum(pot.V_rel(phi_b, np.full_like(phi_b, sA), T) - V_A, 0.0)
        dV_B = float(V_A - pot.V_rel(v_T, 0.0, T))
    sigma = float(np.trapz(np.sqrt(2.0 * Vb), phi_b))
    if u_end is not None and u_end < 0 and E > 0:
        dV = max(-float(u_end) * E, dV_B, 1e-300)
    else:
        dV = max(dV_B, 1e-300)
    if u_barrier is None:
        u_barrier = float(np.max(Vb) / E) if E > 0 else float(np.max(Vb))
    if u_end is None:
        u_end = -dV / E if E > 0 else -1.0
    if sigma > 0 and dV > 0:
        S3 = ((16.0 * math.pi / 3.0) * sigma**3 / dV**2 if d == 3
              else (27.0 * math.pi**2 / 2.0) * sigma**4 / dV**3)
    else:
        S3 = 0.0
    immediate = (T > 0 and S3 / max(T, 1e-300) < 1.0)
    b = Bounce(max(S3, 0.0), d, 0.0, 0.0, float(v_T), float(v_T),
               True, "thin-wall (shallow large-ε)")
    info = {"centre_phi": float(v_T), "centre_S": 0.0, "phi_eq": float(v_T),
            "S_path": float(S3), "bvp_ok": False, "x0_over_reach": float("nan"),
            "path_reach": path_reach, "u_end": float(u_end), "u_barrier": float(u_barrier),
            "method": "thin-wall (shallow large-ε)", "immediate": immediate,
            "sigma_wall": sigma, "deltaV": dV}
    if return_path and phis is not None and svals is not None:
        info["path"] = np.stack([phis, svals], axis=-1)
    return b, info


def _a2v_bounce_ok(pot: PortalPotential, T: float, b: Bounce, info: dict, sA: float) -> bool:
    """Reject clearly junk A->B bounces.

    Thin-wall A->B bubbles often have field centre near A (small phi excursion),
    so we do *not* require the centre to sit at B=(v,0). Spurious high-T spikes
    are removed later by ``scrub_isolated_S`` / ``_validate_a2v_nucleation``.
    """
    _ = (pot, sA)
    if b is None or not np.isfinite(b.S) or b.S < 0 or T <= 0:
        return False
    S_over_T = b.S / T if T > 0 else float("inf")
    if not np.isfinite(S_over_T):
        return False
    u_end = float(info.get("u_end", 0.0))
    u_bar = float(info.get("u_barrier", 0.0))
    method = str(info.get("method", ""))
    # CosmoTransitions thermal-clip: trust finite positive action
    if method.startswith("cosmoTransitions"):
        return S_over_T > 0
    # Immediate / ultra-thin-wall (large ε): S3/T can be ≪ 1
    if method.startswith("thin-wall") or info.get("immediate"):
        return u_bar > 0 and u_end < 0
    if not (S_over_T > 1.0):
        return False
    # Physical barrier may be u ∼ T⁴/(ε V0) ≪ 1e-6 for large ε
    return u_bar > 0 and u_end < -0.05


def bounce_2d(pot: PortalPotential, T: float, d: int = 3, path_reach: float = None,
              refine: bool = True, return_path: bool = False):
    """O(d) bounce A = (0, s_T) -> B, in units F = s_T, E = eps V0.

    Returns (Bounce, info) or None. ``info['S_path']`` is the 1D action along the
    valley path; ``Bounce.S`` is the BVP-refined 2D action when refinement succeeds.

    ``path_reach``: if None, try a short-to-medium ladder. Long reaches undersample the
    thin barrier near A (x≈0) on a uniform path grid and falsely look barrierless —
    that was the intermittent "stuck in A" mis-label for some eps.
    """
    vac = pot.vacua(T)
    if vac["A"] is None or vac["B"] is None:
        return None
    # A / A′ must be a local min in phi (mixed: A′ = (φ_off, s_T))
    phi_A, sA, V_A = float(vac["A"][0]), float(vac["A"][1]), float(vac["A"][2])
    if pot.mass2_phi(phi_A, sA, T) <= 0:
        return None
    F, E = sA, pot.eps * pot.V0
    if not (F > 0 and E > 0):
        return None
    v_T = float(vac["v_T"]) if np.isfinite(vac.get("v_T", np.nan)) else float(pot.v)
    # Large-ε / large-s0: valley path to tree-level B overflows or returns junk.
    # Use CosmoTransitions thermal-clip (scanCouplingTemp pattern) instead.
    # NOTE: V_bar/|ΔV_AB| ≪ 1 always (thermal vs V0), so do NOT use that ratio
    # alone — it would steal the small-ε valley bounce too.
    use_ct = (float(pot.eps) >= 1e-3) or (sA > 0.05 * max(v_T, 1.0))
    if use_ct:
        out_ct = _bounce_a2v_cosmotransitions(pot, T, d=d, return_path=return_path)
        if out_ct is not None and _a2v_bounce_ok(pot, T, out_ct[0], out_ct[1], sA):
            return out_ct

    S_line = 0.0 if pot.cross else sA
    phi_eq = _exit_scale(pot, T, S_line, V_A, sA=sA)
    if not np.isfinite(phi_eq):
        # last try: CT even at small ε if exit scale missing
        return _bounce_a2v_cosmotransitions(pot, T, d=d, return_path=return_path)
    # Resolve thin thermal barriers: when εV0 ≫ T⁴, keep E=εV0 for action
    # conversion but the path ODE uses V_delta (stable).
    alpha = d - 1
    # Large ε ⇒ phi_eq ~ s0 ≲ v.  path_reach*phi_eq then overshoots into the φ^4
    # wall (u_end ≫ 0) and every reach returns None — silent "stuck in A".
    # Cap reaches so the valley path ends near B = (v_T, 0).
    if path_reach is not None:
        reaches = (float(path_reach),)
    else:
        r_to_v = max(3.0, 1.2 * v_T / max(phi_eq, 1e-300))
        if r_to_v < 40.0:
            # large-ε: short ladder ending at ~v
            reaches = tuple(sorted({2.0, 3.0, 5.0, r_to_v, 1.5 * r_to_v}))
        else:
            reaches = (40.0, 80.0, 120.0, 200.0, 400.0)

    best, best_score = None, -np.inf
    last_err = None
    for reach in reaches:
        try:
            out = _bounce_2d_once(pot, T, d, float(reach), refine, return_path,
                                 sA, V_A, phi_eq, F, E, alpha, v_T=v_T, phi_A=phi_A)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            out = None
        if out is None:
            continue
        b, info = out
        if not _a2v_bounce_ok(pot, T, b, info, sA):
            continue
        # prefer BVP-refined, deep true-vacuum end
        score = (100.0 if info.get("bvp_ok") else 0.0) + min(30.0, -float(info.get("u_end", 0.0)))
        score += min(10.0, float(info.get("u_barrier", 0.0)))
        # thin-wall / immediate beats a bogus huge valley-path S
        if info.get("method", "").startswith("thin-wall"):
            score += 1.0e6
        if score > best_score:
            best, best_score = out, score
        # good enough: stop the reach ladder early
        if info.get("bvp_ok") and float(info.get("u_end", 0.0)) < -1.0:
            return out
        if info.get("method", "").startswith("thin-wall"):
            return out
    if best is None and last_err is not None:
        # surface real CosmoTransitions / scipy errors (otherwise all-NaN looks silent)
        print(f"  2D bounce failed at T={T:.4g}: {last_err}")
    return best


def _bounce_2d_once(pot, T, d, path_reach, refine, return_path,
                    sA, V_A, phi_eq, F, E, alpha, v_T: float = None,
                    phi_A: float = 0.0):
    from scipy.integrate import solve_bvp
    from scipy.interpolate import CubicSpline

    # Cap path at ~v_T: going past the true vacuum climbs V ~ λ_φ φ^4 and
    # makes u_end > 0 ("barrier at path end"), which is the large-ε NaN mode.
    v_cap = float(v_T) if (v_T is not None and np.isfinite(v_T) and v_T > 0) else float(pot.v)
    phi_A = float(phi_A)
    phi_hi = min(path_reach * phi_eq, 1.2 * v_cap)
    phi_hi = max(phi_hi, min(3.0 * phi_eq, 1.2 * v_cap), 1.05 * min(phi_eq, v_cap))
    # Absolute-GeV samples near A/A′: thin thermal barrier sits at φ ∼ 10²–10³ GeV,
    # while phi_eq ∼ s0 can be 10^14 for large ε — s0-scaled grids miss it.
    phi_lo = max(phi_A, 0.0)
    phi_near = np.geomspace(max(phi_lo + 1e-3, 1e-6 * max(T, 1.0)),
                            min(1e6, 0.5 * phi_hi), 600)
    phis = np.unique(np.concatenate([
        [phi_lo],
        phi_near,
        np.linspace(phi_lo, min(3 * phi_eq, phi_hi), 800),
        np.geomspace(max(min(3 * phi_eq, phi_hi) * 0.5, phi_lo + 1e-30), phi_hi, 400),
        [v_cap],
    ]))
    phis = phis[(phis >= phi_lo) & (phis <= phi_hi * (1.0 + 1e-12))]
    if len(phis) < 8:
        return None
    svals = _valley(pot, T, phis, sA)
    X = np.stack([phis / F, svals / F], axis=-1)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(X, axis=0).T))]
    keep = np.r_[True, np.diff(seg) > 0]
    X, seg = X[keep], seg[keep]
    if len(seg) < 8 or seg[-1] <= 0:
        return None
    spl = CubicSpline(seg, X, axis=0)
    dspl = spl.derivative()

    def u(Y):
        # V_delta avoids float cancellation when |V_A| ∼ ε V0 is huge (large ε)
        if hasattr(pot, "V_delta"):
            return pot.V_delta(Y[..., 0] * F, Y[..., 1] * F, T, sA, phi_A=phi_A) / E
        return (pot.V_rel(Y[..., 0] * F, Y[..., 1] * F, T) - V_A) / E

    def gu(Y):
        gp, gs = pot.grad(Y[..., 0] * F, Y[..., 1] * F, T)
        return np.stack([gp * F / E, gs * F / E], axis=-1)

    Vx = lambda x: float(u(spl(x)))  # noqa: E731
    dVx = lambda x: float(np.dot(gu(spl(x)), dspl(x)))  # noqa: E731

    # Dense near x=0: barrier sits at x_exit/seg ~ 1e-4 for large s0; uniform
    # linspace(0, seg, N) misses it when path_reach is large.
    xs = np.unique(np.concatenate([
        [0.0],
        np.geomspace(max(seg[-1] * 1e-12, 1e-16), seg[-1], 25000),
    ]))
    ux = u(spl(xs))
    pos = np.nonzero(ux > 0)[0]
    if len(pos) == 0:
        return None
    j = int(pos[-1])
    if j + 1 >= len(xs):
        # Path ends while still above V_A — usually φ^4 overshoot (large ε).
        return None
    x_exit = brentq(Vx, xs[j], xs[j + 1])
    u_end = float(ux[-1])
    u_barrier = float(np.max(ux))
    # Need true-vacuum side (u < 0).  Do NOT demand u_barrier > 1e-6:
    # for large ε, E = ε V0 ≫ T⁴ so the same physical barrier is
    # u ∼ T⁴/(ε V0) ∼ 10^{-24} and that cut falsely returned None.
    if u_end >= 0 or not (u_barrier > 0):
        return None
    # Shallow in (F,E) units only steers to CT for large-ε / large-s0.
    # (V_bar/|ΔV_AB| is always tiny, so do not use that ratio alone.)
    shallow = (u_barrier > 0 and u_end < 0
               and u_barrier / max(-u_end, 1e-300) < 1e-8)
    if shallow and (float(pot.eps) >= 1e-3 or sA > 0.05 * max(v_cap, 1.0)):
        out_ct = _bounce_a2v_cosmotransitions(pot, T, d=d, return_path=return_path)
        if out_ct is not None:
            return out_ct
    shot = _shoot_path(Vx, dVx, x_exit, seg[-1], alpha)
    omega = 4 * np.pi if d == 3 else 2 * np.pi**2
    if shot is None:
        return None
    x0, sol = shot
    r_end = sol.t[-1]
    rr = np.linspace(sol.t[0], r_end, 4000)
    xr, dxr = sol.sol(rr)
    S_path = omega / d * np.trapz(rr**alpha * dxr**2, rr)
    S_tilde, method, bvp_ok, Phi = S_path, "valley path (1D)", False, spl(xr)
    rf = rr
    if (refine and pot.cross) or refine == "always":
        R = r_end * 3.0
        m2A = max(pot.mass2_phi(phi_A, sA, T), 1e-300) * F**2 / E
        R = max(R, r_end + 30.0 / math.sqrt(m2A))
        r = np.concatenate([np.linspace(0, r_end, 300)[:-1], np.geomspace(r_end, R, 200)])
        prof = sol.sol(np.clip(r, sol.t[0], r_end))
        Yg = spl(np.clip(prof[0], 0, seg[-1]))
        dYg = dspl(np.clip(prof[0], 0, seg[-1])) * prof[1][:, None]
        tail = r > r_end
        # outer vacuum = A / A′ = (φ_A, s_A) in field units → (φ_A/F, 1)
        Y_false = np.array([phi_A / F, 1.0])
        Yg[tail] = Y_false
        dYg[tail] = 0.0
        y_init = np.vstack([Yg.T, dYg.T])
        Smat = np.zeros((4, 4))
        Smat[2, 2] = Smat[3, 3] = -alpha

        def f(rv, y):
            g = gu(np.stack([y[0], y[1]], axis=-1))
            return np.vstack([y[2], y[3], g[:, 0], g[:, 1]])

        def bc(ya, yb):
            return np.array([ya[2], ya[3],
                             yb[0] - Y_false[0], yb[1] - Y_false[1]])

        bv = solve_bvp(f, bc, r, y_init, S=Smat, tol=1e-8, max_nodes=200000)
        if bv.success:
            rf = np.linspace(0, R, 40000)
            yf = bv.sol(rf)
            S_bvp = omega / d * np.trapz(rf**alpha * (yf[2]**2 + yf[3]**2), rf)
            if np.isfinite(S_bvp) and S_bvp > 0:
                S_tilde, method, bvp_ok = S_bvp, "2D BVP", True
                Phi = yf[:2].T
    S_phys = S_tilde * F**d * E ** (1.0 - d / 2.0)
    centre = Phi[0] * F
    half = 0.5 * (np.hypot(*Phi[0]) + 1.0)
    i = int(np.argmin(np.abs(np.hypot(*Phi.T) - half)))
    radius = float(rf[min(i, len(rf) - 1)]) * F / math.sqrt(E)
    b = Bounce(S_phys, d, 0.0, 0.0, float(centre[0]), radius, True, method)
    info = {"centre_phi": float(centre[0]), "centre_S": float(centre[1]), "phi_eq": phi_eq,
            "S_path": S_path * F**d * E ** (1.0 - d / 2.0), "bvp_ok": bvp_ok,
            "x0_over_reach": float(x0 / seg[-1]), "path_reach": path_reach,
            "u_end": u_end, "u_barrier": u_barrier}
    if return_path:
        info["path"] = Phi * F
    return b, info


def _exit_scale(pot: PortalPotential, T: float, S_line: float, V_A: float,
                sA: float = None) -> float:
    """Smallest phi > barrier with V(phi, S_line, T) = V_A (escape point scale)."""
    # Absolute GeV grid near the origin — do NOT scale the lower edge with s0,
    # or large-ε (s0 ∼ 10^14) skips the thin thermal barrier at φ ∼ 10^2–10^3 GeV.
    lo = max(1e-3, 1e-6 * max(T, 1.0))
    grid = np.geomspace(lo, pot.v, 4000)
    s_ref = float(sA) if sA is not None else float(S_line)

    def dV(phi):
        if hasattr(pot, "V_delta"):
            return float(pot.V_delta(phi, S_line, T, s_ref))
        return float(pot.V_rel(phi, S_line, T) - V_A)

    Vg = np.array([dV(p) for p in grid])
    k = np.nonzero(Vg > 0)[0]
    j = int(k[-1]) if len(k) else 0
    if j + 1 >= len(grid):
        return float("nan")
    if Vg[j] <= 0:
        return float(grid[j])
    return brentq(dV, grid[j], grid[j + 1])


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def _point(pot: PortalPotential, T: float, do_2d: bool = True) -> dict:
    vac = pot.vacua(T)
    # Mixed potential: the S condensate can exist while the tilt has removed the
    # φ-minimum (A′ "rolled over").  The intermediate phase then exists but
    # decays classically to B.  Keep phi_inter finite and record the flag so
    # that nucleation()/analyze() treat it as immediate A->B, not "stuck in A".
    rolled = bool(vac.get("A_rolled", False)) and vac["A"] is None \
        and np.isfinite(vac.get("s_T", np.nan)) and vac.get("s_T", 0.0) > 0
    row = {"T": T, "O_is_min": vac["O"] is not None,
           "phi_inter": vac["s_T"] if (vac["A"] is not None or rolled) else np.nan,
           "phi_true": vac["v_T"], "V_A": vac["A"][2] if vac["A"] else np.nan,
           "A_rolled": rolled}
    for c in CHANNELS:
        row[f"S3/T {c}"] = np.nan
        row[f"phi_center {c}"] = np.nan
    row["S_center a->v"] = np.nan
    # O->A bounce lives on the S axis; it is defined whenever the S condensate
    # is lower than O, whether or not A′ survives in the φ direction.
    S_axis_ok = vac["O"] is not None and (
        (vac["A"] is not None and vac["A"][2] < 0)
        or (rolled and float(pot.slice_S().V_rel(vac["s_T"], T)) < 0))
    if S_axis_ok:
        b = bounce_action_auto(pot.slice_S(), T, 0.0, vac["s_T"])
        if b and b.converged:
            row["S3/T 0->a"], row["phi_center 0->a"] = b.S / T, b.phi_center
    if rolled:
        # classical rollover: no barrier along the A'->B path
        row["S3/T a->v"] = 0.0
        row["S_center a->v"] = vac["s_T"]
    # phi-axis bounce whenever phi = 0 is metastable along phi: the direct O -> (v,0)
    # channel, the pure-flaton reference, and (no cross term) the phi step at any S.
    if pot.mass2_phi(0.0, 0.0, T) > 0 and np.isfinite(vac["v_T"]):
        b = bounce_action_auto(pot.slice_phi(0.0), T, 0.0, vac["v_T"])
        if b and b.converged:
            row["S3/T 0->v"], row["phi_center 0->v"] = b.S / T, b.phi_center
    if do_2d and vac["A"] is not None and vac["B"] is not None:
        if pot.cross:
            out = None
            try:
                out = bounce_2d(pot, T, path_reach=None, refine=True)
            except Exception as exc:
                print(f"  2D bounce failed at T={T:.4g}: {exc}")
            if out is None:
                try:
                    out = bounce_2d(pot, T, path_reach=None, refine=False)
                except Exception:
                    out = None
            if out is not None:
                b, info = out
                row["S3/T a->v"] = b.S / T
                row["phi_center a->v"], row["S_center a->v"] = info["centre_phi"], info["centre_S"]
        else:
            row["S3/T a->v"], row["phi_center a->v"] = row["S3/T 0->v"], row["phi_center 0->v"]
            row["S_center a->v"] = vac["s_T"]
    return row


def _eval_Ts(pot: PortalPotential, Ts, workers: int, verbose: bool,
             do_2d: bool = True) -> list:
    """Evaluate `_point` at each T, optionally in parallel."""
    Ts = [float(T) for T in Ts]
    if not Ts:
        return []
    n_workers = max(1, int(workers))
    if n_workers == 1 or len(Ts) == 1:
        rows = []
        for T in Ts:
            r = _point(pot, T, do_2d=do_2d)
            rows.append(r)
            if verbose:
                print(f"T={T:11.5g}  " + "  ".join(f"{c}: {r[f'S3/T {c}']:10.4g}"
                                                   for c in CHANNELS), flush=True)
        return rows

    # fork avoids macOS spawn re-importing __main__; fall back to default if unavailable.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    rows_by_T = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_init_worker, initargs=(pot, do_2d)) as ex:
        futs = {ex.submit(_point_worker, T): T for T in Ts}
        for fut in as_completed(futs):
            T = futs[fut]
            r = fut.result()
            rows_by_T[T] = r
            if verbose:
                print(f"T={T:11.5g}  " + "  ".join(f"{c}: {r[f'S3/T {c}']:10.4g}"
                                                   for c in CHANNELS), flush=True)
    return [rows_by_T[T] for T in Ts]


def scrub_isolated_S(T, S, min_run: int = 4, jump_fac: float = 5.0,
                     min_rel_span: float = 0.02):
    """Drop isolated / discontinuous S3/T spikes (2D bounce false positives).

    Keeps only ascending-T runs with:
      * at least ``min_run`` finite positive points,
      * no adjacent jump larger than ``jump_fac``,
      * T-span at least ``min_rel_span`` (ΔT/T), so a 3-point blip in a
        0.1% temperature window cannot seed a fake T_n.

    Immediate / ultra-thin-wall (S3/T ≪ 1 everywhere): adjacent ratios are
    huge in linear space even though the physics is smooth — leave those alone.
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float).copy()
    if T.size == 0:
        return S
    order = np.argsort(T)
    Ta, Sa = T[order], S[order]
    good = np.isfinite(Sa) & (Sa > 0)
    # Large-ε immediate: all finite S3/T ≪ 1 — do not scrub
    if int(np.sum(good)) >= min_run and float(np.median(Sa[good])) < 1.0:
        return S
    keep = np.zeros(len(Sa), dtype=bool)
    i, n = 0, len(Sa)
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and good[j + 1]:
            ratio = Sa[j + 1] / Sa[j]
            if ratio > jump_fac or ratio < 1.0 / jump_fac:
                break
            j += 1
        span_ok = Ta[i] > 0 and (Ta[j] / Ta[i] - 1.0) >= min_rel_span
        if j - i + 1 >= min_run and span_ok:
            keep[i:j + 1] = True
        i = j + 1
    out = np.full_like(S, np.nan)
    out[order[keep]] = S[order[keep]]
    return out


def scrub_a2v_in_rows(rows: list[dict], min_run: int = 4, jump_fac: float = 5.0) -> int:
    """In-place scrub of ``S3/T a->v`` in a list of scan row dicts. Returns #dropped."""
    if not rows:
        return 0
    T = np.array([r["T"] for r in rows], dtype=float)
    S = np.array([r.get("S3/T a->v", np.nan) for r in rows], dtype=float)
    Sc = scrub_isolated_S(T, S, min_run=min_run, jump_fac=jump_fac)
    dropped = 0
    for r, s_new, s_old in zip(rows, Sc, S):
        if np.isfinite(s_old) and not np.isfinite(s_new):
            r["S3/T a->v"] = np.nan
            r["phi_center a->v"] = np.nan
            r["S_center a->v"] = np.nan
            dropped += 1
        else:
            r["S3/T a->v"] = s_new
    return dropped


def scrub_a2v_in_res(res: dict, min_run: int = 4, jump_fac: float = 5.0) -> int:
    """In-place scrub of a scan result dict. Returns #dropped."""
    key = "S3/T a->v"
    if key not in res:
        return 0
    S_old = np.array(res[key], dtype=float)
    S_new = scrub_isolated_S(res["T"], S_old, min_run=min_run, jump_fac=jump_fac)
    dropped = int(np.sum(np.isfinite(S_old) & ~np.isfinite(S_new)))
    res[key] = S_new
    if dropped and "phi_center a->v" in res:
        bad = np.isfinite(S_old) & ~np.isfinite(S_new)
        res["phi_center a->v"] = np.where(bad, np.nan, res["phi_center a->v"])
        if "S_center a->v" in res:
            res["S_center a->v"] = np.where(bad, np.nan, res["S_center a->v"])
    return dropped


def _a2v_fine_window(pot: PortalPotential, rows=None,
                     T_hi_fac: float = 1.25, T_lo_fac: float = 0.85,
                     S_trigger: float = 2000.0) -> tuple:
    """T window for denser A->B sampling (trigger, not full-scan refine).

    Physics prior: T_n(S->v) sits near T0_phi (~1.15 TeV set8 / ~0.58 TeV paper),
    so densify ``[T_lo_fac, T_hi_fac] * T0_phi`` (≈ 1.0–1.4 TeV for set8).
    Also widen if *scrubbed* coarse S3/T(a->v) already fell below ``S_trigger``.
    Cap T_hi so isolated high-T junk cannot drag the window upward.
    """
    T0 = pot.T0_phi()
    T_hi = T_hi_fac * T0
    T_lo = T_lo_fac * T0
    T_hi_cap = 1.35 * T0  # never chase false S3/T blips above ~T0
    if rows and S_trigger > 0:
        Tc = np.array([r["T"] for r in rows], dtype=float)
        S = scrub_isolated_S(Tc, np.array([r.get("S3/T a->v", np.nan) for r in rows],
                                          dtype=float))
        below = np.isfinite(S) & (S < S_trigger) & (S > 0)
        if np.any(below):
            T_hi = min(T_hi_cap, max(T_hi, float(Tc[below].max())))
            T_lo = min(T_lo, 0.75 * float(Tc[below].min()))
    T_hi = min(T_hi, T_hi_cap)
    T_lo = max(T_lo, 5.0)
    if T_hi <= T_lo * 1.01:
        return T_lo, T_lo
    return T_lo, T_hi


def _a2v_trigger_Ts(rows, pot: PortalPotential, S_trigger: float = 2000.0,
                    n_fine: int = 64, T_hi_fac: float = 1.25,
                    T_lo_fac: float = 0.85) -> list:
    """Finer T grid in the A->B nucleation window (triggered after coarse scan)."""
    if n_fine <= 0:
        return []
    T_lo, T_hi = _a2v_fine_window(pot, rows, T_hi_fac, T_lo_fac, S_trigger)
    if T_hi <= T_lo * 1.01:
        return []
    return [float(t) for t in np.geomspace(T_hi, T_lo, n_fine)]


def scan(pot: PortalPotential, Ts, refine: int = 20, verbose: bool = True,
         workers: int = 1, do_2d: bool = True,
         a2v_S_trigger: float = 2000.0, a2v_n_fine: int = 64,
         a2v_T_hi_fac: float = 1.25, a2v_T_lo_fac: float = 0.85) -> dict:
    """Coarse T scan, nucleation-crossing refine, then optional a->v fine pass.

    The a->v fine pass is *triggered* after the coarse grid: densify
    ``[a2v_T_lo_fac, a2v_T_hi_fac] * T0_phi`` (and any S3/T < ``a2v_S_trigger``
    band). Set ``a2v_n_fine <= 0`` to disable.
    """
    rows = _eval_Ts(pot, Ts, workers, verbose, do_2d=do_2d)
    for _ in range(2 if refine else 0):
        if do_2d:
            scrub_a2v_in_rows(rows)  # don't refine on isolated junk spikes
        coarse = sorted({r["T"]: r for r in rows}.values(), key=lambda r: r["T"])
        Tc = np.array([r["T"] for r in coarse])
        seen = {float(t) for t in Tc}
        extra = []
        chans = CHANNELS if do_2d else ("0->a", "0->v")
        for c in chans:
            S = np.array([r[f"S3/T {c}"] for r in coarse])
            thr = 4 * np.log(Tc / pot.hubble(Tc, pot.V0))
            g = S - thr
            for k in range(len(Tc) - 1):
                a_ok, b_ok = np.isfinite(g[k]), np.isfinite(g[k + 1])
                if (a_ok and b_ok and g[k] * g[k + 1] <= 0) or (a_ok != b_ok and (
                        (a_ok and g[k] < 3 * thr[k]) or (b_ok and g[k + 1] < 3 * thr[k + 1]))):
                    lo, hi = Tc[max(k - 1, 0)], Tc[min(k + 2, len(Tc) - 1)]
                    for T in np.linspace(lo, hi, refine + 2)[1:-1]:
                        if float(T) not in seen:
                            extra.append(float(T))
                            seen.add(float(T))
        if extra:
            rows.extend(_eval_Ts(pot, extra, workers, verbose, do_2d=do_2d))

    # Triggered fine pass for A->B around T0_phi (not a uniformly fine full scan)
    if do_2d and a2v_n_fine > 0:
        scrub_a2v_in_rows(rows)
        seen = {float(r["T"]) for r in rows}
        T_lo, T_hi = _a2v_fine_window(pot, rows, a2v_T_hi_fac, a2v_T_lo_fac, a2v_S_trigger)
        fine = [T for T in _a2v_trigger_Ts(rows, pot, a2v_S_trigger, a2v_n_fine,
                                           a2v_T_hi_fac, a2v_T_lo_fac)
                if T not in seen]
        # fill coarse gaps / nan holes inside the window
        coarse = sorted(rows, key=lambda r: -r["T"])
        Tc = np.array([r["T"] for r in coarse])
        S = scrub_isolated_S(Tc, np.array([r.get("S3/T a->v", np.nan) for r in coarse],
                                          dtype=float))
        for k in range(len(Tc) - 1):
            t1, t0 = float(Tc[k]), float(Tc[k + 1])  # descending
            if t1 < T_lo or t0 > T_hi:
                continue
            a_ok, b_ok = np.isfinite(S[k]), np.isfinite(S[k + 1])
            wide = abs(np.log(max(t1, 1e-30) / max(t0, 1e-30))) > 0.04
            if a_ok != b_ok or wide:
                n_fill = 8 if a_ok != b_ok else 5
                for T in np.geomspace(t1, t0, n_fill)[1:-1]:
                    if float(T) not in seen:
                        fine.append(float(T))
                        seen.add(float(T))
        fine = sorted({float(T) for T in fine})
        if fine:
            if verbose:
                print(f"a->v fine pass (trigger): T in [{T_lo:.4g}, {T_hi:.4g}] "
                      f"(~[{a2v_T_lo_fac:g}, {a2v_T_hi_fac:g}] T0_phi={pot.T0_phi():.4g}) "
                      f"→ +{len(fine)} points", flush=True)
            rows.extend(_eval_Ts(pot, fine, workers, verbose, do_2d=True))

    rows.sort(key=lambda r: -r["T"])
    if do_2d:
        n_drop = scrub_a2v_in_rows(rows)
        if verbose and n_drop:
            print(f"a->v scrub: dropped {n_drop} isolated/discontinuous S3/T point(s)",
                  flush=True)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# --------------------------------------------------------------------------
# Nucleation sequence
# --------------------------------------------------------------------------
# Thermal FO transitions rarely have beta/H above a few thousand; values
# ≳1e4 almost always mean a noisy PCHIP derivative on junk S3/T.
_A2V_BETA_H_MAX = 3000.0


def _validate_a2v_nucleation(step2: dict, res: dict, pot: PortalPotential) -> dict:
    """Reject A->B T_n built on scrub survivors that are still unphysical."""
    T_n = step2.get("T_n", float("nan"))
    if not np.isfinite(T_n):
        return step2
    # Large-ε immediate A→B: S3/T ≪ 1 as soon as A exists (T_n ∼ T_n(0→S)
    # ≫ T0_phi).  CosmoTransitions cannot resolve this; thin-wall is trusted.
    note = str(step2.get("note", ""))
    S = np.asarray(res.get("S3/T a->v", np.nan), dtype=float)
    ok = np.isfinite(S) & (S > 0)
    if "immediate" in note or (int(np.sum(ok)) >= 4 and float(np.nanmedian(S[ok])) < 1.0):
        step2 = dict(step2)
        if "immediate" not in note:
            step2["note"] = "immediate A->B (thin-wall, S3/T << 1; CT pathDeformation overflows)"
        return step2
    T = np.asarray(res["T"], dtype=float)
    n_finite = int(np.sum(ok))
    if n_finite < 4:
        step2 = dict(step2)
        step2.update({"T_n": float("nan"), "beta/H": float("nan"),
                      "note": "rejected: fewer than 4 reliable S3/T(a->v) points"})
        return step2
    # need local support near T_n (not a lone distant blip)
    near = ok & (T > 0.85 * T_n) & (T < 1.15 * T_n)
    if int(np.sum(near)) < 3:
        step2 = dict(step2)
        step2.update({"T_n": float("nan"), "beta/H": float("nan"),
                      "note": "rejected: no contiguous S3/T(a->v) support near T_n"})
        return step2
    # A->B nucleation sits at or below ~T0_phi; well above that is bounce junk
    try:
        T0 = pot.T0_phi()
    except Exception:
        T0 = float("nan")
    if np.isfinite(T0) and T_n > 1.2 * T0:
        step2 = dict(step2)
        step2.update({"T_n": float("nan"), "beta/H": float("nan"),
                      "note": f"rejected: T_n(a->v)={T_n:.4g} > 1.2 T0_phi={T0:.4g}"})
        return step2
    bH = step2.get("beta/H", float("nan"))
    if not (np.isfinite(bH) and 0.0 < bH < _A2V_BETA_H_MAX):
        # O(4)-dominated immediate decay is still allowed (beta/H nan + note)
        if step2.get("dominant") == "O(4) vacuum" and "immediate" in note:
            return step2
        step2 = dict(step2)
        step2.update({"T_n": float("nan"), "beta/H": float("nan"),
                      "note": f"rejected: unphysical beta/H={bH} from noisy S3/T(a->v)"})
        return step2
    return step2


def _finish(d, delV):
    d["delV"] = delV
    if np.isfinite(d.get("T_n", np.nan)):
        d["alpha"] = delV / d["rho_rad(T_n)"]
    return d


def analyze(pot: PortalPotential, res: dict, b4=None) -> dict:
    V0, d1, d2 = pot.V0, pot.delV1, pot.delV2
    s_step = nucleation(pot, res, "0->a", V0)
    phi_direct = nucleation(pot, res, "0->v", V0)
    T_s, T_p = s_step["T_n"], phi_direct["T_n"]
    out = {"S step (0->a)": _finish(s_step, d1),
           "phi direct (0->v, single-step reference)": _finish(phi_direct, V0)}
    if pot.cross:
        if np.isfinite(T_p) and (not np.isfinite(T_s) or T_p > T_s):
            out["sequence"] = "single step: phi tunnels O -> (v,0) before S; no intermediate phase"
            out["step1"] = phi_direct
            out["step2"] = {}
            return out
        s_step["T_perc"] = percolation_T(pot, res, "0->a", V0)
        lnG4 = log_gamma4(b4)
        # Final scrub before nucleation (idempotent if scan already scrubbed).
        scrub_a2v_in_res(res)
        step2 = nucleation(pot, res, "a->v", V0 - d1, lnG4=lnG4, T_max=T_s)
        step2 = _validate_a2v_nucleation(step2, res, pot)
        # Mixed potential: classical rollover of A′ (no φ-minimum) at T_roll.
        # The second step then happens at max(T_n^tunnel, T_roll); if the
        # rollover already holds at T_n1 the decay is immediate.
        rolled = np.asarray(res.get("A_rolled", np.zeros(len(res["T"]), bool)), bool)
        Tr = np.asarray(res["T"], float)[rolled & (np.asarray(res["T"], float) <= T_s)]
        if len(Tr):
            T_roll = float(np.max(Tr))
            T_n2_t = step2.get("T_n", float("nan"))
            if not np.isfinite(T_n2_t) or T_roll > T_n2_t:
                step2 = dict(step2)
                immediate = T_roll >= 0.98 * T_s
                step2.update({
                    "T_n": float(T_s) if immediate else T_roll,
                    "beta/H": float("nan"),
                    "dominant": "classical rollover (A' minimum disappears)",
                    "S3/T(T_n)": 0.0,
                    "T_roll": T_roll,
                    "note": ("immediate A->B: A' has no phi-minimum already at T_n(O->A) (rollover)"
                             if immediate else
                             f"A->B by classical rollover at T_roll={T_roll:.4g} (no tunneling needed)"),
                })
                step2["H(T_n)"] = float(pot.hubble(step2["T_n"], V0 - d1))
                step2["rho_rad(T_n)"] = float(pot.rho_rad(step2["T_n"]))
        _finish(step2, d2)
        if b4 is not None:
            step2["S4 (T=0)"] = b4.S
            step2["R4 (T=0) [1/GeV]"] = b4.radius
            step2["ln(Gamma4/H^4) at T=0"] = float(lnG4 - 4 * np.log(pot.hubble(0.0, V0 - d1)))
        out["step1"], out["step2"] = s_step, step2
        T_n2 = step2["T_n"]
        if not np.isfinite(T_s):
            seq = "S never nucleates in the scanned range"
        elif not np.isfinite(T_n2):
            seq = "two-step attempt fails: stuck in A = (0, s0) (eternal inflation at V = (1-eps) V0)"
        elif "immediate" in step2.get("note", ""):
            seq = "Regime II (nested): A -> B faster than H^4 as soon as A-bubbles form"
        elif np.isfinite(s_step["T_perc"]) and s_step["T_perc"] > T_n2:
            seq = "Regime I (well separated): T_n1 > T_perc,1 > T_n2"
        else:
            seq = "Regime II (nested): T_n2 >~ T_perc,1"
        out["sequence"] = seq
        return out
    # No cross term: S and phi transitions are independent; order them by T_n.
    first_is_S = np.isfinite(T_s) and (not np.isfinite(T_p) or T_s >= T_p)
    if first_is_S:
        s_step["T_perc"] = percolation_T(pot, res, "0->a", V0)
        second = nucleation(pot, res, "0->v", V0 - d1)
        _finish(second, d2)
        out["step1"], out["step2"] = s_step, second
        T_perc1, T_n2 = s_step["T_perc"], second["T_n"]
        names = ("S", "phi")
    else:
        phi_direct["T_perc"] = percolation_T(pot, res, "0->v", V0)
        second = nucleation(pot, res, "0->a", d1)
        _finish(second, d1)
        _finish(phi_direct, d2)
        phi_direct["delV"] = d2
        out["step1"], out["step2"] = phi_direct, second
        T_perc1, T_n2 = phi_direct["T_perc"], second["T_n"]
        names = ("phi", "S")
    if not np.isfinite(out["step1"]["T_n"]):
        seq = "nothing nucleates in the scanned range"
    elif not np.isfinite(T_n2):
        seq = f"{names[0]} transition only; {names[1]} never nucleates"
    elif np.isfinite(T_perc1) and T_perc1 > T_n2:
        seq = f"independent, well separated: {names[0]} (T_n={out['step1']['T_n']:.4g}) then {names[1]}"
    else:
        seq = f"independent, overlapping in time: {names[0]} then {names[1]}"
    out["sequence"] = seq
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_potential(pot: PortalPotential, Ts, path: str, bounce_path=None, T_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    fig, ax = plt.subplots(1, 3, figsize=(21, 5.8))
    s0, E = pot.s0, pot.eps * pot.V0
    T_map = 0.0 if T_path is None else T_path
    vac = pot.vacua(T_map)
    V_A = vac["A"][2] if vac["A"] else -E
    phi_eq = _exit_scale(pot, T_map, 0.0 if pot.cross else s0, V_A) if vac["A"] else 3 * s0
    pmax = 4 * (phi_eq if np.isfinite(phi_eq) else s0)
    P, S = np.meshgrid(np.linspace(0, pmax, 400), np.linspace(0, 1.3 * s0, 300))
    U = (pot.V_rel(P, S, T_map) - V_A) / E
    im = ax[0].pcolormesh(P / s0, S / s0, U, cmap="RdBu_r", shading="auto",
                          norm=SymLogNorm(linthresh=1e-2, vmin=-1e2, vmax=1e2))
    ax[0].contour(P / s0, S / s0, U, levels=[0.0], colors="k", linewidths=1.2)
    ax[0].plot([0], [vac["s_T"] / s0 if vac["A"] else 1.0], "o", c="C1", ms=8, label="A = (0, s)")
    if bounce_path is not None:
        ax[0].plot(bounce_path[:, 0] / s0, bounce_path[:, 1] / s0, "w-", lw=2.2, label="bounce path")
        ax[0].plot(bounce_path[0, 0] / s0, bounce_path[0, 1] / s0, "w*", ms=13, label="bounce centre")
    fig.colorbar(im, ax=ax[0], label=r"$[V - V(A)]/(\epsilon V_0)$")
    ax[0].set(xlabel=r"$\phi/s_0$", ylabel=r"$S/s_0$",
              title=f"V(phi,S) near A at T = {T_map:.4g} GeV (black: V = V(A))")
    ax[0].legend(fontsize=8, loc="upper right")
    cmap = plt.cm.viridis
    for k, T in enumerate(Ts):
        col = cmap(k / max(1, len(Ts) - 1))
        s = np.linspace(0, 1.4 * s0, 600)
        ax[1].plot(s / s0, pot.V_rel(0.0, s, T) / E, color=col, label=f"T={T:.4g}")
        p = np.linspace(0, pmax, 600)
        sv = pot.vacua(T)
        if sv["A"] is not None:
            # phi direction through the 2D valley from A (S re-minimised at each phi)
            if pot.cross:
                sl = np.sqrt(np.clip(sv["s_T"]**2 - pot.lam_phiS * p**2 / (2 * pot.lam_S), 0, None))
            else:
                sl = np.full_like(p, sv["s_T"])
            ax[2].plot(p / s0, (pot.V_rel(p, sl, T) - sv["A"][2]) / E, color=col, label=f"T={T:.4g}")
    ax[1].axhline(0, c="gray", lw=0.7)
    ax[1].axhline(-1, c="gray", ls=":", lw=0.7)
    ax[1].set(xlabel=r"$S/s_0$", ylabel=r"$[V(0,S,T)-V(0,0,T)]/(\epsilon V_0)$",
              title="Step 1: S axis (thermal barrier, depth eps V0)")
    ax[1].set_ylim(-1.5, 1.0)
    ax[1].legend(fontsize=7)
    ax[2].axhline(0, c="gray", lw=0.7)
    ax[2].set(xlabel=r"$\phi/s_0$", ylabel=r"$[V - V(A)]/(\epsilon V_0)$ along the valley",
              title="Step 2: from A toward phi" + (" (S re-minimised)" if pot.cross else " (S = s)"))
    ax[2].set_ylim(-3, 1.5)
    ax[2].legend(fontsize=7)
    fig.suptitle(pot.summary().splitlines()[0])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actions(pot: PortalPotential, res: dict, info: dict, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    o = np.argsort(res["T"])
    T = res["T"][o]
    lab = {"0->a": r"step 1: $O\to A$ (S axis)",
           "a->v": r"step 2: $A\to B$" + (" (2D)" if pot.cross else r" ($\phi$ axis)"),
           "0->v": r"direct $O\to(v,0)$ (single-step ref.)"}
    for c, col, ls in (("0->a", "C0", "o-"), ("a->v", "C3", "s-"), ("0->v", "k", "^--")):
        y = res[f"S3/T {c}"][o]
        if np.isfinite(y).any():
            ax.loglog(T, y, ls, ms=3, color=col, label=lab[c])
    ax.loglog(T, 4 * np.log(T / pot.hubble(T, pot.V0)), ":", c="gray", label=r"$4\ln(T/H)$")
    s4 = info.get("step2", {}).get("S4 (T=0)")
    if s4 is not None and np.isfinite(s4):
        ax.axhline(s4, ls="-.", c="C3", lw=1, label=rf"$S_4$ ($A\to B$, T=0) = {s4:.3g}")
    for key, col in (("step1", "C0"), ("step2", "C3")):
        Tn = info.get(key, {}).get("T_n", np.nan)
        if np.isfinite(Tn):
            ax.axvline(Tn, ls=":", color=col)
    ax.set(xlabel="T [GeV]", ylabel=r"$S_3/T$", title=info["sequence"])
    ax.set_ylim(1, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gw(pot: PortalPotential, info: dict, out_dir: str, v_w=1.0, kappa_phi=1.0,
       T_reh: float | None = None):
    """SGWB: Sec. II/IV sequential or Sec. III.C nested multi-boundary spectrum."""
    from analysis.twoStepGW import (
        DEFAULT_T_REH, apply_two_step_gw, make_step, plot, step_quantities,
    )

    T_reh = DEFAULT_T_REH if T_reh is None else float(T_reh)
    qs_raw = []
    beta1 = info.get("step1", {}).get("beta/H", np.nan)
    for key in ("step1", "step2"):
        n = info.get(key, {})
        T_n, bH = n.get("T_n", np.nan), n.get("beta/H", np.nan)
        label = key  # keep short labels for nested weight matching
        if key == "step2" and np.isfinite(T_n) and not (np.isfinite(bH) and bH > 0) \
                and "immediate" in n.get("note", ""):
            bH = beta1
        if np.isfinite(T_n) and np.isfinite(bH) and bH > 0 and np.isfinite(n.get("delV", np.nan)):
            q = step_quantities(make_step(
                label, n["delV"], T_n, bH, pot.V0, T_reh,
                v_w=v_w, kappa=kappa_phi, g_star=pot.g_star))
            q["channel"] = n.get("channel", "")
            qs_raw.append(q)
    stuck = "stuck" in info.get("sequence", "")
    seq = info.get("sequence", "")
    eps = float(getattr(pot, "eps_eff", getattr(pot, "eps", 0.0)) or 0.0)
    if not (0 < eps < 1) and qs_raw and pot.V0 > 0:
        eps = float(qs_raw[0].get("frac_delV_V0", 0.0))
    T1 = info.get("step1", {}).get("T_n", float("nan"))
    T2 = info.get("step2", {}).get("T_n", float("nan"))
    dTn = (T1 - T2) if (np.isfinite(T1) and np.isfinite(T2)) else float("nan")
    force_nested = (
        (not np.isfinite(dTn))
        and ("Regime II" in seq or "nested" in seq.lower())
        and not stuck
    )
    qs, meta = apply_two_step_gw(
        qs_raw, eps=eps, stuck=stuck, dTn=dTn, force_nested=force_nested,
        kappa_phi1=kappa_phi, kappa_phi2=kappa_phi)
    ref = None
    n = info.get("phi direct (0->v, single-step reference)", {})
    if np.isfinite(n.get("T_n", np.nan)) and np.isfinite(n.get("beta/H", np.nan)):
        ref = step_quantities(make_step(
            "single-step O->(v,0)", pot.V0, n["T_n"], n["beta/H"], pot.V0, T_reh,
            v_w=v_w, kappa=kappa_phi, g_star=pot.g_star))
    if qs:
        ttl = info["sequence"]
        if meta.get("nested"):
            ttl = (f"{ttl}\nSec.III.C nested  f_comp={meta.get('f_comp', 0):.3g}  "
                   f"χ={meta.get('chi_nest', float('nan')):.3g}")
        plot(qs, ref, os.path.join(out_dir, "gw_spectrum.png"), title=ttl)
    return qs, ref


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _species(items, default, fermion=False):
    pairs = [tuple(float(x) for x in s.split(",")) for s in items] if items else default
    return make_species(pairs, fermion=fermion)


def build_parser():
    p = argparse.ArgumentParser(
        description="Two-field portal model: bounces, nucleation, SGWB. "
                    "Default phi sector matches set8 (V0=2.5e35, v=1e15, nb=nf=20, g=1.09).")
    p.add_argument("--V0", type=float, default=SET8_V0,
                   help="vacuum energy [GeV^4] (set8 default 2.5e35)")
    p.add_argument("--v", type=float, default=SET8_V, help="flaton VEV [GeV] (set8: 1e15)")
    p.add_argument("--s0", type=float, default=3e3, help="intermediate S VEV [GeV]")
    p.add_argument("--eps", type=float, default=1e-14, help="Delta V_S / V0")
    p.add_argument("--kappa", type=float, default=2.0,
                   help="lam_phiS = kappa m_phi^2/s0^2 (kappa > 1: A is a T=0 minimum)")
    p.add_argument("--no_cross", action="store_true", help="lam_phiS = 0")
    p.add_argument("--boson_phi", action="append",
                   help="n,g bosons on phi (default 20,1.09). Overrides --g_phi.")
    p.add_argument("--fermion_phi", action="append",
                   help="n,g fermions on phi (default 20,1.09)")
    p.add_argument("--boson_S", action="append",
                   help="n,g bosons on S (default 10,1). Overrides --g_S.")
    p.add_argument("--fermion_S", action="append", help="n,g fermions coupled to S")
    p.add_argument("--g_phi", type=float, default=None,
                   help="shorthand: n=20 bosons+fermions on phi with this g (paper m=g|phi|)")
    p.add_argument("--g_S", type=float, default=None,
                   help="shorthand: n=6 bosons on S with this g (paper m=g|S|)")
    p.add_argument("--g_star", type=float, default=100.0)
    p.add_argument("--phi_thermal", choices=("paper", "set8"), default="paper",
                   help="paper = m=g*field; set8 = V_correct Debye+mb2_0+CW on phi")
    p.add_argument("--Tmin", type=float, default=None)
    p.add_argument("--Tmax", type=float, default=None)
    p.add_argument("--nT", type=int, default=40)
    p.add_argument("--refine", type=int, default=20)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel processes for bounce scan (default: ncpu-1)")
    p.add_argument("--check_2d", action="store_true",
                   help="print valley-path vs 2D-refined action and path-length dependence")
    p.add_argument("--out", required=True)
    return p


def build_potential(a) -> PortalPotential:
    # --g_phi / --g_S shorthand → n,g lists (paper thermal m = g * field)
    if getattr(a, "g_phi", None) is not None and a.boson_phi is None:
        a.boson_phi = [f"20,{a.g_phi}"]
        if a.fermion_phi is None:
            a.fermion_phi = [f"20,{a.g_phi}"]
    if getattr(a, "g_S", None) is not None and a.boson_S is None:
        a.boson_S = [f"6,{a.g_S}"]
    sp_phi = (_species(a.boson_phi, SET8_BOSON_PHI, False)
              + _species(a.fermion_phi, SET8_FERMION_PHI, True))
    # If user passes only boson_phi / only fermion_phi, do not force the other set8 default.
    if a.boson_phi is not None and a.fermion_phi is None:
        sp_phi = _species(a.boson_phi, SET8_BOSON_PHI, False)
    elif a.fermion_phi is not None and a.boson_phi is None:
        sp_phi = _species(a.fermion_phi, SET8_FERMION_PHI, True)
    sp_S = (_species(a.boson_S, [(6, 1)], False)
            + _species(a.fermion_S, [], True))
    if a.boson_S is None and a.fermion_S is not None:
        sp_S = _species(a.fermion_S, [], True)
    return PortalPotential(
        V0=a.V0, v=a.v, s0=a.s0, eps=a.eps,
        kappa=None if a.no_cross else a.kappa,
        species_phi=sp_phi, species_S=sp_S, g_star=a.g_star,
        phi_thermal=getattr(a, "phi_thermal", "paper"),
    )


def run(a) -> dict:
    pot = build_potential(a)
    print(pot.summary())
    print(f"scan workers = {a.workers}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    Tmax = a.Tmax or 2.5 * max(pot.T0_S(), pot.T0_phi())
    Tmin = a.Tmin or 0.02 * min(pot.T0_S(), pot.T0_phi())
    b4 = None
    if pot.cross:
        out4 = bounce_2d(pot, 0.0, d=4)
        b4 = out4[0] if out4 else None
        print(f"O(4) A->B at T=0: S4 = {b4.S:.6g}, R = {b4.radius:.4g} /GeV, "
              f"centre (phi,S) = ({out4[1]['centre_phi']:.4g}, {out4[1]['centre_S']:.4g})"
              if b4 else "O(4) A->B: none")
        if a.check_2d:
            for T, d in ((0.0, 4), (0.5 * pot.T0_phi(), 3)):
                for reach in (200.0, 400.0, 800.0):
                    o = bounce_2d(pot, T, d=d, path_reach=reach)
                    if o:
                        print(f"2D check T={T:.4g} d={d} reach={reach:g}: S_path = {o[1]['S_path']:.8g}"
                              f"  S_2D = {o[0].S:.8g} ({o[0].method})")
    Ts = np.geomspace(Tmax, Tmin, a.nT)
    res = scan(pot, Ts, refine=a.refine, workers=a.workers)
    info = analyze(pot, res, b4=b4)
    print(json.dumps(info, indent=2, default=float))

    import pandas as pd

    pd.DataFrame(res).to_csv(os.path.join(a.out, "bounce_scan.csv"), index=False)
    T_path = info.get("step2", {}).get("T_n", np.nan) if pot.cross else np.nan
    bp = None
    if pot.cross:
        T_path = T_path if np.isfinite(T_path) else 0.0
        o = bounce_2d(pot, T_path, d=3 if T_path > 0 else 4, return_path=True)
        bp = o[1]["path"] if o else None
    else:
        T_path = 0.0
    Tn1 = info.get("step1", {}).get("T_n", np.nan)
    Ts_plot = [x for x in (1.1 * pot.T0_S(), Tn1, pot.T0_S() * 0.9, pot.T0_phi(), 0.3 * pot.T0_phi())
               if np.isfinite(x)]
    plot_potential(pot, sorted(Ts_plot, reverse=True), os.path.join(a.out, "potential.png"),
                   bounce_path=bp, T_path=T_path)
    plot_actions(pot, res, info, os.path.join(a.out, "bounce_actions.png"))
    qs, ref = gw(pot, info, a.out)
    meta = {"potential": {"V0": pot.V0, "v": pot.v, "s0": pot.s0, "eps": pot.eps,
                          "kappa": pot.kappa, "cross": pot.cross, "m_phi2": pot.m_phi2,
                          "lam_phi": pot.lam_phi, "m_S2": pot.m_S2, "lam_S": pot.lam_S,
                          "lam_phiS": pot.lam_phiS, "T0_phi": pot.T0_phi(), "T0_S": pot.T0_S(),
                          "species_phi": [asdict(s) for s in pot.species_phi],
                          "species_S": [asdict(s) for s in pot.species_S], "g_star": pot.g_star},
            "nucleation": info, "gw": {"steps": qs, "reference": ref}}
    with open(os.path.join(a.out, "nucleation.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"Wrote {a.out}")
    return meta


if __name__ == "__main__":
    run(build_parser().parse_args())
