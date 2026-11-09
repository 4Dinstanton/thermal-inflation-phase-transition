"""Background between the two stages with a residual first-stage flaton fraction r_S (vectorized).
After stage 1: radiation (1-r) V1 at T_rh = C((1-r)V1)^{1/4}; S matter r V1 diluting as (T/T_rh)^3; V2 constant.
Stage 2 starts at the radiation temperature T_* where V2 = rho_rad(T_*) + r V1 (T_*/T_rh)^3; ends at T_n2."""
import math
import numpy as np
G = 100.0
C = (30 / (math.pi**2 * G))**0.25
A = math.pi**2 / 30 * G


def T_start2(V1, V2, r):
    V1, V2, r = np.broadcast_arrays(np.asarray(V1, float), np.asarray(V2, float), np.asarray(r, float))
    Trh = C * ((1 - r) * V1)**0.25
    lo = np.full(V1.shape, -60.0); hi = np.log(np.maximum(Trh, 1e-30))
    for _ in range(80):                                   # bisection in ln T: f decreasing in T
        mid = 0.5 * (lo + hi); T = np.exp(mid)
        f = V2 - A * T**4 - r * V1 * (T / Trh)**3
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    return np.exp(0.5 * (lo + hi))


def N2(V1, V2, Tn2, r):
    return np.maximum(np.log(T_start2(V1, V2, r) / Tn2), 0.0)


def gw_suppression(V1, V2, Tn2, r, Td=100.0):
    """Omega_GW(r)/Omega_GW(0): energy fraction at nucleation and extra matter era (residual S decays with phi at T_d)."""
    def om(rr):
        Trh = C * ((1 - rr) * V1)**0.25
        rr_r = A * Tn2**4; rs = rr * V1 * (Tn2 / Trh)**3
        tot = V2 + rr_r + rs; M0, R0 = V2 + rs, rr_r
        lo, hi = 0.0, 200.0                                 # y = ln(a_d/a_n)
        for _ in range(100):
            y = 0.5 * (lo + hi); g = M0 * math.exp(-3 * y) + R0 * math.exp(-4 * y) - A * Td**4
            lo, hi = (y, hi) if g > 0 else (lo, y)
        return (V2 / tot)**2 * tot * math.exp(-4 * y), math.sqrt(tot) * math.exp(-y)
    (o1, f1), (o0, f0) = om(r), om(0.0)
    return o1 / o0


def gw_factors(V1, V2, Tn2, r, Td=100.0):
    """(amplitude ratio, peak-frequency ratio) relative to r = 0; f_today ∝ H_* a_n/a_d at fixed beta/H."""
    def om(rr):
        Trh = C * ((1 - rr) * V1)**0.25
        rr_r = A * Tn2**4; rs = rr * V1 * (Tn2 / Trh)**3
        tot = V2 + rr_r + rs; M0, R0 = V2 + rs, rr_r
        lo, hi = 0.0, 200.0
        for _ in range(100):
            y = 0.5 * (lo + hi); g = M0 * math.exp(-3 * y) + R0 * math.exp(-4 * y) - A * Td**4
            lo, hi = (y, hi) if g > 0 else (lo, y)
        return (V2 / tot)**2 * tot * math.exp(-4 * y), math.sqrt(tot) * math.exp(-y)
    (o1, f1), (o0, f0) = om(r), om(0.0)
    return o1 / o0, f1 / f0
