#!/usr/bin/env python
"""
Quick benchmark comparing original vs numba-optimized lattice simulation.
Runs both for a small number of steps to measure speedup.
"""
import numpy as np
import time
import sys
import os

# Suppress plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("LATTICE SIMULATION SPEED COMPARISON")
print("=" * 70)

# Small test parameters
Nx, Ny = 64, 64  # Smaller grid for quick test
Nt_test = 1000    # Just 1000 steps
dx_phys = 1e-3
dt_phys = 1e-2 * dx_phys**2
lam = 1e-16
mphi = 1000.0
eta_phys = 0.3
T0 = 3000.0
cooling_rate = 1.0
mu = mphi
dx = mu * dx_phys
dt = mu * dt_phys
eta = eta_phys / mu
cooling_rate_scaled = cooling_rate / mu

print(f"\nTest parameters: {Nx}×{Ny} grid, {Nt_test} steps")
print("-" * 70)

# ============================================================================
# Test 1: Original (scipy-based)
# ============================================================================
print("\n1. Testing ORIGINAL (latticeSimRescale.py with scipy splines)...")
try:
    from scipy.interpolate import InterpolatedUnivariateSpline
    import cosmoTransitions.finiteT as CTFT
    import math

    # Build splines
    y2_grid = np.linspace(0, 100, 1000)
    dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid])
    dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid])
    dJb_spline = InterpolatedUnivariateSpline(y2_grid, dJb_grid)
    dJf_spline = InterpolatedUnivariateSpline(y2_grid, dJf_grid)

    # Setup
    bosonMassSquared = 1_000_000.0
    bosonCoupling = 1.09
    bosonGaugeCoupling = 1.05
    fermionCoupling = 1.09
    fermionGaugeCoupling = 1.05

    phi_orig = 0.01 * np.random.randn(Nx, Ny)
    pi_orig = np.zeros_like(phi_orig)

    def laplacian_orig(phi):
        return (
            np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
          + np.roll(phi, 1, 1) + np.roll(phi, -1, 1)
          - 4 * phi
        ) / dx**2

    def Vprime_orig(phi, T):
        gb2 = bosonCoupling * bosonCoupling
        gg2 = bosonGaugeCoupling * bosonGaugeCoupling
        gf2 = fermionCoupling * fermionCoupling
        gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
        coef_b_T = 0.25 * gb2 + (2.0 / 3.0) * gg2
        T2 = T * T
        xb_sq = bosonMassSquared + 0.5 * gb2 * phi * phi + coef_b_T * T2
        xf_sq = 0.5 * gf2 * phi * phi + (1.0 / 6.0) * gfg2 * T2
        xb = np.sqrt(np.maximum(xb_sq, 0)) / T
        xf = np.sqrt(np.maximum(xf_sq, 0)) / T
        db = dJb_spline(xb) * 0.5 * gb2 * phi / (T2 + 1e-20)
        df = dJf_spline(xf) * 0.5 * gf2 * phi / (T2 + 1e-20)
        dV = lam * phi**3 - mphi**2 * phi
        return dV + T**4 / (2 * math.pi**2) * (2 * db - df)

    def temp_orig(t):
        return max(T0 - cooling_rate_scaled * t, 0.0)

    # Run
    t0 = time.time()
    for n in range(Nt_test):
        t = n * dt
        T = temp_orig(t)
        T_mid = temp_orig(t + 0.5 * dt)
        noise = np.sqrt(2.0 * eta * T * dt / dx**2) * np.random.randn(Nx, Ny)
        k1_phi = pi_orig
        k1_pi = laplacian_orig(phi_orig) - eta * pi_orig - Vprime_orig(phi_orig, T) / mu**2
        phi_mid = phi_orig + 0.5 * dt * k1_phi
        pi_mid = pi_orig + 0.5 * dt * k1_pi
        k2_phi = pi_mid
        k2_pi = laplacian_orig(phi_mid) - eta * pi_mid - Vprime_orig(phi_mid, T_mid) / mu**2
        phi_orig += dt * k2_phi
        pi_orig += dt * k2_pi + noise
    t1 = time.time()
    time_orig = t1 - t0
    print(f"   Time: {time_orig:.3f} seconds")
    print(f"   Speed: {Nt_test/time_orig:.1f} steps/sec")
except Exception as e:
    print(f"   ERROR: {e}")
    time_orig = None

# ============================================================================
# Test 2: Numba-optimized
# ============================================================================
print("\n2. Testing NUMBA-OPTIMIZED (latticeSimeRescale_numba.py)...")
try:
    import numba as nb
    from scipy.interpolate import CubicSpline

    print(f"   Numba threads: {nb.get_num_threads()}")

    # Build uniform cubic splines
    N_Y = 256
    y2_grid_nb = np.linspace(0.0, 100.0, N_Y, dtype=np.float64)
    dJb_grid_nb = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid_nb], dtype=np.float64)
    dJf_grid_nb = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid_nb], dtype=np.float64)
    cs_b = CubicSpline(y2_grid_nb, dJb_grid_nb, bc_type="not-a-knot")
    cs_f = CubicSpline(y2_grid_nb, dJf_grid_nb, bc_type="not-a-knot")
    c0_b = cs_b.c[0].astype(np.float64)
    c1_b = cs_b.c[1].astype(np.float64)
    c2_b = cs_b.c[2].astype(np.float64)
    c3_b = cs_b.c[3].astype(np.float64)
    c0_f = cs_f.c[0].astype(np.float64)
    c1_f = cs_f.c[1].astype(np.float64)
    c2_f = cs_f.c[2].astype(np.float64)
    c3_f = cs_f.c[3].astype(np.float64)
    x_min = float(y2_grid_nb[0])
    h_y = float(y2_grid_nb[1] - y2_grid_nb[0])
    nseg = int(y2_grid_nb.size - 1)

    @nb.njit(fastmath=True, cache=True)
    def cubic_eval(x, x_min, h, nseg, c0, c1, c2, c3):
        t = (x - x_min) / h
        i = int(t)
        if i < 0: i = 0
        elif i >= nseg: i = nseg - 1
        dx = x - (x_min + i * h)
        return ((c0[i] * dx + c1[i]) * dx + c2[i]) * dx + c3[i]

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def laplacian_nb(out, a, dx_val):
        nx, ny = a.shape
        inv_dx2 = 1.0 / (dx_val * dx_val)
        for i in nb.prange(nx):
            ip = i + 1 if i + 1 < nx else 0
            im = i - 1 if i - 1 >= 0 else nx - 1
            for j in range(ny):
                jp = j + 1 if j + 1 < ny else 0
                jm = j - 1 if j - 1 >= 0 else ny - 1
                out[i, j] = (a[ip, j] + a[im, j] + a[i, jp] + a[i, jm] - 4.0 * a[i, j]) * inv_dx2

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def Vprime_nb(out, phi, T, lam, mphi, bms, bc, bgc, fc, fgc):
        nx, ny = phi.shape
        T2 = T * T
        T4 = T2 * T2
        pref = T4 / (2.0 * math.pi * math.pi)
        gb2 = bc * bc
        gg2 = bgc * bgc
        gf2 = fc * fc
        gfg2 = fgc * fgc
        coef_b_T = 0.25 * gb2 + (2.0 / 3.0) * gg2
        for i in nb.prange(nx):
            for j in range(ny):
                ph = phi[i, j]
                dV = lam * ph * ph * ph - mphi * mphi * ph
                xb_sq = bms + 0.5 * gb2 * ph * ph + coef_b_T * T2
                xf_sq = 0.5 * gf2 * ph * ph + (1.0 / 6.0) * gfg2 * T2
                xb = 0.0
                xf = 0.0
                if xb_sq > 0.0: xb = math.sqrt(xb_sq) / T
                if xf_sq > 0.0: xf = math.sqrt(xf_sq) / T
                xb_c = max(min(xb, x_min + h_y * nseg - 1e-12), x_min)
                xf_c = max(min(xf, x_min + h_y * nseg - 1e-12), x_min)
                dJb = cubic_eval(xb_c, x_min, h_y, nseg, c0_b, c1_b, c2_b, c3_b)
                dJf = cubic_eval(xf_c, x_min, h_y, nseg, c0_f, c1_f, c2_f, c3_f)
                dxb_dphi = 0.5 * gb2 * ph / (T2 * max(xb, 1e-20))
                dxf_dphi = 0.5 * gf2 * ph / (T2 * max(xf, 1e-20))
                dV += pref * (2.0 * dJb * dxb_dphi - dJf * dxf_dphi)
                out[i, j] = dV

    phi_nb = 0.01 * np.random.randn(Nx, Ny)
    pi_nb = np.zeros_like(phi_nb)
    lap = np.empty_like(phi_nb)
    Vp = np.empty_like(phi_nb)

    # Warmup JIT
    laplacian_nb(lap, phi_nb, dx)
    Vprime_nb(Vp, phi_nb, T0, lam, mphi, bosonMassSquared, bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)

    # Run
    t0 = time.time()
    for n in range(Nt_test):
        t = n * dt
        T = max(T0 - cooling_rate_scaled * t, 0.0)
        T_mid = max(T0 - cooling_rate_scaled * (t + 0.5 * dt), 0.0)
        noise = np.sqrt(2.0 * eta * T * dt / dx**2) * np.random.randn(Nx, Ny)
        laplacian_nb(lap, phi_nb, dx)
        Vprime_nb(Vp, phi_nb, T, lam, mphi, bosonMassSquared, bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
        k1_phi = pi_nb
        k1_pi = lap - eta * pi_nb - Vp / mu**2
        phi_mid = phi_nb + 0.5 * dt * k1_phi
        pi_mid = pi_nb + 0.5 * dt * k1_pi
        laplacian_nb(lap, phi_mid, dx)
        Vprime_nb(Vp, phi_mid, T_mid, lam, mphi, bosonMassSquared, bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
        k2_phi = pi_mid
        k2_pi = lap - eta * pi_mid - Vp / mu**2
        phi_nb += dt * k2_phi
        pi_nb += dt * k2_pi + noise
    t1 = time.time()
    time_nb = t1 - t0
    print(f"   Time: {time_nb:.3f} seconds")
    print(f"   Speed: {Nt_test/time_nb:.1f} steps/sec")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    time_nb = None

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
if time_orig and time_nb:
    speedup = time_orig / time_nb
    print(f"Original:         {time_orig:.3f}s  ({Nt_test/time_orig:.1f} steps/sec)")
    print(f"Numba-optimized:  {time_nb:.3f}s  ({Nt_test/time_nb:.1f} steps/sec)")
    print(f"\n🚀 SPEEDUP: {speedup:.2f}× faster!")
    print(f"\nFor {Nt_test:,} steps:")
    print(f"  Original would take:    ~{time_orig * (1_000_000/Nt_test)/60:.1f} minutes")
    print(f"  Numba takes:            ~{time_nb * (1_000_000/Nt_test)/60:.1f} minutes")
    print(f"  Time saved:             ~{(time_orig - time_nb) * (1_000_000/Nt_test)/60:.1f} minutes")
elif time_nb:
    print(f"Numba-optimized:  {time_nb:.3f}s  ({Nt_test/time_nb:.1f} steps/sec)")
    print("(Original test failed)")
else:
    print("Both tests failed - check errors above")
print("=" * 70)

