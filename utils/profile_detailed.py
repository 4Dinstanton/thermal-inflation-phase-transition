#!/usr/bin/env python
"""
Detailed profiling to find remaining bottlenecks.
Run this to see exactly where time is spent.
"""
import numpy as np
import numba as nb
import time
import math
from scipy.interpolate import CubicSpline
import cosmoTransitions.finiteT as CTFT

print("=" * 70)
print("DETAILED PERFORMANCE PROFILING")
print("=" * 70)

# Check threading
print(f"\n1. Threading Status:")
print(f"   Numba threads: {nb.get_num_threads()}")
print(f"   Threading layer: {nb.config.THREADING_LAYER}")

# Setup (same as your simulation)
YMAX = 100.0
N_Y = 256
y2_grid = np.linspace(0.0, YMAX, N_Y, dtype=np.float64)
dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid], dtype=np.float64)
dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid], dtype=np.float64)

cs_b = CubicSpline(y2_grid, dJb_grid, bc_type="not-a-knot")
cs_f = CubicSpline(y2_grid, dJf_grid, bc_type="not-a-knot")

c0_b = cs_b.c[0].astype(np.float64)
c1_b = cs_b.c[1].astype(np.float64)
c2_b = cs_b.c[2].astype(np.float64)
c3_b = cs_b.c[3].astype(np.float64)
c0_f = cs_f.c[0].astype(np.float64)
c1_f = cs_f.c[1].astype(np.float64)
c2_f = cs_f.c[2].astype(np.float64)
c3_f = cs_f.c[3].astype(np.float64)

x_min = float(y2_grid[0])
h_y = float(y2_grid[1] - y2_grid[0])
nseg = int(y2_grid.size - 1)

# Parameters
Nx, Ny = 128, 128
lam = 1e-16
mphi = 1000.0
dx = 1.0
bosonMassSquared = 1_000_000.0
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

print(f"\n2. Grid Size: {Nx}×{Ny} = {Nx*Ny:,} points")

# Test with both precisions
for dtype_name in ["float32", "float64"]:
    field_dtype = np.float32 if dtype_name == "float32" else np.float64

    print(f"\n" + "=" * 70)
    print(f"Testing with {dtype_name}:")
    print("=" * 70)

    phi = np.random.randn(Nx, Ny).astype(field_dtype)
    pi = np.zeros((Nx, Ny), dtype=field_dtype)
    lap = np.empty((Nx, Ny), dtype=field_dtype)
    Vp = np.empty((Nx, Ny), dtype=field_dtype)

    # Define kernels
    @nb.njit(fastmath=True, cache=True)
    def cubic_eval(x, x_min, h, nseg, c0, c1, c2, c3):
        t = (x - x_min) / h
        i = int(t)
        if i < 0:
            i = 0
        elif i >= nseg:
            i = nseg - 1
        dx = x - (x_min + i * h)
        return ((c0[i] * dx + c1[i]) * dx + c2[i]) * dx + c3[i]

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def laplacian_test(out, a, dx_val):
        nx, ny = a.shape
        inv_dx2 = 1.0 / (dx_val * dx_val)
        for i in nb.prange(nx):
            ip = i + 1 if i + 1 < nx else 0
            im = i - 1 if i - 1 >= 0 else nx - 1
            for j in range(ny):
                jp = j + 1 if j + 1 < ny else 0
                jm = j - 1 if j - 1 >= 0 else ny - 1
                out[i, j] = (
                    a[ip, j] + a[im, j] + a[i, jp] + a[i, jm] - 4.0 * a[i, j]
                ) * inv_dx2

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def vprime_test(out, phi, T, lam, mphi, bms, bc, bgc, fc, fgc):
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
                if xb_sq > 0.0:
                    xb = math.sqrt(xb_sq) / T
                if xf_sq > 0.0:
                    xf = math.sqrt(xf_sq) / T
                xb_c = max(min(xb, x_min + h_y * nseg - 1e-12), x_min)
                xf_c = max(min(xf, x_min + h_y * nseg - 1e-12), x_min)
                dJb = cubic_eval(xb_c, x_min, h_y, nseg, c0_b, c1_b, c2_b, c3_b)
                dJf = cubic_eval(xf_c, x_min, h_y, nseg, c0_f, c1_f, c2_f, c3_f)
                dxb_dphi = 0.5 * gb2 * ph / (T2 * max(xb, 1e-20))
                dxf_dphi = 0.5 * gf2 * ph / (T2 * max(xf, 1e-20))
                dV += pref * (2.0 * dJb * dxb_dphi - dJf * dxf_dphi)
                out[i, j] = dV

    # Warmup
    laplacian_test(lap, phi, dx)
    vprime_test(
        Vp,
        phi,
        3000.0,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )

    # Benchmark individual components
    N_iter = 100

    print(f"\n   Component benchmarks ({N_iter} iterations):")

    # Laplacian
    t0 = time.time()
    for _ in range(N_iter):
        laplacian_test(lap, phi, dx)
    t_lap = (time.time() - t0) / N_iter

    # Vprime
    t0 = time.time()
    for _ in range(N_iter):
        vprime_test(
            Vp,
            phi,
            3000.0,
            lam,
            mphi,
            bosonMassSquared,
            bosonCoupling,
            bosonGaugeCoupling,
            fermionCoupling,
            fermionGaugeCoupling,
        )
    t_vp = (time.time() - t0) / N_iter

    # Array operations
    t0 = time.time()
    for _ in range(N_iter):
        temp = lap - 0.3 * pi - Vp / (mphi * mphi)
    t_ops = (time.time() - t0) / N_iter

    # Noise generation (NumPy)
    t0 = time.time()
    for _ in range(N_iter):
        noise = np.random.randn(Nx, Ny).astype(field_dtype) * 0.1
    t_noise = (time.time() - t0) / N_iter

    print(f"   Laplacian:       {t_lap*1000:.3f} ms")
    print(f"   Vprime:          {t_vp*1000:.3f} ms")
    print(f"   Array ops:       {t_ops*1000:.3f} ms")
    print(f"   Noise (NumPy):   {t_noise*1000:.3f} ms")

    t_total = 2 * (t_lap + t_vp) + 2 * t_ops + t_noise  # RK2 ~ 2x each
    print(f"   → Est. per step: {t_total*1000:.3f} ms")
    print(f"   → Est. rate:     {1000/t_total:.1f} steps/sec")

# Threading scalability test
print("\n" + "=" * 70)
print("3. Threading Scalability Test:")
print("=" * 70)

original_threads = nb.get_num_threads()
field_dtype = np.float32

phi = np.random.randn(Nx, Ny).astype(field_dtype)
Vp = np.empty((Nx, Ny), dtype=field_dtype)


@nb.njit(parallel=True, fastmath=True)
def vprime_thread_test(out, phi, T):
    nx, ny = phi.shape
    T2 = T * T
    for i in nb.prange(nx):
        for j in range(ny):
            ph = phi[i, j]
            # Simplified calculation
            out[i, j] = ph * ph * ph - T2 * ph


# Warmup
vprime_thread_test(Vp, phi, 3000.0)

for n_threads in [1, 2, 4, 8]:
    try:
        nb.set_num_threads(n_threads)
        t0 = time.time()
        for _ in range(100):
            vprime_thread_test(Vp, phi, 3000.0)
        elapsed = time.time() - t0
        speedup = (elapsed if n_threads == 1 else t_1thread) / elapsed
        if n_threads == 1:
            t_1thread = elapsed
            speedup = 1.0
        print(
            f"   {n_threads} threads: {elapsed*10:.2f} ms/iter, speedup: {speedup:.2f}×"
        )
    except:
        print(f"   {n_threads} threads: (not available)")

nb.set_num_threads(original_threads)

# Final recommendations
print("\n" + "=" * 70)
print("4. BOTTLENECK ANALYSIS & RECOMMENDATIONS:")
print("=" * 70)

print("\nBased on profiling:")
print("\n📊 If Vprime >> Laplacian:")
print("   → Spline evaluation is the bottleneck")
print("   → Try: Reduce N_Y further (256 → 128 or 64)")
print("   → Or: Simplify thermal corrections")

print("\n📊 If Laplacian ≈ Vprime:")
print("   → Balanced (good!)")
print("   → Limited by compute, not memory")
print("   → Float32 won't help much more")

print("\n📊 If threading speedup < 4× with 8 threads:")
print("   → Memory bandwidth limited")
print("   → Or: Problem too small for threading overhead")
print("   → Try: Larger grid (256×256) for better scaling")

print("\n💡 PRACTICAL SPEEDUP OPTIONS:")
print("\n1. **Reduce accuracy for speed:**")
print("   - N_Y = 128 (instead of 256): ~1.5× faster")
print("   - N_Y = 64: ~2× faster")
print("   - Check comparison plots to verify acceptable")

print("\n2. **Reduce output frequency:**")
print("   - steps = 100_000 (instead of 50_000)")
print("   - Saves time on plotting/I/O")

print("\n3. **Development workflow:**")
print("   - Use 64×64 grid for testing (~4× faster)")
print("   - Use 128×128 for production")

print("\n4. **For very long runs:**")
print("   - Use cloud GPU (Colab is free!)")
print("   - Or: Run overnight with current speed")

print("\n5. **Check your current speed is good:**")
print("   - Is >500 steps/sec? → Already excellent!")
print("   - Is <200 steps/sec? → Threading issue")

print("=" * 70)
