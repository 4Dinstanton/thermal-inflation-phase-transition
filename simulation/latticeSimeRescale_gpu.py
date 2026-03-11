"""
GPU-Accelerated Lattice Simulation using CuPy
==============================================

Requirements:
    pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version

This provides massive speedup (~10-50×) for large grids on NVIDIA GPUs.
Falls back to CPU if CuPy is not available.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("✅ CuPy detected - GPU acceleration available")
    print(f"   GPU: {cp.cuda.Device().name}")
    print(f"   CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    cp = np  # Fallback to NumPy
    GPU_AVAILABLE = False
    print("⚠️  CuPy not found - falling back to CPU (NumPy)")
    print("   Install CuPy for GPU support: pip install cupy-cuda11x")

from scipy.interpolate import CubicSpline
import cosmoTransitions.finiteT as CTFT

# =====================================================
# Performance Settings
# =====================================================
USE_GPU = True and GPU_AVAILABLE  # Auto-disable if GPU not available
USE_FUSED_RK2 = True
USE_FLOAT32 = True  # Recommended for GPU (memory bandwidth)

print("\nConfiguration:")
print(f"  USE_GPU: {USE_GPU}")
print(f"  USE_FUSED_RK2: {USE_FUSED_RK2}")
print(f"  USE_FLOAT32: {USE_FLOAT32}")

# =====================================================
# Thermal dJ tables (CPU-side, copied to GPU if needed)
# =====================================================
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

# Copy spline coefficients to GPU if using GPU
if USE_GPU:
    c0_b_gpu = cp.asarray(c0_b)
    c1_b_gpu = cp.asarray(c1_b)
    c2_b_gpu = cp.asarray(c2_b)
    c3_b_gpu = cp.asarray(c3_b)
    c0_f_gpu = cp.asarray(c0_f)
    c1_f_gpu = cp.asarray(c1_f)
    c2_f_gpu = cp.asarray(c2_f)
    c3_f_gpu = cp.asarray(c3_f)
else:
    c0_b_gpu, c1_b_gpu, c2_b_gpu, c3_b_gpu = c0_b, c1_b, c2_b, c3_b
    c0_f_gpu, c1_f_gpu, c2_f_gpu, c3_f_gpu = c0_f, c1_f, c2_f, c3_f

# =====================================================
# GPU Kernels (CuPy RawKernels)
# =====================================================
if USE_GPU:
    # Laplacian kernel
    laplacian_kernel = cp.RawKernel(
        r"""
    extern "C" __global__
    void laplacian_periodic(const float* a, float* out, int nx, int ny,
                            float inv_dx2) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= nx || j >= ny) return;

        int ip = (i + 1 < nx) ? i + 1 : 0;
        int im = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jp = (j + 1 < ny) ? j + 1 : 0;
        int jm = (j - 1 >= 0) ? j - 1 : ny - 1;

        int idx = i * ny + j;
        out[idx] = (a[ip * ny + j] + a[im * ny + j] +
                   a[i * ny + jp] + a[i * ny + jm] - 4.0f * a[idx]) * inv_dx2;
    }
    """,
        "laplacian_periodic",
    )

    # Vprime kernel with inline cubic spline evaluation
    vprime_kernel = cp.RawKernel(
        r"""
    extern "C" __global__
    void vprime_field(const float* phi, float* out, int nx, int ny,
                     float T, float lam, float mphi,
                     float bms, float bc, float bgc, float fc, float fgc,
                    const double* c0_b, const double* c1_b,
                    const double* c2_b, const double* c3_b,
                    const double* c0_f, const double* c1_f,
                    const double* c2_f, const double* c3_f,
                    float x_min, float h_y, int nseg) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= nx || j >= ny) return;

        int idx = i * ny + j;
        float ph = phi[idx];
        float T2 = T * T;
        float T4 = T2 * T2;
        float pref = T4 / (2.0f * 3.14159265359f * 3.14159265359f);

        // Tree-level derivative
        float dV = lam * ph * ph * ph - mphi * mphi * ph;

        // Thermal inputs
        float gb2 = bc * bc;
        float gg2 = bgc * bgc;
        float gf2 = fc * fc;
        float gfg2 = fgc * fgc;
        float coef_b_T = 0.25f * gb2 + (2.0f / 3.0f) * gg2;

        float xb_sq = bms + 0.5f * gb2 * ph * ph + coef_b_T * T2;
        float xf_sq = 0.5f * gf2 * ph * ph + (1.0f / 6.0f) * gfg2 * T2;

        float xb = (xb_sq > 0.0f) ? sqrtf(xb_sq) / T : 0.0f;
        float xf = (xf_sq > 0.0f) ? sqrtf(xf_sq) / T : 0.0f;

        // Clamp and evaluate cubic spline for dJb
        float xb_c = fmaxf(fminf(xb, x_min + h_y * nseg - 1e-12f), x_min);
        float t_b = (xb_c - x_min) / h_y;
        int seg_b = (int)t_b;
        if (seg_b < 0) seg_b = 0;
        if (seg_b >= nseg) seg_b = nseg - 1;
        float dx_b = xb_c - (x_min + seg_b * h_y);
        float dJb = ((c0_b[seg_b] * dx_b + c1_b[seg_b]) * dx_b +
                     c2_b[seg_b]) * dx_b + c3_b[seg_b];

        // Clamp and evaluate cubic spline for dJf
        float xf_c = fmaxf(fminf(xf, x_min + h_y * nseg - 1e-12f), x_min);
        float t_f = (xf_c - x_min) / h_y;
        int seg_f = (int)t_f;
        if (seg_f < 0) seg_f = 0;
        if (seg_f >= nseg) seg_f = nseg - 1;
        float dx_f = xf_c - (x_min + seg_f * h_y);
        float dJf = ((c0_f[seg_f] * dx_f + c1_f[seg_f]) * dx_f +
                     c2_f[seg_f]) * dx_f + c3_f[seg_f];

        // Derivatives
        float dxb_dphi = 0.5f * gb2 * ph / (T2 * fmaxf(xb, 1e-20f));
        float dxf_dphi = 0.5f * gf2 * ph / (T2 * fmaxf(xf, 1e-20f));

        dV += pref * (2.0f * dJb * dxb_dphi - dJf * dxf_dphi);
        out[idx] = dV;
    }
    """,
        "vprime_field",
    )


# =====================================================
# Wrapper functions
# =====================================================
def laplacian_gpu(phi, dx, out):
    """GPU-accelerated periodic Laplacian."""
    nx, ny = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    block = (16, 16)
    grid = ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1])
    laplacian_kernel(grid, block, (phi, out, nx, ny, inv_dx2))


def vprime_gpu(phi, T, out, lam, mphi, bms, bc, bgc, fc, fgc):
    """GPU-accelerated Vprime field evaluation."""
    nx, ny = phi.shape
    block = (16, 16)
    grid = ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1])
    vprime_kernel(
        grid,
        block,
        (
            phi,
            out,
            nx,
            ny,
            T,
            lam,
            mphi,
            bms,
            bc,
            bgc,
            fc,
            fgc,
            c0_b_gpu,
            c1_b_gpu,
            c2_b_gpu,
            c3_b_gpu,
            c0_f_gpu,
            c1_f_gpu,
            c2_f_gpu,
            c3_f_gpu,
            x_min,
            h_y,
            nseg,
        ),
    )


def rk2_step_gpu(
    phi,
    pi,
    dt,
    dx,
    eta,
    T,
    T_mid,
    mu,
    lam,
    mphi,
    bms,
    bc,
    bgc,
    fc,
    fgc,
    lap,
    Vp,
    phi_mid,
    pi_mid,
):
    """GPU RK2 step (fused version)."""
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt

    # First half-step
    laplacian_gpu(phi, dx, lap)
    vprime_gpu(phi, T, Vp, lam, mphi, bms, bc, bgc, fc, fgc)
    k1_phi = pi
    k1_pi = lap - eta * pi - Vp * inv_mu2
    phi_temp = phi + half_dt * k1_phi
    pi_temp = pi + half_dt * k1_pi

    laplacian_gpu(phi_temp, dx, lap)
    vprime_gpu(phi_temp, T, Vp, lam, mphi, bms, bc, bgc, fc, fgc)
    k2_phi = pi_temp
    k2_pi = lap - eta * pi_temp - Vp * inv_mu2

    phi += half_dt * k2_phi
    pi += half_dt * k2_pi

    # Second half-step
    laplacian_gpu(phi, dx, lap)
    vprime_gpu(phi, T_mid, Vp, lam, mphi, bms, bc, bgc, fc, fgc)
    k1_phi = pi
    k1_pi = lap - eta * pi - Vp * inv_mu2
    phi_mid[:] = phi + half_dt * k1_phi
    pi_mid[:] = pi + half_dt * k1_pi

    laplacian_gpu(phi_mid, dx, lap)
    vprime_gpu(phi_mid, T_mid, Vp, lam, mphi, bms, bc, bgc, fc, fgc)
    k2_phi = pi_mid
    k2_pi = lap - eta * pi_mid - Vp * inv_mu2

    phi += half_dt * k2_phi
    pi += half_dt * k2_pi


# =====================================================
# Physical parameters
# =====================================================
Nx, Ny = 128, 128
dx_phys = 1e-3
dt_phys = 1e-2 * dx_phys**2
Nt = 100_000

lam = 1e-16
mphi = 1_000.0
eta_phys = 0.3
T0 = 3000.0
cooling_rate = 1.0

mu = mphi
dx = mu * dx_phys
dt = mu * dt_phys
eta = eta_phys / mu
cooling_rate = cooling_rate / mu

bosonMassSquared = 1_000_000.0
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

# =====================================================
# Initialize fields
# =====================================================
field_dtype = np.float32 if USE_FLOAT32 else np.float64
xp = cp if USE_GPU else np

phi = xp.asarray((0.01 * np.random.randn(Nx, Ny)).astype(field_dtype))
pi = xp.zeros((Nx, Ny), dtype=field_dtype)

# Preallocate temporaries
lap = xp.empty((Nx, Ny), dtype=field_dtype)
Vp = xp.empty((Nx, Ny), dtype=field_dtype)
phi_mid = xp.empty((Nx, Ny), dtype=field_dtype)
pi_mid = xp.empty((Nx, Ny), dtype=field_dtype)

# =====================================================
# Output
# =====================================================
param_set = "set6"
steps = 50_000
device_str = "gpu" if USE_GPU else "cpu"
save_path = (
    f"figs/latticeSim_{device_str}/{param_set}"
    f"/T0_{int(T0)}_dt_{str(dt)}_interval_{steps}"
)
os.makedirs(save_path, exist_ok=True)

# Create directory for field states
state_path = f"{save_path}/field_states"
os.makedirs(state_path, exist_ok=True)
print(f"Field states will be saved to: {state_path}")


# =====================================================
# Temperature schedule
# =====================================================
def temperature(t):
    return max(T0 - cooling_rate * t, 0.0)


# =====================================================
# Time evolution
# =====================================================
print("\n" + "=" * 60)
print("Starting time evolution...")
print(f"Grid: {Nx}×{Ny}, Steps: {Nt}, dt: {dt:.2e}")
print(f"Device: {'GPU' if USE_GPU else 'CPU'}")
print(f"Precision: {field_dtype.__name__}")
print("=" * 60 + "\n")

# Warmup
print("Warming up...")
if USE_GPU:
    warmup_noise = xp.random.randn(Nx, Ny).astype(field_dtype) * 0.1
    rk2_step_gpu(
        phi,
        pi,
        dt,
        dx,
        eta,
        T0,
        T0,
        mu,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
        lap,
        Vp,
        phi_mid,
        pi_mid,
    )
    cp.cuda.Stream.null.synchronize()  # Ensure GPU completes
print("Warmup complete.\n")

t_start = time.time()

for n in range(Nt):
    t = n * dt
    T = temperature(t)
    T_mid = temperature(t + 0.5 * dt)

    # Generate noise
    noise = xp.random.randn(Nx, Ny).astype(field_dtype)
    noise *= xp.sqrt(2.0 * eta * T * dt / dx**2)

    # RK2 step
    rk2_step_gpu(
        phi,
        pi,
        dt,
        dx,
        eta,
        T,
        T_mid,
        mu,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
        lap,
        Vp,
        phi_mid,
        pi_mid,
    )

    # Add noise
    pi += noise

    if n % steps == 0:
        if USE_GPU:
            cp.cuda.Stream.null.synchronize()  # Wait for GPU
        t_now = time.time()
        elapsed = t_now - t_start
        steps_per_sec = (n + 1) / elapsed if elapsed > 0 else 0
        print(
            f"Step {n}/{Nt} | t={t/mu:.2e} | T={T:.1f} | "
            f"{steps_per_sec:.1f} steps/s"
        )

        # Transfer to CPU for plotting and saving
        phi_cpu = cp.asnumpy(phi) if USE_GPU else phi
        pi_cpu = cp.asnumpy(pi) if USE_GPU else pi

        # Save field state for later revisualization
        state_file = f"{state_path}/state_step_{n:010d}.npz"
        np.savez_compressed(
            state_file,
            phi=phi_cpu,
            pi=pi_cpu,
            step=n,
            time=t,
            temperature=T,
            phi_min=phi_cpu.min(),
            phi_max=phi_cpu.max(),
        )

        # Create snapshot image
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(phi_cpu, origin="lower", cmap="coolwarm", vmin=-1000, vmax=1000)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\phi$")
        ax.set_title(f"t={t/mu:.2e}, T={T:.1f}")
        fig.tight_layout()
        fig.savefig(f"{save_path}/t_{str(t/mu)}.png")
        plt.close(fig)

t_end = time.time()
total_time = t_end - t_start

print("\n" + "=" * 60)
print("Simulation finished!")
print(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
print(f"Average: {Nt/total_time:.1f} steps/second")

# Save simulation metadata
metadata_file = f"{save_path}/simulation_metadata.npz"
np.savez(
    metadata_file,
    # Grid parameters
    Nx=Nx,
    Ny=Ny,
    # Physical parameters
    dx_phys=dx_phys,
    dt_phys=dt_phys,
    mphi=mphi,
    lam=lam,
    eta_phys=eta_phys,
    # Temperature parameters
    T0=T0,
    cooling_rate=cooling_rate,
    # Simulation parameters
    Nt=Nt,
    steps=steps,
    total_time=total_time,
    # Rescaling
    mu=mu,
    dx=dx,
    dt=dt,
    eta=eta,
    # VEV for normalization
    vev=np.sqrt(mphi**2 / lam),
)
print(f"\nMetadata saved to: {metadata_file}")
print(f"Field states saved to: {state_path}/")
print("Use postprocess/revisualize_snapshots.py to replot with different settings")
print(f"Time per step: {total_time*1000/Nt:.3f} ms")
print("=" * 60)
