"""
PyTorch MPS (Metal GPU) accelerated 3D lattice simulation.

Shares physical parameters, CLI args, and checkpoint format with the Numba version
(simulation/latticeSimeRescale_numba.py) for seamless interoperability.

Usage:
    python latticeSim_3D_torch.py --Nx 128 --Ny 128 --Nz 128 --T0 7347 --Nt 100000000 --steps 50000
    python latticeSim_3D_torch.py --resume   # resume from latest checkpoint
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os
import glob
import time
import argparse
from datetime import datetime

import torch

from scipy.interpolate import CubicSpline
import cosmoTransitions.finiteT as CTFT

# =====================================================
# CLI arguments (same as Numba version)
# =====================================================
parser = argparse.ArgumentParser(description="Lattice simulation (PyTorch MPS)")
parser.add_argument("--Nx", type=int, default=256)
parser.add_argument("--Ny", type=int, default=256)
parser.add_argument("--Nz", type=int, default=256)
parser.add_argument("--Nt", type=int, default=100_000_000)
parser.add_argument("--T0", type=float, default=7350.0)
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--resume", action="store_true", default=False)
cli_args = parser.parse_args()

# =====================================================
# Device selection
# =====================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("WARNING: No GPU found, falling back to CPU")

# =====================================================
# Performance settings
# =====================================================
VPRIME_TABLE_SIZE = 16384
TABLE_MARGIN = 50.0

# =====================================================
# Hubble expansion
# =====================================================
HUBBLE_EXPANSION = True
G_STAR = 106.75
M_PL = 2.4e18
DEL_V = 1e28

# =====================================================
# Thermal dJ tables (CPU-side, same as Numba version)
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

# =====================================================
# Physical parameters (identical to Numba version)
# =====================================================
Nx, Ny, Nz = cli_args.Nx, cli_args.Ny, cli_args.Nz
dx_phys = 5e-3
dt_phys = 1e-2 * dx_phys**2
Nt = cli_args.Nt
lam = 1e-16
mphi = 1000.0
eta_phys = 0.3
T0 = cli_args.T0
cooling_rate = 1.0

mu = mphi
dx = mu * dx_phys
dt = mu * dt_phys
eta = eta_phys / mu
cooling_rate_rescaled = cooling_rate / mu

bosonMassSquared = 1_000_000.0
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

vev = np.sqrt(mphi**2 / lam)

# =====================================================
# Hubble helpers
# =====================================================
_hubble_inv_chig2 = G_STAR * np.pi**2 / 30.0
_hubble_inv_3mpl2 = 1.0 / (3.0 * M_PL**2)


def hubble_param(T_val):
    chig2 = 30.0 / (np.pi**2 * G_STAR)
    H2 = (T_val**4 / chig2 + DEL_V) / (3.0 * M_PL**2)
    return np.sqrt(H2)


# =====================================================
# Vprime scalar (CPU, for table building)
# =====================================================
def Vprime_scalar(phi_val, T_val):
    dV = lam * phi_val**3 - mphi**2 * phi_val
    gb2 = bosonCoupling**2
    gg2 = bosonGaugeCoupling**2
    coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2
    xb_sq = bosonMassSquared + 0.5 * gb2 * phi_val**2 + coef_b * T_val**2
    if xb_sq > 0:
        xb = np.sqrt(xb_sq) / T_val
        xb_c = min(max(xb, x_min), x_min + h_y * nseg - 1e-12)
        idx = max(0, min(int((xb_c - x_min) / h_y), nseg - 1))
        dx_s = xb_c - (x_min + idx * h_y)
        dJb_val = ((c0_b[idx] * dx_s + c1_b[idx]) * dx_s + c2_b[idx]) * dx_s + c3_b[idx]
        dxb_dp = 0.5 * gb2 * phi_val / (T_val**2 * max(xb, 1e-20))
        pref = T_val**4 / (2.0 * np.pi**2)
        dV += pref * 2.0 * dJb_val * dxb_dp
    gf2 = fermionCoupling**2
    gfg2 = fermionGaugeCoupling**2
    xf_sq = 0.5 * gf2 * phi_val**2 + (1.0 / 6.0) * gfg2 * T_val**2
    if xf_sq > 0:
        xf = np.sqrt(xf_sq) / T_val
        xf_c = min(max(xf, x_min), x_min + h_y * nseg - 1e-12)
        idx = max(0, min(int((xf_c - x_min) / h_y), nseg - 1))
        dx_s = xf_c - (x_min + idx * h_y)
        dJf_val = ((c0_f[idx] * dx_s + c1_f[idx]) * dx_s + c2_f[idx]) * dx_s + c3_f[idx]
        dxf_dp = 0.5 * gf2 * phi_val / (T_val**2 * max(xf, 1e-20))
        pref = T_val**4 / (2.0 * np.pi**2)
        dV -= pref * dJf_val * dxf_dp
    return dV


def build_vprime_table(T_val, phi_lo, phi_hi, n_table):
    phi_arr = np.linspace(phi_lo, phi_hi, n_table, dtype=np.float64)
    table = np.empty(n_table, dtype=np.float64)
    for i in range(n_table):
        table[i] = Vprime_scalar(phi_arr[i], T_val)
    return table, float(phi_lo), 1.0 / ((phi_hi - phi_lo) / (n_table - 1))


# =====================================================
# PyTorch GPU kernels
# =====================================================
def laplacian_periodic_torch(phi, inv_dx2):
    return (torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0)
            + torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)
            + torch.roll(phi, 1, 2) + torch.roll(phi, -1, 2)
            - 6.0 * phi) * inv_dx2


def vprime_from_table_torch(phi, table, tmin, dinv, npts):
    fidx = (phi - tmin) * dinv
    idx = fidx.long().clamp(0, npts - 2)
    frac = fidx - idx.float()
    return table[idx] + frac * (table[idx + 1] - table[idx])


def rk2_step_table_torch(phi, pi, dt, inv_dx2, eta_eff, inv_mu2, inv_a2,
                          noise, table_T, tmin_T, dinv_T, npts_T,
                          table_Tm, tmin_Tm, dinv_Tm, npts_Tm):
    half_dt = 0.5 * dt

    # First half-step (T)
    lap = laplacian_periodic_torch(phi, inv_dx2)
    dV = vprime_from_table_torch(phi, table_T, tmin_T, dinv_T, npts_T)
    k1_pi = inv_a2 * lap - eta_eff * pi - dV * inv_mu2
    phi_tmp = phi + half_dt * pi
    pi_tmp = pi + half_dt * k1_pi

    lap = laplacian_periodic_torch(phi_tmp, inv_dx2)
    dV = vprime_from_table_torch(phi_tmp, table_T, tmin_T, dinv_T, npts_T)
    k2_pi = inv_a2 * lap - eta_eff * pi_tmp - dV * inv_mu2
    phi.add_(half_dt * pi_tmp)
    pi.add_(half_dt * k2_pi + 0.5 * noise)

    # Second half-step (T_mid)
    lap = laplacian_periodic_torch(phi, inv_dx2)
    dV = vprime_from_table_torch(phi, table_Tm, tmin_Tm, dinv_Tm, npts_Tm)
    k1_pi = inv_a2 * lap - eta_eff * pi - dV * inv_mu2
    phi_tmp = phi + half_dt * pi
    pi_tmp = pi + half_dt * k1_pi

    lap = laplacian_periodic_torch(phi_tmp, inv_dx2)
    dV = vprime_from_table_torch(phi_tmp, table_Tm, tmin_Tm, dinv_Tm, npts_Tm)
    k2_pi = inv_a2 * lap - eta_eff * pi_tmp - dV * inv_mu2
    phi.add_(half_dt * pi_tmp)
    pi.add_(half_dt * k2_pi + 0.5 * noise)


# =====================================================
# Output paths
# =====================================================
param_set = "set6"
steps = cli_args.steps
hubble_tag = "_hubble" if HUBBLE_EXPANSION else "_nohubble"
eta_tag = f"_eta_{eta_phys:g}"
dx_tag = f"_dx_{dx_phys:g}"
dtphys_tag = f"_dtphys_{dt_phys:g}"
save_path = (
    f"data/lattice/{param_set}/{Nx}x{Ny}x{Nz}_T0_{int(T0)}"
    f"{dx_tag}{dtphys_tag}_interval_{steps}_3D_torch{hubble_tag}{eta_tag}"
)
os.makedirs(save_path, exist_ok=True)

state_path = f"{save_path}/field_states"
os.makedirs(state_path, exist_ok=True)

fig_path = f"{save_path}/figs/latticeSnapshot"
os.makedirs(fig_path, exist_ok=True)

print(f"Save path: {save_path}")

# =====================================================
# Fields
# =====================================================
a_current = 1.0
phi_np = 0.01 * np.random.randn(Nx, Ny, Nz).astype(np.float32)
pi_np = np.zeros((Nx, Ny, Nz), dtype=np.float32)
resuming = False
n_start = 0

# =====================================================
# Checkpoint resume
# =====================================================
if cli_args.resume:
    ckpt_files = sorted(glob.glob(f"{state_path}/state_step_*.npz"))
    ckpt_files = [f for f in ckpt_files if "_NaN_debug" not in f]
    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt = np.load(latest)
        ckpt_step = int(ckpt["step"])
        ckpt_phi = ckpt["phi"]
        ckpt_pi = ckpt["pi"]
        if not (np.any(np.isnan(ckpt_phi)) or np.any(np.isnan(ckpt_pi))):
            resuming = True
            n_start = ckpt_step
            phi_np = ckpt_phi.astype(np.float32)
            pi_np = ckpt_pi.astype(np.float32)
            if HUBBLE_EXPANSION and "scale_factor" in ckpt:
                a_current = float(ckpt["scale_factor"])
            print(f"Resuming from step {ckpt_step}, T={float(ckpt['temperature']):.1f}")
        else:
            print("Latest checkpoint has NaN, starting fresh.")
    else:
        print("No checkpoints found, starting fresh.")

# Move fields to GPU
phi = torch.from_numpy(phi_np).to(device)
pi = torch.from_numpy(pi_np).to(device)

inv_dx2 = 1.0 / (dx * dx)
inv_mu2 = 1.0 / (mu * mu)

print(f"\nGrid: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} sites")
print(f"Device: {device}")
print(f"Dtype: {phi.dtype}")
print(f"Memory per field: {phi.nelement() * 4 / 1e6:.1f} MB")
print(f"Total steps: {Nt:,}, snapshot interval: {steps}")

# =====================================================
# Main simulation loop
# =====================================================
print("\nStarting simulation...\n")
t_start = time.time()

vp_table_T_last = -1.0
vp_table_gpu_T = None
vp_table_gpu_Tm = None
vp_tmin_T = 0.0
vp_dinv_T = 1.0
vp_thi_T = 0.0
vp_tmin_Tm = 0.0
vp_dinv_Tm = 1.0
_TABLE_MARGIN_FRAC = 1.0

for n in range(n_start, Nt):
    t_sim = n * dt

    # Hubble / temperature
    if HUBBLE_EXPANSION:
        T = T0 / a_current
        T4 = T * T * T * T
        H_now = math.sqrt((T4 * _hubble_inv_chig2 + DEL_V) * _hubble_inv_3mpl2)
        eta_eff = eta + 3.0 * H_now / mu
        inv_a2 = 1.0 / (a_current * a_current)
        a_current += a_current * H_now * (dt / mu)
        T_mid = T0 / a_current
    else:
        T = max(T0 - cooling_rate_rescaled * t_sim, 0.0)
        T_mid = max(T0 - cooling_rate_rescaled * (t_sim + 0.5 * dt), 0.0)
        eta_eff = eta
        inv_a2 = 1.0
        H_now = 0.0

    # Rebuild V'(phi) table when T changes or phi exceeds table range
    _need_rebuild = vp_table_gpu_T is None
    if not _need_rebuild:
        _need_rebuild = abs(T - vp_table_T_last) / max(abs(T), 1.0) > 1e-4
    if not _need_rebuild:
        _pmin = float(phi.min().item())
        _pmax = float(phi.max().item())
        _need_rebuild = _pmin < vp_tmin_T or _pmax > vp_thi_T
    if _need_rebuild:
        phi_cpu = phi.cpu().numpy()
        _cur_lo = float(phi_cpu.min())
        _cur_hi = float(phi_cpu.max())
        _range = max(_cur_hi - _cur_lo, 1.0)
        _margin = max(_range * _TABLE_MARGIN_FRAC, 20000.0)
        phi_lo = _cur_lo - _margin
        phi_hi = _cur_hi + _margin

        tbl_T, tmin_T, dinv_T = build_vprime_table(T, phi_lo, phi_hi, VPRIME_TABLE_SIZE)
        tbl_Tm, tmin_Tm, dinv_Tm = build_vprime_table(T_mid, phi_lo, phi_hi, VPRIME_TABLE_SIZE)

        vp_table_gpu_T = torch.from_numpy(tbl_T.astype(np.float32)).to(device)
        vp_table_gpu_Tm = torch.from_numpy(tbl_Tm.astype(np.float32)).to(device)
        vp_tmin_T = tmin_T
        vp_dinv_T = dinv_T
        vp_thi_T = phi_hi
        vp_tmin_Tm = tmin_Tm
        vp_dinv_Tm = dinv_Tm
        vp_table_T_last = T

    # Noise
    noise_scale = math.sqrt(2.0 * eta_eff * T * dt / (mu**2 * dx_phys**3))
    noise = torch.randn_like(phi) * noise_scale

    # RK2 step
    rk2_step_table_torch(
        phi, pi, dt, inv_dx2, eta_eff, inv_mu2, inv_a2, noise,
        vp_table_gpu_T, vp_tmin_T, vp_dinv_T, VPRIME_TABLE_SIZE,
        vp_table_gpu_Tm, vp_tmin_Tm, vp_dinv_Tm, VPRIME_TABLE_SIZE,
    )

    # NaN guard (every 100k steps)
    if n % 100000 == 0:
        if torch.any(torch.isnan(phi)) or torch.any(torch.isnan(pi)):
            print(f"NaN detected at step {n}! Aborting.")
            phi_cpu = phi.cpu().numpy()
            pi_cpu = pi.cpu().numpy()
            np.savez_compressed(
                f"{state_path}/state_step_{n:010d}_NaN_debug.npz",
                phi=phi_cpu, pi=pi_cpu, step=n, time=t_sim, temperature=T,
            )
            break

    # Snapshot + progress
    if n % steps == 0:
        elapsed = time.time() - t_start
        done = n - n_start + 1
        sps = done / elapsed if elapsed > 0 else 0
        ms = elapsed / done * 1000 if done > 0 else 0
        eta_min = (Nt - n) / max(sps, 1e-9) / 60

        if HUBBLE_EXPANSION:
            print(
                f"Step {n}/{Nt} | t={t_sim/mu:.2e} | T={T:.1f} | "
                f"a={a_current:.6f} | H={H_now:.2e} | "
                f"{sps:.1f} steps/s | {ms:.2f} ms/step | ETA: {eta_min:.1f} min"
            )
        else:
            print(
                f"Step {n}/{Nt} | t={t_sim/mu:.2e} | T={T:.1f} | "
                f"{sps:.1f} steps/s | {ms:.2f} ms/step | ETA: {eta_min:.1f} min"
            )

        # Save state (transfer to CPU)
        phi_cpu = phi.cpu().numpy()
        pi_cpu = pi.cpu().numpy()

        # Compute laplacian and Vprime on CPU for diagnostic save
        save_dict = dict(
            phi=phi_cpu, pi=pi_cpu,
            step=n, time=t_sim, temperature=T,
            phi_min=phi_cpu.min(), phi_max=phi_cpu.max(),
        )
        if HUBBLE_EXPANSION:
            save_dict["scale_factor"] = a_current
            save_dict["hubble"] = H_now
        state_file = f"{state_path}/state_step_{n:010d}.npz"
        np.savez_compressed(state_file, **save_dict)

        # Snapshot image (z-midplane)
        phi_slice = phi_cpu[:, :, Nz // 2]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(phi_slice, origin="lower", cmap="coolwarm", vmin=-2e11, vmax=2e11)
        fig.colorbar(im, ax=ax, label=r"$\phi$")
        ax.set_title(
            f"Step {n:,} | t={t_sim/mu:.2e} | T={T:.1f}\n"
            f"$\\phi$: [{phi_cpu.min():.2e}, {phi_cpu.max():.2e}]"
        )
        fig.tight_layout()
        fig.savefig(f"{fig_path}/t_{t_sim/mu}.png")
        plt.close(fig)

# =====================================================
# Finish
# =====================================================
t_end = time.time()
total_time = t_end - t_start
steps_run = Nt - n_start
print("\n" + "=" * 60)
print("SIMULATION FINISHED!")
print("=" * 60)
print(f"  Total time: {total_time/60:.2f} min ({total_time:.1f} s)")
print(f"  Average: {steps_run/max(total_time,1e-9):.1f} steps/s")
print(f"  Time per step: {total_time*1000/max(steps_run,1):.3f} ms")
print(f"  Grid: {Nx}x{Ny}x{Nz}, Device: {device}")
print(f"  Output: {save_path}")
print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

metadata_file = f"{save_path}/simulation_metadata.npz"
np.savez(
    metadata_file,
    Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt, T0=T0, steps=steps,
    dx_phys=dx_phys, dt_phys=dt_phys, lam=lam, mphi=mphi,
    eta_phys=eta_phys, cooling_rate=cooling_rate,
    hubble_expansion=HUBBLE_EXPANSION, G_STAR=G_STAR, M_PL=M_PL, DEL_V=DEL_V,
    backend="pytorch_mps",
)
print(f"Metadata saved to: {metadata_file}")
