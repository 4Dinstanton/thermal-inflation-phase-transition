#!/usr/bin/env python
"""
Taichi-Accelerated 3D Lattice Simulation — Full 2nd-Order Langevin (RK2 Midpoint)
==================================================================================

Uses Taichi for GPU acceleration on Apple Silicon (Metal), NVIDIA (CUDA), or CPU.
Each RK2 pass compiles into a **single** GPU kernel with fused Laplacian + V'(phi)
+ RK2 update, matching the Numba rk2_step_inline architecture on GPU.

Requirements:
    pip install taichi numpy matplotlib scipy

Usage:
    python simulation/latticeSimeRescale_taichi.py --T0 1588 --Nx 128 --Nt 100000
    python simulation/latticeSimeRescale_taichi.py --arch metal --T0 1588
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import CubicSpline
    import cosmoTransitions.finiteT as CTFT
except ImportError:
    print("ERROR: scipy and cosmoTransitions are required.")
    sys.exit(1)

import taichi as ti

# =====================================================
# CLI Arguments
# =====================================================
parser = argparse.ArgumentParser(
    description="Taichi GPU Lattice Langevin simulation (RK2, fused kernels)"
)
parser.add_argument("--Nx", type=int, default=256, help="Lattice size x (default: 256)")
parser.add_argument("--Ny", type=int, default=None, help="Default: same as Nx")
parser.add_argument("--Nz", type=int, default=None, help="Default: same as Nx")
parser.add_argument(
    "--T0", type=float, default=7350.0, help="Initial temperature [GeV]"
)
parser.add_argument("--Nt", type=int, default=100_000_000, help="Total timesteps")
parser.add_argument("--steps", type=int, default=100_000, help="Snapshot interval")
parser.add_argument("--dt_factor", type=float, default=1.0, help="dt_phys multiplier")
parser.add_argument(
    "--arch",
    type=str,
    default="auto",
    choices=["auto", "metal", "cuda", "cpu"],
    help="Taichi backend (default: auto-detect metal→cuda→cpu)",
)
parser.add_argument(
    "--boson_coupling",
    type=float,
    default=None,
    help="Boson Yukawa coupling (default: 1.09)",
)
parser.add_argument(
    "--fermion_coupling",
    type=float,
    default=None,
    help="Fermion Yukawa coupling (default: 1.09)",
)
parser.add_argument(
    "--counterterm",
    action="store_true",
    help="Enable lattice counterterm",
)
parser.add_argument(
    "--potential_type",
    type=str,
    default="V_correct",
    choices=["V_p", "V_correct", "fermion_only"],
    help="Thermal potential convention",
)
parser.add_argument("--nb", type=int, default=20, help="Boson species multiplicity")
parser.add_argument("--nf", type=int, default=20, help="Fermion species multiplicity")
parser.add_argument("--no_hubble", action="store_true", help="Disable Hubble expansion")
parser.add_argument(
    "--resume", action="store_true", help="Resume from latest checkpoint"
)
parser.add_argument(
    "--diag_energy", action="store_true", help="Energy diagnostics at save points"
)
parser.add_argument(
    "--disable_noise", action="store_true", help="Disable stochastic noise"
)
parser.add_argument(
    "--eta", type=float, default=None, help="η_phys [GeV] (default: 1000)"
)
parser.add_argument("--dx", type=float, default=None, help="dx_phys (default: 1e-3)")
parser.add_argument(
    "--dt", type=float, default=None, help="dt_phys (default: 2.5e-4 * dt_factor)"
)
parser.add_argument("--phi_threshold", type=float, default=None)
parser.add_argument("--steps_dense", type=int, default=None)
parser.add_argument("--boson_mass_squared", type=float, default=None)
parser.add_argument("--lam", type=float, default=None, help="Quartic coupling λ")
parser.add_argument("--mphi", type=float, default=None, help="Flaton mass m_phi [GeV]")
parser.add_argument(
    "--integrator",
    type=str,
    default="nonfused_inline",
    choices=["nonfused_inline", "fused_inline"],
    help="nonfused_inline: 2 passes/step (default). "
    "fused_inline: 4 passes/step (two dt/2 half-steps with T_mid).",
)
cli_args = parser.parse_args()
if cli_args.Ny is None:
    cli_args.Ny = cli_args.Nx
if cli_args.Nz is None:
    cli_args.Nz = cli_args.Nx


# =====================================================
# Taichi init
# =====================================================
def _select_arch(requested):
    if requested == "metal":
        return ti.metal
    if requested == "cuda":
        return ti.cuda
    if requested == "cpu":
        return ti.cpu
    # auto: try metal → cuda → cpu
    try:
        ti.init(arch=ti.metal, default_fp=ti.f32)
        return None  # already initialized
    except Exception:
        pass
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32)
        return None
    except Exception:
        pass
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    return None


_requested_arch = _select_arch(cli_args.arch)
if _requested_arch is not None:
    ti.init(arch=_requested_arch, default_fp=ti.f32)

_actual_arch = ti.cfg.arch
_arch_name = {ti.metal: "metal", ti.cuda: "cuda", ti.cpu: "cpu"}.get(
    _actual_arch, str(_actual_arch)
)
print(f"Taichi backend: {_arch_name}")

# =====================================================
# Potential Coefficients
# =====================================================
n_b = cli_args.nb
n_f = cli_args.nf
_pot_c = [2.0, -1.0]
if cli_args.potential_type == "V_correct":
    _pot_c = [1.0, 1.0]
elif cli_args.potential_type == "fermion_only":
    _pot_c = [0.0, 1.0]
_pot_c[0] *= n_b
_pot_c[1] *= n_f
print(
    f"Potential: {cli_args.potential_type}  "
    f"(nb={n_b}, nf={n_f}, coeffs: boson={_pot_c[0]:g}, fermion={_pot_c[1]:g})"
)

# =====================================================
# Thermal dJ Splines (CPU, then upload coeffs)
# =====================================================
YMAX = 100.0
N_Y = 256
y2_grid = np.linspace(0.0, YMAX, N_Y, dtype=np.float64)
dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid], dtype=np.float64)
dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid], dtype=np.float64)

cs_b = CubicSpline(y2_grid, dJb_grid, bc_type="not-a-knot")
cs_f = CubicSpline(y2_grid, dJf_grid, bc_type="not-a-knot")

_spline_x_min = float(y2_grid[0])
_spline_h = float(y2_grid[1] - y2_grid[0])
_spline_inv_h = 1.0 / _spline_h
_spline_nseg = int(y2_grid.size - 1)
_spline_xhi = _spline_x_min + _spline_h * _spline_nseg - 1e-12

# CPU arrays (may be corrected by counterterm before upload)
_c0_b_np = cs_b.c[0].astype(np.float32)
_c1_b_np = cs_b.c[1].astype(np.float32)
_c2_b_np = cs_b.c[2].astype(np.float32)
_c3_b_np = cs_b.c[3].astype(np.float32)
_c0_f_np = cs_f.c[0].astype(np.float32)
_c1_f_np = cs_f.c[1].astype(np.float32)
_c2_f_np = cs_f.c[2].astype(np.float32)
_c3_f_np = cs_f.c[3].astype(np.float32)

# =====================================================
# Physical Parameters
# =====================================================
Nx, Ny, Nz = cli_args.Nx, cli_args.Ny, cli_args.Nz

lam = cli_args.lam if cli_args.lam is not None else 1e-16
mphi = cli_args.mphi if cli_args.mphi is not None else 1000.0
dx_phys = cli_args.dx if cli_args.dx is not None else 1e-3
dt_phys = cli_args.dt if cli_args.dt is not None else 2.5e-4 * cli_args.dt_factor
Nt = cli_args.Nt
T0 = cli_args.T0
eta_phys = cli_args.eta if cli_args.eta is not None else 1000.0
HUBBLE_EXPANSION = not cli_args.no_hubble
DISABLE_THERMAL_NOISE = cli_args.disable_noise

bosonMassSquared = (
    cli_args.boson_mass_squared
    if cli_args.boson_mass_squared is not None
    else 1_000_000.0
)
bosonCoupling = 1.09 if cli_args.boson_coupling is None else cli_args.boson_coupling
bosonGaugeCoupling = 1.05
fermionCoupling = (
    1.09 if cli_args.fermion_coupling is None else cli_args.fermion_coupling
)
fermionGaugeCoupling = 1.05

mu = mphi
dx = mu * dx_phys
dt = mu * dt_phys
eta = eta_phys / mu
vev = math.sqrt(mphi**2 / lam)
cooling_rate = 1.0 / mu

gb2 = bosonCoupling**2
gg2 = bosonGaugeCoupling**2
gf2 = fermionCoupling**2
gfg2 = fermionGaugeCoupling**2
coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2
mphi_sq = mphi**2

# Dense snapshot switching
_dense_enabled = cli_args.phi_threshold is not None and cli_args.steps_dense is not None
_dense_threshold = cli_args.phi_threshold if _dense_enabled else 0.0
_dense_steps = cli_args.steps_dense if _dense_enabled else cli_args.steps
_dense_active = False

# CFL check
dt_cfl = dx / math.sqrt(3.0)
cfl_ratio = dt / dt_cfl
print(f"\nGrid: {Nx}x{Ny}x{Nz}  |  dx_phys={dx_phys}  |  dt_phys={dt_phys}")
print(f"CFL: dt/dt_CFL = {cfl_ratio:.4e} ({'SAFE' if cfl_ratio < 1 else 'UNSAFE!'})")
if cfl_ratio >= 1.0:
    print("  WARNING: dt exceeds CFL limit! Simulation may be unstable.")
print(f"eta_phys = {eta_phys}  |  eta (rescaled) = {eta:.4f}")
print(f"Couplings: boson={bosonCoupling:g}, fermion={fermionCoupling:g}")

# =====================================================
# Counterterm
# =====================================================
if cli_args.counterterm:
    from scipy.integrate import quad as _ct_quad

    def _dJ_soft_boson(y, x_cut):
        if x_cut < 1e-10 or y < 1e-30:
            return 0.0

        def _integrand(x):
            z = math.sqrt(x * x + y * y)
            if z > 50.0:
                return 0.0
            return x * x * y / (z * (math.exp(z) - 1.0))

        val, _ = _ct_quad(_integrand, 0, x_cut, limit=200)
        return val / (2.0 * math.pi**2)

    def _dJ_soft_fermion(y, x_cut):
        if x_cut < 1e-10 or y < 1e-30:
            return 0.0

        def _integrand(x):
            z = math.sqrt(x * x + y * y)
            if z > 50.0:
                return 0.0
            return x * x * y / (z * (math.exp(z) + 1.0))

        val, _ = _ct_quad(_integrand, 0, x_cut, limit=200)
        return val / (2.0 * math.pi**2)

    _ct_x_cut = math.pi / (dx_phys * T0)
    print("\n[Counterterm] Applying lattice counterterm:")
    print(f"  dx_phys = {dx_phys:g}, T0 = {T0:g}")
    print(f"  x_cut = {_ct_x_cut:.4f}")
    _t_ct0 = time.time()
    dJb_soft = np.array(
        [_dJ_soft_boson(float(y), _ct_x_cut) for y in y2_grid], dtype=np.float64
    )
    dJf_soft = np.array(
        [_dJ_soft_fermion(float(y), _ct_x_cut) for y in y2_grid], dtype=np.float64
    )
    cs_b_ct = CubicSpline(y2_grid, dJb_grid - dJb_soft, bc_type="not-a-knot")
    cs_f_ct = CubicSpline(y2_grid, dJf_grid - dJf_soft, bc_type="not-a-knot")
    _c0_b_np[:] = cs_b_ct.c[0].astype(np.float32)
    _c1_b_np[:] = cs_b_ct.c[1].astype(np.float32)
    _c2_b_np[:] = cs_b_ct.c[2].astype(np.float32)
    _c3_b_np[:] = cs_b_ct.c[3].astype(np.float32)
    _c0_f_np[:] = cs_f_ct.c[0].astype(np.float32)
    _c1_f_np[:] = cs_f_ct.c[1].astype(np.float32)
    _c2_f_np[:] = cs_f_ct.c[2].astype(np.float32)
    _c3_f_np[:] = cs_f_ct.c[3].astype(np.float32)
    print(f"  Done in {time.time() - _t_ct0:.2f}s")

# Hubble constants
G_STAR = 106.75
M_PL = 2.435e18
DEL_V = 0.25 * lam * vev**4


def hubble_param(T_val):
    chig2 = 30.0 / (math.pi**2 * G_STAR)
    H2 = (T_val**4 / chig2 + DEL_V) / (3.0 * M_PL**2)
    return math.sqrt(H2)


# =====================================================
# Taichi Fields
# =====================================================
phi_field = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
pi_field = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
phi_mid = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
pi_mid = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))

# Spline coefficient fields (1D)
c0_b = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c1_b = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c2_b = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c3_b = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c0_f = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c1_f = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c2_f = ti.field(dtype=ti.f32, shape=(_spline_nseg,))
c3_f = ti.field(dtype=ti.f32, shape=(_spline_nseg,))

# Upload spline coefficients
c0_b.from_numpy(_c0_b_np)
c1_b.from_numpy(_c1_b_np)
c2_b.from_numpy(_c2_b_np)
c3_b.from_numpy(_c3_b_np)
c0_f.from_numpy(_c0_f_np)
c1_f.from_numpy(_c1_f_np)
c2_f.from_numpy(_c2_f_np)
c3_f.from_numpy(_c3_f_np)

# Energy reduction fields
e_kin_field = ti.field(dtype=ti.f32, shape=())
e_grad_field = ti.field(dtype=ti.f32, shape=())
e_pot_field = ti.field(dtype=ti.f32, shape=())

# =====================================================
# Taichi Kernels
# =====================================================
# All physics constants are captured as Python variables in ti.func/ti.kernel.
# Taichi compiles them as shader constants.
_TI_NX = Nx
_TI_NY = Ny
_TI_NZ = Nz
_TI_LAM = float(lam)
_TI_MPHI_SQ = float(mphi_sq)
_TI_BMS = float(bosonMassSquared)
_TI_GB2 = float(gb2)
_TI_GF2 = float(gf2)
_TI_GFG2 = float(gfg2)
_TI_COEF_B = float(coef_b)
_TI_POT_C0 = float(_pot_c[0])
_TI_POT_C1 = float(_pot_c[1])
_TI_X_MIN = float(_spline_x_min)
_TI_H_Y = float(_spline_h)
_TI_INV_HY = float(_spline_inv_h)
_TI_XHI = float(_spline_xhi)
_TI_NSEG = int(_spline_nseg)
_TI_INV_MU2 = 1.0 / (mu * mu)


@ti.func
def _vprime_inline(ph: ti.f32, T_val: ti.f32, T2: ti.f32, pref: ti.f32) -> ti.f32:
    """Per-site V'(phi) with cubic spline thermal corrections."""
    dV = _TI_LAM * ph * ph * ph - _TI_MPHI_SQ * ph

    xb_sq = _TI_BMS + 0.5 * _TI_GB2 * ph * ph + _TI_COEF_B * T2
    xb = ti.f32(0.0)
    if xb_sq > 0.0:
        xb = ti.sqrt(xb_sq) / T_val

    xf_sq = 0.5 * _TI_GF2 * ph * ph + (1.0 / 6.0) * _TI_GFG2 * T2
    xf = ti.f32(0.0)
    if xf_sq > 0.0:
        xf = ti.sqrt(xf_sq) / T_val

    # Boson spline eval
    xb_c = ti.max(ti.min(xb, _TI_XHI), _TI_X_MIN)
    t_b = (xb_c - _TI_X_MIN) * _TI_INV_HY
    si_b = ti.cast(t_b, ti.i32)
    si_b = ti.max(ti.min(si_b, _TI_NSEG - 1), 0)
    dx_b = xb_c - (_TI_X_MIN + ti.cast(si_b, ti.f32) * _TI_H_Y)
    dJb = ((c0_b[si_b] * dx_b + c1_b[si_b]) * dx_b + c2_b[si_b]) * dx_b + c3_b[si_b]

    # Fermion spline eval
    xf_c = ti.max(ti.min(xf, _TI_XHI), _TI_X_MIN)
    t_f = (xf_c - _TI_X_MIN) * _TI_INV_HY
    si_f = ti.cast(t_f, ti.i32)
    si_f = ti.max(ti.min(si_f, _TI_NSEG - 1), 0)
    dx_f = xf_c - (_TI_X_MIN + ti.cast(si_f, ti.f32) * _TI_H_Y)
    dJf = ((c0_f[si_f] * dx_f + c1_f[si_f]) * dx_f + c2_f[si_f]) * dx_f + c3_f[si_f]

    # Chain rule
    dxb_dphi = 0.5 * _TI_GB2 * ph / (T2 * ti.max(xb, ti.f32(1e-20)))
    dxf_dphi = 0.5 * _TI_GF2 * ph / (T2 * ti.max(xf, ti.f32(1e-20)))

    dV += pref * (_TI_POT_C0 * dJb * dxb_dphi + _TI_POT_C1 * dJf * dxf_dphi)
    return dV


@ti.kernel
def rk2_pass1(
    dt_val: ti.f32,
    inv_dx2: ti.f32,
    eta_eff: ti.f32,
    T_val: ti.f32,
    inv_a2: ti.f32,
):
    """Pass 1: compute k1 at current state, write midpoint to phi_mid/pi_mid."""
    T2 = T_val * T_val
    pref = T2 * T2 / (2.0 * 3.14159265358979323846 * 3.14159265358979323846)
    half_dt = 0.5 * dt_val
    for i, j, k in phi_field:
        ip = (i + 1) % _TI_NX
        im = (i - 1 + _TI_NX) % _TI_NX
        jp = (j + 1) % _TI_NY
        jm = (j - 1 + _TI_NY) % _TI_NY
        kp = (k + 1) % _TI_NZ
        km = (k - 1 + _TI_NZ) % _TI_NZ

        lap = (
            phi_field[ip, j, k]
            + phi_field[im, j, k]
            + phi_field[i, jp, k]
            + phi_field[i, jm, k]
            + phi_field[i, j, kp]
            + phi_field[i, j, km]
            - 6.0 * phi_field[i, j, k]
        ) * inv_dx2

        ph = phi_field[i, j, k]
        dV = _vprime_inline(ph, T_val, T2, pref)
        pi_v = pi_field[i, j, k]

        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * _TI_INV_MU2
        phi_mid[i, j, k] = ph + half_dt * pi_v
        pi_mid[i, j, k] = pi_v + half_dt * k_pi


@ti.kernel
def rk2_pass2(
    dt_val: ti.f32,
    inv_dx2: ti.f32,
    eta_eff: ti.f32,
    T_val: ti.f32,
    inv_a2: ti.f32,
    noise_scale: ti.f32,
    use_noise: ti.i32,
):
    """Pass 2: compute k2 at midpoint, advance phi/pi + stochastic noise."""
    T2 = T_val * T_val
    pref = T2 * T2 / (2.0 * 3.14159265358979323846 * 3.14159265358979323846)
    for i, j, k in phi_mid:
        ip = (i + 1) % _TI_NX
        im = (i - 1 + _TI_NX) % _TI_NX
        jp = (j + 1) % _TI_NY
        jm = (j - 1 + _TI_NY) % _TI_NY
        kp = (k + 1) % _TI_NZ
        km = (k - 1 + _TI_NZ) % _TI_NZ

        lap = (
            phi_mid[ip, j, k]
            + phi_mid[im, j, k]
            + phi_mid[i, jp, k]
            + phi_mid[i, jm, k]
            + phi_mid[i, j, kp]
            + phi_mid[i, j, km]
            - 6.0 * phi_mid[i, j, k]
        ) * inv_dx2

        ph_m = phi_mid[i, j, k]
        dV = _vprime_inline(ph_m, T_val, T2, pref)
        pi_m = pi_mid[i, j, k]

        k_pi = inv_a2 * lap - eta_eff * pi_m - dV * _TI_INV_MU2

        ns = ti.f32(0.0)
        if use_noise:
            ns = noise_scale * ti.randn()

        phi_field[i, j, k] += dt_val * pi_m
        pi_field[i, j, k] += dt_val * k_pi + ns


# =====================================================
# Fused Inline Kernels (4-pass: two dt/2 half-steps)
# =====================================================
# Pass 1 & 3 reuse rk2_pass1 (same structure).
# Passes 2 & 4 differ: they advance by half_dt and add half the noise.


@ti.kernel
def fused_pass2(
    half_dt: ti.f32,
    inv_dx2: ti.f32,
    eta_eff: ti.f32,
    T_val: ti.f32,
    inv_a2: ti.f32,
    half_noise_scale: ti.f32,
    use_noise: ti.i32,
):
    """Fused pass 2/4: k2 at midpoint, advance phi/pi by half_dt + half noise."""
    T2 = T_val * T_val
    pref = T2 * T2 / (2.0 * 3.14159265358979323846 * 3.14159265358979323846)
    for i, j, k in phi_mid:
        ip = (i + 1) % _TI_NX
        im = (i - 1 + _TI_NX) % _TI_NX
        jp = (j + 1) % _TI_NY
        jm = (j - 1 + _TI_NY) % _TI_NY
        kp = (k + 1) % _TI_NZ
        km = (k - 1 + _TI_NZ) % _TI_NZ

        lap = (
            phi_mid[ip, j, k]
            + phi_mid[im, j, k]
            + phi_mid[i, jp, k]
            + phi_mid[i, jm, k]
            + phi_mid[i, j, kp]
            + phi_mid[i, j, km]
            - 6.0 * phi_mid[i, j, k]
        ) * inv_dx2

        ph_m = phi_mid[i, j, k]
        dV = _vprime_inline(ph_m, T_val, T2, pref)
        pi_m = pi_mid[i, j, k]

        k_pi = inv_a2 * lap - eta_eff * pi_m - dV * _TI_INV_MU2

        ns = ti.f32(0.0)
        if use_noise:
            ns = half_noise_scale * ti.randn()

        phi_field[i, j, k] += half_dt * pi_m
        pi_field[i, j, k] += half_dt * k_pi + ns


@ti.kernel
def compute_energy(inv_dx2: ti.f32, inv_a2: ti.f32):
    """Energy diagnostics: E_kin, E_grad, E_pot (tree-level)."""
    n_sites = ti.f32(_TI_NX * _TI_NY * _TI_NZ)
    for i, j, k in phi_field:
        ip = (i + 1) % _TI_NX
        jp = (j + 1) % _TI_NY
        kp = (k + 1) % _TI_NZ
        ph = phi_field[i, j, k]
        pi_v = pi_field[i, j, k]

        e_kin_field[None] += ti.f32(0.5) * pi_v * pi_v / n_sites

        dx_f = phi_field[ip, j, k] - ph
        dy_f = phi_field[i, jp, k] - ph
        dz_f = phi_field[i, j, kp] - ph
        e_grad_field[None] += (
            ti.f32(0.5)
            * inv_a2
            * inv_dx2
            * (dx_f * dx_f + dy_f * dy_f + dz_f * dz_f)
            / n_sites
        )

        ph2 = ph * ph
        e_pot_field[None] += (
            (ti.f32(0.25) * _TI_LAM * ph2 * ph2 - ti.f32(0.5) * _TI_MPHI_SQ * ph2)
            * _TI_INV_MU2
            / n_sites
        )


@ti.kernel
def init_random(scale: ti.f32):
    for i, j, k in phi_field:
        phi_field[i, j, k] = scale * ti.randn()
        pi_field[i, j, k] = 0.0


# =====================================================
# Initialize Fields
# =====================================================
init_random(0.01)

# =====================================================
# Output Paths
# =====================================================
param_set = "set7"
steps = cli_args.steps
hubble_tag = "_hubble" if HUBBLE_EXPANSION else "_nohubble"
eta_tag = f"_eta_{eta_phys:g}"
dx_tag = f"_dx_{dx_phys:g}"
dtphys_tag = f"_dtphys_{dt_phys:g}"
coupling_tag = f"_gb_{bosonCoupling:g}_gf_{fermionCoupling:g}"
ct_tag = "_CT" if cli_args.counterterm else ""
pot_tag = f"_{cli_args.potential_type}" if cli_args.potential_type != "V_p" else ""
nf_nb_tag = f"_nf{n_f}_nb{n_b}" if (n_f != 1 or n_b != 1) else ""
backend_tag = f"_taichi_{_arch_name}"
save_path = (
    f"data/lattice/{param_set}/{Nx}x{Ny}x{Nz}_T0_{int(T0)}"
    f"{dx_tag}{dtphys_tag}_interval_{steps}_3D{hubble_tag}{eta_tag}"
    f"{coupling_tag}_rk2_inline{ct_tag}{pot_tag}{nf_nb_tag}{backend_tag}"
)
os.makedirs(save_path, exist_ok=True)
state_path = os.path.join(save_path, "field_states")
fig_path = os.path.join(save_path, "figs", "latticeSnapshot")
os.makedirs(state_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

CHECKPOINT_EVERY = 5

# =====================================================
# Resume
# =====================================================
import glob as _glob

n_start = 0
a_current = 1.0
resuming = False

if cli_args.resume:
    ckpt_files = sorted(_glob.glob(os.path.join(state_path, "state_step_*.npz")))
    for ckpt_path in reversed(ckpt_files):
        d = np.load(ckpt_path)
        if "pi" not in d:
            continue
        ckpt_phi = d["phi"]
        if np.any(np.isnan(ckpt_phi)):
            continue
        phi_field.from_numpy(ckpt_phi.astype(np.float32))
        pi_field.from_numpy(d["pi"].astype(np.float32))
        n_start = int(d["step"])
        if "scale_factor" in d:
            a_current = float(d["scale_factor"])
        resuming = True
        print(f"Resumed from {ckpt_path} (step {n_start}, a={a_current:.6f})")
        break
    if not resuming:
        print("No valid checkpoint found — starting fresh")

# =====================================================
# Print Summary
# =====================================================
print(f"\n{'=' * 60}")
print("Taichi Lattice Simulation — Full 2nd-Order Langevin (RK2)")
print(f"{'=' * 60}")
print(f"Grid: {Nx}x{Ny}x{Nz}  |  Backend: {_arch_name}")
print(f"dx_phys={dx_phys}  dt_phys={dt_phys}  eta_phys={eta_phys}")
print(f"T0={T0}  mphi={mphi}  lam={lam}  VEV={vev:.4e}")
print(f"Couplings: gB={bosonCoupling}, gF={fermionCoupling}")
print(f"Hubble: {'ON' if HUBBLE_EXPANSION else 'OFF'}")
print(f"Noise: {'OFF' if DISABLE_THERMAL_NOISE else 'ON'}")
print(f"Counterterm: {'ON' if cli_args.counterterm else 'OFF'}")
print(f"Steps: {Nt}  |  Save every: {steps}")
if _dense_enabled:
    print(f"Dense snapshots: threshold={_dense_threshold:.1f}, interval={_dense_steps}")
if cli_args.diag_energy:
    _ekin_eq = mu * mu * T0 / (2.0 * dx_phys**3)
    print(f"Energy diagnostics: ENABLED (E_kin_equip(T0) = {_ekin_eq:.4e})")
print(f"Output: {save_path}")
print(f"{'=' * 60}\n")

# =====================================================
# JIT warmup
# =====================================================
USE_FUSED = cli_args.integrator == "fused_inline"
_intg_name = (
    "rk2_fused_inline (4 passes)" if USE_FUSED else "rk2_nonfused_inline (2 passes)"
)
print(f"Integrator: {_intg_name}")

print("Warming up Taichi kernels (first call triggers compilation)...")
_t_warmup = time.time()
_inv_dx2_f = float(1.0 / (dx * dx))
rk2_pass1(float(dt), _inv_dx2_f, float(eta), float(T0), 1.0)
rk2_pass2(float(dt), _inv_dx2_f, float(eta), float(T0), 1.0, 0.0, 0)
if USE_FUSED:
    fused_pass2(float(0.5 * dt), _inv_dx2_f, float(eta), float(T0), 1.0, 0.0, 0)
if cli_args.diag_energy:
    e_kin_field[None] = 0.0
    e_grad_field[None] = 0.0
    e_pot_field[None] = 0.0
    compute_energy(_inv_dx2_f, 1.0)
ti.sync()
print(f"Warmup done in {time.time() - _t_warmup:.2f}s")

# Re-initialize after warmup
if not resuming:
    init_random(0.01)

# =====================================================
# Main Simulation Loop
# =====================================================
_total_saves = 0
t_start = time.time()

_hubble_inv_chig2 = G_STAR * math.pi**2 / 30.0
_hubble_inv_3mpl2 = 1.0 / (3.0 * M_PL**2)

_inv_dx2_val = float(1.0 / (dx * dx))
_use_noise_int = 0 if DISABLE_THERMAL_NOISE else 1

for n in range(n_start, Nt):
    t = n * dt

    if HUBBLE_EXPANSION:
        T = T0 / a_current
        _T4 = T**4
        H_now = math.sqrt((_T4 * _hubble_inv_chig2 + DEL_V) * _hubble_inv_3mpl2)
        eta_eff = eta + 3.0 * H_now / mu
        inv_a2 = 1.0 / (a_current * a_current)
        a_current += a_current * H_now * (dt / mu)
    else:
        T = max(T0 - cooling_rate * t, 0.0)
        eta_eff = eta
        inv_a2 = 1.0
        H_now = 0.0

    # Noise amplitude (FDT)
    if DISABLE_THERMAL_NOISE:
        ns_val = 0.0
    else:
        ns_val = math.sqrt(2.0 * eta_eff * T * dt / (mu**2 * dx_phys**3))

    if USE_FUSED:
        # Fused RK2: 4 passes, two dt/2 half-steps
        if HUBBLE_EXPANSION:
            T_m = T0 / a_current
        else:
            T_m = max(T0 - cooling_rate * (t + 0.5 * dt), 0.0)
        _half_dt = float(0.5 * dt)
        _half_ns = float(0.5 * ns_val)
        # Half-step 1 (at T)
        rk2_pass1(_half_dt, _inv_dx2_val, float(eta_eff), float(T), float(inv_a2))
        fused_pass2(
            _half_dt,
            _inv_dx2_val,
            float(eta_eff),
            float(T),
            float(inv_a2),
            _half_ns,
            _use_noise_int,
        )
        # Half-step 2 (at T_mid)
        rk2_pass1(_half_dt, _inv_dx2_val, float(eta_eff), float(T_m), float(inv_a2))
        fused_pass2(
            _half_dt,
            _inv_dx2_val,
            float(eta_eff),
            float(T_m),
            float(inv_a2),
            _half_ns,
            _use_noise_int,
        )
    else:
        # Nonfused RK2: 2 passes, single full dt step
        rk2_pass1(float(dt), _inv_dx2_val, float(eta_eff), float(T), float(inv_a2))
        rk2_pass2(
            float(dt),
            _inv_dx2_val,
            float(eta_eff),
            float(T),
            float(inv_a2),
            float(ns_val),
            _use_noise_int,
        )

    # Dense snapshot switching
    _cur_steps = steps
    if _dense_enabled and not _dense_active:
        if n % 1000 == 0 and n > n_start:
            phi_np = phi_field.to_numpy()
            _phi_absmax = max(abs(phi_np.min()), abs(phi_np.max()))
            if _phi_absmax > _dense_threshold:
                _dense_active = True
                _cur_steps = _dense_steps
                print(
                    f"\n*** phi threshold exceeded: max|phi|={_phi_absmax:.1f} > {_dense_threshold:.1f}"
                )
                print(f"*** Switching to dense snapshots: every {_dense_steps} steps\n")
    elif _dense_active:
        _cur_steps = _dense_steps

    _should_save = n % _cur_steps == 0

    if _should_save:
        ti.sync()
        elapsed = time.time() - t_start
        done = n - n_start + 1
        steps_sec = done / elapsed if elapsed > 0 else 0
        ms_step = (elapsed / done * 1000) if done > 0 else 0
        eta_min = (Nt - n) * ms_step / 60000

        if HUBBLE_EXPANSION:
            print(
                f"Step {n}/{Nt} | t={t / mu:.2e} | T={T:.1f} | "
                f"a={a_current:.6f} | H={H_now:.2e} | "
                f"{steps_sec:.1f} steps/s | {ms_step:.2f} ms/step | ETA: {eta_min:.1f} min"
            )
        else:
            print(
                f"Step {n}/{Nt} | t={t / mu:.2e} | T={T:.1f} | "
                f"{steps_sec:.1f} steps/s | {ms_step:.2f} ms/step | ETA: {eta_min:.1f} min"
            )

        # Energy diagnostics
        _ek, _eg, _ep = 0.0, 0.0, 0.0
        if cli_args.diag_energy:
            e_kin_field[None] = 0.0
            e_grad_field[None] = 0.0
            e_pot_field[None] = 0.0
            compute_energy(_inv_dx2_val, float(inv_a2))
            _ek = float(e_kin_field[None])
            _eg = float(e_grad_field[None])
            _ep = float(e_pot_field[None])
            _et = _ek + _eg + _ep
            _ek_expect = mu**2 * T / (2.0 * dx_phys**3)
            _ratio = _ek / _ek_expect if _ek_expect > 0 else 0.0
            print(
                f"  Energy: E_kin={_ek:.4e}  E_grad={_eg:.4e}  "
                f"E_pot={_ep:.4e}  E_tot={_et:.4e}  E_kin/equip={_ratio:.4f}"
            )

        # Save snapshot
        phi_cpu = phi_field.to_numpy()

        # NaN check
        if np.any(np.isnan(phi_cpu)):
            print(f"\nERROR: NaN detected at step {n}! Aborting.")
            break

        save_dict = dict(
            phi=phi_cpu,
            step=n,
            time=t,
            temperature=T,
            phi_min=phi_cpu.min(),
            phi_max=phi_cpu.max(),
        )
        _is_ckpt = _total_saves % CHECKPOINT_EVERY == 0
        _total_saves += 1
        if _is_ckpt:
            save_dict["pi"] = pi_field.to_numpy()
        if HUBBLE_EXPANSION:
            save_dict["scale_factor"] = a_current
            save_dict["hubble"] = H_now
        if cli_args.diag_energy:
            save_dict["E_kin"] = _ek
            save_dict["E_grad"] = _eg
            save_dict["E_pot"] = _ep

        state_file = os.path.join(state_path, f"state_step_{n:010d}.npz")
        np.savez_compressed(state_file, **save_dict)

        # Snapshot figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        _vmax = max(abs(phi_cpu.min()), abs(phi_cpu.max()), 1.0)
        slices_info = [
            (phi_cpu[:, :, Nz // 2], f"z={Nz // 2}"),
            (phi_cpu[:, Ny // 2, :], f"y={Ny // 2}"),
            (phi_cpu[Nx // 2, :, :], f"x={Nx // 2}"),
        ]
        for ax, (sl, label) in zip(axes, slices_info):
            im = ax.imshow(sl, origin="lower", cmap="coolwarm", vmin=-_vmax, vmax=_vmax)
            ax.set_title(label, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Step {n:,} | t={t / mu:.2e} | T={T:.1f} | "
            f"phi: [{phi_cpu.min():.2e}, {phi_cpu.max():.2e}]",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, f"snapshot_step_{n:010d}.png"), dpi=100)
        plt.close(fig)

# =====================================================
# Finish
# =====================================================
t_end = time.time()
total_time = t_end - t_start
actual_steps = Nt - n_start

print(f"\n{'=' * 60}")
print("Simulation finished!")
print(f"Total time: {total_time / 60:.2f} min ({total_time:.1f} s)")
if actual_steps > 0 and total_time > 0:
    print(
        f"Average: {actual_steps / total_time:.1f} steps/s ({total_time * 1000 / actual_steps:.2f} ms/step)"
    )
print(f"Backend: {_arch_name}")
print(f"\nKernel structure ({_intg_name}):")
if USE_FUSED:
    print("  4 fused GPU dispatches per step (2 half-steps x 2 passes each)")
else:
    print("  2 fused GPU dispatches per step (pass1 + pass2)")
print("  Each dispatch: Laplacian + V'(phi) + RK2 update + noise — all in one kernel")
print(f"\nOutput: {save_path}")
print(f"{'=' * 60}")

# Save metadata
np.savez(
    os.path.join(save_path, "simulation_metadata.npz"),
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    dx_phys=dx_phys,
    dt_phys=dt_phys,
    mphi=mphi,
    lam=lam,
    eta_phys=eta_phys,
    T0=T0,
    Nt=Nt,
    steps=steps,
    total_time=total_time,
    mu=mu,
    dx=dx,
    dt=dt,
    eta=eta,
    vev=vev,
    integrator=f"rk2_{cli_args.integrator}_taichi",
    potential_type=cli_args.potential_type,
    seed_bubbles=False,
    counterterm=cli_args.counterterm,
    cooling_rate=cooling_rate,
    bosonCoupling=bosonCoupling,
    fermionCoupling=fermionCoupling,
    bosonGaugeCoupling=bosonGaugeCoupling,
    fermionGaugeCoupling=fermionGaugeCoupling,
    bosonMassSquared=bosonMassSquared,
    n_b=n_b,
    n_f=n_f,
    backend=_arch_name,
)
print("Metadata saved.")
