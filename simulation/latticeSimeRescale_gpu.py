#!/usr/bin/env python
"""
GPU-Accelerated 3D Lattice Simulation — Full 2nd-Order Langevin (RK2 Midpoint)
===============================================================================

Uses PyTorch for GPU acceleration on Apple Silicon (MPS), NVIDIA (CUDA), or CPU.
Implements the same physics as latticeSimeRescale_numba.py (rk2_nonfused_inline).

EOM (rescaled):
    d²φ/dt̃² + η̃ dφ/dt̃ = (1/a²) ∇̃²φ − V'(φ)/μ² + ξ̃

Requirements:
    pip install torch numpy matplotlib scipy

Usage:
    python simulation/latticeSimeRescale_gpu.py --T0 1588 --Nx 256 --Nt 100000
    python simulation/latticeSimeRescale_gpu.py --device mps --T0 1588
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
    print("  pip install scipy")
    sys.exit(1)

# =====================================================
# CLI Arguments
# =====================================================
parser = argparse.ArgumentParser(description="GPU Lattice Langevin simulation (RK2)")
parser.add_argument(
    "--Nx", type=int, default=256, help="Lattice size in x (default: 256)"
)
parser.add_argument(
    "--Ny", type=int, default=None, help="Lattice size in y (default: same as Nx)"
)
parser.add_argument(
    "--Nz", type=int, default=None, help="Lattice size in z (default: same as Nx)"
)
parser.add_argument(
    "--T0", type=float, default=7350.0, help="Initial temperature [GeV] (default: 7350)"
)
parser.add_argument(
    "--Nt", type=int, default=100_000_000, help="Total timesteps (default: 100000000)"
)
parser.add_argument(
    "--steps", type=int, default=100_000, help="Snapshot interval (default: 100000)"
)
parser.add_argument("--dt_factor", type=float, default=1.0, help="dt_phys multiplier")
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "mps", "cuda", "cpu"],
    help="Compute device (default: auto-detect MPS→CUDA→CPU)",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="float32",
    choices=["float32", "float64"],
    help="Field tensor precision (default: float32)",
)
parser.add_argument(
    "--boson_coupling",
    type=float,
    default=None,
    help="Boson Yukawa coupling (default: use value in code)",
)
parser.add_argument(
    "--fermion_coupling",
    type=float,
    default=None,
    help="Fermion Yukawa coupling (default: use value in code)",
)
parser.add_argument(
    "--counterterm",
    action="store_true",
    help="Enable lattice counterterm (subtract double-counted soft-mode thermal contribution from V')",
)
parser.add_argument(
    "--potential_type",
    type=str,
    default="V_correct",
    choices=["V_p", "V_correct", "fermion_only"],
    help="Thermal potential convention: V_p (2*Jb-Jf, default), V_correct (Jb+Jf), fermion_only (Jf only)",
)
parser.add_argument(
    "--nb", type=int, default=20, help="Boson species multiplicity (default: 20)"
)
parser.add_argument(
    "--nf", type=int, default=20, help="Fermion species multiplicity (default: 20)"
)
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
    "--eta",
    type=float,
    default=None,
    help="Damping coefficient η_phys [GeV] (default: 1000)",
)
parser.add_argument(
    "--dx",
    type=float,
    default=None,
    help="Physical lattice spacing dx_phys [GeV^-1] (default: 1e-3)",
)
parser.add_argument(
    "--dt",
    type=float,
    default=None,
    help="Physical time step dt_phys [GeV^-1] (default: 2.5e-4 * dt_factor)",
)
parser.add_argument(
    "--phi_threshold",
    type=float,
    default=None,
    help="When max|phi| exceeds this, switch to dense snapshot interval",
)
parser.add_argument(
    "--steps_dense",
    type=int,
    default=None,
    help="Dense snapshot interval (used when max|phi| > phi_threshold)",
)
parser.add_argument(
    "--boson_mass_squared",
    type=float,
    default=None,
    help="Bare boson mass squared (default: 1e6 GeV^2)",
)
parser.add_argument(
    "--lam",
    type=float,
    default=None,
    help="Quartic coupling λ (default: 1e-16)",
)
parser.add_argument(
    "--mphi",
    type=float,
    default=None,
    help="Flaton mass m_phi [GeV] (default: 1000)",
)
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
# Device Selection
# =====================================================
import torch


def select_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("WARNING: MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


device = select_device(cli_args.device)
# MPS does not support float64 — force float32 on MPS
if device.type == "mps" and cli_args.dtype == "float64":
    print("WARNING: MPS does not support float64. Forcing float32.")
    cli_args.dtype = "float32"
field_dtype = torch.float32 if cli_args.dtype == "float32" else torch.float64
# Precision used for internal accumulation (spline eval, energy diagnostics)
_acc_dtype = torch.float32 if device.type == "mps" else torch.float64
print(f"Device: {device}  |  dtype: {field_dtype}  |  accumulation: {_acc_dtype}")

# =====================================================
# Potential Coefficients
# =====================================================
n_b = cli_args.nb
n_f = cli_args.nf
pot_coeffs = [2.0, -1.0]  # [boson, fermion]
if cli_args.potential_type == "V_correct":
    pot_coeffs = [1.0, 1.0]
elif cli_args.potential_type == "fermion_only":
    pot_coeffs = [0.0, 1.0]
pot_coeffs[0] *= n_b
pot_coeffs[1] *= n_f
print(
    f"Potential: {cli_args.potential_type}  "
    f"(nb={n_b}, nf={n_f}, coeffs: boson={pot_coeffs[0]:g}, fermion={pot_coeffs[1]:g})"
)

# =====================================================
# Thermal dJ Splines (CPU-side, used to build V' on GPU)
# =====================================================
YMAX = 100.0
N_Y = 256
y2_grid = np.linspace(0.0, YMAX, N_Y, dtype=np.float64)
dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid], dtype=np.float64)
dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid], dtype=np.float64)

cs_b = CubicSpline(y2_grid, dJb_grid, bc_type="not-a-knot")
cs_f = CubicSpline(y2_grid, dJf_grid, bc_type="not-a-knot")

# Spline coefficients (nseg intervals, 4 coefficients each)
_spline_x_min = float(y2_grid[0])
_spline_h = float(y2_grid[1] - y2_grid[0])
_spline_nseg = int(y2_grid.size - 1)
_spline_xhi = _spline_x_min + _spline_h * _spline_nseg - 1e-12

# CPU-side spline coefficient arrays (may be modified by counterterm before GPU upload)
_c0_b_np = cs_b.c[0].copy()
_c1_b_np = cs_b.c[1].copy()
_c2_b_np = cs_b.c[2].copy()
_c3_b_np = cs_b.c[3].copy()
_c0_f_np = cs_f.c[0].copy()
_c1_f_np = cs_f.c[1].copy()
_c2_f_np = cs_f.c[2].copy()
_c3_f_np = cs_f.c[3].copy()

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

# Precomputed coupling constants
gb2 = bosonCoupling**2
gg2 = bosonGaugeCoupling**2
gf2 = fermionCoupling**2
gfg2 = fermionGaugeCoupling**2
coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2
mphi_sq = mphi**2
inv_dx2 = 1.0 / (dx * dx)
inv_mu2 = 1.0 / (mu * mu)

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
if cli_args.dt_factor != 1.0 and cli_args.dt is None:
    print(f"  dt_factor = {cli_args.dt_factor}")
print(f"Couplings: boson={bosonCoupling:g}, fermion={fermionCoupling:g}")
print(f"bosonMassSquared = {bosonMassSquared:g}")

# =====================================================
# Lattice counterterm: subtract soft-mode double-counting
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
    print(f"\n[Counterterm] Applying lattice counterterm:")
    print(f"  dx_phys = {dx_phys:g}, T0 = {T0:g}")
    print(f"  UV cutoff: k_max = pi/dx = {math.pi / dx_phys:.1f} GeV")
    print(f"  Dimensionless cutoff: x_cut = k_max/T = {_ct_x_cut:.4f}")

    _t_ct_start = time.time()
    dJb_soft_grid = np.array(
        [_dJ_soft_boson(float(y), _ct_x_cut) for y in y2_grid], dtype=np.float64
    )
    dJf_soft_grid = np.array(
        [_dJ_soft_fermion(float(y), _ct_x_cut) for y in y2_grid], dtype=np.float64
    )

    _frac_b = np.abs(dJb_soft_grid).sum() / max(np.abs(dJb_grid).sum(), 1e-30)
    _frac_f = np.abs(dJf_soft_grid).sum() / max(np.abs(dJf_grid).sum(), 1e-30)
    print(
        f"  Soft-mode fraction: boson={_frac_b:.4f} ({_frac_b * 100:.2f}%), "
        f"fermion={_frac_f:.4f} ({_frac_f * 100:.2f}%)"
    )

    dJb_corrected = dJb_grid - dJb_soft_grid
    dJf_corrected = dJf_grid - dJf_soft_grid

    cs_b_ct = CubicSpline(y2_grid, dJb_corrected, bc_type="not-a-knot")
    cs_f_ct = CubicSpline(y2_grid, dJf_corrected, bc_type="not-a-knot")
    _c0_b_np[:] = cs_b_ct.c[0]
    _c1_b_np[:] = cs_b_ct.c[1]
    _c2_b_np[:] = cs_b_ct.c[2]
    _c3_b_np[:] = cs_b_ct.c[3]
    _c0_f_np[:] = cs_f_ct.c[0]
    _c1_f_np[:] = cs_f_ct.c[1]
    _c2_f_np[:] = cs_f_ct.c[2]
    _c3_f_np[:] = cs_f_ct.c[3]

    _t_ct_end = time.time()
    print(f"  Spline coefficients corrected in {_t_ct_end - _t_ct_start:.2f}s")

# Transfer (possibly corrected) spline coefficients to GPU
_c0_b = torch.tensor(_c0_b_np, dtype=_acc_dtype, device=device)
_c1_b = torch.tensor(_c1_b_np, dtype=_acc_dtype, device=device)
_c2_b = torch.tensor(_c2_b_np, dtype=_acc_dtype, device=device)
_c3_b = torch.tensor(_c3_b_np, dtype=_acc_dtype, device=device)
_c0_f = torch.tensor(_c0_f_np, dtype=_acc_dtype, device=device)
_c1_f = torch.tensor(_c1_f_np, dtype=_acc_dtype, device=device)
_c2_f = torch.tensor(_c2_f_np, dtype=_acc_dtype, device=device)
_c3_f = torch.tensor(_c3_f_np, dtype=_acc_dtype, device=device)

# Hubble
G_STAR = 106.75
M_PL = 2.435e18
DEL_V = 0.25 * lam * vev**4


def hubble_param(T_val):
    chig2 = 30.0 / (math.pi**2 * G_STAR)
    H2 = (T_val**4 / chig2 + DEL_V) / (3.0 * M_PL**2)
    return math.sqrt(H2)


# =====================================================
# GPU Kernels: Laplacian, Vprime, RK2
# =====================================================
# torch.compile fuses elementwise ops into fewer kernel launches.
# On CUDA this uses Triton; on MPS it uses aot_eager (partial fusion).
_compile_backend = "inductor" if device.type == "cuda" else "aot_eager"

# Spline coefficient constants captured by closure for torch.compile
_pot_c0 = float(pot_coeffs[0])
_pot_c1 = float(pot_coeffs[1])
_lam_f = float(lam)
_mphi_sq_f = float(mphi_sq)
_bms_f = float(bosonMassSquared)
_gb2_f = float(gb2)
_gf2_f = float(gf2)
_gg2_f = float(gg2)
_gfg2_f = float(gfg2)
_coef_b_f = float(coef_b)


def _vprime_core(ph, T_val, c0_b, c1_b, c2_b, c3_b, c0_f, c1_f, c2_f, c3_f):
    """V'(phi) body — split out so torch.compile can trace it."""
    T2 = T_val * T_val
    T4 = T2 * T2
    pref = T4 / (2.0 * math.pi ** 2)

    dV = _lam_f * ph * ph * ph - _mphi_sq_f * ph

    xb_sq = _bms_f + 0.5 * _gb2_f * ph * ph + _coef_b_f * T2
    xb = torch.where(
        xb_sq > 0, torch.sqrt(xb_sq.clamp(min=0)) / T_val, torch.zeros_like(ph)
    )

    xf_sq = 0.5 * _gf2_f * ph * ph + (1.0 / 6.0) * _gfg2_f * T2
    xf = torch.where(
        xf_sq > 0, torch.sqrt(xf_sq.clamp(min=0)) / T_val, torch.zeros_like(ph)
    )

    xb_c = xb.clamp(min=_spline_x_min, max=_spline_xhi)
    t_b = (xb_c - _spline_x_min) / _spline_h
    si_b = t_b.long().clamp(min=0, max=_spline_nseg - 1)
    dx_b = xb_c - (_spline_x_min + si_b.to(ph.dtype) * _spline_h)
    dJb = ((c0_b[si_b] * dx_b + c1_b[si_b]) * dx_b + c2_b[si_b]) * dx_b + c3_b[si_b]

    xf_c = xf.clamp(min=_spline_x_min, max=_spline_xhi)
    t_f = (xf_c - _spline_x_min) / _spline_h
    si_f = t_f.long().clamp(min=0, max=_spline_nseg - 1)
    dx_f = xf_c - (_spline_x_min + si_f.to(ph.dtype) * _spline_h)
    dJf = ((c0_f[si_f] * dx_f + c1_f[si_f]) * dx_f + c2_f[si_f]) * dx_f + c3_f[si_f]

    dxb_dphi = torch.where(xb > 1e-20, 0.5 * _gb2_f * ph / (T2 * xb), torch.zeros_like(ph))
    dxf_dphi = torch.where(xf > 1e-20, 0.5 * _gf2_f * ph / (T2 * xf), torch.zeros_like(ph))

    dV = dV + pref * (_pot_c0 * dJb * dxb_dphi + _pot_c1 * dJf * dxf_dphi)
    return dV


def laplacian_3d(phi, inv_dx2):
    """3D periodic Laplacian via torch.roll (7-point stencil)."""
    return (
        torch.roll(phi, -1, 0)
        + torch.roll(phi, 1, 0)
        + torch.roll(phi, -1, 1)
        + torch.roll(phi, 1, 1)
        + torch.roll(phi, -1, 2)
        + torch.roll(phi, 1, 2)
        - 6.0 * phi
    ) * inv_dx2


def _rk2_full(phi, pi, noise, T_val, dt_val, inv_dx2, inv_mu2, eta_eff, inv_a2,
              c0_b, c1_b, c2_b, c3_b, c0_f, c1_f, c2_f, c3_f):
    """Full RK2 midpoint step — compilable as one graph."""
    half_dt = 0.5 * dt_val

    # Pass 1
    lap = laplacian_3d(phi, inv_dx2)
    dV = _vprime_core(phi, T_val, c0_b, c1_b, c2_b, c3_b, c0_f, c1_f, c2_f, c3_f)
    k_pi = inv_a2 * lap - eta_eff * pi - dV * inv_mu2
    phi_mid = phi + half_dt * pi
    pi_mid = pi + half_dt * k_pi

    # Pass 2
    lap_mid = laplacian_3d(phi_mid, inv_dx2)
    dV_mid = _vprime_core(phi_mid, T_val, c0_b, c1_b, c2_b, c3_b, c0_f, c1_f, c2_f, c3_f)
    k_pi_mid = inv_a2 * lap_mid - eta_eff * pi_mid - dV_mid * inv_mu2
    phi_new = phi + dt_val * pi_mid
    pi_new = pi + dt_val * k_pi_mid + noise
    return phi_new, pi_new


try:
    _rk2_compiled = torch.compile(_rk2_full, backend=_compile_backend)
    _USE_COMPILED = True
    print(f"torch.compile: ENABLED (backend={_compile_backend})")
except Exception as _comp_err:
    _rk2_compiled = None
    _USE_COMPILED = False
    print(f"torch.compile: DISABLED ({_comp_err})")


def rk2_step(phi, pi, dt_val, dx_val, eta_eff, T_val, mu_val, inv_a2, noise):
    """Full RK2 midpoint step for the 2nd-order Langevin equation."""
    _inv_dx2 = 1.0 / (dx_val * dx_val)
    _inv_mu2 = 1.0 / (mu_val * mu_val)

    if _USE_COMPILED:
        phi_new, pi_new = _rk2_compiled(
            phi, pi, noise, T_val, dt_val, _inv_dx2, _inv_mu2, eta_eff, inv_a2,
            _c0_b, _c1_b, _c2_b, _c3_b, _c0_f, _c1_f, _c2_f, _c3_f,
        )
        phi.copy_(phi_new)
        pi.copy_(pi_new)
    else:
        half_dt = 0.5 * dt_val
        lap = laplacian_3d(phi, _inv_dx2)
        dV = _vprime_core(phi, T_val, _c0_b, _c1_b, _c2_b, _c3_b,
                          _c0_f, _c1_f, _c2_f, _c3_f)
        k_pi = inv_a2 * lap - eta_eff * pi - dV * _inv_mu2
        phi_mid = phi + half_dt * pi
        pi_mid = pi + half_dt * k_pi

        lap_mid = laplacian_3d(phi_mid, _inv_dx2)
        dV_mid = _vprime_core(phi_mid, T_val, _c0_b, _c1_b, _c2_b, _c3_b,
                              _c0_f, _c1_f, _c2_f, _c3_f)
        k_pi_mid = inv_a2 * lap_mid - eta_eff * pi_mid - dV_mid * _inv_mu2
        phi.add_(dt_val * pi_mid)
        pi.add_(dt_val * k_pi_mid)
        pi.add_(noise)


def compute_energies(phi, pi, dx_val, inv_a2, lam_val, mphi_val, mu_val):
    """Energy densities (kinetic, gradient, tree-level potential) for diagnostics."""
    _inv_dx2 = 1.0 / (dx_val * dx_val)
    _inv_mu2 = 1.0 / (mu_val * mu_val)
    ph = phi.to(_acc_dtype)
    pi_d = pi.to(_acc_dtype)

    e_kin = 0.5 * (pi_d * pi_d).mean().item()

    dx_f = torch.roll(ph, -1, 0) - ph
    dy_f = torch.roll(ph, -1, 1) - ph
    dz_f = torch.roll(ph, -1, 2) - ph
    e_grad = (
        (0.5 * inv_a2 * _inv_dx2 * (dx_f * dx_f + dy_f * dy_f + dz_f * dz_f))
        .mean()
        .item()
    )

    ph2 = ph * ph
    e_pot = (
        ((0.25 * lam_val * ph2 * ph2 - 0.5 * mphi_val**2 * ph2) * _inv_mu2)
        .mean()
        .item()
    )
    return e_kin, e_grad, e_pot


# =====================================================
# Initialize Fields
# =====================================================
phi = torch.randn(Nx, Ny, Nz, dtype=field_dtype, device=device) * 0.01
pi = torch.zeros(Nx, Ny, Nz, dtype=field_dtype, device=device)

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
device_tag = f"_gpu_{device.type}"
save_path = (
    f"data/lattice/{param_set}/{Nx}x{Ny}x{Nz}_T0_{int(T0)}"
    f"{dx_tag}{dtphys_tag}_interval_{steps}_3D{hubble_tag}{eta_tag}"
    f"{coupling_tag}_rk2_inline{ct_tag}{pot_tag}{nf_nb_tag}{device_tag}"
)
os.makedirs(save_path, exist_ok=True)

state_path = os.path.join(save_path, "field_states")
fig_path = os.path.join(save_path, "figs", "latticeSnapshot")
os.makedirs(state_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

CHECKPOINT_EVERY = 5

# =====================================================
# Resume from checkpoint
# =====================================================
import glob

n_start = 0
a_current = 1.0
resuming = False

if cli_args.resume:
    ckpt_files = sorted(glob.glob(os.path.join(state_path, "state_step_*.npz")))
    ckpt_files = [f for f in ckpt_files if "NaN" not in f]
    for ckpt_path in reversed(ckpt_files):
        d = np.load(ckpt_path)
        if "pi" not in d:
            continue
        ckpt_phi = d["phi"]
        if np.any(np.isnan(ckpt_phi)):
            continue
        phi = torch.tensor(ckpt_phi, dtype=field_dtype, device=device)
        pi = torch.tensor(d["pi"], dtype=field_dtype, device=device)
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
print(f"GPU Lattice Simulation — Full 2nd-Order Langevin (RK2)")
print(f"{'=' * 60}")
print(f"Grid: {Nx}x{Ny}x{Nz}  |  Device: {device}  |  dtype: {field_dtype}")
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
# Integrator Selection
# =====================================================
_USE_FUSED_INLINE = cli_args.integrator == "fused_inline"
_intg_name = "rk2_fused_inline (4 passes)" if _USE_FUSED_INLINE else "rk2_nonfused_inline (2 passes)"
print(f"Integrator: {_intg_name}")

# =====================================================
# Main Simulation Loop
# =====================================================
_total_saves = 0
t_start = time.time()

_hubble_inv_chig2 = G_STAR * math.pi**2 / 30.0
_hubble_inv_3mpl2 = 1.0 / (3.0 * M_PL**2)

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

    # Generate FDT noise
    if DISABLE_THERMAL_NOISE:
        ns_val = 0.0
    else:
        ns_val = math.sqrt(2.0 * eta_eff * T * dt / (mu**2 * dx_phys**3))

    if _USE_FUSED_INLINE:
        # Fused RK2: two dt/2 half-steps (4 passes)
        if HUBBLE_EXPANSION:
            T_m = T0 / a_current
        else:
            T_m = max(T0 - cooling_rate * (t + 0.5 * dt), 0.0)
        _half_ns = 0.5 * ns_val
        noise1 = torch.zeros_like(pi) if DISABLE_THERMAL_NOISE else torch.randn_like(pi) * _half_ns
        rk2_step(phi, pi, 0.5 * dt, dx, eta_eff, T, mu, inv_a2, noise1)
        noise2 = torch.zeros_like(pi) if DISABLE_THERMAL_NOISE else torch.randn_like(pi) * _half_ns
        rk2_step(phi, pi, 0.5 * dt, dx, eta_eff, T_m, mu, inv_a2, noise2)
    else:
        # Nonfused RK2: single full dt step (2 passes)
        noise = torch.zeros_like(pi) if DISABLE_THERMAL_NOISE else torch.randn_like(pi) * ns_val
        rk2_step(phi, pi, dt, dx, eta_eff, T, mu, inv_a2, noise)

    # NaN check
    if n % (steps // 2 or 1) == 0 and n > n_start:
        if torch.isnan(phi).any():
            print(f"\nERROR: NaN detected at step {n}! Aborting.")
            break

    # Dense snapshot switching
    _cur_steps = steps
    if _dense_enabled and not _dense_active:
        _phi_absmax = max(abs(float(phi.min())), abs(float(phi.max())))
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

    # Save snapshot
    if _should_save:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

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
        if cli_args.diag_energy:
            _ek, _eg, _ep = compute_energies(phi, pi, dx, inv_a2, lam, mphi, mu)
            _et = _ek + _eg + _ep
            _ek_expect = mu**2 * T / (2.0 * dx_phys**3)
            _ratio = _ek / _ek_expect if _ek_expect > 0 else 0.0
            print(
                f"  Energy: E_kin={_ek:.4e}  E_grad={_eg:.4e}  "
                f"E_pot={_ep:.4e}  E_tot={_et:.4e}  E_kin/equip={_ratio:.4f}"
            )

        # Transfer to CPU and save
        phi_cpu = phi.cpu().numpy()
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
            save_dict["pi"] = pi.cpu().numpy()
        if HUBBLE_EXPANSION:
            save_dict["scale_factor"] = a_current
            save_dict["hubble"] = H_now
        if cli_args.diag_energy:
            save_dict["E_kin"] = _ek
            save_dict["E_grad"] = _eg
            save_dict["E_pot"] = _ep

        state_file = os.path.join(state_path, f"state_step_{n:010d}.npz")
        np.savez_compressed(state_file, **save_dict)

        # Snapshot figure (3 midplane slices)
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
print(f"Simulation finished!")
print(f"Total time: {total_time / 60:.2f} min ({total_time:.1f} s)")
if actual_steps > 0:
    print(
        f"Average: {actual_steps / total_time:.1f} steps/s ({total_time * 1000 / actual_steps:.2f} ms/step)"
    )
print(f"Device: {device}")
print(f"\nOptimizations Used:")
print(f"  Integrator: RK2 midpoint (full 2nd-order Langevin)")
print(f"  V'(phi): inline vectorized spline eval")
print(f"  Laplacian: torch.roll 7-point stencil")
print(f"  Precision: {field_dtype}")
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
    integrator=f"rk2_{cli_args.integrator}_gpu",
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
    device=str(device),
    dtype=cli_args.dtype,
)
print(f"Metadata saved.")
