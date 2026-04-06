import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob
import re
import time
import sys
from datetime import datetime

import argparse
import numba as nb
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
import cosmoTransitions.finiteT as CTFT
from cosmoTransitions.tunneling1D import SingleFieldInstanton

# =====================================================
# CLI arguments (override defaults for Nx, Ny, Nt, T0, steps)
# =====================================================
parser = argparse.ArgumentParser(description="Lattice scalar field simulation")
parser.add_argument(
    "--Nx", type=int, default=256, help="Lattice size in x (default: 256)"
)
parser.add_argument(
    "--Ny", type=int, default=256, help="Lattice size in y (default: 256)"
)
parser.add_argument(
    "--Nz", type=int, default=256, help="Lattice size in z (default: 256)"
)
parser.add_argument(
    "--Nt", type=int, default=100_000_000, help="Total timesteps (default: 100000000)"
)
parser.add_argument(
    "--T0",
    type=float,
    default=7350.0,
    help="Initial temperature in GeV (default: 7350)",
)
parser.add_argument(
    "--steps", type=int, default=100_000, help="Snapshot interval (default: 100000)"
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from the latest checkpoint in the output directory",
)
parser.add_argument(
    "--dt_factor",
    type=float,
    default=1.0,
    help="Multiply dt_phys by this factor (default: 1.0, i.e. use dt_phys as set in code)",
)
parser.add_argument(
    "--replay_dir",
    type=str,
    default=None,
    help="Replay mode: path to an existing simulation output directory. "
    "Loads a checkpoint and re-runs with finer snapshot interval.",
)
parser.add_argument(
    "--start_step",
    type=int,
    default=None,
    help="Replay mode: step number of the checkpoint to start from",
)
parser.add_argument(
    "--replay_steps",
    type=int,
    default=3000,
    help="Replay mode: number of steps to run forward (default: 3000)",
)
parser.add_argument(
    "--replay_save_every",
    type=int,
    default=10,
    help="Replay mode: save snapshot every N steps (default: 10)",
)
parser.add_argument(
    "--integrator",
    type=str,
    default=None,
    choices=[
        "baoab",
        "rk2_nonfused",
        "rk2_fused",
        "rk2_fused_table",
        "rk2_fused_inline",
        "rk2",
        "overdamped",
        "rk2_nonfused_inline",
    ],
    help="Integrator kernel (default: use flags in code). "
    "'overdamped' uses first-order Langevin (no momentum); auto-selected when eta > 100.",
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
    default="V_p",
    choices=["V_p", "V_correct", "fermion_only"],
    help="Thermal potential convention: V_p (2*Jb-Jf, default), V_correct (Jb+Jf), fermion_only (Jf only)",
)
parser.add_argument(
    "--nb",
    type=int,
    default=1,
    help="Number of boson species (thermal multiplicity, default: 1)",
)
parser.add_argument(
    "--nf",
    type=int,
    default=1,
    help="Number of fermion species (thermal multiplicity, default: 1)",
)
cli_args = parser.parse_args()

# =====================================================
# Replay mode detection
# =====================================================
REPLAY_MODE = cli_args.replay_dir is not None
_replay_dx_phys = None
_replay_dt_phys = None
_replay_eta_phys = None
_replay_hubble = None

if REPLAY_MODE:
    if cli_args.start_step is None:
        print("ERROR: --start_step is required in replay mode")
        sys.exit(1)
    _replay_base = os.path.basename(cli_args.replay_dir.rstrip("/"))
    _meta_path = os.path.join(cli_args.replay_dir, "simulation_metadata.npz")
    if os.path.exists(_meta_path):
        _meta = np.load(_meta_path)
        cli_args.Nx = int(_meta["Nx"])
        cli_args.Ny = int(_meta["Ny"])
        cli_args.Nz = int(_meta["Nz"])
        cli_args.T0 = float(_meta["T0"])
        _replay_dx_phys = float(_meta["dx_phys"])
        _replay_dt_phys = float(_meta["dt_phys"])
        _replay_eta_phys = float(_meta["eta_phys"])
        _replay_hubble = True
    else:
        _m = re.match(
            r"(\d+)x(\d+)x(\d+)_T0_(\d+)_dx_([\d.eE+-]+)_dtphys_([\d.eE+-]+)"
            r"_interval_\d+_3D_(hubble|nohubble)_eta_([\d.eE+-]+)",
            _replay_base,
        )
        if _m is None:
            print(
                f"ERROR: Cannot parse simulation parameters from directory name: {_replay_base}"
            )
            print(
                "  Expected format: NxNxN_T0_XXXX_dx_X_dtphys_X_interval_X_3D_hubble_eta_X"
            )
            sys.exit(1)
        cli_args.Nx = int(_m.group(1))
        cli_args.Ny = int(_m.group(2))
        cli_args.Nz = int(_m.group(3))
        cli_args.T0 = float(_m.group(4))
        _replay_dx_phys = float(_m.group(5))
        _replay_dt_phys = float(_m.group(6))
        _replay_hubble = _m.group(7) == "hubble"
        _replay_eta_phys = float(_m.group(8))

# =====================================================
# Thermal potential coefficient array (Numba reads at runtime from global arrays)
# =====================================================
# _POT_COEFFS[0] = boson coefficient, _POT_COEFFS[1] = fermion coefficient
# V_p:          2*dJb - dJf  →  [2.0, -1.0]
# V_correct:    dJb + dJf    →  [1.0,  1.0]
# fermion_only: dJf only     →  [0.0,  1.0]
# Multiplied by nb / nf (species multiplicities)
n_b = cli_args.nb
n_f = cli_args.nf
_POT_COEFFS = np.array([2.0, -1.0], dtype=np.float64)
if cli_args.potential_type == "V_correct":
    _POT_COEFFS[0] = 1.0
    _POT_COEFFS[1] = 1.0
elif cli_args.potential_type == "fermion_only":
    _POT_COEFFS[0] = 0.0
    _POT_COEFFS[1] = 1.0
_POT_COEFFS[0] *= n_b
_POT_COEFFS[1] *= n_f

# =====================================================
# Logging Setup
# =====================================================


class Logger:
    """Simple logger that writes to both console and file."""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Create log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
sys.stdout = Logger(log_file)

print("=" * 70)
print("LATTICE SIMULATION LOG")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print("=" * 70)

# Print threading info
print("\nNumba Configuration:")
print(f"  Version: {nb.__version__}")
print(f"  Threading layer: {nb.config.THREADING_LAYER}")
print(f"  Configured threads: {nb.config.NUMBA_NUM_THREADS}")
print(f"  Active threads: {nb.get_num_threads()}")
print(
    f"\nPotential type: {cli_args.potential_type}  "
    f"(nb={n_b}, nf={n_f}, coeffs: boson={_POT_COEFFS[0]:g}, fermion={_POT_COEFFS[1]:g})"
)
print("-" * 70)

# =====================================================
# Performance Settings
# =====================================================
# For further speedup, set NUMBA_NUM_THREADS=<core_count> before running
# See SPEED_OPTIMIZATIONS.md for detailed tuning guide
USE_FUSED_RK2 = False  # Fused RK2: two dt/2 half-steps in one kernel (4 passes)
USE_SINGLE_PASS_RK2 = (
    False  # Fully fused: lap+Vprime inline per site (4 passes, fastest fused)
)
USE_NONFUSED_TABLE_RK2 = (
    True  # Non-fused RK2: single full dt step with table V' (2 passes)
)
USE_NONFUSED_INLINE_RK2 = (
    False  # Non-fused RK2: single full dt step, inline V' (2 passes)
)
USE_VPRIME_TABLE = True  # Pre-tabulate V'(phi) for fast lookup (~2-4x speedup)
VPRIME_TABLE_SIZE = 16384  # Number of table entries (power of 2 for alignment)
USE_BAOAB = True  # BAOAB Langevin integrator: 2 passes instead of 4 (~2x faster)
USE_OVERDAMPED = False  # Overdamped (first-order) Langevin: no momentum, Heun method

# Override kernel flags from CLI --integrator
if cli_args.integrator is not None:
    USE_BAOAB = False
    USE_FUSED_RK2 = False
    USE_SINGLE_PASS_RK2 = False
    USE_NONFUSED_TABLE_RK2 = False
    USE_NONFUSED_INLINE_RK2 = False
    USE_OVERDAMPED = False
    _intg = cli_args.integrator
    if _intg == "baoab":
        USE_BAOAB = True
        USE_VPRIME_TABLE = True
    elif _intg == "rk2_nonfused":
        USE_NONFUSED_TABLE_RK2 = True
        USE_VPRIME_TABLE = True
    elif _intg == "rk2_nonfused_inline":
        USE_NONFUSED_INLINE_RK2 = True
        USE_VPRIME_TABLE = False
    elif _intg == "rk2_fused_table":
        USE_SINGLE_PASS_RK2 = True
        USE_VPRIME_TABLE = True
    elif _intg == "rk2_fused_inline":
        USE_SINGLE_PASS_RK2 = True
        USE_VPRIME_TABLE = False
    elif _intg == "rk2_fused":
        USE_FUSED_RK2 = True
    elif _intg == "rk2":
        pass  # all flags already False
    elif _intg == "overdamped":
        USE_OVERDAMPED = True
        USE_VPRIME_TABLE = True

# Auto-select overdamped integrator when eta_phys > 100 (unless CLI overrode)
# Reading eta_phys early here; it will be set properly in Physical Parameters section.
# This block is re-checked after eta_phys is finalized (see below).
_OVERDAMPED_AUTO_THRESHOLD = 100.0

USE_NUMBA_RNG = True  # In-kernel RNG (eliminates Python overhead)
USE_FLOAT32 = True  # ~1.5-2× faster (memory); safe per check

# =====================================================
# Bubble Seeding Options
# =====================================================
# Set SEED_BUBBLES = True to place pre-formed true vacuum bubbles
# This is useful for studying bubble expansion/collision dynamics
# without waiting for rare stochastic nucleation events
SEED_BUBBLES = False  # Set to True to enable bubble seeding

# Bubble configuration: list of (center_x, center_y, center_z, radius, sign)
# - center_x, center_y, center_z: position in lattice units
# - radius: bubble radius in lattice units
# - sign: +1 for positive VEV, -1 for negative VEV
BUBBLE_CONFIG = [(128, 128, 128, 20, 1)]  # Single bubble at center

# Bubble wall profile:
# - 'sharp': step function (instant transition)
# - 'tanh': smooth tanh profile (more physical, mimics kink solution)
# - 'bounce': use CosmoTransitions bounce solution (most physical)
# - wall_width controls the thickness for 'tanh' profile
BUBBLE_PROFILE = "tanh"  # 'sharp', 'tanh', or 'bounce'
BUBBLE_WALL_WIDTH = 5.0  # Wall thickness in lattice units (for 'tanh' only)

# For 'bounce' profile: scale the bounce to this radius in lattice units
# The physical bounce radius is typically very small, so we scale it up
# to span a reasonable number of lattice cells for visualization
BOUNCE_TARGET_RADIUS = 40.0  # Target bubble radius in lattice units

# Settling period: run with high damping to reduce initial oscillations
# This lets the bubble relax to a more physical profile before simulation.
# NOTE: Settling uses the same RK2 integrator with tiny dt = 1e-5 and is
# ~25,000x slower than gradient flow relaxation (see GRADIENT_FLOW_RELAX).
# With eta_rescaled = 0.05, the damping timescale is 1/eta = 20, requiring
# ~2,000,000 steps -- far more than the 10,000 configured here.
# Prefer GRADIENT_FLOW_RELAX = True instead.
SETTLING_ENABLED = False  # DISABLED - use gradient flow relaxation instead
SETTLING_STEPS = 10000  # Number of settling steps
SETTLING_ETA = 5.0  # High damping during settling (normal eta is ~0.3)

# Disable thermal noise for clean bubble dynamics (testing)
# When True, bubble evolves deterministically without thermal kicks
DISABLE_THERMAL_NOISE = False  # Set to True to disable noise

# Checkpoint interval: full state (phi+pi) saved every N-th snapshot.
# Other snapshots save phi only (sufficient for visualization).
# Set to 1 to save full state every snapshot (old behavior).
CHECKPOINT_EVERY = 10

# Overdamped mode: use high damping to suppress wall oscillations
# In this mode, the field follows gradient descent (no ringing).
# NOTE: Even in overdamped mode, the RK2 integrator is limited by the tiny
# simulation timestep dt = 1e-5, so convergence is still very slow.
# Gradient flow relaxation (GRADIENT_FLOW_RELAX) uses dt_gf ~ 0.1 and
# first-order dynamics (no momentum), making it ~25,000x faster.
OVERDAMPED_MODE = False  # Set to True for smooth bubble expansion
OVERDAMPED_ETA = 50.0  # VERY high damping for smooth dynamics

# Gradient flow relaxation: first-order relaxation of bubble profile
# This eliminates initial force imbalance (ring artifacts) by solving
#   phi_{n+1} = phi_n + dt_gf * (laplacian(phi_n) - V'(phi_n)/mu^2)
# until convergence. This is MUCH faster than overdamped settling because:
#   1. No momentum -> no oscillations, strictly monotone relaxation
#   2. CFL allows dt_gf ~ 0.1 vs dt = 1e-5 (25,000x larger steps)
GRADIENT_FLOW_RELAX = True  # Enable gradient flow before simulation
GF_DT = 0.1  # Gradient flow timestep (CFL limit: dx^2/(2*d) ≈ 0.167 for d=3)
GF_MAX_ITER = 50000  # Maximum relaxation iterations
GF_TOL = 1e-6  # Convergence: max|delta_phi| / max|phi| < tol
GF_PRINT_EVERY = 1000  # Print progress every N iterations
GF_SAVE_EVERY = 500  # Save diagnostic snapshot every N iterations (0 to disable)
GF_WALL_MARGIN = 2.0  # Relax only within margin * wall_width of the bubble wall

# Hubble expansion: include FRW dynamics in the Langevin equation
# When enabled, the EOM becomes:
#   phi_tt = (1/a^2) * laplacian(phi) - (eta + 3H) * phi_t - V'(phi)/mu^2
# and temperature tracks the scale factor: T = T0 * a0 / a(t)
HUBBLE_EXPANSION = True
G_STAR = 106.75  # SM relativistic degrees of freedom
M_PL = 2.4e18  # Reduced Planck mass (GeV)
DEL_V = 1e28  # Vacuum energy V0 for inflation background (GeV^4)

# =====================================================
# Thermal dJ tables (build once), evaluate via Numba cubic on uniform grid
# =====================================================
YMAX = 100.0
N_Y = 256  # Lower to 256 or 128 for speed; check comparison plots for accuracy
y2_grid = np.linspace(0.0, YMAX, N_Y, dtype=np.float64)
dJb_grid = np.array([CTFT.dJb_exact(y2) for y2 in y2_grid], dtype=np.float64)
dJf_grid = np.array([CTFT.dJf_exact(y2) for y2 in y2_grid], dtype=np.float64)

# Build cubic splines and extract piecewise polynomial coefficients
cs_b = CubicSpline(y2_grid, dJb_grid, bc_type="not-a-knot")
cs_f = CubicSpline(y2_grid, dJf_grid, bc_type="not-a-knot")
# cs.c has shape (4, n-1): coefficients for powers [3,2,1,0]
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
inv_hy = 1.0 / h_y
xhi_clamp = x_min + h_y * (y2_grid.size - 1) - 1e-12
nseg = int(y2_grid.size - 1)

# =====================================================
# Lattice counterterm: subtract double-counted soft-mode contribution
# =====================================================
if cli_args.counterterm:
    from scipy.integrate import quad as _ct_quad

    def _dJ_soft_boson(y, x_cut):
        """Soft-mode (k < k_max) contribution to dJ_+/dy (boson)."""
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
        """Soft-mode (k < k_max) contribution to dJ_-/dy (fermion)."""
        if x_cut < 1e-10 or y < 1e-30:
            return 0.0

        def _integrand(x):
            z = math.sqrt(x * x + y * y)
            if z > 50.0:
                return 0.0
            return x * x * y / (z * (math.exp(z) + 1.0))

        val, _ = _ct_quad(_integrand, 0, x_cut, limit=200)
        return val / (2.0 * math.pi**2)

    _CT_HELPERS_READY = True
    print(
        "[Counterterm] Helpers defined; splines will be corrected after dx_phys is set."
    )
else:
    _CT_HELPERS_READY = False

# =====================================================
# Comparison plots (put first)
# =====================================================


def _dJ_uniform_array(x_arr, c0, c1, c2, c3):
    out = np.empty_like(x_arr, dtype=np.float64)
    xlo = x_min
    xhi = x_min + h_y * nseg
    for i in range(x_arr.size):
        x = x_arr[i]
        if x < xlo:
            x = xlo
        elif x > xhi:
            x = xhi - 1e-12
        out[i] = cubic_eval_uniform(x, x_min, h_y, nseg, c0, c1, c2, c3)
    return out


def _Vprime_1d(phi_arr, T, use_uniform):
    gb2 = bosonCoupling * bosonCoupling
    gg2 = bosonGaugeCoupling * bosonGaugeCoupling
    gf2 = fermionCoupling * fermionCoupling
    gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
    coef_b_T = 0.25 * gb2 + (2.0 / 3.0) * gg2
    T2 = T * T
    T4 = T2 * T2
    pref = T4 / (2.0 * math.pi * math.pi)
    out = np.empty_like(phi_arr, dtype=np.float64)
    for i in range(phi_arr.size):
        ph = float(phi_arr[i])
        dV = lam * ph * ph * ph - mphi * mphi * ph
        xb_sq = bosonMassSquared + 0.5 * gb2 * ph * ph + coef_b_T * T2
        xf_sq = 0.5 * gf2 * ph * ph + (1.0 / 6.0) * gfg2 * T2
        xb = 0.0
        xf = 0.0
        if xb_sq > 0.0:
            xb = math.sqrt(xb_sq) / T
        if xf_sq > 0.0:
            xf = math.sqrt(xf_sq) / T
        if use_uniform:
            dJb = _dJ_uniform_array(np.array([xb]), c0_b, c1_b, c2_b, c3_b)[0]
            dJf = _dJ_uniform_array(np.array([xf]), c0_f, c1_f, c2_f, c3_f)[0]
        else:
            dJb = CTFT.dJb_exact(xb)
            dJf = CTFT.dJf_exact(xf)
        dxb_dphi = 0.5 * gb2 * ph / (T2 * max(xb, 1e-20))
        dxf_dphi = 0.5 * gf2 * ph / (T2 * max(xf, 1e-20))
        dV += pref * (_POT_COEFFS[0] * dJb * dxb_dphi + _POT_COEFFS[1] * dJf * dxf_dphi)
        out[i] = dV
    return out


def comparison_plots():
    comp_dir = "figs/latticeSim_rescaled_numba/comparison"
    os.makedirs(comp_dir, exist_ok=True)
    # dJ comparison over x in [0, YMAX]
    xs = np.linspace(0.0, YMAX, 400)
    dJb_ref = np.array([CTFT.dJb_exact(x) for x in xs])
    dJf_ref = np.array([CTFT.dJf_exact(x) for x in xs])
    dJb_new = _dJ_uniform_array(xs, c0_b, c1_b, c2_b, c3_b)
    dJf_new = _dJ_uniform_array(xs, c0_f, c1_f, c2_f, c3_f)
    eps = 1e-12
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xs, dJb_ref, label="dJb exact")
    plt.plot(xs, dJb_new, "--", label="dJb uniform-cubic")
    plt.xlabel("x")
    plt.ylabel("dJb")
    plt.title("Bosonic dJ")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(xs, np.abs(dJb_new - dJb_ref), label="abs err")
    plt.plot(
        xs,
        np.abs(dJb_new - dJb_ref) / (np.abs(dJb_ref) + eps),
        label="rel err",
    )
    plt.xlabel("x")
    plt.title("dJb error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/dJb_compare.png", dpi=200)
    plt.clf()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xs, dJf_ref, label="dJf exact")
    plt.plot(xs, dJf_new, "--", label="dJf uniform-cubic")
    plt.xlabel("x")
    plt.ylabel("dJf")
    plt.title("Fermionic dJ")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(xs, np.abs(dJf_new - dJf_ref), label="abs err")
    plt.plot(
        xs,
        np.abs(dJf_new - dJf_ref) / (np.abs(dJf_ref) + eps),
        label="rel err",
    )
    plt.xlabel("x")
    plt.title("dJf error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/dJf_compare.png", dpi=200)
    plt.clf()
    # V' comparison at representative T and phi range
    T_sample = float(T0)
    # choose phi range around 0 with width related to thermal mass
    phi_grid = np.linspace(-1.0, 1.0, 400) * 1e3
    Vp_ref = _Vprime_1d(phi_grid, T_sample, use_uniform=False)
    Vp_new = _Vprime_1d(phi_grid, T_sample, use_uniform=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(phi_grid, Vp_ref, label="V' exact dJ")
    plt.plot(phi_grid, Vp_new, "--", label="V' uniform-cubic")
    plt.xlabel("phi")
    plt.ylabel("V'(phi, T)")
    plt.title(f"V' at T={T_sample:.1f}")
    plt.legend()
    plt.subplot(1, 2, 2)
    diff = Vp_new - Vp_ref
    rel = np.abs(diff) / (np.maximum(np.abs(Vp_ref), eps))
    plt.plot(phi_grid, np.abs(diff), label="abs err")
    plt.plot(phi_grid, rel, label="rel err")
    plt.xlabel("phi")
    plt.title("V' error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comp_dir}/Vprime_compare.png", dpi=200)
    plt.clf()


# =====================================================
# Physical parameters
# =====================================================
Nx, Ny, Nz = cli_args.Nx, cli_args.Ny, cli_args.Nz

dx_phys = 1 * 1e-3
dt_phys = 2.5 * 1e-4 * cli_args.dt_factor
# dt_phys = 5 * 1e-5 * cli_args.dt_factor
Nt = cli_args.Nt

lam = 1e-16
mphi = 1_000.0

T0 = cli_args.T0
eta_phys = 1000
cooling_rate = 1.0

# Override physical parameters in replay mode
if REPLAY_MODE:
    if _replay_dx_phys is not None:
        dx_phys = _replay_dx_phys
    if _replay_dt_phys is not None:
        dt_phys = _replay_dt_phys
    if _replay_eta_phys is not None:
        eta_phys = _replay_eta_phys
    Nt = cli_args.start_step + cli_args.replay_steps

# Auto-select overdamped integrator when eta_phys exceeds threshold
if (
    cli_args.integrator is None
    and not USE_OVERDAMPED
    and eta_phys > _OVERDAMPED_AUTO_THRESHOLD
):
    USE_OVERDAMPED = True
    USE_BAOAB = False
    USE_FUSED_RK2 = False
    USE_SINGLE_PASS_RK2 = False
    USE_NONFUSED_TABLE_RK2 = False
    USE_NONFUSED_INLINE_RK2 = False
    USE_VPRIME_TABLE = True
    print(
        f"\n*** eta_phys = {eta_phys} > {_OVERDAMPED_AUTO_THRESHOLD}: "
        f"auto-selecting OVERDAMPED (first-order Langevin) integrator ***\n"
    )

# =====================================================
# Rescaling
# =====================================================
# The lattice uses COMOVING coordinates: dx is the comoving spacing (fixed).
# Physical distance = a(t) * dx_comoving. The Laplacian is computed in
# comoving coords, so the EOM gets a 1/a^2 prefactor (via inv_a2).
mu = mphi

dx = mu * dx_phys  # comoving lattice spacing (rescaled)
dt = mu * dt_phys

# Stability check
if USE_OVERDAMPED:
    # Diffusion-type CFL: dt < eta_eff * dx^2 / (2*d), d=3
    _eta_resc = eta_phys / mu
    dt_cfl_diff = _eta_resc * dx * dx / 6.0
    cfl_ratio = dt / dt_cfl_diff
    print(f"\nTime step: dt_phys = {dt_phys:.2e}, dt (rescaled) = {dt:.4e}")
    print(f"  Overdamped CFL limit (diffusive): dt_CFL = {dt_cfl_diff:.4f} (rescaled)")
    print(f"  dt/dt_CFL = {cfl_ratio:.4e} ({'SAFE' if cfl_ratio < 1.0 else 'UNSAFE!'})")
    if cfl_ratio >= 1.0:
        print("  WARNING: dt exceeds diffusive CFL limit! Reduce dt_phys.")
else:
    dt_cfl = dx / np.sqrt(3.0)
    cfl_ratio = dt / dt_cfl
    print(f"\nTime step: dt_phys = {dt_phys:.2e}, dt (rescaled) = {dt:.4e}")
    print(f"  CFL limit: dt_CFL = {dt_cfl:.4f} (rescaled)")
    print(f"  dt/dt_CFL = {cfl_ratio:.4e} ({'SAFE' if cfl_ratio < 1.0 else 'UNSAFE!'})")
    if cfl_ratio >= 1.0:
        print("  WARNING: dt exceeds CFL limit! Simulation may be unstable.")
        print("  Reduce --dt_factor or dt_phys.")
if cli_args.dt_factor != 1.0:
    print(f"  dt_factor = {cli_args.dt_factor}")

# Use overdamped eta if enabled
if OVERDAMPED_MODE:
    eta = OVERDAMPED_ETA / mu
    print(f"OVERDAMPED MODE: η = {OVERDAMPED_ETA} (rescaled: {eta:.2e})")
else:
    eta = eta_phys / mu
cooling_rate = cooling_rate / mu


# =====================================================
# Hubble expansion setup
# =====================================================
def hubble_param(T_val):
    """H(T) in GeV, including radiation + vacuum energy (inflation -> radiation era)."""
    chig2 = 30.0 / (np.pi**2 * G_STAR)
    H2 = (T_val**4 / chig2 + DEL_V) / (3.0 * M_PL**2)
    return np.sqrt(H2)


a_current = 1.0  # Initial scale factor at T = T0

# Precompute Hubble constants for fast inline evaluation in main loop
_hubble_inv_chig2 = G_STAR * np.pi**2 / 30.0
_hubble_inv_3mpl2 = 1.0 / (3.0 * M_PL**2)

if HUBBLE_EXPANSION:
    H0 = hubble_param(T0)
    print(f"\nHubble expansion: ENABLED (comoving lattice)")
    print(f"  g_* = {G_STAR}, M_Pl = {M_PL:.2e} GeV, ΔV = {DEL_V:.2e} GeV^4")
    print(f"  H² = (T⁴/χ_g² + ΔV) / (3 M_Pl²)")
    print(f"  H(T0={T0}) = {H0:.4e} GeV")
    print(f"  3H = {3*H0:.4e} GeV")
    print(f"  η_phys = {eta_phys} GeV  →  3H/η = {3*H0/eta_phys:.2e}")
    print(f"  T(t) = T0 / a(t)  (entropy conservation)")
    print(f"  Initial a = {a_current}")
else:
    print(f"\nHubble expansion: DISABLED (using linear cooling)")

# =====================================================
# Fields
# =====================================================
# Choose precision (float32 for speed, float64 for accuracy)
field_dtype = np.float32 if USE_FLOAT32 else np.float64
print(f"Field precision: {field_dtype.__name__}")

# Calculate tree-level VEV
vev = np.sqrt(mphi**2 / lam)
print(f"Tree-level VEV: ±{vev:.4e}")

# Potential parameters (needed for thermal VEV calculation)
bosonMassSquared = 1_000_000.0
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

if cli_args.boson_coupling is not None:
    bosonCoupling = cli_args.boson_coupling
if cli_args.fermion_coupling is not None:
    fermionCoupling = cli_args.fermion_coupling
print(f"Couplings: boson={bosonCoupling:g}, fermion={fermionCoupling:g}")

# =====================================================
# Apply lattice counterterm (subtract soft-mode double-counting)
# =====================================================
if _CT_HELPERS_READY:
    _ct_x_cut = math.pi / (dx_phys * T0)
    print(f"\n[Counterterm] Applying lattice counterterm:")
    print(f"  dx_phys = {dx_phys:g}, T0 = {T0:g}")
    print(f"  UV cutoff: k_max = pi/dx = {math.pi/dx_phys:.1f} GeV")
    print(f"  Dimensionless cutoff: x_cut = k_max/T = {_ct_x_cut:.4f}")

    _t_ct_start = time.time()
    dJb_soft_grid = np.array(
        [_dJ_soft_boson(float(y), _ct_x_cut) for y in y2_grid],
        dtype=np.float64,
    )
    dJf_soft_grid = np.array(
        [_dJ_soft_fermion(float(y), _ct_x_cut) for y in y2_grid],
        dtype=np.float64,
    )

    _frac_b = np.abs(dJb_soft_grid).sum() / max(np.abs(dJb_grid).sum(), 1e-30)
    _frac_f = np.abs(dJf_soft_grid).sum() / max(np.abs(dJf_grid).sum(), 1e-30)
    print(
        f"  Soft-mode fraction: boson={_frac_b:.4f} ({_frac_b*100:.2f}%), "
        f"fermion={_frac_f:.4f} ({_frac_f*100:.2f}%)"
    )

    dJb_corrected = dJb_grid - dJb_soft_grid
    dJf_corrected = dJf_grid - dJf_soft_grid

    cs_b_ct = CubicSpline(y2_grid, dJb_corrected, bc_type="not-a-knot")
    cs_f_ct = CubicSpline(y2_grid, dJf_corrected, bc_type="not-a-knot")
    # In-place update so Numba cached kernels see the corrected data
    c0_b[:] = cs_b_ct.c[0]
    c1_b[:] = cs_b_ct.c[1]
    c2_b[:] = cs_b_ct.c[2]
    c3_b[:] = cs_b_ct.c[3]
    c0_f[:] = cs_f_ct.c[0]
    c1_f[:] = cs_f_ct.c[1]
    c2_f[:] = cs_f_ct.c[2]
    c3_f[:] = cs_f_ct.c[3]

    _t_ct_end = time.time()
    print(f"  Spline coefficients replaced in {_t_ct_end - _t_ct_start:.2f}s")
    print(f"  [No runtime overhead: kernel code unchanged, only spline values differ]")

# Calculate THERMAL VEV at T0


def Vprime_scalar(phi_val, T_val):
    """Compute V'(φ) at given φ and T (for finding thermal VEV)."""
    dV = lam * phi_val**3 - mphi**2 * phi_val
    # Thermal boson
    gb2 = bosonCoupling**2
    gg2 = bosonGaugeCoupling**2
    coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2
    xb_sq = bosonMassSquared + 0.5 * gb2 * phi_val**2 + coef_b * T_val**2
    if xb_sq > 0:
        xb = np.sqrt(xb_sq) / T_val
        xb_c = min(max(xb, x_min), x_min + h_y * nseg - 1e-12)
        idx = int((xb_c - x_min) / h_y)
        idx = max(0, min(idx, nseg - 1))
        dx_s = xb_c - (x_min + idx * h_y)
        dJb_val = ((c0_b[idx] * dx_s + c1_b[idx]) * dx_s + c2_b[idx]) * dx_s + c3_b[idx]
        dxb_dp = 0.5 * gb2 * phi_val / (T_val**2 * max(xb, 1e-20))
        pref = T_val**4 / (2.0 * np.pi**2)
        dV += pref * _POT_COEFFS[0] * dJb_val * dxb_dp
    # Thermal fermion
    gf2 = fermionCoupling**2
    gfg2 = fermionGaugeCoupling**2
    xf_sq = 0.5 * gf2 * phi_val**2 + (1.0 / 6.0) * gfg2 * T_val**2
    if xf_sq > 0:
        xf = np.sqrt(xf_sq) / T_val
        xf_c = min(max(xf, x_min), x_min + h_y * nseg - 1e-12)
        idx = int((xf_c - x_min) / h_y)
        idx = max(0, min(idx, nseg - 1))
        dx_s = xf_c - (x_min + idx * h_y)
        dJf_val = ((c0_f[idx] * dx_s + c1_f[idx]) * dx_s + c2_f[idx]) * dx_s + c3_f[idx]
        dxf_dp = 0.5 * gf2 * phi_val / (T_val**2 * max(xf, 1e-20))
        pref = T_val**4 / (2.0 * np.pi**2)
        dV += pref * _POT_COEFFS[1] * dJf_val * dxf_dp
    return dV


def build_vprime_table(T_val, phi_min, phi_max, n_table):
    """Build a dense 1D lookup table of V'(phi) at fixed temperature T_val.

    Returns (table, phi_min, dphi_inv) where table[i] = V'(phi_min + i*dphi).
    """
    phi_arr = np.linspace(phi_min, phi_max, n_table, dtype=np.float64)
    table = np.empty(n_table, dtype=np.float64)
    for i in range(n_table):
        table[i] = Vprime_scalar(phi_arr[i], T_val)
    return table, float(phi_min), 1.0 / ((phi_max - phi_min) / (n_table - 1))


print(f"\nFinding thermal VEV at T = {T0}...")
print(f"  V'(0) = {Vprime_scalar(1e-10, T0):.4e}")
print(f"  V'(VEV) = {Vprime_scalar(vev, T0):.4e}")

try:
    # Find where V'(φ) = 0 for φ > 0
    vev_thermal = brentq(lambda p: Vprime_scalar(p, T0), vev * 0.001, vev * 2.0)
    print(f"  Thermal VEV found: {vev_thermal:.4e}")
    print(f"  Ratio thermal/tree: {vev_thermal/vev:.4f}")
except Exception as e:
    print(f"  Could not find thermal VEV: {e}")
    vev_thermal = vev

# Store tree-level VEV for reference
vev_tree = vev  # vev was set to tree-level above

# Find the ESCAPE POINT (where field will roll toward true vacuum)
print(f"\nFinding escape point...")

# The escape point is slightly past the barrier top, where V' < 0
# so the field naturally rolls toward the true vacuum
try:
    # Search for where V' changes from positive to negative
    # Start from very small phi (not 0.001*VEV which is too large!)
    # Use logarithmic spacing to capture barrier at any scale
    test_points_log = np.logspace(0, np.log10(vev_thermal * 0.99), 500)
    test_points = np.concatenate([[1e-10], test_points_log])  # Include near-zero
    Vp_values = [Vprime_scalar(p, T0) for p in test_points]

    # Debug: show V' at different scales
    print(f"  V'(1e-10) = {Vprime_scalar(1e-10, T0):.4e}")
    print(f"  V'(1e0)   = {Vprime_scalar(1e0, T0):.4e}")
    print(f"  V'(1e5)   = {Vprime_scalar(1e5, T0):.4e}")
    print(f"  V'(1e8)   = {Vprime_scalar(1e8, T0):.4e}")
    print(f"  V'(1e10)  = {Vprime_scalar(1e10, T0):.4e}")

    # Find where V' transitions from + to - (barrier top)
    barrier_idx = None
    for i in range(len(Vp_values) - 1):
        if Vp_values[i] > 0 and Vp_values[i + 1] < 0:
            barrier_idx = i
            print(
                f"  Barrier found between {test_points[i]:.4e} and "
                f"{test_points[i+1]:.4e}"
            )
            break

    if barrier_idx is not None:
        # Barrier found - escape point is slightly past it
        barrier_top = brentq(
            lambda p: Vprime_scalar(p, T0),
            test_points[barrier_idx],
            test_points[barrier_idx + 1],
        )
        # Use a point past the barrier where V' < 0
        escape_point = barrier_top * 1.1  # Slightly past barrier
        print(f"  Barrier top at: {barrier_top:.4e}")
        print(f"  Escape point: {escape_point:.4e}")
        print(f"  V'(escape) = {Vprime_scalar(escape_point, T0):.4e} (should be < 0)")
    else:
        # No barrier found - check if V' is always negative or always positive
        if all(v < 0 for v in Vp_values):
            # V' < 0 everywhere: no metastable false vacuum
            # Use thermal VEV directly
            escape_point = vev_thermal
            print(f"  No barrier: V' < 0 everywhere (spinodal regime)")
            print(f"  Using thermal VEV: {escape_point:.4e}")
        else:
            # V' > 0 somewhere: barrier exists but wasn't found
            # Use a moderate fraction
            escape_point = vev_thermal * 0.3
            print(f"  No barrier crossing found in search range")
            print(f"  Using {escape_point:.4e} as starting point")
        print(f"  V'(escape) = {Vprime_scalar(escape_point, T0):.4e}")

except Exception as e:
    print(f"  Error finding escape point: {e}")
    escape_point = vev_thermal * 0.3

# Choose seeding value based on profile type
# For 'bounce': use escape point (field will roll)
# For 'tanh'/'sharp': use tree-level VEV directly (already at true vacuum)
if BUBBLE_PROFILE == "bounce":
    vev = escape_point
    print(f"\nUsing ESCAPE POINT for seeding: {vev:.4e}")
    print("The field will dynamically roll toward the true vacuum!")
else:
    # Use tree-level VEV for tanh/sharp profiles
    # This seeds the bubble directly at the true vacuum
    vev = vev_tree  # vev_tree = sqrt(mphi^2 / lam) ~ 1e11
    print(f"\nUsing TREE-LEVEL VEV for seeding: {vev:.4e}")
    print("The bubble is seeded at the true vacuum (no rolling needed).")

# =====================================================
# Compute CosmoTransitions Bounce Profile (if using 'bounce' profile)
# =====================================================
bounce_profile_interp = None  # Will hold interpolator if computed
bounce_R_scale = 1.0  # Will be set after computing bounce

if BUBBLE_PROFILE == "bounce":
    print("\n" + "=" * 60)
    print("Computing CosmoTransitions bounce profile...")
    print("=" * 60)

    try:
        # Define potential for CosmoTransitions
        # V(phi) = tree-level + thermal corrections at T0
        def V_for_bounce(phi_val):
            """
            Full potential V(phi, T0) for CosmoTransitions.
            Uses the same physics as the lattice simulation.
            """
            phi_val = float(phi_val)
            # Tree-level potential
            V_tree = -0.5 * mphi**2 * phi_val**2 + 0.25 * lam * phi_val**4

            # Thermal corrections using same formulas as Vprime_scalar
            T_val = T0
            gb2 = bosonCoupling**2
            gg2 = bosonGaugeCoupling**2
            coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2
            gf2 = fermionCoupling**2
            gfg2 = fermionGaugeCoupling**2

            # Boson mass squared
            mb2 = bosonMassSquared + 0.5 * gb2 * phi_val**2 + coef_b * T_val**2
            # Fermion mass squared (matches Vprime_scalar formula)
            mf2 = 0.5 * gf2 * phi_val**2 + (1.0 / 6.0) * gfg2 * T_val**2

            # Thermal integrals Jb and Jf (using exact CTFT)
            T2 = T_val**2
            if mb2 > 0 and T_val > 0:
                yb2 = mb2 / T2
                Jb_val = CTFT.Jb_exact(yb2)
            else:
                Jb_val = 0.0

            if mf2 > 0 and T_val > 0:
                yf2 = mf2 / T2
                Jf_val = CTFT.Jf_exact(yf2)
            else:
                Jf_val = 0.0

            # Thermal potential
            T4 = T_val**4
            pref = T4 / (2.0 * np.pi**2)
            V_thermal = pref * (2.0 * Jb_val + Jf_val)

            return V_tree + V_thermal

        # Find true vacuum and false vacuum positions
        # At finite T, the thermal minimum is at a MUCH smaller scale than tree-level
        # Search for the actual minimum of V(phi, T) in the range 10^5 to 10^8
        from scipy.optimize import minimize_scalar

        phi_false = 0.0

        # Find thermal minimum by minimizing V(phi) in the right range
        # At T ~ 7375, the thermal VEV is around 32000 (3.2e4)
        # Search in a range that includes this value
        result = minimize_scalar(V_for_bounce, bounds=(1e3, 1e6), method="bounded")
        phi_true_thermal = result.x

        print(f"  Searching for thermal minimum in range [1e3, 1e6]...")
        print(f"  Found thermal minimum at: phi = {phi_true_thermal:.4e}")
        print(f"  V(thermal min) = {result.fun:.4e}")

        # Also check at different scales for comparison
        print(f"\n  V at different scales:")
        for scale in [1e3, 1e4, 3e4, 5e4, 1e5, 1e6]:
            print(f"    V({scale:.0e}) = {V_for_bounce(scale):.4e}")

        phi_true = phi_true_thermal

        print(f"\n  False vacuum: phi = {phi_false:.4e}")
        print(f"  True vacuum:  phi = {phi_true:.4e}")
        print(f"  V(false) = {V_for_bounce(phi_false):.4e}")
        print(f"  V(true)  = {V_for_bounce(phi_true):.4e}")

        # Create SingleFieldInstanton
        # Arguments: phi_absMin, phi_metaMin, V
        instanton = SingleFieldInstanton(
            phi_absMin=phi_true, phi_metaMin=phi_false, V=V_for_bounce
        )

        # Find the bounce profile
        profile = instanton.findProfile()

        # profile.R: radial coordinates (1D array)
        # profile.Phi: field values at each r (1D array)
        R_bounce = np.array(profile.R)
        Phi_bounce = np.array(profile.Phi).flatten()

        print(f"\n  Bounce profile computed!")
        print(f"  Number of profile points: {len(R_bounce)}")
        print(f"  R range: [{R_bounce.min():.4e}, {R_bounce.max():.4e}]")
        print(f"  Phi range: [{Phi_bounce.min():.4e}, {Phi_bounce.max():.4e}]")
        print(f"  Phi at r=0 (center): {Phi_bounce[0]:.4e}")
        print(f"  Phi at r=max (edge): {Phi_bounce[-1]:.4e}")

        # Check if bounce is valid (phi should vary from true vac to false vac)
        phi_variation = abs(Phi_bounce[0] - Phi_bounce[-1])
        if phi_variation < 1e-6 * abs(phi_true):
            print(f"\n  WARNING: Phi is nearly constant! Bounce may have failed.")
            print(f"  Phi variation: {phi_variation:.4e}")

        # Compute the action
        try:
            action = instanton.findAction(profile)
            print(f"  Bounce action S3 = {action:.4e}")
            print(f"  S3/T = {action / T0:.4e}")
        except Exception as e:
            print(f"  (Could not compute action: {e})")

        # Create interpolator for the bounce profile
        bounce_profile_interp = interp1d(
            R_bounce,
            Phi_bounce,
            kind="cubic",
            bounds_error=False,
            fill_value=(Phi_bounce[0], phi_false),  # (inside, outside)
        )

        # Convert R_bounce to lattice units
        R_max_physical = R_bounce.max()
        print(f"\n  Bounce radius in physical units: ~{R_max_physical:.4e}")
        print(f"  Lattice spacing dx = {dx:.4e}")
        R_bounce_lattice_natural = R_max_physical / dx
        print(
            f"  Natural bounce radius in lattice units: ~{R_bounce_lattice_natural:.4f}"
        )

        # Scale the bounce to target radius in lattice units
        # r_physical = r_lattice * (R_max_physical / BOUNCE_TARGET_RADIUS)
        bounce_R_scale = R_max_physical / BOUNCE_TARGET_RADIUS
        print(f"\n  Scaling bounce to {BOUNCE_TARGET_RADIUS:.0f} lattice units")
        print(f"  Scale factor: {bounce_R_scale:.4e}")
        print(f"  (r_physical = r_lattice * {bounce_R_scale:.4e})")

        print("\n  Bounce profile ready for seeding!")

    except Exception as e:
        print(f"\n  ERROR computing bounce profile: {e}")
        print("  Falling back to 'tanh' profile")
        import traceback

        traceback.print_exc()
        BUBBLE_PROFILE = "tanh"


def seed_bubble(
    phi_field,
    center_x,
    center_y,
    center_z,
    radius,
    sign,
    vev_val,
    profile="tanh",
    wall_width=3.0,
    bounce_interp=None,
    bounce_scale=1.0,
):
    """
    Seed a true vacuum bubble at specified location (3D).

    Parameters:
    -----------
    phi_field : ndarray
        The field array to modify (in-place)
    center_x, center_y, center_z : float
        Bubble center position in lattice units
    radius : float
        Bubble radius in lattice units (ignored for 'bounce' profile)
    sign : int
        +1 for positive VEV, -1 for negative VEV
    vev_val : float
        The VEV value (ignored for 'bounce' profile)
    profile : str
        'sharp', 'tanh', or 'bounce'
    wall_width : float
        Wall thickness for tanh profile (in lattice units)
    bounce_interp : callable
        Interpolator for bounce profile phi(r) in physical units
    bounce_scale : float
        Scale factor to convert lattice units to physical units

    Returns:
    --------
    phi_field : ndarray
        Modified field array
    """
    Nx_local, Ny_local, Nz_local = phi_field.shape

    x = np.arange(Nx_local)
    y = np.arange(Ny_local)
    z = np.arange(Nz_local)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dx_arr = X - center_x
    dy_arr = Y - center_y
    dz_arr = Z - center_z

    dx_arr = np.where(dx_arr > Nx_local / 2, dx_arr - Nx_local, dx_arr)
    dx_arr = np.where(dx_arr < -Nx_local / 2, dx_arr + Nx_local, dx_arr)
    dy_arr = np.where(dy_arr > Ny_local / 2, dy_arr - Ny_local, dy_arr)
    dy_arr = np.where(dy_arr < -Ny_local / 2, dy_arr + Ny_local, dy_arr)
    dz_arr = np.where(dz_arr > Nz_local / 2, dz_arr - Nz_local, dz_arr)
    dz_arr = np.where(dz_arr < -Nz_local / 2, dz_arr + Nz_local, dz_arr)

    r = np.sqrt(dx_arr**2 + dy_arr**2 + dz_arr**2)

    if profile == "sharp":
        # Step function: instant transition at radius
        bubble_mask = r < radius
        phi_field[bubble_mask] = sign * vev_val
    elif profile == "tanh":
        # Smooth tanh profile (kink-like solution)
        # Correct: transition from +vev (inside) to original field (outside)
        # step = 1 inside bubble, step = 0 outside bubble
        step = 0.5 * (1 + np.tanh((radius - r) / wall_width))
        # Interpolate: inside → sign*vev, outside → original phi (≈0)
        # This avoids the negative ring artifact from the old formula
        phi_field[:] = sign * vev_val * step + (1 - step) * phi_field
    elif profile == "bounce":
        # Use CosmoTransitions bounce profile
        if bounce_interp is None:
            raise ValueError("bounce_interp required for 'bounce' profile")
        # Convert lattice radius to physical radius
        # r is in lattice units, bounce_scale = dx (physical per lattice)
        r_physical = r * bounce_scale  # lattice → physical
        # Get field values from bounce profile
        phi_bounce = bounce_interp(r_physical)
        # Apply sign (for positive/negative VEV bubbles)
        phi_field[:] = sign * phi_bounce
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return phi_field


def seed_multiple_bubbles(
    phi_field,
    bubble_config,
    vev_val,
    profile="tanh",
    wall_width=3.0,
    bounce_interp=None,
    bounce_scale=1.0,
):
    """
    Seed multiple true vacuum bubbles.

    Parameters:
    -----------
    phi_field : ndarray
        The field array to modify
    bubble_config : list of tuples
        Each tuple: (center_x, center_y, center_z, radius, sign)
    vev_val : float
        The VEV value (ignored for 'bounce' profile)
    profile : str
        'sharp', 'tanh', or 'bounce'
    wall_width : float
        Wall thickness for tanh profile
    bounce_interp : callable
        Interpolator for bounce profile (for 'bounce' profile)
    bounce_scale : float
        Scale factor for bounce profile

    Returns:
    --------
    phi_field : ndarray
        Modified field array
    """
    for cx, cy, cz, r, s in bubble_config:
        seed_bubble(
            phi_field,
            cx,
            cy,
            cz,
            r,
            s,
            vev_val,
            profile,
            wall_width,
            bounce_interp=bounce_interp,
            bounce_scale=bounce_scale,
        )
        if profile == "bounce":
            print(
                f"  Seeded bubble: center=({cx}, {cy}, {cz}), "
                f"sign={'+' if s > 0 else '-'}, profile=BOUNCE"
            )
        else:
            print(
                f"  Seeded bubble: center=({cx}, {cy}, {cz}), radius={r}, "
                f"sign={'+' if s > 0 else '-'}, profile={profile}"
            )
    return phi_field


# These will be set to True / nonzero if --resume finds a valid checkpoint
resuming = False
n_start = 0

# Initialize fields (3D)
phi = (0.01 * np.random.randn(Nx, Ny, Nz)).astype(field_dtype)
pi = np.zeros((Nx, Ny, Nz), dtype=field_dtype)

# Apply bubble seeding if enabled (skip when resuming from checkpoint)
if SEED_BUBBLES and not resuming:
    print("\n" + "=" * 60)
    print("BUBBLE SEEDING ENABLED")
    print("=" * 60)
    print(f"Number of bubbles: {len(BUBBLE_CONFIG)}")
    print(f"Profile type: {BUBBLE_PROFILE}")
    if BUBBLE_PROFILE == "tanh":
        print(f"Wall width: {BUBBLE_WALL_WIDTH} lattice units")
    elif BUBBLE_PROFILE == "bounce":
        if bounce_profile_interp is not None:
            print("Using CosmoTransitions bounce solution")
            print(f"Scaled to radius: {BOUNCE_TARGET_RADIUS:.0f} lattice units")
        else:
            print("WARNING: Bounce profile not computed, falling back to tanh")
            BUBBLE_PROFILE = "tanh"
    print(f"VEV magnitude: {vev:.4e}")
    print("-" * 60)

    # Set up bounce parameters if using bounce profile
    # bounce_R_scale converts lattice r to physical r for the bounce interpolator
    bounce_scale_param = bounce_R_scale if BUBBLE_PROFILE == "bounce" else 1.0

    phi = seed_multiple_bubbles(
        phi,
        BUBBLE_CONFIG,
        vev,
        profile=BUBBLE_PROFILE,
        wall_width=BUBBLE_WALL_WIDTH,
        bounce_interp=bounce_profile_interp,
        bounce_scale=bounce_scale_param,
    )
    print("-" * 60)
    print(f"Field after seeding: min={phi.min():.4e}, max={phi.max():.4e}")
    print("=" * 60 + "\n")
elif not resuming:
    print("Bubble seeding: DISABLED (thermal nucleation only)")

# (Potential parameters already defined above for thermal VEV calculation)


# =====================================================
# Numba utilities: periodic Laplacian, cubic eval, V' kernel, RK2 step
# =====================================================
@nb.njit(cache=True)
def _find_span(t, k, x):
    n = t.size - k - 1
    if x <= t[k]:
        return k
    if x >= t[n]:
        return n - 1
    lo = k
    hi = n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < t[mid]:
            hi = mid
        else:
            lo = mid
    return lo


@nb.njit(fastmath=True, cache=True)
def bspline_de_boor(t, c, k, x):
    # Clamp x to knot domain to avoid NaNs
    if x <= t[k]:
        x = t[k]
    else:
        n = t.size - k - 1
        if x >= t[n]:
            x = t[n] - 1e-12
    i = _find_span(t, k, x)
    d = np.empty(k + 1, dtype=np.float64)
    base = i - k
    for j in range(k + 1):
        d[j] = c[base + j]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left = t[i + j - k]
            right = t[i + 1 + j - r]
            denom = right - left
            alpha = 0.0 if denom == 0.0 else (x - left) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


@nb.njit(parallel=True, cache=True)
def generate_noise_field(noise, scale, seed):
    """Generate Gaussian noise field in-kernel using Numba's RNG (3D)."""
    nx, ny, nz = noise.shape
    np.random.seed(seed)
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                u1 = np.random.random()
                u2 = np.random.random()
                if u1 < 1e-12:
                    u1 = 1e-12
                z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
                noise[i, j, k] = scale * z0


@nb.njit(parallel=True, fastmath=True, cache=True)
def laplacian_periodic(out, a, dx):
    nx, ny, nz = a.shape
    inv_dx2 = 1.0 / (dx * dx)
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                out[i, j, k] = (
                    a[ip, j, k]
                    + a[im, j, k]
                    + a[i, jp, k]
                    + a[i, jm, k]
                    + a[i, j, kp]
                    + a[i, j, km]
                    - 6.0 * a[i, j, k]
                ) * inv_dx2


@nb.njit(parallel=True, fastmath=True, cache=True)
def gradient_flow_update(phi, lap, Vp, dt_gf, mu_val, weight):
    """
    Apply one gradient flow update with smooth spatial weighting:
        phi += weight * dt_gf * (lap - Vp / mu^2)

    The weight array is a smooth Gaussian centered on the bubble wall,
    tapering to zero in the interior and exterior. This:
      1. Concentrates relaxation at the wall (where force imbalance lives)
      2. Prevents saddle-point instability (interior/exterior barely move)
      3. Avoids sharp profile discontinuities at the band edges (unlike a
         hard binary mask, the smooth taper produces a continuous profile)

    NOTE: max|delta| for convergence is computed *outside* this kernel via
    NumPy to avoid a prange race condition on scalar max-reductions (Numba's
    prange only supports += / *= reductions, NOT if-based max reductions).
    """
    nx, ny, nz = phi.shape
    mu2 = mu_val * mu_val
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                phi[i, j, k] += (
                    weight[i, j, k] * dt_gf * (lap[i, j, k] - Vp[i, j, k] / mu2)
                )


@nb.njit(cache=True)
def cubic_eval_uniform(x, x_min, h, nseg, c0, c1, c2, c3):
    # map x to segment index
    t = (x - x_min) / h
    i = int(t)
    if i < 0:
        i = 0
    elif i >= nseg:
        i = nseg - 1
    dx = x - (x_min + i * h)
    # Horner evaluation: a*dx^3 + b*dx^2 + c*dx + d
    return ((c0[i] * dx + c1[i]) * dx + c2[i]) * dx + c3[i]


@nb.njit(parallel=True, fastmath=True, cache=True)
def Vprime_field(
    out,
    phi,
    T,
    lam,
    mphi,
    bosonMassSquared,
    bosonCoupling,
    bosonGaugeCoupling,
    fermionCoupling,
    fermionGaugeCoupling,
):
    nx, ny, nz = phi.shape
    T2 = T * T
    T4 = T2 * T2
    pref = T4 / (2.0 * math.pi * math.pi)
    gb2 = bosonCoupling * bosonCoupling
    gg2 = bosonGaugeCoupling * bosonGaugeCoupling
    gf2 = fermionCoupling * fermionCoupling
    gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
    coef_b_T = 0.25 * gb2 + (2.0 / 3.0) * gg2
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                ph = phi[i, j, k]
                dV = lam * ph * ph * ph - mphi * mphi * ph
                xb_sq = bosonMassSquared + 0.5 * gb2 * ph * ph + coef_b_T * T2
                xb = 0.0
                if xb_sq > 0.0:
                    xb = math.sqrt(xb_sq) / T
                xf_sq = 0.5 * gf2 * ph * ph + (1.0 / 6.0) * gfg2 * T2
                xf = 0.0
                if xf_sq > 0.0:
                    xf = math.sqrt(xf_sq) / T
                xb_clamped = xb
                if xb_clamped < x_min:
                    xb_clamped = x_min
                elif xb_clamped > x_min + h_y * nseg:
                    xb_clamped = x_min + h_y * nseg - 1e-12
                xf_clamped = xf
                if xf_clamped < x_min:
                    xf_clamped = x_min
                elif xf_clamped > x_min + h_y * nseg:
                    xf_clamped = x_min + h_y * nseg - 1e-12
                dJb = cubic_eval_uniform(
                    xb_clamped, x_min, h_y, nseg, c0_b, c1_b, c2_b, c3_b
                )
                dJf = cubic_eval_uniform(
                    xf_clamped, x_min, h_y, nseg, c0_f, c1_f, c2_f, c3_f
                )
                dxb_dphi = 0.5 * gb2 * ph / (T2 * max(xb, 1e-20))
                dxf_dphi = 0.5 * gf2 * ph / (T2 * max(xf, 1e-20))
                dV += pref * (
                    _POT_COEFFS[0] * dJb * dxb_dphi + _POT_COEFFS[1] * dJf * dxf_dphi
                )
                out[i, j, k] = dV


@nb.njit(cache=True)
def rk2_step(
    phi,
    pi,
    dt,
    dx,
    eta_eff,
    T,
    mu,
    lam,
    mphi,
    bosonMassSquared,
    bosonCoupling,
    bosonGaugeCoupling,
    fermionCoupling,
    fermionGaugeCoupling,
    noise,
    lap,
    Vp,
    phi_mid_buf,
    pi_mid_buf,
    inv_a2,
):
    # k1
    laplacian_periodic(lap, phi, dx)
    Vprime_field(
        Vp,
        phi,
        T,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    k1_phi = pi
    k1_pi = inv_a2 * lap - eta_eff * pi - Vp / (mu * mu)
    # midpoint
    phi_mid = phi_mid_buf
    pi_mid = pi_mid_buf
    phi_mid[:, :, :] = phi + 0.5 * dt * k1_phi
    pi_mid[:, :, :] = pi + 0.5 * dt * k1_pi
    # k2
    laplacian_periodic(lap, phi_mid, dx)
    Vprime_field(
        Vp,
        phi_mid,
        T,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    k2_phi = pi_mid
    k2_pi = inv_a2 * lap - eta_eff * pi_mid - Vp / (mu * mu)
    phi += dt * k2_phi
    pi += dt * k2_pi + noise


@nb.njit(cache=True)
def rk2_step_fused(
    phi,
    pi,
    dt,
    dx,
    eta_eff,
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
    noise,
    lap,
    Vp,
    phi_mid_buf,
    pi_mid_buf,
    inv_a2,
):
    """
    Fused RK2 that does two half-steps in one kernel call using T and T_mid.
    Slightly faster by reducing kernel launch overhead.
    inv_a2 = 1/a^2 accounts for Hubble redshift of spatial gradients.
    """
    half_dt = 0.5 * dt
    half_noise = 0.5 * noise
    # k1 at t
    laplacian_periodic(lap, phi, dx)
    Vprime_field(
        Vp,
        phi,
        T,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    k1_phi = pi
    k1_pi = inv_a2 * lap - eta_eff * pi - Vp / (mu * mu)
    phi_temp = phi + half_dt * k1_phi
    pi_temp = pi + half_dt * k1_pi
    # k2 at midpoint
    laplacian_periodic(lap, phi_temp, dx)
    Vprime_field(
        Vp,
        phi_temp,
        T,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    k2_phi = pi_temp
    k2_pi = inv_a2 * lap - eta_eff * pi_temp - Vp / (mu * mu)
    phi += half_dt * k2_phi
    pi += half_dt * k2_pi + half_noise
    # Second half-step from midpoint to t+dt using T_mid
    laplacian_periodic(lap, phi, dx)
    Vprime_field(
        Vp,
        phi,
        T_mid,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    k1_phi = pi
    k1_pi = inv_a2 * lap - eta_eff * pi - Vp / (mu * mu)
    phi_mid = phi_mid_buf
    pi_mid = pi_mid_buf
    phi_mid[:, :, :] = phi + half_dt * k1_phi
    pi_mid[:, :, :] = pi + half_dt * k1_pi
    # k2 at t+dt
    laplacian_periodic(lap, phi_mid, dx)
    Vprime_field(
        Vp,
        phi_mid,
        T_mid,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    k2_phi = pi_mid
    k2_pi = inv_a2 * lap - eta_eff * pi_mid - Vp / (mu * mu)
    phi += half_dt * k2_phi
    pi += half_dt * k2_pi + half_noise


@nb.njit(cache=True)
def _vprime_at_site(
    ph, T_val, T2, pref, lam_val, mphi_sq, bms, gb2, gf2, gg2, gfg2, coef_b
):
    """V'(phi) at a single site with thermal corrections.

    Captures spline coefficient arrays (x_min, h_y, nseg, c0_b, ...) from module scope.
    """
    dV = lam_val * ph * ph * ph - mphi_sq * ph
    xb_sq = bms + 0.5 * gb2 * ph * ph + coef_b * T2
    xb = 0.0
    if xb_sq > 0.0:
        xb = math.sqrt(xb_sq) / T_val
    xf_sq = 0.5 * gf2 * ph * ph + (1.0 / 6.0) * gfg2 * T2
    xf = 0.0
    if xf_sq > 0.0:
        xf = math.sqrt(xf_sq) / T_val
    xb_c = xb
    if xb_c < x_min:
        xb_c = x_min
    elif xb_c > x_min + h_y * nseg:
        xb_c = x_min + h_y * nseg - 1e-12
    t_b = (xb_c - x_min) / h_y
    si = int(t_b)
    if si < 0:
        si = 0
    elif si >= nseg:
        si = nseg - 1
    dx_b = xb_c - (x_min + si * h_y)
    dJb = ((c0_b[si] * dx_b + c1_b[si]) * dx_b + c2_b[si]) * dx_b + c3_b[si]
    xf_c = xf
    if xf_c < x_min:
        xf_c = x_min
    elif xf_c > x_min + h_y * nseg:
        xf_c = x_min + h_y * nseg - 1e-12
    t_f = (xf_c - x_min) / h_y
    si = int(t_f)
    if si < 0:
        si = 0
    elif si >= nseg:
        si = nseg - 1
    dx_f = xf_c - (x_min + si * h_y)
    dJf = ((c0_f[si] * dx_f + c1_f[si]) * dx_f + c2_f[si]) * dx_f + c3_f[si]
    dxb_dphi = 0.5 * gb2 * ph / (T2 * max(xb, 1e-20))
    dxf_dphi = 0.5 * gf2 * ph / (T2 * max(xf, 1e-20))
    dV += pref * (_POT_COEFFS[0] * dJb * dxb_dphi + _POT_COEFFS[1] * dJf * dxf_dphi)
    return dV


@nb.njit(parallel=True, fastmath=True, cache=True)
def rk2_fused_table(
    phi,
    pi,
    dt,
    dx,
    eta_eff,
    mu,
    noise,
    phi_tmp,
    pi_tmp,
    inv_a2,
    vp_table_T,
    vp_tmin_T,
    vp_dinv_T,
    vp_npts_T,
    vp_table_Tm,
    vp_tmin_Tm,
    vp_dinv_Tm,
    vp_npts_Tm,
):
    """Fully fused RK2 with V'(phi) evaluated via pre-built lookup table.

    Two separate tables are passed: one for temperature T (first half-step)
    and one for T_mid (second half-step). Each table is a 1D array of V' values
    on a uniform phi grid; lookup is a single linear interpolation per site.
    """
    nx, ny, nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    last_T = vp_npts_T - 2
    last_Tm = vp_npts_Tm - 2

    # ---- Pass 1: first half-step, k1 at (phi, T) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                ph = phi[i, j, k]
                pi_v = pi[i, j, k]
                lap = (
                    phi[ip, j, k]
                    + phi[im, j, k]
                    + phi[i, jp, k]
                    + phi[i, jm, k]
                    + phi[i, j, kp]
                    + phi[i, j, km]
                    - 6.0 * ph
                ) * inv_dx2
                fidx = (ph - vp_tmin_T) * vp_dinv_T
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_T:
                    idx = last_T
                frac = fidx - idx
                dV = vp_table_T[idx] + frac * (vp_table_T[idx + 1] - vp_table_T[idx])
                k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                phi_tmp[i, j, k] = ph + half_dt * pi_v
                pi_tmp[i, j, k] = pi_v + half_dt * k_pi

    # ---- Pass 2: first half-step, k2 at (phi_tmp, T) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                ph = phi_tmp[i, j, k]
                pi_v = pi_tmp[i, j, k]
                lap = (
                    phi_tmp[ip, j, k]
                    + phi_tmp[im, j, k]
                    + phi_tmp[i, jp, k]
                    + phi_tmp[i, jm, k]
                    + phi_tmp[i, j, kp]
                    + phi_tmp[i, j, km]
                    - 6.0 * ph
                ) * inv_dx2
                fidx = (ph - vp_tmin_T) * vp_dinv_T
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_T:
                    idx = last_T
                frac = fidx - idx
                dV = vp_table_T[idx] + frac * (vp_table_T[idx + 1] - vp_table_T[idx])
                k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                phi[i, j, k] += half_dt * pi_v
                pi[i, j, k] += half_dt * k_pi + 0.5 * noise[i, j, k]

    # ---- Pass 3: second half-step, k1 at (phi, T_mid) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                ph = phi[i, j, k]
                pi_v = pi[i, j, k]
                lap = (
                    phi[ip, j, k]
                    + phi[im, j, k]
                    + phi[i, jp, k]
                    + phi[i, jm, k]
                    + phi[i, j, kp]
                    + phi[i, j, km]
                    - 6.0 * ph
                ) * inv_dx2
                fidx = (ph - vp_tmin_Tm) * vp_dinv_Tm
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_Tm:
                    idx = last_Tm
                frac = fidx - idx
                dV = vp_table_Tm[idx] + frac * (vp_table_Tm[idx + 1] - vp_table_Tm[idx])
                k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                phi_tmp[i, j, k] = ph + half_dt * pi_v
                pi_tmp[i, j, k] = pi_v + half_dt * k_pi

    # ---- Pass 4: second half-step, k2 at (phi_tmp, T_mid) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                ph = phi_tmp[i, j, k]
                pi_v = pi_tmp[i, j, k]
                lap = (
                    phi_tmp[ip, j, k]
                    + phi_tmp[im, j, k]
                    + phi_tmp[i, jp, k]
                    + phi_tmp[i, jm, k]
                    + phi_tmp[i, j, kp]
                    + phi_tmp[i, j, km]
                    - 6.0 * ph
                ) * inv_dx2
                fidx = (ph - vp_tmin_Tm) * vp_dinv_Tm
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_Tm:
                    idx = last_Tm
                frac = fidx - idx
                dV = vp_table_Tm[idx] + frac * (vp_table_Tm[idx + 1] - vp_table_Tm[idx])
                k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                phi[i, j, k] += half_dt * pi_v
                pi[i, j, k] += half_dt * k_pi + 0.5 * noise[i, j, k]


@nb.njit(parallel=True, fastmath=True, cache=True)
def rk2_step_table(
    phi,
    pi,
    dt,
    dx,
    eta_eff,
    mu,
    noise,
    phi_tmp,
    pi_tmp,
    inv_a2,
    vp_table,
    vp_tmin,
    vp_dinv,
    vp_npts,
):
    """Non-fused RK2: single full step of size dt with Vprime lookup table.

    Only 2 grid passes (vs 4 for fused).
    """
    nx, ny, nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    last = vp_npts - 2

    # ---- Pass 1: k1 at current state, predict midpoint ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                ph = phi[i, j, k]
                pi_v = pi[i, j, k]
                lap = (
                    phi[ip, j, k]
                    + phi[im, j, k]
                    + phi[i, jp, k]
                    + phi[i, jm, k]
                    + phi[i, j, kp]
                    + phi[i, j, km]
                    - 6.0 * ph
                ) * inv_dx2
                fidx = (ph - vp_tmin) * vp_dinv
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last:
                    idx = last
                frac = fidx - idx
                dV = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                phi_tmp[i, j, k] = ph + half_dt * pi_v
                pi_tmp[i, j, k] = pi_v + half_dt * k_pi

    # ---- Pass 2: k2 at midpoint, advance full dt + noise ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            jm = j - 1 if j - 1 >= 0 else ny - 1
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                km = k - 1 if k - 1 >= 0 else nz - 1
                ph = phi_tmp[i, j, k]
                pi_v = pi_tmp[i, j, k]
                lap = (
                    phi_tmp[ip, j, k]
                    + phi_tmp[im, j, k]
                    + phi_tmp[i, jp, k]
                    + phi_tmp[i, jm, k]
                    + phi_tmp[i, j, kp]
                    + phi_tmp[i, j, km]
                    - 6.0 * ph
                ) * inv_dx2
                fidx = (ph - vp_tmin) * vp_dinv
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last:
                    idx = last
                frac = fidx - idx
                dV = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                phi[i, j, k] += dt * pi_v
                pi[i, j, k] += dt * k_pi + noise[i, j, k]


BAOAB_TILE_J = 16
BAOAB_TILE_K = 16


@nb.njit(parallel=True, fastmath=True, cache=True)
def baoab_step_table(
    phi,
    pi,
    dt,
    dx,
    gamma,
    mu,
    inv_a2,
    vp_table,
    vp_tmin,
    vp_dinv,
    vp_npts,
    noise_scale,
    seed,
):
    """BAOAB Langevin integrator with table-based V'(phi) and cache-blocking.

    BAOAB splitting for:  phi_tt = (1/a^2)*lap(phi) - gamma*pi - V'(phi)/mu^2 + noise

      B: pi += (dt/2) * F(phi)          [half-kick]
      A: phi += (dt/2) * pi             [half-drift]
      O: pi = c1*pi + c2*noise          [exact OU thermostat]
      A: phi += (dt/2) * pi             [half-drift]
      B: pi += (dt/2) * F(phi)          [half-kick]

    where F(phi) = (1/a^2)*lap(phi) - V'(phi)/mu^2.

    Only 2 grid passes needed (first B+A fused, then A+B fused with O in between).
    """
    nx, ny, nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    last = vp_npts - 2

    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1.0 - c1 * c1) * noise_scale

    tile_j = BAOAB_TILE_J
    tile_k = BAOAB_TILE_K

    # ---- Pass 1: B (half-kick) + A (half-drift) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = jj + tile_j
            if j_end > ny:
                j_end = ny
            for kk in range(0, nz, tile_k):
                k_end = kk + tile_k
                if k_end > nz:
                    k_end = nz
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi[i, j, k]
                        lap = (
                            phi[ip, j, k]
                            + phi[im, j, k]
                            + phi[i, jp, k]
                            + phi[i, jm, k]
                            + phi[i, j, kp]
                            + phi[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        fidx = (ph - vp_tmin) * vp_dinv
                        idx = int(fidx)
                        if idx < 0:
                            idx = 0
                        elif idx > last:
                            idx = last
                        frac = fidx - idx
                        dV = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                        force = inv_a2 * lap - dV * inv_mu2
                        pi_v = pi[i, j, k] + half_dt * force
                        phi[i, j, k] = ph + half_dt * pi_v
                        pi[i, j, k] = pi_v

    # ---- O step: exact Ornstein-Uhlenbeck thermostat ----
    np.random.seed(seed)
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                u1 = np.random.random()
                u2 = np.random.random()
                if u1 < 1e-12:
                    u1 = 1e-12
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                pi[i, j, k] = c1 * pi[i, j, k] + c2 * z

    # ---- Pass 2: A (half-drift) + B (half-kick) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = jj + tile_j
            if j_end > ny:
                j_end = ny
            for kk in range(0, nz, tile_k):
                k_end = kk + tile_k
                if k_end > nz:
                    k_end = nz
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        pi_v = pi[i, j, k]
                        ph = phi[i, j, k] + half_dt * pi_v
                        phi[i, j, k] = ph
                        lap = (
                            phi[ip, j, k]
                            + phi[im, j, k]
                            + phi[i, jp, k]
                            + phi[i, jm, k]
                            + phi[i, j, kp]
                            + phi[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        fidx = (ph - vp_tmin) * vp_dinv
                        idx = int(fidx)
                        if idx < 0:
                            idx = 0
                        elif idx > last:
                            idx = last
                        frac = fidx - idx
                        dV = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                        force = inv_a2 * lap - dV * inv_mu2
                        pi[i, j, k] = pi_v + half_dt * force


OVERDAMPED_TILE_J = 16
OVERDAMPED_TILE_K = 16


@nb.njit(parallel=True, fastmath=True, cache=True)
def overdamped_euler_step_table(
    phi,
    dt,
    dx,
    eta_eff,
    mu,
    inv_a2,
    vp_table,
    vp_tmin,
    vp_dinv,
    vp_npts,
    noise_scale,
    seed,
):
    """Overdamped (first-order) Langevin: Euler-Maruyama, single pass, in-place.

    Solves:  eta_eff * dphi/dt = (1/a^2)*lap(phi) - V'(phi)/mu^2 + xi

    Euler-Maruyama update:
      phi[i] += (dt/eta_eff) * F(phi[i]) + sqrt(2*T*dt/(eta_eff*mu^2*dx^3)) * z

    For additive noise, Euler-Maruyama has weak order 1 — identical to Heun for
    computing equilibrium distributions and nucleation rates.  Only 1 grid pass
    (same in-place neighbor race as BAOAB Pass 1; O(dt^2) error, standard practice).
    """
    nx, ny, nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    dt_over_eta = dt / eta_eff
    last = vp_npts - 2

    tile_j = OVERDAMPED_TILE_J
    tile_k = OVERDAMPED_TILE_K

    np.random.seed(seed)
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = jj + tile_j
            if j_end > ny:
                j_end = ny
            for kk in range(0, nz, tile_k):
                k_end = kk + tile_k
                if k_end > nz:
                    k_end = nz
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi[i, j, k]
                        lap = (
                            phi[ip, j, k]
                            + phi[im, j, k]
                            + phi[i, jp, k]
                            + phi[i, jm, k]
                            + phi[i, j, kp]
                            + phi[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        fidx = (ph - vp_tmin) * vp_dinv
                        idx = int(fidx)
                        if idx < 0:
                            idx = 0
                        elif idx > last:
                            idx = last
                        frac = fidx - idx
                        dV = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                        force = inv_a2 * lap - dV * inv_mu2
                        u1 = np.random.random()
                        u2 = np.random.random()
                        if u1 < 1e-12:
                            u1 = 1e-12
                        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                            2.0 * math.pi * u2
                        )
                        phi[i, j, k] = ph + dt_over_eta * force + noise_scale * z


RK2_INLINE_TILE_J = 16
RK2_INLINE_TILE_K = 16


@nb.njit(cache=True, fastmath=True)
def _vprime_inline(
    ph,
    T_val,
    T2,
    pref,
    lam_val,
    mphi_sq,
    bms,
    gb2,
    gf2,
    gg2,
    gfg2,
    coef_b,
    _x_min,
    _h_y,
    _inv_hy,
    _xhi,
    _nseg,
    _c0_b,
    _c1_b,
    _c2_b,
    _c3_b,
    _c0_f,
    _c1_f,
    _c2_f,
    _c3_f,
):
    """V'(phi) with precomputed inv_hy and xhi to avoid per-site redundant ops."""
    dV = lam_val * ph * ph * ph - mphi_sq * ph
    xb_sq = bms + 0.5 * gb2 * ph * ph + coef_b * T2
    xb = 0.0
    if xb_sq > 0.0:
        xb = math.sqrt(xb_sq) / T_val
    xf_sq = 0.5 * gf2 * ph * ph + (1.0 / 6.0) * gfg2 * T2
    xf = 0.0
    if xf_sq > 0.0:
        xf = math.sqrt(xf_sq) / T_val
    xb_c = xb
    if xb_c < _x_min:
        xb_c = _x_min
    elif xb_c > _xhi:
        xb_c = _xhi
    t_b = (xb_c - _x_min) * _inv_hy
    si = int(t_b)
    if si < 0:
        si = 0
    elif si >= _nseg:
        si = _nseg - 1
    dx_b = xb_c - (_x_min + si * _h_y)
    dJb = ((_c0_b[si] * dx_b + _c1_b[si]) * dx_b + _c2_b[si]) * dx_b + _c3_b[si]
    xf_c = xf
    if xf_c < _x_min:
        xf_c = _x_min
    elif xf_c > _xhi:
        xf_c = _xhi
    t_f = (xf_c - _x_min) * _inv_hy
    si = int(t_f)
    if si < 0:
        si = 0
    elif si >= _nseg:
        si = _nseg - 1
    dx_f = xf_c - (_x_min + si * _h_y)
    dJf = ((_c0_f[si] * dx_f + _c1_f[si]) * dx_f + _c2_f[si]) * dx_f + _c3_f[si]
    dxb_dphi = 0.5 * gb2 * ph / (T2 * max(xb, 1e-20))
    dxf_dphi = 0.5 * gf2 * ph / (T2 * max(xf, 1e-20))
    dV += pref * (_POT_COEFFS[0] * dJb * dxb_dphi + _POT_COEFFS[1] * dJf * dxf_dphi)
    return dV


@nb.njit(cache=True, fastmath=True)
def _hash_rng_pair(seed):
    """Fast hash-based RNG: returns two uniform [0,1) values from a uint64 seed."""
    x = nb.uint64(seed)
    x = (x ^ (x >> nb.uint64(30))) * nb.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> nb.uint64(27))) * nb.uint64(0x94D049BB133111EB)
    x = x ^ (x >> nb.uint64(31))
    u1 = (x >> nb.uint64(11)) * (1.0 / 9007199254740992.0)
    x = (x * nb.uint64(0x2545F4914F6CDD1D)) ^ nb.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> nb.uint64(30))) * nb.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> nb.uint64(31))
    u2 = (x >> nb.uint64(11)) * (1.0 / 9007199254740992.0)
    if u1 < 1e-12:
        u1 = 1e-12
    return u1, u2


@nb.njit(parallel=True, fastmath=True, cache=True)
def rk2_fused_single_pass(
    phi,
    pi,
    dt,
    dx,
    eta_eff,
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
    noise_scale,
    noise_seed,
    phi_tmp,
    pi_tmp,
    inv_a2,
):
    """Fully fused RK2 with inline noise generation (no separate noise array).

    Optimised with cache-blocking, precomputed spline constants, and hash-based
    inline RNG to eliminate the separate noise generation pass.
    """
    nx, ny, nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    mphi_sq = mphi * mphi
    gb2 = bosonCoupling * bosonCoupling
    gg2 = bosonGaugeCoupling * bosonGaugeCoupling
    gf2 = fermionCoupling * fermionCoupling
    gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
    coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2

    # Spline coefficients — local refs avoid repeated global lookup
    _xm = x_min
    _hy = h_y
    _ihy = inv_hy
    _xhi = xhi_clamp
    _ns = nseg
    _c0b = c0_b
    _c1b = c1_b
    _c2b = c2_b
    _c3b = c3_b
    _c0f = c0_f
    _c1f = c1_f
    _c2f = c2_f
    _c3f = c3_f

    T2 = T * T
    pref = T2 * T2 / (2.0 * math.pi * math.pi)

    tile_j = RK2_INLINE_TILE_J
    tile_k = RK2_INLINE_TILE_K

    nyz = nb.int64(ny) * nb.int64(nz)

    # ---- Pass 1: first half-step, k1 at (phi, T) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = min(jj + tile_j, ny)
            for kk in range(0, nz, tile_k):
                k_end = min(kk + tile_k, nz)
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi[i, j, k]
                        pi_v = pi[i, j, k]
                        lap = (
                            phi[ip, j, k]
                            + phi[im, j, k]
                            + phi[i, jp, k]
                            + phi[i, jm, k]
                            + phi[i, j, kp]
                            + phi[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        dV = _vprime_inline(
                            ph,
                            T,
                            T2,
                            pref,
                            lam,
                            mphi_sq,
                            bosonMassSquared,
                            gb2,
                            gf2,
                            gg2,
                            gfg2,
                            coef_b,
                            _xm,
                            _hy,
                            _ihy,
                            _xhi,
                            _ns,
                            _c0b,
                            _c1b,
                            _c2b,
                            _c3b,
                            _c0f,
                            _c1f,
                            _c2f,
                            _c3f,
                        )
                        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                        phi_tmp[i, j, k] = ph + half_dt * pi_v
                        pi_tmp[i, j, k] = pi_v + half_dt * k_pi

    # ---- Pass 2: first half-step, k2 at (phi_tmp, T) + inline noise ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = min(jj + tile_j, ny)
            for kk in range(0, nz, tile_k):
                k_end = min(kk + tile_k, nz)
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi_tmp[i, j, k]
                        pi_v = pi_tmp[i, j, k]
                        lap = (
                            phi_tmp[ip, j, k]
                            + phi_tmp[im, j, k]
                            + phi_tmp[i, jp, k]
                            + phi_tmp[i, jm, k]
                            + phi_tmp[i, j, kp]
                            + phi_tmp[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        dV = _vprime_inline(
                            ph,
                            T,
                            T2,
                            pref,
                            lam,
                            mphi_sq,
                            bosonMassSquared,
                            gb2,
                            gf2,
                            gg2,
                            gfg2,
                            coef_b,
                            _xm,
                            _hy,
                            _ihy,
                            _xhi,
                            _ns,
                            _c0b,
                            _c1b,
                            _c2b,
                            _c3b,
                            _c0f,
                            _c1f,
                            _c2f,
                            _c3f,
                        )
                        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                        # Inline noise via hash RNG (eliminates separate noise pass)
                        _site = (
                            nb.int64(i) * nyz + nb.int64(j) * nb.int64(nz) + nb.int64(k)
                        )
                        _seed = nb.uint64(_site * nb.int64(73856093)) ^ nb.uint64(
                            noise_seed
                        )
                        _u1, _u2 = _hash_rng_pair(_seed)
                        _z = math.sqrt(-2.0 * math.log(_u1)) * math.cos(
                            6.283185307179586 * _u2
                        )
                        phi[i, j, k] += half_dt * pi_v
                        pi[i, j, k] += half_dt * k_pi + 0.5 * noise_scale * _z

    T2m = T_mid * T_mid
    prefm = T2m * T2m / (2.0 * math.pi * math.pi)

    # ---- Pass 3: second half-step, k1 at (phi, T_mid) ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = min(jj + tile_j, ny)
            for kk in range(0, nz, tile_k):
                k_end = min(kk + tile_k, nz)
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi[i, j, k]
                        pi_v = pi[i, j, k]
                        lap = (
                            phi[ip, j, k]
                            + phi[im, j, k]
                            + phi[i, jp, k]
                            + phi[i, jm, k]
                            + phi[i, j, kp]
                            + phi[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        dV = _vprime_inline(
                            ph,
                            T_mid,
                            T2m,
                            prefm,
                            lam,
                            mphi_sq,
                            bosonMassSquared,
                            gb2,
                            gf2,
                            gg2,
                            gfg2,
                            coef_b,
                            _xm,
                            _hy,
                            _ihy,
                            _xhi,
                            _ns,
                            _c0b,
                            _c1b,
                            _c2b,
                            _c3b,
                            _c0f,
                            _c1f,
                            _c2f,
                            _c3f,
                        )
                        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                        phi_tmp[i, j, k] = ph + half_dt * pi_v
                        pi_tmp[i, j, k] = pi_v + half_dt * k_pi

    # ---- Pass 4: second half-step, k2 at (phi_tmp, T_mid) + inline noise ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = min(jj + tile_j, ny)
            for kk in range(0, nz, tile_k):
                k_end = min(kk + tile_k, nz)
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi_tmp[i, j, k]
                        pi_v = pi_tmp[i, j, k]
                        lap = (
                            phi_tmp[ip, j, k]
                            + phi_tmp[im, j, k]
                            + phi_tmp[i, jp, k]
                            + phi_tmp[i, jm, k]
                            + phi_tmp[i, j, kp]
                            + phi_tmp[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        dV = _vprime_inline(
                            ph,
                            T_mid,
                            T2m,
                            prefm,
                            lam,
                            mphi_sq,
                            bosonMassSquared,
                            gb2,
                            gf2,
                            gg2,
                            gfg2,
                            coef_b,
                            _xm,
                            _hy,
                            _ihy,
                            _xhi,
                            _ns,
                            _c0b,
                            _c1b,
                            _c2b,
                            _c3b,
                            _c0f,
                            _c1f,
                            _c2f,
                            _c3f,
                        )
                        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                        _site = (
                            nb.int64(i) * nyz + nb.int64(j) * nb.int64(nz) + nb.int64(k)
                        )
                        _seed = nb.uint64(_site * nb.int64(19349669)) ^ nb.uint64(
                            noise_seed
                        )
                        _u1, _u2 = _hash_rng_pair(_seed)
                        _z = math.sqrt(-2.0 * math.log(_u1)) * math.cos(
                            6.283185307179586 * _u2
                        )
                        phi[i, j, k] += half_dt * pi_v
                        pi[i, j, k] += half_dt * k_pi + 0.5 * noise_scale * _z


RK2_NF_TILE_J = 16
RK2_NF_TILE_K = 16


@nb.njit(parallel=True, fastmath=True, cache=True)
def rk2_step_inline(
    phi,
    pi,
    dt,
    dx,
    eta_eff,
    T,
    mu,
    lam_val,
    mphi_val,
    bosonMassSquared,
    bosonCoupling,
    bosonGaugeCoupling,
    fermionCoupling,
    fermionGaugeCoupling,
    noise,
    phi_tmp,
    pi_tmp,
    inv_a2,
):
    """Non-fused RK2 with inline V'(phi): single full step, 2 passes only.

    Same midpoint-method (Heun) as rk2_step_table but evaluates V'(phi)
    inline via _vprime_inline with precomputed dJb/dJf spline coefficients.
    Cache-blocking (tiling) for better L1/L2 locality.
    """
    nx, ny, nz = phi.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    mphi_sq = mphi_val * mphi_val
    gb2 = bosonCoupling * bosonCoupling
    gg2 = bosonGaugeCoupling * bosonGaugeCoupling
    gf2 = fermionCoupling * fermionCoupling
    gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
    coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2

    _xm = x_min
    _hy = h_y
    _ihy = inv_hy
    _xhi = xhi_clamp
    _ns = nseg
    _c0b = c0_b
    _c1b = c1_b
    _c2b = c2_b
    _c3b = c3_b
    _c0f = c0_f
    _c1f = c1_f
    _c2f = c2_f
    _c3f = c3_f

    T2 = T * T
    pref = T2 * T2 / (2.0 * math.pi * math.pi)

    tile_j = RK2_NF_TILE_J
    tile_k = RK2_NF_TILE_K

    # ---- Pass 1: k1 at current state, predict midpoint ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = min(jj + tile_j, ny)
            for kk in range(0, nz, tile_k):
                k_end = min(kk + tile_k, nz)
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi[i, j, k]
                        pi_v = pi[i, j, k]
                        lap = (
                            phi[ip, j, k]
                            + phi[im, j, k]
                            + phi[i, jp, k]
                            + phi[i, jm, k]
                            + phi[i, j, kp]
                            + phi[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        dV = _vprime_inline(
                            ph,
                            T,
                            T2,
                            pref,
                            lam_val,
                            mphi_sq,
                            bosonMassSquared,
                            gb2,
                            gf2,
                            gg2,
                            gfg2,
                            coef_b,
                            _xm,
                            _hy,
                            _ihy,
                            _xhi,
                            _ns,
                            _c0b,
                            _c1b,
                            _c2b,
                            _c3b,
                            _c0f,
                            _c1f,
                            _c2f,
                            _c3f,
                        )
                        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                        phi_tmp[i, j, k] = ph + half_dt * pi_v
                        pi_tmp[i, j, k] = pi_v + half_dt * k_pi

    # ---- Pass 2: k2 at midpoint, advance full dt + noise ----
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        im = i - 1 if i - 1 >= 0 else nx - 1
        for jj in range(0, ny, tile_j):
            j_end = min(jj + tile_j, ny)
            for kk in range(0, nz, tile_k):
                k_end = min(kk + tile_k, nz)
                for j in range(jj, j_end):
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j - 1 >= 0 else ny - 1
                    for k in range(kk, k_end):
                        kp = k + 1 if k + 1 < nz else 0
                        km = k - 1 if k - 1 >= 0 else nz - 1
                        ph = phi_tmp[i, j, k]
                        pi_v = pi_tmp[i, j, k]
                        lap = (
                            phi_tmp[ip, j, k]
                            + phi_tmp[im, j, k]
                            + phi_tmp[i, jp, k]
                            + phi_tmp[i, jm, k]
                            + phi_tmp[i, j, kp]
                            + phi_tmp[i, j, km]
                            - 6.0 * ph
                        ) * inv_dx2
                        dV = _vprime_inline(
                            ph,
                            T,
                            T2,
                            pref,
                            lam_val,
                            mphi_sq,
                            bosonMassSquared,
                            gb2,
                            gf2,
                            gg2,
                            gfg2,
                            coef_b,
                            _xm,
                            _hy,
                            _ihy,
                            _xhi,
                            _ns,
                            _c0b,
                            _c1b,
                            _c2b,
                            _c3b,
                            _c0f,
                            _c1f,
                            _c2f,
                            _c3f,
                        )
                        k_pi = inv_a2 * lap - eta_eff * pi_v - dV * inv_mu2
                        phi[i, j, k] += dt * pi_v
                        pi[i, j, k] += dt * k_pi + noise[i, j, k]


# =====================================================
# Temperature schedule
# =====================================================
def temperature(t):
    return max(T0 - cooling_rate * t, 0.0)


# =====================================================
# Output
# =====================================================
param_set = "set7"
steps = cli_args.steps

# Determine integrator name
if USE_OVERDAMPED:
    INTEGRATOR_NAME = "overdamped"
elif USE_BAOAB and USE_VPRIME_TABLE:
    INTEGRATOR_NAME = "baoab"
elif USE_VPRIME_TABLE and USE_NONFUSED_TABLE_RK2:
    INTEGRATOR_NAME = "rk2_nonfused_table"
elif USE_NONFUSED_INLINE_RK2:
    INTEGRATOR_NAME = "rk2_nonfused_inline"
elif USE_VPRIME_TABLE and USE_SINGLE_PASS_RK2:
    INTEGRATOR_NAME = "rk2_fused_table"
elif USE_SINGLE_PASS_RK2:
    INTEGRATOR_NAME = "rk2_fused_inline"
elif USE_FUSED_RK2:
    INTEGRATOR_NAME = "rk2_fused"
else:
    INTEGRATOR_NAME = "rk2"

# Build save path
hubble_tag = "_hubble" if HUBBLE_EXPANSION else "_nohubble"
eta_val = OVERDAMPED_ETA if OVERDAMPED_MODE else eta_phys
eta_tag = f"_eta_{eta_val:g}"
overdamped_tag = "_OD" if USE_OVERDAMPED else ""
if SEED_BUBBLES:
    n_bubbles = len(BUBBLE_CONFIG)
    profile_tag = BUBBLE_PROFILE
    seed_tag = f"_seeded_{n_bubbles}bubbles_{profile_tag}"
else:
    seed_tag = ""
dx_tag = f"_dx_{dx_phys:g}"
dtphys_tag = f"_dtphys_{dt_phys:g}"
coupling_tag = f"_gb_{bosonCoupling:g}_gf_{fermionCoupling:g}"
integrator_tag = f"_{INTEGRATOR_NAME}"
counterterm_tag = "_CT" if cli_args.counterterm else ""
pot_type_tag = f"_{cli_args.potential_type}" if cli_args.potential_type != "V_p" else ""
save_path = (
    f"data/lattice/{param_set}/{Nx}x{Ny}x{Nz}_T0_{int(T0)}"
    f"{dx_tag}{dtphys_tag}_interval_{steps}_3D{hubble_tag}{eta_tag}{coupling_tag}{overdamped_tag}{integrator_tag}{counterterm_tag}{pot_type_tag}{seed_tag}"
)
os.makedirs(save_path, exist_ok=True)

# Create directory for field states
state_path = f"{save_path}/field_states"
os.makedirs(state_path, exist_ok=True)
print(f"Field states will be saved to: {state_path}")

fig_path = f"{save_path}/figs/latticeSnapshot"
os.makedirs(fig_path, exist_ok=True)
print(f"Figures will be saved to: {fig_path}")

# =====================================================
# Replay mode: override output paths and load checkpoint
# =====================================================
if REPLAY_MODE:
    _replay_start = cli_args.start_step
    _replay_end = _replay_start + cli_args.replay_steps
    _replay_save_every = cli_args.replay_save_every
    steps = _replay_save_every

    _replay_src_state_path = os.path.join(cli_args.replay_dir, "field_states")
    _replay_ckpt_file = os.path.join(
        _replay_src_state_path, f"state_step_{_replay_start:010d}.npz"
    )
    if not os.path.exists(_replay_ckpt_file):
        print(f"ERROR: Checkpoint file not found: {_replay_ckpt_file}")
        sys.exit(1)

    _replay_out = os.path.join(
        cli_args.replay_dir,
        f"detailed_replay/step_{_replay_start}_to_{_replay_end}",
    )
    save_path = _replay_out
    state_path = f"{save_path}/field_states"
    fig_path = f"{save_path}/figs/latticeSnapshot"
    os.makedirs(state_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    ckpt_data = np.load(_replay_ckpt_file)
    ckpt_phi = ckpt_data["phi"]
    if not USE_OVERDAMPED:
        if "pi" not in ckpt_data:
            print(f"ERROR: {_replay_ckpt_file} has no 'pi' (lightweight snapshot).")
            print("  Replay requires a full checkpoint. Use a file with 'pi'.")
            sys.exit(1)
        ckpt_pi = ckpt_data["pi"]
        if np.any(np.isnan(ckpt_pi)):
            print(
                f"ERROR: Checkpoint {_replay_ckpt_file} pi contains NaN. Cannot replay."
            )
            sys.exit(1)
    if np.any(np.isnan(ckpt_phi)):
        print(f"ERROR: Checkpoint {_replay_ckpt_file} phi contains NaN. Cannot replay.")
        sys.exit(1)

    resuming = True
    n_start = _replay_start
    phi = ckpt_phi.astype(field_dtype)
    if not USE_OVERDAMPED:
        pi = ckpt_pi.astype(field_dtype)
    if HUBBLE_EXPANSION and "scale_factor" in ckpt_data:
        a_current = float(ckpt_data["scale_factor"])

    print(f"\n{'='*60}")
    print("REPLAY MODE - DETAILED BUBBLE DYNAMICS")
    print(f"{'='*60}")
    print(f"  Source dir: {cli_args.replay_dir}")
    print(f"  Checkpoint: {_replay_ckpt_file}")
    print(
        f"  Step range: {_replay_start} -> {_replay_end} ({cli_args.replay_steps} steps)"
    )
    print(
        f"  Save every: {_replay_save_every} steps ({cli_args.replay_steps // _replay_save_every} snapshots)"
    )
    print(f"  Time: {float(ckpt_data['time'])/mu:.6e} (physical)")
    print(f"  Temperature: {float(ckpt_data['temperature']):.1f}")
    print(f"  phi range: [{ckpt_phi.min():.4e}, {ckpt_phi.max():.4e}]")
    if HUBBLE_EXPANSION:
        print(f"  scale_factor: {a_current:.10f}")
    print(f"  Output: {_replay_out}")
    print(f"{'='*60}\n")

# =====================================================
# Checkpoint resume (normal mode only)
# =====================================================
if not REPLAY_MODE and cli_args.resume:
    checkpoint_files = sorted(glob.glob(f"{state_path}/state_step_*.npz"))
    checkpoint_files = [f for f in checkpoint_files if "_NaN_debug" not in f]
    if checkpoint_files:
        ckpt_data = None
        for latest_ckpt in reversed(checkpoint_files):
            _d = np.load(latest_ckpt)
            if not USE_OVERDAMPED and "pi" not in _d:
                continue
            ckpt_phi = _d["phi"]
            if np.any(np.isnan(ckpt_phi)):
                print(f"  Skipping {latest_ckpt} (contains NaN)")
                continue
            if not USE_OVERDAMPED:
                ckpt_pi = _d["pi"]
                if np.any(np.isnan(ckpt_pi)):
                    print(f"  Skipping {latest_ckpt} (pi contains NaN)")
                    continue
            ckpt_data = _d
            ckpt_step = int(_d["step"])
            break
        if ckpt_data is None:
            if USE_OVERDAMPED:
                print("  No valid checkpoint found. Starting from scratch.")
            else:
                print("  No valid checkpoint (with pi) found. Starting from scratch.")
        if ckpt_data is not None:
            resuming = True
            n_start = ckpt_step
            phi = ckpt_phi.astype(field_dtype)
            if not USE_OVERDAMPED:
                pi = ckpt_pi.astype(field_dtype)
            if HUBBLE_EXPANSION and "scale_factor" in ckpt_data:
                a_current = float(ckpt_data["scale_factor"])
            print(f"\n{'='*60}")
            print("RESUMING FROM CHECKPOINT")
            print(f"{'='*60}")
            print(f"  File: {latest_ckpt}")
            print(f"  Step: {ckpt_step}  ->  resuming from step {n_start}")
            print(f"  Time: {float(ckpt_data['time'])/mu:.6e} (physical)")
            print(f"  Temperature: {float(ckpt_data['temperature']):.1f}")
            print(f"  phi range: [{ckpt_phi.min():.4e}, {ckpt_phi.max():.4e}]")
            if HUBBLE_EXPANSION:
                print(f"  scale_factor: {a_current:.10f}")
            print(f"  Remaining steps: {Nt - n_start:,}")
            print(f"{'='*60}\n")
    else:
        print("\n--resume specified but no checkpoints found. Starting from scratch.")

# =====================================================
# Time evolution
# =====================================================
# Preallocate temporaries (match field dtype for efficiency)
lap_tmp = np.empty((Nx, Ny, Nz), dtype=field_dtype)
Vp_tmp = np.empty((Nx, Ny, Nz), dtype=field_dtype)
phi_mid = np.empty((Nx, Ny, Nz), dtype=field_dtype)
pi_mid = np.empty((Nx, Ny, Nz), dtype=field_dtype)
noise = np.empty((Nx, Ny, Nz), dtype=field_dtype)

# Run comparison plots first (skip in replay mode)
if not REPLAY_MODE:
    comparison_plots()

print("\n" + "=" * 60)
print("Starting time evolution...")
print(f"Grid: {Nx}×{Ny}×{Nz}, Steps: {Nt}, dt: {dt:.2e}")
print(f"Integrator: {INTEGRATOR_NAME}")
print(f"Using Vprime lookup table: {USE_VPRIME_TABLE} (size={VPRIME_TABLE_SIZE})")
print(f"Using in-kernel RNG: {USE_NUMBA_RNG}")
print(f"Field precision: {field_dtype.__name__}")
print(f"Spline resolution N_Y: {N_Y}")
if SEED_BUBBLES:
    print(f"Bubble seeding: ENABLED ({len(BUBBLE_CONFIG)} bubbles)")
    for i, (cx, cy, cz, r, s) in enumerate(BUBBLE_CONFIG):
        sign_str = "+" if s > 0 else "-"
        print(f"  Bubble {i+1}: ({cx}, {cy}, {cz}), R={r}, {sign_str}VEV")
else:
    print("Bubble seeding: DISABLED")
if DISABLE_THERMAL_NOISE:
    print("Thermal noise: DISABLED (deterministic evolution)")
else:
    if USE_OVERDAMPED:
        _ns0 = np.sqrt(2.0 * T0 * dt / (eta * mu**2 * dx_phys**3))
        print(f"Thermal noise: ENABLED (overdamped FDT noise_scale(T0) = {_ns0:.4e})")
    else:
        _ns0 = np.sqrt(2.0 * eta * T0 * dt / (mu**2 * dx_phys**3))
        print(f"Thermal noise: ENABLED (FDT noise_scale(T0) = {_ns0:.4e})")
if USE_OVERDAMPED:
    print(f"Overdamped (first-order) Langevin: ENABLED (η = {eta_phys}, Heun method)")
    print(f"  No momentum (pi) — pure dissipative dynamics")
elif OVERDAMPED_MODE:
    print(f"Overdamped mode: ENABLED (η = {OVERDAMPED_ETA}, no ringing)")
else:
    print(f"Overdamped mode: DISABLED (η = {eta_phys})")
if HUBBLE_EXPANSION:
    H0_diag = hubble_param(T0)
    print(f"Hubble expansion: ENABLED")
    print(f"  H(T0) = {H0_diag:.4e} GeV, 3H/η = {3*H0_diag/eta_phys:.2e}")
    print(f"  g_* = {G_STAR}, M_Pl = {M_PL:.2e} GeV")
else:
    print("Hubble expansion: DISABLED (linear cooling)")
print("=" * 60 + "\n")

t_start = time.time()

# Warmup: force JIT compilation by running a few steps
print("Warming up JIT compilation...")
warmup_noise = np.empty((Nx, Ny, Nz), dtype=field_dtype)
warmup_scale = np.sqrt(2.0 * eta * T0 * dt / (mu**2 * dx_phys**3))
if USE_NUMBA_RNG:
    generate_noise_field(warmup_noise, warmup_scale, 0)
else:
    temp = np.random.randn(Nx, Ny, Nz) * warmup_scale
    warmup_noise[:] = temp.astype(field_dtype)
_warmup_tbl, _warmup_tmin, _warmup_dinv = build_vprime_table(
    T0, -30000.0, 30000.0, VPRIME_TABLE_SIZE
)
if USE_OVERDAMPED and USE_VPRIME_TABLE:
    _warmup_od_ns = np.sqrt(2.0 * T0 * dt / (eta * mu**2 * dx_phys**3))
    overdamped_euler_step_table(
        phi,
        dt,
        dx,
        eta,
        mu,
        1.0,
        _warmup_tbl,
        _warmup_tmin,
        _warmup_dinv,
        VPRIME_TABLE_SIZE,
        _warmup_od_ns,
        0,
    )
elif USE_BAOAB and USE_VPRIME_TABLE:
    _warmup_ns = np.sqrt(T0 / (mu**2 * dx_phys**3))
    baoab_step_table(
        phi,
        pi,
        dt,
        dx,
        eta,
        mu,
        1.0,
        _warmup_tbl,
        _warmup_tmin,
        _warmup_dinv,
        VPRIME_TABLE_SIZE,
        _warmup_ns,
        0,
    )
elif USE_VPRIME_TABLE and USE_NONFUSED_TABLE_RK2:
    rk2_step_table(
        phi,
        pi,
        dt,
        dx,
        eta,
        mu,
        warmup_noise,
        phi_mid,
        pi_mid,
        1.0,
        _warmup_tbl,
        _warmup_tmin,
        _warmup_dinv,
        VPRIME_TABLE_SIZE,
    )
elif USE_NONFUSED_INLINE_RK2:
    rk2_step_inline(
        phi,
        pi,
        dt,
        dx,
        eta,
        T0,
        mu,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
        warmup_noise,
        phi_mid,
        pi_mid,
        1.0,
    )
elif USE_VPRIME_TABLE and USE_SINGLE_PASS_RK2:
    rk2_fused_table(
        phi,
        pi,
        dt,
        dx,
        eta,
        mu,
        warmup_noise,
        phi_mid,
        pi_mid,
        1.0,
        _warmup_tbl,
        _warmup_tmin,
        _warmup_dinv,
        VPRIME_TABLE_SIZE,
        _warmup_tbl,
        _warmup_tmin,
        _warmup_dinv,
        VPRIME_TABLE_SIZE,
    )
elif USE_SINGLE_PASS_RK2:
    rk2_fused_single_pass(
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
        warmup_scale,
        nb.uint64(0),
        phi_mid,
        pi_mid,
        1.0,
    )
elif USE_FUSED_RK2:
    rk2_step_fused(
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
        warmup_noise,
        lap_tmp,
        Vp_tmp,
        phi_mid,
        pi_mid,
        1.0,
    )
else:
    rk2_step(
        phi,
        pi,
        0.5 * dt,
        dx,
        eta,
        T0,
        mu,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
        warmup_noise * 0.5,
        lap_tmp,
        Vp_tmp,
        phi_mid,
        pi_mid,
        1.0,
    )
print("JIT warmup complete.")

# =====================================================
# Force imbalance diagnostic helper
# =====================================================


def plot_force_imbalance(phi_arr, label_suffix, save_name):
    """
    Compute and plot the initial force imbalance F = laplacian(phi) - V'(phi)/mu^2.

    A large, localized F at the bubble wall confirms that the seeded profile
    does not satisfy the static field equation and will emit ring artifacts.
    After gradient flow relaxation, F should be nearly zero everywhere.
    """
    diag_lap = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    diag_Vp = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    laplacian_periodic(diag_lap, phi_arr, dx)
    Vprime_field(
        diag_Vp,
        phi_arr,
        T0,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    F = diag_lap - diag_Vp / (mu * mu)

    max_F = np.max(np.abs(F))
    max_loc = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    print(f"  Force imbalance {label_suffix}:")
    print(f"    max|F| = {max_F:.4e}  at lattice site {max_loc}")
    print(f"    phi range: [{phi_arr.min():.4e}, {phi_arr.max():.4e}]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    zmid = phi_arr.shape[2] // 2
    im0 = axes[0].imshow(
        np.asarray(phi_arr[:, :, zmid], dtype=np.float64).T,
        origin="lower",
        cmap="RdBu_r",
    )
    axes[0].set_title(f"phi field (z={zmid}) {label_suffix}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(
        np.asarray(F[:, :, zmid], dtype=np.float64).T,
        origin="lower",
        cmap="RdBu_r",
    )
    axes[1].set_title(f"Force imbalance F = lap(phi) - V'(phi)/mu^2 {label_suffix}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(
        f"max|F| = {max_F:.4e}  at {max_loc}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{fig_path}/{save_name}", dpi=200)
    plt.close(fig)
    print(f"    Saved: {fig_path}/{save_name}")


# =====================================================
# Pre-relaxation force imbalance diagnostic
# =====================================================
if SEED_BUBBLES:
    print("\n" + "-" * 60)
    print("FORCE IMBALANCE DIAGNOSTIC (before relaxation)")
    print("-" * 60)
    plot_force_imbalance(phi, "(before relaxation)", "initial_force_imbalance.png")
    print("-" * 60 + "\n")

# =====================================================
# Gradient flow relaxation (optional): relax bubble to near-equilibrium
# =====================================================
if SEED_BUBBLES and GRADIENT_FLOW_RELAX and not resuming:
    print("\n" + "=" * 60)
    print("GRADIENT FLOW RELAXATION")
    print("=" * 60)
    print(f"  dt_gf = {GF_DT}, max_iter = {GF_MAX_ITER}, tol = {GF_TOL}")
    print(f"  wall_margin = {GF_WALL_MARGIN} * wall_width")

    # ---- Build smooth Gaussian weight ----
    # Instead of a hard binary mask (which creates profile discontinuities
    # at the band edges), we use a smooth Gaussian weight centered on the
    # bubble wall. The update is: phi += weight * dt_gf * F
    # - At the wall (r = radius): weight = 1.0, full relaxation
    # - Away from wall: weight tapers smoothly to ~0, preventing the
    #   saddle-point instability (bubble cannot expand/collapse)
    # - No sharp edges: the profile remains continuous, avoiding the
    #   force spikes that a hard mask creates at its boundaries
    sigma_gf = GF_WALL_MARGIN * BUBBLE_WALL_WIDTH  # Gaussian width
    gf_weight = np.zeros((Nx, Ny, Nz), dtype=field_dtype)
    x_arr = np.arange(Nx)
    y_arr = np.arange(Ny)
    z_arr = np.arange(Nz)
    X_gf, Y_gf, Z_gf = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")
    for cx, cy, cz, bub_radius, bub_sign in BUBBLE_CONFIG:
        dx_gf = X_gf - cx
        dy_gf = Y_gf - cy
        dz_gf = Z_gf - cz
        dx_gf = np.where(dx_gf > Nx / 2, dx_gf - Nx, dx_gf)
        dx_gf = np.where(dx_gf < -Nx / 2, dx_gf + Nx, dx_gf)
        dy_gf = np.where(dy_gf > Ny / 2, dy_gf - Ny, dy_gf)
        dy_gf = np.where(dy_gf < -Ny / 2, dy_gf + Ny, dy_gf)
        dz_gf = np.where(dz_gf > Nz / 2, dz_gf - Nz, dz_gf)
        dz_gf = np.where(dz_gf < -Nz / 2, dz_gf + Nz, dz_gf)
        r_gf = np.sqrt(dx_gf**2 + dy_gf**2 + dz_gf**2)
        w = np.exp(-(((r_gf - bub_radius) / sigma_gf) ** 2)).astype(field_dtype)
        gf_weight = np.maximum(gf_weight, w)
    n_total = Nx * Ny * Nz
    n_active = int(np.sum(gf_weight > 0.01))
    print(f"  Smooth Gaussian weight: sigma = {sigma_gf:.1f} lattice units")
    print(
        f"  Active sites (weight > 0.01): {n_active} / {n_total} "
        f"({100.0*n_active/n_total:.1f}%)"
    )
    print(f"  Weight range: [{gf_weight.min():.4f}, {gf_weight.max():.4f}]")

    # ---- JIT warmup for gradient_flow_update ----
    _gf_warmup_lap = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    _gf_warmup_Vp = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    _gf_warmup_phi = phi.copy()
    laplacian_periodic(_gf_warmup_lap, _gf_warmup_phi, dx)
    Vprime_field(
        _gf_warmup_Vp,
        _gf_warmup_phi,
        T0,
        lam,
        mphi,
        bosonMassSquared,
        bosonCoupling,
        bosonGaugeCoupling,
        fermionCoupling,
        fermionGaugeCoupling,
    )
    gradient_flow_update(
        _gf_warmup_phi, _gf_warmup_lap, _gf_warmup_Vp, GF_DT, mu, gf_weight
    )
    del _gf_warmup_lap, _gf_warmup_Vp, _gf_warmup_phi
    print("  JIT warmup for gradient_flow_update complete.")

    # ---- Gradient flow loop ----
    # Create directory for gradient flow snapshots
    gf_snap_path = f"{fig_path}/gradient_flow_snapshots"
    os.makedirs(gf_snap_path, exist_ok=True)
    print(f"  Snapshots will be saved to: {gf_snap_path}")

    # Save initial state (iter 0)
    if GF_SAVE_EVERY > 0:
        plot_force_imbalance(
            phi, "(GF iter 0)", f"gradient_flow_snapshots/gf_iter_0000.png"
        )

    gf_start = time.time()
    gf_converged = False
    # Threshold for "significant weight" in convergence metric
    gf_sig_mask = gf_weight > 0.1
    # Early snapshot iterations to catch the fast initial changes
    gf_early_iters = {1, 10, 50, 100, 200}
    for gf_iter in range(GF_MAX_ITER):
        laplacian_periodic(lap_tmp, phi, dx)
        Vprime_field(
            Vp_tmp,
            phi,
            T0,
            lam,
            mphi,
            bosonMassSquared,
            bosonCoupling,
            bosonGaugeCoupling,
            fermionCoupling,
            fermionGaugeCoupling,
        )
        # Compute max|weighted delta| BEFORE the update.
        # Only count sites with significant weight (> 0.1) for convergence,
        # since low-weight sites barely move and shouldn't dominate the metric.
        force = lap_tmp - Vp_tmp / (mu * mu)
        weighted_delta = gf_weight * np.abs(force) * GF_DT
        sig_deltas = weighted_delta[gf_sig_mask]
        if sig_deltas.size > 0:
            max_delta = float(np.max(sig_deltas))
        else:
            max_delta = 0.0

        gradient_flow_update(phi, lap_tmp, Vp_tmp, GF_DT, mu, gf_weight)

        max_phi = float(np.max(np.abs(phi)))
        rel_change = max_delta / max(max_phi, 1e-30)

        if (gf_iter + 1) % GF_PRINT_EVERY == 0:
            print(
                f"  iter {gf_iter+1:6d}: max|dphi|={max_delta:.2e}, "
                f"rel={rel_change:.2e}, "
                f"phi range=[{phi.min():.2e}, {phi.max():.2e}]"
            )

        # Save snapshot at regular intervals + early snapshots to catch fast changes
        save_now = GF_SAVE_EVERY > 0 and (
            (gf_iter + 1) % GF_SAVE_EVERY == 0 or (gf_iter + 1) in gf_early_iters
        )
        if save_now:
            snap_name = f"gradient_flow_snapshots/gf_iter_{gf_iter+1:04d}.png"
            plot_force_imbalance(phi, f"(GF iter {gf_iter+1})", snap_name)

        if rel_change < GF_TOL:
            print(
                f"  Converged at iter {gf_iter+1}! "
                f"rel_change={rel_change:.2e} < tol={GF_TOL:.0e}"
            )
            gf_converged = True
            # Save final converged state
            if GF_SAVE_EVERY > 0:
                snap_name = (
                    f"gradient_flow_snapshots/gf_iter_{gf_iter+1:04d}_converged.png"
                )
                plot_force_imbalance(
                    phi, f"(GF iter {gf_iter+1}, converged)", snap_name
                )
            break

    if not gf_converged:
        print(f"  WARNING: Did not converge in {GF_MAX_ITER} iterations")
        print(f"  Final rel_change = {rel_change:.2e} (tol = {GF_TOL:.0e})")

    gf_time = time.time() - gf_start
    print(f"  Gradient flow took {gf_time:.2f}s ({gf_iter+1} iterations)")

    # Reset momentum after relaxation
    pi[:] = 0.0
    print("  Momentum reset to zero.")
    print("=" * 60 + "\n")

    # Post-relaxation diagnostic
    print("-" * 60)
    print("FORCE IMBALANCE DIAGNOSTIC (after gradient flow)")
    print("-" * 60)
    plot_force_imbalance(
        phi, "(after gradient flow)", "initial_force_imbalance_after_gf.png"
    )
    print("-" * 60 + "\n")

# =====================================================
# Settling period (optional): let bubble relax with high damping
# NOTE: Gradient flow relaxation (above) is preferred. This settling
# approach uses the RK2 integrator with tiny dt and is ~25,000x slower.
# =====================================================
if SEED_BUBBLES and SETTLING_ENABLED and not resuming:
    print(f"\nRunning settling ({SETTLING_STEPS} steps, η={SETTLING_ETA})...")
    settling_eta = SETTLING_ETA / mu  # Rescaled damping
    settling_noise = np.zeros((Nx, Ny, Nz), dtype=field_dtype)  # No noise
    T_settle = T0  # Fixed temperature during settling

    for s in range(SETTLING_STEPS):
        if USE_FUSED_RK2:
            rk2_step_fused(
                phi,
                pi,
                dt,
                dx,
                settling_eta,
                T_settle,
                T_settle,
                mu,
                lam,
                mphi,
                bosonMassSquared,
                bosonCoupling,
                bosonGaugeCoupling,
                fermionCoupling,
                fermionGaugeCoupling,
                settling_noise,
                lap_tmp,
                Vp_tmp,
                phi_mid,
                pi_mid,
                1.0,  # inv_a2 (no Hubble during settling)
            )
        else:
            rk2_step(
                phi,
                pi,
                0.5 * dt,
                dx,
                settling_eta,
                T_settle,
                mu,
                lam,
                mphi,
                bosonMassSquared,
                bosonCoupling,
                bosonGaugeCoupling,
                fermionCoupling,
                fermionGaugeCoupling,
                settling_noise * 0.5,
                lap_tmp,
                Vp_tmp,
                phi_mid,
                pi_mid,
                1.0,  # inv_a2 (no Hubble during settling)
            )
            rk2_step(
                phi,
                pi,
                0.5 * dt,
                dx,
                settling_eta,
                T_settle,
                mu,
                lam,
                mphi,
                bosonMassSquared,
                bosonCoupling,
                bosonGaugeCoupling,
                fermionCoupling,
                fermionGaugeCoupling,
                settling_noise * 0.5,
                lap_tmp,
                Vp_tmp,
                phi_mid,
                pi_mid,
                1.0,  # inv_a2 (no Hubble during settling)
            )
        if (s + 1) % 2000 == 0:
            print(
                f"  Settling step {s+1}/{SETTLING_STEPS}, "
                f"φ range: [{phi.min():.2e}, {phi.max():.2e}]"
            )

    print(f"Settling complete. φ: [{phi.min():.2e}, {phi.max():.2e}]")
    # Reset momentum after settling
    pi[:] = 0.0
    print("Momentum reset to zero.\n")

print("Starting main loop...\n")

t_start = time.time()  # Reset after warmup/settling

# Vprime lookup table state (rebuilt when T or phi range changes)
if USE_VPRIME_TABLE:
    _vp_table_T_last = -1.0
    _vp_table = None
    _vp_tmin = 0.0
    _vp_dinv = 1.0
    _vp_thi = 0.0
    _vp_table_mid = None
    _vp_tmin_mid = 0.0
    _vp_dinv_mid = 1.0
    _TABLE_MARGIN_FRAC = 1.0  # fractional padding on phi range

noise_scale = 0.0  # will be set each step if noise is enabled

# Dense snapshot state
_dense_enabled = cli_args.phi_threshold is not None and cli_args.steps_dense is not None
_dense_threshold = cli_args.phi_threshold if _dense_enabled else 0.0
_dense_steps = cli_args.steps_dense if _dense_enabled else steps
_dense_active = False
_total_saves = 0  # counter for checkpoint interval
if _dense_enabled:
    print(
        f"Dense snapshots: enabled (threshold={_dense_threshold:.1f}, interval={_dense_steps})"
    )

for n in range(n_start, Nt):
    t = n * dt

    if HUBBLE_EXPANSION:
        T = T0 / a_current
        _T4 = T * T * T * T
        H_now = math.sqrt((_T4 * _hubble_inv_chig2 + DEL_V) * _hubble_inv_3mpl2)
        eta_eff = eta + 3.0 * H_now / mu
        inv_a2 = 1.0 / (a_current * a_current)
        a_current += a_current * H_now * (dt / mu)
        T_mid = T0 / a_current
    else:
        T = temperature(t)
        T_mid = temperature(t + 0.5 * dt)
        eta_eff = eta
        inv_a2 = 1.0

    # Rebuild V'(phi) table when T changes or phi exceeds table range
    if USE_VPRIME_TABLE:
        _need_rebuild = _vp_table is None
        if not _need_rebuild:
            _need_rebuild = abs(T - _vp_table_T_last) / max(abs(T), 1.0) > 1e-4
        if not _need_rebuild:
            _cur_lo = float(phi.min())
            _cur_hi = float(phi.max())
            _need_rebuild = _cur_lo < _vp_tmin or _cur_hi > _vp_thi
        if _need_rebuild:
            _cur_lo = float(phi.min())
            _cur_hi = float(phi.max())
            _range = max(_cur_hi - _cur_lo, 1.0)
            _margin = max(_range * _TABLE_MARGIN_FRAC, 20000.0)
            _phi_lo = _cur_lo - _margin
            _phi_hi = _cur_hi + _margin
            _vp_table, _vp_tmin, _vp_dinv = build_vprime_table(
                T, _phi_lo, _phi_hi, VPRIME_TABLE_SIZE
            )
            _vp_table_mid, _vp_tmin_mid, _vp_dinv_mid = build_vprime_table(
                T_mid, _phi_lo, _phi_hi, VPRIME_TABLE_SIZE
            )
            _vp_thi = _phi_hi
            _vp_table_T_last = T

    if USE_OVERDAMPED and USE_VPRIME_TABLE:
        if DISABLE_THERMAL_NOISE:
            _od_ns = 0.0
        else:
            _od_ns = np.sqrt(2.0 * T * dt / (eta_eff * mu**2 * dx_phys**3))
        overdamped_euler_step_table(
            phi,
            dt,
            dx,
            eta_eff,
            mu,
            inv_a2,
            _vp_table,
            _vp_tmin,
            _vp_dinv,
            VPRIME_TABLE_SIZE,
            _od_ns,
            n,
        )
    elif USE_BAOAB and USE_VPRIME_TABLE:
        if DISABLE_THERMAL_NOISE:
            _baoab_ns = 0.0
        else:
            _baoab_ns = np.sqrt(T / (mu**2 * dx_phys**3))
        baoab_step_table(
            phi,
            pi,
            dt,
            dx,
            eta_eff,
            mu,
            inv_a2,
            _vp_table,
            _vp_tmin,
            _vp_dinv,
            VPRIME_TABLE_SIZE,
            _baoab_ns,
            n,
        )
    else:
        _ns_val = (
            0.0
            if DISABLE_THERMAL_NOISE
            else np.sqrt(2.0 * eta_eff * T * dt / (mu**2 * dx_phys**3))
        )

        if USE_SINGLE_PASS_RK2 and not USE_VPRIME_TABLE:
            rk2_fused_single_pass(
                phi,
                pi,
                dt,
                dx,
                eta_eff,
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
                _ns_val,
                nb.uint64(n),
                phi_mid,
                pi_mid,
                inv_a2,
            )
        else:
            if DISABLE_THERMAL_NOISE:
                noise[:] = 0.0
            else:
                if USE_NUMBA_RNG:
                    generate_noise_field(noise, _ns_val, n)
                else:
                    noise[:] = np.random.randn(Nx, Ny, Nz) * _ns_val

            if USE_VPRIME_TABLE and USE_NONFUSED_TABLE_RK2:
                rk2_step_table(
                    phi,
                    pi,
                    dt,
                    dx,
                    eta_eff,
                    mu,
                    noise,
                    phi_mid,
                    pi_mid,
                    inv_a2,
                    _vp_table,
                    _vp_tmin,
                    _vp_dinv,
                    VPRIME_TABLE_SIZE,
                )
            elif USE_NONFUSED_INLINE_RK2:
                rk2_step_inline(
                    phi,
                    pi,
                    dt,
                    dx,
                    eta_eff,
                    T,
                    mu,
                    lam,
                    mphi,
                    bosonMassSquared,
                    bosonCoupling,
                    bosonGaugeCoupling,
                    fermionCoupling,
                    fermionGaugeCoupling,
                    noise,
                    phi_mid,
                    pi_mid,
                    inv_a2,
                )
            elif USE_VPRIME_TABLE and USE_SINGLE_PASS_RK2:
                rk2_fused_table(
                    phi,
                    pi,
                    dt,
                    dx,
                    eta_eff,
                    mu,
                    noise,
                    phi_mid,
                    pi_mid,
                    inv_a2,
                    _vp_table,
                    _vp_tmin,
                    _vp_dinv,
                    VPRIME_TABLE_SIZE,
                    _vp_table_mid,
                    _vp_tmin_mid,
                    _vp_dinv_mid,
                    VPRIME_TABLE_SIZE,
                )
            elif USE_FUSED_RK2:
                rk2_step_fused(
                    phi,
                    pi,
                    dt,
                    dx,
                    eta_eff,
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
                    noise,
                    lap_tmp,
                    Vp_tmp,
                    phi_mid,
                    pi_mid,
                    inv_a2,
                )
            else:
                half_noise = 0.5 * noise
                rk2_step(
                    phi,
                    pi,
                    0.5 * dt,
                    dx,
                    eta_eff,
                    T,
                    mu,
                    lam,
                    mphi,
                    bosonMassSquared,
                    bosonCoupling,
                    bosonGaugeCoupling,
                    fermionCoupling,
                    fermionGaugeCoupling,
                    half_noise,
                    lap_tmp,
                    Vp_tmp,
                    phi_mid,
                    pi_mid,
                    inv_a2,
                )
                rk2_step(
                    phi,
                    pi,
                    0.5 * dt,
                    dx,
                    eta_eff,
                    T_mid,
                    mu,
                    lam,
                    mphi,
                    bosonMassSquared,
                    bosonCoupling,
                    bosonGaugeCoupling,
                    fermionCoupling,
                    fermionGaugeCoupling,
                    half_noise,
                    lap_tmp,
                    Vp_tmp,
                    phi_mid,
                    pi_mid,
                    inv_a2,
                )

    # NaN / divergence guard (more frequent in replay mode)
    _nan_check_interval = 1000 if REPLAY_MODE else 100000
    if n % _nan_check_interval == 0:
        has_nan = np.any(np.isnan(phi))
        has_inf = np.any(np.isinf(phi))
        if not USE_OVERDAMPED:
            has_nan = has_nan or np.any(np.isnan(pi))
            has_inf = has_inf or np.any(np.isinf(pi))
        phi_absmax = (
            np.max(np.abs(phi[np.isfinite(phi)])) if np.any(np.isfinite(phi)) else 0.0
        )
        diverging = phi_absmax > 1e6
        if has_nan or has_inf or diverging:
            print(f"\n{'!'*60}")
            tag = "NaN" if has_nan else ("Inf" if has_inf else "DIVERGENCE")
            print(f"{tag} DETECTED at step {n} (t_phys={t/mu:.6e})")
            nan_phi = int(np.sum(np.isnan(phi)))
            print(f"  NaN sites: phi={nan_phi}/{phi.size}")
            print(f"  |phi|_max = {phi_absmax:.4e}")
            if nan_phi < phi.size:
                valid = phi[~np.isnan(phi)]
                print(f"  Finite phi: [{valid.min():.4e}, {valid.max():.4e}]")
            if not USE_OVERDAMPED:
                nan_pi = int(np.sum(np.isnan(pi)))
                print(f"  NaN sites: pi={nan_pi}/{pi.size}")
                if nan_pi < pi.size:
                    valid_pi = pi[~np.isnan(pi)]
                    print(f"  Finite pi:  [{valid_pi.min():.4e}, {valid_pi.max():.4e}]")
            print(f"  noise_scale = {noise_scale:.6e}")
            print(f"  eta_eff = {eta_eff:.6e}, T = {T:.1f}, inv_a2 = {inv_a2:.10f}")
            print(f"{'!'*60}")
            laplacian_periodic(lap_tmp, phi, dx)
            Vprime_field(
                Vp_tmp,
                phi,
                T,
                lam,
                mphi,
                bosonMassSquared,
                bosonCoupling,
                bosonGaugeCoupling,
                fermionCoupling,
                fermionGaugeCoupling,
            )
            state_file = f"{state_path}/state_step_{n:010d}_NaN_debug.npz"
            _nan_save = dict(
                phi=phi,
                laplacian=lap_tmp,
                Vprime=Vp_tmp,
                step=n,
                time=t,
                temperature=T,
                noise_scale=noise_scale,
                eta_eff=eta_eff,
                inv_a2=inv_a2,
            )
            if not USE_OVERDAMPED:
                _nan_save["pi"] = pi
                _nan_save["noise"] = noise
            np.savez_compressed(state_file, **_nan_save)
            print(f"  Debug state saved: {state_file}")
            print("  ABORTING simulation.")
            break

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
    _should_save = (
        (n - n_start) % _cur_steps == 0 if REPLAY_MODE else n % _cur_steps == 0
    )
    if _should_save:
        elapsed = time.time() - t_start
        done = n - n_start + 1
        steps_per_sec = done / elapsed if elapsed > 0 else 0
        time_per_step = elapsed / done if done > 0 else 0
        eta_total = (Nt - n) * time_per_step
        ms_per_step = time_per_step * 1000
        eta_min = eta_total / 60
        if HUBBLE_EXPANSION:
            print(
                f"Step {n}/{Nt} | t={t/mu:.2e} | T={T:.1f} | "
                f"a={a_current:.6f} | H={H_now:.2e} | "
                f"{steps_per_sec:.1f} steps/s | "
                f"{ms_per_step:.2f} ms/step | ETA: {eta_min:.1f} min"
            )
        else:
            print(
                f"Step {n}/{Nt} | t={t/mu:.2e} | T={T:.1f} | "
                f"{steps_per_sec:.1f} steps/s | "
                f"{ms_per_step:.2f} ms/step | ETA: {eta_min:.1f} min"
            )

        # Save field state for later revisualization / checkpointing
        state_file = f"{state_path}/state_step_{n:010d}.npz"
        _is_checkpoint = _total_saves % CHECKPOINT_EVERY == 0
        _total_saves += 1
        save_dict = dict(
            phi=phi,
            step=n,
            time=t,
            temperature=T,
            phi_min=phi.min(),
            phi_max=phi.max(),
        )
        if _is_checkpoint and not USE_OVERDAMPED:
            save_dict["pi"] = pi
        if HUBBLE_EXPANSION:
            save_dict["scale_factor"] = a_current
            save_dict["hubble"] = H_now
        np.savez_compressed(state_file, **save_dict)

        # Create snapshot image
        if REPLAY_MODE:
            _t_phys = t / mu
            _title = (
                f"Step {n:,} | t={_t_phys:.4f} | T={T:.1f}\n"
                f"$\\phi$: [{phi.min():.2e}, {phi.max():.2e}]"
            )
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            _slices = [
                (phi[:, :, Nz // 2], "XY mid-plane (z={})".format(Nz // 2)),
                (phi[:, Ny // 2, :], "XZ mid-plane (y={})".format(Ny // 2)),
                (phi[Nx // 2, :, :], "YZ mid-plane (x={})".format(Nx // 2)),
            ]
            for ax, (sl, label) in zip(axes, _slices):
                im = ax.imshow(
                    sl, origin="lower", cmap="coolwarm", vmin=-2e11, vmax=2e11
                )
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(label)
            fig.suptitle(_title, fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{fig_path}/t_{_t_phys:.6f}.png", dpi=150)
            plt.close(fig)
        else:
            phi_slice = phi[:, :, Nz // 2]
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(
                phi_slice, origin="lower", cmap="coolwarm", vmin=-2e11, vmax=2e11
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\phi$")
            if HUBBLE_EXPANSION:
                ax.set_title(
                    f"Step {n:,} | t={t/mu:.2e} | T={T:.1f}\n"
                    f"$\\phi$ range: [{phi.min():.2e}, {phi.max():.2e}]"
                )
            else:
                ax.set_title(f"t={t/mu:.2e}, T={T:.1f}")
            fig.tight_layout()
            fig.savefig(f"{fig_path}/t_{str(t/mu)}.png")
            plt.close(fig)

t_end = time.time()
total_time = t_end - t_start
print("\n" + "=" * 60)
if REPLAY_MODE:
    print("REPLAY FINISHED!")
else:
    print("SIMULATION FINISHED!")
print("=" * 60)
print("\nTiming:")
print(f"  Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
steps_run = Nt - n_start
print(f"  Average: {steps_run/total_time:.1f} steps/second")
print(f"  Time per step: {total_time*1000/max(steps_run,1):.3f} ms")

if REPLAY_MODE:
    _n_saved = steps_run // steps + (1 if steps_run % steps == 0 else 0)
    print(f"\nReplay Summary:")
    print(f"  Step range: {n_start} -> {Nt}")
    print(f"  Save interval: every {steps} steps")
    print(f"  Snapshots saved: ~{_n_saved}")
    print(f"  Output: {save_path}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
else:
    print("\nSimulation Parameters:")
    print(f"  Grid: {Nx}×{Ny}×{Nz}")
    print(f"  Total steps: {Nt:,}")
    print(f"  Initial temperature: {T0}")
    print(f"  Cooling rate: {cooling_rate}")
    print(f"  dx_phys: {dx_phys}")
    print(f"  dt_phys: {dt_phys}")

    print("\nOptimizations Used:")
    if USE_OVERDAMPED:
        print(f"  Integrator: OVERDAMPED Heun (first-order, 2 passes/step)")
    elif USE_BAOAB and USE_VPRIME_TABLE:
        print(f"  Integrator: BAOAB (2 passes/step, cache-blocked)")
    else:
        print(f"  Single-pass fused RK2: {USE_SINGLE_PASS_RK2}")
        print(f"  Fused RK2: {USE_FUSED_RK2}")
    print(f"  Vprime table: {USE_VPRIME_TABLE} (size={VPRIME_TABLE_SIZE})")
    print(f"  In-kernel RNG: {USE_NUMBA_RNG}")
    print(f"  Field precision: {field_dtype.__name__}")
    print(f"  Spline resolution (N_Y): {N_Y}")
    print(f"  Numba threads: {nb.get_num_threads()}")

    print("\nOutput:")
    print(f"  Snapshots saved: {save_path}")
    print(f"  Snapshot interval: every {steps} steps")
    print(f"  Total snapshots: {Nt // steps}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save simulation metadata
    metadata_file = f"{save_path}/simulation_metadata.npz"
    np.savez(
        metadata_file,
        # Grid parameters
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
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
        # Bubble seeding info
        seed_bubbles=SEED_BUBBLES,
        bubble_config=np.array(BUBBLE_CONFIG if SEED_BUBBLES else []),
        bubble_profile=BUBBLE_PROFILE if SEED_BUBBLES else "none",
        bubble_wall_width=BUBBLE_WALL_WIDTH if SEED_BUBBLES else 0.0,
        integrator=INTEGRATOR_NAME,
        counterterm=cli_args.counterterm,
        potential_type=cli_args.potential_type,
    )
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"Field states saved to: {state_path}/")
    print("Use postprocess/revisualize_snapshots.py to replot with different settings")
    print("=" * 60)

# =====================================================
# Optional: quick validation vs exact CTFT for dJ and V' at a few points
# =====================================================
if __name__ == "__main__" and False:
    # Toggle to True to run a brief validation (slow due to CTFT calls).
    xs = np.linspace(0.0, YMAX, 200)
    dJb_ref = np.array([CTFT.dJb_exact(x) for x in xs])
    dJf_ref = np.array([CTFT.dJf_exact(x) for x in xs])
    dJb_new = np.array(
        [
            cubic_eval_uniform(
                max(min(x, x_min + h_y * nseg - 1e-12), x_min),
                x_min,
                h_y,
                nseg,
                c0_b,
                c1_b,
                c2_b,
                c3_b,
            )
            for x in xs
        ]
    )
    dJf_new = np.array(
        [
            cubic_eval_uniform(
                max(min(x, x_min + h_y * nseg - 1e-12), x_min),
                x_min,
                h_y,
                nseg,
                c0_f,
                c1_f,
                c2_f,
                c3_f,
            )
            for x in xs
        ]
    )
    eps = 1e-12
    print(
        "dJb max abs/rel err:",
        np.max(np.abs(dJb_new - dJb_ref)),
        np.max(np.abs(dJb_new - dJb_ref) / (np.abs(dJb_ref) + eps)),
    )
    print(
        "dJf max abs/rel err:",
        np.max(np.abs(dJf_new - dJf_ref)),
        np.max(np.abs(dJf_new - dJf_ref) / (np.abs(dJf_ref) + eps)),
    )
