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
parser = argparse.ArgumentParser(description="Complex scalar field lattice simulation (cosmic strings + domain walls)")
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
    "--zn_order",
    type=int,
    default=0,
    help="Z_N symmetry breaking order (0 = pure U(1), N>0 adds cos(N*theta) term)",
)
parser.add_argument(
    "--zn_strength",
    type=float,
    default=0.0,
    help="Strength delta_V of the Z_N breaking term (GeV^4)",
)
parser.add_argument(
    "--zn_turn_on_T",
    type=float,
    default=0.0,
    help="Temperature below which Z_N breaking activates (0 = always on if zn_order>0)",
)
parser.add_argument(
    "--init_rho",
    type=float,
    default=0.01,
    help="Initial radial fluctuation amplitude for complex field (default: 0.01)",
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
_POT_COEFFS = np.array([2.0, -1.0], dtype=np.float64)
if cli_args.potential_type == "V_correct":
    _POT_COEFFS[0] = 1.0
    _POT_COEFFS[1] = 1.0
elif cli_args.potential_type == "fermion_only":
    _POT_COEFFS[0] = 0.0
    _POT_COEFFS[1] = 1.0

# =====================================================
# Z_N breaking parameters (Numba reads at runtime from global array)
# =====================================================
# _ZN_PARAMS[0] = N (order), _ZN_PARAMS[1] = delta_V (strength), _ZN_PARAMS[2] = active flag
_ZN_PARAMS = np.array([float(cli_args.zn_order), cli_args.zn_strength, 0.0], dtype=np.float64)
if cli_args.zn_order > 0 and cli_args.zn_turn_on_T <= 0.0:
    _ZN_PARAMS[2] = 1.0  # always on

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
    f"\nPotential type: {cli_args.potential_type}  (coeffs: boson={_POT_COEFFS[0]}, fermion={_POT_COEFFS[1]})"
)
print(f"Complex field simulation (two-component: phi1 + i*phi2)")
if cli_args.zn_order > 0:
    print(f"Z_{cli_args.zn_order} breaking: delta_V={cli_args.zn_strength:.2e}, turn_on_T={cli_args.zn_turn_on_T:.1f}")
else:
    print("Pure U(1) symmetry (no Z_N breaking)")
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

# Initialize complex fields (3D): phi1 = Re(Phi), phi2 = Im(Phi)
_init_rho = cli_args.init_rho
_init_theta = 2.0 * np.pi * np.random.rand(Nx, Ny, Nz)
_init_r = _init_rho * np.abs(np.random.randn(Nx, Ny, Nz))
phi1 = (_init_r * np.cos(_init_theta)).astype(field_dtype)
phi2 = (_init_r * np.sin(_init_theta)).astype(field_dtype)
pi1 = np.zeros((Nx, Ny, Nz), dtype=field_dtype)
pi2 = np.zeros((Nx, Ny, Nz), dtype=field_dtype)
del _init_theta, _init_r

# Apply bubble seeding if enabled (skip when resuming from checkpoint)
# For complex field: seed into phi1 (radial direction), phi2 stays near zero
if SEED_BUBBLES and not resuming:
    print("\n" + "=" * 60)
    print("BUBBLE SEEDING ENABLED (into radial mode phi1)")
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

    bounce_scale_param = bounce_R_scale if BUBBLE_PROFILE == "bounce" else 1.0

    phi1 = seed_multiple_bubbles(
        phi1,
        BUBBLE_CONFIG,
        vev,
        profile=BUBBLE_PROFILE,
        wall_width=BUBBLE_WALL_WIDTH,
        bounce_interp=bounce_profile_interp,
        bounce_scale=bounce_scale_param,
    )
    print("-" * 60)
    print(f"phi1 after seeding: min={phi1.min():.4e}, max={phi1.max():.4e}")
    print(f"phi2 (unchanged): min={phi2.min():.4e}, max={phi2.max():.4e}")
    _rho_seeded = np.sqrt(phi1**2 + phi2**2)
    print(f"|Phi| range: [{_rho_seeded.min():.4e}, {_rho_seeded.max():.4e}]")
    del _rho_seeded
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
    out1, out2,
    phi1_arr, phi2_arr,
    T,
    lam,
    mphi,
    bosonMassSquared,
    bosonCoupling,
    bosonGaugeCoupling,
    fermionCoupling,
    fermionGaugeCoupling,
):
    """Compute V'(phi1, phi2) for entire 3D complex field."""
    nx, ny, nz = phi1_arr.shape
    T2 = T * T
    T4 = T2 * T2
    pref = T4 / (2.0 * math.pi * math.pi)
    gb2 = bosonCoupling * bosonCoupling
    gg2 = bosonGaugeCoupling * bosonGaugeCoupling
    gf2 = fermionCoupling * fermionCoupling
    gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
    coef_b_T = 0.25 * gb2 + (2.0 / 3.0) * gg2
    mphi_sq = mphi * mphi
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                p1 = phi1_arr[i, j, k]
                p2 = phi2_arr[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                inv_rho = 1.0 / rho_safe
                dV = lam * rho_safe * rho_safe * rho_safe - mphi_sq * rho_safe
                xb_sq = bosonMassSquared + 0.5 * gb2 * rho_safe * rho_safe + coef_b_T * T2
                xb = 0.0
                if xb_sq > 0.0:
                    xb = math.sqrt(xb_sq) / T
                xf_sq = 0.5 * gf2 * rho_safe * rho_safe + (1.0 / 6.0) * gfg2 * T2
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
                dJb = cubic_eval_uniform(xb_clamped, x_min, h_y, nseg, c0_b, c1_b, c2_b, c3_b)
                dJf = cubic_eval_uniform(xf_clamped, x_min, h_y, nseg, c0_f, c1_f, c2_f, c3_f)
                dxb_dphi = 0.5 * gb2 * rho_safe / (T2 * max(xb, 1e-20))
                dxf_dphi = 0.5 * gf2 * rho_safe / (T2 * max(xf, 1e-20))
                dV += pref * (_POT_COEFFS[0] * dJb * dxb_dphi + _POT_COEFFS[1] * dJf * dxf_dphi)
                dV1 = dV * p1 * inv_rho
                dV2 = dV * p2 * inv_rho
                zn_n = _ZN_PARAMS[0]; zn_dv = _ZN_PARAMS[1]; zn_active = _ZN_PARAMS[2]
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = inv_rho * inv_rho
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                out1[i, j, k] = dV1
                out2[i, j, k] = dV2


@nb.njit(cache=True)
def rk2_step(
    phi1, phi2, pi1, pi2,
    dt, dx, eta_eff, T, mu, lam, mphi,
    bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
    fermionCoupling, fermionGaugeCoupling,
    noise1, noise2, lap1, lap2, Vp1, Vp2,
    phi1_mid_buf, phi2_mid_buf, pi1_mid_buf, pi2_mid_buf,
    inv_a2,
):
    mu2 = mu * mu
    laplacian_periodic(lap1, phi1, dx)
    laplacian_periodic(lap2, phi2, dx)
    Vprime_field(Vp1, Vp2, phi1, phi2, T, lam, mphi, bosonMassSquared,
                 bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
    k1_pi1 = inv_a2 * lap1 - eta_eff * pi1 - Vp1 / mu2
    k1_pi2 = inv_a2 * lap2 - eta_eff * pi2 - Vp2 / mu2
    phi1_mid_buf[:,:,:] = phi1 + 0.5 * dt * pi1
    phi2_mid_buf[:,:,:] = phi2 + 0.5 * dt * pi2
    pi1_mid_buf[:,:,:] = pi1 + 0.5 * dt * k1_pi1
    pi2_mid_buf[:,:,:] = pi2 + 0.5 * dt * k1_pi2
    laplacian_periodic(lap1, phi1_mid_buf, dx)
    laplacian_periodic(lap2, phi2_mid_buf, dx)
    Vprime_field(Vp1, Vp2, phi1_mid_buf, phi2_mid_buf, T, lam, mphi, bosonMassSquared,
                 bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
    k2_pi1 = inv_a2 * lap1 - eta_eff * pi1_mid_buf - Vp1 / mu2
    k2_pi2 = inv_a2 * lap2 - eta_eff * pi2_mid_buf - Vp2 / mu2
    phi1 += dt * pi1_mid_buf
    phi2 += dt * pi2_mid_buf
    pi1 += dt * k2_pi1 + noise1
    pi2 += dt * k2_pi2 + noise2


@nb.njit(cache=True)
def rk2_step_fused(
    phi1, phi2, pi1, pi2,
    dt, dx, eta_eff, T, T_mid, mu, lam, mphi,
    bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
    fermionCoupling, fermionGaugeCoupling,
    noise1, noise2, lap1, lap2, Vp1, Vp2,
    phi1_mid_buf, phi2_mid_buf, pi1_mid_buf, pi2_mid_buf,
    inv_a2,
):
    """Fused RK2 for complex field: two half-steps."""
    half_dt = 0.5 * dt
    half_noise1 = 0.5 * noise1; half_noise2 = 0.5 * noise2
    mu2 = mu * mu
    # First half: k1
    laplacian_periodic(lap1, phi1, dx); laplacian_periodic(lap2, phi2, dx)
    Vprime_field(Vp1, Vp2, phi1, phi2, T, lam, mphi, bosonMassSquared,
                 bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
    k1_pi1 = inv_a2 * lap1 - eta_eff * pi1 - Vp1 / mu2
    k1_pi2 = inv_a2 * lap2 - eta_eff * pi2 - Vp2 / mu2
    phi1_t = phi1 + half_dt * pi1; phi2_t = phi2 + half_dt * pi2
    pi1_t = pi1 + half_dt * k1_pi1; pi2_t = pi2 + half_dt * k1_pi2
    # First half: k2
    laplacian_periodic(lap1, phi1_t, dx); laplacian_periodic(lap2, phi2_t, dx)
    Vprime_field(Vp1, Vp2, phi1_t, phi2_t, T, lam, mphi, bosonMassSquared,
                 bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
    k2_pi1 = inv_a2 * lap1 - eta_eff * pi1_t - Vp1 / mu2
    k2_pi2 = inv_a2 * lap2 - eta_eff * pi2_t - Vp2 / mu2
    phi1 += half_dt * pi1_t; phi2 += half_dt * pi2_t
    pi1 += half_dt * k2_pi1 + half_noise1; pi2 += half_dt * k2_pi2 + half_noise2
    # Second half: k1
    laplacian_periodic(lap1, phi1, dx); laplacian_periodic(lap2, phi2, dx)
    Vprime_field(Vp1, Vp2, phi1, phi2, T_mid, lam, mphi, bosonMassSquared,
                 bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
    k1_pi1 = inv_a2 * lap1 - eta_eff * pi1 - Vp1 / mu2
    k1_pi2 = inv_a2 * lap2 - eta_eff * pi2 - Vp2 / mu2
    phi1_mid_buf[:,:,:] = phi1 + half_dt * pi1; phi2_mid_buf[:,:,:] = phi2 + half_dt * pi2
    pi1_mid_buf[:,:,:] = pi1 + half_dt * k1_pi1; pi2_mid_buf[:,:,:] = pi2 + half_dt * k1_pi2
    # Second half: k2
    laplacian_periodic(lap1, phi1_mid_buf, dx); laplacian_periodic(lap2, phi2_mid_buf, dx)
    Vprime_field(Vp1, Vp2, phi1_mid_buf, phi2_mid_buf, T_mid, lam, mphi, bosonMassSquared,
                 bosonCoupling, bosonGaugeCoupling, fermionCoupling, fermionGaugeCoupling)
    k2_pi1 = inv_a2 * lap1 - eta_eff * pi1_mid_buf - Vp1 / mu2
    k2_pi2 = inv_a2 * lap2 - eta_eff * pi2_mid_buf - Vp2 / mu2
    phi1 += half_dt * pi1_mid_buf; phi2 += half_dt * pi2_mid_buf
    pi1 += half_dt * k2_pi1 + half_noise1; pi2 += half_dt * k2_pi2 + half_noise2


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
    phi1,
    phi2,
    pi1,
    pi2,
    dt,
    dx,
    eta_eff,
    mu,
    noise1,
    noise2,
    phi1_tmp,
    phi2_tmp,
    pi1_tmp,
    pi2_tmp,
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
    """Fully fused RK2 with V'(rho) evaluated via pre-built lookup table.

    Complex field: phi1=Re(Phi), phi2=Im(Phi). Potential V depends on rho=sqrt(phi1^2+phi2^2).
    Table lookup uses rho; dV1 = V'(rho)*phi1/rho, dV2 = V'(rho)*phi2/rho.
    Two separate tables: T (first half-step) and T_mid (second half-step).
    """
    nx, ny, nz = phi1.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    last_T = vp_npts_T - 2
    last_Tm = vp_npts_Tm - 2
    zn_n = _ZN_PARAMS[0]
    zn_dv = _ZN_PARAMS[1]
    zn_active = _ZN_PARAMS[2]

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
                p1 = phi1[i, j, k]
                p2 = phi2[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                pv1 = pi1[i, j, k]
                pv2 = pi2[i, j, k]
                lap1 = (
                    phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                    + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                    - 6.0 * p1
                ) * inv_dx2
                lap2 = (
                    phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                    + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                    - 6.0 * p2
                ) * inv_dx2
                fidx = (rho - vp_tmin_T) * vp_dinv_T
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_T:
                    idx = last_T
                frac = fidx - idx
                dV_drho = vp_table_T[idx] + frac * (vp_table_T[idx + 1] - vp_table_T[idx])
                dV1 = dV_drho * p1 / rho_safe
                dV2 = dV_drho * p2 / rho_safe
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                phi1_tmp[i, j, k] = p1 + half_dt * pv1
                phi2_tmp[i, j, k] = p2 + half_dt * pv2
                pi1_tmp[i, j, k] = pv1 + half_dt * k_pi1
                pi2_tmp[i, j, k] = pv2 + half_dt * k_pi2

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
                p1 = phi1_tmp[i, j, k]
                p2 = phi2_tmp[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                pv1 = pi1_tmp[i, j, k]
                pv2 = pi2_tmp[i, j, k]
                lap1 = (
                    phi1_tmp[ip, j, k] + phi1_tmp[im, j, k] + phi1_tmp[i, jp, k]
                    + phi1_tmp[i, jm, k] + phi1_tmp[i, j, kp] + phi1_tmp[i, j, km]
                    - 6.0 * p1
                ) * inv_dx2
                lap2 = (
                    phi2_tmp[ip, j, k] + phi2_tmp[im, j, k] + phi2_tmp[i, jp, k]
                    + phi2_tmp[i, jm, k] + phi2_tmp[i, j, kp] + phi2_tmp[i, j, km]
                    - 6.0 * p2
                ) * inv_dx2
                fidx = (rho - vp_tmin_T) * vp_dinv_T
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_T:
                    idx = last_T
                frac = fidx - idx
                dV_drho = vp_table_T[idx] + frac * (vp_table_T[idx + 1] - vp_table_T[idx])
                dV1 = dV_drho * p1 / rho_safe
                dV2 = dV_drho * p2 / rho_safe
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                phi1[i, j, k] += half_dt * pv1
                phi2[i, j, k] += half_dt * pv2
                pi1[i, j, k] += half_dt * k_pi1 + 0.5 * noise1[i, j, k]
                pi2[i, j, k] += half_dt * k_pi2 + 0.5 * noise2[i, j, k]

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
                p1 = phi1[i, j, k]
                p2 = phi2[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                pv1 = pi1[i, j, k]
                pv2 = pi2[i, j, k]
                lap1 = (
                    phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                    + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                    - 6.0 * p1
                ) * inv_dx2
                lap2 = (
                    phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                    + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                    - 6.0 * p2
                ) * inv_dx2
                fidx = (rho - vp_tmin_Tm) * vp_dinv_Tm
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_Tm:
                    idx = last_Tm
                frac = fidx - idx
                dV_drho = vp_table_Tm[idx] + frac * (vp_table_Tm[idx + 1] - vp_table_Tm[idx])
                dV1 = dV_drho * p1 / rho_safe
                dV2 = dV_drho * p2 / rho_safe
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                phi1_tmp[i, j, k] = p1 + half_dt * pv1
                phi2_tmp[i, j, k] = p2 + half_dt * pv2
                pi1_tmp[i, j, k] = pv1 + half_dt * k_pi1
                pi2_tmp[i, j, k] = pv2 + half_dt * k_pi2

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
                p1 = phi1_tmp[i, j, k]
                p2 = phi2_tmp[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                pv1 = pi1_tmp[i, j, k]
                pv2 = pi2_tmp[i, j, k]
                lap1 = (
                    phi1_tmp[ip, j, k] + phi1_tmp[im, j, k] + phi1_tmp[i, jp, k]
                    + phi1_tmp[i, jm, k] + phi1_tmp[i, j, kp] + phi1_tmp[i, j, km]
                    - 6.0 * p1
                ) * inv_dx2
                lap2 = (
                    phi2_tmp[ip, j, k] + phi2_tmp[im, j, k] + phi2_tmp[i, jp, k]
                    + phi2_tmp[i, jm, k] + phi2_tmp[i, j, kp] + phi2_tmp[i, j, km]
                    - 6.0 * p2
                ) * inv_dx2
                fidx = (rho - vp_tmin_Tm) * vp_dinv_Tm
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last_Tm:
                    idx = last_Tm
                frac = fidx - idx
                dV_drho = vp_table_Tm[idx] + frac * (vp_table_Tm[idx + 1] - vp_table_Tm[idx])
                dV1 = dV_drho * p1 / rho_safe
                dV2 = dV_drho * p2 / rho_safe
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                phi1[i, j, k] += half_dt * pv1
                phi2[i, j, k] += half_dt * pv2
                pi1[i, j, k] += half_dt * k_pi1 + 0.5 * noise1[i, j, k]
                pi2[i, j, k] += half_dt * k_pi2 + 0.5 * noise2[i, j, k]


@nb.njit(parallel=True, fastmath=True, cache=True)
def rk2_step_table(
    phi1,
    phi2,
    pi1,
    pi2,
    dt,
    dx,
    eta_eff,
    mu,
    noise1,
    noise2,
    phi1_tmp,
    phi2_tmp,
    pi1_tmp,
    pi2_tmp,
    inv_a2,
    vp_table,
    vp_tmin,
    vp_dinv,
    vp_npts,
):
    """Non-fused RK2: single full step of size dt with V'(rho) lookup table.

    Complex field: phi1=Re(Phi), phi2=Im(Phi). V depends on rho=sqrt(phi1^2+phi2^2).
    Only 2 grid passes (vs 4 for fused).
    """
    nx, ny, nz = phi1.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    last = vp_npts - 2
    zn_n = _ZN_PARAMS[0]
    zn_dv = _ZN_PARAMS[1]
    zn_active = _ZN_PARAMS[2]

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
                p1 = phi1[i, j, k]
                p2 = phi2[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                pv1 = pi1[i, j, k]
                pv2 = pi2[i, j, k]
                lap1 = (
                    phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                    + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                    - 6.0 * p1
                ) * inv_dx2
                lap2 = (
                    phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                    + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                    - 6.0 * p2
                ) * inv_dx2
                fidx = (rho - vp_tmin) * vp_dinv
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last:
                    idx = last
                frac = fidx - idx
                dV_drho = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                dV1 = dV_drho * p1 / rho_safe
                dV2 = dV_drho * p2 / rho_safe
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                phi1_tmp[i, j, k] = p1 + half_dt * pv1
                phi2_tmp[i, j, k] = p2 + half_dt * pv2
                pi1_tmp[i, j, k] = pv1 + half_dt * k_pi1
                pi2_tmp[i, j, k] = pv2 + half_dt * k_pi2

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
                p1 = phi1_tmp[i, j, k]
                p2 = phi2_tmp[i, j, k]
                rho = math.sqrt(p1 * p1 + p2 * p2)
                rho_safe = max(rho, 1e-30)
                pv1 = pi1_tmp[i, j, k]
                pv2 = pi2_tmp[i, j, k]
                lap1 = (
                    phi1_tmp[ip, j, k] + phi1_tmp[im, j, k] + phi1_tmp[i, jp, k]
                    + phi1_tmp[i, jm, k] + phi1_tmp[i, j, kp] + phi1_tmp[i, j, km]
                    - 6.0 * p1
                ) * inv_dx2
                lap2 = (
                    phi2_tmp[ip, j, k] + phi2_tmp[im, j, k] + phi2_tmp[i, jp, k]
                    + phi2_tmp[i, jm, k] + phi2_tmp[i, j, kp] + phi2_tmp[i, j, km]
                    - 6.0 * p2
                ) * inv_dx2
                fidx = (rho - vp_tmin) * vp_dinv
                idx = int(fidx)
                if idx < 0:
                    idx = 0
                elif idx > last:
                    idx = last
                frac = fidx - idx
                dV_drho = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                dV1 = dV_drho * p1 / rho_safe
                dV2 = dV_drho * p2 / rho_safe
                if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                    theta = math.atan2(p2, p1)
                    sin_ntheta = math.sin(zn_n * theta)
                    inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                    dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                    dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                phi1[i, j, k] += dt * pv1
                phi2[i, j, k] += dt * pv2
                pi1[i, j, k] += dt * k_pi1 + noise1[i, j, k]
                pi2[i, j, k] += dt * k_pi2 + noise2[i, j, k]


BAOAB_TILE_J = 16
BAOAB_TILE_K = 16


@nb.njit(parallel=True, fastmath=True, cache=True)
def baoab_step_table(
    phi1,
    phi2,
    pi1,
    pi2,
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
    """BAOAB Langevin integrator with table-based V'(rho) and cache-blocking.

    Complex field: phi1=Re(Phi), phi2=Im(Phi). V depends on rho=sqrt(phi1^2+phi2^2).

    BAOAB splitting:
      B: pi1, pi2 += (dt/2) * F1, F2   [half-kick]
      A: phi1, phi2 += (dt/2) * pi1, pi2 [half-drift]
      O: pi1, pi2 = c1*pi + c2*noise   [exact OU thermostat, independent noise]
      A: phi1, phi2 += (dt/2) * pi1, pi2 [half-drift]
      B: pi1, pi2 += (dt/2) * F1, F2   [half-kick]

    where F = (1/a^2)*lap(phi) - dV/mu^2 (projected from V'(rho)).
    """
    nx, ny, nz = phi1.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    last = vp_npts - 2
    zn_n = _ZN_PARAMS[0]
    zn_dv = _ZN_PARAMS[1]
    zn_active = _ZN_PARAMS[2]

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
                        p1 = phi1[i, j, k]
                        p2 = phi2[i, j, k]
                        rho = math.sqrt(p1 * p1 + p2 * p2)
                        rho_safe = max(rho, 1e-30)
                        lap1 = (
                            phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                            + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                            - 6.0 * p1
                        ) * inv_dx2
                        lap2 = (
                            phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                            + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                            - 6.0 * p2
                        ) * inv_dx2
                        fidx = (rho - vp_tmin) * vp_dinv
                        idx = int(fidx)
                        if idx < 0:
                            idx = 0
                        elif idx > last:
                            idx = last
                        frac = fidx - idx
                        dV_drho = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                        dV1 = dV_drho * p1 / rho_safe
                        dV2 = dV_drho * p2 / rho_safe
                        if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                            theta = math.atan2(p2, p1)
                            sin_ntheta = math.sin(zn_n * theta)
                            inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                            dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                            dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                        force1 = inv_a2 * lap1 - dV1 * inv_mu2
                        force2 = inv_a2 * lap2 - dV2 * inv_mu2
                        pi_v1 = pi1[i, j, k] + half_dt * force1
                        pi_v2 = pi2[i, j, k] + half_dt * force2
                        phi1[i, j, k] = p1 + half_dt * pi_v1
                        phi2[i, j, k] = p2 + half_dt * pi_v2
                        pi1[i, j, k] = pi_v1
                        pi2[i, j, k] = pi_v2

    # ---- O step: exact Ornstein-Uhlenbeck thermostat (independent noise per component) ----
    np.random.seed(seed)
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                u1 = np.random.random()
                u2 = np.random.random()
                if u1 < 1e-12:
                    u1 = 1e-12
                z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                z2 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                pi1[i, j, k] = c1 * pi1[i, j, k] + c2 * z1
                pi2[i, j, k] = c1 * pi2[i, j, k] + c2 * z2

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
                        pi_v1 = pi1[i, j, k]
                        pi_v2 = pi2[i, j, k]
                        p1 = phi1[i, j, k] + half_dt * pi_v1
                        p2 = phi2[i, j, k] + half_dt * pi_v2
                        phi1[i, j, k] = p1
                        phi2[i, j, k] = p2
                        rho = math.sqrt(p1 * p1 + p2 * p2)
                        rho_safe = max(rho, 1e-30)
                        lap1 = (
                            phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                            + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                            - 6.0 * p1
                        ) * inv_dx2
                        lap2 = (
                            phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                            + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                            - 6.0 * p2
                        ) * inv_dx2
                        fidx = (rho - vp_tmin) * vp_dinv
                        idx = int(fidx)
                        if idx < 0:
                            idx = 0
                        elif idx > last:
                            idx = last
                        frac = fidx - idx
                        dV_drho = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                        dV1 = dV_drho * p1 / rho_safe
                        dV2 = dV_drho * p2 / rho_safe
                        if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                            theta = math.atan2(p2, p1)
                            sin_ntheta = math.sin(zn_n * theta)
                            inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                            dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                            dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                        force1 = inv_a2 * lap1 - dV1 * inv_mu2
                        force2 = inv_a2 * lap2 - dV2 * inv_mu2
                        pi1[i, j, k] = pi_v1 + half_dt * force1
                        pi2[i, j, k] = pi_v2 + half_dt * force2


OVERDAMPED_TILE_J = 16
OVERDAMPED_TILE_K = 16


@nb.njit(parallel=True, fastmath=True, cache=True)
def overdamped_euler_step_table(
    phi1,
    phi2,
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

    Complex field: phi1=Re(Phi), phi2=Im(Phi). V depends on rho=sqrt(phi1^2+phi2^2).

    Solves:  eta_eff * dphi/dt = (1/a^2)*lap(phi) - dV/mu^2 + xi

    Euler-Maruyama update for both components with independent noise.
    """
    nx, ny, nz = phi1.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    dt_over_eta = dt / eta_eff
    last = vp_npts - 2
    zn_n = _ZN_PARAMS[0]
    zn_dv = _ZN_PARAMS[1]
    zn_active = _ZN_PARAMS[2]

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
                        p1 = phi1[i, j, k]
                        p2 = phi2[i, j, k]
                        rho = math.sqrt(p1 * p1 + p2 * p2)
                        rho_safe = max(rho, 1e-30)
                        lap1 = (
                            phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                            + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                            - 6.0 * p1
                        ) * inv_dx2
                        lap2 = (
                            phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                            + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                            - 6.0 * p2
                        ) * inv_dx2
                        fidx = (rho - vp_tmin) * vp_dinv
                        idx = int(fidx)
                        if idx < 0:
                            idx = 0
                        elif idx > last:
                            idx = last
                        frac = fidx - idx
                        dV_drho = vp_table[idx] + frac * (vp_table[idx + 1] - vp_table[idx])
                        dV1 = dV_drho * p1 / rho_safe
                        dV2 = dV_drho * p2 / rho_safe
                        if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
                            theta = math.atan2(p2, p1)
                            sin_ntheta = math.sin(zn_n * theta)
                            inv_rho2 = (1.0 / rho_safe) * (1.0 / rho_safe)
                            dV1 += -zn_n * zn_dv * sin_ntheta * p2 * inv_rho2
                            dV2 += zn_n * zn_dv * sin_ntheta * p1 * inv_rho2
                        force1 = inv_a2 * lap1 - dV1 * inv_mu2
                        force2 = inv_a2 * lap2 - dV2 * inv_mu2
                        u1 = np.random.random()
                        u2 = np.random.random()
                        if u1 < 1e-12:
                            u1 = 1e-12
                        z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                        z2 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
                        phi1[i, j, k] = p1 + dt_over_eta * force1 + noise_scale * z1
                        phi2[i, j, k] = p2 + dt_over_eta * force2 + noise_scale * z2


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
def _vprime_complex_inline(
    ph1, ph2,
    T_val, T2, pref, lam_val, mphi_sq, bms,
    gb2, gf2, gg2, gfg2, coef_b,
    _x_min, _h_y, _inv_hy, _xhi, _nseg,
    _c0_b, _c1_b, _c2_b, _c3_b,
    _c0_f, _c1_f, _c2_f, _c3_f,
):
    """V'(phi1, phi2) for complex field: radial V'(rho) projected + optional Z_N breaking."""
    rho = math.sqrt(ph1 * ph1 + ph2 * ph2)
    rho_safe = max(rho, 1e-30)

    dV_drho = _vprime_inline(
        rho_safe, T_val, T2, pref, lam_val, mphi_sq, bms,
        gb2, gf2, gg2, gfg2, coef_b,
        _x_min, _h_y, _inv_hy, _xhi, _nseg,
        _c0_b, _c1_b, _c2_b, _c3_b,
        _c0_f, _c1_f, _c2_f, _c3_f,
    )

    inv_rho = 1.0 / rho_safe
    dV1 = dV_drho * ph1 * inv_rho
    dV2 = dV_drho * ph2 * inv_rho

    zn_n = _ZN_PARAMS[0]
    zn_dv = _ZN_PARAMS[1]
    zn_active = _ZN_PARAMS[2]
    if zn_active > 0.5 and zn_n > 0.5 and rho > 1e-20:
        theta = math.atan2(ph2, ph1)
        sin_ntheta = math.sin(zn_n * theta)
        inv_rho2 = inv_rho * inv_rho
        dV1 += -zn_n * zn_dv * sin_ntheta * ph2 * inv_rho2
        dV2 += zn_n * zn_dv * sin_ntheta * ph1 * inv_rho2

    return dV1, dV2


@nb.njit(parallel=True, fastmath=True, cache=True)
def compute_winding_number(phi1, phi2, winding_out):
    """Compute winding number density for cosmic string detection.

    For each site, sums the phase winding around plaquettes in XY, XZ, YZ planes.
    A winding of |W| = 1 indicates a cosmic string passing through.
    """
    nx, ny, nz = phi1.shape
    for i in nb.prange(nx):
        ip = i + 1 if i + 1 < nx else 0
        for j in range(ny):
            jp = j + 1 if j + 1 < ny else 0
            for k in range(nz):
                kp = k + 1 if k + 1 < nz else 0
                w_total = 0.0

                # XY plaquette: (i,j,k) -> (ip,j,k) -> (ip,jp,k) -> (i,jp,k)
                th00 = math.atan2(phi2[i, j, k], phi1[i, j, k])
                th10 = math.atan2(phi2[ip, j, k], phi1[ip, j, k])
                th11 = math.atan2(phi2[ip, jp, k], phi1[ip, jp, k])
                th01 = math.atan2(phi2[i, jp, k], phi1[i, jp, k])
                d1 = th10 - th00
                d2 = th11 - th10
                d3 = th01 - th11
                d4 = th00 - th01
                d1 = math.atan2(math.sin(d1), math.cos(d1))
                d2 = math.atan2(math.sin(d2), math.cos(d2))
                d3 = math.atan2(math.sin(d3), math.cos(d3))
                d4 = math.atan2(math.sin(d4), math.cos(d4))
                w_total += (d1 + d2 + d3 + d4) / (2.0 * math.pi)

                # XZ plaquette
                th0k = th00
                th1k = th10
                th1kp = math.atan2(phi2[ip, j, kp], phi1[ip, j, kp])
                th0kp = math.atan2(phi2[i, j, kp], phi1[i, j, kp])
                d1 = th1k - th0k
                d2 = th1kp - th1k
                d3 = th0kp - th1kp
                d4 = th0k - th0kp
                d1 = math.atan2(math.sin(d1), math.cos(d1))
                d2 = math.atan2(math.sin(d2), math.cos(d2))
                d3 = math.atan2(math.sin(d3), math.cos(d3))
                d4 = math.atan2(math.sin(d4), math.cos(d4))
                w_total += (d1 + d2 + d3 + d4) / (2.0 * math.pi)

                # YZ plaquette
                th_j0 = th00
                th_j1 = th01
                th_j1kp = math.atan2(phi2[i, jp, kp], phi1[i, jp, kp])
                th_j0kp = th0kp
                d1 = th_j1 - th_j0
                d2 = th_j1kp - th_j1
                d3 = th_j0kp - th_j1kp
                d4 = th_j0 - th_j0kp
                d1 = math.atan2(math.sin(d1), math.cos(d1))
                d2 = math.atan2(math.sin(d2), math.cos(d2))
                d3 = math.atan2(math.sin(d3), math.cos(d3))
                d4 = math.atan2(math.sin(d4), math.cos(d4))
                w_total += (d1 + d2 + d3 + d4) / (2.0 * math.pi)

                winding_out[i, j, k] = w_total


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
    phi1, phi2, pi1, pi2,
    dt, dx, eta_eff, T, T_mid, mu, lam, mphi,
    bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
    fermionCoupling, fermionGaugeCoupling,
    noise_scale, noise_seed,
    phi1_tmp, phi2_tmp, pi1_tmp, pi2_tmp,
    inv_a2,
):
    """Fully fused RK2 for complex field with inline noise generation."""
    nx, ny, nz = phi1.shape
    inv_dx2 = 1.0 / (dx * dx)
    inv_mu2 = 1.0 / (mu * mu)
    half_dt = 0.5 * dt
    mphi_sq = mphi * mphi
    gb2 = bosonCoupling * bosonCoupling
    gg2 = bosonGaugeCoupling * bosonGaugeCoupling
    gf2 = fermionCoupling * fermionCoupling
    gfg2 = fermionGaugeCoupling * fermionGaugeCoupling
    coef_b = 0.25 * gb2 + (2.0 / 3.0) * gg2
    _xm = x_min; _hy = h_y; _ihy = inv_hy; _xhi = xhi_clamp; _ns = nseg
    _c0b = c0_b; _c1b = c1_b; _c2b = c2_b; _c3b = c3_b
    _c0f = c0_f; _c1f = c1_f; _c2f = c2_f; _c3f = c3_f
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
                        p1 = phi1[i, j, k]; p2 = phi2[i, j, k]
                        pv1 = pi1[i, j, k]; pv2 = pi2[i, j, k]
                        lap1 = (phi1[ip,j,k]+phi1[im,j,k]+phi1[i,jp,k]+phi1[i,jm,k]+phi1[i,j,kp]+phi1[i,j,km]-6.0*p1)*inv_dx2
                        lap2 = (phi2[ip,j,k]+phi2[im,j,k]+phi2[i,jp,k]+phi2[i,jm,k]+phi2[i,j,kp]+phi2[i,j,km]-6.0*p2)*inv_dx2
                        dV1, dV2 = _vprime_complex_inline(
                            p1, p2, T, T2, pref, lam, mphi_sq, bosonMassSquared,
                            gb2, gf2, gg2, gfg2, coef_b,
                            _xm, _hy, _ihy, _xhi, _ns,
                            _c0b, _c1b, _c2b, _c3b, _c0f, _c1f, _c2f, _c3f)
                        kp1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                        kp2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                        phi1_tmp[i,j,k] = p1 + half_dt * pv1
                        phi2_tmp[i,j,k] = p2 + half_dt * pv2
                        pi1_tmp[i,j,k] = pv1 + half_dt * kp1
                        pi2_tmp[i,j,k] = pv2 + half_dt * kp2

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
                        p1 = phi1_tmp[i,j,k]; p2 = phi2_tmp[i,j,k]
                        pv1 = pi1_tmp[i,j,k]; pv2 = pi2_tmp[i,j,k]
                        lap1 = (phi1_tmp[ip,j,k]+phi1_tmp[im,j,k]+phi1_tmp[i,jp,k]+phi1_tmp[i,jm,k]+phi1_tmp[i,j,kp]+phi1_tmp[i,j,km]-6.0*p1)*inv_dx2
                        lap2 = (phi2_tmp[ip,j,k]+phi2_tmp[im,j,k]+phi2_tmp[i,jp,k]+phi2_tmp[i,jm,k]+phi2_tmp[i,j,kp]+phi2_tmp[i,j,km]-6.0*p2)*inv_dx2
                        dV1, dV2 = _vprime_complex_inline(
                            p1, p2, T, T2, pref, lam, mphi_sq, bosonMassSquared,
                            gb2, gf2, gg2, gfg2, coef_b,
                            _xm, _hy, _ihy, _xhi, _ns,
                            _c0b, _c1b, _c2b, _c3b, _c0f, _c1f, _c2f, _c3f)
                        kp1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                        kp2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                        _site = nb.int64(i) * nyz + nb.int64(j) * nb.int64(nz) + nb.int64(k)
                        _seed = nb.uint64(_site * nb.int64(73856093)) ^ nb.uint64(noise_seed)
                        _u1, _u2 = _hash_rng_pair(_seed)
                        _z1 = math.sqrt(-2.0 * math.log(_u1)) * math.cos(6.283185307179586 * _u2)
                        _z2 = math.sqrt(-2.0 * math.log(_u1)) * math.sin(6.283185307179586 * _u2)
                        phi1[i,j,k] += half_dt * pv1
                        phi2[i,j,k] += half_dt * pv2
                        pi1[i,j,k] += half_dt * kp1 + 0.5 * noise_scale * _z1
                        pi2[i,j,k] += half_dt * kp2 + 0.5 * noise_scale * _z2

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
                        p1 = phi1[i,j,k]; p2 = phi2[i,j,k]
                        pv1 = pi1[i,j,k]; pv2 = pi2[i,j,k]
                        lap1 = (phi1[ip,j,k]+phi1[im,j,k]+phi1[i,jp,k]+phi1[i,jm,k]+phi1[i,j,kp]+phi1[i,j,km]-6.0*p1)*inv_dx2
                        lap2 = (phi2[ip,j,k]+phi2[im,j,k]+phi2[i,jp,k]+phi2[i,jm,k]+phi2[i,j,kp]+phi2[i,j,km]-6.0*p2)*inv_dx2
                        dV1, dV2 = _vprime_complex_inline(
                            p1, p2, T_mid, T2m, prefm, lam, mphi_sq, bosonMassSquared,
                            gb2, gf2, gg2, gfg2, coef_b,
                            _xm, _hy, _ihy, _xhi, _ns,
                            _c0b, _c1b, _c2b, _c3b, _c0f, _c1f, _c2f, _c3f)
                        kp1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                        kp2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                        phi1_tmp[i,j,k] = p1 + half_dt * pv1
                        phi2_tmp[i,j,k] = p2 + half_dt * pv2
                        pi1_tmp[i,j,k] = pv1 + half_dt * kp1
                        pi2_tmp[i,j,k] = pv2 + half_dt * kp2

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
                        p1 = phi1_tmp[i,j,k]; p2 = phi2_tmp[i,j,k]
                        pv1 = pi1_tmp[i,j,k]; pv2 = pi2_tmp[i,j,k]
                        lap1 = (phi1_tmp[ip,j,k]+phi1_tmp[im,j,k]+phi1_tmp[i,jp,k]+phi1_tmp[i,jm,k]+phi1_tmp[i,j,kp]+phi1_tmp[i,j,km]-6.0*p1)*inv_dx2
                        lap2 = (phi2_tmp[ip,j,k]+phi2_tmp[im,j,k]+phi2_tmp[i,jp,k]+phi2_tmp[i,jm,k]+phi2_tmp[i,j,kp]+phi2_tmp[i,j,km]-6.0*p2)*inv_dx2
                        dV1, dV2 = _vprime_complex_inline(
                            p1, p2, T_mid, T2m, prefm, lam, mphi_sq, bosonMassSquared,
                            gb2, gf2, gg2, gfg2, coef_b,
                            _xm, _hy, _ihy, _xhi, _ns,
                            _c0b, _c1b, _c2b, _c3b, _c0f, _c1f, _c2f, _c3f)
                        kp1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                        kp2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                        _site = nb.int64(i) * nyz + nb.int64(j) * nb.int64(nz) + nb.int64(k)
                        _seed = nb.uint64(_site * nb.int64(19349669)) ^ nb.uint64(noise_seed)
                        _u1, _u2 = _hash_rng_pair(_seed)
                        _z1 = math.sqrt(-2.0 * math.log(_u1)) * math.cos(6.283185307179586 * _u2)
                        _z2 = math.sqrt(-2.0 * math.log(_u1)) * math.sin(6.283185307179586 * _u2)
                        phi1[i,j,k] += half_dt * pv1
                        phi2[i,j,k] += half_dt * pv2
                        pi1[i,j,k] += half_dt * kp1 + 0.5 * noise_scale * _z1
                        pi2[i,j,k] += half_dt * kp2 + 0.5 * noise_scale * _z2


RK2_NF_TILE_J = 16
RK2_NF_TILE_K = 16


@nb.njit(parallel=True, fastmath=True, cache=True)
def rk2_step_inline(
    phi1,
    phi2,
    pi1,
    pi2,
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
    noise1,
    noise2,
    phi1_tmp,
    phi2_tmp,
    pi1_tmp,
    pi2_tmp,
    inv_a2,
):
    """Non-fused RK2 with inline V'(phi1,phi2): single full step, 2 passes only.

    Complex field: phi1=Re(Phi), phi2=Im(Phi). Uses _vprime_complex_inline.
    Same midpoint-method (Heun) as rk2_step_table. Cache-blocking for L1/L2 locality.
    """
    nx, ny, nz = phi1.shape
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
                        p1 = phi1[i, j, k]
                        p2 = phi2[i, j, k]
                        pv1 = pi1[i, j, k]
                        pv2 = pi2[i, j, k]
                        lap1 = (
                            phi1[ip, j, k] + phi1[im, j, k] + phi1[i, jp, k]
                            + phi1[i, jm, k] + phi1[i, j, kp] + phi1[i, j, km]
                            - 6.0 * p1
                        ) * inv_dx2
                        lap2 = (
                            phi2[ip, j, k] + phi2[im, j, k] + phi2[i, jp, k]
                            + phi2[i, jm, k] + phi2[i, j, kp] + phi2[i, j, km]
                            - 6.0 * p2
                        ) * inv_dx2
                        dV1, dV2 = _vprime_complex_inline(
                            p1, p2,
                            T, T2, pref, lam_val, mphi_sq, bosonMassSquared,
                            gb2, gf2, gg2, gfg2, coef_b,
                            _xm, _hy, _ihy, _xhi, _ns,
                            _c0b, _c1b, _c2b, _c3b, _c0f, _c1f, _c2f, _c3f,
                        )
                        k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                        k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                        phi1_tmp[i, j, k] = p1 + half_dt * pv1
                        phi2_tmp[i, j, k] = p2 + half_dt * pv2
                        pi1_tmp[i, j, k] = pv1 + half_dt * k_pi1
                        pi2_tmp[i, j, k] = pv2 + half_dt * k_pi2

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
                        p1 = phi1_tmp[i, j, k]
                        p2 = phi2_tmp[i, j, k]
                        pv1 = pi1_tmp[i, j, k]
                        pv2 = pi2_tmp[i, j, k]
                        lap1 = (
                            phi1_tmp[ip, j, k] + phi1_tmp[im, j, k] + phi1_tmp[i, jp, k]
                            + phi1_tmp[i, jm, k] + phi1_tmp[i, j, kp] + phi1_tmp[i, j, km]
                            - 6.0 * p1
                        ) * inv_dx2
                        lap2 = (
                            phi2_tmp[ip, j, k] + phi2_tmp[im, j, k] + phi2_tmp[i, jp, k]
                            + phi2_tmp[i, jm, k] + phi2_tmp[i, j, kp] + phi2_tmp[i, j, km]
                            - 6.0 * p2
                        ) * inv_dx2
                        dV1, dV2 = _vprime_complex_inline(
                            p1, p2,
                            T, T2, pref, lam_val, mphi_sq, bosonMassSquared,
                            gb2, gf2, gg2, gfg2, coef_b,
                            _xm, _hy, _ihy, _xhi, _ns,
                            _c0b, _c1b, _c2b, _c3b, _c0f, _c1f, _c2f, _c3f,
                        )
                        k_pi1 = inv_a2 * lap1 - eta_eff * pv1 - dV1 * inv_mu2
                        k_pi2 = inv_a2 * lap2 - eta_eff * pv2 - dV2 * inv_mu2
                        phi1[i, j, k] += dt * pv1
                        phi2[i, j, k] += dt * pv2
                        pi1[i, j, k] += dt * k_pi1 + noise1[i, j, k]
                        pi2[i, j, k] += dt * k_pi2 + noise2[i, j, k]


# =====================================================
# Temperature schedule
# =====================================================
def temperature(t):
    return max(T0 - cooling_rate * t, 0.0)


# =====================================================
# Output
# =====================================================
param_set = "set6"
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
zn_tag = f"_ZN{cli_args.zn_order}" if cli_args.zn_order > 0 else ""
save_path = (
    f"data/lattice/{param_set}/{Nx}x{Ny}x{Nz}_T0_{int(T0)}_complex"
    f"{zn_tag}{dx_tag}{dtphys_tag}_interval_{steps}_3D{hubble_tag}{eta_tag}{coupling_tag}{overdamped_tag}{integrator_tag}{counterterm_tag}{pot_type_tag}{seed_tag}"
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
    _has_complex = "phi1" in ckpt_data
    _ckpt_p1 = ckpt_data["phi1"] if _has_complex else ckpt_data["phi"]
    if np.any(np.isnan(_ckpt_p1)):
        print(f"ERROR: Checkpoint {_replay_ckpt_file} phi contains NaN. Cannot replay.")
        sys.exit(1)
    if not USE_OVERDAMPED:
        _need_pi = "pi1" if _has_complex else "pi"
        if _need_pi not in ckpt_data:
            print(f"ERROR: {_replay_ckpt_file} has no '{_need_pi}' (lightweight snapshot).")
            print("  Replay requires a full checkpoint. Use a file with momentum.")
            sys.exit(1)
        _ckpt_pv1 = ckpt_data[_need_pi]
        if np.any(np.isnan(_ckpt_pv1)):
            print(f"ERROR: Checkpoint {_replay_ckpt_file} pi contains NaN. Cannot replay.")
            sys.exit(1)

    resuming = True
    n_start = _replay_start
    phi1 = _ckpt_p1.astype(field_dtype)
    phi2 = ckpt_data["phi2"].astype(field_dtype) if "phi2" in ckpt_data else np.zeros_like(phi1)
    if not USE_OVERDAMPED:
        pi1 = _ckpt_pv1.astype(field_dtype)
        pi2 = ckpt_data["pi2"].astype(field_dtype) if "pi2" in ckpt_data else np.zeros_like(pi1)
    if HUBBLE_EXPANSION and "scale_factor" in ckpt_data:
        a_current = float(ckpt_data["scale_factor"])

    print(f"\n{'='*60}")
    print("REPLAY MODE - DETAILED BUBBLE DYNAMICS (COMPLEX FIELD)")
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
    _rho_ckpt = np.sqrt(phi1**2 + phi2**2)
    print(f"  |Phi| range: [{_rho_ckpt.min():.4e}, {_rho_ckpt.max():.4e}]")
    del _rho_ckpt
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
            if "phi1" not in _d and "phi" not in _d:
                continue
            if not USE_OVERDAMPED and "pi1" not in _d and "pi" not in _d:
                continue
            _ckpt_phi1 = _d["phi1"] if "phi1" in _d else _d["phi"]
            if np.any(np.isnan(_ckpt_phi1)):
                print(f"  Skipping {latest_ckpt} (contains NaN)")
                continue
            if not USE_OVERDAMPED:
                _ckpt_pi1 = _d["pi1"] if "pi1" in _d else _d["pi"]
                if np.any(np.isnan(_ckpt_pi1)):
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
            phi1 = (ckpt_data["phi1"] if "phi1" in ckpt_data else ckpt_data["phi"]).astype(field_dtype)
            phi2 = (ckpt_data["phi2"] if "phi2" in ckpt_data else np.zeros_like(phi1)).astype(field_dtype)
            if not USE_OVERDAMPED:
                pi1 = (ckpt_data["pi1"] if "pi1" in ckpt_data else ckpt_data["pi"]).astype(field_dtype)
                pi2 = (ckpt_data["pi2"] if "pi2" in ckpt_data else np.zeros_like(pi1)).astype(field_dtype)
            if HUBBLE_EXPANSION and "scale_factor" in ckpt_data:
                a_current = float(ckpt_data["scale_factor"])
            print(f"\n{'='*60}")
            print("RESUMING FROM CHECKPOINT (COMPLEX FIELD)")
            print(f"{'='*60}")
            print(f"  File: {latest_ckpt}")
            print(f"  Step: {ckpt_step}  ->  resuming from step {n_start}")
            print(f"  Time: {float(ckpt_data['time'])/mu:.6e} (physical)")
            print(f"  Temperature: {float(ckpt_data['temperature']):.1f}")
            _rho_ckpt = np.sqrt(phi1**2 + phi2**2)
            print(f"  |Phi| range: [{_rho_ckpt.min():.4e}, {_rho_ckpt.max():.4e}]")
            del _rho_ckpt
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
lap_tmp2 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
Vp_tmp = np.empty((Nx, Ny, Nz), dtype=field_dtype)
Vp_tmp2 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
phi1_mid = np.empty((Nx, Ny, Nz), dtype=field_dtype)
phi2_mid = np.empty((Nx, Ny, Nz), dtype=field_dtype)
pi1_mid = np.empty((Nx, Ny, Nz), dtype=field_dtype)
pi2_mid = np.empty((Nx, Ny, Nz), dtype=field_dtype)
noise = np.empty((Nx, Ny, Nz), dtype=field_dtype)
noise2 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
winding_buf = np.empty((Nx, Ny, Nz), dtype=np.float32)

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
        phi1,
        phi2,
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
        phi1,
        phi2,
        pi1,
        pi2,
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
    if USE_NUMBA_RNG:
        generate_noise_field(noise, warmup_scale, 0)
        generate_noise_field(noise2, warmup_scale, 1)
    else:
        noise[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
        noise2[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
    rk2_step_table(
        phi1,
        phi2,
        pi1,
        pi2,
        dt,
        dx,
        eta,
        mu,
        noise,
        noise2,
        phi1_mid,
        phi2_mid,
        pi1_mid,
        pi2_mid,
        1.0,
        _warmup_tbl,
        _warmup_tmin,
        _warmup_dinv,
        VPRIME_TABLE_SIZE,
    )
elif USE_NONFUSED_INLINE_RK2:
    if USE_NUMBA_RNG:
        generate_noise_field(noise, warmup_scale, 0)
        generate_noise_field(noise2, warmup_scale, 1)
    else:
        noise[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
        noise2[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
    rk2_step_inline(
        phi1,
        phi2,
        pi1,
        pi2,
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
        noise,
        noise2,
        phi1_mid,
        phi2_mid,
        pi1_mid,
        pi2_mid,
        1.0,
    )
elif USE_VPRIME_TABLE and USE_SINGLE_PASS_RK2:
    if USE_NUMBA_RNG:
        generate_noise_field(noise, warmup_scale, 0)
        generate_noise_field(noise2, warmup_scale, 1)
    else:
        noise[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
        noise2[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
    rk2_fused_table(
        phi1,
        phi2,
        pi1,
        pi2,
        dt,
        dx,
        eta,
        mu,
        noise,
        noise2,
        phi1_mid,
        phi2_mid,
        pi1_mid,
        pi2_mid,
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
        phi1,
        phi2,
        pi1,
        pi2,
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
        phi1_mid,
        phi2_mid,
        pi1_mid,
        pi2_mid,
        1.0,
    )
elif USE_FUSED_RK2:
    warmup_noise2 = np.empty_like(warmup_noise)
    if USE_NUMBA_RNG:
        generate_noise_field(warmup_noise2, warmup_scale, 1)
    else:
        warmup_noise2[:] = (np.random.randn(Nx, Ny, Nz) * warmup_scale).astype(field_dtype)
    rk2_step_fused(
        phi1, phi2, pi1, pi2,
        dt, dx, eta, T0, T0, mu, lam, mphi,
        bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
        fermionCoupling, fermionGaugeCoupling,
        warmup_noise, warmup_noise2,
        lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
        phi1_mid, phi2_mid, pi1_mid, pi2_mid,
        1.0,
    )
else:
    if USE_NUMBA_RNG:
        generate_noise_field(noise, warmup_scale, 0)
        generate_noise_field(noise2, warmup_scale, 1)
    else:
        noise[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
        noise2[:] = np.random.randn(Nx, Ny, Nz) * warmup_scale
    rk2_step(
        phi1,
        phi2,
        pi1,
        pi2,
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
        noise * 0.5,
        noise2 * 0.5,
        lap_tmp,
        lap_tmp2,
        Vp_tmp,
        Vp_tmp2,
        phi1_mid,
        phi2_mid,
        pi1_mid,
        pi2_mid,
        1.0,
    )
print("JIT warmup complete.")

# =====================================================
# Force imbalance diagnostic helper
# =====================================================


def plot_force_imbalance(phi1_arr, phi2_arr, label_suffix, save_name):
    """Compute and plot force imbalance for complex field (uses phi1 component for display)."""
    diag_lap1 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    diag_lap2 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    diag_Vp1 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    diag_Vp2 = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    laplacian_periodic(diag_lap1, phi1_arr, dx)
    laplacian_periodic(diag_lap2, phi2_arr, dx)
    Vprime_field(diag_Vp1, diag_Vp2, phi1_arr, phi2_arr, T0, lam, mphi,
                 bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                 fermionCoupling, fermionGaugeCoupling)
    F1 = diag_lap1 - diag_Vp1 / (mu * mu)
    F2 = diag_lap2 - diag_Vp2 / (mu * mu)
    F_mag = np.sqrt(F1**2 + F2**2)
    rho_arr = np.sqrt(phi1_arr**2 + phi2_arr**2)

    max_F = np.max(F_mag)
    max_loc = np.unravel_index(np.argmax(F_mag), F_mag.shape)
    print(f"  Force imbalance {label_suffix}:")
    print(f"    max|F| = {max_F:.4e}  at lattice site {max_loc}")
    print(f"    |Phi| range: [{rho_arr.min():.4e}, {rho_arr.max():.4e}]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    zmid = phi1_arr.shape[2] // 2
    im0 = axes[0].imshow(np.asarray(rho_arr[:, :, zmid], dtype=np.float64).T,
                         origin="lower", cmap="viridis")
    axes[0].set_title(f"|Phi| (z={zmid}) {label_suffix}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(np.asarray(F_mag[:, :, zmid], dtype=np.float64).T,
                         origin="lower", cmap="hot")
    axes[1].set_title(f"|F| = |lap - V'/mu^2| {label_suffix}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f"max|F| = {max_F:.4e}  at {max_loc}", fontsize=11)
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
    plot_force_imbalance(phi1, phi2, "(before relaxation)", "initial_force_imbalance.png")
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
    _gf_w_lap = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    _gf_w_Vp = np.empty((Nx, Ny, Nz), dtype=field_dtype)
    _gf_w_phi = phi1.copy()
    laplacian_periodic(_gf_w_lap, _gf_w_phi, dx)
    gradient_flow_update(_gf_w_phi, _gf_w_lap, _gf_w_Vp, GF_DT, mu, gf_weight)
    del _gf_w_lap, _gf_w_Vp, _gf_w_phi
    print("  JIT warmup for gradient_flow_update complete.")

    gf_snap_path = f"{fig_path}/gradient_flow_snapshots"
    os.makedirs(gf_snap_path, exist_ok=True)
    print(f"  Snapshots will be saved to: {gf_snap_path}")

    if GF_SAVE_EVERY > 0:
        plot_force_imbalance(phi1, phi2, "(GF iter 0)", f"gradient_flow_snapshots/gf_iter_0000.png")

    gf_start = time.time()
    gf_converged = False
    gf_sig_mask = gf_weight > 0.1
    gf_early_iters = {1, 10, 50, 100, 200}
    for gf_iter in range(GF_MAX_ITER):
        laplacian_periodic(lap_tmp, phi1, dx)
        laplacian_periodic(lap_tmp2, phi2, dx)
        Vprime_field(Vp_tmp, Vp_tmp2, phi1, phi2, T0, lam, mphi,
                     bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                     fermionCoupling, fermionGaugeCoupling)
        force1 = lap_tmp - Vp_tmp / (mu * mu)
        force2 = lap_tmp2 - Vp_tmp2 / (mu * mu)
        force_mag = np.sqrt(force1**2 + force2**2)
        weighted_delta = gf_weight * force_mag * GF_DT
        sig_deltas = weighted_delta[gf_sig_mask]
        max_delta = float(np.max(sig_deltas)) if sig_deltas.size > 0 else 0.0

        gradient_flow_update(phi1, lap_tmp, Vp_tmp, GF_DT, mu, gf_weight)
        gradient_flow_update(phi2, lap_tmp2, Vp_tmp2, GF_DT, mu, gf_weight)

        rho_gf = np.sqrt(phi1**2 + phi2**2)
        max_rho = float(np.max(rho_gf))
        rel_change = max_delta / max(max_rho, 1e-30)

        if (gf_iter + 1) % GF_PRINT_EVERY == 0:
            print(f"  iter {gf_iter+1:6d}: max|dphi|={max_delta:.2e}, rel={rel_change:.2e}, |Phi| range=[{rho_gf.min():.2e}, {rho_gf.max():.2e}]")

        save_now = GF_SAVE_EVERY > 0 and ((gf_iter + 1) % GF_SAVE_EVERY == 0 or (gf_iter + 1) in gf_early_iters)
        if save_now:
            plot_force_imbalance(phi1, phi2, f"(GF iter {gf_iter+1})", f"gradient_flow_snapshots/gf_iter_{gf_iter+1:04d}.png")

        if rel_change < GF_TOL:
            print(f"  Converged at iter {gf_iter+1}! rel_change={rel_change:.2e} < tol={GF_TOL:.0e}")
            gf_converged = True
            if GF_SAVE_EVERY > 0:
                plot_force_imbalance(phi1, phi2, f"(GF iter {gf_iter+1}, converged)", f"gradient_flow_snapshots/gf_iter_{gf_iter+1:04d}_converged.png")
            break

    if not gf_converged:
        print(f"  WARNING: Did not converge in {GF_MAX_ITER} iterations")
        print(f"  Final rel_change = {rel_change:.2e} (tol = {GF_TOL:.0e})")

    gf_time = time.time() - gf_start
    print(f"  Gradient flow took {gf_time:.2f}s ({gf_iter+1} iterations)")

    pi1[:] = 0.0; pi2[:] = 0.0
    print("  Momentum reset to zero.")
    print("=" * 60 + "\n")

    print("-" * 60)
    print("FORCE IMBALANCE DIAGNOSTIC (after gradient flow)")
    print("-" * 60)
    plot_force_imbalance(phi1, phi2, "(after gradient flow)", "initial_force_imbalance_after_gf.png")
    print("-" * 60 + "\n")

# =====================================================
# Settling period (optional): let bubble relax with high damping
# NOTE: Gradient flow relaxation (above) is preferred. This settling
# approach uses the RK2 integrator with tiny dt and is ~25,000x slower.
# =====================================================
if SEED_BUBBLES and SETTLING_ENABLED and not resuming:
    print(f"\nRunning settling ({SETTLING_STEPS} steps, η={SETTLING_ETA})...")
    settling_eta = SETTLING_ETA / mu
    settling_noise1 = np.zeros((Nx, Ny, Nz), dtype=field_dtype)
    settling_noise2 = np.zeros((Nx, Ny, Nz), dtype=field_dtype)
    T_settle = T0

    for s in range(SETTLING_STEPS):
        if USE_FUSED_RK2:
            rk2_step_fused(
                phi1, phi2, pi1, pi2, dt, dx, settling_eta, T_settle, T_settle, mu, lam, mphi,
                bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                fermionCoupling, fermionGaugeCoupling,
                settling_noise1, settling_noise2, lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
                phi1_mid, phi2_mid, pi1_mid, pi2_mid, 1.0,
            )
        else:
            rk2_step(
                phi1, phi2, pi1, pi2, 0.5 * dt, dx, settling_eta, T_settle, mu, lam, mphi,
                bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                fermionCoupling, fermionGaugeCoupling,
                settling_noise1, settling_noise2, lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
                phi1_mid, phi2_mid, pi1_mid, pi2_mid, 1.0,
            )
            rk2_step(
                phi1, phi2, pi1, pi2, 0.5 * dt, dx, settling_eta, T_settle, mu, lam, mphi,
                bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                fermionCoupling, fermionGaugeCoupling,
                settling_noise1, settling_noise2, lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
                phi1_mid, phi2_mid, pi1_mid, pi2_mid, 1.0,
            )
        if (s + 1) % 2000 == 0:
            _rho_s = np.sqrt(phi1**2 + phi2**2)
            print(f"  Settling step {s+1}/{SETTLING_STEPS}, |Phi| range: [{_rho_s.min():.2e}, {_rho_s.max():.2e}]")

    _rho_s = np.sqrt(phi1**2 + phi2**2)
    print(f"Settling complete. |Phi|: [{_rho_s.min():.2e}, {_rho_s.max():.2e}]")
    pi1[:] = 0.0; pi2[:] = 0.0
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

    # Runtime Z_N activation based on temperature
    if cli_args.zn_order > 0 and cli_args.zn_turn_on_T > 0.0:
        if T < cli_args.zn_turn_on_T and _ZN_PARAMS[2] < 0.5:
            _ZN_PARAMS[2] = 1.0
            print(f"\n*** Z_{cli_args.zn_order} breaking activated at T={T:.1f} < {cli_args.zn_turn_on_T:.1f} ***\n")

    # Rebuild V'(rho) table when T changes or rho exceeds table range
    if USE_VPRIME_TABLE:
        _need_rebuild = _vp_table is None
        if not _need_rebuild:
            _need_rebuild = abs(T - _vp_table_T_last) / max(abs(T), 1.0) > 1e-4
        if not _need_rebuild:
            _rho_max = float(np.max(np.sqrt(phi1**2 + phi2**2)))
            _cur_lo = 0.0
            _cur_hi = _rho_max
            _need_rebuild = _cur_lo < _vp_tmin or _cur_hi > _vp_thi
        if _need_rebuild:
            _rho_max = float(np.max(np.sqrt(phi1**2 + phi2**2)))
            _cur_lo = 0.0
            _cur_hi = _rho_max
            _range = max(_cur_hi - _cur_lo, 1.0)
            _margin = max(_range * _TABLE_MARGIN_FRAC, 20000.0)
            _phi_lo = max(0.0, _cur_lo - _margin)
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
            phi1,
            phi2,
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
            phi1,
            phi2,
            pi1,
            pi2,
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
                phi1, phi2, pi1, pi2,
                dt, dx, eta_eff, T, T_mid, mu, lam, mphi,
                bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                fermionCoupling, fermionGaugeCoupling,
                _ns_val, nb.uint64(n),
                phi1_mid, phi2_mid, pi1_mid, pi2_mid,
                inv_a2,
            )
        else:
            if DISABLE_THERMAL_NOISE:
                noise[:] = 0.0
                noise2[:] = 0.0
            else:
                if USE_NUMBA_RNG:
                    generate_noise_field(noise, _ns_val, n)
                    generate_noise_field(noise2, _ns_val, n + 1)
                else:
                    noise[:] = np.random.randn(Nx, Ny, Nz) * _ns_val
                    noise2[:] = np.random.randn(Nx, Ny, Nz) * _ns_val

            if USE_VPRIME_TABLE and USE_NONFUSED_TABLE_RK2:
                rk2_step_table(
                    phi1,
                    phi2,
                    pi1,
                    pi2,
                    dt,
                    dx,
                    eta_eff,
                    mu,
                    noise,
                    noise2,
                    phi1_mid,
                    phi2_mid,
                    pi1_mid,
                    pi2_mid,
                    inv_a2,
                    _vp_table,
                    _vp_tmin,
                    _vp_dinv,
                    VPRIME_TABLE_SIZE,
                )
            elif USE_NONFUSED_INLINE_RK2:
                rk2_step_inline(
                    phi1,
                    phi2,
                    pi1,
                    pi2,
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
                    noise2,
                    phi1_mid,
                    phi2_mid,
                    pi1_mid,
                    pi2_mid,
                    inv_a2,
                )
            elif USE_VPRIME_TABLE and USE_SINGLE_PASS_RK2:
                rk2_fused_table(
                    phi1,
                    phi2,
                    pi1,
                    pi2,
                    dt,
                    dx,
                    eta_eff,
                    mu,
                    noise,
                    noise2,
                    phi1_mid,
                    phi2_mid,
                    pi1_mid,
                    pi2_mid,
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
                    phi1, phi2, pi1, pi2,
                    dt, dx, eta_eff, T, T_mid, mu, lam, mphi,
                    bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                    fermionCoupling, fermionGaugeCoupling,
                    noise, noise2, lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
                    phi1_mid, phi2_mid, pi1_mid, pi2_mid, inv_a2,
                )
            else:
                half_noise1 = 0.5 * noise
                half_noise2 = 0.5 * noise2
                rk2_step(
                    phi1, phi2, pi1, pi2, 0.5 * dt, dx, eta_eff, T, mu, lam, mphi,
                    bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                    fermionCoupling, fermionGaugeCoupling,
                    half_noise1, half_noise2, lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
                    phi1_mid, phi2_mid, pi1_mid, pi2_mid, inv_a2,
                )
                rk2_step(
                    phi1, phi2, pi1, pi2, 0.5 * dt, dx, eta_eff, T_mid, mu, lam, mphi,
                    bosonMassSquared, bosonCoupling, bosonGaugeCoupling,
                    fermionCoupling, fermionGaugeCoupling,
                    half_noise1, half_noise2, lap_tmp, lap_tmp2, Vp_tmp, Vp_tmp2,
                    phi1_mid, phi2_mid, pi1_mid, pi2_mid, inv_a2,
                )

    # NaN / divergence guard
    _nan_check_interval = 1000 if REPLAY_MODE else 100000
    if n % _nan_check_interval == 0:
        has_nan = np.any(np.isnan(phi1)) or np.any(np.isnan(phi2))
        has_inf = np.any(np.isinf(phi1)) or np.any(np.isinf(phi2))
        if not USE_OVERDAMPED:
            has_nan = has_nan or np.any(np.isnan(pi1)) or np.any(np.isnan(pi2))
            has_inf = has_inf or np.any(np.isinf(pi1)) or np.any(np.isinf(pi2))
        _rho_check = np.sqrt(phi1**2 + phi2**2)
        rho_absmax = float(np.max(_rho_check[np.isfinite(_rho_check)])) if np.any(np.isfinite(_rho_check)) else 0.0
        diverging = rho_absmax > 1e6
        if has_nan or has_inf or diverging:
            print(f"\n{'!'*60}")
            tag = "NaN" if has_nan else ("Inf" if has_inf else "DIVERGENCE")
            print(f"{tag} DETECTED at step {n} (t_phys={t/mu:.6e})")
            print(f"  |Phi|_max = {rho_absmax:.4e}")
            print(f"  noise_scale = {noise_scale:.6e}")
            print(f"  eta_eff = {eta_eff:.6e}, T = {T:.1f}, inv_a2 = {inv_a2:.10f}")
            print(f"{'!'*60}")
            state_file = f"{state_path}/state_step_{n:010d}_NaN_debug.npz"
            _nan_save = dict(phi1=phi1, phi2=phi2, step=n, time=t, temperature=T,
                             noise_scale=noise_scale, eta_eff=eta_eff, inv_a2=inv_a2)
            if not USE_OVERDAMPED:
                _nan_save["pi1"] = pi1; _nan_save["pi2"] = pi2
            np.savez_compressed(state_file, **_nan_save)
            print(f"  Debug state saved: {state_file}")
            print("  ABORTING simulation.")
            break

    _cur_steps = steps
    if _dense_enabled and not _dense_active:
        _rho_snap = np.sqrt(phi1**2 + phi2**2)
        _phi_absmax = float(np.max(_rho_snap))
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

        # Compute derived quantities
        _rho_save = np.sqrt(phi1.astype(np.float64)**2 + phi2.astype(np.float64)**2)
        _theta_save = np.arctan2(phi2.astype(np.float64), phi1.astype(np.float64))
        compute_winding_number(phi1.astype(np.float64), phi2.astype(np.float64), winding_buf)

        # Save field state
        state_file = f"{state_path}/state_step_{n:010d}.npz"
        _is_checkpoint = _total_saves % CHECKPOINT_EVERY == 0
        _total_saves += 1
        save_dict = dict(
            phi1=phi1, phi2=phi2,
            rho=_rho_save.astype(np.float32),
            theta=_theta_save.astype(np.float32),
            winding=winding_buf.copy(),
            step=n, time=t, temperature=T,
            rho_min=float(_rho_save.min()), rho_max=float(_rho_save.max()),
            zn_active=float(_ZN_PARAMS[2]),
        )
        if _is_checkpoint and not USE_OVERDAMPED:
            save_dict["pi1"] = pi1; save_dict["pi2"] = pi2
        if HUBBLE_EXPANSION:
            save_dict["scale_factor"] = a_current
            save_dict["hubble"] = H_now
        np.savez_compressed(state_file, **save_dict)

        # 4-panel snapshot visualization
        zmid = Nz // 2
        _rho_sl = np.asarray(_rho_save[:, :, zmid], dtype=np.float64)
        _theta_sl = np.asarray(_theta_save[:, :, zmid], dtype=np.float64)
        _wind_sl = np.asarray(winding_buf[:, :, zmid], dtype=np.float64)
        _n_strings = int(np.sum(np.abs(winding_buf) > 0.5))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _t_phys = t / mu
        _title = (f"Step {n:,} | t={_t_phys:.4e} | T={T:.1f} | "
                  f"|Phi| [{_rho_save.min():.2e}, {_rho_save.max():.2e}] | "
                  f"strings: {_n_strings}")

        im0 = axes[0, 0].imshow(_rho_sl.T, origin="lower", cmap="viridis")
        axes[0, 0].set_title(r"$\rho = |\Phi|$ (z-midplane)")
        fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

        im1 = axes[0, 1].imshow(_theta_sl.T, origin="lower", cmap="hsv",
                                vmin=-np.pi, vmax=np.pi)
        axes[0, 1].set_title(r"$\theta = \arg(\Phi)$ (z-midplane)")
        fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

        _wmax = max(float(np.max(np.abs(_wind_sl))), 0.1)
        im2 = axes[1, 0].imshow(_wind_sl.T, origin="lower", cmap="RdBu_r",
                                vmin=-_wmax, vmax=_wmax)
        axes[1, 0].set_title(f"Winding number (z-midplane, |W|>0.5: {_n_strings})")
        fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)

        _rho_flat = _rho_save.flatten()
        axes[1, 1].hist(_rho_flat, bins=100, density=True, color="steelblue", alpha=0.8)
        axes[1, 1].set_xlabel(r"$\rho = |\Phi|$")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title(r"$\rho$ histogram")
        axes[1, 1].axvline(float(vev_thermal), color="r", ls="--", label=f"VEV={vev_thermal:.1e}")
        axes[1, 1].legend(fontsize=8)

        fig.suptitle(_title, fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{fig_path}/t_{_t_phys:.6e}.png", dpi=150)
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
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx_phys=dx_phys, dt_phys=dt_phys,
        mphi=mphi, lam=lam, eta_phys=eta_phys,
        T0=T0, cooling_rate=cooling_rate,
        Nt=Nt, steps=steps, total_time=total_time,
        mu=mu, dx=dx, dt=dt, eta=eta,
        vev=np.sqrt(mphi**2 / lam),
        seed_bubbles=SEED_BUBBLES,
        bubble_config=np.array(BUBBLE_CONFIG if SEED_BUBBLES else []),
        bubble_profile=BUBBLE_PROFILE if SEED_BUBBLES else "none",
        bubble_wall_width=BUBBLE_WALL_WIDTH if SEED_BUBBLES else 0.0,
        integrator=INTEGRATOR_NAME,
        counterterm=cli_args.counterterm,
        potential_type=cli_args.potential_type,
        field_type="complex",
        zn_order=cli_args.zn_order,
        zn_strength=cli_args.zn_strength,
        zn_turn_on_T=cli_args.zn_turn_on_T,
        init_rho=cli_args.init_rho,
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
