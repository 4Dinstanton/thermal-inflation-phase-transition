#!/usr/bin/env python3
"""
Single-site / uniform-field roll diagnostic: numba vs CosmoLattice RK2.

Tests whether post-spinodal roll fails due to:
  (A) deterministic integrator / V' mismatch, or
  (B) stochastic amplitude / spatial Laplacian drag.

Usage
-----
    python tools/diagnose_roll_dynamics.py
    python tools/diagnose_roll_dynamics.py --plot
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "figs")

# --- physics (256^3 T0=1230 V_correct, nb=nf=20) --------------------------------
M_PL = 2.4e18
G_STAR = 106.75
DEL_V = 1e36 / 4
GAMMA = 4.1667e-4
FSTAR = GAMMA * M_PL
MU = 1000.0
Mphi = 1000.0
LAM = Mphi**2 / FSTAR**2
T0 = 1230.0
ETA_PHYS = 1230.0
ETA = ETA_PHYS / MU
DX_PHYS = 1e-3
DX_PROG = MU * DX_PHYS  # program lattice spacing (= numba dx)
DT_PHYS = 1e-4
DT = DT_PHYS * Mphi  # program dt = 0.1
NB = NF = 20.0
YB, GB = 1.09, 1.05
YF, GF = 1.09, 1.05
MB2 = 1e6
COEF_B = 0.25 * YB**2 + (2.0 / 3.0) * GB**2


def hubble(T: float) -> float:
    chig2 = 30.0 / (math.pi**2 * G_STAR)
    return math.sqrt((T**4 / chig2 + DEL_V) / (3.0 * M_PL**2))


_CTFT = None


def _get_ctft():
    global _CTFT
    if _CTFT is None:
        sys.path.insert(0, os.path.join(REPO, "potential"))
        import cosmoTransitions.finiteT as CTFT  # noqa: WPS433

        _CTFT = CTFT
    return _CTFT


def vprime_field(phi: np.ndarray, T: float) -> np.ndarray:
    """Vectorized V'(phi,T) — numba / CL table style (no CW)."""
    CTFT = _get_ctft()
    pref = T**4 / (2.0 * math.pi**2)
    dV = LAM * phi**3 - Mphi**2 * phi
    xb_sq = MB2 + 0.5 * YB**2 * phi**2 + COEF_B * T**2
    mask = xb_sq > 0
    xb = np.sqrt(np.maximum(xb_sq[mask], 0.0)) / T
    dJb = np.real(CTFT.dJb_exact(xb))
    dub = 0.5 * YB**2 * phi[mask] / (T**2 * np.maximum(xb, 1e-20))
    dV[mask] += pref * NB * dJb * dub
    xf_sq = 0.5 * YF**2 * phi**2 + (1.0 / 6.0) * GF**2 * T**2
    mask = xf_sq > 0
    xf = np.sqrt(np.maximum(xf_sq[mask], 0.0)) / T
    dJf = np.real(CTFT.dJf_exact(xf))
    duf = 0.5 * YF**2 * phi[mask] / (T**2 * np.maximum(xf, 1e-20))
    dV[mask] += pref * NF * dJf * duf
    return dV


def vprime(phi: float, T: float) -> float:
    return float(vprime_field(np.array([phi], dtype=np.float64), T)[0])


def noise_scale(T: float, a: float, eta_eff: float) -> float:
    inv_a3 = 1.0 / a**3
    val = 2.0 * eta_eff * T * DT * inv_a3 / (MU**2 * DX_PHYS**3)
    return math.sqrt(max(val, 0.0))


class HubbleState:
    __slots__ = ("a", "T0")

    def __init__(self, T0_val: float = T0):
        self.T0 = T0_val
        self.a = 1.0

    def T(self) -> float:
        return self.T0 / self.a

    def advance(self) -> tuple[float, float]:
        """Return (T_now, T_mid) for this step; update a."""
        T_now = self.T()
        H = hubble(T_now)
        eta_eff = ETA + 3.0 * H / MU
        inv_a2 = 1.0 / self.a**2
        inv_a3 = 1.0 / self.a**3
        ns = noise_scale(T_now, self.a, eta_eff)
        self.a *= 1.0 + H * DT / MU
        T_mid = self.T0 / self.a
        return T_now, T_mid, eta_eff, inv_a2, inv_a3, ns


def rk2_fused_4pass_field(
    phi: np.ndarray,
    pi: np.ndarray,
    T_now: float,
    T_mid: float,
    eta_eff: float,
    inv_a2: float,
    ns: float,
    rng: np.random.Generator,
    cl_sqrt2: bool,
    dx: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One fused 4-pass RK2 step on full arrays (matches rk2_fused_single_pass)."""
    half = 0.5 * DT
    half_ns = 0.5 * ns * (math.sqrt(2.0) if cl_sqrt2 else 1.0)
    inv_mu2 = 1.0 / (MU**2)
    use_lap = dx is not None

    if use_lap:
        lap = laplacian_3d(phi, dx)
    else:
        lap = np.zeros_like(phi)

    ph_tmp = phi + half * pi
    pi_tmp = pi + half * (inv_a2 * lap - eta_eff * pi - vprime_field(phi, T_now) * inv_mu2)

    if use_lap:
        lap2 = laplacian_3d(ph_tmp, dx)
    else:
        lap2 = lap
    z = rng.standard_normal(phi.shape) if half_ns > 0 else 0.0
    phi = phi + half * pi_tmp
    pi = pi + half * (inv_a2 * lap2 - eta_eff * pi_tmp - vprime_field(ph_tmp, T_now) * inv_mu2) + half_ns * z

    if use_lap:
        lap = laplacian_3d(phi, dx)
    else:
        lap = np.zeros_like(phi)
    ph_tmp = phi + half * pi
    pi_tmp = pi + half * (inv_a2 * lap - eta_eff * pi - vprime_field(phi, T_mid) * inv_mu2)

    if use_lap:
        lap2 = laplacian_3d(ph_tmp, dx)
    else:
        lap2 = lap
    z = rng.standard_normal(phi.shape) if half_ns > 0 else 0.0
    phi = phi + half * pi_tmp
    pi = pi + half * (inv_a2 * lap2 - eta_eff * pi_tmp - vprime_field(ph_tmp, T_mid) * inv_mu2) + half_ns * z
    return phi, pi


def rk2_fused_4pass(
    phi: float,
    pi: float,
    T_now: float,
    T_mid: float,
    eta_eff: float,
    inv_a2: float,
    ns: float,
    rng: np.random.Generator,
    cl_sqrt2: bool,
    lap: float = 0.0,
) -> tuple[float, float]:
    p = np.array([phi], dtype=np.float64)
    v = np.array([pi], dtype=np.float64)
    p, v = rk2_fused_4pass_field(p, v, T_now, T_mid, eta_eff, inv_a2, ns, rng, cl_sqrt2, dx=None)
    return float(p[0]), float(v[0])


def run_0d(
    phi0: float,
    pi0: float,
    n_steps: int,
    *,
    cl_sqrt2: bool = False,
    noise: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform field (lap=0): all sites share (phi, pi)."""
    rng = np.random.default_rng(seed)
    hs = HubbleState()
    phis = np.empty(n_steps + 1)
    pis = np.empty(n_steps + 1)
    Ts = np.empty(n_steps + 1)
    phis[0], pis[0] = phi0, pi0
    Ts[0] = hs.T()
    phi, pi = phi0, pi0
    for n in range(n_steps):
        T_now, T_mid, eta_eff, inv_a2, inv_a3, ns = hs.advance()
        if not noise:
            ns = 0.0
        phi, pi = rk2_fused_4pass(phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, rng, cl_sqrt2)
        phis[n + 1] = phi
        pis[n + 1] = pi
        Ts[n + 1] = hs.T()
    return phis, pis, Ts


def laplacian_3d(phi: np.ndarray, dx: float) -> np.ndarray:
    inv_dx2 = 1.0 / dx**2
    lap = (
        np.roll(phi, 1, 0)
        + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1)
        + np.roll(phi, -1, 1)
        + np.roll(phi, 1, 2)
        + np.roll(phi, -1, 2)
        - 6.0 * phi
    ) * inv_dx2
    return lap


def run_hot_spot(
    n: int,
    phi_hot: float,
    phi_bg: float,
    n_steps: int,
    *,
    cl_sqrt2: bool = False,
    noise: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Center site seeded high; rest at background. Returns max|phi| history."""
    rng = np.random.default_rng(seed)
    hs = HubbleState()
    phi = np.full((n, n, n), phi_bg, dtype=np.float64)
    pi = np.zeros_like(phi)
    c = n // 2
    phi[c, c, c] = phi_hot
    max_hist = np.empty(n_steps + 1)
    max_hist[0] = np.max(np.abs(phi))
    for step in range(n_steps):
        T_now, T_mid, eta_eff, inv_a2, _, ns = hs.advance()
        if not noise:
            ns = 0.0
        phi, pi = rk2_fused_4pass_field(phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, rng, cl_sqrt2, dx=DX_PROG)
        max_hist[step + 1] = np.max(np.abs(phi))
    return max_hist, phi


def test_deterministic_roll():
    print("\n" + "=" * 70)
    print("[1] Deterministic 0D roll (no noise, lap=0, phi0=1000 GeV, pi0=0)")
    print("=" * 70)
    target_T = 1172.0
    for label, sqrt2 in [("numba (bare ns)", False), ("CL (sqrt2 ns)", True)]:
        hs = HubbleState()
        while hs.T() > target_T + 0.1:
            hs.advance()
        phi, pi = 1000.0, 0.0
        for step in range(1, 51):
            T_now, T_mid, eta_eff, inv_a2, _, _ = hs.advance()
            phi, pi = rk2_fused_4pass(phi, pi, T_now, T_mid, eta_eff, inv_a2, 0.0, np.random.default_rng(0), sqrt2)
            if not np.isfinite(phi):
                print(f"  {label:20s}  roll diverges by step {step} (|phi| > float64 range) — V' roll works")
                break
            if step in (1, 5, 10, 20, 50):
                print(f"  {label:20s}  step {step:2d}: phi={phi:+.4e} GeV  pi={pi:+.4e} GeV  V'={vprime(phi, hs.T()):+.4e}")
        else:
            print(f"  {label:20s}  step 50: phi={phi:+.4e} GeV (still finite)")


def test_vprime_at_roll():
    print("\n" + "=" * 70)
    print("[2] V'(phi) at T=1172 GeV (spinodal ~470 GeV)")
    print("=" * 70)
    T = 1172.0
    for ph in [0, 470, 1000, 2000, 5000, 10000, 50000]:
        vp = vprime(ph, T)
        print(f"  phi={ph:6d} GeV:  V'={vp:+.4e}  {'roll' if vp < 0 else 'restored'}")


def test_stochastic_ensemble():
    print("\n" + "=" * 70)
    print("[3] Stochastic 0D ensemble (200 traj, 80 steps from T~1172, phi0=1000)")
    print("=" * 70)
    n_traj, n_steps = 200, 80
    sigma_eq = math.sqrt(T0 / (DX_PHYS**3 * MU**2))
    for label, sqrt2 in [("numba (bare ns)", False), ("CL (sqrt2 ns)", True)]:
        max_phi = []
        rolled = 0
        for seed in range(n_traj):
            hs = HubbleState()
            while hs.T() > 1172.5:
                hs.advance()
            phi, pi = 1000.0, float(np.random.default_rng(seed).normal(0.0, sigma_eq))
            rng = np.random.default_rng(seed + 1000)
            for _ in range(n_steps):
                T_now, T_mid, eta_eff, inv_a2, _, ns = hs.advance()
                phi, pi = rk2_fused_4pass(phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, rng, sqrt2)
                if not np.isfinite(phi):
                    break
            mx = float(np.abs(phi)) if np.isfinite(phi) else 1e30
            max_phi.append(mx)
            if mx > 5000:
                rolled += 1
        max_phi = np.array(max_phi)
        finite = max_phi[np.isfinite(max_phi) & (max_phi < 1e20)]
        print(
            f"  {label:20s}  finite runs={len(finite)}/{n_traj}  "
            f"median max|phi|={np.median(finite) if len(finite) else float('nan'):.0f}  "
            f"P(max>5e3)={rolled/n_traj:.3f}  "
            f"P(diverge)={(n_traj-len(finite))/n_traj:.3f}"
        )


def test_hot_spot():
    print("\n" + "=" * 70)
    print("[4] Spatial hot-spot (7^3, center phi=2000, bg=400 GeV, 30 steps, T~1172)")
    print("=" * 70)
    hs = HubbleState()
    while hs.T() > 1172.5:
        hs.advance()
    for label, sqrt2, noise in [
        ("numba det", False, False),
        ("CL det", True, False),
        ("numba stoch", False, True),
        ("CL stoch", True, True),
    ]:
        # restart from same hot spot each time
        rng = np.random.default_rng(42)
        hs2 = HubbleState()
        hs2.a = hs.a
        phi = np.full((7, 7, 7), 400.0, dtype=np.float64)
        pi = np.zeros_like(phi)
        phi[3, 3, 3] = 2000.0
        hist = [np.max(np.abs(phi))]
        for _ in range(30):
            T_now, T_mid, eta_eff, inv_a2, _, ns = hs2.advance()
            if not noise:
                ns = 0.0
            phi, pi = rk2_fused_4pass_field(phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, rng, cl_sqrt2=sqrt2, dx=DX_PROG)
            hist.append(np.max(np.abs(phi)))
            if not np.isfinite(hist[-1]):
                break
        end = hist[-1] if np.isfinite(hist[-1]) else float("inf")
        ctr = abs(phi[3, 3, 3]) if np.isfinite(phi[3, 3, 3]) else float("nan")
        print(f"  {label:14s}  max|phi| start={hist[0]:.0f}  end={end:.4e}  center={ctr:.4e}")


def test_match_lattice_bulk():
    print("\n" + "=" * 70)
    print("[5] 0D thermal equilibrium (200 steps, phi0=0, pi0=0, noise on)")
    print("=" * 70)
    tgt = T0 / (DX_PHYS**3 * FSTAR**2 * MU**2)
    for label, sqrt2 in [("numba (bare ns)", False), ("CL (sqrt2 ns)", True)]:
        phis, pis, _ = run_0d(0.0, 0.0, 200, cl_sqrt2=sqrt2, noise=True, seed=123)
        ok = np.isfinite(phis).all() and np.isfinite(pis).all()
        piS2 = float(np.mean(pis**2) / FSTAR**2) if ok else float("nan")
        ratio = piS2 / tgt if ok else float("nan")
        print(f"  {label:20s}  phi_rms={np.std(phis):.1f} GeV  <piS^2>/FDT={ratio:.3f}")


def make_plot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # deterministic roll
    ax = axes[0, 0]
    for label, sqrt2, ls in [("numba", False, "-"), ("CL sqrt2", True, "--")]:
        hs = HubbleState()
        while hs.T() > 1172.5:
            hs.advance()
        phi, pi = 1000.0, 0.0
        hist = [phi]
        for _ in range(300):
            T_now, T_mid, eta_eff, inv_a2, _, _ = hs.advance()
            phi, pi = rk2_fused_4pass(phi, pi, T_now, T_mid, eta_eff, inv_a2, 0.0, np.random.default_rng(0), sqrt2)
            hist.append(phi)
        ax.semilogy(range(len(hist)), np.abs(hist), ls=ls, label=label)
    ax.set_xlabel("steps after T≈1172")
    ax.set_ylabel("|phi| (GeV)")
    ax.set_title("Deterministic 0D roll (phi0=1000, pi0=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # stochastic single traj
    ax = axes[0, 1]
    for label, sqrt2, ls in [("numba", False, "-"), ("CL sqrt2", True, "--")]:
        phis, _, Ts = run_0d(1000.0, 0.0, 400, cl_sqrt2=sqrt2, noise=True, seed=7)
        ax.semilogy(Ts, np.abs(phis), ls=ls, label=label)
    ax.set_xlabel("T (GeV)")
    ax.set_ylabel("|phi| (GeV)")
    ax.set_title("Stochastic 0D (phi0=1000, pi0=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ensemble CDF
    ax = axes[1, 0]
    for label, sqrt2 in [("numba", False), ("CL sqrt2", True)]:
        maxes = []
        for seed in range(200):
            phis, _, _ = run_0d(1000.0, 0.0, 400, cl_sqrt2=sqrt2, noise=True, seed=seed)
            maxes.append(np.max(np.abs(phis)))
        xs = np.sort(maxes)
        ax.semilogx(xs, np.linspace(0, 1, len(xs)), label=label)
    ax.axvline(5000, color="k", ls=":", lw=0.8, label="5e3 GeV")
    ax.set_xlabel("max |phi| over 400 steps")
    ax.set_ylabel("CDF")
    ax.set_title("Stochastic ensemble (200 traj)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # hot spot
    ax = axes[1, 1]
    for label, sqrt2, noise, ls in [
        ("numba det", False, False, "-"),
        ("CL det", True, False, "--"),
        ("numba stoch", False, True, "-."),
        ("CL stoch", True, True, ":"),
    ]:
        hist, _ = run_hot_spot(7, 2000.0, 400.0, 100, cl_sqrt2=sqrt2, noise=noise, seed=42)
        ax.plot(hist, ls=ls, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("max |phi| (GeV)")
    ax.set_title("7^3 hot-spot (center=2000, bg=400)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "diagnose_roll_dynamics.png")
    fig.savefig(path, dpi=120)
    print(f"\nSaved plot: {os.path.relpath(path, REPO)}")


def test_thermal_only():
    print("\n" + "=" * 70)
    print("[6] Thermal noise only (0D, phi0=0, 400 steps at T~1172) — mimics lattice bulk")
    print("=" * 70)
    for label, sqrt2 in [("numba (bare ns)", False), ("CL (sqrt2 ns)", True)]:
        hs = HubbleState()
        while hs.T() > 1172.5:
            hs.advance()
        phi, pi = 0.0, 0.0
        rng = np.random.default_rng(99)
        max_hist = []
        for _ in range(400):
            T_now, T_mid, eta_eff, inv_a2, _, ns = hs.advance()
            phi, pi = rk2_fused_4pass(phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, rng, sqrt2)
            if np.isfinite(phi):
                max_hist.append(abs(phi))
            else:
                break
        if max_hist:
            print(
                f"  {label:20s}  phi_rms={np.std(max_hist):.1f}  max|phi|={max(max_hist):.0f}  "
                f"(thermal max~{6.5*np.std(max_hist):.0f})"
            )
        else:
            print(f"  {label:20s}  diverged immediately")


def main():
    ap = argparse.ArgumentParser(description="CL vs numba roll diagnostic")
    ap.add_argument("--plot", action="store_true", help="Write figs/diagnose_roll_dynamics.png")
    args = ap.parse_args()

    print("Roll dynamics diagnostic (nb=nf=20, T0=1230, eta=1230, dt_phys=1e-4)")
    test_vprime_at_roll()
    test_deterministic_roll()
    test_match_lattice_bulk()
    test_stochastic_ensemble()
    test_hot_spot()
    test_thermal_only()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        "  If deterministic 0D rolls match but lattice does not → spatial / snapshot issue.\n"
        "  If deterministic 0D already differs → integrator or V' bug.\n"
        "  If only stochastic ensemble differs → noise amplitude / RNG structure."
    )

    if args.plot:
        make_plot()


if __name__ == "__main__":
    main()
