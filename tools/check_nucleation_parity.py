#!/usr/bin/env python3
"""
Nucleation parity checks (CL vs numba hypotheses 1–3) + seeded bubble roll test.

  [1] IC comparison: numba step-0 vs CL-style white noise; forward evolution on N^3 grid
  [2] pi roll-alignment at |phi|>1500 on numba step-4000 (pi from checkpoint)
  [3] 32^3 side-by-side trajectory: max|phi| and n(>5k) vs step (2500–4000 window)
  [4] Deterministic bubble seed phi=10000 on small grid — does it roll?

Usage
-----
    python tools/check_nucleation_parity.py
    python tools/check_nucleation_parity.py --n 32 --steps 4000 --quick
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time

import numpy as np

try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from diagnose_roll_dynamics import (  # noqa: E402
    DX_PROG,
    DT,
    ETA,
    FSTAR,
    MU,
    T0,
    HubbleState,
    laplacian_3d,
    vprime_field,
)

NUMBA_DIR = os.path.join(
    REPO,
    "data/lattice/set8/256x256x256_T0_1230_dx_0.001_dtphys_0.0001_interval_4000"
    "_3D_hubble_eta_1230_gb_1.09_gf_1.09_rk2_fused_inline_V_correct/field_states",
)
HDR = struct.calcsize("<IIq5d")
K_PASS2 = 73856093
K_PASS4 = 19349669
TWOPI = 6.283185307179586


def hash_mix64(x: int) -> int:
    x &= (1 << 64) - 1
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= x >> 31
    return x


def hash_gaussian(seed: int) -> float:
    x = hash_mix64(seed)
    u1 = (x >> 11) * (1.0 / 9007199254740992.0)
    x = (x * 0x2545F4914F6CDD1D) ^ 0x9E3779B97F4A7C15
    x = hash_mix64(x)
    u2 = (x >> 11) * (1.0 / 9007199254740992.0)
    if u1 < 1e-12:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(TWOPI * u2)


def hash_noise_pass(nx: int, ny: int, nz: int, step: int, mul: int) -> np.ndarray:
    if HAS_NUMBA:
        return _hash_noise_pass_nb(nx, ny, nz, step, mul)
    nyz = ny * nz
    out = np.empty((nx, ny, nz), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                site = i * nyz + j * nz + k
                seed = (site * mul) ^ step
                out[i, j, k] = hash_gaussian(seed)
    return out


if HAS_NUMBA:

    @nb.njit(cache=True)
    def _hash_mix64_nb(x):
        x = np.uint64(x)
        x ^= x >> np.uint64(30)
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x = x * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return x

    @nb.njit(cache=True)
    def _hash_gaussian_nb(seed):
        x = _hash_mix64_nb(seed)
        u1 = (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)
        x = (x * np.uint64(0x2545F4914F6CDD1D)) ^ np.uint64(0x9E3779B97F4A7C15)
        x = _hash_mix64_nb(x)
        u2 = (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)
        if u1 < 1e-12:
            u1 = 1e-12
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(TWOPI * u2)

    @nb.njit(parallel=True, cache=True)
    def _hash_noise_pass_nb(nx, ny, nz, step, mul):
        nyz = ny * nz
        out = np.empty((nx, ny, nz), dtype=np.float64)
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    site = i * nyz + j * nz + k
                    seed = np.uint64(site * mul) ^ np.uint64(step)
                    out[i, j, k] = _hash_gaussian_nb(seed)
        return out


def noise_scale(T: float, a: float, eta_eff: float) -> float:
    inv_a3 = 1.0 / a**3
    val = 2.0 * eta_eff * T * DT * inv_a3 / (MU**2 * 1e-3**3)
    return math.sqrt(max(val, 0.0))


def rk2_hash_step(
    phi: np.ndarray,
    pi: np.ndarray,
    T_now: float,
    T_mid: float,
    eta_eff: float,
    inv_a2: float,
    ns: float,
    step: int,
    *,
    use_noise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One numba/CL-parity RK2 step with per-site hash noise."""
    nx, ny, nz = phi.shape
    half = 0.5 * DT
    half_ns = 0.5 * ns
    inv_mu2 = 1.0 / (MU**2)

    lap = laplacian_3d(phi, DX_PROG)
    ph_tmp = phi + half * pi
    pi_tmp = pi + half * (inv_a2 * lap - eta_eff * pi - vprime_field(phi, T_now) * inv_mu2)

    lap2 = laplacian_3d(ph_tmp, DX_PROG)
    z2 = hash_noise_pass(nx, ny, nz, step, K_PASS2) if use_noise and half_ns > 0 else 0.0
    phi = phi + half * pi_tmp
    pi = pi + half * (inv_a2 * lap2 - eta_eff * pi_tmp - vprime_field(ph_tmp, T_now) * inv_mu2) + half_ns * z2

    lap = laplacian_3d(phi, DX_PROG)
    ph_tmp = phi + half * pi
    pi_tmp = pi + half * (inv_a2 * lap - eta_eff * pi - vprime_field(phi, T_mid) * inv_mu2)

    lap2 = laplacian_3d(ph_tmp, DX_PROG)
    z4 = hash_noise_pass(nx, ny, nz, step, K_PASS4) if use_noise and half_ns > 0 else 0.0
    phi = phi + half * pi_tmp
    pi = pi + half * (inv_a2 * lap2 - eta_eff * pi_tmp - vprime_field(ph_tmp, T_mid) * inv_mu2) + half_ns * z4
    return phi, pi


def evolve_lattice(
    phi: np.ndarray,
    pi: np.ndarray,
    n_steps: int,
    *,
    use_noise: bool,
    report_every: int = 0,
    label: str = "",
) -> list[tuple[int, float, float, int]]:
    """Return list of (step, T, max|phi|, n>5k)."""
    hs = HubbleState()
    rows: list[tuple[int, float, float, int]] = []
    ap0 = np.abs(phi)
    rows.append((0, hs.T(), float(ap0.max()), int((ap0 > 5000).sum())))

    for n in range(n_steps):
        T_now, T_mid, eta_eff, inv_a2, _, ns = hs.advance()
        if not use_noise:
            ns = 0.0
        phi, pi = rk2_hash_step(phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, n, use_noise=use_noise)
        if report_every and (n + 1) % report_every == 0:
            ap = np.abs(phi)
            rows.append((n + 1, hs.T(), float(ap.max()), int((ap > 5000).sum())))
    if not report_every or rows[-1][0] != n_steps:
        ap = np.abs(phi)
        rows.append((n_steps, hs.T(), float(ap.max()), int((ap > 5000).sum())))
    return rows


def load_numba_step(step: int) -> dict:
    path = os.path.join(NUMBA_DIR, f"state_step_{step:010d}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return dict(np.load(path))


def check_ic(n: int, steps: int, quick: bool):
    print("\n" + "=" * 72)
    print("[1] Initial conditions & forward evolution (hash-RK2 parity evolver)")
    print("=" * 72)

    nb0 = load_numba_step(0)
    phi_nb = nb0["phi"]
    if phi_nb.shape[0] != n:
        phi_nb = None
        print(f"  numba step-0 is {load_numba_step(0)['phi'].shape[0]}^3 — using downsampled stats only")
    else:
        print(f"  numba step-0: std={phi_nb.std():.2f}  max|phi|={np.abs(phi_nb).max():.1f}")

    rng = np.random.default_rng(42)
    phi_cl = (0.01 * rng.standard_normal((n, n, n))).astype(np.float64)
    pi_cl = np.zeros_like(phi_cl)
    print(f"  CL-style IC (0.01*randn seed=42): std={phi_cl.std():.4f}  max|phi|={np.abs(phi_cl).max():.4f}")

    if phi_nb is not None:
        pi_nb = nb0["pi"] if "pi" in nb0 else np.zeros_like(phi_nb)
        if steps_run := (500 if quick else steps):
            print(f"\n  Evolving numba IC on {n}^3 for {steps_run} steps (noise on)...")
            t0 = time.time()
            rows_nb = evolve_lattice(phi_nb.copy(), pi_nb.copy(), steps_run, use_noise=True, report_every=100 if quick else 500, label="numba IC")
            print(f"    done in {time.time()-t0:.1f}s  final max|phi|={rows_nb[-1][2]:.0f}  >5k={rows_nb[-1][3]}")
    else:
        rows_nb = None

    steps_run = 500 if quick else steps
    print(f"\n  Evolving CL-style IC on {n}^3 for {steps_run} steps (noise on)...")
    t0 = time.time()
    rows_cl = evolve_lattice(phi_cl.copy(), pi_cl.copy(), steps_run, use_noise=True, report_every=100 if quick else 500, label="CL IC")
    print(f"    done in {time.time()-t0:.1f}s  final max|phi|={rows_cl[-1][2]:.0f}  >5k={rows_cl[-1][3]}")

    print("\n  Trajectory sample (CL IC):")
    print(f"  {'step':>6s}  {'T':>8s}  {'max|phi|':>12s}  {'>5k':>8s}")
    for row in rows_cl:
        if row[0] % (100 if quick else 500) == 0 or row[0] == steps_run:
            print(f"  {row[0]:6d}  {row[1]:8.1f}  {row[2]:12.0f}  {row[3]:8d}")

    if rows_nb:
        print("\n  Trajectory sample (numba IC on same grid):")
        for row in rows_nb:
            if row[0] % (100 if quick else 500) == 0 or row[0] == steps_run:
                print(f"  {row[0]:6d}  {row[1]:8.1f}  {row[2]:12.0f}  {row[3]:8d}")


def check_pi_alignment():
    print("\n" + "=" * 72)
    print("[2] pi roll-alignment at |phi|>1500 (numba step 4000)")
    print("=" * 72)

    path = os.path.join(NUMBA_DIR, "state_step_0000004000.npz")
    d = np.load(path)
    phi = d["phi"]
    T = float(d["temperature"])
    if "pi" not in d:
        # checkpoint every 10 steps includes pi
        ckpt = os.path.join(NUMBA_DIR, "state_step_0000004000.npz")
        d = np.load(ckpt)
        if "pi" not in d:
            print("  WARNING: no pi in numba snapshot — skipping pi alignment")
            return
    pi = d["pi"] if "pi" in d else None
    if pi is None:
        print("  WARNING: pi not saved at step 4000")
        return

    mask = np.abs(phi) > 1500
    n_tail = int(mask.sum())
    print(f"  T={T:.1f} GeV  tail sites |phi|>1500: {n_tail}")

    vp = vprime_field(phi, T)
    roll_dir = -np.sign(vp)  # direction of -V'/mu^2 push on pi (sign)
    pi_sign = np.sign(pi)
    aligned = (pi_sign == roll_dir) & mask
    nonzero_pi = (np.abs(pi) > 1.0) & mask
    print(f"  fraction tail with |pi|>1 GeV: {nonzero_pi.sum()/max(n_tail,1):.3f}")
    print(f"  fraction tail pi aligned with roll direction: {aligned.sum()/max(n_tail,1):.3f}")

    # blob vs isolated on rolled sites
    ap = np.abs(phi)
    nb_mean = np.zeros_like(ap)
    nx = phi.shape[0]
    for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nb_mean += np.abs(np.roll(phi, (-di, -dj, -dk), axis=(0, 1, 2)))
    nb_mean /= 6.0
    iso = ap / np.maximum(nb_mean, 1.0)

    blob = mask & (ap > 5000)
    iso_tail = mask & (ap <= 5000)
    if blob.sum():
        print(f"  rolled blob (|phi|>5k): n={blob.sum()}  median isolation={np.median(iso[blob]):.2f}  "
              f"pi-align={aligned[blob].mean():.3f}  median|pi|={np.median(np.abs(pi[blob])):.0f}")
    if iso_tail.sum():
        print(f"  isolated tail (1.5k<|phi|<5k): n={iso_tail.sum()}  median isolation={np.median(iso[iso_tail]):.2f}  "
              f"pi-align={aligned[iso_tail].mean():.3f}  median|pi|={np.median(np.abs(pi[iso_tail])):.0f}")

    # global max site
    imax = np.unravel_index(np.argmax(ap), ap.shape)
    print(f"\n  global max at {imax}: phi={phi[imax]:.0f}  pi={pi[imax]:.0f}  "
          f"V'={vp[imax]:.2e}  roll_dir={roll_dir[imax]:+.0f}  aligned={aligned[imax]}")


def check_side_by_side(n: int, steps: int, quick: bool):
    print("\n" + "=" * 72)
    print(f"[3] {n}^3 side-by-side: first step with max|phi|>5k (hash-RK2, noise on)")
    print("=" * 72)

    steps_run = min(steps, 800) if quick else steps
    rng = np.random.default_rng(42)
    phi_cl = 0.01 * rng.standard_normal((n, n, n))
    pi_cl = np.zeros_like(phi_cl)

    try:
        nb0 = load_numba_step(0)
        if nb0["phi"].shape[0] == n:
            phi_nb, pi_nb = nb0["phi"].copy(), np.zeros_like(nb0["phi"])
        else:
            phi_nb = phi_cl.copy()
            pi_nb = pi_cl.copy()
            print(f"  numba step-0 is {nb0['phi'].shape[0]}^3 — using CL IC for both (same seed)")
    except FileNotFoundError:
        phi_nb, pi_nb = phi_cl.copy(), pi_cl.copy()

    report = 50 if not quick else 25
    print(f"  Running {steps_run} steps, report every {report}...")
    rows_cl = evolve_lattice(phi_cl, pi_cl, steps_run, use_noise=True, report_every=report)
    rows_nb = evolve_lattice(phi_nb, pi_nb, steps_run, use_noise=True, report_every=report)

    def first_5k(rows):
        for r in rows:
            if r[2] > 5000:
                return r
        return None

    f_cl, f_nb = first_5k(rows_cl), first_5k(rows_nb)
    print(f"\n  {'label':12s}  {'first>5k step':>14s}  {'T':>8s}  {'max|phi|':>12s}")
    print(f"  {'CL IC':12s}  {str(f_cl[0] if f_cl else 'none'):>14s}  "
          f"{f_cl[1] if f_cl else 0:8.1f}  {rows_cl[-1][2]:12.0f}")
    print(f"  {'numba IC':12s}  {str(f_nb[0] if f_nb else 'none'):>14s}  "
          f"{f_nb[1] if f_nb else 0:8.1f}  {rows_nb[-1][2]:12.0f}")

    print(f"\n  Steps 2500–4000 window (if steps>2500):")
    for label, rows in [("CL IC", rows_cl), ("numba IC", rows_nb)]:
        win = [r for r in rows if 2500 <= r[0] <= 4000 or (steps_run < 2500 and r[0] == steps_run)]
        if not win:
            win = rows[-5:]
        print(f"  --- {label} ---")
        for r in win:
            print(f"    step {r[0]:5d}  T={r[1]:7.1f}  max={r[2]:10.0f}  >5k={r[3]}")


def check_bubble_seed(n: int, phi_seed: float, n_steps: int, bg: float):
    print("\n" + "=" * 72)
    print(f"[4] Seeded bubble roll: centre phi={phi_seed:.0f} GeV, bg={bg:.0f}, {n}^3, no noise")
    print("=" * 72)

    phi = np.full((n, n, n), bg, dtype=np.float64)
    pi = np.zeros_like(phi)
    c = n // 2
    phi[c, c, c] = phi_seed

    hs = HubbleState()
    # advance to T ~ 1172 like step 4000
    target_T = 1172.0
    while hs.T() > target_T + 0.5:
        hs.advance()

    center_hist = [phi_seed]
    max_hist = [float(np.abs(phi).max())]
    T_hist = [hs.T()]

    for step in range(n_steps):
        T_now, T_mid, eta_eff, inv_a2, _, _ = hs.advance()
        phi, pi = rk2_hash_step(phi, pi, T_now, T_mid, eta_eff, inv_a2, 0.0, step, use_noise=False)
        center_hist.append(float(phi[c, c, c]))
        max_hist.append(float(np.abs(phi).max()))
        T_hist.append(hs.T())

    print(f"  T: {T_hist[0]:.1f} -> {T_hist[-1]:.1f} GeV over {n_steps} steps")
    print(f"  centre phi: {center_hist[0]:.0f} -> {center_hist[-1]:.0f} GeV")
    print(f"  max|phi|:  {max_hist[0]:.0f} -> {max_hist[-1]:.0f} GeV")
    rolled = max_hist[-1] > max(phi_seed * 2, 20000)
    print(f"  ROLL {'YES' if rolled else 'NO'} (threshold: max > {max(phi_seed*2, 20000):.0f})")

    # Laplacian vs V' at centre step 0
    lap = laplacian_3d(phi, DX_PROG)[c, c, c] / (T0 / T_hist[-1]) ** 2
    vp = float(vprime_field(np.array([center_hist[-1]]), T_hist[-1])[0])
    print(f"  final centre: lap={lap:.2e}  V'={vp:.2e}  -V'/mu2={-vp/MU**2:.2e}")
    return rolled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--quick", action="store_true", help="Shorter runs for CI (~500 steps)")
    ap.add_argument("--phi-seed", type=float, default=10000.0)
    ap.add_argument("--bg", type=float, default=400.0)
    args = ap.parse_args()

    print("Nucleation parity checks")
    print(f"  grid={args.n}^3  steps={args.steps}  quick={args.quick}")

    check_ic(args.n, args.steps, args.quick)
    check_pi_alignment()
    check_side_by_side(args.n, args.steps, args.quick)
    check_bubble_seed(args.n, args.phi_seed, min(200, args.steps if args.quick else 400), args.bg)

    print("\n" + "=" * 72)
    print("Done. For CL binary bubble-seed test, run run_cosmolattice.py with --bubble_seed_phi")
    print("=" * 72)


if __name__ == "__main__":
    main()
