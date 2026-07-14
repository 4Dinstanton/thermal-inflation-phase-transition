#!/usr/bin/env python3
"""
Spinodal roll diagnostic for CL vs expected linear growth.

Compares snapshot trajectories against the unstable-mode growth rate
  gamma ~ (-eta + sqrt(eta^2 + 4*|V''(0)|/mu^2)) / 2   (program time)
at T below the spinodal.

Usage
-----
    python tools/diagnose_spinodal.py data/lattice/roll_gaussian_T900/.../
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from diagnose_roll_dynamics import laplacian_3d, vprime, MU, DT, DX_PROG  # noqa: E402
from export_thermal_splines import TableEval, DEFAULT_PARAMS  # noqa: E402


def load_table():
    d = np.load(os.path.join(REPO, "data", "thermal_splines", "thermal_tables.npz"))
    params = dict(DEFAULT_PARAMS)
    gamma = 4.1667e-4
    params["gamma"] = gamma
    params["lam"] = params["mphi"] ** 2 / (gamma * 2.4e18) ** 2
    return TableEval(
        d["u"], d["Jb"], d["Jf"], d["dJb"], d["dJf"], d["d2Jb"], d["d2Jf"], params
    )


def vpp_at_zero(ev: TableEval, T: float) -> float:
    h = 10.0
    return (ev.Vprime(h, T, include_cw=False) - ev.Vprime(-h, T, include_cw=False)) / (2 * h)


def spinodal_growth_rate(T: float, eta_phys: float) -> float:
    ev = load_table()
    vpp = vpp_at_zero(ev, T)
    eta = eta_phys / MU
    inv_mu2 = 1.0 / (MU * MU)
    # phi_tt + eta phi_t + (V''/mu^2) phi = 0, V''/mu^2 = vpp * inv_mu2
    # (vpp < 0 => spinodal unstable). Note: always * inv_mu2, never / inv_mu2.
    disc = eta * eta - 4.0 * vpp * inv_mu2
    if disc < 0:
        return 0.0
    return 0.5 * (-eta + math.sqrt(disc))


def load_snapshots(run_dir: str) -> list[tuple[int, np.ndarray, np.ndarray | None]]:
    state_dir = os.path.join(run_dir, "field_states")
    out = []
    for path in sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz"))):
        d = np.load(path)
        step = int(d["step"]) if "step" in d else int(os.path.basename(path).split("_")[-1].split(".")[0])
        pi = d["pi"] if "pi" in d else None
        out.append((step, d["phi"].astype(np.float64), pi))
    return out


def kspace_rms(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-shell |phi_k| rms vs |k| index (integer mode number)."""
    n = phi.shape[0]
    fk = np.fft.rfftn(phi)
    shells: dict[int, list[float]] = {}
    for i in range(n):
        for j in range(n):
            for k in range(fk.shape[2]):
                ki = i if i <= n // 2 else i - n
                kj = j if j <= n // 2 else j - n
                kk = k if k <= n // 2 else k - n
                kn = int(round(math.sqrt(ki * ki + kj * kj + kk * kk)))
                shells.setdefault(kn, []).append(abs(fk[i, j, k]))
    ns = sorted(shells)
    rms = [float(np.sqrt(np.mean(np.square(shells[nn])))) for nn in ns]
    return np.array(ns, dtype=float), np.array(rms)


def force_balance(phi: np.ndarray, T: float) -> dict:
    lap = laplacian_3d(phi, DX_PROG)
    vp = np.vectorize(lambda x: vprime(float(x), T))(phi)
    roll = -vp / (MU * MU)
    return {
        "lap_rms": float(np.sqrt(np.mean(lap**2))),
        "roll_rms": float(np.sqrt(np.mean(roll**2))),
        "roll_max": float(np.max(np.abs(roll))),
        "lap_max": float(np.max(np.abs(lap))),
    }


def main():
    ap = argparse.ArgumentParser(description="Spinodal roll diagnostic")
    ap.add_argument("run_dir", help="CL run directory with field_states/*.npz")
    ap.add_argument("--T", type=float, default=None, help="Temperature (GeV); default from metadata")
    ap.add_argument("--eta", type=float, default=None, help="eta_phys (GeV); default from metadata")
    args = ap.parse_args()

    meta_path = os.path.join(args.run_dir, "simulation_metadata.npz")
    T = args.T
    eta_phys = args.eta
    if os.path.isfile(meta_path):
        m = np.load(meta_path, allow_pickle=True)
        if T is None and "T0" in m:
            T = float(m["T0"])
        if eta_phys is None and "eta_phys" in m:
            eta_phys = float(m["eta_phys"])

    snaps = load_snapshots(args.run_dir)
    if not snaps:
        sys.exit(f"No snapshots in {args.run_dir}/field_states")

    if T is None:
        T = 900.0
    if eta_phys is None:
        eta_phys = T

    ev = load_table()
    vpp0 = vpp_at_zero(ev, T)
    gamma = spinodal_growth_rate(T, eta_phys)

    print("=" * 70)
    print(f"Spinodal diagnostic  T={T:.1f} GeV  eta={eta_phys:.1f} GeV")
    print(f"V''(0) = {vpp0:.4e} GeV^2  (negative => spinodal unstable)")
    print(f"Predicted k=0 growth rate gamma ~ {gamma:.4f} / program time")
    print(f"  => exp(gamma * t) at step 400 (t=40): {math.exp(gamma * 40):.2e}x from IC amplitude")
    print("=" * 70)

    print(f"\n{'step':>6} {'max|phi|':>12} {'rms':>10} {'frac>500':>10} {'frac>50':>10}")
    for step, phi, _ in snaps[:: max(1, len(snaps) // 20)]:
        print(
            f"{step:6d} {np.abs(phi).max():12.3f} "
            f"{np.sqrt(np.mean(phi**2)):10.4f} "
            f"{(np.abs(phi) > 500).mean():10.4f} "
            f"{(np.abs(phi) > 50).mean():10.4f}"
        )

    phi0 = snaps[0][1]
    fb = force_balance(phi0, T)
    print("\nForce balance at IC (step 0):")
    print(f"  lap rms={fb['lap_rms']:.4f}  max={fb['lap_max']:.4f}")
    print(f"  |V'/mu^2| rms={fb['roll_rms']:.4f}  max={fb['roll_max']:.4f}")
    print("  (If lap >> |V'/mu^2|, spatial diffusion dominates before roll takes over.)")

    kn, rms0 = kspace_rms(phi0)
    _, rms_mid = kspace_rms(snaps[len(snaps) // 2][1])
    print("\nLow-k shell amplitude (|phi_k| rms):")
    for n in [0, 1, 2, 3, 4]:
        i0 = np.where(kn == n)[0]
        if len(i0):
            i = int(i0[0])
            print(f"  n={n}: IC={rms0[i]:.5f}  mid-run={rms_mid[i]:.5f}  ratio={rms_mid[i]/max(rms0[i],1e-12):.2f}")

    # Coherence at last snapshot with |phi|>500 if any
    late = snaps[-1][1]
    for step, phi, _ in reversed(snaps):
        if (np.abs(phi) > 500).mean() > 0.1:
            late = phi
            late_step = step
            break
    else:
        late_step = snaps[-1][0]
    frac500 = float((np.abs(late) > 500).mean())
    pos = float((late > 0).mean())
    # neighbor sign agreement (1 = uniform domain, 0.5 = random)
    s = np.sign(late)
    agree = 0
    pairs = 0
    for ax in range(3):
        a, b = s, np.roll(s, -1, axis=ax)
        mask = (a != 0) & (b != 0)
        pairs += int(mask.sum())
        agree += int((a[mask] == b[mask]).sum())
    agree_frac = agree / max(pairs, 1)

    print(f"\nLate-time coherence (step {late_step}):")
    print(f"  frac |phi|>500 = {frac500:.3f}  (≈1 => lattice-wide roll)")
    print(f"  positive fraction = {pos:.3f}  (≈0.5 expected for Z2-symmetric V)")
    print(f"  neighbor same-sign = {agree_frac:.3f}  (0.5=random, →1=large domains)")

    print("\nInterpretation:")
    print("  Below spinodal, |phi| should grow lattice-wide (frac>500 → 1).")
    print("  Z2-symmetric V ⇒ ± domains (mean≈0, pos_frac≈0.5) is correct spinodal.")
    print("  Sparse corner spikes + frac>500≪1 ⇒ broken ghosts / wrong dynamics.")


if __name__ == "__main__":
    main()
