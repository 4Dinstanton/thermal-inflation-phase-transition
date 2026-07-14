#!/usr/bin/env python3
"""
3D roll diagnostics: spatial Laplacian vs V', V' table parity, hot-spot roll,
and tail-site analysis on CL vs numba snapshots.

Usage
-----
    python tools/diagnose_roll_3d.py
    python tools/diagnose_roll_3d.py --plot
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

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
    rk2_fused_4pass_field,
    vprime,
    vprime_field,
)

TABLE_BIN = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")
NUMBA_SNAP = os.path.join(
    REPO,
    "data/lattice/set8/256x256x256_T0_1230_dx_0.001_dtphys_0.0001_interval_4000"
    "_3D_hubble_eta_1230_gb_1.09_gf_1.09_rk2_fused_inline_V_correct/field_states"
    "/state_step_0000004000.npz",
)
CL_SNAP = os.path.join(
    REPO,
    "data/lattice/set8/256x256x256_T0_1230_dx_0.001_dtphys_0.0001_interval_4000"
    "_3D_hubble_eta_1230_gb_1.09_gf_1.09_nb_20_nf_20_stochasticrk_V_correct_CL"
    "/field_states/snapshot_0000004000.raw",
)
HDR = struct.calcsize("<IIq5d")


def load_cl_raw(path: str) -> tuple[np.ndarray, float, int]:
    with open(path, "rb") as f:
        magic, n, step, t, T, a, H, fStar = struct.unpack("<IIq5d", f.read(HDR))
        phi = np.frombuffer(f.read(4 * n**3), dtype=np.float32).reshape(n, n, n) * fStar
    return phi, float(T), int(step)


def load_table_eval():
    """Load C++-matching TableEval from thermal_tables.npz."""
    from export_thermal_splines import TableEval, DEFAULT_PARAMS  # noqa: WPS433

    npz_path = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.npz")
    d = np.load(npz_path)
    params = dict(DEFAULT_PARAMS)
    gamma = 4.1667e-4
    params["gamma"] = gamma
    params["lam"] = params["mphi"] ** 2 / (gamma * 2.4e18) ** 2
    return TableEval(d["u"], d["Jb"], d["Jf"], d["dJb"], d["dJf"], d["d2Jb"], d["d2Jf"], params)


def test_vprime_parity(T: float = 1172.22):
    print("\n" + "=" * 70)
    print(f"[A] V'(phi, T={T:.1f}) — numba inline vs CL table (include_cw=0 vs 1)")
    print("=" * 70)
    ev = load_table_eval()
    print(f"  {'phi':>8s}  {'numba':>14s}  {'CL cw=0':>14s}  {'CL cw=1':>14s}  {'rel err cw=0':>12s}")
    for phi in [0, 470, 1000, 1500, 2000, 2646, 5000, 10000]:
        vn = vprime(phi, T)
        vc0 = float(ev.Vprime(phi, T, include_cw=False))
        vc1 = float(ev.Vprime(phi, T, include_cw=True))
        rel = abs(vc0 - vn) / max(abs(vn), 1.0)
        flag = " ***" if rel > 0.05 and phi >= 1000 else ""
        print(f"  {phi:8.0f}  {vn:14.4e}  {vc0:14.4e}  {vc1:14.4e}  {rel:12.3e}{flag}")


def roll_3d_hot_spot(
    n: int,
    phi_hot: float,
    phi_bg: float,
    n_steps: int,
    *,
    pi_hot: float = 0.0,
    noise: bool = False,
    seed: int = 42,
) -> dict:
    """Deterministic 3D hot spot; returns center phi history."""
    rng = np.random.default_rng(seed)
    hs = HubbleState()
    while hs.T() > 1172.5:
        hs.advance()
    inv_a2 = 1.0 / hs.a**2
    eta_eff = ETA + 3.0 * hs.T() / hs.a  # approximate; refreshed each step below

    phi = np.full((n, n, n), phi_bg, dtype=np.float64)
    pi = np.zeros_like(phi)
    c = n // 2
    phi[c, c, c] = phi_hot
    pi[c, c, c] = pi_hot

    center_hist = [phi_hot]
    max_hist = [float(np.max(np.abs(phi)))]
    for _ in range(n_steps):
        T_now, T_mid, eta_eff, inv_a2, _, ns = hs.advance()
        if not noise:
            ns = 0.0
        phi, pi = rk2_fused_4pass_field(
            phi, pi, T_now, T_mid, eta_eff, inv_a2, ns, rng, cl_sqrt2=False, dx=DX_PROG
        )
        center_hist.append(float(phi[c, c, c]))
        max_hist.append(float(np.max(np.abs(phi))))
        if not np.isfinite(max_hist[-1]):
            break
    return {
        "center": np.array(center_hist),
        "max": np.array(max_hist),
        "final_phi": phi,
        "final_pi": pi,
    }


def roll_3d_uniform(phi0: float, n: int, n_steps: int) -> dict:
    """Uniform phi=phi0, pi=0 — Laplacian=0 everywhere (true 3D but homogeneous)."""
    rng = np.random.default_rng(0)
    hs = HubbleState()
    while hs.T() > 1172.5:
        hs.advance()
    phi = np.full((n, n, n), phi0, dtype=np.float64)
    pi = np.zeros_like(phi)
    max_hist = [phi0]
    for _ in range(n_steps):
        T_now, T_mid, eta_eff, inv_a2, _, _ = hs.advance()
        phi, pi = rk2_fused_4pass_field(
            phi, pi, T_now, T_mid, eta_eff, inv_a2, 0.0, rng, False, dx=DX_PROG
        )
        max_hist.append(float(np.max(np.abs(phi))))
        if not np.isfinite(max_hist[-1]):
            break
    return {"max": np.array(max_hist)}


def test_3d_roll_scenarios():
    print("\n" + "=" * 70)
    print("[B] 3D deterministic roll scenarios (numba integrator, T~1172)")
    print("=" * 70)

    cases = [
        ("7^3 hot 2000/bg400 pi=0", dict(n=7, phi_hot=2000, phi_bg=400, n_steps=50, pi_hot=0)),
        ("7^3 hot 2000/bg400 pi=+500", dict(n=7, phi_hot=2000, phi_bg=400, n_steps=50, pi_hot=500)),
        ("7^3 hot 1500/bg400 pi=0", dict(n=7, phi_hot=1500, phi_bg=400, n_steps=80, pi_hot=0)),
        ("32^3 hot 2000/bg400 pi=0", dict(n=32, phi_hot=2000, phi_bg=400, n_steps=50, pi_hot=0)),
        ("32^3 hot 2000/bg400 pi=+800", dict(n=32, phi_hot=2000, phi_bg=400, n_steps=50, pi_hot=800)),
        ("32^3 uniform phi=1500", None),
    ]
    for label, kw in cases:
        if kw is None:
            r = roll_3d_uniform(1500.0, 32, 80)
            end = r["max"][-1] if np.isfinite(r["max"][-1]) else float("inf")
            print(f"  {label:32s}  start=1500  end max|phi|={end:.4e}  rolled={'YES' if end > 1e4 else 'no'}")
            continue
        r = roll_3d_hot_spot(**kw)
        c0, c_end = r["center"][0], r["center"][-1]
        m_end = r["max"][-1] if np.isfinite(r["max"][-1]) else float("inf")
        rolled = m_end > 1e4 or (np.isfinite(m_end) and m_end > 5000)
        print(
            f"  {label:32s}  center {c0:.0f}->{c_end:.4e}  max|phi|={m_end:.4e}  "
            f"rolled={'YES' if rolled else 'NO (lap/friction wins)'}"
        )


def force_balance_at_site(phi: np.ndarray, i: int, j: int, k: int, T: float, pi: float = 0.0):
    """Estimate terms in k_pi at one site (GeV units)."""
    from diagnose_roll_dynamics import hubble  # noqa: WPS433

    hs = HubbleState()
    hs.a = T0 / T
    inv_a2 = 1.0 / hs.a**2
    H = hubble(T)
    eta_eff = ETA + 3.0 * H / MU
    inv_mu2 = 1.0 / MU**2
    lap = laplacian_3d(phi, DX_PROG)
    ph = phi[i, j, k]
    lap_g = inv_a2 * lap[i, j, k]
    vp = vprime(float(ph), T)
    kpi = lap_g - eta_eff * pi - vp * inv_mu2
    return dict(phi=float(ph), lap=lap_g, vp=vp, friction=-eta_eff * pi, kpi=kpi)


def analyze_snapshot_tail(path: str, label: str, is_npz: bool):
    print("\n" + "=" * 70)
    print(f"[C] Tail-site force balance on {label} snapshot @ step 4000")
    print("=" * 70)
    if is_npz:
        d = np.load(path)
        phi = d["phi"]
        T = float(d["temperature"])
    else:
        phi, T, _ = load_cl_raw(path)

    abs_phi = np.abs(phi)
    n_tail = int(np.sum(abs_phi > 1500))
    print(f"  T={T:.2f} GeV  sites |phi|>1500: {n_tail}  max|phi|={abs_phi.max():.1f}")

    # Global phi stats
    print(f"  std(phi)={phi.std():.1f}  p99.9={np.percentile(abs_phi, 99.9):.1f}  "
          f"frac>5k={np.mean(abs_phi > 5000):.4e}")

    # Top-5 max sites: force balance with pi=0 (instantaneous roll tendency)
    flat_idx = np.argsort(abs_phi.ravel())[-5:][::-1]
    nx = phi.shape[0]
    print(f"\n  Top 5 |phi| sites (k_pi with pi=0 — negative kpi => roll tendency on pi):")
    print(f"  {'rank':>4s}  {'|phi|':>8s}  {'lap':>12s}  {'Vprime':>12s}  {'-Vp/mu2':>12s}  {'lap-Vp/mu2':>14s}")
    for rank, fi in enumerate(flat_idx, 1):
        k = fi % nx
        j = (fi // nx) % nx
        i = fi // (nx * nx)
        fb = force_balance_at_site(phi, i, j, k, T, pi=0.0)
        roll_term = -fb["vp"] / (MU**2)
        net = fb["lap"] + roll_term
        print(
            f"  {rank:4d}  {fb['phi']:8.1f}  {fb['lap']:12.2e}  {fb['vp']:12.2e}  "
            f"{roll_term:12.2e}  {net:14.2e}"
        )

    # Numba rolled sites: check if max site has strong roll tendency
    if is_npz:
        imax = np.unravel_index(np.argmax(abs_phi), phi.shape)
        fb = force_balance_at_site(phi, *imax, T, pi=0.0)
        print(f"\n  Global max site {imax}: |phi|={fb['phi']:.1f}  "
              f"lap={fb['lap']:.2e}  V'={fb['vp']:.2e}  net k_pi|_pi0={fb['kpi']:.2e}")


def compare_numba_max_bubble(path: str):
    """Track numba max site neighborhood."""
    d = np.load(path)
    phi = d["phi"]
    T = float(d["temperature"])
    imax = np.unravel_index(np.argmax(np.abs(phi)), phi.shape)
    i, j, k = imax
    nx = phi.shape[0]
    print("\n" + "=" * 70)
    print("[D] Numba rolled bubble — neighborhood at global max")
    print("=" * 70)
    print(f"  max at ({i},{j},{k}): phi={phi[i,j,k]:.1f} GeV  T={T:.2f}")
    for di, dj, dk, name in [
        (0, 0, 0, "center"),
        (1, 0, 0, "i+1"),
        (-1, 0, 0, "i-1"),
        (0, 1, 0, "j+1"),
    ]:
        ii = (i + di) % nx
        jj = (j + dj) % nx
        kk = (k + dk) % nx
        print(f"    {name:6s}: phi={phi[ii,jj,kk]:12.1f} GeV")


def test_laplacian_drag():
    print("\n" + "=" * 70)
    print("[E] Laplacian drag on isolated hot spot (analytic at step 0)")
    print("=" * 70)
    n = 7
    phi = np.full((n, n, n), 400.0)
    phi[n // 2, n // 2, n // 2] = 2000.0
    lap = laplacian_3d(phi, DX_PROG)
    c = n // 2
    hs = HubbleState()
    hs.a = T0 / 1172.22
    inv_a2 = 1.0 / hs.a**2
    lap_g = inv_a2 * lap[c, c, c]
    vp = vprime(2000.0, 1172.22)
    roll_push = -vp / MU**2
    print(f"  Center phi=2000, neighbors=400, dx_prog={DX_PROG}")
    print(f"  lap (GeV) = {lap_g:.4e}  (positive => pushes phi down at center)")
    print(f"  -V'/mu^2 = {roll_push:.4e}  (negative => roll push on pi)")
    print(f"  net k_pi (pi=0) = {lap_g + roll_push:.4e}")
    print(f"  |lap|/|V'/mu^2| = {abs(lap_g)/abs(roll_push):.4f}")
    if abs(lap_g) > abs(roll_push):
        print("  => Laplacian CAN overwhelm roll at isolated hot spot without pi seed")
    else:
        print("  => V' roll term dominates at hot spot (should roll if pi builds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    test_vprime_parity()
    test_laplacian_drag()
    test_3d_roll_scenarios()

    if os.path.exists(NUMBA_SNAP):
        analyze_snapshot_tail(NUMBA_SNAP, "numba", True)
        compare_numba_max_bubble(NUMBA_SNAP)
    else:
        print(f"\n  (skip numba snapshot: not found)")

    if os.path.exists(CL_SNAP):
        analyze_snapshot_tail(CL_SNAP, "CL", False)
    else:
        print(f"\n  (skip CL snapshot: not found)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("  • If V' table (cw=0) matches numba → roll force is not the bug.")
    print("  • If 3D hot spot does NOT roll without pi seed → Laplacian + friction stall roll.")
    print("  • If numba max site has phi~48k but neighbors ~400 → bubble already rolled.")
    print("  • CL tail sites with |phi|~2.6k but k_pi~0 → stuck at thermal-Laplacian balance.")


if __name__ == "__main__":
    main()
