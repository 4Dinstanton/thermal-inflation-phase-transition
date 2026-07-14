#!/usr/bin/env python3
"""
Deep V'(phi,T) audit: compare all implementations used in numba vs CosmoLattice.

Implementations compared
------------------------
  exact     : CTFT.dJb_exact / dJf_exact (quadrature reference)
  numba     : cubic spline on dJ(u), u in [0,100], 256 pts (latticeSimeRescale_numba.py)
  cl_table  : C++ thermal_tables.hpp Hermite (via compiled vprime_audit)
  py_linear : Python TableEval with np.interp (legacy; NOT what C++ uses)

Also reports spinodal / barrier / roll-sign at T=1172 and T=1230.

Usage
-----
    python tools/diagnose_vprime.py
    python tools/diagnose_vprime.py --plot
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicSpline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "potential"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import cosmoTransitions.finiteT as CTFT  # noqa: E402
from export_thermal_splines import TableEval, DEFAULT_PARAMS  # noqa: E402

# Model params (256^3 T0=1230 set)
M_PL = 2.4e18
GAMMA = 4.1667e-4
FSTAR = GAMMA * M_PL
MPHI = 1000.0
LAM = MPHI**2 / FSTAR**2
NB = NF = 20.0
YB, GB = 1.09, 1.05
YF, GF = 1.09, 1.05
MB2 = 1e6
COEF_B = 0.25 * YB**2 + (2.0 / 3.0) * GB**2
MU = 1000.0

VP_LINE = re.compile(
    r"phi=\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)\s+"
    r"Vp=\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)\s+"
    r"Vpp=\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
)

TABLE_BIN = os.path.join(REPO, "data", "thermal_splines", "thermal_tables.bin")
AUDIT_CPP = os.path.join(REPO, "cosmolattice_ext", "tests", "vprime_audit.cpp")
AUDIT_BIN = "/tmp/vprime_audit"


def build_numba_spline():
    """Replicate latticeSimeRescale_numba.py dJ spline (YMAX=100, N_Y=256)."""
    y_grid = np.linspace(0.0, 100.0, 256, dtype=np.float64)
    dJb = np.array([float(np.real(CTFT.dJb_exact(y))) for y in y_grid])
    dJf = np.array([float(np.real(CTFT.dJf_exact(y))) for y in y_grid])
    cs_b = CubicSpline(y_grid, dJb, bc_type="not-a-knot")
    cs_f = CubicSpline(y_grid, dJf, bc_type="not-a-knot")
    return y_grid, cs_b, cs_f


Y_GRID, CS_B, CS_F = build_numba_spline()
X_MIN = float(Y_GRID[0])
H_Y = float(Y_GRID[1] - Y_GRID[0])
NSEG = len(Y_GRID) - 1


def vprime_numba(phi: float, T: float) -> float:
    """Vprime_scalar / _vprime_at_site from latticeSimeRescale_numba.py."""
    dV = LAM * phi**3 - MPHI**2 * phi
    pref = T**4 / (2.0 * math.pi**2)

    xb_sq = MB2 + 0.5 * YB**2 * phi**2 + COEF_B * T**2
    if xb_sq > 0:
        xb = math.sqrt(xb_sq) / T
        xb_c = min(max(xb, X_MIN), X_MIN + H_Y * NSEG - 1e-12)
        idx = int((xb_c - X_MIN) / H_Y)
        idx = max(0, min(idx, NSEG - 1))
        dx = xb_c - (X_MIN + idx * H_Y)
        c0, c1, c2, c3 = CS_B.c[:, idx]
        dJb = ((c0 * dx + c1) * dx + c2) * dx + c3
        dub = 0.5 * YB**2 * phi / (T**2 * max(xb, 1e-20))
        dV += pref * NB * dJb * dub

    xf_sq = 0.5 * YF**2 * phi**2 + (1.0 / 6.0) * GF**2 * T**2
    if xf_sq > 0:
        xf = math.sqrt(xf_sq) / T
        xf_c = min(max(xf, X_MIN), X_MIN + H_Y * NSEG - 1e-12)
        idx = int((xf_c - X_MIN) / H_Y)
        idx = max(0, min(idx, NSEG - 1))
        dx = xf_c - (X_MIN + idx * H_Y)
        c0, c1, c2, c3 = CS_F.c[:, idx]
        dJf = ((c0 * dx + c1) * dx + c2) * dx + c3
        duf = 0.5 * YF**2 * phi / (T**2 * max(xf, 1e-20))
        dV += pref * NF * dJf * duf
    return dV


def vprime_exact(phi: float, T: float) -> float:
    pref = T**4 / (2.0 * math.pi**2)
    dV = LAM * phi**3 - MPHI**2 * phi
    xb_sq = MB2 + 0.5 * YB**2 * phi**2 + COEF_B * T**2
    if xb_sq > 0:
        xb = math.sqrt(xb_sq) / T
        dJb = float(np.real(CTFT.dJb_exact(xb)))
        dub = 0.5 * YB**2 * phi / (T**2 * max(xb, 1e-20))
        dV += pref * NB * dJb * dub
    xf_sq = 0.5 * YF**2 * phi**2 + (1.0 / 6.0) * GF**2 * T**2
    if xf_sq > 0:
        xf = math.sqrt(xf_sq) / T
        dJf = float(np.real(CTFT.dJf_exact(xf)))
        duf = 0.5 * YF**2 * phi / (T**2 * max(xf, 1e-20))
        dV += pref * NF * dJf * duf
    return dV


def vprime_second_exact(phi: float, T: float, h: float | None = None) -> float:
    if h is None:
        h = max(abs(phi) * 1e-5, 1.0)
    return (vprime_exact(phi + h, T) - vprime_exact(phi - h, T)) / (2.0 * h)


def load_py_table_eval() -> TableEval:
    d = np.load(os.path.join(REPO, "data", "thermal_splines", "thermal_tables.npz"))
    params = dict(DEFAULT_PARAMS)
    params["gamma"] = GAMMA
    params["lam"] = LAM
    return TableEval(d["u"], d["Jb"], d["Jf"], d["dJb"], d["dJf"], d["d2Jb"], d["d2Jf"], params)


def compile_cl_audit() -> None:
    inc = os.path.join(REPO, "cosmolattice_ext", "models")
    subprocess.check_call(
        ["g++", "-std=c++14", "-O2", "-I", inc, AUDIT_CPP, "-o", AUDIT_BIN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_cl_audit(include_cw: bool) -> dict[tuple[float, float], tuple[float, float]]:
    """Return {(T, phi): (Vp, Vpp)} from C++ audit."""
    flag = "1" if include_cw else "0"
    out = subprocess.check_output([AUDIT_BIN, TABLE_BIN, flag], text=True)
    data: dict[tuple[float, float], tuple[float, float]] = {}
    T_cur = None
    for line in out.splitlines():
        if line.startswith("T="):
            T_cur = float(line.split()[0].split("=")[1])
            continue
        m = VP_LINE.search(line)
        if m:
            phi = float(m.group(1))
            vp = float(m.group(2))
            vpp = float(m.group(3))
            data[(T_cur, phi)] = (vp, vpp)
            continue
        if line.strip().startswith("phi="):
            continue
    return data


def find_spinodal(T: float, vprime_fn, phi_max: float = 2000.0) -> float | None:
    """First phi>0 where V'(phi,T) crosses zero from + to - (barrier top / spinodal)."""
    phis = np.logspace(-2, math.log10(phi_max), 500)
    prev_p, prev_v = phis[0], vprime_fn(phis[0], T)
    for phi in phis[1:]:
        v = vprime_fn(phi, T)
        if prev_v > 0 and v < 0:
            # linear refine
            return float(prev_p + (0 - prev_v) * (phi - prev_p) / (v - prev_v))
        prev_p, prev_v = phi, v
    return None


def roll_push(phi: float, T: float, vp: float) -> float:
    """-V'/mu^2 term in k_pi (GeV units)."""
    return -vp / (MU**2)


def audit_grid(T: float, phis: list[float], cl_data: dict, py_ev: TableEval):
    print(f"\n{'='*78}")
    print(f"V' audit at T={T:.2f} GeV  (include_cw=0 for CL)")
    print(f"{'='*78}")
    print(
        f"{'phi':>8s}  {'exact':>12s}  {'numba':>12s}  {'CL C++':>12s}  "
        f"{'nb-ex':>8s}  {'CL-ex':>8s}  {'sign roll':>10s}"
    )
    max_nb_err = 0.0
    max_cl_err = 0.0
    for phi in phis:
        ve = vprime_exact(phi, T)
        vn = vprime_numba(phi, T)
        vc = cl_data.get((T, phi), cl_data.get((round(T, 2), phi)))
        if vc is None:
            # try nearest T key
            keys = [k for k in cl_data if abs(k[0] - T) < 1 and k[1] == phi]
            vc = cl_data[keys[0]] if keys else (float("nan"), float("nan"))
        vp_cl = vc[0] if isinstance(vc, tuple) else float("nan")

        err_nb = abs(vn - ve) / max(abs(ve), 1.0)
        err_cl = abs(vp_cl - ve) / max(abs(ve), 1.0)
        max_nb_err = max(max_nb_err, err_nb)
        max_cl_err = max(max_cl_err, err_cl)

        roll = "DOWN" if ve < 0 else "UP  "
        flag = " ***" if err_cl > 0.05 or err_nb > 0.05 else ""
        print(
            f"{phi:8.0f}  {ve:12.4e}  {vn:12.4e}  {vp_cl:12.4e}  "
            f"{err_nb:8.3e}  {err_cl:8.3e}  {roll}{flag}"
        )
    print(f"\n  max |numba-exact|/|exact| = {max_nb_err:.3e}")
    print(f"  max |CL C++ -exact|/|exact| = {max_cl_err:.3e}")

    # py linear TableEval (wrong mirror)
    print(f"\n  Python TableEval (linear dJ interp) vs C++ Hermite at sample points:")
    for phi in [470, 1000, 2000]:
        vp_py = float(py_ev.Vprime(phi, T, include_cw=False))
        vp_cl = cl_data.get((T, phi), (0, 0))[0]
        print(f"    phi={phi:.0f}: py_linear={vp_py:+.4e}  CL_cpp={vp_cl:+.4e}  diff={vp_py-vp_cl:+.4e}")


def spinodal_report(T: float, cl_data: dict):
    print(f"\n--- Spinodal / barrier structure at T={T:.2f} ---")
    for name, fn in [("exact", vprime_exact), ("numba spline", vprime_numba)]:
        sp = find_spinodal(T, fn)
        v0 = vprime_exact(0, T)
        vpp0 = vprime_second_exact(0, T)
        print(f"  {name:14s}: V'(0)={v0:+.4e}  V''(0)={vpp0:+.4e}  spinodal/barrier phi={sp:.1f} GeV" if sp else
              f"  {name:14s}: V'(0)={v0:+.4e}  V''(0)={vpp0:+.4e}  no + to - crossing found")

    # CL V'' at phi=0 from C++
    vp0, vpp0 = cl_data.get((T, 0.0), (0, 0))
    print(f"  CL C++ table  : V'(0)={vp0:+.4e}  V''(0)={vpp0:+.4e}")

    # Roll push at CL tail values
    print(f"\n  Roll push -V'/mu^2 at tail phi (exact):")
    for phi in [400, 470, 1000, 1500, 2000, 2646]:
        ve = vprime_exact(phi, T)
        print(f"    phi={phi:5.0f}: V'={ve:+.4e}  roll push={roll_push(phi,T,ve):+.2e} GeV")


def cw_impact(T: float, phi: float, cl_cw0: dict, cl_cw1: dict):
    vp0 = cl_cw0.get((T, phi), (0, 0))[0]
    vp1 = cl_cw1.get((T, phi), (0, 0))[0]
    return vp0, vp1, vp1 - vp0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print("Compiling C++ V' audit...")
    compile_cl_audit()
    cl0 = run_cl_audit(include_cw=False)
    cl1 = run_cl_audit(include_cw=True)
    py_ev = load_py_table_eval()

    phis = [
        0, 100, 200, 300, 400, 470, 500, 600, 700, 800, 900, 1000,
        1200, 1500, 2000, 2646, 5000, 10000,
    ]
    for T in [1230.0, 1172.22]:
        audit_grid(T, phis, cl0, py_ev)
        spinodal_report(T, cl0)

    print(f"\n{'='*78}")
    print("include_cw=1 vs 0 (CL C++ table) — runs with default include_cw=1 are WRONG")
    print(f"{'='*78}")
    for T in [1172.22]:
        for phi in [0, 470, 1000, 2000, 2646]:
            vp0, vp1, d = cw_impact(T, phi, cl0, cl1)
            print(f"  T={T:.1f} phi={phi:5.0f}: cw=0 V'={vp0:+.4e}  cw=1 V'={vp1:+.4e}  delta={d:+.4e}")

    # Check input.in for CL runs
    cl_in = os.path.join(
        REPO,
        "data/lattice/set8/256x256x256_T0_1230_dx_0.001_dtphys_0.0001_interval_4000"
        "_3D_hubble_eta_1230_gb_1.09_gf_1.09_nb_20_nf_20_stochasticrk_V_correct_CL/input.in",
    )
    if os.path.exists(cl_in):
        with open(cl_in) as f:
            for line in f:
                if "include_cw" in line:
                    print(f"\n  Current CL input.in: {line.strip()}")

    print(f"\n{'='*78}")
    print("Verdict")
    print(f"{'='*78}")
    print("  1. Numba runtime uses 256-pt cubic spline on dJ(u); CL uses 4096-pt Hermite.")
    print("  2. Both should match CTFT exact; large errors indicate a real bug.")
    print("  3. Python TableEval (linear) != C++ Hermite — do not use for CL parity.")
    print("  4. include_cw=1 adds CW force numba does NOT have — breaks roll parity.")
    print("  5. Spinodal phi sets minimum |phi| for roll; check vs CL tail max ~2646.")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        T = 1172.22
        phis_d = np.logspace(-1, 3.5, 400)
        ve = [vprime_exact(p, T) for p in phis_d]
        vn = [vprime_numba(p, T) for p in phis_d]
        vc = [cl0.get((T, p), (float("nan"),))[0] for p in phis_d]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].semilogx(phis_d, ve, label="exact")
        ax[0].semilogx(phis_d, vn, "--", label="numba spline")
        ax[0].semilogx(phis_d, vc, ":", label="CL C++")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_xlabel("|phi| (GeV)")
        ax[0].set_ylabel("V'(phi,T)")
        ax[0].set_title(f"V' at T={T:.0f}")
        ax[0].legend()
        rel_cl = [abs(c - e) / max(abs(e), 1) for c, e in zip(vc, ve)]
        rel_nb = [abs(n - e) / max(abs(e), 1) for n, e in zip(vn, ve)]
        ax[1].semilogx(phis_d, rel_nb, label="|numba-exact|/|exact|")
        ax[1].semilogx(phis_d, rel_cl, label="|CL-exact|/|exact|")
        ax[1].set_xlabel("|phi| (GeV)")
        ax[1].set_ylabel("relative error")
        ax[1].set_title("V' relative error")
        ax[1].legend()
        fig.tight_layout()
        out = os.path.join(REPO, "figs", "diagnose_vprime.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=120)
        print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
