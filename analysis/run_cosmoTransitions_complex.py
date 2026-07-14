#!/usr/bin/env python3
"""
CosmoTransitions pipeline for the complex-scalar (O(2)/U(1)) thermal model.

For zn_order = 0 the finite-T potential depends only on rho = |phi|, so the
radial bounce on V(rho) matches the real-scalar nucleation physics.  With
zn_order > 0 an angular Z_N term is added (approximate: still tunnel on rho
with V_eff(rho) = V_correct(rho); full 2D CT deferred).

Outputs (under data/cosmotransitions/<tag>/):
  - S3T_scan.csv          : T, S3, S3/T, r_c, phi_esc
  - transition_summary.json : T_c2, T_n, T_p, T_c1, gamma, delV, ...
  - nucleation_condition.png
  - S3T_vs_T.png

No lattice simulation — compare these numbers to CosmoLattice T_p / snapshots later.

Example (set8, matches run_cosmolattice.py defaults):
  python analysis/run_cosmoTransitions_complex.py \\
      --param_set set8 --gamma 4.1667e-4 --mphi 1000 --T0 1230 \\
      --zn_order 0 --potential_type V_correct
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "potential"))
sys.path.insert(0, os.path.join(REPO, "analysis"))

import Potential as p  # noqa: E402
import cosmoTransitions.pathDeformation as CTPD  # noqa: E402
from tunneling_utils import fullTunneling  # noqa: E402

M_PL = 2.4e18  # GeV


def _pot_callables(VT, potential_type):
    if potential_type == "fermion_only":
        return VT.V_fermion_only, VT.dV_p_fermion_only
    return VT.V_correct, VT.dV_p_correct


def _V_at(VT, phi_val, potential_type, h=1.0):
    X = np.array([[phi_val]])
    V, _ = _pot_callables(VT, potential_type)
    return float(V(X))


def _V_second_derivative_at_origin(VT, potential_type, h=1.0):
    return (
        _V_at(VT, h, potential_type)
        - 2.0 * _V_at(VT, 0.0, potential_type)
        + _V_at(VT, -h, potential_type)
    ) / h**2


def find_T_c2(VT, potential_type, T_lo=100.0, T_hi=50000.0, n_coarse=600, fd_step=1.0):
    """T where V''(phi=0)=0 (spinodal / barrier at origin disappears)."""
    T_arr = np.linspace(T_lo, T_hi, n_coarse)
    d2 = np.empty(n_coarse)
    for i, T in enumerate(T_arr):
        VT.update_T(T)
        d2[i] = _V_second_derivative_at_origin(VT, potential_type, h=fd_step)
    sc = np.where(np.diff(np.sign(d2)))[0]
    if len(sc) == 0:
        return None, T_arr, d2

    idx = sc[0]

    def res(T):
        VT.update_T(T)
        return _V_second_derivative_at_origin(VT, potential_type, h=fd_step)

    T_c2 = opt.brentq(res, T_arr[idx], T_arr[idx + 1], xtol=1e-4)
    return float(T_c2), T_arr, d2


def _make_clipped_potential(V_func, dV_func, phi_cutoff, wall_k=1.0):
    def V_clipped(X):
        val = np.asarray(V_func(X), dtype=float)
        phi = X[..., 0]
        excess = np.abs(phi) - phi_cutoff
        mask = excess > 0
        if np.any(mask):
            val = np.array(val, copy=True)
            val[mask] += 0.5 * wall_k * excess[mask] ** 2
        return val

    def dV_clipped(X):
        val = np.asarray(dV_func(X), dtype=float).copy()
        phi = X[..., 0]
        excess = np.abs(phi) - phi_cutoff
        mask = excess > 0
        if np.any(mask):
            val[mask] += wall_k * excess[mask] * np.sign(phi[mask])
        return val

    return V_clipped, dV_clipped


def compute_S3_over_T(T, param_dict, spline_arrays, potential_type, zn_order, zn_strength):
    """O(3) radial bounce on V(rho). Z_N angular term omitted when zn_order=0."""
    VT = p.finiteTemperaturePotential(param_dict)
    VT.update_T(T)
    VT.set_fast_thermal_from_arrays(*spline_arrays)

    V_raw, dV_raw = _pot_callables(VT, potential_type)
    if zn_order > 0 and zn_strength > 0:
        # Approximate: add constant Z_N well depth at fixed rho (conservative placeholder).
        delta = zn_strength

        def V_zn(X):
            return V_raw(X) + delta * (1.0 - np.cos(zn_order * 0.0))

        def dV_zn(X):
            return dV_raw(X)

        V_raw, dV_raw = V_zn, dV_zn

    phi_cutoff = max(15.0 * T, 50000.0)
    mphi = param_dict["mphi"]
    wall_k = 100.0 * mphi**2
    V_func, dV_func = _make_clipped_potential(V_raw, dV_raw, phi_cutoff, wall_k)

    tv = max(10.0 * T, 50000.0)
    fv = 0.0
    try:
        tunnel = fullTunneling(
            path_pts=np.array([[tv], [fv]]),
            V=V_func,
            dV=dV_func,
            maxiter=1,
            V_spline_samples=200,
            tunneling_init_params=dict(alpha=2),
            tunneling_findProfile_params=dict(
                xtol=1e-5, phitol=1e-5, rmin=1e-5, npoints=200
            ),
            deformation_class=CTPD.Deformation_Spline,
            extend_to_minima=False,
        )
    except Exception as exc:
        return dict(ok=False, error=str(exc), T=T)

    S3 = float(tunnel.action)
    S3T = S3 / T
    R = tunnel.profile1D.R
    Phi = tunnel.profile1D.Phi
    phi_mid = 0.5 * (Phi[0] + Phi[-1])
    if Phi[0] > Phi[-1]:
        r_c = float(np.interp(phi_mid, Phi[::-1], R[::-1]))
    else:
        r_c = float(np.interp(phi_mid, Phi, R))
    phi_esc = float(tv - Phi[0])
    return dict(ok=True, T=T, S3=S3, S3_over_T=S3T, r_c=r_c, phi_esc=phi_esc)


def nucleation_rhs(T, mphi, gamma):
    return 4.0 * math.log(2.0 * math.sqrt(3.0) * T / (mphi * gamma))


def find_T_n(T_grid, S3T_grid, mphi, gamma):
    rhs = np.array([nucleation_rhs(T, mphi, gamma) for T in T_grid])
    diff = S3T_grid - rhs
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return None
    i = sc[0]
    spl = CubicSpline(T_grid, diff)

    def res(T):
        return float(spl(T))

    return float(opt.brentq(res, T_grid[i], T_grid[i + 1], xtol=1e-3))


def hubble(T, delV):
    chig2 = 30.0 / (math.pi**2 * 106.75)
    return math.sqrt((T**4 / chig2 + delV) / (3.0 * M_PL**2))


def fit_S3T_spline(T_arr, S3T_arr):
    return CubicSpline(T_arr, S3T_arr, extrapolate=False)


def fit_log_gamma(T_arr, S3T_arr):
    """Fit log(Gamma) = -S3/T + 4 ln T + 3/2 ln(S3/T / 2pi) with rev()."""
    log_G = (
        -S3T_arr
        + 4.0 * np.log(T_arr)
        + 1.5 * np.log(S3T_arr / (2.0 * math.pi))
    )

    def rev(x, a, b, c, d, e, f):
        return a + b * x + c * x**2 + d * x**3 + e * np.exp(-f * x)

    p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1e-4]
    popt, _ = curve_fit(rev, T_arr, log_G, maxfev=10000, p0=p0)
    return popt, rev


def _inner_integral(T, Tp, delV):
    return quad(lambda x: 1.0 / hubble(x, delV), T, Tp, limit=100)[0]


def nT_integral(T, popt, rev, delV, T_max=15000.0):
    H4 = hubble(T, delV) ** 4

    def f(y):
        return math.exp(rev(y, *popt)) / (H4 * y)

    return quad(f, T, T_max, limit=100)[0]


def percolation_I(T, popt, rev, delV, T_max=15000.0):
    """I(T) percolation integral (plotCouplingComparison / drawAction)."""
    prefactor = 4.0 * math.pi / 3.0

    def outer_integrand(Tp):
        J = _inner_integral(T, Tp, delV)
        return math.exp(rev(Tp, *popt)) / (hubble(Tp, delV) * Tp**4) * J**3

    return prefactor * quad(outer_integrand, T, T_max, limit=100)[0]


def find_T_from_percolation_scan(I_vals, T_arr, P_target):
    """T minimizing |P/P_target - 1| on a precomputed grid."""
    valid = np.isfinite(I_vals)
    if valid.sum() < 3:
        return None
    t_v = T_arr[valid]
    I_v = I_vals[valid]
    P_v = np.exp(np.clip(-I_v, -700.0, 700.0))
    if abs(P_target - 0.7) < 1e-12:
        idx = int(np.argmin(np.abs(P_v / 0.7 - 1.0)))
    else:
        idx = int(np.argmin(np.abs(P_v - P_target)))
    return float(t_v[idx])


def main():
    ap = argparse.ArgumentParser(description="CosmoTransitions set8 complex-scalar table")
    ap.add_argument("--param_set", default="set8")
    ap.add_argument("--gamma", type=float, default=4.1667e-4, help="phi0/M_Pl")
    ap.add_argument("--mphi", type=float, default=1000.0, help="flaton mass [GeV]")
    ap.add_argument("--nb", type=int, default=20)
    ap.add_argument("--nf", type=int, default=20)
    ap.add_argument("--boson_coupling", type=float, default=1.09)
    ap.add_argument("--fermion_coupling", type=float, default=1.09)
    ap.add_argument("--boson_gauge_coupling", type=float, default=1.05)
    ap.add_argument("--fermion_gauge_coupling", type=float, default=1.05)
    ap.add_argument(
        "--potential_type",
        default="V_correct",
        choices=["V_correct", "fermion_only"],
    )
    ap.add_argument("--zn_order", type=int, default=0)
    ap.add_argument("--zn_strength", type=float, default=0.0)
    ap.add_argument("--T0", type=float, default=1230.0, help="reference T [GeV] (annotation only)")
    ap.add_argument("--T_span", type=float, default=400.0, help="scan width above T_c2 [GeV]")
    ap.add_argument("--dT", type=float, default=10.0, help="temperature step [GeV]")
    ap.add_argument("--tag", default=None, help="output subfolder (default: param_set_complex_znN)")
    ap.add_argument(
        "--resume-from-csv",
        action="store_true",
        help="skip bounce scan; load existing S3T_scan.csv in output dir",
    )
    args = ap.parse_args()

    gamma = args.gamma
    mphi = args.mphi
    lam = mphi**2 / (gamma * M_PL) ** 2
    delV = gamma**2 * mphi**2 * M_PL**2 / 4.0

    param_dict = {
        "lambda": lam,
        "mphi": mphi,
        "epsilon": 0.0,
        "lambdaSix": 0.0,
        "bosonMassSquared": 1.0e6,
        "bosonCoupling": args.boson_coupling,
        "bosonGaugeCoupling": args.boson_gauge_coupling,
        "fermionCoupling": args.fermion_coupling,
        "fermionGaugeCoupling": args.fermion_gauge_coupling,
        "nb": args.nb,
        "nf": args.nf,
    }

    tag = args.tag or f"{args.param_set}_complex_zn{args.zn_order}"
    out_dir = os.path.join(REPO, "data", "cosmotransitions", tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("CosmoTransitions — complex scalar (radial V(|phi|)) pipeline")
    print("=" * 72)
    print(f"  tag              : {tag}")
    print(f"  gamma            : {gamma:.6e}  (phi0 = {gamma * M_PL:.4e} GeV)")
    print(f"  mphi             : {mphi} GeV")
    print(f"  lambda           : {lam:.6e}")
    print(f"  delV             : {delV:.6e} GeV^4")
    print(f"  potential        : {args.potential_type}")
    print(f"  zn_order/strength: {args.zn_order} / {args.zn_strength}")
    print(f"  T0 (reference)   : {args.T0} GeV")
    if args.zn_order == 0:
        print("  Note: zn=0 => V depends only on rho; CT radial bounce = real-scalar nucleation.")
    print("=" * 72)

    VT = p.finiteTemperaturePotential(param_dict)
    VT.build_fast_thermal(x_max=150.0, n_pts=4096)
    spline_arrays = VT._fast_arrays

    T_c2, _, _ = find_T_c2(VT, args.potential_type)
    if T_c2 is None:
        sys.exit("ERROR: could not find T_c2 (V''(0)=0)")
    print(f"\nT_c2 (spinodal) = {T_c2:.4f} GeV  ({T_c2/1e3:.6f} TeV)")

    csv_path = os.path.join(out_dir, "S3T_scan.csv")
    rows = []
    if args.resume_from_csv and os.path.isfile(csv_path):
        print(f"\nResuming from {os.path.relpath(csv_path, REPO)}")
        with open(csv_path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                rows.append(
                    dict(
                        ok=True,
                        T=float(parts[0]),
                        S3=float(parts[1]),
                        S3_over_T=float(parts[2]),
                        r_c=float(parts[3]),
                        phi_esc=float(parts[4]),
                    )
                )
        print(f"  Loaded {len(rows)} points")
    else:
        T_list = np.arange(T_c2 + 1.0, T_c2 + args.T_span, args.dT)
        t0 = time.time()
        for k, T in enumerate(T_list):
            res = compute_S3_over_T(
                float(T),
                param_dict,
                spline_arrays,
                args.potential_type,
                args.zn_order,
                args.zn_strength,
            )
            if res.get("ok"):
                rows.append(res)
                print(
                    f"  [{k+1}/{len(T_list)}] T={T:8.2f} GeV  S3/T={res['S3_over_T']:8.3f}  "
                    f"r_c={res['r_c']:.3e}  phi_esc={res['phi_esc']:.3e}"
                )
            else:
                print(f"  [{k+1}/{len(T_list)}] T={T:8.2f} GeV  FAILED: {res.get('error')}")
        print(f"Scan done in {time.time()-t0:.1f}s ({len(rows)} successful points)")

    if len(rows) < 3:
        sys.exit("ERROR: too few successful tunneling points")

    T_ok = np.array([r["T"] for r in rows])
    S3T_ok = np.array([r["S3_over_T"] for r in rows])
    S3T_spl = fit_S3T_spline(T_ok, S3T_ok)
    T_n = find_T_n(T_ok, S3T_ok, mphi, gamma)

    if not args.resume_from_csv:
        with open(csv_path, "w") as f:
            f.write("T_GeV,S3,S3_over_T,r_c,phi_esc\n")
            for r in rows:
                f.write(
                    f"{r['T']:.8f},{r['S3']:.8e},{r['S3_over_T']:.8f},"
                    f"{r['r_c']:.8e},{r['phi_esc']:.8e}\n"
                )

    popt, rev_fn = fit_log_gamma(T_ok, S3T_ok)
    T_center = T_n if T_n else 0.5 * (T_ok[0] + T_ok[-1])
    T_scan_lo = max(T_c2 + 5.0, T_center - 120.0)
    T_scan_hi = min(T_ok[-1] + 80.0, T_center + 120.0)
    print(f"\nPercolation window: [{T_scan_lo:.1f}, {T_scan_hi:.1f}] GeV (rev fit on log Gamma)")

    t_sparse = np.linspace(T_scan_lo, T_scan_hi, 60)
    n_arr = []
    I_arr = []
    for t in t_sparse:
        try:
            n_arr.append(nT_integral(t, popt, rev_fn, delV))
            I_arr.append(percolation_I(t, popt, rev_fn, delV))
        except Exception:
            n_arr.append(np.nan)
            I_arr.append(np.nan)
    n_arr = np.array(n_arr)
    I_arr = np.array(I_arr)

    valid_n = np.isfinite(n_arr) & (n_arr > 0)
    T_n_rate = (
        float(t_sparse[valid_n][np.argmin(np.abs(n_arr[valid_n] - 1.0))])
        if valid_n.any()
        else None
    )

    T_p = find_T_from_percolation_scan(I_arr, t_sparse, P_target=0.7)
    T_c1 = find_T_from_percolation_scan(I_arr, t_sparse, P_target=7.0e-6)

    summary = {
        "tag": tag,
        "model": "complex_scalar_radial_bounce",
        "zn_order": args.zn_order,
        "zn_strength": args.zn_strength,
        "potential_type": args.potential_type,
        "gamma": gamma,
        "phi0_GeV": gamma * M_PL,
        "mphi_GeV": mphi,
        "lambda": lam,
        "delV_GeV4": delV,
        "nb": args.nb,
        "nf": args.nf,
        "boson_coupling": args.boson_coupling,
        "fermion_coupling": args.fermion_coupling,
        "T0_reference_GeV": args.T0,
        "T_c2_GeV": T_c2,
        "T_n_GeV": T_n,
        "T_n_rate_GeV": T_n_rate,
        "T_p_GeV": T_p,
        "T_c1_GeV": T_c1,
        "nucleation_condition": "S3/T = 4 ln(2 sqrt(3) T / (m phi gamma))",
        "T_n_rate_note": "n(T)=1 from Gamma/H^4 integral (drawAction convention)",
        "percolation_note": "T_p at P=0.7, T_c1 at P=7e-6 (drawAction I(T) integral)",
        "compare_to_cosmolattice": {
            "lattice_T_p": "postprocess/revisualize_snapshots.py --view3d volume_fraction>0.99999",
            "lattice_phi_threshold_crossing": "max|rho| > phi_threshold in snapshots",
        },
    }
    json_path = os.path.join(out_dir, "transition_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(T_ok / 1e3, S3T_ok, "k-", lw=2.5, label=r"$S_3/T$ (CT bounce)")
    T_fine = np.linspace(T_ok[0], T_ok[-1], 300)
    rhs_fine = [nucleation_rhs(T, mphi, gamma) for T in T_fine]
    ax.plot(T_fine / 1e3, rhs_fine, "b--", lw=2.0, label=r"$4\ln(2\sqrt{3}T/m\gamma)$")
    if T_n:
        ax.axvline(T_n / 1e3, color="red", ls=":", lw=2, label=rf"$T_n={T_n/1e3:.4f}$ TeV")
    ax.axvline(T_c2 / 1e3, color="gray", ls=":", lw=1.5, label=rf"$T_{{c2}}={T_c2/1e3:.4f}$ TeV")
    ax.set_xlabel(r"$T$ [TeV]")
    ax.set_ylabel(r"$S_3/T$")
    ax.set_title(f"CosmoTransitions nucleation ({tag})")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "nucleation_condition.png"), dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(T_ok / 1e3, S3T_ok, "o-", ms=4, lw=1.5)
    ax2.set_xlabel(r"$T$ [TeV]")
    ax2.set_ylabel(r"$S_3/T$")
    ax2.set_title("S3/T temperature scan")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "S3T_vs_T.png"), dpi=180)
    plt.close(fig2)

    print("\n" + "=" * 72)
    print("RESULTS (compare to CosmoLattice later)")
    print("=" * 72)
    print(f"  T_c2  = {T_c2:.4f} GeV")
    print(f"  T_n   = {T_n:.4f} GeV" if T_n else "  T_n   = (not in scan range)")
    if T_n_rate:
        print(f"  T_n (Gamma/H^4=1) = {T_n_rate:.4f} GeV")
    print(f"  T_p   = {T_p:.4f} GeV" if T_p else "  T_p   = (not resolved in scan)")
    print(f"  T_c1  = {T_c1:.4f} GeV" if T_c1 else "  T_c1  = (not resolved in scan)")
    print(f"\nWrote:\n  {os.path.relpath(csv_path, REPO)}\n  {os.path.relpath(json_path, REPO)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
