#!/usr/bin/env python3
"""
Find two critical temperatures for the finite-temperature potential:

  T_c  : V(origin) = V(true vacuum)   -- degenerate minima
  T_sp : V''(origin) = 0              -- barrier near origin disappears (spinodal)

Each temperature is searched in its own range with high resolution,
then refined with Brent's method.
"""
import numpy as np
import scipy.optimize as opt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os

_sys.path.insert(
    0,
    _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"
    ),
)
import Potential as p


M_PL = 2.4e18  # Reduced Planck mass (GeV)

PARAM_SETS = {
    "A": {"gamma": 4.1667e-8, "g": 1.05, "nb": 20, "nf": 20},
    "B": {"gamma": 4.1667e-4, "g": 1.05, "nb": 20, "nf": 20},
    "C": {"gamma": 4.1667e-4, "g": 1.05, "nb": 0, "nf": 20},
}


def setup_potential(gamma=4.1667e-8, nb=20, nf=20, y=1.09, g=1.05):
    mphi = 1000.0
    phi0 = gamma * M_PL
    lam = mphi**2 / phi0**2

    param = {
        "lambda": lam,
        "mphi": mphi,
        "epsilon": 0,
        "lambdaSix": 0,
        "bosonMassSquared": 1_000_000,
        "bosonCoupling": y,
        "bosonGaugeCoupling": g,
        "fermionCoupling": y,
        "fermionGaugeCoupling": g,
        "nb": nb,
        "nf": nf,
    }

    VT = p.finiteTemperaturePotential(param)
    VT.update_T(1.0)
    VT.build_fast_thermal(x_max=150.0, n_pts=4096)
    return VT, lam, phi0


def V_at(VT, phi_val, nb=0):
    """Evaluate the thermal potential at a single field value."""
    X = np.array([[phi_val]])
    if nb > 0:
        return VT.V_p_correct(X).item()
    return VT.V_p_fermion_only(X).item()


def find_true_minimum(VT, vev_guess, nb=0):
    """Find the position of the true vacuum minimum near vev_guess."""
    res = opt.minimize_scalar(
        lambda phi: V_at(VT, phi, nb=nb),
        bounds=(0.01 * vev_guess, 2.0 * vev_guess),
        method="bounded",
    )
    return res.x, res.fun


def V_second_derivative_at_origin(VT, h=1.0, nb=0):
    """V''(0) via central finite difference with small step h."""
    return (V_at(VT, h, nb=nb) - 2.0 * V_at(VT, 0.0, nb=nb) + V_at(VT, -h, nb=nb)) / h**2


def delta_V_func(T, VT, vev_guess, nb=0):
    """V(origin) - V(true_min) at temperature T."""
    VT.update_T(T)
    V_origin = V_at(VT, 0.0, nb=nb)
    _, V_min = find_true_minimum(VT, vev_guess, nb=nb)
    return V_origin - V_min


def d2V_func(T, VT, h, nb=0):
    """V''(0) at temperature T."""
    VT.update_T(T)
    return V_second_derivative_at_origin(VT, h=h, nb=nb)


def scan_and_find_root(func, T_arr, label):
    """Coarse scan + Brent refinement for a sign-change root."""
    vals = np.array([func(T) for T in T_arr])

    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) == 0:
        print(f"  [{label}] No sign change found in [{T_arr[0]:.1f}, {T_arr[-1]:.1f}]")
        print(f"           val range: [{vals.min():.6e}, {vals.max():.6e}]")
        return None, T_arr, vals

    idx = sign_changes[0]
    T_root = opt.brentq(func, T_arr[idx], T_arr[idx + 1], xtol=1e-6)
    print(f"  [{label}] Root found: T = {T_root:.6f} GeV")
    return T_root, T_arr, vals


def find_Tc_for_set(set_name, gamma, nb, nf, y=1.09, g=1.05):
    """Find the degeneracy temperature T_c for a given parameter set."""
    print(f"\n{'='*70}")
    print(f"  Set {set_name}:  γ = {gamma:.4e},  nb = {nb},  nf = {nf},  y = {y},  g = {g}")
    print(f"{'='*70}")

    VT, lam, phi0 = setup_potential(gamma=gamma, nb=nb, nf=nf, y=y, g=g)
    vev_0 = VT.v
    V0 = 0.25 * lam * vev_0**4
    print(f"  φ₀ = {phi0:.4e} GeV,  λ = {lam:.4e},  V₀ = {V0:.4e} GeV⁴")
    print(f"  VEV = {vev_0:.4e} GeV")

    # Estimate T_c search range from Eq. (29): T_c ~ [1/(4λ · nf|JF|)]^{1/4} · m
    # For small λ, T_c can be very large
    Tc_est = (1.0 / (4.0 * lam * max(nf, 1) * 0.0177)) ** 0.25 * 1000.0
    T_c_lo = max(Tc_est * 0.1, 1000.0)
    T_c_hi = Tc_est * 5.0
    N_c = 2000
    print(f"  T_c estimate: {Tc_est:.0f} GeV  →  scan [{T_c_lo:.0f}, {T_c_hi:.0f}]")

    T_c_arr = np.linspace(T_c_lo, T_c_hi, N_c)
    T_c, _, _ = scan_and_find_root(
        lambda T: delta_V_func(T, VT, vev_0, nb=nb),
        T_c_arr,
        f"T_c (set {set_name})",
    )

    # Also find T_c2 (barrier disappears, V''(0)=0)
    FD_STEP = 1.0
    T_sp_lo, T_sp_hi = 1000.0, 2000.0
    T_sp_arr = np.linspace(T_sp_lo, T_sp_hi, 1000)
    T_sp, _, _ = scan_and_find_root(
        lambda T: d2V_func(T, VT, FD_STEP, nb=nb),
        T_sp_arr,
        f"T_c2 (set {set_name})",
    )

    if T_c is not None:
        VT.update_T(T_c)
        V0_Tc = V_at(VT, 0.0, nb=nb)
        phi_m, Vm_Tc = find_true_minimum(VT, vev_0, nb=nb)
        print(f"  T_c  = {T_c:.2f} GeV  ({T_c/1e3:.4f} TeV)")
        print(f"  V(0) - V(φ_min) = {V0_Tc - Vm_Tc:.4e}  (should be ~0)")
    if T_sp is not None:
        print(f"  T_c2 = {T_sp:.2f} GeV  ({T_sp/1e3:.4f} TeV)")

    return T_c, T_sp


def main():
    print("Computing degeneracy temperature T_c for all parameter sets")
    print("  m = 1 TeV,  y = 1.09,  g = 1.05")

    results = {}
    for name, ps in PARAM_SETS.items():
        T_c, T_sp = find_Tc_for_set(
            name, gamma=ps["gamma"], nb=ps["nb"], nf=ps["nf"], g=ps["g"]
        )
        results[name] = {"T_c": T_c, "T_c2": T_sp, **ps}

    # Summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY:  Degeneracy temperature T_c  (V(0,T_c) = V(φ_min,T_c))")
    print("=" * 80)
    print(f"  {'Set':<6} {'γ':<16} {'nb':<5} {'nf':<5} {'T_c (GeV)':<16} {'T_c (TeV)':<14} {'T_c2 (GeV)':<14}")
    print("-" * 80)
    for name in ["A", "B", "C"]:
        r = results[name]
        Tc_str = f"{r['T_c']:.2f}" if r["T_c"] else "NOT FOUND"
        Tc_TeV = f"{r['T_c']/1e3:.4f}" if r["T_c"] else "—"
        Tc2_str = f"{r['T_c2']:.2f}" if r["T_c2"] else "NOT FOUND"
        print(f"  {name:<6} {r['gamma']:<16.4e} {r['nb']:<5} {r['nf']:<5} {Tc_str:<16} {Tc_TeV:<14} {Tc2_str:<14}")
    print("=" * 80)

    plt.close("all")


if __name__ == "__main__":
    main()
