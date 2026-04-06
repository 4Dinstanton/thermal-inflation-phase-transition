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


def setup_potential():
    lam = 1e-16
    mphi = 1000
    epsil = 0
    lambdaSix = 0

    param = {
        "lambda": lam,
        "mphi": mphi,
        "epsilon": epsil,
        "lambdaSix": lambdaSix,
        "bosonMassSquared": 1_000_000,
        "bosonCoupling": 1.09,
        "bosonGaugeCoupling": 1.05,
        "fermionCoupling": 1.09,
        "fermionGaugeCoupling": 1.05,
        "nb": 20,
        "nf": 20,
    }

    VT = p.finiteTemperaturePotential(param)
    VT.update_T(1.0)
    VT.build_fast_thermal(x_max=150.0, n_pts=4096)
    return VT


def V_at(VT, phi_val):
    """Evaluate V_p_fermion_only at a single field value."""
    X = np.array([[phi_val]])
    return VT.V_p_fermion_only(X).item()


def find_true_minimum(VT, vev_guess):
    """Find the position of the true vacuum minimum near vev_guess."""
    res = opt.minimize_scalar(
        lambda phi: V_at(VT, phi),
        bounds=(0.01 * vev_guess, 2.0 * vev_guess),
        method="bounded",
    )
    return res.x, res.fun


def V_second_derivative_at_origin(VT, h=1.0):
    """V''(0) via central finite difference with small step h."""
    return (V_at(VT, h) - 2.0 * V_at(VT, 0.0) + V_at(VT, -h)) / h**2


def delta_V_func(T, VT, vev_guess):
    """V(origin) - V(true_min) at temperature T."""
    VT.update_T(T)
    V_origin = V_at(VT, 0.0)
    _, V_min = find_true_minimum(VT, vev_guess)
    return V_origin - V_min


def d2V_func(T, VT, h):
    """V''(0) at temperature T."""
    VT.update_T(T)
    return V_second_derivative_at_origin(VT, h=h)


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


def main():
    VT = setup_potential()
    vev_0 = VT.v
    print(f"Zero-temperature VEV: {vev_0:.6e} GeV")
    print(f"Tree-level V(VEV):    {VT.V_tree(np.array([[vev_0]])).item():.6e}")

    # ----------------------------------------------------------------
    # T_sp: barrier near origin disappears  (V''(0) = 0)
    #   Search range: 6600 -- 6780  with 1000 points
    #   Use small h for finite differences (barrier width ~ few 1000 GeV
    #   in field space, so h ~ 1 GeV is appropriate)
    # ----------------------------------------------------------------
    FD_STEP = 1.0  # 1 GeV step for V''(0) finite difference

    T_sp_lo, T_sp_hi, N_sp = 1000.0, 2000.0, 1000
    T_sp_arr = np.linspace(T_sp_lo, T_sp_hi, N_sp)
    print(
        f"\n--- Scanning T_sp in [{T_sp_lo}, {T_sp_hi}], {N_sp} pts, h={FD_STEP} GeV ---"
    )

    T_sp, T_sp_scan, d2V_vals = scan_and_find_root(
        lambda T: d2V_func(T, VT, FD_STEP),
        T_sp_arr,
        "T_sp",
    )

    # ----------------------------------------------------------------
    # T_c: degenerate minima  (V(0) = V(true min))
    #   V_tree(VEV) ~ -2.5e27;  V_thermal(0) ~ T^4 * Jf / (2pi^2)
    #   => T_c ~ O(1e7) GeV
    # ----------------------------------------------------------------
    T_c_lo, T_c_hi, N_c = 1.0e6, 5.0e7, 1000
    T_c_arr = np.linspace(T_c_lo, T_c_hi, N_c)
    print(f"\n--- Scanning T_c in [{T_c_lo:.0f}, {T_c_hi:.0f}], {N_c} pts ---")

    T_c, T_c_scan, dV_vals = scan_and_find_root(
        lambda T: delta_V_func(T, VT, vev_0),
        T_c_arr,
        "T_c",
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if T_sp is not None:
        print(f"  T_sp (barrier disappears)    = {T_sp:.4f} GeV  ({T_sp/1e3:.4f} TeV)")
    else:
        print(f"  T_sp: NOT FOUND in [{T_sp_lo}, {T_sp_hi}]")
    if T_c is not None:
        VT.update_T(T_c)
        phi_min_Tc, _ = find_true_minimum(VT, vev_0)
        print(f"  T_c  (degenerate minima)     = {T_c:.4f} GeV  ({T_c/1e3:.2f} TeV)")
        print(
            f"         phi_min(T_c)          = {phi_min_Tc:.6e} GeV  ({phi_min_Tc/1e3:.2f} TeV)"
        )
    else:
        print(f"  T_c:  NOT FOUND in [{T_c_lo:.0f}, {T_c_hi:.0f}]")
    if T_c is not None and T_sp is not None:
        print(f"\n  T_c / T_sp = {T_c / T_sp:.2f}")
        print(f"  T_c > T_sp ?  {T_c > T_sp}  (expected: T_c > T_sp for 1st-order PT)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Plot 1: Scan curves with root markers
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(T_sp_scan, d2V_vals, "g-", lw=1.2)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    if T_sp is not None:
        ax1.axvline(
            T_sp, color="red", ls="--", lw=1.2, label=f"$T_{{sp}}$ = {T_sp:.2f} GeV"
        )
    ax1.set_xlabel("T (GeV)")
    ax1.set_ylabel(r"$V''(0, T)$")
    ax1.set_title(r"Barrier near origin disappears at $T_{sp}$")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(T_c_scan, dV_vals, "b-", lw=1.2)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    if T_c is not None:
        ax2.axvline(T_c, color="red", ls="--", lw=1.2, label=f"$T_c$ = {T_c:.1f} GeV")
    ax2.set_xlabel("T (GeV)")
    ax2.set_ylabel(r"$V(0,T) - V(\phi_{\min},T)$")
    ax2.set_title(r"Degenerate minima at $T_c$")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Critical Temperatures (fermion_only, set7)", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        "figs/finiteTemp/critical_temperatures.png", dpi=200, bbox_inches="tight"
    )
    print(f"\nPlot saved: figs/finiteTemp/critical_temperatures.png")

    # ----------------------------------------------------------------
    # Plot 2: Potential at key temperatures
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Plot 2: Barrier disappearance at T_sp  (near-origin zoom)
    # ----------------------------------------------------------------
    if T_sp is not None:
        phi_near = np.linspace(-5000, 5000, 2000).reshape(-1, 1)
        sp_temps = [
            (T_sp * 1.02, f"1.02 $T_{{sp}}$ = {T_sp*1.02:.1f}"),
            (T_sp * 1.01, f"1.01 $T_{{sp}}$ = {T_sp*1.01:.1f}"),
            (T_sp, f"$T_{{sp}}$ = {T_sp:.2f}"),
            (T_sp * 0.99, f"0.99 $T_{{sp}}$ = {T_sp*0.99:.1f}"),
            (T_sp * 0.98, f"0.98 $T_{{sp}}$ = {T_sp*0.98:.1f}"),
        ]

        fig_sp, ax_sp = plt.subplots(figsize=(9, 6))
        for T_plot, lab in sp_temps:
            VT.update_T(T_plot)
            V_zero = V_at(VT, 0.0)
            V_vals = VT.V_p_fermion_only(phi_near) - V_zero
            ax_sp.plot(phi_near.ravel(), V_vals.ravel(), label=lab, lw=1.5)

        ax_sp.axhline(0, color="gray", ls="--", lw=0.8)
        ax_sp.set_xlabel(r"$\phi$ (GeV)")
        ax_sp.set_ylabel(r"$V(\phi, T) - V(0, T)$")
        ax_sp.set_title(
            rf"Barrier disappearance at $T_{{sp}}$ = {T_sp:.2f} GeV"
            "\n(origin ceases to be a local minimum)",
            fontsize=12,
        )
        ax_sp.legend(fontsize=10)
        ax_sp.grid(True, alpha=0.3)
        fig_sp.tight_layout()
        fig_sp.savefig(
            "figs/finiteTemp/potential_barrier_disappearance.png",
            dpi=200,
            bbox_inches="tight",
        )
        print(f"Plot saved: figs/finiteTemp/potential_barrier_disappearance.png")

    # ----------------------------------------------------------------
    # Plot 3: Vacuum degeneracy at T_c  (full range showing both minima)
    # ----------------------------------------------------------------
    if T_c is not None:
        # Build a high-resolution grid that INCLUDES the exact VEV point
        n_full = 20000
        phi_full_vals = np.linspace(-0.1 * vev_0, 1.3 * vev_0, n_full)
        # Insert exact VEV into the grid so the minimum is captured precisely
        phi_full_vals = np.sort(np.append(phi_full_vals, vev_0))
        phi_full = phi_full_vals.reshape(-1, 1)

        # Plot order: background curves first, T_c last on top
        tc_temps_bg = [
            (T_c * 1.05, f"1.05 $T_c$ = {T_c*1.05:.0f}"),
            (T_c * 1.01, f"1.01 $T_c$ = {T_c*1.01:.0f}"),
            (T_c * 0.99, f"0.99 $T_c$ = {T_c*0.99:.0f}"),
            (T_c * 0.95, f"0.95 $T_c$ = {T_c*0.95:.0f}"),
        ]
        tc_temp_main = (T_c, f"$T_c$ = {T_c:.0f}")

        fig_tc, axes_tc = plt.subplots(1, 2, figsize=(16, 6))

        # --- Left panel: full range ---
        for T_plot, lab in tc_temps_bg:
            VT.update_T(T_plot)
            V_zero = V_at(VT, 0.0)
            V_vals = VT.V_p_fermion_only(phi_full).ravel() - V_zero
            axes_tc[0].plot(
                phi_full.ravel() / 1e3, V_vals, label=lab, lw=1.2, alpha=0.7
            )
            phi_m, V_m = find_true_minimum(VT, vev_0)
            col = axes_tc[0].get_lines()[-1].get_color()
            axes_tc[0].plot(phi_m / 1e3, V_m - V_zero, "o", color=col, ms=5, zorder=5)

        # T_c drawn LAST with thicker line and high zorder
        T_plot, lab = tc_temp_main
        VT.update_T(T_plot)
        V_zero = V_at(VT, 0.0)
        V_vals = VT.V_p_fermion_only(phi_full).ravel() - V_zero
        axes_tc[0].plot(
            phi_full.ravel() / 1e3, V_vals, label=lab, lw=2.5, color="black", zorder=10
        )
        phi_m, V_m = find_true_minimum(VT, vev_0)
        axes_tc[0].plot(phi_m / 1e3, V_m - V_zero, "o", color="black", ms=8, zorder=11)

        axes_tc[0].axhline(0, color="gray", ls="--", lw=0.8)
        axes_tc[0].set_xlabel(r"$\phi$ (TeV)")
        axes_tc[0].set_ylabel(r"$V(\phi, T) - V(0, T)$")
        axes_tc[0].set_title("Full range: both minima visible")
        axes_tc[0].legend(fontsize=9)
        axes_tc[0].grid(True, alpha=0.3)

        # --- Right panel: zoom near the VEV ---
        n_zoom = 20000
        phi_zoom_vals = np.linspace(0.9 * vev_0, 1.1 * vev_0, n_zoom)
        phi_zoom_vals = np.sort(np.append(phi_zoom_vals, vev_0))
        phi_vev_zoom = phi_zoom_vals.reshape(-1, 1)

        for T_plot, lab in tc_temps_bg:
            VT.update_T(T_plot)
            V_zero = V_at(VT, 0.0)
            V_vals = VT.V_p_fermion_only(phi_vev_zoom).ravel() - V_zero
            axes_tc[1].plot(
                phi_vev_zoom.ravel() / 1e3, V_vals, label=lab, lw=1.2, alpha=0.7
            )
            phi_m, V_m = find_true_minimum(VT, vev_0)
            col = axes_tc[1].get_lines()[-1].get_color()
            axes_tc[1].plot(phi_m / 1e3, V_m - V_zero, "o", color=col, ms=5, zorder=5)

        # T_c drawn LAST
        T_plot, lab = tc_temp_main
        VT.update_T(T_plot)
        V_zero = V_at(VT, 0.0)
        V_vals = VT.V_p_fermion_only(phi_vev_zoom).ravel() - V_zero
        axes_tc[1].plot(
            phi_vev_zoom.ravel() / 1e3,
            V_vals,
            label=lab,
            lw=2.5,
            color="black",
            zorder=10,
        )
        phi_m, V_m = find_true_minimum(VT, vev_0)
        axes_tc[1].plot(phi_m / 1e3, V_m - V_zero, "o", color="black", ms=8, zorder=11)

        axes_tc[1].axhline(
            0, color="red", ls="--", lw=1.0, alpha=0.7, label="V(0) level"
        )
        axes_tc[1].set_xlabel(r"$\phi$ (TeV)")
        axes_tc[1].set_ylabel(r"$V(\phi, T) - V(0, T)$")
        axes_tc[1].set_title(r"Zoom near VEV: degeneracy at $T_c$")
        axes_tc[1].legend(fontsize=9)
        axes_tc[1].grid(True, alpha=0.3)

        # Print verification
        VT.update_T(T_c)
        V0_Tc = V_at(VT, 0.0)
        phi_m_Tc, Vm_Tc = find_true_minimum(VT, vev_0)
        print(f"\n  [Degeneracy check at T_c]")
        print(f"    V(0)        = {V0_Tc:.15e}")
        print(f"    V(phi_min)  = {Vm_Tc:.15e}")
        print(f"    V(0)-V(min) = {V0_Tc - Vm_Tc:.6e}  (should be ~0)")

        T_c_TeV = T_c / 1e3
        fig_tc.suptitle(
            rf"Vacuum degeneracy at $T_c$ = {T_c:.0f} GeV ({T_c_TeV:.1f} TeV)",
            fontsize=13,
        )
        fig_tc.tight_layout()
        fig_tc.savefig(
            "figs/finiteTemp/potential_vacuum_degeneracy.png",
            dpi=200,
            bbox_inches="tight",
        )
        print(f"Plot saved: figs/finiteTemp/potential_vacuum_degeneracy.png")

    plt.close("all")


if __name__ == "__main__":
    main()
