"""
Scan tunneling parameters over both coupling and temperature.
Outer loop: COUPLING_LIST
Inner loop: TEMP_LIST (parallelized across CPU cores)

Outputs:
- CSV file with columns: coupling, T, S3/T, r_c, phi_esc
- Separate CSV for each coupling value
- Combined CSV with all data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.optimize as opt

import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD
import sys as _sys, os as _os

_sys.path.insert(
    0,
    _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"
    ),
)
import Potential as p
from tunneling_utils import fullTunneling

# "V_correct" = boson + fermion (Jb + Jf)
# "fermion_only" = fermion only (Jf)
potential_flag = "fermion_only"


def format_e(n):
    """Format number in scientific notation."""
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "E" + a.split("E")[1]


def _V_at(VT, phi_val, pot_flag="fermion_only"):
    """Evaluate the potential at a single field value."""
    X = np.array([[phi_val]])
    if pot_flag == "fermion_only":
        return VT.V_p_fermion_only(X).item()
    return VT.V_p_correct(X).item()


def _V_second_derivative_at_origin(VT, pot_flag="fermion_only", h=1.0):
    """V''(0) via central finite difference."""
    return (
        _V_at(VT, h, pot_flag)
        - 2.0 * _V_at(VT, 0.0, pot_flag)
        + _V_at(VT, -h, pot_flag)
    ) / h**2


def find_barrier_temperature(
    VT, pot_flag="fermion_only", T_lo=100.0, T_hi=50000.0, n_coarse=500, fd_step=1.0
):
    """
    Find T_sp where V''(0) = 0 (barrier near origin disappears).

    Above T_sp the origin is a local minimum (barrier exists);
    below T_sp the curvature is negative (no barrier, no tunneling).
    """
    T_arr = np.linspace(T_lo, T_hi, n_coarse)
    d2V_vals = np.empty(n_coarse)
    for i, T in enumerate(T_arr):
        VT.update_T(T)
        d2V_vals[i] = _V_second_derivative_at_origin(VT, pot_flag, h=fd_step)

    sign_changes = np.where(np.diff(np.sign(d2V_vals)))[0]
    if len(sign_changes) == 0:
        print(f"  [find_barrier_temperature] No sign change in [{T_lo}, {T_hi}]")
        print(f"  V''(0) range: [{d2V_vals.min():.6e}, {d2V_vals.max():.6e}]")
        return None

    idx = sign_changes[0]

    def _d2V_at_T(T):
        VT.update_T(T)
        return _V_second_derivative_at_origin(VT, pot_flag, h=fd_step)

    T_sp = opt.brentq(_d2V_at_T, T_arr[idx], T_arr[idx + 1], xtol=1e-4)
    return T_sp


def _tunneling_worker(args):
    """Worker: create own VT, reconstruct fast splines, run fullTunneling."""
    TEMP, param_dict, epsil, spline_arrays, pot_flag, tv_estimate = args

    VT_w = p.finiteTemperaturePotential(param_dict)
    VT_w.update_T(TEMP)
    VT_w.set_fast_thermal_from_arrays(*spline_arrays)

    fv = 0 if epsil == 0 else None
    if fv is None:
        _, fv = VT_w.find_new_minima()

    if pot_flag == "fermion_only":
        V_func = VT_w.V_fermion_only
        dV_func = VT_w.dV_p_fermion_only
    else:
        V_func = VT_w.V_correct
        dV_func = VT_w.dV_p_correct

    try:
        tunneling_result = fullTunneling(
            path_pts=np.array([[tv_estimate], [fv]]),
            V=V_func,
            dV=dV_func,
            maxiter=1,
            V_spline_samples=200,
            tunneling_init_params=dict(alpha=2),
            tunneling_findProfile_params=dict(
                xtol=0.00001, phitol=0.00001, rmin=0.00001, npoints=200
            ),
            deformation_class=CTPD.Deformation_Spline,
            extend_to_minima=False,
        )
    except Exception as e:
        print(f"  T={TEMP:.1f}  tunneling FAILED (T_c2 / no barrier): {e}", flush=True)
        return TEMP, 0.0, 0.0, 0.0

    S3_T = tunneling_result.action / TEMP
    _R = tunneling_result.profile1D.R
    _Phi = tunneling_result.profile1D.Phi
    _phi_mid = 0.5 * (_Phi[0] + _Phi[-1])
    r_c = (
        np.interp(_phi_mid, _Phi[::-1], _R[::-1])
        if _Phi[0] > _Phi[-1]
        else np.interp(_phi_mid, _Phi, _R)
    )
    phi_esc = tv_estimate - _Phi[0]

    print(
        f"  T={TEMP:.1f}  S3/T={S3_T:.2f}  r_c={r_c:.2e}  phi_esc={phi_esc:.2e}",
        flush=True,
    )
    return TEMP, S3_T, r_c, phi_esc


if __name__ == "__main__":
    import time as _time

    # =============================================================================
    # Physical parameters
    # =============================================================================
    lam = 1e-16
    mphi = 1000
    epsil = 0
    lambdaSix = 0

    bosonMassSquared = 1000000
    bosonCoupling = 1.09
    bosonGaugeCoupling = 1.05
    fermionCoupling = 1.09
    fermionGaugeCoupling = 1.05
    nb = 20
    nf = 20

    param_set = "set7"

    param = {
        param_set: {
            "lambda": lam,
            "mphi": mphi,
            "epsilon": epsil,
            "lambdaSix": lambdaSix,
            "bosonMassSquared": bosonMassSquared,
            "bosonCoupling": bosonCoupling,
            "bosonGaugeCoupling": bosonGaugeCoupling,
            "fermionCoupling": fermionCoupling,
            "fermionGaugeCoupling": fermionGaugeCoupling,
            "nb": nb,
            "nf": nf,
        }
    }

    # =============================================================================
    # Scan ranges
    # =============================================================================
    COUPLING_LIST = np.arange(1.00, 1.21, 0.01)
    TEMP_RANGE = 1000.0
    TEMP_STEP = 20.0

    # =============================================================================
    # Output directory
    # =============================================================================
    output_dir = f"data/tunneling/{param_set}/coupling_temp_scan_{potential_flag}"
    os.makedirs(output_dir, exist_ok=True)

    # =============================================================================
    # Build fast thermal splines (once, shared across all workers)
    # =============================================================================
    VT = p.finiteTemperaturePotential(param[param_set])
    VT.build_fast_thermal(x_max=150.0, n_pts=4096)
    spline_arrays = VT._fast_arrays

    N_WORKERS = min(multiprocessing.cpu_count(), 4)
    N_WORKERS = 1

    # =============================================================================
    # Main scan loops
    # =============================================================================
    all_results = []

    print("=" * 70)
    print("COUPLING-TEMPERATURE SCAN (parallelized)")
    print("=" * 70)
    print(
        f"Scanned: bosonCoupling = fermionCoupling = "
        f"{COUPLING_LIST[0]:.3f} to {COUPLING_LIST[-1]:.3f}"
    )
    print(f"Fixed:   bosonGaugeCoupling = {bosonGaugeCoupling:.3f}")
    print(f"Fixed:   fermionGaugeCoupling = {fermionGaugeCoupling:.3f}")
    print(f"Temperature range: T_sp to T_sp + {TEMP_RANGE:.0f} GeV (per coupling)")
    print(f"Workers: {N_WORKERS}")
    print(f"Potential: {potential_flag}")
    print("=" * 70)

    for i_coup, COUP in enumerate(COUPLING_LIST):
        print(f"\n{'='*70}")
        print(f"Coupling {i_coup+1}/{len(COUPLING_LIST)}: g = {COUP:.4f}")
        print("=" * 70)

        param[param_set]["bosonCoupling"] = float(COUP)
        param[param_set]["fermionCoupling"] = float(COUP)

        # --- Find T_sp for this coupling ---
        VT_scan = p.finiteTemperaturePotential(param[param_set])
        VT_scan.update_T(1.0)
        VT_scan.set_fast_thermal_from_arrays(*spline_arrays)

        T_sp = find_barrier_temperature(
            VT_scan,
            pot_flag=potential_flag,
            T_lo=100.0,
            T_hi=50000.0,
            n_coarse=500,
            fd_step=1.0,
        )
        if T_sp is not None:
            print(f"  T_sp = {T_sp:.2f} GeV")
        else:
            print(f"  T_sp not found, skipping coupling g = {COUP:.4f}")
            continue

        TEMP_LIST = np.arange(T_sp, T_sp + TEMP_RANGE, TEMP_STEP)
        print(
            f"  Temperature scan: [{TEMP_LIST[0]:.2f}, {TEMP_LIST[-1]:.2f}] GeV, "
            f"{len(TEMP_LIST)} points"
        )

        S3_T_list = []
        r_c_list = []
        phi_esc_list = []
        successful_temps = []

        _t0 = _time.time()

        S3T_CUTOFF = 1000.0
        n_done = 0
        n_fail = 0
        hit_cutoff = False

        worker_args = []
        for TEMP in TEMP_LIST:
            tv = max(10.0 * TEMP, 80000.0)
            worker_args.append(
                (
                    float(TEMP),
                    param[param_set].copy(),
                    epsil,
                    spline_arrays,
                    potential_flag,
                    tv,
                )
            )

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            future_to_temp = {
                pool.submit(_tunneling_worker, a): a[0] for a in worker_args
            }
            results_map = {}
            for fut in as_completed(future_to_temp):
                temp_key = future_to_temp[fut]
                try:
                    TEMP_r, S3_T, r_c, phi_esc = fut.result()
                    results_map[TEMP_r] = (S3_T, r_c, phi_esc)
                    n_done += 1
                except Exception as e:
                    print(f"  T={temp_key:.1f} FAILED: {e}", flush=True)
                    n_fail += 1

        for TEMP in TEMP_LIST:
            if hit_cutoff:
                break
            T_val = float(TEMP)
            if T_val in results_map:
                S3_T, r_c, phi_esc = results_map[T_val]
                S3_T_list.append(S3_T)
                r_c_list.append(r_c)
                phi_esc_list.append(phi_esc)
                successful_temps.append(T_val)
                all_results.append(
                    {
                        "coupling": COUP,
                        "T": T_val,
                        "S3/T": S3_T,
                        "r_c": r_c,
                        "phi_esc": phi_esc,
                    }
                )
                if S3_T > S3T_CUTOFF:
                    print(
                        f"  S3/T = {S3_T:.2f} > {S3T_CUTOFF} at T={T_val:.1f}"
                        f" -> skipping to next coupling"
                    )
                    hit_cutoff = True

        _elapsed = _time.time() - _t0
        print(
            f"\n  Done: {n_done} ok, {n_fail} failed in {_elapsed:.1f}s "
            f"({_elapsed/max(n_done,1):.2f}s/point)"
        )

        if len(successful_temps) > 0:
            df_coup = pd.DataFrame(
                {
                    "T": successful_temps,
                    "S3/T": S3_T_list,
                    "r_c": r_c_list,
                    "phi_esc": phi_esc_list,
                }
            )
            coup_str = f"{COUP:.4f}".replace(".", "p")
            df_coup.to_csv(f"{output_dir}/coupling_{coup_str}.csv", index=False)
            print(f"  Saved: {output_dir}/coupling_{coup_str}.csv")

    # =============================================================================
    # Save combined results
    # =============================================================================
    if len(all_results) > 0:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(f"{output_dir}/all_coupling_temp_scan.csv", index=False)
        print(f"\nSaved combined results: {output_dir}/all_coupling_temp_scan.csv")

        json.dump(param[param_set], open(f"{output_dir}/parameters.json", "w"))
        print(f"Saved parameters: {output_dir}/parameters.json")

    # =============================================================================
    # Summary plot
    # =============================================================================
    print("\nGenerating summary plot...")

    try:
        df_all = pd.read_csv(f"{output_dir}/all_coupling_temp_scan.csv")

        print("\nData coverage summary:")
        print("-" * 50)
        unique_couplings = sorted(df_all["coupling"].unique())
        for coup in unique_couplings:
            subset = df_all[df_all["coupling"] == coup]
            if len(subset) > 0:
                T_min, T_max = subset["T"].min(), subset["T"].max()
                print(
                    f"  g={coup:.3f}: {len(subset):3d} points, "
                    f"T = [{T_min:.0f}, {T_max:.0f}]"
                )
        print("-" * 50)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        n_couplings = len(unique_couplings)
        step = max(1, n_couplings // 10)

        for coup in unique_couplings[::step]:
            subset = df_all[df_all["coupling"] == coup]
            if len(subset) > 0:
                subset = subset.sort_values("T")
                axes[0].plot(
                    subset["T"],
                    subset["S3/T"],
                    marker=".",
                    markersize=3,
                    label=f"g={coup:.3f}",
                )
        axes[0].set_xlabel("T")
        axes[0].set_ylabel("S3/T")
        axes[0].set_title("Bounce Action")
        axes[0].legend(fontsize=7, loc="best")
        axes[0].grid(True, alpha=0.3)

        for coup in unique_couplings[::step]:
            subset = df_all[df_all["coupling"] == coup]
            if len(subset) > 0:
                subset = subset.sort_values("T")
                axes[1].plot(
                    subset["T"],
                    subset["r_c"],
                    marker=".",
                    markersize=3,
                    label=f"g={coup:.3f}",
                )
        axes[1].set_xlabel("T")
        axes[1].set_ylabel("r_c")
        axes[1].set_title("Critical Bubble Radius")
        axes[1].legend(fontsize=7, loc="best")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")

        for coup in unique_couplings[::step]:
            subset = df_all[df_all["coupling"] == coup]
            if len(subset) > 0:
                subset = subset.sort_values("T")
                axes[2].plot(
                    subset["T"],
                    subset["phi_esc"],
                    marker=".",
                    markersize=3,
                    label=f"g={coup:.3f}",
                )
        axes[2].set_xlabel("T")
        axes[2].set_ylabel("phi_esc")
        axes[2].set_title("Escape Point")
        axes[2].legend(fontsize=7, loc="best")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_plot.png", dpi=200)
        print(f"Saved plot: {output_dir}/summary_plot.png")
        plt.show()

        if len(unique_couplings) > 3 and len(df_all) > 20:
            print("\nGenerating 2D heatmap...")
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

            for idx, (col, title) in enumerate(
                [("S3/T", "S3/T"), ("r_c", "log10(r_c)"), ("phi_esc", "phi_esc")]
            ):
                try:
                    pivot = df_all.pivot_table(
                        values=col, index="coupling", columns="T", aggfunc="mean"
                    )
                    if col == "r_c":
                        pivot = np.log10(pivot)
                    im = axes2[idx].imshow(
                        pivot.values,
                        aspect="auto",
                        origin="lower",
                        extent=[
                            pivot.columns.min(),
                            pivot.columns.max(),
                            pivot.index.min(),
                            pivot.index.max(),
                        ],
                        cmap="viridis",
                    )
                    axes2[idx].set_xlabel("T")
                    axes2[idx].set_ylabel("coupling")
                    axes2[idx].set_title(title)
                    plt.colorbar(im, ax=axes2[idx])
                except Exception as e:
                    print(f"  Could not create heatmap for {col}: {e}")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/heatmap_plot.png", dpi=200)
            print(f"Saved heatmap: {output_dir}/heatmap_plot.png")
            plt.show()

    except Exception as e:
        print(f"Could not generate plot: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("SCAN COMPLETE")
    print("=" * 70)
