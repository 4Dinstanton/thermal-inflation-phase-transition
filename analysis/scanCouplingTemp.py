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

import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))
import Potential as p

# "V_correct" = boson + fermion (Jb + Jf)
# "fermion_only" = fermion only (Jf)
potential_flag = "fermion_only"


def format_e(n):
    """Format number in scientific notation."""
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "E" + a.split("E")[1]


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

    tunneling_result = CTPD.fullTunneling(
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
    )

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

    param_set = "set6"

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
        }
    }

    # =============================================================================
    # Scan ranges
    # =============================================================================
    COUPLING_LIST = np.arange(1.00, 1.21, 0.01)
    TEMP_LIST = np.arange(6000, 12000, 20)

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

    N_WORKERS = min(len(TEMP_LIST), multiprocessing.cpu_count())
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
    print(f"Temperature range: {TEMP_LIST[0]:.1f} to {TEMP_LIST[-1]:.1f}")
    print(
        f"Total combinations: {len(COUPLING_LIST)} x {len(TEMP_LIST)} = "
        f"{len(COUPLING_LIST) * len(TEMP_LIST)}"
    )
    print(f"Workers: {N_WORKERS}")
    print(f"Potential: {potential_flag}")
    print("=" * 70)

    for i_coup, COUP in enumerate(COUPLING_LIST):
        print(f"\n{'='*70}")
        print(f"Coupling {i_coup+1}/{len(COUPLING_LIST)}: g = {COUP:.4f}")
        print("=" * 70)

        param[param_set]["bosonCoupling"] = float(COUP)
        param[param_set]["fermionCoupling"] = float(COUP)

        S3_T_list = []
        r_c_list = []
        phi_esc_list = []
        successful_temps = []

        _t0 = _time.time()

        S3T_CUTOFF = 1000.0
        BATCH_SIZE = max(1, N_WORKERS * 2)
        n_done = 0
        n_fail = 0
        hit_cutoff = False

        for batch_start in range(0, len(TEMP_LIST), BATCH_SIZE):
            if hit_cutoff:
                break

            batch_temps = TEMP_LIST[batch_start : batch_start + BATCH_SIZE]
            worker_args = []
            for TEMP in batch_temps:
                if TEMP <= 80000:
                    tv = 100000
                elif TEMP < 15000:
                    tv = 120000 * TEMP / 10000
                elif TEMP < 60000:
                    tv = 1000000 * TEMP / 12000
                else:
                    tv = 3500000 * TEMP / 12000
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

            for TEMP in batch_temps:
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
                        break

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
