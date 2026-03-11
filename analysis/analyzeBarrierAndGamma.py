import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))

if True:
    """
    Analyze:
    1. T_c2 (barrier disappearance temperature) vs lambda
    2. Gamma/H^4 for lambda < 0.5

    T_c2 is the temperature where the potential barrier between false and
    true vacuum disappears.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import glob
    import math
    from scipy.optimize import minimize_scalar
    from scipy.integrate import quad

    import cosmoTransitions.finiteT as CTFT
    import cosmoTransitions.pathDeformation as CTPD

    import Potential as p

    # "V_correct" = boson + fermion, "fermion_only" = fermion only
    potential_flag = "V_correct"

    # =============================================================================
    # Physical parameters (base)
    # =============================================================================
    lam = 1e-16
    mphi = 1000
    epsil = 0
    lambdaSix = 0

    bosonMassSquared = 1000000
    bosonGaugeCoupling = 1.05  # FIXED
    fermionGaugeCoupling = 1.05  # FIXED

    param_set = "set6"

    # Cosmological constants for Hubble calculation
    MPL = 2.4e18  # Reduced Planck mass
    delV = 10**28  # Vacuum energy difference
    chig2 = 30 / (math.pi**2 * 106.75)


    def Hubble(T, delV=delV):
        """Hubble parameter H(T)."""
        if isinstance(T, np.ndarray):
            Hub2 = (T**4 / chig2 + delV) / (3 * MPL**2)
            return np.sqrt(Hub2.astype("float"))
        else:
            return np.sqrt((T**4 / chig2 + delV) / (3 * MPL**2))


    def create_potential(coupling):
        """Create potential with given coupling (lambda_b = lambda_f)."""
        param_dict = {
            "lambda": lam,
            "mphi": mphi,
            "epsilon": epsil,
            "lambdaSix": lambdaSix,
            "bosonMassSquared": bosonMassSquared,
            "bosonCoupling": coupling,
            "bosonGaugeCoupling": bosonGaugeCoupling,
            "fermionCoupling": coupling,
            "fermionGaugeCoupling": fermionGaugeCoupling,
        }
        return p.finiteTemperaturePotential(param_dict)


    def V_at_T(VT, phi, T):
        """Evaluate potential at given phi and T."""
        VT.update_T(T)
        X = np.array([[phi]])
        if potential_flag == "fermion_only":
            return VT.V_p_fermion_only(X)[0]
        return VT.V_p_correct(X)[0]


    def dV_at_T(VT, phi, T):
        """Evaluate derivative of potential at given phi and T."""
        VT.update_T(T)
        X = np.array([[phi]])
        if potential_flag == "fermion_only":
            return VT.dV_p_fermion_only(X)[0]
        return VT.dV_p_correct(X)[0]


    def find_barrier_height(VT, T, phi_range=(1e3, 1e8)):
        """
        Find barrier height at temperature T using minimize on V.
        Returns (barrier_exists, barrier_top_phi, barrier_height).
        """
        VT.update_T(T)

        # Define V as a function of phi for this T
        def V_func(phi):
            return V_at_T(VT, phi, T)

        # Find the false vacuum (minimum near phi=0)
        V_false = V_at_T(VT, 0, T)

        # Find the true vacuum (global minimum at large phi)
        # Use minimize_scalar to find the minimum in the range
        result_true = minimize_scalar(
            V_func,
            bounds=(phi_range[0], phi_range[1]),
            method="bounded",
        )
        if not result_true.success:
            return False, None, 0.0

        phi_true = result_true.x
        V_true = result_true.fun

        # Find the barrier top (local maximum between false and true vacuum)
        # Minimize -V to find the maximum
        def neg_V_func(phi):
            return -V_at_T(VT, phi, T)

        # Search for barrier between phi=0 and phi_true
        # Use a reasonable upper bound for barrier search
        barrier_search_max = min(phi_true * 0.9, phi_range[1] * 0.5)
        barrier_search_min = phi_range[0] * 0.1

        result_barrier = minimize_scalar(
            neg_V_func,
            bounds=(barrier_search_min, barrier_search_max),
            method="bounded",
        )

        if not result_barrier.success:
            return False, None, 0.0

        barrier_phi = result_barrier.x
        V_barrier = -result_barrier.fun  # Convert back from -V

        # Check if this is actually a barrier (V_barrier > V_false)
        barrier_height = V_barrier - V_false

        if barrier_height <= 0:
            # No barrier exists
            return False, None, 0.0

        # Also check that true vacuum is lower than false vacuum
        if V_true >= V_false:
            # No proper phase transition
            return False, None, 0.0

        return True, barrier_phi, barrier_height


    def find_Tc2(VT, T_range=(1000, 1e7)):
        """
        Find T_c2: temperature where barrier disappears.
        Binary search for the temperature where barrier height goes to zero.
        """
        T_low, T_high = T_range

        # Check endpoints
        has_barrier_low, _, _ = find_barrier_height(VT, T_low)
        has_barrier_high, _, _ = find_barrier_height(VT, T_high)

        if has_barrier_low and has_barrier_high:
            print("  Warning: Barrier exists at both T endpoints")
            return None
        if not has_barrier_low and not has_barrier_high:
            print("  Warning: No barrier at either T endpoint")
            return None

        # Binary search
        for _ in range(50):
            T_mid = (T_low + T_high) / 2
            has_barrier, _, _ = find_barrier_height(VT, T_mid)

            if has_barrier:
                T_low = T_mid
            else:
                T_high = T_mid

            if abs(T_high - T_low) < 1:
                break

        return (T_low + T_high) / 2


    # =============================================================================
    # Part 1: Find T_c2 vs lambda
    # =============================================================================
    print("=" * 70)
    print("PART 1: Finding T_c2 (barrier disappearance) vs lambda")
    print("=" * 70)

    # Coupling values to scan
    coupling_list_Tc2 = np.arange(0.15, 1.25, 0.05)

    Tc2_results = []

    for coupling in coupling_list_Tc2:
        print(f"\nλ = {coupling:.3f}:")
        VT = create_potential(coupling)

        # Find T_c2
        Tc2 = find_Tc2(VT, T_range=(1000, 2e6))

        if Tc2 is not None:
            print(f"  T_c2 = {Tc2:.1f}")
            Tc2_results.append({"lambda": coupling, "Tc2": Tc2})
        else:
            print("  T_c2 not found in range")

    # Convert to DataFrame
    df_Tc2 = pd.DataFrame(Tc2_results)
    print("\n" + "=" * 70)
    print("T_c2 Results:")
    print(df_Tc2.to_string(index=False))

    # Save results
    output_dir = f"data/tunneling/{param_set}/analysis_{potential_flag}"
    os.makedirs(output_dir, exist_ok=True)
    df_Tc2.to_csv(f"{output_dir}/Tc2_vs_lambda.csv", index=False)
    print(f"\nSaved: {output_dir}/Tc2_vs_lambda.csv")

    # =============================================================================
    # Part 2: Check Gamma/H^4 for lambda < 0.5
    # =============================================================================
    print("\n" + "=" * 70)
    print("PART 2: Checking Gamma/H^4 for lambda < 0.5")
    print("=" * 70)

    # Load CSV files for lambda < 0.5
    data_dir = f"data/tunneling/{param_set}/coupling_temp_scan_{potential_flag}"
    csv_files = glob.glob(f"{data_dir}/coupling_*.csv")

    small_lambda_data = []

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        if filename == "all_coupling_temp_scan.csv":
            continue

        # Extract coupling value
        coup_str = filename.replace("coupling_", "").replace(".csv", "")
        coup_val = float(coup_str.replace("p", "."))

        if coup_val >= 0.5:
            continue

        # Load data
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue

        # Filter out S3/T < 10 (potential calculation error)
        df = df[df["S3/T"] >= 10]
        if len(df) == 0:
            print(f"\nλ = {coup_val:.4f}: All data has S3/T < 10, skipping")
            continue

        print(f"\nλ = {coup_val:.4f}:")
        print(f"  Data points: {len(df)} (after filtering S3/T >= 10)")
        print(f"  T range: [{df['T'].min():.0f}, {df['T'].max():.0f}]")

        # Compute Gamma/H^4
        T_arr = df["T"].values
        S3_T = df["S3/T"].values

        # log(Gamma) ≈ -S3/T + 4*log(T) + 3/2*log(S3/T / 2π)
        log_Gamma = -S3_T + 4 * np.log(T_arr) + 1.5 * np.log(S3_T / (2 * np.pi))
        Gamma = np.exp(log_Gamma)

        H = Hubble(T_arr)
        Gamma_H4 = Gamma / (H**4)

        max_Gamma_H4 = np.max(Gamma_H4)
        idx_max = np.argmax(Gamma_H4)

        print(f"  max(Γ/H⁴) = {max_Gamma_H4:.2e} at T = {T_arr[idx_max]:.0f}")
        print(f"  Γ/H⁴ > 1? {'YES ✓' if max_Gamma_H4 > 1 else 'NO ✗'}")

        # Check if we need finer temperature grid
        if max_Gamma_H4 < 1:
            print("  → May need finer T grid or lower T range")

        small_lambda_data.append({
            "lambda": coup_val,
            "T_min": df["T"].min(),
            "T_max": df["T"].max(),
            "max_Gamma_H4": max_Gamma_H4,
            "T_at_max": T_arr[idx_max],
            "exceeds_1": max_Gamma_H4 > 1,
        })

    # Summary
    print("\n" + "=" * 70)
    print("Summary for λ < 0.5:")
    print("=" * 70)
    df_small = pd.DataFrame(small_lambda_data)
    if len(df_small) > 0:
        print(df_small.to_string(index=False))
        df_small.to_csv(f"{output_dir}/small_lambda_Gamma_H4.csv", index=False)
        print(f"\nSaved: {output_dir}/small_lambda_Gamma_H4.csv")
    else:
        print("No data found for λ < 0.5")

    # =============================================================================
    # Part 3: Run extra tunneling at lower temperatures for lambda < 0.5
    # =============================================================================
    print("\n" + "=" * 70)
    print("PART 3: Running extra tunneling at lower T (step = 0.5 GeV = 500)")
    print("=" * 70)

    # Temperature step (finer resolution for precise Gamma/H^4 search)
    T_STEP = 0.05  # Small step for fine temperature resolution

    extended_results = []

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        if filename == "all_coupling_temp_scan.csv":
            continue

        # Extract coupling value
        coup_str = filename.replace("coupling_", "").replace(".csv", "")
        coup_val = float(coup_str.replace("p", "."))

        if coup_val >= 0.5:
            continue

        # Load existing data
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue

        # Filter out S3/T < 10 (potential calculation error)
        df = df[df["S3/T"] >= 10]
        if len(df) == 0:
            print(f"\nλ = {coup_val:.4f}: All data has S3/T < 10, skipping")
            continue

        # Find the temperature where S3/T > 100 (lower T = higher S3/T)
        # Sort by T ascending (low to high)
        df_sorted = df.sort_values("T", ascending=True)

        # Find first T where S3/T > 100
        mask = df_sorted["S3/T"] > 100
        if mask.any():
            T_start = df_sorted[mask]["T"].iloc[0]
            S3_T_at_start = df_sorted[mask]["S3/T"].iloc[0]
        else:
            # If no S3/T > 100, use minimum T
            T_start = df["T"].min()
            S3_T_at_start = df[df["T"] == T_start]["S3/T"].iloc[0]

        print(f"\nλ = {coup_val:.4f}:")
        print(f"  Found S3/T > 100 at T = {T_start:.0f} (S3/T = {S3_T_at_start:.1f})")
        print(f"  Starting from T = {T_start:.0f}, stepping down by {T_STEP}")

        # Create potential with this coupling
        VT = create_potential(coup_val)

        # Storage for new data
        new_T_list = []
        new_S3_T_list = []
        new_r_c_list = []
        new_phi_esc_list = []

        # Estimate true vacuum position
        tv_estimate = 70000  # Initial estimate

        # Run tunneling at lower temperatures
        T_current = T_start - T_STEP
        max_iterations = 5000  # Increased for finer T_STEP

        for iteration in range(max_iterations):
            if T_current <= 0:
                print(f"  T reached 0, stopping")
                break

            print(f"  T = {T_current:.1f} ... ", end="", flush=True)

            VT.update_T(T_current)

            if potential_flag == "fermion_only":
                _V_fn = VT.V_fermion_only
                _dV_fn = VT.dV_p_fermion_only
            else:
                _V_fn = VT.V_correct
                _dV_fn = VT.dV_p_correct

            try:
                # Compute tunneling
                tunneling_result = CTPD.fullTunneling(
                    path_pts=np.array([[tv_estimate], [0]]),
                    V=_V_fn,
                    dV=_dV_fn,
                    maxiter=1,
                    V_spline_samples=200,
                    tunneling_init_params=dict(alpha=2),
                    tunneling_findProfile_params=dict(
                        xtol=0.00001, phitol=0.00001, rmin=0.00001, npoints=200
                    ),
                    deformation_class=CTPD.Deformation_Spline,
                )

                S3_T = tunneling_result.action / T_current
                _R = tunneling_result.profile1D.R
                _Phi = tunneling_result.profile1D.Phi
                _phi_mid = 0.5 * (_Phi[0] + _Phi[-1])
                r_c = np.interp(_phi_mid, _Phi[::-1], _R[::-1]) if _Phi[0] > _Phi[-1] else np.interp(_phi_mid, _Phi, _R)
                phi_esc = _Phi[0]

                # Skip results with S3/T < 10 (potential calculation error)
                if S3_T < 10:
                    print(f"S3/T={S3_T:.2f} < 10, skipping (potential error)")
                    T_current -= T_STEP
                    continue

                new_T_list.append(T_current)
                new_S3_T_list.append(S3_T)
                new_r_c_list.append(r_c)
                new_phi_esc_list.append(phi_esc)

                # Compute Gamma/H^4
                log_Gamma = -S3_T + 4 * np.log(T_current) + 1.5 * np.log(
                    S3_T / (2 * np.pi)
                )
                Gamma_H4 = np.exp(log_Gamma) / (Hubble(T_current) ** 4)

                print(f"S3/T={S3_T:.2f}, Γ/H⁴={Gamma_H4:.2e}")

                # Check if we found Gamma/H^4 > 1
                if Gamma_H4 > 1:
                    print(f"  *** Found Γ/H⁴ > 1 at T = {T_current:.1f}! ***")
                    # Continue a bit more to find the peak, then stop
                    # (or set STOP_AFTER_FOUND = True to stop immediately)
                    STOP_AFTER_FOUND = False
                    if STOP_AFTER_FOUND:
                        print("  Stopping (STOP_AFTER_FOUND=True)")
                        break

                # Move to next temperature
                T_current -= T_STEP

            except Exception as e:
                print(f"FAILED: {e}")
                print(f"  Stopping at T = {T_current + T_STEP:.1f}")
                break

        # Combine old and new data
        if len(new_T_list) > 0:
            # Compute Gamma/H^4 for all new points
            new_T_arr = np.array(new_T_list)
            new_S3_T_arr = np.array(new_S3_T_list)
            log_Gamma_new = (
                -new_S3_T_arr
                + 4 * np.log(new_T_arr)
                + 1.5 * np.log(new_S3_T_arr / (2 * np.pi))
            )
            Gamma_H4_new = np.exp(log_Gamma_new) / (Hubble(new_T_arr) ** 4)

            max_new_Gamma_H4 = np.max(Gamma_H4_new)
            idx_max_new = np.argmax(Gamma_H4_new)

            print(f"\n  Extended results for λ = {coup_val:.4f}:")
            print(f"    New T range: [{new_T_arr.min():.0f}, {new_T_arr.max():.0f}]")
            print(f"    New points: {len(new_T_arr)}")
            print(
                f"    max(Γ/H⁴) in new data: {max_new_Gamma_H4:.2e} "
                f"at T = {new_T_arr[idx_max_new]:.0f}"
            )
            print(
                f"    Γ/H⁴ > 1 in new data? "
                f"{'YES ✓' if max_new_Gamma_H4 > 1 else 'NO ✗'}"
            )

            # Save extended data (including Gamma_H4)
            df_new = pd.DataFrame({
                "T": new_T_list,
                "S3/T": new_S3_T_list,
                "r_c": new_r_c_list,
                "phi_esc": new_phi_esc_list,
                "Gamma_H4": Gamma_H4_new.tolist(),
            })
            extended_file = f"{output_dir}/extended_coupling_{coup_str}.csv"
            df_new.to_csv(extended_file, index=False)
            print(f"    Saved: {extended_file}")

            # Also store for combined CSV
            for i in range(len(new_T_list)):
                extended_results.append({
                    "lambda": coup_val,
                    "T": new_T_list[i],
                    "S3/T": new_S3_T_list[i],
                    "r_c": new_r_c_list[i],
                    "phi_esc": new_phi_esc_list[i],
                    "Gamma_H4": Gamma_H4_new[i],
                })

    # Save all extended results to combined CSV
    print("\n" + "=" * 70)
    print("Summary of Extended Tunneling Results:")
    print("=" * 70)
    if len(extended_results) > 0:
        df_extended_all = pd.DataFrame(extended_results)

        # Save combined CSV with all data points
        combined_file = f"{output_dir}/all_extended_tunneling.csv"
        df_extended_all.to_csv(combined_file, index=False)
        print(f"Saved all extended data: {combined_file}")
        print(f"Total data points: {len(df_extended_all)}")

        # Create summary per coupling
        summary_list = []
        for coup in df_extended_all["lambda"].unique():
            df_coup = df_extended_all[df_extended_all["lambda"] == coup]
            max_G_H4 = df_coup["Gamma_H4"].max()
            idx_max = df_coup["Gamma_H4"].idxmax()
            summary_list.append({
                "lambda": coup,
                "T_min": df_coup["T"].min(),
                "T_max": df_coup["T"].max(),
                "n_points": len(df_coup),
                "max_Gamma_H4": max_G_H4,
                "T_at_max_Gamma_H4": df_coup.loc[idx_max, "T"],
                "exceeds_1": max_G_H4 > 1,
            })

        df_summary = pd.DataFrame(summary_list)
        print("\nSummary per coupling:")
        print(df_summary.to_string(index=False))

        summary_file = f"{output_dir}/extended_tunneling_summary.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"\nSaved summary: {summary_file}")
    else:
        print("No extended tunneling results")

    # =============================================================================
    # Plot results
    # =============================================================================
    print("\n" + "=" * 70)
    print("Creating plots...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: T_c2 vs lambda
    ax1 = axes[0]
    if len(df_Tc2) > 0:
        ax1.plot(df_Tc2["lambda"], df_Tc2["Tc2"] / 1000, "bo-", linewidth=2)
        ax1.set_xlabel(r"$\lambda_b = \lambda_f$", fontsize=12)
        ax1.set_ylabel(r"$T_{c2}$ (TeV)", fontsize=12)
        ax1.set_title(r"Barrier Disappearance Temperature $T_{c2}$", fontsize=14)
        ax1.grid(True, alpha=0.3)

    # Plot 2: max(Gamma/H^4) vs lambda for small lambda (original data)
    ax2 = axes[1]
    if len(df_small) > 0:
        colors = ["green" if x else "red" for x in df_small["exceeds_1"]]
        ax2.scatter(
            df_small["lambda"],
            df_small["max_Gamma_H4"],
            c=colors,
            s=100,
            edgecolors="black",
            label="Original data",
        )
        ax2.axhline(1, linestyle="--", color="black", linewidth=1, alpha=0.7)
        ax2.set_xlabel(r"$\lambda_b = \lambda_f$", fontsize=12)
        ax2.set_ylabel(r"max($\Gamma/H^4$)", fontsize=12)
        ax2.set_title(r"Max $\Gamma/H^4$ for $\lambda < 0.5$ (Original)", fontsize=14)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    # Plot 3: max(Gamma/H^4) vs lambda for extended data
    ax3 = axes[2]
    if len(extended_results) > 0 and "df_summary" in dir():
        colors_ext = ["green" if x else "red" for x in df_summary["exceeds_1"]]
        ax3.scatter(
            df_summary["lambda"],
            df_summary["max_Gamma_H4"],
            c=colors_ext,
            s=100,
            edgecolors="black",
            marker="s",
        )
        ax3.axhline(1, linestyle="--", color="black", linewidth=1, alpha=0.7)
        ax3.set_xlabel(r"$\lambda_b = \lambda_f$", fontsize=12)
        ax3.set_ylabel(r"max($\Gamma/H^4$)", fontsize=12)
        ax3.set_title(r"Max $\Gamma/H^4$ for $\lambda < 0.5$ (Extended)", fontsize=14)
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5, 0.5, "No extended data",
            ha="center", va="center", transform=ax3.transAxes
        )
        ax3.set_title("Extended Data (none)", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"figs/Tc2_and_Gamma_H4_analysis_{potential_flag}.png", dpi=200)
    print(f"Saved: figs/Tc2_and_Gamma_H4_analysis_{potential_flag}.png")

    # Also create a combined plot showing original vs extended
    if len(extended_results) > 0 and len(df_small) > 0 and "df_summary" in dir():
        fig2, ax_comb = plt.subplots(figsize=(10, 6))

        # Original data
        ax_comb.scatter(
            df_small["lambda"],
            df_small["max_Gamma_H4"],
            c="blue",
            s=80,
            edgecolors="black",
            label="Original data",
            marker="o",
        )

        # Extended data (use summary)
        ax_comb.scatter(
            df_summary["lambda"],
            df_summary["max_Gamma_H4"],
            c="orange",
            s=80,
            edgecolors="black",
            label="Extended (lower T)",
            marker="s",
        )

        ax_comb.axhline(1, linestyle="--", color="black", linewidth=1.5, alpha=0.7)
        ax_comb.set_xlabel(r"$\lambda_b = \lambda_f$", fontsize=12)
        ax_comb.set_ylabel(r"max($\Gamma/H^4$)", fontsize=12)
        ax_comb.set_title(
            r"Max $\Gamma/H^4$ for $\lambda < 0.5$: Original vs Extended",
            fontsize=14,
        )
        ax_comb.set_yscale("log")
        ax_comb.grid(True, alpha=0.3)
        ax_comb.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f"figs/Gamma_H4_original_vs_extended_{potential_flag}.png", dpi=200)
        print(f"Saved: figs/Gamma_H4_original_vs_extended_{potential_flag}.png")

    plt.show()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
