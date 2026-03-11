"""
Compare bounce profiles phi(r) for different coupling values lambda.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import cosmoTransitions.pathDeformation as CTPD
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))

import Potential as p

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

# Temperature for bounce calculation
TEMP = 7700  # Temperature where tunneling is computed

# Coupling values to compare
COUPLING_LIST = [1.05, 1.09, 1.19]

# Output directory
output_dir = f"data/tunneling/{param_set}/analysis"
os.makedirs(output_dir, exist_ok=True)


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


# Colors for different couplings
colors = {1.05: "blue", 1.09: "red", 1.19: "green", 0.9: "purple"}

# =============================================================================
# Compute bounce profiles
# =============================================================================
print("\n" + "=" * 70)
print(f"Computing bounce profiles at T = {TEMP}")
print("=" * 70)

profiles = {}
profile_data = []
fv = 0

for coupling in COUPLING_LIST:
    print(f"\nλ = {coupling:.2f}:")

    # Create potential
    VT = create_potential(coupling)
    VT.update_T(TEMP)

    # Estimate true vacuum position (depends on coupling)
    tv_estimate = 70000 if coupling > 0.8 else 50000

    try:
        # Compute tunneling
        tunneling_result = CTPD.fullTunneling(
            path_pts=np.array([[tv_estimate], [fv]]),
            V=VT.V,
            dV=VT.dV_p,
            maxiter=1,
            V_spline_samples=200,
            tunneling_init_params=dict(alpha=2),
            tunneling_findProfile_params=dict(
                xtol=0.00001, phitol=0.00001, rmin=0.00001, npoints=500
            ),
            deformation_class=CTPD.Deformation_Spline,
        )

        # Extract profile
        R = tunneling_result.profile1D.R.copy()
        Phi = tunneling_result.profile1D.Phi.copy()

        # Get key parameters
        S3 = tunneling_result.action
        S3_T = S3 / TEMP
        _phi_mid = 0.5 * (Phi[0] + Phi[-1])
        r_c = np.interp(_phi_mid, Phi[::-1], R[::-1]) if Phi[0] > Phi[-1] else np.interp(_phi_mid, Phi, R)
        phi_diff = Phi[-1] - fv
        Phi = phi_diff - Phi
        phi_esc = Phi[0] # Field value at center (r=0) = escape point in physical manner


        # False vacuum should be at φ = 0 (boundary condition)
        # CosmoTransitions may give small non-zero value numerically
        phi_fv = fv  # Physical false vacuum

        print(f"  S₃/T = {S3_T:.2f}")
        print(f"  r_c = {r_c:.4f}")
        print(f"  φ_esc = φ(0) = {phi_esc:.2e}  (escape point)")
        print(f"  φ_fv = φ(∞) = {phi_fv:.2e}  (false vacuum, by definition)")

        profiles[coupling] = {
            "R": R,
            "Phi": Phi,
            "S3_T": S3_T,
            "r_c": r_c,
            "phi_esc": phi_esc,
            "phi_fv": phi_fv,
        }

        # Save individual profile to CSV
        df_profile = pd.DataFrame(
            {
                "r": R,
                "phi": Phi,
            }
        )
        coup_str = f"{coupling:.2f}".replace(".", "p")
        profile_file = f"{output_dir}/bounce_profile_lambda_{coup_str}_T_{TEMP}.csv"
        df_profile.to_csv(profile_file, index=False)
        print(f"  Saved: {profile_file}")

    except Exception as e:
        print(f"  FAILED: {e}")
        profiles[coupling] = None

# =============================================================================
# Plot comparison
# =============================================================================
print("\n" + "=" * 70)
print("Creating comparison plot...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: phi(r) - linear scale
ax1 = axes[0]
for coupling in COUPLING_LIST:
    if profiles[coupling] is not None:
        R = profiles[coupling]["R"]
        Phi = profiles[coupling]["Phi"]
        S3_T = profiles[coupling]["S3_T"]
        phi_esc = profiles[coupling]["phi_esc"]  # Escape point at r=0
        phi_fv = profiles[coupling]["phi_fv"]  # False vacuum = 0
        color = colors.get(coupling, "black")

        label = f"λ = {coupling:.2f} (S₃/T = {S3_T:.1f})"
        ax1.plot(R, Phi, color=color, linewidth=2, label=label)

        # Mark escape point (r=0)
        ax1.scatter(
            [R[0]],
            [phi_esc],
            color=color,
            s=80,
            marker="o",
            edgecolors="black",
            zorder=5,
        )
        # Mark false vacuum (r→∞) at φ = 0
        ax1.scatter(
            [R[-1]], [phi_fv], color=color, s=80, marker="x", linewidths=2, zorder=5
        )

# Add dummy scatter entries to legend for marker explanation
ax1.scatter([], [], color="gray", s=80, marker="o", edgecolors="black",
            label=r"$\phi_{\rm esc}$ (escape)")
ax1.scatter([], [], color="gray", s=80, marker="x", linewidths=2,
            label=r"$\phi_{\rm fv}$ (false vacuum)")

ax1.set_xlabel(r"$r$ (GeV$^{-1}$)", fontsize=12)
ax1.set_ylabel(r"$\phi(r)$ (GeV)", fontsize=12)
ax1.set_title(
    f"Bounce Profile at T = {TEMP} GeV\n"
    r"$\phi(0) = \phi_{\rm esc}$, $\phi(\infty) = \phi_{\rm fv}$",
    fontsize=13,
)
ax1.legend(fontsize=10, loc="right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

# Plot 2: phi(r) - log scale for phi
ax2 = axes[1]
for coupling in COUPLING_LIST:
    if profiles[coupling] is not None:
        R = profiles[coupling]["R"]
        Phi = profiles[coupling]["Phi"]
        S3_T = profiles[coupling]["S3_T"]
        phi_esc = profiles[coupling]["phi_esc"]
        phi_fv = profiles[coupling]["phi_fv"]  # = 0
        color = colors.get(coupling, "black")

        # Only plot positive phi values for log scale
        mask = Phi > 0
        label = f"λ = {coupling:.2f} (S₃/T = {S3_T:.1f})"
        ax2.semilogy(R[mask], Phi[mask], color=color, linewidth=2, label=label)

        # Mark escape point (if positive)
        if phi_esc > 0:
            ax2.scatter(
                [R[0]],
                [phi_esc],
                color=color,
                s=80,
                marker="o",
                edgecolors="black",
                zorder=5,
            )
        # Note: phi_fv = 0 cannot be shown on log scale

# Add dummy scatter entry to legend for escape point marker
ax2.scatter([], [], color="gray", s=80, marker="o", edgecolors="black",
            label=r"$\phi_{\rm esc}$ (escape)")

ax2.set_xlabel(r"$r$ (GeV$^{-1}$)", fontsize=12)
ax2.set_ylabel(r"$\phi(r)$ (GeV)", fontsize=12)
ax2.set_title(
    f"Bounce Profile (log scale) at T = {TEMP} GeV\n"
    r"$\phi_{\rm fv} = 0$ (not shown on log scale)",
    fontsize=13,
)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)

plt.tight_layout()
plt.savefig(f"figs/bounce_profile_comparison_T_{TEMP}.png", dpi=200)
print(f"Saved: figs/bounce_profile_comparison_T_{TEMP}.png")

# =============================================================================
# Plot: Potential V(phi) comparison with escape points from pre-computed CSVs
# =============================================================================
print("\nCreating potential comparison plot...")

# Load escape points from existing coupling scan CSVs (no tunneling needed)
scan_dir = f"data/tunneling/{param_set}/coupling_temp_scan"
phi_esc_from_csv = {}
for coupling in COUPLING_LIST:
    coup_str = f"{coupling:.4f}".replace(".", "p")
    csv_path = f"{scan_dir}/coupling_{coup_str}.csv"
    if os.path.exists(csv_path):
        df_scan = pd.read_csv(csv_path)
        # Find the row closest to TEMP
        idx = (df_scan["T"] - TEMP).abs().idxmin()
        row = df_scan.iloc[idx]
        # CSV stores raw CosmoTransitions coordinate; convert to false-vacuum=0 frame
        tv_estimate = 70000 if coupling > 0.8 else 50000
        phi_esc_physical = tv_estimate - row["phi_esc"]
        phi_esc_from_csv[coupling] = phi_esc_physical
        print(f"  λ={coupling:.2f}: φ_esc_raw={row['phi_esc']:.1f}, φ_esc={phi_esc_physical:.1f} at T={row['T']:.0f}")
    else:
        print(f"  λ={coupling:.2f}: CSV not found at {csv_path}")

phi_max = 30000.0
phi_arr = np.linspace(1e-3, phi_max, 5000)
X_arr = phi_arr.reshape(-1, 1)

fig_pot, ax_pot = plt.subplots(figsize=(8, 6))

for coupling in COUPLING_LIST:
    VT = create_potential(coupling)
    VT.update_T(TEMP)

    V_vals = VT.V(X_arr)
    V0 = VT.V(np.array([[1e-3]]))
    V_shifted = np.nan_to_num(V_vals - V0, nan=0.0)

    color = colors.get(coupling, "black")
    label = f"λ = {coupling:.2f}"
    ax_pot.plot(phi_arr, V_shifted, color=color, linewidth=2, label=label)

    # Mark escape point from CSV if available
    if coupling in phi_esc_from_csv:
        phi_esc = phi_esc_from_csv[coupling]
        X_esc = np.array([[phi_esc]])
        V_esc = np.nan_to_num(VT.V(X_esc) - V0, nan=0.0)
        ax_pot.scatter(
            [phi_esc], [V_esc], color=color, s=80,
            marker="o", edgecolors="black", zorder=5,
        )

# Dummy scatter for legend
ax_pot.scatter([], [], color="gray", s=80, marker="o", edgecolors="black",
               label=r"$\phi_{\rm esc}$ (escape point)")

ax_pot.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax_pot.set_xlabel(r"$\phi$ (GeV)", fontsize=13)
ax_pot.set_ylabel(r"$V(\phi) - V(0)$ (GeV$^4$)", fontsize=13)
ax_pot.set_title(f"Effective Potential at T = {TEMP} GeV", fontsize=14)
ax_pot.legend(fontsize=10)
ax_pot.grid(True, alpha=0.3)
ax_pot.set_xlim(left=0, right=phi_max)

plt.tight_layout()
pot_file = f"figs/potential_comparison_T_{TEMP}.png"
plt.savefig(pot_file, dpi=200)
print(f"Saved: {pot_file}")

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

summary_data = []
for coupling in COUPLING_LIST:
    if profiles[coupling] is not None:
        summary_data.append(
            {
                "lambda": coupling,
                "S3/T": profiles[coupling]["S3_T"],
                "r_c": profiles[coupling]["r_c"],
                "phi_esc (r=0)": profiles[coupling]["phi_esc"],
                "phi_fv (r=inf)": profiles[coupling]["phi_fv"],  # = 0
            }
        )

if summary_data:
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Save summary
    summary_file = f"{output_dir}/bounce_profile_summary_T_{TEMP}.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")

plt.show()

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
