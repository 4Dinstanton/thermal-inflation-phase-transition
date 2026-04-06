"""
Plot comparison of tunneling parameters across different couplings.
Reads coupling_*p*.csv files and plots:
- S_3 (bounce action)
- beta/H (nucleation rate parameter)
- r_c (critical bubble radius)
- phi_esc (escape point)

Uses the most overlapping temperature range across all couplings.
"""

import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad


# =============================================================================
# Fitting functions for log(Gamma)
# =============================================================================
def rev(x, a, b, c, d, e, f):
    """Fitting function for log(Gamma)."""
    return a + b * x + c * x**2 + d * x**3 + e * np.exp(-f * x)


def drev(x, a, b, c, d, e, f):
    """Derivative of rev for beta/H calculation."""
    return b + 2 * c * x + 3 * d * x**2 - e * f * np.exp(-f * x)


# =============================================================================
# Cosmological functions (from drawAction.py)
# =============================================================================
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


def nT(T, popt, T_max):
    """Number of bubbles per Hubble volume n(T)."""

    def f(y):
        return np.exp(rev(y, *popt)) / (Hubble(T) ** 4 * y)

    return quad(f, T, T_max, limit=100)[0]


def inner_integral(T, Tp):
    """Inner integral for percolation calculation."""
    return quad(lambda x: 1 / Hubble(x), T, Tp, limit=100)[0]


def percol(T, popt, T_max):
    """Percolation integral I(T)."""
    prefactor = 4 * math.pi / 3

    def outer_integrand(Tp):
        J = inner_integral(T, Tp)
        return np.exp(rev(Tp, *popt)) / (Hubble(Tp) * Tp**4) * J**3

    integral, _ = quad(outer_integrand, T, T_max, limit=100)
    return prefactor * integral


# =============================================================================
# Configuration
# =============================================================================
param_set = "set7"
# "V_correct" = boson + fermion, "fermion_only" = fermion only
potential_flag = "fermion_only"

data_dir = f"data/tunneling/{param_set}/coupling_temp_scan_{potential_flag}"
output_dir = f"figs/coupling_comparison_{potential_flag}"
os.makedirs(output_dir, exist_ok=True)

# Coupling range to include
COUPLING_MIN = 1.03
COUPLING_MAX = 1.16

# =============================================================================
# Load all coupling CSV files
# =============================================================================
print("=" * 70)
print("Loading coupling data files...")
print("=" * 70)

# Find all coupling CSV files
csv_files = glob.glob(f"{data_dir}/coupling_*.csv")
print(f"Found {len(csv_files)} coupling files")

# Load and filter by coupling range
all_data = {}
for csv_file in sorted(csv_files):
    # Extract coupling value from filename (coupling_0p9000.csv -> 0.9000)
    filename = os.path.basename(csv_file)
    coup_str = filename.replace("coupling_", "").replace(".csv", "")
    coup_val = float(coup_str.replace("p", "."))

    # Filter by coupling range
    if COUPLING_MIN <= coup_val <= COUPLING_MAX:
        df = pd.read_csv(csv_file)
        # Treat failed tunneling points (r_c=0) as missing data
        failed = df["r_c"] == 0
        df.loc[failed, ["S3/T", "r_c", "phi_esc"]] = np.nan
        df = df.dropna(subset=["S3/T"])
        if len(df) > 0:
            all_data[coup_val] = df
            print(
                f"  λ={coup_val:.4f}: {len(df)} points, "
                f"T=[{df['T'].min():.0f}, {df['T'].max():.0f}]"
            )

n_couplings = len(all_data)
print(f"\nLoaded {n_couplings} couplings in range " f"[{COUPLING_MIN}, {COUPLING_MAX}]")

if len(all_data) == 0:
    print("ERROR: No data found! Check the data directory and coupling range.")
    exit(1)

# =============================================================================
# Find most overlapping temperature range
# =============================================================================
print("\n" + "=" * 70)
print("Finding overlapping temperature range...")
print("=" * 70)

# Get T ranges for each coupling
T_mins = [df["T"].min() for df in all_data.values()]
T_maxs = [df["T"].max() for df in all_data.values()]

# Temperature range: min of all to max of overlapping (min of T_maxs)
T_overlap_min = min(T_mins)  # Start from minimum
T_overlap_max = min(T_maxs)  # End at max of overlapping region

print("Individual T ranges:")
print(f"  Min of all: {min(T_mins):.0f}")
print(f"  Max of all: {max(T_maxs):.0f}")
print(f"  Max of overlapping: {min(T_maxs):.0f}")
print(f"Plot range: [{T_overlap_min:.0f}, {T_overlap_max:.0f}]")

if T_overlap_min >= T_overlap_max:
    print("WARNING: Invalid temperature range! Using full range.")
    T_overlap_min = min(T_mins)
    T_overlap_max = max(T_maxs)

# Coupling to highlight
HIGHLIGHT_COUPLING = 1.09

# Common temperature array for plotting
T_common = np.linspace(T_overlap_min, T_overlap_max, 500)

# =============================================================================
# Prepare data for plotting
# =============================================================================
print("\n" + "=" * 70)
print("Preparing data for plotting...")
print("=" * 70)

plot_data = {}
for coup, df in sorted(all_data.items()):
    # Filter to overlapping range
    mask = (df["T"] >= T_overlap_min) & (df["T"] <= T_overlap_max)
    df_filtered = df[mask].sort_values("T")

    if len(df_filtered) < 5:
        print(f"  λ={coup:.4f}: Skipping " f"(only {len(df_filtered)} points)")
        continue

    # Compute S_3 = (S3/T) * T
    df_filtered = df_filtered.copy()
    df_filtered["S3"] = df_filtered["S3/T"] * df_filtered["T"]

    # Compute log(Gamma) and beta/H using curve fitting
    # log(Gamma) ≈ -S3/T + 4*log(T) + 3/2*log(S3/T / 2π)
    T_arr = df_filtered["T"].values
    S3_T = df_filtered["S3/T"].values
    log_Gamma = -S3_T + 4 * np.log(T_arr) + 1.5 * np.log(S3_T / (2 * np.pi))

    df_filtered["log_Gamma"] = log_Gamma

    # Fit rev function and compute beta/H = -d(ln Gamma)/d(ln T)
    try:
        p0 = [0, 0, 0, 0, 0, 1e-4]
        popt, _ = curve_fit(rev, T_arr, log_Gamma, maxfev=10000, p0=p0)
        beta_H = -T_arr * drev(T_arr, *popt)
        df_filtered["beta_H"] = beta_H

        s3_min = df_filtered["S3"].min()
        s3_max = df_filtered["S3"].max()
        print(
            f"  λ={coup:.4f}: {len(df_filtered)} points, "
            f"S3=[{s3_min:.2e}, {s3_max:.2e}]"
        )

    except Exception as e:
        print(f"  λ={coup:.4f}: Fit failed ({e}), " "using numerical gradient")
        # Fallback: numpy gradient
        d_logGamma_dT = np.gradient(log_Gamma, T_arr)
        df_filtered["beta_H"] = -T_arr * d_logGamma_dT

    plot_data[coup] = df_filtered

print(f"\nPrepared {len(plot_data)} couplings for plotting")

# =============================================================================
# Create plots
# =============================================================================
print("\n" + "=" * 70)
print("Creating plots...")
print("=" * 70)

# Color map for couplings
couplings = sorted(plot_data.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(couplings)))


def get_plot_style(coup):
    """Return plot style based on whether coupling should be highlighted."""
    is_highlight = abs(coup - HIGHLIGHT_COUPLING) < 0.005
    if is_highlight:
        return {
            "color": "red",
            "linewidth": 3.0,
            "zorder": 10,
            "label": rf"$\lambda$={coup:.2f} ★",
        }
    else:
        idx = couplings.index(coup)
        return {
            "color": colors[idx],
            "linewidth": 1.2,
            "zorder": 1,
            "label": rf"$\lambda$={coup:.2f}",
        }


# Figure 1: S_3 (Bounce Action)
fig1, ax1 = plt.subplots(figsize=(10, 7))
for coup in couplings:
    df = plot_data[coup]
    style = get_plot_style(coup)
    ax1.plot(df["T"] / 1000, df["S3"], **style)
ax1.set_xlabel("T (TeV)", fontsize=12)
ax1.set_ylabel(r"$S_3$", fontsize=12)
ax1.set_title(r"Bounce Action $S_3$ vs Temperature", fontsize=14)
ax1.legend(fontsize=8, loc="best", ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([T_overlap_min / 1000, T_overlap_max / 1000])
plt.tight_layout()
plt.savefig(f"{output_dir}/S3_comparison.png", dpi=200)
print(f"Saved: {output_dir}/S3_comparison.png")

# Figure 2: beta/H
fig2, ax2 = plt.subplots(figsize=(10, 7))
for coup in couplings:
    df = plot_data[coup]
    style = get_plot_style(coup)
    ax2.plot(df["T"] / 1000, df["beta_H"], **style)
ax2.set_xlabel("T (TeV)", fontsize=12)
ax2.set_ylabel(r"$\beta/H$", fontsize=12)
ax2.set_title(r"Nucleation Rate $\beta/H$ vs Temperature", fontsize=14)
ax2.legend(fontsize=8, loc="best", ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([T_overlap_min / 1000, T_overlap_max / 1000])
plt.tight_layout()
plt.savefig(f"{output_dir}/beta_H_comparison.png", dpi=200)
print(f"Saved: {output_dir}/beta_H_comparison.png")

# Figure 3: r_c (Critical Bubble Radius)
fig3, ax3 = plt.subplots(figsize=(10, 7))
for coup in couplings:
    df = plot_data[coup]
    if "r_c" in df.columns:
        style = get_plot_style(coup)
        ax3.plot(df["T"] / 1000, df["r_c"], **style)
ax3.set_xlabel("T (TeV)", fontsize=12)
ax3.set_ylabel(r"$r_c$", fontsize=12)
ax3.set_title(r"Critical Bubble Radius $r_c$ vs Temperature", fontsize=14)
ax3.legend(fontsize=8, loc="best", ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_yscale("log")
ax3.set_xlim([T_overlap_min / 1000, T_overlap_max / 1000])
plt.tight_layout()
plt.savefig(f"{output_dir}/r_c_comparison.png", dpi=200)
print(f"Saved: {output_dir}/r_c_comparison.png")

# Figure 4: phi_esc (Escape Point)
fig4, ax4 = plt.subplots(figsize=(10, 7))
for coup in couplings:
    df = plot_data[coup]
    if "phi_esc" in df.columns:
        style = get_plot_style(coup)
        ax4.plot(df["T"] / 1000, df["phi_esc"], **style)
ax4.set_xlabel("T (TeV)", fontsize=12)
ax4.set_ylabel(r"$\phi_{esc}$", fontsize=12)
ax4.set_title(r"Escape Point $\phi_{esc}$ vs Temperature", fontsize=14)
ax4.legend(fontsize=8, loc="best", ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([T_overlap_min / 1000, T_overlap_max / 1000])
plt.tight_layout()
plt.savefig(f"{output_dir}/phi_esc_comparison.png", dpi=200)
print(f"Saved: {output_dir}/phi_esc_comparison.png")

# Figure 5: Combined 2x2 plot
fig5, axes = plt.subplots(2, 2, figsize=(14, 12))

# S_3
for coup in couplings:
    df = plot_data[coup]
    style = get_plot_style(coup)
    style["linewidth"] = style["linewidth"] * 0.8  # Thinner for combined
    axes[0, 0].plot(df["T"] / 1000, df["S3"], **style)
axes[0, 0].set_xlabel("T (TeV)")
axes[0, 0].set_ylabel(r"$S_3$")
axes[0, 0].set_title(r"Bounce Action $S_3$")
axes[0, 0].legend(fontsize=6, loc="best", ncol=2)
axes[0, 0].grid(True, alpha=0.3)

# beta/H
for coup in couplings:
    df = plot_data[coup]
    style = get_plot_style(coup)
    style["linewidth"] = style["linewidth"] * 0.8
    axes[0, 1].plot(df["T"] / 1000, df["beta_H"], **style)
axes[0, 1].set_xlabel("T (TeV)")
axes[0, 1].set_ylabel(r"$\beta/H$")
axes[0, 1].set_title(r"Nucleation Rate $\beta/H$")
axes[0, 1].legend(fontsize=6, loc="best", ncol=2)
axes[0, 1].grid(True, alpha=0.3)

# r_c
for coup in couplings:
    df = plot_data[coup]
    if "r_c" in df.columns:
        style = get_plot_style(coup)
        style["linewidth"] = style["linewidth"] * 0.8
        axes[1, 0].plot(df["T"] / 1000, df["r_c"], **style)
axes[1, 0].set_xlabel("T (TeV)")
axes[1, 0].set_ylabel(r"$r_c$")
axes[1, 0].set_title(r"Critical Bubble Radius $r_c$")
axes[1, 0].legend(fontsize=6, loc="best", ncol=2)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale("log")

# phi_esc
for coup in couplings:
    df = plot_data[coup]
    if "phi_esc" in df.columns:
        style = get_plot_style(coup)
        style["linewidth"] = style["linewidth"] * 0.8
        axes[1, 1].plot(df["T"] / 1000, df["phi_esc"], **style)
axes[1, 1].set_xlabel("T (TeV)")
axes[1, 1].set_ylabel(r"$\phi_{esc}$")
axes[1, 1].set_title(r"Escape Point $\phi_{esc}$")
axes[1, 1].legend(fontsize=6, loc="best", ncol=2)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(
    rf"Coupling Comparison ($\lambda$ = {COUPLING_MIN} to {COUPLING_MAX}), "
    rf"★ = $\lambda$ = {HIGHLIGHT_COUPLING}",
    fontsize=14,
    y=1.02,
)
plt.tight_layout()
plt.savefig(f"{output_dir}/all_comparison.png", dpi=200)
print(f"Saved: {output_dir}/all_comparison.png")

# =============================================================================
# Figure 6: Nucleation & Percolation (drawAction.py style)
# Plots: n(T), Γ/H⁴, I/0.34, P/0.7 for different couplings
# =============================================================================
print("\n" + "=" * 70)
print("Computing nucleation & percolation quantities...")
print("=" * 70)

# Store computed quantities for each coupling
nucleation_data = {}

for coup in couplings:
    df = plot_data[coup]
    T_arr = df["T"].values
    log_Gamma = df["log_Gamma"].values

    # Fit rev function
    try:
        p0 = [0, 0, 0, 0, 0, 1e-4]
        popt, _ = curve_fit(rev, T_arr, log_Gamma, maxfev=10000, p0=p0)
    except Exception:
        print(f"  λ={coup:.4f}: Fitting failed, skipping")
        continue

    # First pass: find T range where Γ/H⁴ is in interesting region
    # Use wide range to find bounds
    t_wide = np.linspace(T_arr.min() * 0.9, T_arr.max() * 1.1, 500)
    Gamma_H4_wide = np.exp(rev(t_wide, *popt)) / (Hubble(t_wide) ** 4)

    # t_wide is sorted low T -> high T
    # Γ/H⁴ is HIGH at low T, LOW at high T

    # Find T_low: lowest T where Γ/H⁴ just reaches 10⁴
    # (extend low enough to see Γ/H⁴ >= 10⁴)
    idx_high = np.where(Gamma_H4_wide >= 1e4)[0]
    if len(idx_high) > 0:
        T_low = t_wide[idx_high[-1]]  # Highest T where Γ/H⁴ >= 10⁴
    else:
        T_low = T_arr.min()

    # Find T_high: highest T where Γ/H⁴ is still >= 10⁻⁴
    # (cut off where Γ/H⁴ drops below Y_MIN to remove blank space)
    idx_visible = np.where(Gamma_H4_wide >= 1e-4)[0]
    if len(idx_visible) > 0:
        T_high = t_wide[idx_visible[-1]]  # Highest T where Γ/H⁴ >= 10⁻⁴
    else:
        T_high = T_arr.max()

    # Add buffer (8% on each side)
    T_range = T_high - T_low
    T_low = T_low - 0.08 * T_range
    T_high = T_high + 0.08 * T_range

    # Ensure valid range
    if T_low >= T_high:
        T_low = T_arr.min()
        T_high = T_arr.max()

    # Define T range for computation (focus on nucleation region)
    T_max_calc = T_high

    # Evaluation arrays within the interesting range
    t_eval = np.linspace(T_low, T_high, 300)

    # Compute Γ/H⁴
    Gamma_H4 = np.exp(rev(t_eval, *popt)) / (Hubble(t_eval) ** 4)

    # Compute n(T) and I(T) for selected points (expensive)
    print(f"  λ={coup:.4f}: T range [{T_low/1000:.3f}, {T_high/1000:.3f}] TeV")
    print("    Computing n(T) and I(T)...")
    n_arr = []
    I_arr = []

    # Use fewer points for expensive integrals
    t_sparse = np.linspace(T_low, T_high, 80)
    for t in t_sparse:
        try:
            n_val = nT(t, popt, T_max_calc)
            I_val = percol(t, popt, T_max_calc)
        except Exception:
            n_val = np.nan
            I_val = np.nan
        n_arr.append(n_val)
        I_arr.append(I_val)

    n_arr = np.array(n_arr)
    I_arr = np.array(I_arr)
    P_arr = np.exp(-I_arr)

    # Fit log(n) and log(I) with rev function for smooth interpolation
    valid_n = ~np.isnan(n_arr) & (n_arr > 0)
    valid_I = ~np.isnan(I_arr) & (I_arr > 0)

    # Fit n(T): use log(n) vs T
    n_popt = None
    if np.sum(valid_n) > 6:
        try:
            log_n = np.log(n_arr[valid_n])
            p0 = [0, 0, 0, 0, 0, 1e-4]
            n_popt, _ = curve_fit(rev, t_sparse[valid_n], log_n, maxfev=10000, p0=p0)
        except Exception:
            n_popt = None

    # Fit I(T): use log(I) vs T
    I_popt = None
    if np.sum(valid_I) > 6:
        try:
            log_I = np.log(I_arr[valid_I])
            p0 = [0, 0, 0, 0, 0, 1e-4]
            I_popt, _ = curve_fit(rev, t_sparse[valid_I], log_I, maxfev=10000, p0=p0)
        except Exception:
            I_popt = None

    # Generate smooth interpolated curves
    if n_popt is not None:
        n_smooth = np.exp(rev(t_eval, *n_popt))
    else:
        n_smooth = None

    if I_popt is not None:
        I_smooth = np.exp(rev(t_eval, *I_popt))
        P_smooth = np.exp(-I_smooth)
    else:
        I_smooth = None
        P_smooth = None

    # Find T_n (where n(T) ~ 1)
    if np.any(valid_n):
        idx_Tn = np.argmin(np.abs(n_arr[valid_n] - 1))
        T_n = t_sparse[valid_n][idx_Tn]
    else:
        T_n = None

    # Find T_p (where P ~ 0.7)
    valid_P = ~np.isnan(P_arr) & (P_arr > 0)
    if np.any(valid_P):
        idx_Tp = np.argmin(np.abs(P_arr[valid_P] / 0.7 - 1))
        T_p = t_sparse[valid_P][idx_Tp]
    else:
        T_p = None

    nucleation_data[coup] = {
        "t_eval": t_eval,
        "Gamma_H4": Gamma_H4,
        "t_sparse": t_sparse,
        "n_arr": n_arr,
        "I_arr": I_arr,
        "P_arr": P_arr,
        "n_smooth": n_smooth,
        "I_smooth": I_smooth,
        "P_smooth": P_smooth,
        "T_n": T_n,
        "T_p": T_p,
        "T_low": T_low,
        "T_high": T_high,
        "popt": popt,
    }

    if T_n is not None and T_p is not None:
        print(f"    T_n = {T_n/1000:.3f} TeV, T_p = {T_p/1000:.3f} TeV")

# Determine common plot range (union of all individual ranges)
all_T_low = [nucleation_data[c]["T_low"] for c in nucleation_data]
all_T_high = [nucleation_data[c]["T_high"] for c in nucleation_data]
plot_T_min = min(all_T_low) if all_T_low else T_overlap_min
plot_T_max = max(all_T_high) if all_T_high else T_overlap_max
print(f"\nCommon plot range: [{plot_T_min/1000:.3f}, " f"{plot_T_max/1000:.3f}] TeV")

# Recompute curves for the common T range (so all curves extend fully)
t_common = np.linspace(plot_T_min, plot_T_max, 400)
for coup in list(nucleation_data.keys()):
    data = nucleation_data[coup]
    popt = data["popt"]

    # Recompute Γ/H⁴ for full common range
    Gamma_H4_full = np.exp(rev(t_common, *popt)) / (Hubble(t_common) ** 4)
    data["t_common"] = t_common
    data["Gamma_H4_full"] = Gamma_H4_full

    # Recompute smooth n, I, P for full range if fitting succeeded
    if data.get("n_smooth") is not None:
        # Get n_popt by refitting (we didn't store it)
        valid_n = ~np.isnan(data["n_arr"]) & (data["n_arr"] > 0)
        if np.sum(valid_n) > 6:
            try:
                log_n = np.log(data["n_arr"][valid_n])
                p0 = [0, 0, 0, 0, 0, 1e-4]
                n_popt, _ = curve_fit(
                    rev, data["t_sparse"][valid_n], log_n, maxfev=10000, p0=p0
                )
                data["n_smooth_full"] = np.exp(rev(t_common, *n_popt))
            except Exception:
                data["n_smooth_full"] = None
        else:
            data["n_smooth_full"] = None
    else:
        data["n_smooth_full"] = None

    if data.get("I_smooth") is not None:
        valid_I = ~np.isnan(data["I_arr"]) & (data["I_arr"] > 0)
        if np.sum(valid_I) > 6:
            try:
                log_I = np.log(data["I_arr"][valid_I])
                p0 = [0, 0, 0, 0, 0, 1e-4]
                I_popt, _ = curve_fit(
                    rev, data["t_sparse"][valid_I], log_I, maxfev=10000, p0=p0
                )
                data["I_smooth_full"] = np.exp(rev(t_common, *I_popt))
                data["P_smooth_full"] = np.exp(-data["I_smooth_full"])
            except Exception:
                data["I_smooth_full"] = None
                data["P_smooth_full"] = None
        else:
            data["I_smooth_full"] = None
            data["P_smooth_full"] = None
    else:
        data["I_smooth_full"] = None
        data["P_smooth_full"] = None

# Create 2x2 subplot figure
print("\nCreating nucleation & percolation plots...")
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 12))

# Y-axis range (log scale): 10⁻⁴ to 10⁴
Y_MIN = 1e-4
Y_MAX = 1e4

# X-axis range (TeV)
X_MIN = plot_T_min / 1000
X_MAX = plot_T_max / 1000

# Plot n(T) - use fitted smooth curve for full range
ax_n = axes6[0, 0]
for coup in couplings:
    if coup not in nucleation_data:
        continue
    data = nucleation_data[coup]
    style = get_plot_style(coup)
    style["linewidth"] = style["linewidth"] * 0.8
    # Use full-range smooth curve if available
    if data.get("n_smooth_full") is not None:
        n_s = data["n_smooth_full"]
        valid = (n_s > 0) & (n_s >= Y_MIN * 0.1)
        ax_n.plot(data["t_common"][valid] / 1000, n_s[valid], **style)
    elif data["n_smooth"] is not None:
        n_s = data["n_smooth"]
        valid = (n_s > 0) & (n_s >= Y_MIN * 0.1)
        ax_n.plot(data["t_eval"][valid] / 1000, n_s[valid], **style)
    else:
        n_raw = data["n_arr"]
        valid = ~np.isnan(n_raw) & (n_raw > 0)
        ax_n.plot(data["t_sparse"][valid] / 1000, n_raw[valid], **style)
ax_n.axhline(1, linestyle="--", color="black", linewidth=1, alpha=0.7)
ax_n.set_xlabel("T (TeV)", fontsize=11)
ax_n.set_ylabel(r"$n(T)$", fontsize=12)
ax_n.set_title(r"Number of Bubbles per Hubble Volume $n(T)$", fontsize=12)
ax_n.set_yscale("log")
ax_n.set_ylim([Y_MIN, Y_MAX])
ax_n.set_xlim([X_MIN, X_MAX])
ax_n.grid(True, alpha=0.3)
ax_n.legend(fontsize=6, loc="best", ncol=2)

# Plot Γ/H⁴ - use full common range
ax_G = axes6[0, 1]
for coup in couplings:
    if coup not in nucleation_data:
        continue
    data = nucleation_data[coup]
    style = get_plot_style(coup)
    style["linewidth"] = style["linewidth"] * 0.8
    # Use full-range Gamma_H4
    G_H4 = data["Gamma_H4_full"]
    valid = (G_H4 > 0) & (G_H4 >= Y_MIN * 0.1)
    ax_G.plot(data["t_common"][valid] / 1000, G_H4[valid], **style)
ax_G.set_xlabel("T (TeV)", fontsize=11)
ax_G.set_ylabel(r"$\Gamma/H^4$", fontsize=12)
ax_G.set_title(r"Nucleation Rate $\Gamma(T)/H^4$", fontsize=12)
ax_G.set_yscale("log")
ax_G.set_ylim([Y_MIN, Y_MAX])
ax_G.set_xlim([X_MIN, X_MAX])
ax_G.grid(True, alpha=0.3)
ax_G.legend(fontsize=6, loc="best", ncol=2)

# Plot I/0.34 - use fitted smooth curve for full range
ax_I = axes6[1, 0]
for coup in couplings:
    if coup not in nucleation_data:
        continue
    data = nucleation_data[coup]
    style = get_plot_style(coup)
    style["linewidth"] = style["linewidth"] * 0.8
    # Use full-range smooth curve if available
    if data.get("I_smooth_full") is not None:
        I_norm = data["I_smooth_full"] / 0.34
        valid = (I_norm > 0) & (I_norm >= Y_MIN * 0.1)
        ax_I.plot(data["t_common"][valid] / 1000, I_norm[valid], **style)
    elif data["I_smooth"] is not None:
        I_norm = data["I_smooth"] / 0.34
        valid = (I_norm > 0) & (I_norm >= Y_MIN * 0.1)
        ax_I.plot(data["t_eval"][valid] / 1000, I_norm[valid], **style)
    else:
        valid = ~np.isnan(data["I_arr"]) & (data["I_arr"] > 0)
        t_plot = data["t_sparse"][valid] / 1000
        I_plot = data["I_arr"][valid] / 0.34
        ax_I.plot(t_plot, I_plot, **style)
ax_I.axhline(1, linestyle="--", color="black", linewidth=1, alpha=0.7)
ax_I.set_xlabel("T (TeV)", fontsize=11)
ax_I.set_ylabel(r"$I/0.34$", fontsize=12)
ax_I.set_title(r"Percolation Integral $I(T)/0.34$", fontsize=12)
ax_I.set_yscale("log")
ax_I.set_ylim([Y_MIN, Y_MAX])
ax_I.set_xlim([X_MIN, X_MAX])
ax_I.grid(True, alpha=0.3)
ax_I.legend(fontsize=6, loc="best", ncol=2)

# Plot P/0.7 - use fitted smooth curve for full range
ax_P = axes6[1, 1]
for coup in couplings:
    if coup not in nucleation_data:
        continue
    data = nucleation_data[coup]
    style = get_plot_style(coup)
    style["linewidth"] = style["linewidth"] * 0.8
    # Use full-range smooth curve if available
    if data.get("P_smooth_full") is not None:
        P_norm = data["P_smooth_full"] / 0.7
        valid = (P_norm > 0) & (P_norm >= Y_MIN * 0.1)
        ax_P.plot(data["t_common"][valid] / 1000, P_norm[valid], **style)
    elif data["P_smooth"] is not None:
        P_norm = data["P_smooth"] / 0.7
        valid = (P_norm > 0) & (P_norm >= Y_MIN * 0.1)
        ax_P.plot(data["t_eval"][valid] / 1000, P_norm[valid], **style)
    else:
        valid = ~np.isnan(data["P_arr"]) & (data["P_arr"] > 0)
        t_plot = data["t_sparse"][valid] / 1000
        P_plot = data["P_arr"][valid] / 0.7
        ax_P.plot(t_plot, P_plot, **style)
ax_P.axhline(1, linestyle="--", color="black", linewidth=1, alpha=0.7)
ax_P.set_xlabel("T (TeV)", fontsize=11)
ax_P.set_ylabel(r"$P/0.7$", fontsize=12)
ax_P.set_title(r"Survival Probability $P(T)/0.7$", fontsize=12)
ax_P.set_yscale("log")
ax_P.set_ylim([Y_MIN, Y_MAX])
ax_P.set_xlim([X_MIN, X_MAX])
ax_P.grid(True, alpha=0.3)
ax_P.legend(fontsize=6, loc="best", ncol=2)

plt.suptitle(
    rf"Nucleation & Percolation ($\lambda$ = {COUPLING_MIN:.2f}"
    rf" to {COUPLING_MAX:.2f})",
    fontsize=14,
    y=1.02,
)
plt.tight_layout()
plt.savefig(f"{output_dir}/nucleation_percolation.png", dpi=200)
print(f"Saved: {output_dir}/nucleation_percolation.png")

plt.show()

print("\n" + "=" * 70)
print("PLOTTING COMPLETE")
print("=" * 70)
