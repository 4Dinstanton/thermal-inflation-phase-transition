"""
Overlay phi_esc locations on the finite-temperature potential V(phi).

phi_esc in the CSV is the actual field value at the escape point
(cosmoTransitions returns the profile in reversed order, so the CSV
already stores the corrected value).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))
import Potential as p

# ── parameters (must match drawPotential / scanCouplingTemp) ──────────────
potential_flag = "fermion_only"

lam = 1e-16
mphi = 1000
epsil = 0
lambdaSix = 0
bosonMassSquared = 1000000
bosonCoupling = 1.09
bosonGaugeCoupling = 1.05
fermionCoupling = 1.09
fermionGaugeCoupling = 1.05

param = {
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

# ── load CSV ──────────────────────────────────────────────────────────────
csv_path = "data/tunneling/set6/T-S_param_set6_lambdaSix_0E+00_fermion_only.csv"
df = pd.read_csv(csv_path)
T_arr = df["T"].values
phi_esc_arr = df["phi_esc"].values

# ── build potential ───────────────────────────────────────────────────────
VT = p.finiteTemperaturePotential(param)
VT.build_fast_thermal(x_max=150.0, n_pts=4096)

if potential_flag == "fermion_only":
    V_func = VT.V_p_fermion_only
else:
    V_func = VT.V_p_correct

# ── pick subset of temperatures to plot (avoid clutter) ───────────────────
n_total = len(T_arr)
if n_total <= 8:
    idx_sel = np.arange(n_total)
else:
    idx_sel = np.linspace(0, n_total - 1, 8, dtype=int)

cmap = plt.cm.coolwarm
colors = [cmap(i / max(len(idx_sel) - 1, 1)) for i in range(len(idx_sel))]

# ── figure 1: full potential with phi_esc marked ──────────────────────────
phi_esc_max = phi_esc_arr.max()
phi_range = np.linspace(-0.05 * phi_esc_max, 1.5 * phi_esc_max, 1000).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(10, 7))

for ci, idx in enumerate(idx_sel):
    TEMP = T_arr[idx]
    pe = phi_esc_arr[idx]

    VT.update_T(TEMP)

    V_vals = V_func(phi_range)
    V_at_zero = V_func(np.zeros((1, 1))).item()
    V_shifted = V_vals - V_at_zero

    col = colors[ci]
    ax.plot(phi_range[:, 0] / 1e3, V_shifted, color=col, lw=1.5,
            label=f"T={TEMP:.0f}")

    pe_arr = np.array([[pe]])
    V_at_esc = V_func(pe_arr).item() - V_at_zero
    ax.plot(pe / 1e3, V_at_esc, "o", color=col,
            markersize=8, markeredgecolor="k", markeredgewidth=0.8, zorder=5)

ax.set_xlabel(r"$\phi$  (TeV)", fontsize=12)
ax.set_ylabel(r"$V(\phi) - V(0)$", fontsize=12)
ax.set_title(
    r"Potential with escape point $\phi_{\mathrm{esc}}$"
    f"  ({potential_flag})",
    fontsize=12,
)
ax.legend(fontsize=9, title=r"$\bigcirc$ = $\phi_{\mathrm{esc}}$")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/finiteTemp/phi_esc_on_potential_fermion_only_set6.png", dpi=200)
plt.close(fig)

# ── figure 2: zoomed near phi_esc ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7))

for ci, idx in enumerate(idx_sel):
    TEMP = T_arr[idx]
    pe = phi_esc_arr[idx]

    VT.update_T(TEMP)

    margin = max(0.3 * pe, 2000)
    phi_lo = pe - margin
    phi_hi = pe + margin
    phi_zoom = np.linspace(phi_lo, phi_hi, 600).reshape(-1, 1)
    V_vals = V_func(phi_zoom)
    V_at_zero = V_func(np.zeros((1, 1))).item()
    V_shifted = V_vals - V_at_zero

    col = colors[ci]
    ax2.plot(phi_zoom[:, 0] / 1e3, V_shifted, color=col, lw=1.5,
             label=f"T={TEMP:.0f}")

    pe_arr = np.array([[pe]])
    V_at_esc = V_func(pe_arr).item() - V_at_zero
    ax2.plot(pe / 1e3, V_at_esc, "o", color=col,
             markersize=8, markeredgecolor="k", markeredgewidth=0.8, zorder=5)
    ax2.axvline(pe / 1e3, color=col, ls=":", lw=0.7, alpha=0.5)

ax2.set_xlabel(r"$\phi$  (TeV)", fontsize=12)
ax2.set_ylabel(r"$V(\phi) - V(0)$", fontsize=12)
ax2.set_title(r"Zoomed near $\phi_{\mathrm{esc}}$", fontsize=12)
ax2.legend(fontsize=9, title=r"$\bigcirc$ = $\phi_{\mathrm{esc}}$, dotted = $\phi_{\mathrm{esc}}$ line")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figs/finiteTemp/phi_esc_on_potential_zoom_fermion_only_set6.png", dpi=200)
plt.close(fig2)

print("Plots saved:")
print("  figs/finiteTemp/phi_esc_on_potential_fermion_only_set6.png")
print("  figs/finiteTemp/phi_esc_on_potential_zoom_fermion_only_set6.png")
