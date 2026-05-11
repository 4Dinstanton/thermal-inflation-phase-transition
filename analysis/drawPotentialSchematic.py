#!/usr/bin/env python3
"""
Publication-quality SCHEMATIC of the flaton thermal effective potential.

Single wide panel with enlarged inset zoom.  Schematic (not computed)
because of the extreme scale separation (T_c ~ 10^3 vs phi_0 ~ 10^11 GeV).

Physics:
  - Tree-level: V_0 - (1/2)m^2 phi^2 + (1/4)lambda phi^4
  - Thermal correction adds +c T^2 phi^2 → valley at phi=0 when c T^2 > m^2,
    creating a barrier at phi_b before the potential rolls down to phi_0.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import argparse, os

_p = argparse.ArgumentParser()
_p.add_argument("--nf", type=int, default=20)
_p.add_argument("--nb", type=int, default=20)
args = _p.parse_args()

plt.rcParams.update(
    {
        "font.size": 13,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    }
)

# ── Schematic curves  (x = phi / phi_0) ──
x = np.linspace(0, 1.12, 6000)

V_tree = (1.0 - x**2) ** 2


# Thermal correction LOWERS V(0): the false vacuum energy drops,
# creating a valley at the origin.  The barrier stays near the
# tree-level height while the zero-point descends.
def thermal_dip(x, B, sigma):
    return B * np.exp(-0.5 * (x / sigma) ** 2)


V_Tc = V_tree - thermal_dip(x, B=0.03, sigma=0.07)
V_hot = V_tree - thermal_dip(x, B=0.3, sigma=0.10)

# ── Main figure ──
fig, ax = plt.subplots(figsize=(12, 4.2))

ax.plot(x, V_hot, color="#d62728", lw=2.0, zorder=2)
ax.plot(x, V_Tc, color="#ff7f0e", lw=2.0, zorder=3)
ax.plot(x, V_tree, color="#4a90d9", lw=2.0, zorder=4)

# ── Axes: both as arrows, intersecting at origin ──
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# x-axis arrow
ax.annotate(
    "",
    xy=(1.15, 0),
    xytext=(-0.01, 0),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
)
ax.text(1.16, -0.008, r"$\phi$", fontsize=17, ha="left", va="center")

# y-axis arrow
ax.annotate(
    "",
    xy=(0, 1.28),
    xytext=(0, -0.06),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
)
ax.text(-0.02, 1.30, r"$V(\phi)$", fontsize=16, ha="center", va="bottom")

# phi_0 marker — lowered
ax.plot(1.0, 0.0, "o", color="#4a90d9", ms=5, zorder=6)
ax.text(1.005, -0.04, r"$\phi_0$", fontsize=14, ha="center", va="top", color="#4a90d9")

ax.set_xlim(-0.04, 1.22)
ax.set_ylim(-0.10, 1.35)

# ═══════════════════════════════════════════════════════════════
#  INSET: enlarged zoom into barrier region
#  Shows V(phi) - V(0) to reveal valley + barrier shape
# ═══════════════════════════════════════════════════════════════
ax_in = fig.add_axes([0.32, 0.34, 0.42, 0.55])

x_in = np.linspace(0, 0.35, 1200)

for V_full, color in [(V_hot, "#d62728"), (V_Tc, "#ff7f0e"), (V_tree, "#4a90d9")]:
    V_shifted = np.interp(x_in, x, V_full) - 1.0
    ax_in.plot(x_in, V_shifted, color=color, lw=1.8)

ax_in.set_xticks([])
ax_in.set_yticks([])
for s in ax_in.spines.values():
    s.set_edgecolor("#999999")
    s.set_linewidth(0.6)
ax_in.patch.set_facecolor("white")
ax_in.patch.set_alpha(0.97)

# Precompute shifted curves for label placement
V_hot_in = np.interp(x_in, x, V_hot) - 1.0
V_Tc_in = np.interp(x_in, x, V_Tc) - 1.0
V_T0_in = np.interp(x_in, x, V_tree) - 1.0
i_pk_hot = np.argmax(V_hot_in)
i_pk_Tc = np.argmax(V_Tc_in)

# Temperature labels without arrows: stacked cleanly
# Blue label: ABOVE the blue curve, moved higher
blue_x = 0.12
blue_y = np.interp(blue_x, x_in, V_T0_in)
ax_in.text(
    blue_x,
    blue_y + 0.015,
    r"$T = 0$",
    fontsize=12,
    color="#4a90d9",
    va="bottom",
    ha="center",
)

# Orange label: IN THE MIDDLE of Orange and Red curves, moved left
orange_x = 0.12
orange_y = np.interp(orange_x, x_in, V_Tc_in)
red_y_at_orange = np.interp(orange_x, x_in, V_hot_in)
mid_y = (orange_y + red_y_at_orange) / 2.0
ax_in.text(
    orange_x,
    orange_y - 0.03,
    r"$T \gtrsim T_{c_2}$",
    fontsize=12,
    color="#ff7f0e",
    va="center",
    ha="center",
)

# Red label: BELOW the red curve, moved left
red_x = 0.12
red_y = np.interp(red_x, x_in, V_hot_in)
ax_in.text(
    red_x,
    red_y - 0.04,
    r"$T \gg T_{c_2}$",
    fontsize=12,
    color="#d62728",
    va="top",
    ha="center",
)

# Dashed box on main axes — fitted to encompass all three curves near origin
bx0, bx1 = -0.008, 0.3
by0 = 0.67
by1 = 1.05
rect = plt.Rectangle(
    (bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor="#999999", ls="--", lw=0.6
)
ax.add_patch(rect)

# Connection lines
con1 = ConnectionPatch(
    xyA=(bx1, by1),
    coordsA=ax.transData,
    xyB=(0.0, 1.0),
    coordsB=ax_in.transAxes,
    color="#999999",
    ls="--",
    lw=0.6,
)
con2 = ConnectionPatch(
    xyA=(bx1, by0),
    coordsA=ax.transData,
    xyB=(0.0, 0.0),
    coordsB=ax_in.transAxes,
    color="#999999",
    ls="--",
    lw=0.6,
)
fig.add_artist(con1)
fig.add_artist(con2)

fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.96)

os.makedirs("figs/potential", exist_ok=True)
out = f"figs/potential/schematic_minimal_nf{args.nf}_nb{args.nb}"
fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print(f"Saved: {out}.pdf / .png")
plt.close(fig)
