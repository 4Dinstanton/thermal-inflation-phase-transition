from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from published_plisc import (
    DETECTORS,
    LS_TURBULENCE,
    load_all as load_published_plisc,
    place_detector_label,
)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

m_TeV = 1.0
Mpl_TeV = 2.4e15
Td_TeV = 0.1
gstar_d = 100.0
gstars_d = 100.0
gstars_0 = 3.91
T0_K = 2.7255
kB_eV_per_K = 8.617333262e-5
T0_TeV = T0_K * kB_eV_per_K / 1e12
beta_over_H = 1000.0
vw = 1.0
Uf = math.sqrt(3.0 / 4.0)
epsilon_turb = 0.05
kappa_v = 1.0
hbar_eV_s = 6.582119569e-16
TeV_to_sinv = 1e12 / hbar_eV_s
gammas = [1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3]
frequencies = np.logspace(-5, 5, 2200)


def S_sw(x):
    return x**3 * (7.0 / (4.0 + 3.0 * x**2))**3.5

def S_turb(f, f_turb, f_H):
    x = f / f_turb
    return x**3 / ((1.0 + x)**(11.0 / 3.0) * (1.0 + 8.0 * math.pi * f / f_H))

def S_phi(x):
    return 3.8 * x**2.8 / (1.0 + 2.8 * x**3.8)

def model_quantities(gamma):
    V04 = math.sqrt(gamma * m_TeV * Mpl_TeV / 2.0)
    V0 = V04**4
    Rmd = (math.pi**2 * gstar_d * Td_TeV**4 / (30.0 * V0))**(1.0 / 3.0)
    Hstar_TeV = gamma * m_TeV / (2.0 * math.sqrt(3.0))
    a_d_over_a0 = (gstars_0 / gstars_d)**(1.0 / 3.0) * T0_TeV / Td_TeV
    f_H0 = Hstar_TeV * TeV_to_sinv * Rmd * a_d_over_a0
    f_sw = 1.16 * beta_over_H * f_H0 / vw
    f_turb = 1.64 * beta_over_H * f_H0 / vw
    f_phi = (0.62 / (1.8 - 0.1 * vw + vw**2)) * beta_over_H * f_H0
    H_Rstar = (8.0 * math.pi)**(1.0 / 3.0) * vw / beta_over_H
    ups_sw = min(1.0, H_Rstar / Uf)
    sw_amp_opt = 8.21e-16 * (Td_TeV / 0.1)**(4.0 / 3.0) * (V04 / 1.0e4)**(-4.0 / 3.0) * (beta_over_H / 1000.0)**(-1.0)
    sw_amp = sw_amp_opt * ups_sw
    turb_pref = 3.35e-4 * (1.0 / beta_over_H) * (epsilon_turb * kappa_v)**1.5 * (100.0 / gstar_d)**(1.0 / 3.0) * vw * Rmd
    phi_pref = 1.67e-5 * (1.0 / beta_over_H)**2 * (100.0 / gstar_d)**(1.0 / 3.0) * (0.11 * vw**3 / (0.42 + vw**2)) * Rmd
    return {"fH": f_H0, "fsw": f_sw, "fturb": f_turb, "fphi": f_phi, "sw_amp": sw_amp, "turb_pref": turb_pref, "phi_pref": phi_pref}

plisc = load_published_plisc()


def add_detector_plisc(ax):
    for name, (_, color, f_target) in DETECTORS.items():
        f, y = plisc[name]
        ax.loglog(f, y, color=color, linestyle="-.", linewidth=1.5, alpha=0.85)
        place_detector_label(ax, name, f, y, f_target)

def style_axes(ax, ylabel, title):
    ax.set_xlabel(r"$f\ \mathrm{[Hz]}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1e-5, 1e5)
    ax.set_ylim(1e-25, 1e-5)
    ax.grid(True, which="both", alpha=0.25)

def gamma_label(gamma):
    return rf"$\gamma=10^{{{int(round(math.log10(gamma)))}}}$"

# Detailed plasma-coupled figure
fig, ax = plt.subplots(figsize=(11.2, 7.2))
add_detector_plisc(ax)
ax.set_prop_cycle(None)
totals, sws, turbs = [], [], []
for gamma in gammas:
    q = model_quantities(gamma)
    sw = q["sw_amp"] * S_sw(frequencies / q["fsw"])
    turb = q["turb_pref"] * S_turb(frequencies, q["fturb"], q["fH"])
    total = sw + turb
    totals.append((gamma, total)); sws.append((gamma, sw)); turbs.append((gamma, turb))
    ax.loglog(frequencies, total, linewidth=2.15)
ax.set_prop_cycle(None)
for _, sw in sws: ax.loglog(frequencies, sw, linestyle="--", linewidth=1.0)
ax.set_prop_cycle(None)
for _, turb in turbs: ax.loglog(frequencies, turb, linestyle=":", linewidth=1.0)
for gamma, total in totals:
    idx = int(np.argmax(total))
    ax.annotate(gamma_label(gamma), xy=(frequencies[idx], total[idx]), xytext=(7, 6), textcoords="offset points", fontsize=8.5)
style_axes(ax, r"$h^2\Omega_{\mathrm{GW}}$", r"Plasma-coupled benchmark with published PLISC curves")
ax.legend(
    handles=[
        Line2D([0], [0], linewidth=2.15, label="Plasma total"),
        Line2D([0], [0], linestyle="--", linewidth=1.0, label="Sound wave"),
        Line2D([0], [0], linestyle=LS_TURBULENCE, linewidth=1.0, label="Turbulence"),
        Line2D([0], [0], linestyle="-.", linewidth=1.15, label="Published PLISC"),
    ],
    loc="lower left",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(FIG / "gw_plasma_coupled_published_PLISC.png", dpi=240, bbox_inches="tight")
fig.savefig(FIG / "gw_plasma_coupled_published_PLISC.pdf", bbox_inches="tight")

# Detailed collision upper-envelope figure
fig, ax = plt.subplots(figsize=(11.2, 7.2))
add_detector_plisc(ax)
ax.set_prop_cycle(None)
collisions = []
for gamma in gammas:
    q = model_quantities(gamma)
    phi = q["phi_pref"] * S_phi(frequencies / q["fphi"])
    collisions.append((gamma, phi))
    ax.loglog(frequencies, phi, linewidth=2.15)
for gamma, phi in collisions:
    idx = int(np.argmax(phi))
    ax.annotate(gamma_label(gamma), xy=(frequencies[idx], phi[idx]), xytext=(7, 6), textcoords="offset points", fontsize=8.5)
style_axes(ax, r"$h^2\Omega_{\phi}$", r"Weak-friction scalar-collision upper envelope with published PLISC curves")
ax.legend(handles=[Line2D([0],[0],linewidth=2.15,label=r"Scalar collision: $\kappa_\phi=1$"), Line2D([0],[0],linestyle="-.",linewidth=1.15,label="Published PLISC")], loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "gw_scalar_collision_upper_envelope_published_PLISC.png", dpi=240, bbox_inches="tight")
fig.savefig(FIG / "gw_scalar_collision_upper_envelope_published_PLISC.pdf", bbox_inches="tight")

# Combined publication figure
fig, ax = plt.subplots(figsize=(11.6, 7.4))
add_detector_plisc(ax)
ax.set_prop_cycle(None)
combined = []
for gamma in gammas:
    q = model_quantities(gamma)
    sw = q["sw_amp"] * S_sw(frequencies / q["fsw"])
    turb = q["turb_pref"] * S_turb(frequencies, q["fturb"], q["fH"])
    total = sw + turb
    combined.append((gamma, total))
    ax.loglog(frequencies, total, linewidth=2.2)
ax.set_prop_cycle(None)
for gamma in gammas:
    q = model_quantities(gamma)
    phi = q["phi_pref"] * S_phi(frequencies / q["fphi"])
    ax.loglog(frequencies, phi, linestyle="--", linewidth=1.55)
for gamma, total in combined:
    idx = int(np.argmax(total))
    ax.annotate(gamma_label(gamma), xy=(frequencies[idx], total[idx]), xytext=(7, 6), textcoords="offset points", fontsize=8.5)
style_axes(ax, r"$h^2\Omega_{\mathrm{GW}}$", r"Thermal-inflation GW benchmarks and published PLISC curves")
ax.legend(handles=[Line2D([0],[0],linewidth=2.2,label="Plasma-coupled benchmark"), Line2D([0],[0],linestyle="--",linewidth=1.55,label=r"Weak-friction collision upper envelope"), Line2D([0],[0],linestyle="-.",linewidth=1.15,label="Published PLISC")], loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "gw_combined_publication_published_PLISC.png", dpi=260, bbox_inches="tight")
fig.savefig(FIG / "gw_combined_publication_published_PLISC.pdf", bbox_inches="tight")
