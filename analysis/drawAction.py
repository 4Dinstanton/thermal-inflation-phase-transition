import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit


def format_e(n):
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "E" + a.split("E")[1]


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def quartic(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def rev(x, a, b, c, d, e, f):
    # return a * x**2 + b * x + c
    return a + b * x + c * x**2 + d * x**3 + e * np.exp(-f * x)


def HubbleSquare(T, delV):
    MPL = 2.4e18
    chig2 = 30 / (math.pi**2 * 106.75)
    return (T**4 / chig2 + delV) / (3 * MPL**2)


def Hubble(T, delV):
    MPL = 2.4e18
    chig2 = 30 / (math.pi**2 * 106.75)
    if isinstance(T, np.ndarray):
        Hub2 = (T**4 / chig2 + delV) / (3 * MPL**2)
        return np.sqrt(Hub2.astype("float"))
    else:
        return np.sqrt((T**4 / chig2 + delV) / (3 * MPL**2))


MPL = 2.4e18

PANEL_CONFIGS = [
    {
        "param_set": "set9",
        "potential_flag": "boson_and_fermion",
        "gamma": 4.1667e-8,
        "title": r"Set A: $V_0 = 2.5\times10^{27}$, $n_b{=}n_f{=}20$",
    },
    {
        "param_set": "set10",
        "potential_flag": "boson_and_fermion",
        "gamma": 4.1667e-4,
        "title": r"Set B: $V_0 = 2.5\times10^{35}$, $n_b{=}n_f{=}20$",
    },
    {
        "param_set": "set11",
        "potential_flag": "boson_and_Fermion",
        "gamma": 4.1667e-4,
        "title": r"Set C: $V_0 = 2.5\times10^{35}$, $n_b{=}0$, $n_f{=}20$",
    },
]

LW = 2.2
FONTSIZE_LABEL = 16
FONTSIZE_TICK = 13
FONTSIZE_LEGEND = 12
FONTSIZE_ANNOT = 14


def compute_panel(cfg):
    """Load data, fit, and compute nucleation/percolation arrays for one panel."""
    csv_path = f"data/tunneling/{cfg['param_set']}"
    df = pd.read_csv(
        f"{csv_path}/T-S_param_{cfg['param_set']}_lambdaSix_0E+00_{cfg['potential_flag']}.csv"
    ).iloc[1:]
    gamma = cfg["gamma"]
    delV = gamma**2 * 1000**2 * MPL**2 / 4

    G = (-df["S3/T"]) + 4 * np.log(df["T"]) + 3 * np.log(df["S3/T"] / (2 * math.pi)) / 2
    popt, _ = curve_fit(rev, df["T"].values, G.values)

    T_c = 15000
    H_ = Hubble(200_000, delV)

    def nT_local(T):
        def f(y):
            return np.exp(rev(y, *popt)) / (H_**4 * y)

        return quad(f, T, T_c)[0]

    def percol_local(T):
        prefactor = 4 * math.pi / 3

        def outer_integrand(Tp):
            return np.exp(rev(Tp, *popt) - 4 * np.log(H_)) * (Tp / T - 1) ** 3 / T

        return prefactor * quad(outer_integrand, T, T_c)[0]

    t_coarse = np.linspace(df["T"].min(), df["T"].max() + 5000, 5000)
    log_gh4 = rev(t_coarse, *popt) - 4 * np.log(Hubble(t_coarse, delV))
    cross_idx = np.where(np.diff(np.sign(log_gh4)))[0]
    T_n_approx = (
        0.5 * (t_coarse[cross_idx[-1]] + t_coarse[cross_idx[-1] + 1])
        if len(cross_idx)
        else 0.5 * (df["T"].min() + df["T"].max())
    )
    print(f"  [{cfg['param_set']}/{cfg['potential_flag']}] T_n ≈ {T_n_approx:.2f} GeV")

    HALF_WIDTH, N_FINE = 200.0, 500
    t_arr = np.linspace(T_n_approx - HALF_WIDTH, T_n_approx + HALF_WIDTH, N_FINE)
    nt_arr = np.array([nT_local(t) for t in t_arr])
    perc_arr = np.array([percol_local(t) for t in t_arr])

    YLIM_LO, YLIM_HI = 1e-4, 1e4
    gh4 = np.exp(rev(t_arr, *popt)) / (Hubble(t_arr, delV) ** 4)
    I034 = perc_arr / 0.34
    right_mask = np.where(gh4 < YLIM_LO)[0]
    idx_right = right_mask[0] if len(right_mask) else N_FINE - 1
    left_mask = np.where(I034 > YLIM_HI)[0]
    idx_left = left_mask[-1] if len(left_mask) else 0
    margin = max(3, int(0.05 * (idx_right - idx_left)))
    idx_left = max(0, idx_left - margin)
    idx_right = min(N_FINE - 1, idx_right + margin)
    sl = slice(idx_left, idx_right + 1)

    return {
        "t": t_arr[sl],
        "nt": nt_arr[sl],
        "perc": perc_arr[sl],
        "popt": popt,
        "delV": delV,
    }


import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 4, hspace=0.30, wspace=0.35)
ax_positions = [
    gs[0, 0:2],  # top-left  → Set A
    gs[0, 2:4],  # top-right → Set B
    gs[1, 1:3],  # bottom-center → Set C
]

for idx, (cfg, gs_pos) in enumerate(zip(PANEL_CONFIGS, ax_positions)):
    ax = fig.add_subplot(gs_pos)
    print(f"\nPanel {idx}: {cfg['title']}")
    d = compute_panel(cfg)
    t, nt, perc, popt, delV = d["t"], d["nt"], d["perc"], d["popt"], d["delV"]
    x = t / 1000.0

    l1 = ax.plot(x, nt, color="red", lw=LW, label=r"$n(T)$")
    l2 = ax.plot(
        x,
        np.exp(rev(t, *popt)) / (Hubble(t, delV) ** 4),
        color="blue",
        lw=LW,
        label=r"$\Gamma/H^4$",
    )
    l3 = ax.plot(x, perc / 0.34, color="green", lw=LW, label=r"$I\,/\,0.34$")
    l4 = ax.plot(x, np.exp(-perc) / 0.7, color="orange", lw=LW, label=r"$P\,/\,0.7$")
    ax.axhline(1, ls="--", color="black", lw=1.2)

    T_n = t[np.argmin(np.abs(nt - 1))] / 1000
    T_p = t[np.argmin(np.abs(np.exp(-perc) / 0.7 - 1))] / 1000
    T_c1 = t[np.argmin(np.abs(np.exp(-perc) / 0.7 - 1e-5))] / 1000
    print(f"  T_n = {T_n:.4f} TeV,  T_p = {T_p:.4f} TeV,  T_c1 = {T_c1:.4f} TeV")

    ax.axvline(T_n, ls="--", color="red", lw=1.4)
    ax.text(
        T_n + 0.001,
        15,
        r"$T_n$",
        color="red",
        fontsize=FONTSIZE_ANNOT,
        fontweight="bold",
    )
    ax.axvline(T_p, ls="--", color="orange", lw=1.4)
    ax.text(
        T_p + 0.001,
        15,
        r"$T_p$",
        color="orange",
        fontsize=FONTSIZE_ANNOT,
        fontweight="bold",
    )
    ax.axvline(T_c1, ls="--", color="black", lw=1.4)
    ax.text(
        T_c1 - 0.004,
        15,
        r"$T_{c_1}$",
        color="black",
        fontsize=FONTSIZE_ANNOT,
        fontweight="bold",
    )

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([1e-4, 1e4])
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [TeV]", fontsize=FONTSIZE_LABEL)
    ax.set_title(cfg["title"], fontsize=FONTSIZE_LABEL - 1, pad=10)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK, width=1.2, length=5)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=FONTSIZE_LEGEND, loc="upper right")

os.makedirs("figs/action", exist_ok=True)
out_path = "figs/action/action_tripanel_ABC_2.png"
fig.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
