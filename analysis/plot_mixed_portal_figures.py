#!/usr/bin/env python3
"""Potential + nested-bubble-limit figures for the mixed portal potential.

Paper-style figures (Oct 2026 mixed draft):
  * FIG.1 / FIG.2 — finite-T surface & contour of V(φ,S) (unmixed vs μ²_mix>0)
  * FIG.3 — ΔT_n contour / nested window in (m_S, κ) and (ε, φ_off)

Examples
--------
    # potential panels only (fast)
    PYTHONPATH=. python analysis/plot_mixed_portal_figures.py --pot_only \\
        --out data/portal/mixed_figures

    # potential + nested-limit scan
    PYTHONPATH=. python analysis/plot_mixed_portal_figures.py \\
        --out data/portal/mixed_figures --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.portalBounce import (  # noqa: E402
    SET8_BOSON_PHI,
    SET8_FERMION_PHI,
    SET8_V,
    SET8_V0,
)
from analysis.portalEpsScan import (  # noqa: E402
    DEFAULT_T_REH,
    _Tee,
    build_pot,
    log,
    run_one,
    s0_from_mS,
)
from analysis.run_mixed_portal_scan import mu2_for_phi_off  # noqa: E402


# --------------------------------------------------------------------------
# Potential figures (FIG.1 / FIG.2 style)
# --------------------------------------------------------------------------
def _make_pot(eps, m_S, kappa, phi_off_tgt, V0=SET8_V0, v=SET8_V):
    s0 = s0_from_mS(eps, V0, m_S)
    mu2 = 0.0
    if phi_off_tgt and phi_off_tgt > 0:
        mu2 = mu2_for_phi_off(
            V0, v, s0, eps, kappa,
            SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)],
            100.0, "set8", float(phi_off_tgt))
    pot = build_pot(
        V0, v, s0, eps, kappa,
        SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], 100.0,
        phi_thermal="set8", mu2_mix=mu2)
    return pot, mu2, s0


def plot_vacuum_landscape(out_dir: str, eps: float = 1e-20, m_S: float = 1000.0,
                          kappa: float = 0.3, T: float = 2000.0,
                          phi_off_tgt: float = 0.0) -> str:
    """Clear O / A(A′) / B landscape: 3D surface + 2D contour as subplots.

    One potential only (no μ comparison).  Parameters chosen so all three
    vacua coexist at the plotted T.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    os.makedirs(out_dir, exist_ok=True)
    pot, mu2, s0 = _make_pot(eps, m_S, kappa, phi_off_tgt)
    vac = pot.vacua(T)
    if vac.get("A") is None or vac.get("B") is None:
        raise RuntimeError(
            f"Need A and B at T={T}; got A={vac.get('A')}, B={vac.get('B')}. "
            "Try T nearer T0_S or smaller phi_off.")
    v = float(pot.v)
    V_B = float(vac["B"][2])
    V_A = float(vac["A"][2])
    V_O = float(vac["O"][2]) if vac.get("O") is not None else float(
        pot.V_rel(0.0, 0.0, T))
    E = float(pot.V0)
    u_A = (V_A - V_B) / E
    u_O = (V_O - V_B) / E

    # denser grid
    n = 300
    phi = np.linspace(0.0, 1.12 * v, n)
    s = np.linspace(0.0, 1.22 * s0, n)
    P, S = np.meshgrid(phi, s)
    U = (pot.V_rel(P, S, T) - V_B) / E
    # clip just above O so the O–A shelf and the drop into B stay sharp
    zmax = max(u_O * 1.02, 1.02)
    zmin = -0.02
    Uplot = np.clip(U, zmin, zmax)

    fig = plt.figure(figsize=(13.2, 5.6))
    # ---- 3D ----
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3.plot_surface(
        P / v, S / s0, Uplot, cmap=cm.coolwarm, linewidth=0,
        antialiased=True, rstride=2, cstride=2, alpha=0.93,
        vmin=zmin, vmax=zmax)

    def _Vnorm(phi_, S_):
        return float(np.clip((pot.V_rel(phi_, S_, T) - V_B) / E, zmin, zmax))

    if vac.get("O") is not None:
        ax3.scatter([0], [0], [_Vnorm(0.0, 0.0)], c="k", s=70, depthshade=False,
                    label=rf"$O$  ($U={u_O:.3f}$)", zorder=10)
    ax3.scatter([vac["A"][0] / v], [vac["A"][1] / s0],
                [_Vnorm(vac["A"][0], vac["A"][1])],
                c="C1", s=80, depthshade=False,
                label=(rf"$A'$  ($U={u_A:.3f}$)" if vac["A"][0] > 1
                       else rf"$A$  ($U={u_A:.3f}$)"), zorder=10)
    ax3.scatter([vac["B"][0] / v], [vac["B"][1] / s0], [0.0],
                c="C3", s=110, marker="*", depthshade=False,
                label=r"$B$  ($U=0$)", zorder=10)
    ax3.set_xlabel(r"$\phi/v$", labelpad=6)
    ax3.set_ylabel(r"$S/s_0$", labelpad=6)
    ax3.set_zlabel(r"$U=[V-V(B)]/V_0$", labelpad=6)
    ax3.set_title("3D surface")
    ax3.view_init(elev=26, azim=-60)
    ax3.legend(fontsize=9, loc="upper left")
    fig.colorbar(surf, ax=ax3, shrink=0.55, pad=0.08, label=r"$U$")

    # ---- 2D contour ----
    ax2 = fig.add_subplot(1, 2, 2)
    # densify levels between A and O so the intermediate shelf is visible
    lo = min(0.0, u_A - 0.02)
    mid = sorted(set([
        0.0, max(0.0, u_A - 0.01), u_A, 0.5 * (u_A + u_O),
        u_O - 0.01, u_O, zmax,
    ]))
    levels = np.unique(np.concatenate([
        np.linspace(lo, max(u_A, 0.05), 10),
        np.linspace(max(u_A, 0.05), u_O, 12),
        np.linspace(u_O, zmax, 4),
    ]))
    cf = ax2.contourf(P / v, S / s0, Uplot, levels=levels, cmap="RdBu_r",
                      extend="both")
    clev = [0.0, u_A, u_O]
    cs = ax2.contour(P / v, S / s0, U, levels=clev,
                     colors=["k", "C1", "0.25"],
                     linewidths=[2.0, 1.6, 1.2])
    fmt = {0.0: r"$V=V(B)$", u_A: r"$V=V(A)$", u_O: r"$V=V(O)$"}
    ax2.clabel(cs, fmt=fmt, fontsize=8)
    if vac.get("O") is not None:
        ax2.plot(0, 0, "ko", ms=9, label=r"$O$", zorder=5)
    ax2.plot(vac["A"][0] / v, vac["A"][1] / s0, "o", c="C1", ms=10,
             label=(r"$A'$" if vac["A"][0] > 1 else r"$A$"), zorder=5)
    ax2.plot(vac["B"][0] / v, vac["B"][1] / s0, "*", c="C3", ms=14,
             label=r"$B$", zorder=5)
    ax2.plot([0, vac["A"][0] / v, vac["B"][0] / v],
             [0, vac["A"][1] / s0, vac["B"][1] / s0],
             "k--", lw=1.0, alpha=0.55, label=r"$O\!\to\!A\!\to\!B$")
    ax2.set(xlabel=r"$\phi/v$", ylabel=r"$S/s_0$",
            title="2D contour", xlim=(0, 1.12), ylim=(0, 1.22))
    ax2.legend(fontsize=9, loc="upper right")
    fig.colorbar(cf, ax=ax2, fraction=0.046, pad=0.03, label=r"$U=[V-V(B)]/V_0$")

    fig.tight_layout()
    path = os.path.join(out_dir, "vacuum_landscape.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"wrote {path}")
    return path


def plot_saddle_approach(out_dir: str, eps: float = 1e-20, m_S: float = 1000.0,
                         kappa: float = 0.3, T: float = 2000.0,
                         phi_off_tgts=(0.0, 100.0, 200.0, 400.0, 600.0)) -> str:
    """Show the A→B saddle moving closer as the bilinear mix grows.

    Left: 2D zoom near A/A′ with V=V(A) contours for each μ.
    Right: 1D slice at fixed S=s_T — barrier tip / zero-crossing vs φ_off.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.cm.viridis
    tgts = [float(t) for t in phi_off_tgts]
    colors = {t: cmap(i / max(1, len(tgts) - 1)) for i, t in enumerate(tgts)}

    # collect pots / vacua
    data = []
    for tgt in tgts:
        pot, mu2, s0 = _make_pot(eps, m_S, kappa, tgt)
        vac = pot.vacua(T)
        if vac.get("A") is None:
            log(f"  skip phi_off={tgt}: no A′ at T={T}")
            continue
        data.append((tgt, pot, mu2, s0, vac))
    if len(data) < 2:
        raise RuntimeError("need ≥2 phi_off values with A′ at this T")

    s0_ref = data[0][3]
    # common zoom window: cover largest first-neg (φ in GeV internally; plot in TeV)
    phi_hi = 1.8e4
    GeV_to_TeV = 1e-3
    s_pad = 0.18
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # ---- left: contour zoom, overlay V=V(A) for each mix ----
    pot0, _, s0, vac0 = data[0][1], data[0][2], data[0][3], data[0][4]
    phi_A0, s_A0, V_A0 = vac0["A"]
    s_T_TeV = float(s_A0) * GeV_to_TeV
    phi = np.linspace(0.0, phi_hi, 320)
    s = np.linspace((1.0 - s_pad) * s_A0, (1.0 + s_pad) * s_A0, 260)
    P, S = np.meshgrid(phi, s)
    U0 = (pot0.V_rel(P, S, T) - V_A0) / max(T, 1.0) ** 4
    im = ax[0].pcolormesh(P * GeV_to_TeV, S / s0_ref, np.clip(U0, -2, 2),
                          cmap="RdBu_r", shading="auto", vmin=-2, vmax=2)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02,
                 label=r"$[V-V(A_{\mu=0})]/T^4$ (bg)")

    for tgt, pot, mu2, s0, vac in data:
        phi_A, s_A, V_A = vac["A"]
        U = pot.V_rel(P, S, T) - V_A
        col = colors[tgt]
        ax[0].contour(P * GeV_to_TeV, S / s0_ref, U, levels=[0.0], colors=[col],
                      linewidths=2.0)
        ax[0].plot(phi_A * GeV_to_TeV, s_A / s0_ref, "o", color=col, ms=7, zorder=5)

    ax[0].set(xlabel=r"$\phi$ [TeV]", ylabel=r"$S/s_0$",
              title=r"$V=V(A/A')$",
              xlim=(0, phi_hi * GeV_to_TeV),
              ylim=((1.0 - s_pad) * s_A0 / s0_ref, (1.0 + s_pad) * s_A0 / s0_ref))
    handles = [
        Line2D([0], [0], color=colors[t], lw=2.2,
               label=(r"$\mu=0$" if t <= 0
                      else (r"$\phi_{\rm off}^{\rm tgt}="
                            + f"{t * GeV_to_TeV:g}" + r"$ TeV")))
        for t, *_ in data
    ]
    ax[0].legend(handles=handles, fontsize=8, loc="upper right",
                 title=r"$V=V(A)$ contour")

    # ---- right: 1D slices at each A′'s s_T ----
    first_negs = []
    for tgt, pot, mu2, s0, vac in data:
        phi_A, s_A, V_A = vac["A"]
        p = np.linspace(0.0, phi_hi, 800)
        dV = (pot.V_rel(p, s_A, T) - V_A) / max(T, 1.0) ** 4
        col = colors[tgt]
        lab = (r"$\mu=0$" if tgt <= 0
               else (r"$\phi_{\rm off}=" + f"{tgt * GeV_to_TeV:g}" + r"$ TeV"))
        ax[1].plot(p * GeV_to_TeV, dV, color=col, lw=1.9, label=lab)
        ax[1].axvline(phi_A * GeV_to_TeV, color=col, ls=":", lw=1.0, alpha=0.8)
        neg = np.where(dV < 0)[0]
        if len(neg):
            i = int(neg[0])
            if i > 0:
                x0, x1 = p[i - 1], p[i]
                y0, y1 = dV[i - 1], dV[i]
                x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
            else:
                x_cross = p[i]
            first_negs.append((tgt, phi_A, x_cross))
            ax[1].plot(x_cross * GeV_to_TeV, 0.0, "v", color=col, ms=7)

    ax[1].axhline(0, c="0.4", lw=0.8)
    # format S value compactly (TeV)
    if s_T_TeV >= 10:
        s_label = rf"$S={s_T_TeV:.1f}$ TeV plane"
    else:
        s_label = rf"$S={s_T_TeV:.3g}$ TeV plane"
    ax[1].set(xlabel=r"$\phi$ [TeV]",
              ylabel=r"$[V(\phi,s_T)-V(A)]/T^4$",
              title=s_label,
              xlim=(0, phi_hi * GeV_to_TeV))
    ax[1].legend(fontsize=8, loc="lower left")
    if len(first_negs) >= 2:
        t0, a0, x0 = first_negs[0]
        t1, a1, x1 = first_negs[-1]
        ax[1].annotate(
            rf"$\Delta\phi_{{\rm sad}}$: "
            rf"${x0*GeV_to_TeV:.2f}\!\to\!{x1*GeV_to_TeV:.2f}$ TeV",
            xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.9))

    fig.tight_layout()
    path = os.path.join(out_dir, "saddle_approach_mixed.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    log(f"wrote {path}")

    if len(first_negs) >= 2:
        fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
        ts = [t * GeV_to_TeV for t, _, _ in first_negs]
        xs = [x * GeV_to_TeV for _, _, x in first_negs]
        phis = [a * GeV_to_TeV for _, a, _ in first_negs]
        ax2.plot(ts, xs, "o-", c="C3", lw=1.8, label=r"$\phi_{\rm sad}$ ($V=V(A)$)")
        ax2.plot(ts, phis, "s--", c="C1", lw=1.4, label=r"$\phi_{A/A'}$")
        ax2.set(xlabel=r"$\phi_{\rm off}^{\rm tgt}$ [TeV]",
                ylabel=r"$\phi$ [TeV]",
                title=r"Saddle location vs mixing")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        fig2.tight_layout()
        path2 = os.path.join(out_dir, "saddle_phi_vs_phi_off.png")
        fig2.savefig(path2, dpi=160)
        plt.close(fig2)
        log(f"wrote {path2}")
    return path


def plot_potential_panels(out_dir: str, eps: float = 1e-20, m_S: float = 1000.0,
                          kappa: float = 0.3, T: float = 2000.0,
                          phi_off_tgts=(0.0, 400.0)) -> None:
    """Surface + contour of V(φ,S) for unmixed and mixed (paper FIG.1–2).

    Default T=2 TeV (not 1.5) so mixed A′ still exists for φ_off∼400 GeV.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import SymLogNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    os.makedirs(out_dir, exist_ok=True)

    # ---- 2×2 contour: global (φ/v, S/s0) + zoom near A/A′ ----
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    for col, tgt in enumerate(phi_off_tgts):
        pot, mu2, s0 = _make_pot(eps, m_S, kappa, tgt)
        vac = pot.vacua(T)
        V_B = vac["B"][2] if vac["B"] else float(pot.V_rel(pot.v, 0.0, T))
        v = float(pot.v)

        # Global: both vacua on one panel
        ax = axes[0, col]
        phi_g = np.linspace(0.0, 1.15 * v, 200)
        s_g = np.linspace(0.0, 1.25 * s0, 200)
        Pg, Sg = np.meshgrid(phi_g, s_g)
        Ug = (pot.V_rel(Pg, Sg, T) - V_B) / pot.V0
        Udisp = np.clip(Ug, -1.2, 0.5)
        im = ax.pcolormesh(Pg / v, Sg / s0, Udisp, cmap="RdBu_r",
                           shading="auto", vmin=-1.2, vmax=0.5)
        ax.contour(Pg / v, Sg / s0, Ug, levels=[0.0], colors="k", linewidths=1.0)
        _mark_vacua(ax, vac, s0, v=v, mode="global")
        tag = (r"$\mu_{\rm mix}=0$" if tgt <= 0
               else rf"$\phi_{{\rm off}}^{{\rm tgt}}={tgt:g}$ GeV")
        ax.set(xlabel=r"$\phi/v$", ylabel=r"$S/s_0$",
               title=rf"Global  $T={T/1e3:g}$ TeV, {tag}",
               xlim=(0, 1.15), ylim=(0, 1.25))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label=r"$[V-V(B)]/V_0$ (clipped)")

        # Zoom near A / A′ (thermal tip, GeV)
        axz = axes[1, col]
        phi_A = float(vac["A"][0]) if vac["A"] else 0.0
        s_A = float(vac["A"][1]) if vac["A"] else s0
        phi_hi = max(phi_A + 1.2e4, 1.5e4)
        s_lo = max(0.0, 0.80 * s_A)
        s_hi_z = 1.20 * s_A
        phi_z = np.linspace(0.0, phi_hi, 260)
        s_z = np.linspace(s_lo, s_hi_z, 240)
        Pz, Sz = np.meshgrid(phi_z, s_z)
        if vac["A"] is not None:
            V_ref = vac["A"][2]
            ylab = r"$[V-V(A)]/(\varepsilon V_0)$"
        else:
            V_ref = float(pot.V_rel(0.0, s_A, T))
            ylab = r"$[V-V(0,s_T)]/(\varepsilon V_0)$"
        Uz = (pot.V_rel(Pz, Sz, T) - V_ref) / max(abs(pot.eps * pot.V0), 1.0)
        imz = axz.pcolormesh(
            Pz, Sz / s0, Uz, cmap="RdBu_r", shading="auto",
            norm=SymLogNorm(linthresh=1e-3, vmin=-1.0, vmax=1.0))
        axz.contour(Pz, Sz / s0, Uz, levels=[0.0], colors="k", linewidths=1.1)
        _mark_vacua(axz, vac, s0, v=v, mode="zoom")
        axz.set(xlabel=r"$\phi$ [GeV]", ylabel=r"$S/s_0$",
                title=rf"Near A/A′  ($\mu^2_{{\rm mix}}={mu2:.3g}$ GeV$^2$)",
                xlim=(0, phi_hi), ylim=(s_lo / s0, s_hi_z / s0))
        fig.colorbar(imz, ax=axz, fraction=0.046, pad=0.03, label=ylab)

    fig.suptitle(
        rf"Mixed portal potential  ($m_S={m_S:g}$ GeV, $\kappa={kappa:g}$, "
        rf"$\varepsilon={eps:.0e}$)",
        fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "potential_contours_mixed.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    log(f"wrote {path}")

    # ---- 3D surface near the thermal tip (φ in TeV of the tip, not v) ----
    fig = plt.figure(figsize=(13.5, 5.8))
    for i, tgt in enumerate(phi_off_tgts):
        pot, mu2, s0 = _make_pot(eps, m_S, kappa, tgt)
        vac = pot.vacua(T)
        V_A = vac["A"][2] if vac["A"] else float(pot.V_rel(0.0, s0, T))
        phi_A = float(vac["A"][0]) if vac["A"] else 0.0
        phi = np.linspace(0.0, max(phi_A + 1.5e4, 2.0e4), 100)
        s = np.linspace(0.7 * s0, 1.25 * s0, 100)
        P, S = np.meshgrid(phi, s)
        U = (pot.V_rel(P, S, T) - V_A) / max(abs(pot.eps * pot.V0), 1.0)
        U = np.clip(U, -2.0, 2.0)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(P / 1e3, S / s0, U, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True, rstride=1, cstride=1,
                        alpha=0.95)
        if vac.get("A") is not None:
            ax.scatter([vac["A"][0] / 1e3], [vac["A"][1] / s0], [0.0],
                       c="C1", s=50, depthshade=False, label=r"$A/A'$")
        if vac.get("O") is not None and vac["O"][1] >= 0.7 * s0:
            Vo = (float(pot.V_rel(0.0, vac["O"][1], T)) - V_A) / max(
                abs(pot.eps * pot.V0), 1.0)
            ax.scatter([0.0], [vac["O"][1] / s0], [np.clip(Vo, -2, 2)],
                       c="k", s=40, depthshade=False, label=r"$O$")
        tag = (r"$\mu_{\rm mix}=0$" if tgt <= 0
               else rf"$\phi_{{\rm off}}^{{\rm tgt}}={tgt:g}$ GeV")
        ax.set(xlabel=r"$\phi$ [TeV]", ylabel=r"$S/s_0$",
               zlabel=r"$[V-V(A)]/(\varepsilon V_0)$", title=tag)
        ax.legend(fontsize=8, loc="upper left")
        ax.view_init(elev=28, azim=-55)
    fig.suptitle(
        rf"Finite-$T$ portal potential near $A/A'$ at $T={T/1e3:g}$ TeV "
        rf"($m_S={m_S:g}$ GeV, $\kappa={kappa:g}$, $\varepsilon={eps:.0e}$)",
        fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "potential_surface_mixed.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    log(f"wrote {path}")

    # ---- 1D slices: S-axis and φ through A/A′ ----
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = {0.0: "k", 400.0: "C3", 100.0: "C0", 800.0: "C2"}
    for tgt in phi_off_tgts:
        pot, mu2, s0 = _make_pot(eps, m_S, kappa, tgt)
        vac = pot.vacua(T)
        col = colors.get(float(tgt), "C1")
        lab = (r"$\mu=0$" if tgt <= 0
               else rf"$\phi_{{\rm off}}^{{\rm tgt}}={tgt:g}$")
        s = np.linspace(0, 1.3 * s0, 600)
        # S-axis: for mixed, evaluate at φ=φ_off if A′ exists else φ=0
        phi_s = float(vac["A"][0]) if vac["A"] is not None else 0.0
        V0s = float(pot.V_rel(phi_s, 0.0, T))
        ax[0].plot(s / s0, (pot.V_rel(phi_s, s, T) - V0s) / (pot.eps * pot.V0),
                   color=col, lw=1.8, label=lab)
        if vac["A"] is not None:
            phi_A, s_A, V_A = vac["A"]
            p = np.linspace(0, max(2.0e4, phi_A + 1.2e4), 600)
            # thermal barrier ≪ ε V0 — show in T^4 units
            dV = pot.V_rel(p, s_A, T) - V_A
            ax[1].plot(p, dV / max(T, 1.0) ** 4, color=col, lw=1.8, label=lab)
            ax[1].axvline(phi_A, color=col, ls=":", lw=1.0)
    ax[0].axhline(0, c="0.5", lw=0.7)
    ax[0].axhline(-1, c="0.5", ls="--", lw=0.7)
    ax[0].set(xlabel=r"$S/s_0$",
              ylabel=r"$[V(\phi_*,S)-V(\phi_*,0)]/(\varepsilon V_0)$",
              title=r"Step 1: $S$-axis ($\phi_*=\phi_{A/A'}$)", ylim=(-1.5, 1.0))
    ax[0].legend(fontsize=9)
    ax[1].axhline(0, c="0.5", lw=0.7)
    ax[1].set(xlabel=r"$\phi$ [GeV]",
              ylabel=r"$[V(\phi,s_A)-V(A)]/T^4$",
              title=r"Near $A/A'$ along fixed $S=s_T$")
    ax[1].legend(fontsize=9)
    fig.suptitle(rf"$T={T/1e3:g}$ TeV, $m_S={m_S:g}$, $\kappa={kappa:g}$, "
                 rf"$\varepsilon={eps:.0e}$")
    fig.tight_layout()
    path = os.path.join(out_dir, "potential_slices_mixed.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    log(f"wrote {path}")


def _mark_vacua(ax, vac, s0, v: float = 1.0, mode: str = "global"):
    """Mark O, A/A′, B. mode='global' uses φ/v; mode='zoom' uses φ [GeV]."""
    handles = False
    if mode == "global":
        if vac.get("O") is not None:
            ax.plot(vac["O"][0] / v, vac["O"][1] / s0, "ko", ms=7, label=r"$O$")
            handles = True
        if vac.get("A") is not None:
            lab = r"$A'$" if vac["A"][0] > 1.0 else r"$A$"
            ax.plot(vac["A"][0] / v, vac["A"][1] / s0, "o", c="C1", ms=8, label=lab)
            handles = True
        if vac.get("B") is not None:
            ax.plot(vac["B"][0] / v, vac["B"][1] / s0, "*", c="C3", ms=12, label=r"$B$")
            handles = True
    else:
        if vac.get("A") is not None:
            lab = r"$A'$" if vac["A"][0] > 1.0 else r"$A$"
            ax.plot(vac["A"][0], vac["A"][1] / s0, "o", c="C1", ms=8, label=lab)
            handles = True
        if vac.get("O") is not None:
            ax.plot(0.0, vac["O"][1] / s0, "ko", ms=6, label=r"$O$")
            handles = True
    if handles:
        ax.legend(fontsize=8, loc="upper right")


# --------------------------------------------------------------------------
# Nested-bubble limit figures (FIG.3 style)
# --------------------------------------------------------------------------
def plot_nested_mixed(df, out_dir: str, dTn_max: float = 50.0) -> None:
    """ΔT_n / nested-window panels for mixed-potential scan rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import pandas as pd
    from analysis.portalNestedSearch import plot_nested_search

    os.makedirs(out_dir, exist_ok=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty:
        log("no rows for nested plots")
        return

    # Reuse unmixed-style (m_S, κ) plots per phi_off slice
    if "phi_off_target" in df.columns:
        for tgt, sub in df.groupby("phi_off_target"):
            sub_dir = os.path.join(out_dir, f"phi_off_{float(tgt):g}")
            if sub["m_S"].nunique() >= 2 and sub["kappa"].nunique() >= 2:
                plot_nested_search(sub, sub_dir, dTn_max=dTn_max)
                log(f"nested (m_S,κ) panels → {sub_dir}")

    # ---- (ε, φ_off) plane at fixed (m_S, κ) ----
    if ("eps" in df.columns and "phi_off_target" in df.columns
            and df["eps"].nunique() >= 2 and df["phi_off_target"].nunique() >= 2):
        # pick the most-populated (m_S, κ)
        counts = (df.groupby(["m_S", "kappa"]).size()
                    .sort_values(ascending=False))
        (m_fix, k_fix) = counts.index[0]
        sub = df[(df["m_S"] == m_fix) & (df["kappa"] == k_fix)].copy()
        piv = sub.pivot_table(index="phi_off_target", columns="eps",
                              values="delta_T_n_S", aggfunc="first")
        if piv.shape[0] >= 2 and piv.shape[1] >= 2:
            eps_vals = np.array(sorted(piv.columns), float)
            po_vals = np.array(sorted(piv.index), float)
            Z = piv.reindex(index=po_vals, columns=eps_vals).to_numpy(dtype=float)
            EE, PP = np.meshgrid(eps_vals, po_vals)
            fig, ax = plt.subplots(figsize=(7.8, 5.4))
            ax.set_facecolor("#d0d0d0")
            Zplot = np.ma.masked_invalid(Z)
            if np.isfinite(Z).any():
                vmax = float(np.nanmax(Zplot))
                levels = np.linspace(0.0, max(vmax, dTn_max * 1.05), 18)
                cf = ax.contourf(EE, PP, Zplot, levels=levels,
                                 cmap="viridis_r", extend="max")
                cb = fig.colorbar(cf, ax=ax, pad=0.02)
                cb.set_label(r"$\Delta T_n$ [GeV]")
                if np.nanmin(Z) < dTn_max < np.nanmax(Z):
                    ax.contour(EE, PP, Zplot, levels=[dTn_max],
                               colors="w", linewidths=2.0)
            # scatter regime
            dTn = sub["delta_T_n_S"].to_numpy(dtype=float)
            ok = np.isfinite(dTn)
            ax.scatter(sub.loc[~ok, "eps"], sub.loc[~ok, "phi_off_target"],
                       s=40, c="0.2", marker="x", zorder=5, label=r"$0\to v$/stuck")
            ax.scatter(sub.loc[ok & (dTn <= dTn_max), "eps"],
                       sub.loc[ok & (dTn <= dTn_max), "phi_off_target"],
                       s=45, c="lime", edgecolors="k", zorder=5,
                       label=rf"nested $\Delta T_n\leq{dTn_max:g}$")
            ax.scatter(sub.loc[ok & (dTn > dTn_max), "eps"],
                       sub.loc[ok & (dTn > dTn_max), "phi_off_target"],
                       s=45, c="w", edgecolors="k", zorder=5,
                       label=rf"separated $\Delta T_n>{dTn_max:g}$")
            ax.set_xscale("log")
            ax.set(xlabel=r"$\varepsilon$", ylabel=r"$\phi_{\rm off}^{\rm tgt}$ [GeV]",
                   title=rf"Nested limit in $(\varepsilon,\phi_{{\rm off}})$  "
                         rf"($m_S={m_fix:g}$, $\kappa={k_fix:g}$)")
            ax.legend(fontsize=8, loc="best", framealpha=0.92)
            fig.tight_layout()
            path = os.path.join(out_dir, "nested_window_eps_phi_off.png")
            fig.savefig(path, dpi=170)
            plt.close(fig)
            log(f"wrote {path}")

        # regime map in (ε, φ_off)
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        code = np.full(len(sub), 0.0)
        dTn = sub["delta_T_n_S"].to_numpy(dtype=float)
        finite = np.isfinite(dTn)
        code[finite & (dTn > dTn_max)] = 1.0
        code[finite & (dTn > 0) & (dTn <= dTn_max)] = 2.0
        # also mark two-step nested by regime string
        if "regime" in sub.columns:
            nest_reg = sub["regime"].astype(str).str.contains("nested", case=False)
            code[nest_reg.to_numpy()] = 2.0
        from scipy.interpolate import griddata
        pts = sub[["eps", "phi_off_target"]].to_numpy(dtype=float)
        # log-ε for interpolation
        pts_log = np.column_stack([np.log10(np.clip(pts[:, 0], 1e-30, None)), pts[:, 1]])
        e_lin = np.linspace(np.log10(sub["eps"].min()), np.log10(sub["eps"].max()), 80)
        p_lin = np.linspace(sub["phi_off_target"].min(), sub["phi_off_target"].max(), 80)
        EE, PP = np.meshgrid(e_lin, p_lin)
        Zr = griddata(pts_log, code, (EE, PP), method="nearest")
        cmap = mcolors.ListedColormap(["#bdbdbd", "#f4e6c8", "#2ca02c"])
        ax.contourf(10 ** EE, PP, Zr, levels=[-0.5, 0.5, 1.5, 2.5],
                    cmap=cmap, alpha=0.95)
        ax.scatter(sub["eps"], sub["phi_off_target"], s=36, c="k", zorder=4)
        ax.set_xscale("log")
        ax.set(xlabel=r"$\varepsilon$", ylabel=r"$\phi_{\rm off}^{\rm tgt}$ [GeV]",
               title=rf"Nested window $(\varepsilon,\phi_{{\rm off}})$  "
                     rf"($m_S={m_fix:g}$, $\kappa={k_fix:g}$)")
        handles = [
            Line2D([0], [0], color="#2ca02c", lw=8,
                   label=rf"nested ($\Delta T_n\leq{dTn_max:g}$)"),
            Line2D([0], [0], color="#f4e6c8", lw=8,
                   label=rf"separated ($\Delta T_n>{dTn_max:g}$)"),
            Line2D([0], [0], color="#bdbdbd", lw=8, label=r"$0\to v$ / stuck"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")
        fig.tight_layout()
        path = os.path.join(out_dir, "nested_regime_eps_phi_off.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        log(f"wrote {path}")

    # ---- ΔT_n vs φ_off at fixed ε (line plot) ----
    if "phi_off_target" in df.columns and "eps" in df.columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for eps, sub in df.groupby("eps"):
            # average over m_S/κ if multiple
            g = sub.groupby("phi_off_target", as_index=False)["delta_T_n_S"].mean()
            g = g.sort_values("phi_off_target")
            ax.plot(g["phi_off_target"], g["delta_T_n_S"], "o-",
                    label=rf"$\varepsilon={eps:.0e}$")
        ax.axhline(dTn_max, color="0.35", ls="--",
                   label=rf"$\Delta T_n={dTn_max:g}$ GeV")
        ax.axhspan(0, dTn_max, color="C2", alpha=0.10, zorder=0)
        ax.set(xlabel=r"$\phi_{\rm off}^{\rm tgt}$ [GeV]",
               ylabel=r"$\langle\Delta T_n\rangle$ [GeV]",
               title=r"Mixing shrinks $\Delta T_n$ (nested limit)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, "delta_Tn_vs_phi_off.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        log(f"wrote {path}")


def run_nested_limit_scan(out_dir: str, workers: int = 8, dTn_max: float = 50.0,
                          nT: int = 18, refine: int = 6) -> list[dict]:
    """Compact scan for nested-limit figures with mixed potential."""
    os.makedirs(out_dir, exist_ok=True)

    # (m_S, κ) plane at ε=1e-20 for φ_off ∈ {0, 400}
    # (ε, φ_off) plane at m_S=500, κ=0.1 (nested-friendly m_S)
    m_Ss_plane = [460.0, 500.0, 550.0, 600.0, 800.0, 1000.0]
    kappas_plane = [0.05, 0.1, 0.2, 0.3]
    phi_offs_plane = [0.0, 400.0]
    eps_plane = 1e-20

    m_fix, k_fix = 500.0, 0.1
    epss_mix = [1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 0.01]
    phi_offs_mix = [0.0, 100.0, 400.0, 800.0]

    jobs = []
    for m in m_Ss_plane:
        for k in kappas_plane:
            for po in phi_offs_plane:
                jobs.append((eps_plane, k, m, po, "mS_kappa"))
    for e in epss_mix:
        for po in phi_offs_mix:
            jobs.append((e, k_fix, m_fix, po, "eps_phi"))

    # dedup
    uniq, jobs_u = set(), []
    for e, k, m, po, tag in jobs:
        key = (round(e, 20), round(k, 6), round(m, 3), round(po, 4))
        if key in uniq:
            continue
        uniq.add(key)
        jobs_u.append((e, k, m, po, tag))
    jobs = jobs_u
    log(f"nested-limit scan: {len(jobs)} jobs → {out_dir}")

    rows = []
    t0 = time.time()
    for i, (eps, kappa, m_S, po, tag) in enumerate(jobs, 1):
        s0 = s0_from_mS(eps, SET8_V0, m_S)
        mu2 = (mu2_for_phi_off(
            SET8_V0, SET8_V, s0, eps, kappa,
            SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)],
            100.0, "set8", po) if po > 0 else 0.0)
        # T window densified near T0
        pot_tmp = build_pot(
            SET8_V0, SET8_V, s0, eps, kappa,
            SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], 100.0,
            phi_thermal="set8", mu2_mix=mu2)
        T0lo = min(pot_tmp.T0_S(), pot_tmp.T0_phi())
        T0hi = max(pot_tmp.T0_S(), pot_tmp.T0_phi())
        args = SimpleNamespace(
            V0=SET8_V0, v=SET8_V, m_S=m_S, s0=None,
            boson_phi=SET8_BOSON_PHI, fermion_phi=SET8_FERMION_PHI,
            boson_S=[(6.0, 1.0)], g_star=100.0, phi_thermal="set8",
            Tmin=max(50.0, 0.4 * T0lo), Tmax=1.35 * T0hi,
            nT=nT, refine=refine, workers=workers,
            a2v_n_fine=20, a2v_T_hi_fac=1.25, a2v_T_lo_fac=0.85,
            a2v_S_trigger=2000.0, T_reh=DEFAULT_T_REH,
            rmix=None, mu2_mix=mu2,
        )
        log(f"\n======== nested [{i}/{len(jobs)}] ({tag}) m_S={m_S:g} κ={kappa:g} "
            f"ε={eps:.3g} φ_off={po:g} μ²={mu2:.3g} ========")
        try:
            row = run_one(eps, kappa, args, i, len(jobs))
        except Exception as exc:
            log(f"FAILED: {exc}")
            row = {"eps": eps, "kappa": kappa, "m_S": m_S, "regime": f"error: {exc}",
                   "delta_T_n_S": float("nan")}
        row["phi_off_target"] = po
        row["mu2_mix"] = mu2
        row["scan_tag"] = tag
        dTn = row.get("delta_T_n_S", float("nan"))
        seq = str(row.get("sequence", ""))
        nested_seq = ("nested" in seq.lower()) or ("Regime II" in seq)
        nested_dTn = (isinstance(dTn, (int, float)) and math.isfinite(dTn)
                      and 0.0 < dTn <= dTn_max)
        row["nested_by_sequence"] = bool(nested_seq)
        row["nested_by_dTn"] = bool(nested_dTn)
        row["nested"] = bool(nested_seq or nested_dTn)
        rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "nested_mixed.csv"),
                                  index=False)
        log(f"  → regime={row.get('regime')}  ΔT_n={dTn}  nested={row['nested']}")

    with open(os.path.join(out_dir, "nested_mixed.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    log(f"nested-limit scan done in {time.time() - t0:.0f}s")
    return rows


def main():
    from analysis import portalEpsScan as _pes

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/portal/mixed_figures")
    p.add_argument("--landscape_only", action="store_true",
                   help="vacuum landscape (3D+2D) + saddle-approach figures only")
    p.add_argument("--pot_only", action="store_true",
                   help="legacy μ-comparison potential panels (no nested scan)")
    p.add_argument("--nested_only", action="store_true",
                   help="only nested-limit scan/plots (skip potential)")
    p.add_argument("--replot_nested", action="store_true",
                   help="replot nested figures from nested_mixed.csv")
    p.add_argument("--eps", type=float, default=1e-20)
    p.add_argument("--m_S", type=float, default=1000.0)
    p.add_argument("--kappa", type=float, default=0.3)
    p.add_argument("--T", type=float, default=2000.0,
                   help="temperature [GeV] for saddle / legacy panels")
    p.add_argument("--landscape_eps", type=float, default=0.05,
                   help="ε for vacuum-landscape figure (larger ⇒ A shelf visible)")
    p.add_argument("--landscape_m_S", type=float, default=1500.0)
    p.add_argument("--landscape_kappa", type=float, default=0.1)
    p.add_argument("--landscape_T", type=float, default=3200.0,
                   help="T [GeV] with O, A, B all present for landscape")
    p.add_argument("--phi_off", type=str, default="0,100,200,400,600",
                   help="φ_off targets for saddle-approach figure")
    p.add_argument("--dTn_max", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nT", type=int, default=18)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "progress.log")
    _pes._LOG = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _pes._LOG)
    sys.stderr = _Tee(sys.__stderr__, _pes._LOG)

    phi_offs = [float(x) for x in args.phi_off.split(",") if x.strip()]

    if args.landscape_only or (not args.nested_only and not args.pot_only):
        log("=== vacuum landscape (3D + 2D) ===")
        plot_vacuum_landscape(
            args.out, eps=args.landscape_eps, m_S=args.landscape_m_S,
            kappa=args.landscape_kappa, T=args.landscape_T, phi_off_tgt=0.0)
        log("=== saddle approach vs mixing ===")
        plot_saddle_approach(
            args.out, eps=args.eps, m_S=args.m_S, kappa=args.kappa,
            T=args.T, phi_off_tgts=tuple(phi_offs))
        if args.landscape_only:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _pes._LOG.close()
            return

    if args.pot_only:
        log("=== legacy μ-comparison panels ===")
        plot_potential_panels(
            args.out, eps=args.eps, m_S=args.m_S, kappa=args.kappa,
            T=args.T, phi_off_tgts=tuple(phi_offs[:2] if len(phi_offs) > 2
                                         else phi_offs))
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _pes._LOG.close()
        return

    nest_dir = os.path.join(args.out, "nested")
    os.makedirs(nest_dir, exist_ok=True)
    csv_path = os.path.join(nest_dir, "nested_mixed.csv")

    if args.replot_nested and os.path.isfile(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        log(f"replot nested from {csv_path}")
    else:
        log("=== nested-bubble limit scan ===")
        rows = run_nested_limit_scan(
            nest_dir, workers=args.workers, dTn_max=args.dTn_max, nT=args.nT)
        import pandas as pd
        df = pd.DataFrame(rows)

    plot_nested_mixed(df, nest_dir, dTn_max=args.dTn_max)
    log(f"\nAll figures under {args.out}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _pes._LOG.close()


if __name__ == "__main__":
    main()
