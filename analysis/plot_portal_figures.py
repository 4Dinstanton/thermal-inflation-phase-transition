#!/usr/bin/env python3
"""Publication figures for the portal model with λ_φS |φ|²|S|² only (no μ_mix).

Produces:
  * vacuum_landscape.pdf/.png  — 3D surface + contour of V(φ,S) at finite T
  * potential_slices.pdf/.png  — 1D cuts along O→A and A→B
  * gw_separated_*.pdf/.png    — Sec. II two-step separated spectra

No bilinear φ*S mixing.  Parameters: (ε, κ, m_S) only.

Example
-------
    PYTHONPATH=. python analysis/plot_portal_figures.py \\
        --out data/portal/portal_figures --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
    build_pot,
    log,
    run_one,
    s0_from_mS,
)
from analysis.twoStepGW import (  # noqa: E402
    apply_two_step_gw,
    make_step,
    shape,
    spectrum,
    step_quantities,
)


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, path_base: str):
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        p = f"{path_base}.{ext}"
        fig.savefig(p)
        log(f"wrote {p}")


def _make_pot(eps, m_S, kappa, V0=SET8_V0, v=SET8_V):
    s0 = s0_from_mS(eps, V0, m_S)
    pot = build_pot(
        V0, v, s0, eps, kappa,
        SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], 100.0,
        phi_thermal="set8", mu2_mix=None, rmix=None)
    return pot, s0


# --------------------------------------------------------------------------
# Potential figures
# --------------------------------------------------------------------------
def plot_vacuum_landscape(out_dir: str, eps=1e-20, m_S=1000.0, kappa=0.3,
                          T=2000.0) -> None:
    """Finite-T portal landscape: O, A=(0,s_T), B=(v,0) — λ_φS |φ|²|S|² only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _style()
    pot, s0 = _make_pot(eps, m_S, kappa)
    vac = pot.vacua(T)
    if vac.get("A") is None or vac.get("B") is None:
        raise RuntimeError(f"Need A and B at T={T}; got {vac}")

    v = float(pot.v)
    V_B = float(vac["B"][2])
    V_A = float(vac["A"][2])
    V_O = float(vac["O"][2]) if vac.get("O") is not None else float(
        pot.V_rel(0.0, 0.0, T))
    E = float(pot.V0)
    u_A = (V_A - V_B) / E
    u_O = (V_O - V_B) / E

    n = 280
    phi = np.linspace(0.0, 1.08 * v, n)
    s = np.linspace(0.0, 1.18 * s0, n)
    P, S = np.meshgrid(phi, s)
    U = (pot.V_rel(P, S, T) - V_B) / E
    zmax = max(u_O * 1.015, 1.015)
    Uplot = np.clip(U, -0.015, zmax)

    fig = plt.figure(figsize=(11.5, 5.0))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3.plot_surface(
        P / v, S / s0, Uplot, cmap="cividis", linewidth=0,
        antialiased=True, rstride=3, cstride=3, alpha=0.95,
        vmin=-0.015, vmax=zmax)

    def _Vn(ph, ss):
        return float(np.clip((pot.V_rel(ph, ss, T) - V_B) / E, -0.015, zmax))

    ax3.scatter([0], [0], [_Vn(0.0, 0.0)], c="0.1", s=55, depthshade=False,
                label=rf"$O$ ($U={u_O:.3f}$)")
    ax3.scatter([0], [vac["A"][1] / s0], [_Vn(0.0, vac["A"][1])],
                c="#d95f02", s=65, depthshade=False,
                label=rf"$A$ ($U={u_A:.3f}$)")
    ax3.scatter([vac["B"][0] / v], [0], [0.0],
                c="#1b9e77", s=95, marker="*", depthshade=False,
                label=r"$B$ ($U=0$)")
    ax3.set_xlabel(r"$\phi/v$")
    ax3.set_ylabel(r"$S/s_0$")
    ax3.set_zlabel(r"$[V-V(B)]/V_0$")
    ax3.view_init(elev=24, azim=-58)
    ax3.legend(loc="upper left", fontsize=8)
    fig.colorbar(surf, ax=ax3, shrink=0.55, pad=0.06, label=r"$U$")

    ax2 = fig.add_subplot(1, 2, 2)
    lo_u = min(-0.01, u_A - 0.01, 0.0)
    hi_u = max(zmax, u_O, u_A, 0.05)
    mid_lo = min(max(u_A, 0.02), hi_u)
    mid_hi = min(max(u_O, mid_lo + 1e-6), hi_u)
    levels = np.unique(np.concatenate([
        np.linspace(lo_u, mid_lo, 8),
        np.linspace(mid_lo, mid_hi, 12),
        np.linspace(mid_hi, hi_u, 5),
    ]))
    levels = levels[np.isfinite(levels)]
    cf = ax2.contourf(P / v, S / s0, Uplot, levels=levels, cmap="cividis",
                      extend="both")
    clev = sorted({float(x) for x in (0.0, u_A, u_O) if np.isfinite(x)})
    # ensure strictly increasing
    clev_use = []
    for lv in clev:
        if not clev_use or lv > clev_use[-1] + 1e-12:
            clev_use.append(lv)
    cs = None
    if len(clev_use) >= 1:
        cols = {0.0: "#1b9e77", u_A: "#d95f02", u_O: "0.25"}
        cs = ax2.contour(
            P / v, S / s0, U, levels=clev_use,
            colors=[cols.get(lv, "0.3") for lv in clev_use],
            linewidths=[1.8] * len(clev_use))
        fmt = {0.0: r"$V(B)$", u_A: r"$V(A)$", u_O: r"$V(O)$"}
        ax2.clabel(cs, fmt={lv: fmt.get(lv, f"{lv:.3g}") for lv in clev_use},
                   fontsize=8)
    ax2.plot(0, 0, "o", c="0.1", ms=8, label=r"$O$")
    ax2.plot(0, vac["A"][1] / s0, "o", c="#d95f02", ms=9, label=r"$A$")
    ax2.plot(vac["B"][0] / v, 0, "*", c="#1b9e77", ms=13, label=r"$B$")
    ax2.plot([0, 0, vac["B"][0] / v],
             [0, vac["A"][1] / s0, 0],
             ls="--", c="0.35", lw=1.0, alpha=0.7, label=r"$O\!\to\!A\!\to\!B$")
    ax2.set(xlabel=r"$\phi/v$", ylabel=r"$S/s_0$",
            xlim=(0, 1.08), ylim=(0, 1.18))
    ax2.legend(loc="upper right", fontsize=8)
    fig.colorbar(cf, ax=ax2, fraction=0.046, pad=0.03, label=r"$[V-V(B)]/V_0$")

    fig.suptitle(
        rf"Portal $V\supset\lambda_{{\phi S}}|\phi|^2|S|^2$  "
        rf"($\varepsilon={eps:.0e}$, $m_S={m_S:g}$ GeV, $\kappa={kappa:g}$, "
        rf"$T={T:g}$ GeV)",
        fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "vacuum_landscape"))
    plt.close(fig)


def plot_potential_slices(out_dir: str, eps=1e-20, m_S=1000.0, kappa=0.3,
                          T=2000.0) -> None:
    """1D potential cuts: φ=0 (O→A) and S=s_T (A→B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    pot, s0 = _make_pot(eps, m_S, kappa)
    vac = pot.vacua(T)
    if vac.get("A") is None or vac.get("B") is None:
        raise RuntimeError(f"Need A and B at T={T}")

    s_A = float(vac["A"][1])
    V_A = float(vac["A"][2])
    V_B = float(vac["B"][2])
    v = float(pot.v)
    E = float(pot.V0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # O → A along S at φ=0
    s = np.linspace(0, 1.25 * s0, 600)
    U_s = (pot.V_rel(0.0, s, T) - V_B) / E
    axes[0].plot(s / s0, U_s, c="#333333", lw=1.8)
    axes[0].axhline((V_A - V_B) / E, ls=":", c="#d95f02", lw=1.0)
    axes[0].plot(0, (float(vac["O"][2]) - V_B) / E if vac.get("O") else U_s[0],
                 "o", c="0.1", ms=8)
    axes[0].plot(s_A / s0, (V_A - V_B) / E, "o", c="#d95f02", ms=9)
    axes[0].set(xlabel=r"$S/s_0$", ylabel=r"$[V(0,S)-V(B)]/V_0$",
                title=r"$O\to A$  ($\phi=0$)")
    axes[0].set_xlim(0, 1.25)

    # A → B along φ at S = s_A
    phi = np.linspace(0, 1.15 * v, 600)
    U_p = (pot.V_rel(phi, s_A, T) - V_A) / E
    axes[1].plot(phi / v, U_p, c="#333333", lw=1.8)
    axes[1].axhline(0, ls=":", c="#d95f02", lw=1.0)
    axes[1].axhline((V_B - V_A) / E, ls=":", c="#1b9e77", lw=1.0)
    axes[1].plot(0, 0, "o", c="#d95f02", ms=9, label=r"$A$")
    axes[1].plot(vac["B"][0] / v, (V_B - V_A) / E, "*", c="#1b9e77", ms=12,
                 label=r"$B$")
    axes[1].set(xlabel=r"$\phi/v$", ylabel=r"$[V(\phi,s_T)-V(A)]/V_0$",
                title=rf"$A\to B$  ($S=s_T={s_A:.3g}$ GeV)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xlim(0, 1.15)

    fig.suptitle(
        rf"$\varepsilon={eps:.0e}$, $m_S={m_S:g}$ GeV, $\kappa={kappa:g}$, "
        rf"$T={T:g}$ GeV",
        fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "potential_slices"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Separated two-step GW figures
# --------------------------------------------------------------------------
def _qs_from_row(row: dict, V0: float, T_reh: float = DEFAULT_T_REH,
                 g_star: float = 100.0):
    eps = float(row.get("eps_eff", row.get("eps", float("nan"))))
    T1 = float(row.get("T_n_step1", row.get("T_n_0_to_S", float("nan"))))
    T2 = float(row.get("T_n_step2", row.get("T_n_S_to_v", float("nan"))))
    b1 = float(row.get("betaH_step1", float("nan")))
    b2 = float(row.get("betaH_step2", float("nan")))
    b0 = float(row.get("betaH_phi", float("nan")))
    T0 = float(row.get("T_n_0_to_v", row.get("T_n_phi", float("nan"))))
    qs_raw = []
    if np.isfinite(eps) and eps > 0 and np.isfinite(T1) and np.isfinite(b1) and b1 > 0:
        qs_raw.append(step_quantities(make_step(
            "step1", eps * V0, T1, b1, V0, T_reh, g_star=g_star)))
    if (np.isfinite(eps) and 0 < eps < 1 and np.isfinite(T2)
            and np.isfinite(b2) and b2 > 0):
        qs_raw.append(step_quantities(make_step(
            "step2", (1.0 - eps) * V0, T2, b2, V0, T_reh, g_star=g_star)))
    qs, meta = apply_two_step_gw(qs_raw, eps=eps if np.isfinite(eps) else 0.0)
    ref = None
    if np.isfinite(T0) and np.isfinite(b0) and b0 > 0:
        ref = step_quantities(make_step(
            "0->v", V0, T0, b0, V0, T_reh, g_star=g_star))
    return qs, ref, meta


def plot_gw_separated(rows: list[dict], out_dir: str, V0: float = SET8_V0,
                      T_reh: float = DEFAULT_T_REH,
                      max_panels: int = 6) -> None:
    """GW spectra for two-step separated points (Sec. II sequential)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    sep = [r for r in rows
           if "separat" in str(r.get("regime", "")).lower()
           or (str(r.get("gw_formula", "")).startswith("Sec.IV")
               and np.isfinite(r.get("T_n_step2", float("nan")))
               and np.isfinite(r.get("T_n_step1", float("nan")))
               and float(r.get("delta_T_n_S", 1e9)) > 50)]
    # prefer finite GW peaks
    sep = [r for r in sep
           if np.isfinite(r.get("h2Omega_peak_tot", float("nan")))
           or np.isfinite(r.get("betaH_step1", float("nan")))]
    if not sep:
        log("no separated two-step rows for GW figures")
        return

    # diversify: unique (eps, kappa, m_S). Prefer clear separation + finite peaks,
    # spanning small→large ε so dilution / ε² suppression is visible.
    key = lambda r: (round(float(r.get("eps", 0)), 20),
                     round(float(r.get("kappa", 0)), 6),
                     round(float(r.get("m_S", 0)), 3))

    def _score(r):
        dTn = float(r.get("delta_T_n_S", float("nan")))
        eps = float(r.get("eps", 1.0))
        h2 = float(r.get("h2Omega_peak_tot", float("nan")))
        # want separated (large ΔT_n), detectable peak, and mix of ε
        return (
            0 if (np.isfinite(h2) and h2 > 0) else 1,
            -min(dTn, 5000.0) if np.isfinite(dTn) else 0.0,
            abs(math.log10(max(eps, 1e-30)) + 10),  # prefer near 1e-10…1e-2
        )

    seen, picks = set(), []
    for r in sorted(sep, key=_score):
        k = key(r)
        if k in seen:
            continue
        # skip if step peaks missing
        if not (np.isfinite(r.get("betaH_step1", float("nan")))
                and np.isfinite(r.get("betaH_step2", float("nan")))):
            continue
        seen.add(k)
        picks.append(r)
        if len(picks) >= max_panels:
            break

    # --- multi-panel spectra ---
    n = len(picks)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows),
                             squeeze=False)
    f = np.geomspace(1e-4, 1e4, 2500)

    try:
        from analysis.published_plisc import add_detector_curves
        have_det = True
    except Exception:
        have_det = False

    for i, row in enumerate(picks):
        ax = axes[i // ncols][i % ncols]
        qs, ref, meta = _qs_from_row(row, V0, T_reh)
        if have_det:
            try:
                add_detector_curves(ax)
            except Exception:
                pass
        tot = np.zeros_like(f)
        styles = {
            "step1": ("#7570b3", "--", r"$0\to S$"),
            "step2": ("#d95f02", "-.", r"$S\to v$"),
            "composite": ("#1b9e77", "-", r"composite"),
            "cross": ("#e7298a", ":", r"cross"),
        }
        for q in qs:
            if not (q.get("h2Omega_peak", 0) > 0 and q.get("f_peak [Hz]", 0) > 0):
                continue
            lab = q["label"]
            col, ls, nice = styles.get(lab, ("0.4", "-", lab))
            s = spectrum(f, q)
            tot = tot + s
            ax.loglog(f, s, ls=ls, color=col, lw=1.4,
                      label=rf"{nice}: $f_p$={q['f_peak [Hz]']:.3g}")
        if tot.max() > 0:
            ax.loglog(f, tot, color="0.1", lw=2.0, label="total")
        if ref and ref.get("h2Omega_peak", 0) > 0:
            ax.loglog(f, spectrum(f, ref), ":", color="#1b9e77", lw=1.6,
                      label=rf"$0\to v$ ref")
        ax.set_ylim(1e-25, 1e-5)
        ax.set_xlim(1e-4, 1e4)
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_ylabel(r"$h^2\Omega_{\rm GW}$")
        eps = float(row.get("eps", float("nan")))
        kap = float(row.get("kappa", float("nan")))
        mS = float(row.get("m_S", float("nan")))
        dTn = float(row.get("delta_T_n_S", float("nan")))
        ax.set_title(
            rf"$\varepsilon={eps:.2g}$, $\kappa={kap:g}$, $m_S={mS:g}$"
            + (rf", $\Delta T_n={dTn:.0f}$ GeV" if np.isfinite(dTn) else ""),
            fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.22, which="both")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(r"Sec.~II: two-step separated SGWB  "
                 r"($\lambda_{\phi S}|\phi|^2|S|^2$ only)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "gw_separated_panels"))
    plt.close(fig)

    # --- overlay of totals vs single-step ref (one figure) ---
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if have_det:
        try:
            add_detector_curves(ax)
        except Exception:
            pass
    cmap = plt.cm.plasma
    for i, row in enumerate(picks):
        qs, ref, meta = _qs_from_row(row, V0, T_reh)
        tot = np.zeros_like(f)
        for q in qs:
            if q.get("h2Omega_peak", 0) > 0 and q.get("f_peak [Hz]", 0) > 0:
                tot = tot + spectrum(f, q)
        if tot.max() <= 0:
            continue
        col = cmap(i / max(1, len(picks) - 1))
        eps = float(row.get("eps", float("nan")))
        mS = float(row.get("m_S", float("nan")))
        ax.loglog(f, tot, color=col, lw=1.7,
                  label=rf"$\varepsilon={eps:.1g}$, $m_S={mS:g}$")
    # one reference if available
    for row in picks:
        qs, ref, meta = _qs_from_row(row, V0, T_reh)
        if ref and ref.get("h2Omega_peak", 0) > 0:
            ax.loglog(f, spectrum(f, ref), "k--", lw=1.5, alpha=0.75,
                      label=r"$0\to v$ reference")
            break
    ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{\rm GW}$",
           xlim=(1e-4, 1e4), ylim=(1e-25, 1e-5),
           title=r"Separated two-step totals vs single-step reference")
    ax.legend(fontsize=8, loc="upper left", ncol=1)
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "gw_separated_overlay"))
    plt.close(fig)


def plot_gw_from_live_points(out_dir: str, points: list[tuple],
                             workers: int = 4) -> list[dict]:
    """Run a few portal points (no μ_mix) and plot their separated GW."""
    rows = []
    for i, (eps, kappa, m_S) in enumerate(points, 1):
        args = SimpleNamespace(
            V0=SET8_V0, v=SET8_V, m_S=m_S, s0=None,
            boson_phi=SET8_BOSON_PHI, fermion_phi=SET8_FERMION_PHI,
            boson_S=[(6.0, 1.0)], g_star=100.0, phi_thermal="set8",
            Tmin=None, Tmax=None, nT=20, refine=8, workers=workers,
            a2v_n_fine=20, a2v_T_hi_fac=1.25, a2v_T_lo_fac=0.85,
            a2v_S_trigger=2000.0, T_reh=DEFAULT_T_REH,
            rmix=None, mu2_mix=None,
        )
        log(f"\n--- GW figure point [{i}/{len(points)}] "
            f"ε={eps:.3g} κ={kappa:g} m_S={m_S:g} ---")
        try:
            row = run_one(eps, kappa, args, i, len(points))
        except Exception as exc:
            log(f"  FAILED: {exc}")
            continue
        rows.append(row)
        log(f"  regime={row.get('regime')}  ΔT_n={row.get('delta_T_n_S')}")
    if rows:
        with open(os.path.join(out_dir, "gw_figure_points.json"), "w") as fh:
            json.dump(rows, fh, indent=2, default=float)
        plot_gw_separated(rows, out_dir, V0=SET8_V0)
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/portal/portal_figures")
    p.add_argument("--eps", type=float, default=1e-20)
    p.add_argument("--m_S", type=float, default=1000.0)
    p.add_argument("--kappa", type=float, default=0.3)
    p.add_argument("--T", type=float, default=2000.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pot_only", action="store_true")
    p.add_argument("--gw_only", action="store_true")
    p.add_argument("--from_csv", action="append", default=None,
                   help="reuse separated rows from existing scan CSV(s)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    if not a.gw_only:
        log("potential figures (λ_φS |φ|²|S|² only, no μ_mix)")
        plot_vacuum_landscape(a.out, eps=a.eps, m_S=a.m_S, kappa=a.kappa, T=a.T)
        plot_potential_slices(a.out, eps=a.eps, m_S=a.m_S, kappa=a.kappa, T=a.T)

    if not a.pot_only:
        rows = []
        if a.from_csv:
            import pandas as pd
            for path in a.from_csv:
                df = pd.read_csv(path)
                # drop mixed points if present
                if "phi_off_target" in df.columns:
                    df = df[df["phi_off_target"].fillna(0) == 0]
                if "mu2_mix" in df.columns:
                    df = df[df["mu2_mix"].fillna(0).abs() < 1e-30]
                if "rmix" in df.columns:
                    df = df[df["rmix"].fillna(0).abs() < 1e-30]
                rows.extend(df.to_dict("records"))
            log(f"loaded {len(rows)} unmixed rows from CSV")
            plot_gw_separated(rows, a.out)
        else:
            # representative separated benchmarks
            points = [
                (1e-20, 0.1, 800.0),
                (1e-20, 0.3, 1000.0),
                (1e-3, 0.1, 800.0),
                (0.01, 0.1, 1000.0),
                (0.05, 0.1, 1500.0),
                (0.1, 0.05, 1000.0),
            ]
            plot_gw_from_live_points(a.out, points, workers=a.workers)

    log(f"done → {a.out}")


if __name__ == "__main__":
    main()
