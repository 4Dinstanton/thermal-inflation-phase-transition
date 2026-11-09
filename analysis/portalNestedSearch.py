#!/usr/bin/env python3
"""Search (eps, kappa, m_S) for nested two-step / small ΔT_n portal bubbles.

Existing fixed-m_S=1 TeV scans have ΔT_n = T_n(0→S)-T_n(S→v) ~ 1.4 TeV.
T0_S = 2 m_S (n_S=6, g=1), and T_n(0→S) ~ 1.26 T0_S, while T_n(S→v) sits
near T0_phi ~ 1.15 TeV. Closing the gap requires lowering m_S until
T_n(0→S) approaches T_n(S→v), while keeping T_n(0→S) > T_n(0→v) so S still
nucleates first.

Nested criteria (either):
  * code Regime II / "nested" in sequence
  * ΔT_n <= dTn_max (default 50 GeV) with a completed two-step

Example
-------
    PYTHONPATH=. python analysis/portalNestedSearch.py \\
        --m_S 480 --m_S 500 --m_S 520 --kappa 0.05 --kappa 0.1 \\
        --eps 1e-20 --workers 8 --out data/portal/nested_search
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
    _LOG,
    _Tee,
    _parse_ng,
    log,
    run_one,
)


def _pivot_field(df, field: str, dense: bool = True):
    """Return (m_S 1d, kappa 1d, Z 2d) for a (m_S, kappa) scan.

    Prefer the native rectangular grid (NaNs kept for 0→v / stuck cells).
    Optionally upsample with linear interpolation over finite cells only.
    """
    from scipy.interpolate import griddata

    d = df.dropna(subset=["m_S", "kappa"]).copy()
    if d.empty or field not in d.columns:
        return None
    # one value per (m_S, kappa)
    d = d.sort_values(field, na_position="last").drop_duplicates(
        subset=["m_S", "kappa"], keep="first")
    m_vals = np.sort(d["m_S"].unique())
    k_vals = np.sort(d["kappa"].unique())
    if len(m_vals) < 2 or len(k_vals) < 2:
        return None
    piv = d.pivot(index="kappa", columns="m_S", values=field)
    piv = piv.reindex(index=k_vals, columns=m_vals)
    Z = piv.to_numpy(dtype=float)
    if not dense or np.isfinite(Z).sum() < 3:
        return m_vals, k_vals, Z

    # dense upsample for smooth color contours; leave exterior NaN
    finite = np.isfinite(Z)
    pts = np.column_stack([np.repeat(m_vals[None, :], len(k_vals), axis=0)[finite],
                           np.repeat(k_vals[:, None], len(m_vals), axis=1)[finite]])
    vals = Z[finite]
    m_lin = np.linspace(m_vals.min(), m_vals.max(), max(60, 8 * len(m_vals)))
    k_lin = np.linspace(k_vals.min(), k_vals.max(), max(60, 8 * len(k_vals)))
    MM, KK = np.meshgrid(m_lin, k_lin)
    ZZ = griddata(pts, vals, (MM, KK), method="cubic")
    # cubic can leave holes / go negative — fall back to linear then nearest in-hull
    if ZZ is None or np.isnan(ZZ).all():
        ZZ = griddata(pts, vals, (MM, KK), method="linear")
    else:
        miss = np.isnan(ZZ)
        if miss.any():
            ZZl = griddata(pts, vals, (MM, KK), method="linear")
            ZZ[miss] = ZZl[miss]
    # clip tiny negatives from cubic ringing
    if ZZ is not None:
        ZZ = np.where(np.isfinite(ZZ), np.maximum(ZZ, 0.0), ZZ)
    return m_lin, k_lin, ZZ


def plot_nested_search(df, out_dir: str, dTn_max: float = 50.0) -> None:
    """Color contours in the (m_S, κ) plane + compact hit table."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)

    eps_label = ""
    if "eps" in df.columns and df["eps"].nunique(dropna=True) == 1:
        eps_label = rf" ($\varepsilon={df['eps'].dropna().iloc[0]:.0e}$)"

    # ========== primary: ΔT_n color contour in (m_S, κ) ==========
    grid = _pivot_field(df, "delta_T_n_S", dense=True)
    if grid is not None:
        m_vals, k_vals, Z = grid
        MM, KK = np.meshgrid(m_vals, k_vals)
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        Zplot = np.ma.masked_invalid(Z)
        vmax = float(np.nanmax(Zplot)) if np.isfinite(Zplot).any() else dTn_max
        vmin = float(np.nanmin(Zplot)) if np.isfinite(Zplot).any() else 0.0
        levels = np.linspace(max(0.0, vmin), max(vmax, dTn_max * 1.05), 20)
        ax.set_facecolor("#d0d0d0")  # no-two-step / masked
        cf = ax.contourf(MM, KK, Zplot, levels=levels, cmap="viridis_r", extend="max")
        if np.isfinite(Z).any() and vmin < dTn_max < vmax:
            cs = ax.contour(MM, KK, Zplot, levels=[dTn_max], colors="w", linewidths=2.2)
            ax.clabel(cs, fmt=rf"$\Delta T_n={dTn_max:g}$", fontsize=9, colors="w")
        dTn_pt = df["delta_T_n_S"].to_numpy(dtype=float)
        ok = np.isfinite(dTn_pt)
        ax.scatter(df.loc[~ok, "m_S"], df.loc[~ok, "kappa"], s=36, c="0.25",
                   marker="x", zorder=5, label=r"$0\to v$ / stuck")
        ax.scatter(df.loc[ok & (dTn_pt <= dTn_max), "m_S"],
                   df.loc[ok & (dTn_pt <= dTn_max), "kappa"],
                   s=40, c="lime", edgecolors="k", linewidths=0.5, zorder=5,
                   label=rf"$\Delta T_n\leq{dTn_max:g}$")
        ax.scatter(df.loc[ok & (dTn_pt > dTn_max), "m_S"],
                   df.loc[ok & (dTn_pt > dTn_max), "kappa"],
                   s=40, c="w", edgecolors="k", linewidths=0.7, zorder=5,
                   label=rf"$\Delta T_n>{dTn_max:g}$")
        cb = fig.colorbar(cf, ax=ax, pad=0.02)
        cb.set_label(r"$\Delta T_n = T_n(0\!\to\!S)-T_n(S\!\to\!v)$ [GeV]")
        ax.set(xlabel=r"$m_S$ [GeV]", ylabel=r"$\kappa$",
               title=r"$\Delta T_n$ contour in $(m_S,\kappa)$" + eps_label)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "delta_Tn_contour_mS_kappa.png"), dpi=170)
        plt.close(fig)

        # ========== nested / separated / 0→v map ==========
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        regime_code = np.full(len(df), 0.0)
        dTn_all = df["delta_T_n_S"].to_numpy(dtype=float)
        finite = np.isfinite(dTn_all)
        regime_code[finite & (dTn_all > dTn_max)] = 1.0
        regime_code[finite & (dTn_all > 0) & (dTn_all <= dTn_max)] = 2.0
        from scipy.interpolate import griddata as _gd
        pts = df[["m_S", "kappa"]].to_numpy(dtype=float)
        m_lin = np.linspace(df["m_S"].min(), df["m_S"].max(), 100)
        k_lin = np.linspace(df["kappa"].min(), df["kappa"].max(), 100)
        MMr, KKr = np.meshgrid(m_lin, k_lin)
        Zr = _gd(pts, regime_code, (MMr, KKr), method="nearest")
        cmap = mcolors.ListedColormap(["#bdbdbd", "#f4e6c8", "#2ca02c"])
        ax.contourf(MMr, KKr, Zr, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.95)
        if np.isfinite(Z).any() and vmin < dTn_max < vmax:
            ax.contour(MM, KK, Zplot, levels=[dTn_max], colors="k", linewidths=1.8)
            extra_lv = [10, 25, 75, 100, 150]
            extra_lv = [lv for lv in extra_lv
                        if np.nanmin(Z) < lv < np.nanmax(Z) and abs(lv - dTn_max) > 1]
            if extra_lv:
                cs2 = ax.contour(MM, KK, Zplot, levels=extra_lv, colors="0.25",
                                 linewidths=0.8, alpha=0.7)
                ax.clabel(cs2, fmt="%g", fontsize=8)
        ax.scatter(df["m_S"], df["kappa"], s=32, c="k", marker="o", alpha=0.75, zorder=4)
        ax.set(xlabel=r"$m_S$ [GeV]", ylabel=r"$\kappa$",
               title=rf"Nested window in $(m_S,\kappa)$" + eps_label)
        handles = [
            Line2D([0], [0], color="#2ca02c", lw=8,
                   label=rf"nested ($\Delta T_n\leq{dTn_max:g}$)"),
            Line2D([0], [0], color="#f4e6c8", lw=8,
                   label=rf"separated ($\Delta T_n>{dTn_max:g}$)"),
            Line2D([0], [0], color="#bdbdbd", lw=8, label=r"$0\to v$ / stuck"),
            Line2D([0], [0], color="k", lw=1.8, label=rf"$\Delta T_n={dTn_max:g}$"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "nested_window_mS_kappa.png"), dpi=170)
        plt.close(fig)

    # --- line plot ΔT_n vs m_S (secondary) ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    kappas = sorted(df["kappa"].dropna().unique())
    colors = {k: c for k, c in zip(kappas, plt.cm.tab10(np.linspace(0, 0.45, max(len(kappas), 1))))}
    for k in kappas:
        sub = df[df["kappa"] == k].sort_values("m_S")
        nested = sub["nested"].astype(bool) if "nested" in sub.columns else np.zeros(len(sub), dtype=bool)
        ax.plot(sub["m_S"], sub["delta_T_n_S"], "-", color=colors[k], lw=1.6,
                label=rf"$\kappa={k:g}$")
        ax.scatter(sub.loc[nested, "m_S"], sub.loc[nested, "delta_T_n_S"],
                   s=70, color=colors[k], marker="o", zorder=3, edgecolors="k", linewidths=0.6)
        ax.scatter(sub.loc[~nested, "m_S"], sub.loc[~nested, "delta_T_n_S"],
                   s=55, color=colors[k], marker="s", zorder=3, edgecolors="k", linewidths=0.6,
                   alpha=0.85)
    ax.axhline(dTn_max, color="0.35", ls="--", lw=1.2,
               label=rf"$\Delta T_n={dTn_max:g}$ GeV")
    ax.axhspan(0, dTn_max, color="C2", alpha=0.10, zorder=0)
    ax.set(xlabel=r"$m_S$ [GeV]", ylabel=r"$\Delta T_n$ [GeV]",
           title=r"$\Delta T_n$ vs $m_S$" + eps_label)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    extra = [
        Line2D([0], [0], marker="o", color="k", ls="None", ms=7, label="nested"),
        Line2D([0], [0], marker="s", color="k", ls="None", ms=7, label="separated"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + [e.get_label() for e in extra], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_Tn_vs_mS.png"), dpi=160)
    plt.close(fig)

    # --- compact hit table ---
    hit = df[df["nested"].astype(bool)].copy() if "nested" in df.columns else df.iloc[0:0]
    show_cols = [c for c in ("m_S", "kappa", "eps", "T_n_0_to_S", "T_n_S_to_v",
                             "delta_T_n_S", "T_n_0_to_v", "regime") if c in df.columns]
    tab = (hit if len(hit) else df).sort_values(["delta_T_n_S", "m_S", "kappa"])[show_cols]
    disp = tab.copy()
    for c in disp.columns:
        if c == "eps":
            disp[c] = disp[c].map(lambda x: f"{x:.0e}")
        elif c in ("m_S", "kappa"):
            disp[c] = disp[c].map(lambda x: f"{x:g}")
        elif np.issubdtype(disp[c].dtype, np.number):
            disp[c] = disp[c].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
    col_labels = {
        "m_S": r"$m_S$", "kappa": r"$\kappa$", "eps": r"$\varepsilon$",
        "T_n_0_to_S": r"$T_n(0\!\to\!S)$", "T_n_S_to_v": r"$T_n(S\!\to\!v)$",
        "delta_T_n_S": r"$\Delta T_n$", "T_n_0_to_v": r"$T_n(0\!\to\!v)$",
        "regime": "regime",
    }
    # cap table length for readability
    disp_show = disp.head(12)
    fig_h = 1.2 + 0.38 * max(len(disp_show), 1)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.axis("off")
    title = "Nested hits" if len(hit) else "Scan points (no nested flag)"
    ax.set_title(title + rf"  (top {len(disp_show)} by $\Delta T_n$)", fontsize=11, pad=8)
    table = ax.table(
        cellText=disp_show.values.tolist(),
        colLabels=[col_labels.get(c, c) for c in disp_show.columns],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
        elif len(hit):
            cell.set_facecolor("#eef8ee")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "nested_hits_table.png"), dpi=160)
    plt.close(fig)

    md_path = os.path.join(out_dir, "nested_hits.md")
    with open(md_path, "w") as fh:
        fh.write("# Nested-bubble scan hits\n\n")
        fh.write(f"Criterion: Regime II / nested sequence, or `0 < ΔT_n ≤ {dTn_max:g}` GeV.\n\n")
        fh.write(tab.to_markdown(index=False))
        fh.write("\n")


def main():
    global _LOG
    p = argparse.ArgumentParser(description="Portal nested-bubble parameter search")
    p.add_argument("--V0", type=float, default=SET8_V0)
    p.add_argument("--v", type=float, default=SET8_V)
    p.add_argument("--m_S", type=float, action="append", default=None,
                   help="S tree mass [GeV]; repeatable (default: band around ~500 GeV)")
    p.add_argument("--kappa", type=float, action="append", default=None)
    p.add_argument("--eps", type=float, action="append", default=None,
                   help="epsilon values; repeatable (default: 1e-20, 1e-18)")
    p.add_argument("--dTn_max", type=float, default=50.0,
                   help="flag nested_by_dTn if 0 < ΔT_n <= this (GeV)")
    p.add_argument("--g_phi", type=float, default=None)
    p.add_argument("--g_S", type=float, default=None)
    p.add_argument("--boson_phi", action="append", default=None)
    p.add_argument("--fermion_phi", action="append", default=None)
    p.add_argument("--boson_S", action="append", default=None)
    p.add_argument("--g_star", type=float, default=100.0)
    p.add_argument("--phi_thermal", choices=("paper", "set8"), default="set8")
    p.add_argument("--nT", type=int, default=28,
                   help="T grid size; with default Tmin~0.4*min(T0) this densifies the nucleation window")
    p.add_argument("--refine", type=int, default=10)
    p.add_argument("--a2v_n_fine", type=int, default=40)
    p.add_argument("--a2v_T_hi_fac", type=float, default=1.25)
    p.add_argument("--a2v_T_lo_fac", type=float, default=0.85)
    p.add_argument("--a2v_S_trigger", type=float, default=2000.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--Tmin", type=float, default=None,
                   help="default: 0.4 * min(T0_S, T0_phi) — avoid ultra-low T that sparsifies the grid")
    p.add_argument("--Tmax", type=float, default=None)
    p.add_argument("--out", default="data/portal/nested_search")
    p.add_argument("--replot", action="store_true",
                   help="only redraw figures from existing nested_search.csv / .json")
    args_ns = p.parse_args()

    if args_ns.replot:
        import pandas as pd
        csv_path = os.path.join(args_ns.out, "nested_search.csv")
        json_path = os.path.join(args_ns.out, "nested_search.json")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
        elif os.path.isfile(json_path):
            with open(json_path) as fh:
                df = pd.DataFrame(json.load(fh))
        else:
            raise SystemExit(f"No nested_search.csv/.json under {args_ns.out}")
        plot_nested_search(df, args_ns.out, dTn_max=args_ns.dTn_max)
        print(f"Replotted → {args_ns.out}")
        return

    if args_ns.g_phi is not None and args_ns.boson_phi is None:
        args_ns.boson_phi = [f"20,{args_ns.g_phi}"]
        if args_ns.fermion_phi is None:
            args_ns.fermion_phi = [f"20,{args_ns.g_phi}"]
    if args_ns.g_S is not None and args_ns.boson_S is None:
        args_ns.boson_S = [f"6,{args_ns.g_S}"]
    boson_phi = _parse_ng(args_ns.boson_phi, SET8_BOSON_PHI)
    fermion_phi = _parse_ng(args_ns.fermion_phi, SET8_FERMION_PHI)
    boson_S = _parse_ng(args_ns.boson_S, [(6.0, 1.0)])

    m_Ss = args_ns.m_S or [460.0, 480.0, 500.0, 520.0, 550.0, 600.0]
    kappas = args_ns.kappa or [0.05, 0.1, 0.2]
    epss = args_ns.eps or [1e-20, 1e-18]

    os.makedirs(args_ns.out, exist_ok=True)
    log_path = os.path.join(args_ns.out, "progress.log")
    _LOG = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)

    log(f"portalNestedSearch → {args_ns.out}")
    log(f"m_S={m_Ss}  kappa={kappas}  eps={epss}")
    log(f"nested if Regime II / nested in sequence, or 0 < ΔT_n <= {args_ns.dTn_max:g} GeV")
    log(f"phi_thermal={args_ns.phi_thermal}  workers={args_ns.workers}")

    jobs = [(float(e), float(k), float(m)) for m in m_Ss for k in kappas for e in epss]
    rows = []
    t_all = time.time()
    for i, (eps, kappa, m_S) in enumerate(jobs, 1):
        # Densify around T0: default Tmin in run_one is 0.02*T0 (too low → sparse near Tn).
        from analysis.portalEpsScan import s0_from_mS, build_pot
        pot_tmp = build_pot(args_ns.V0, args_ns.v, s0_from_mS(eps, args_ns.V0, m_S),
                            eps, kappa, boson_phi, fermion_phi, boson_S, args_ns.g_star,
                            phi_thermal=args_ns.phi_thermal)
        T0lo = min(pot_tmp.T0_S(), pot_tmp.T0_phi())
        T0hi = max(pot_tmp.T0_S(), pot_tmp.T0_phi())
        Tmin = args_ns.Tmin if args_ns.Tmin is not None else max(50.0, 0.4 * T0lo)
        Tmax = args_ns.Tmax if args_ns.Tmax is not None else 1.35 * T0hi
        args = SimpleNamespace(
            V0=args_ns.V0, v=args_ns.v, m_S=m_S, s0=None,
            boson_phi=boson_phi, fermion_phi=fermion_phi, boson_S=boson_S,
            g_star=args_ns.g_star, phi_thermal=args_ns.phi_thermal,
            Tmin=Tmin, Tmax=Tmax, nT=args_ns.nT,
            refine=args_ns.refine, workers=args_ns.workers,
            a2v_n_fine=args_ns.a2v_n_fine, a2v_T_hi_fac=args_ns.a2v_T_hi_fac,
            a2v_T_lo_fac=args_ns.a2v_T_lo_fac, a2v_S_trigger=args_ns.a2v_S_trigger,
        )
        try:
            row = run_one(eps, kappa, args, i, len(jobs))
        except Exception as exc:
            log(f"FAILED m_S={m_S} kappa={kappa} eps={eps:.3e}: {exc}")
            row = {
                "eps": eps, "kappa": kappa, "m_S": m_S, "regime": "error",
                "sequence": f"ERROR: {exc}",
                "T_n_0_to_S": float("nan"), "T_n_S_to_v": float("nan"),
                "T_n_0_to_v": float("nan"), "delta_T_n_S": float("nan"),
            }
        dTn = row.get("delta_T_n_S", float("nan"))
        seq = str(row.get("sequence", ""))
        nested_seq = ("nested" in seq.lower()) or ("Regime II" in seq)
        nested_dTn = (isinstance(dTn, (int, float)) and math.isfinite(dTn)
                      and 0.0 < dTn <= args_ns.dTn_max)
        row["nested_by_sequence"] = bool(nested_seq)
        row["nested_by_dTn"] = bool(nested_dTn)
        row["nested"] = bool(nested_seq or nested_dTn)
        rows.append(row)

        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(args_ns.out, "nested_search.csv"), index=False)
        flag = "NESTED" if row["nested"] else row.get("regime", "")
        log(f"  → flag={flag}  ΔT_n={dTn}  nested_seq={nested_seq} nested_dTn={nested_dTn}")

    import pandas as pd
    df = pd.DataFrame(rows)
    front = [c for c in ("eps", "kappa", "m_S", "regime", "nested", "nested_by_dTn",
                         "nested_by_sequence", "T_n_0_to_S", "T_n_S_to_v", "delta_T_n_S",
                         "T_n_0_to_v", "T0_S", "T0_phi", "sequence")
             if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    csv_path = os.path.join(args_ns.out, "nested_search.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(args_ns.out, "nested_search.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)

    log(f"\nDone in {time.time() - t_all:.1f}s → {csv_path}")
    hit = df[df.get("nested", False) == True] if "nested" in df.columns else df.iloc[0:0]
    log(f"NESTED hits: {len(hit)} / {len(df)}")
    show = [c for c in ("eps", "kappa", "m_S", "regime", "T_n_0_to_S", "T_n_S_to_v",
                        "delta_T_n_S", "T_n_0_to_v", "nested", "sequence")
            if c in df.columns]
    if len(hit):
        log("\n" + hit[show].to_string(index=False))
    else:
        # still print closest ΔT_n
        d = df[np.isfinite(df.get("delta_T_n_S", np.nan))] if "delta_T_n_S" in df.columns else df
        if len(d) and "delta_T_n_S" in d.columns:
            log("\nNo nested hit. Closest ΔT_n:")
            log(d.nsmallest(min(10, len(d)), "delta_T_n_S")[show].to_string(index=False))
        log("\nFull table:")
        log(df[show].to_string(index=False))

    plot_nested_search(df, args_ns.out, dTn_max=args_ns.dTn_max)
    log(f"Figures → {args_ns.out}/{{delta_Tn_vs_mS,Tn_vs_mS,nested_hits_table}}.png")
    _LOG.close()


if __name__ == "__main__":
    main()
