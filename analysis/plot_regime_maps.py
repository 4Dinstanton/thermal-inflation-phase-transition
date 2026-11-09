#!/usr/bin/env python3
"""Regime-division and Sec. II stuck-exclusion maps from portal GW scans.

Which plane to look at
----------------------
The scan has (ε, κ, m_S, φ_off).  Empirically the useful 2D slices are:

  1. (m_S, φ_off)  — **mixed scan primary**.  Any φ_off>0 tends to stick in A;
                     unmixed (φ_off=0) hosts separated two-step.
  2. (m_S, κ)      — **nested / Sec. III primary** (paper Fig. 4 style).
                     At ε≪1 and low m_S this shows nested vs separated vs stuck.
  3. (ε, m_S)      — **Sec. II stuck-exclusion**.  At φ_off=0, shows which
                     (ε, m_S) never complete A→B (stuck in A).
  4. (ε, φ_off)    — mixing × latent-heat; secondary for mixed potential.

Examples
--------
    # after mixed scan finishes
    PYTHONPATH=. python analysis/plot_regime_maps.py \\
        --csv data/portal/mixed_scan/mixed_scan.csv \\
        --out data/portal/regime_maps/mixed

    # nested contour + large-ε Sec. II exclusions
    PYTHONPATH=. python analysis/plot_regime_maps.py \\
        --csv data/portal/nested_contour/nested_search.csv \\
        --csv data/portal/eps_scan_large_eps/eps_scan.csv \\
        --out data/portal/regime_maps
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# discrete regime codes for colour maps
REGIME_ORDER = ("stuck in A", "two-step separated", "two-step nested", "0->v wins")
REGIME_COLORS = {
    "stuck in A": "#c44e52",          # red — excluded / incomplete exit
    "two-step separated": "#4c72b0",  # blue — Sec. II sequential
    "two-step nested": "#55a868",     # green — Sec. III proximate
    "0->v wins": "#837b7b",           # grey — no intermediate A
}
REGIME_LABELS = {
    "stuck in A": r"stuck in $A$ (excluded)",
    "two-step separated": r"two-step separated",
    "two-step nested": r"two-step nested",
    "0->v wins": r"$0\to v$ wins",
}


def normalize_regime(s) -> str:
    t = str(s).strip().lower()
    if "stuck" in t:
        return "stuck in A"
    if "nested" in t or "regime ii" in t:
        return "two-step nested"
    if "separat" in t or "regime i" in t or "two-step" in t:
        return "two-step separated"
    if "0->v" in t or "0→v" in t or "wins" in t or "single" in t or "never" in t:
        return "0->v wins"
    if "error" in t:
        return "0->v wins"
    return "0->v wins"


def _phi_off_col(df: pd.DataFrame) -> str | None:
    for c in ("phi_off_target", "phi_off_tgt", "phi_off"):
        if c in df.columns:
            return c
    return None


def load_scan(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "regime" not in df.columns:
        raise SystemExit(f"{path}: no 'regime' column")
    df = df.copy()
    df["regime_n"] = df["regime"].map(normalize_regime)
    # tidy float noise on m_S
    if "m_S" in df.columns:
        df["m_S"] = pd.to_numeric(df["m_S"], errors="coerce").round(6)
    if "eps" in df.columns:
        df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    if "kappa" in df.columns:
        df["kappa"] = pd.to_numeric(df["kappa"], errors="coerce")
    po = _phi_off_col(df)
    if po and po != "phi_off_target":
        df["phi_off_target"] = df[po]
    return df


def _legend_handles(present: set[str]):
    from matplotlib.lines import Line2D
    handles = []
    for key in REGIME_ORDER:
        if key in present:
            handles.append(Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=REGIME_COLORS[key], markeredgecolor="k",
                markersize=9, label=REGIME_LABELS[key]))
    return handles


def _scatter_regime(ax, x, y, regimes, *, logx=False, logy=False,
                    xlabel="", ylabel="", title=""):
    present = set()
    for key in REGIME_ORDER:
        m = regimes == key
        if not m.any():
            continue
        present.add(key)
        ax.scatter(x[m], y[m], s=55, c=REGIME_COLORS[key],
                   edgecolors="k", linewidths=0.4, zorder=3, label=REGIME_LABELS[key])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(handles=_legend_handles(present), fontsize=8, loc="best",
              framealpha=0.92)
    ax.grid(alpha=0.25, which="both")
    return present


def _fill_nearest(ax, x, y, codes, *, logx=False, nx=80, ny=80):
    """Background nearest-neighbour fill (optional context under scatter)."""
    from matplotlib import colors as mcolors
    from scipy.interpolate import griddata

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    codes = np.asarray(codes, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(codes)
    if ok.sum() < 3:
        return
    xx, yy = x[ok], y[ok]
    pts = np.column_stack([np.log10(xx) if logx else xx, yy])
    if logx:
        xs = np.linspace(np.log10(xx.min()), np.log10(xx.max()), nx)
    else:
        xs = np.linspace(xx.min(), xx.max(), nx)
    ys = np.linspace(yy.min(), yy.max(), ny)
    XX, YY = np.meshgrid(xs, ys)
    Z = griddata(pts, codes[ok], (XX, YY), method="nearest")
    cmap = mcolors.ListedColormap([REGIME_COLORS[k] for k in REGIME_ORDER])
    levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
    xp = 10 ** XX if logx else XX
    ax.contourf(xp, YY, Z, levels=levels, cmap=cmap, alpha=0.35, zorder=1)


def regime_code(series: pd.Series) -> np.ndarray:
    idx = {k: i for i, k in enumerate(REGIME_ORDER)}
    return series.map(lambda r: idx.get(r, 3)).to_numpy(dtype=float)


def plot_plane(df: pd.DataFrame, xcol: str, ycol: str, path: str,
               *, title: str, logx: bool = False, logy: bool = False,
               fill: bool = True, xlabel: str | None = None,
               ylabel: str | None = None) -> str | None:
    if xcol not in df.columns or ycol not in df.columns:
        return None
    sub = df.dropna(subset=[xcol, ycol, "regime_n"]).copy()
    if len(sub) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    if fill:
        _fill_nearest(ax, sub[xcol], sub[ycol], regime_code(sub["regime_n"]),
                      logx=logx)
    _scatter_regime(
        ax, sub[xcol].to_numpy(), sub[ycol].to_numpy(),
        sub["regime_n"].to_numpy(),
        logx=logx, logy=logy,
        xlabel=xlabel or xcol, ylabel=ylabel or ycol, title=title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_stuck_exclusion(df: pd.DataFrame, xcol: str, ycol: str, path: str,
                         *, title: str, logx: bool = False,
                         xlabel: str | None = None,
                         ylabel: str | None = None) -> str | None:
    """Sec. II style: highlight stuck-in-A as excluded; other regimes as viable."""
    if xcol not in df.columns or ycol not in df.columns:
        return None
    sub = df.dropna(subset=[xcol, ycol, "regime_n"]).copy()
    if len(sub) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    stuck = sub["regime_n"] == "stuck in A"
    viable = ~stuck
    # background: stuck = excluded hatch
    if stuck.any():
        ax.scatter(sub.loc[stuck, xcol], sub.loc[stuck, ycol],
                   s=120, marker="X", c="#c44e52", edgecolors="k",
                   linewidths=0.5, zorder=3, label=r"stuck in $A$ (excluded)")
    # viable two-step
    for key, mk in (("two-step separated", "o"), ("two-step nested", "s"),
                    ("0->v wins", "^")):
        m = viable & (sub["regime_n"] == key)
        if not m.any():
            continue
        ax.scatter(sub.loc[m, xcol], sub.loc[m, ycol],
                   s=70, marker=mk, c=REGIME_COLORS[key], edgecolors="k",
                   linewidths=0.4, zorder=4, label=REGIME_LABELS[key])
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel or xcol, ylabel=ylabel or ycol, title=title)
    ax.grid(alpha=0.25, which="both")
    # force legend order
    handles = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#c44e52",
               markeredgecolor="k", markersize=10,
               label=r"stuck in $A$ — Sec.~II excluded"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=REGIME_COLORS["two-step separated"],
               markeredgecolor="k", markersize=9, label=r"two-step separated (viable)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=REGIME_COLORS["two-step nested"],
               markeredgecolor="k", markersize=9, label=r"two-step nested (viable)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=REGIME_COLORS["0->v wins"],
               markeredgecolor="k", markersize=9, label=r"$0\to v$ wins"),
    ]
    # only keep present
    present_labels = {h.get_label() for h in ax.get_legend_handles_labels()[0]}
    # simpler: rebuild from data
    show = []
    if stuck.any():
        show.append(handles[0])
    if (sub["regime_n"] == "two-step separated").any():
        show.append(handles[1])
    if (sub["regime_n"] == "two-step nested").any():
        show.append(handles[2])
    if (sub["regime_n"] == "0->v wins").any():
        show.append(handles[3])
    ax.legend(handles=show, fontsize=8, loc="best", framealpha=0.92)
    # annotation
    n_st = int(stuck.sum())
    n_ok = int((sub["regime_n"].isin(
        ["two-step separated", "two-step nested"])).sum())
    ax.text(0.02, 0.02,
            rf"excluded {n_st}/{len(sub)}  ·  two-step exit {n_ok}/{len(sub)}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.85))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_summary_bars(df: pd.DataFrame, path: str, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df["regime_n"].value_counts().reindex(REGIME_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    cols = [REGIME_COLORS[k] for k in counts.index]
    ax.bar(range(len(counts)), counts.values, color=cols, edgecolor="k")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([REGIME_LABELS[k] for k in counts.index],
                       rotation=15, ha="right")
    ax.set_ylabel("scan points")
    ax.set_title(title or "Regime counts")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, f"{int(v)}", ha="center", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def recommend_planes(df: pd.DataFrame) -> list[tuple]:
    """Return list of (xcol, ycol, kwargs, tag) to plot."""
    planes = []
    has = set(df.columns)
    nuniq = {c: df[c].nunique(dropna=True) for c in has if c in df.columns}

    def vary(c, min_n=2):
        return c in has and nuniq.get(c, 0) >= min_n

    # 1. mixed primary
    if vary("m_S") and vary("phi_off_target"):
        planes.append(("m_S", "phi_off_target",
                       dict(title=r"Regime map $(m_S,\phi_{\rm off})$ — mixed primary",
                            xlabel=r"$m_S$ [GeV]",
                            ylabel=r"$\phi_{\rm off}^{\rm tgt}$ [GeV]"),
                       "mS_phi_off"))
    # 2. nested primary
    if vary("m_S") and vary("kappa"):
        planes.append(("m_S", "kappa",
                       dict(title=r"Regime map $(m_S,\kappa)$ — nested / Sec.~III",
                            xlabel=r"$m_S$ [GeV]", ylabel=r"$\kappa$"),
                       "mS_kappa"))
    # 3. Sec II (ε, m_S)
    if vary("eps") and vary("m_S"):
        planes.append(("eps", "m_S",
                       dict(title=r"Regime map $(\varepsilon,m_S)$ — Sec.~II",
                            xlabel=r"$\varepsilon$", ylabel=r"$m_S$ [GeV]",
                            logx=True),
                       "eps_mS"))
    # 4. (ε, φ_off)
    if vary("eps") and vary("phi_off_target"):
        planes.append(("eps", "phi_off_target",
                       dict(title=r"Regime map $(\varepsilon,\phi_{\rm off})$",
                            xlabel=r"$\varepsilon$",
                            ylabel=r"$\phi_{\rm off}^{\rm tgt}$ [GeV]",
                            logx=True),
                       "eps_phi_off"))
    # 5. (ε, κ)
    if vary("eps") and vary("kappa"):
        planes.append(("eps", "kappa",
                       dict(title=r"Regime map $(\varepsilon,\kappa)$",
                            xlabel=r"$\varepsilon$", ylabel=r"$\kappa$",
                            logx=True),
                       "eps_kappa"))
    # 6. (κ, φ_off)
    if vary("kappa") and vary("phi_off_target"):
        planes.append(("kappa", "phi_off_target",
                       dict(title=r"Regime map $(\kappa,\phi_{\rm off})$",
                            xlabel=r"$\kappa$",
                            ylabel=r"$\phi_{\rm off}^{\rm tgt}$ [GeV]"),
                       "kappa_phi_off"))
    return planes


def run_one_csv(csv_path: str, out_dir: str, tag: str | None = None) -> list[str]:
    df = load_scan(csv_path)
    base = tag or os.path.splitext(os.path.basename(csv_path))[0]
    out = os.path.join(out_dir, base)
    os.makedirs(out, exist_ok=True)
    written = []

    # summary counts
    p = plot_summary_bars(df, os.path.join(out, "regime_counts.png"),
                          title=f"Regimes — {base}  (N={len(df)})")
    written.append(p)
    print(f"[{base}] regimes:\n{df['regime_n'].value_counts().to_string()}")

    for xcol, ycol, kw, plane_tag in recommend_planes(df):
        path = os.path.join(out, f"regime_{plane_tag}.png")
        w = plot_plane(df, xcol, ycol, path, **kw)
        if w:
            written.append(w)
            print(f"  wrote {w}")

        # Sec. II exclusion twin for planes involving stuck physics
        if plane_tag in ("eps_mS", "mS_phi_off", "eps_phi_off", "mS_kappa",
                         "eps_kappa", "kappa_phi_off"):
            path2 = os.path.join(out, f"secII_excluded_{plane_tag}.png")
            title = kw.get("title", "").replace("Regime map", "Sec.~II exclusion")
            w2 = plot_stuck_exclusion(
                df, xcol, ycol, path2,
                title=title,
                logx=kw.get("logx", False),
                xlabel=kw.get("xlabel"), ylabel=kw.get("ylabel"))
            if w2:
                written.append(w2)
                print(f"  wrote {w2}")

    # φ_off=0 slice of (ε, m_S) if mix present — clean Sec. II view
    if "phi_off_target" in df.columns and df["phi_off_target"].nunique() > 1:
        unmixed = df[df["phi_off_target"].fillna(-1) == 0]
        if len(unmixed) >= 4 and unmixed["eps"].nunique() >= 2 and unmixed["m_S"].nunique() >= 2:
            path = os.path.join(out, "secII_excluded_eps_mS_unmixed.png")
            w = plot_stuck_exclusion(
                unmixed, "eps", "m_S", path,
                title=r"Sec.~II exclusion $(\varepsilon,m_S)$ at $\phi_{\rm off}=0$",
                logx=True, xlabel=r"$\varepsilon$", ylabel=r"$m_S$ [GeV]")
            if w:
                written.append(w)
                print(f"  wrote {w}")
            path = os.path.join(out, "regime_eps_mS_unmixed.png")
            w = plot_plane(
                unmixed, "eps", "m_S", path,
                title=r"Regime map $(\varepsilon,m_S)$ at $\phi_{\rm off}=0$",
                logx=True, xlabel=r"$\varepsilon$", ylabel=r"$m_S$ [GeV]")
            if w:
                written.append(w)
                print(f"  wrote {w}")

    # write a short README for the user
    readme = os.path.join(out, "README_planes.txt")
    with open(readme, "w") as fh:
        fh.write(f"Source: {csv_path}\nN={len(df)}\n\n")
        fh.write("Recommended viewing order\n")
        fh.write("-------------------------\n")
        fh.write("1. regime_mS_phi_off.png     — if mixing scanned (dominant divider)\n")
        fh.write("2. regime_mS_kappa.png       — nested vs separated (Sec. III)\n")
        fh.write("3. secII_excluded_eps_mS*.png — parameters excluded by stuck-in-A\n")
        fh.write("4. regime_counts.png         — overall fractions\n")
    written.append(readme)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description="Regime maps + Sec. II stuck exclusions")
    p.add_argument("--csv", action="append", required=True,
                   help="scan CSV (repeatable)")
    p.add_argument("--out", default="data/portal/regime_maps")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    all_w = []
    for csv in a.csv:
        if not os.path.isfile(csv):
            print(f"skip missing {csv}")
            continue
        all_w.extend(run_one_csv(csv, a.out))
    print(f"\n{len(all_w)} files → {a.out}/")


if __name__ == "__main__":
    main()
