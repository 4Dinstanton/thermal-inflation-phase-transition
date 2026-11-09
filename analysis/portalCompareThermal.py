#!/usr/bin/env python3
"""Compare paper vs set8 portal eps-scan GW / nucleation results.

Example
-------
    PYTHONPATH=. python analysis/portalCompareThermal.py \\
        --paper data/portal/eps_scan --set8 data/portal/eps_scan_set8 \\
        --out data/portal/compare_thermal
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _load(path: str) -> pd.DataFrame:
    csv = os.path.join(path, "eps_scan.csv") if os.path.isdir(path) else path
    return pd.read_csv(csv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", default="data/portal/eps_scan")
    p.add_argument("--set8", default="data/portal/eps_scan_set8")
    p.add_argument("--out", default="data/portal/compare_thermal")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df_p = _load(args.paper)
    df_s = _load(args.set8)
    df_p["thermal"] = "paper"
    df_s["thermal"] = "set8"
    # overlapping kappa for side-by-side
    k_common = sorted(set(df_p["kappa"]) & set(df_s["kappa"]))
    if not k_common:
        k_common = sorted(set(df_s["kappa"]))  # plot set8 alone + paper all

    # ---- T_n comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, df, title in (
        (axes[0], df_p, r"paper ($m=g|\phi|$)"),
        (axes[1], df_s, r"set8 ($V_{\rm correct}$)"),
    ):
        for k in sorted(df["kappa"].unique()):
            rs = df[df["kappa"] == k].sort_values("eps")
            ax.loglog(rs["eps"], rs["T_n_0_to_v"], "^--", ms=6, label=rf"$\kappa={k:g}$: $0\to v$")
            ax.loglog(rs["eps"], rs["T_n_0_to_S"], "o-", ms=5, label=rf"$\kappa={k:g}$: $0\to S$")
            ax.loglog(rs["eps"], rs["T_n_S_to_v"], "s-", ms=5, alpha=0.85,
                      label=rf"$\kappa={k:g}$: $S\to v$")
        ax.set(xlabel=r"$\epsilon$", ylabel=r"$T_n$ [GeV]", title=title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=1)
    fig.suptitle(r"Nucleation temperatures: paper vs set8 $\phi$ thermal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "Tn_paper_vs_set8.png"), dpi=150)
    plt.close(fig)

    # ---- peak GW amplitude
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for df, ls, name in ((df_p, "--", "paper"), (df_s, "-", "set8")):
        for k in sorted(df["kappa"].unique()):
            rs = df[df["kappa"] == k].sort_values("eps")
            ax.loglog(rs["eps"], rs["h2Omega_peak_tot"], ls, marker="o", ms=5,
                      label=rf"{name} $\kappa={k:g}$")
    ax.set(xlabel=r"$\epsilon$", ylabel=r"$h^2\Omega_{\rm GW}^{\rm peak}$",
           title="Peak GW amplitude: paper vs set8")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "h2Omega_paper_vs_set8.png"), dpi=150)
    plt.close(fig)

    # ---- sensitivity ratios for common kappa
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, det in zip(axes, ("LISA", "DECIGO", "BBO")):
        for df, ls, name in ((df_p, "--", "paper"), (df_s, "-", "set8")):
            for k in sorted(df["kappa"].unique()):
                rs = df[df["kappa"] == k].sort_values("eps")
                ax.loglog(rs["eps"], rs[f"max_ratio_{det}"], ls, marker="o", ms=4,
                          label=rf"{name} $\kappa={k:g}$")
        ax.axhline(1.0, color="k", lw=1)
        ax.set(xlabel=r"$\epsilon$", title=det,
               ylabel=r"$\max_f\,\Omega/\Omega_{\rm sens}$" if det == "LISA" else "")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
    fig.suptitle("Spectrum vs detector floor ( >1 ⇒ observable )")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sensitivity_paper_vs_set8.png"), dpi=150)
    plt.close(fig)

    # ---- summary table at overlapping (eps, kappa)
    rows = []
    for k in k_common:
        for eps in sorted(set(df_p[df_p.kappa == k].eps) & set(df_s[df_s.kappa == k].eps)):
            rp = df_p[(df_p.kappa == k) & (np.isclose(df_p.eps, eps))].iloc[0]
            rs = df_s[(df_s.kappa == k) & (np.isclose(df_s.eps, eps))].iloc[0]
            rows.append({
                "eps": eps, "kappa": k,
                "regime_paper": rp["regime"], "regime_set8": rs["regime"],
                "Tn_0v_paper": rp["T_n_0_to_v"], "Tn_0v_set8": rs["T_n_0_to_v"],
                "Tn_0S_paper": rp["T_n_0_to_S"], "Tn_0S_set8": rs["T_n_0_to_S"],
                "Tn_Sv_paper": rp["T_n_S_to_v"], "Tn_Sv_set8": rs["T_n_S_to_v"],
                "h2Om_paper": rp["h2Omega_peak_tot"], "h2Om_set8": rs["h2Omega_peak_tot"],
                "LISA_paper": rp["meets_LISA"], "LISA_set8": rs["meets_LISA"],
                "DECIGO_paper": rp["meets_DECIGO"], "DECIGO_set8": rs["meets_DECIGO"],
                "BBO_paper": rp["meets_BBO"], "BBO_set8": rs["meets_BBO"],
            })
    out_csv = os.path.join(args.out, "compare_summary.csv")
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({len(rows)} overlapping points)")
    else:
        print("No overlapping (eps,kappa); wrote plots only.")
    print(f"Plots → {args.out}")
    # print quick headline numbers
    print("\n--- headline T_n(0→v) ---")
    print(f"  paper mean: {df_p['T_n_0_to_v'].mean():.4g}")
    print(f"  set8  mean: {df_s['T_n_0_to_v'].mean():.4g}")
    print("\n--- regimes set8 ---")
    print(df_s.groupby(["kappa", "regime"]).size())


if __name__ == "__main__":
    main()
