#!/usr/bin/env python3
"""Contour of Delta T_n = T_n(O->A) - T_n(A->B) in the (m_S, mu_mix^2) plane (mixed model).

Companion of Fig. 4 (delta_Tn_contour_mS_kappa.png, mu_mix = 0).  Reads the
mixed scan CSV (analysis/run_mixed_portal_scan.py output), marks
  * the nested band |Delta T_n| <= dTn_max,
  * the immediate region (A->B at T_n(O->A): Delta T_n = 0, roll-over),
  * points where the direct channel O->B wins,
  * roll-over points (no beta_2).

    python analysis/plot_mixed_contour.py --csv data/portal/mixed_contour_mS_mu/mixed_scan.csv \
        --out figs/delta_Tn_contour_mS_mu.png --paper "<paper figs dir>"
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def fl(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/portal/mixed_contour_mS_mu/mixed_scan.csv')
    ap.add_argument('--dTn_max', type=float, default=3.0)
    ap.add_argument('--out', default='figs/delta_Tn_contour_mS_mu.png')
    ap.add_argument('--paper', default='')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    mS = np.array([fl(r, 'm_S') for r in rows]); mu2 = np.array([fl(r, 'mu2_mix') for r in rows])
    tn1 = np.array([fl(r, 'T_n_0_to_S') for r in rows]); tn2 = np.array([fl(r, 'T_n_S_to_v') for r in rows])
    tn0 = np.array([fl(r, 'T_n_0_to_v') for r in rows]); b2 = np.array([fl(r, 'betaH_step2') for r in rows])
    reg = np.array([r.get('regime', '') for r in rows]); seq = np.array([r.get('sequence', '') for r in rows])
    direct = np.array(['0->v' in s or 'single step' in s for s in seq])
    two = np.isfinite(tn1) & np.isfinite(tn2) & ~direct
    dT = np.where(two, tn1 - tn2, np.nan)
    imm = two & (np.abs(dT) < 1e-6)
    roll = two & ~np.isfinite(b2) & ~imm
    nested = two & (np.abs(dT) <= a.dTn_max)
    print(f"{len(rows)} points: two-step {two.sum()}, immediate {imm.sum()}, roll-over(tunnel-free) {roll.sum()}, nested(|dT|<= {a.dTn_max}) {nested.sum()}, direct wins {direct.sum()}")

    # log(mu2) axis with mu2 = 0 placed at a small value
    mu_plot = np.where(mu2 > 0, mu2, np.nanmin(mu2[mu2 > 0]) / 4 if (mu2 > 0).any() else 1e-2)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ok = two & np.isfinite(dT)
    if ok.sum() >= 4 and len(set(mS[ok])) >= 2 and len(set(mu_plot[ok])) >= 2:
      try:
          xi = np.linspace(mS[ok].min(), mS[ok].max(), 120)
          yi = np.geomspace(mu_plot[ok].min(), mu_plot[ok].max(), 120)
          XI, YI = np.meshgrid(xi, yi)
          ZI = griddata((mS[ok], np.log10(mu_plot[ok])), dT[ok], (XI, np.log10(YI)), method='linear')
          lv = np.linspace(0, np.nanmax(ZI), 21)
          cf = ax.contourf(XI, YI, np.clip(ZI, 0, None), levels=lv, cmap='viridis')
          cb = fig.colorbar(cf, ax=ax); cb.set_label(r'$\Delta T_n = T_n(O\to A)-T_n(A\to B)$ [GeV]')
          try:
              cs = ax.contour(XI, YI, ZI, levels=[a.dTn_max], colors='w', linewidths=2.0)
              ax.clabel(cs, fmt=rf'$\Delta T_n={a.dTn_max:g}$ GeV', fontsize=8, colors='w')
          except Exception:
              pass
      except Exception as e:
        print('contour skipped:', e)
    sel = two & ~imm & ~nested
    if sel.any():
        ax.scatter(mS[sel], mu_plot[sel], s=30, facecolors='white', edgecolors='k', zorder=5, label=r'two-step, $\Delta T_n>3$ GeV')
    if (nested & ~imm).any():
        ax.scatter(mS[nested & ~imm], mu_plot[nested & ~imm], s=50, c='lime', edgecolors='k', zorder=6, label=r'nested: $0<\Delta T_n\leq 3$ GeV')
    if imm.any():
        ax.scatter(mS[imm], mu_plot[imm], s=70, marker='*', c='red', edgecolors='k', zorder=7, label=r'immediate ($A\to B$ at $T_n(O\to A)$)')
    if direct.any():
        ax.scatter(mS[direct], mu_plot[direct], s=40, marker='x', c='0.3', zorder=5, label=r'direct $O\to B$ first')
    if roll.any():
        ax.scatter(mS[roll], mu_plot[roll], s=110, facecolors='none', edgecolors='C3', linewidths=1.4, zorder=6, label=r"$A\to B$ by classical roll-over (no $\beta_2$)")
    ax.set_yscale('log'); ax.set_xlabel(r'$m_S$ [GeV]'); ax.set_ylabel(r'$\mu_{\rm mix}^2$ [GeV$^2$]  (bottom row: $\mu_{\rm mix}=0$)')
    ax.set_ylim(np.nanmin(mu_plot) / 1.6, np.nanmax(mu_plot) * 1.3)
    ax.set_title(r'$\kappa=0.1$, $\varepsilon=10^{-12}$', fontsize=10)
    ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=200); print('wrote', a.out)
    if a.paper:
        shutil.copy(a.out, a.paper); print('copied to', a.paper)
    # table
    for r_mS in sorted(set(mS)):
        sel = mS == r_mS
        print(f"m_S={r_mS:.0f}: " + "  ".join(f"mu2={m:.2g}:{'imm' if i else ('dir' if d else (f'{x:.0f}' if np.isfinite(x) else 'NaN'))}" for m, x, i, d in sorted(zip(mu2[sel], dT[sel], imm[sel], direct[sel]))))


if __name__ == '__main__':
    main()
