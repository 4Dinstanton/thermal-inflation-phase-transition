#!/usr/bin/env python3
"""Dependence of the nucleation parameters on m_S and mu_mix (mixed portal scan).

Reads data/portal/mixed_scan/mixed_scan.csv and data/portal/fcomp_benchmark/{hunt.csv,
benchmark.json}; writes figs/mu_dependence.png and prints LaTeX table rows.

    python analysis/plot_mu_dependence.py --paper "paper/twoStepTIPT/.../figs"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fl(r, k):
    v = r.get(k, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', default='')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(REPO, 'data/portal/mixed_scan/mixed_scan.csv'))))
    hunt = list(csv.DictReader(open(os.path.join(REPO, 'data/portal/fcomp_benchmark/hunt.csv'))))
    bench = json.load(open(os.path.join(REPO, 'data/portal/fcomp_benchmark/benchmark.json')))['row']

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    # (a) beta1/H and T_n(O->A) vs m_S, mu = 0 vs mu > 0
    for mixed, mk, lab in ((False, 'o', r'$\mu_{\rm mix}=0$'), (True, 'x', r'$\mu_{\rm mix}>0$')):
        sel = [r for r in rows if (fl(r, 'mu2_mix') > 0) == mixed and np.isfinite(fl(r, 'betaH_step1'))]
        ms = np.array([fl(r, 'm_S') for r in sel]); b1 = np.array([fl(r, 'betaH_step1') for r in sel])
        tn = np.array([fl(r, 'T_n_0_to_S') for r in sel])
        ax[0].plot(ms, b1, mk, ms=5, alpha=0.7, label=lab)
        ax[2].plot(ms, tn / 1e3, mk, ms=5, alpha=0.7, label=lab)
    # small-kappa hunt (m_S ~ 476-480) and benchmark
    hs = [r for r in hunt if np.isfinite(fl(r, 'T_n_0_to_S'))]
    ax[2].plot([fl(r, 'm_S') for r in hs], [fl(r, 'T_n_0_to_S') / 1e3 for r in hs], 's', ms=4, color='C3', alpha=0.6, label=r'$\kappa \leq 0.02$ hunt')
    ax[2].plot([fl(bench, 'm_S')], [fl(bench, 'T_n_0_to_S') / 1e3], '*', ms=11, color='C3', label='mixed nested benchmark')
    ax[0].plot([fl(bench, 'm_S')], [fl(bench, 'betaH_step1')], '*', ms=11, color='C3', label='mixed nested benchmark')
    ax[0].axhline(384.9, color='0.5', ls=':', lw=1); ax[0].text(1950, 384.9 + 4, 'Table I ($m_S=1$ TeV)', fontsize=7, color='0.4', ha='right')
    ax[2].axhline(1.1984, color='k', ls='--', lw=0.8); ax[2].text(1950, 1.25, r'$T_n(O\to B)$ (direct)', fontsize=7, ha='right')
    ax[0].set_xlabel(r'$m_S$ [GeV]'); ax[0].set_ylabel(r'$\beta_1/H_*$'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[2].set_xlabel(r'$m_S$ [GeV]'); ax[2].set_ylabel(r'$T_n(O\to A)$ [TeV]'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    ax[2].set_yscale('log')
    # (b) second step vs mu2 from the re-scan with the corrected A' finder
    fr = []
    for sub in ('mixed_scan_fixed', 'mixed_scan_fixed_mS800'):
        fx = os.path.join(REPO, 'data/portal', sub, 'mixed_scan.csv')
        if os.path.exists(fx):
            fr += [r for r in csv.DictReader(open(fx)) if r['eps'] == '1e-12']
    if fr:
        fr = sorted(fr, key=lambda r: fl(r, 'mu2_mix'))
        mu = np.array([max(fl(r, 'mu2_mix'), 1e-2) for r in fr])
        tn2 = np.array([fl(r, 'T_n_S_to_v') for r in fr]) / 1e3
        tn1 = np.array([fl(r, 'T_n_0_to_S') for r in fr]) / 1e3
        b2 = np.array([fl(r, 'betaH_step2') for r in fr])
        tun = np.isfinite(b2)
        ax[1].plot(mu, tn1, 's-', color='C0', ms=5, label=r'$T_n(O\to A)$')
        ax[1].plot(mu[tun], tn2[tun], 'o', color='C3', ms=7, label=r'$T_n(A\to B)$: tunnelling')
        ax[1].plot(mu[~tun], tn2[~tun], 'o', mfc='none', color='C3', ms=8, mew=1.6, label=r"$T_n(A\to B)$: classical roll-over of $A'$")
        ax[1].plot(mu, tn2, '-', color='C3', lw=0.8, alpha=0.6)
        for m, t, b in zip(mu, tn2, b2):
            if np.isfinite(b):
                ax[1].annotate(rf'$\beta_2/H_*={b:.0f}$', (m, t), textcoords='offset points', xytext=(6, -12), fontsize=7)
        ax[1].axhline(1.1984, color='k', ls='--', lw=0.8); ax[1].text(0.012, 1.21, r'$T_n(O\to B)$ (direct)', fontsize=7)
        ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\mu_{\rm mix}^2$ [GeV$^2$]  ($10^{-2}\equiv 0$)'); ax[1].set_ylabel(r'$T_n$ [TeV]')
        ax[1].set_title(r'$\kappa=0.1$, $m_S=800$ GeV, $\varepsilon=10^{-12}$', fontsize=9)
        ax[1].legend(fontsize=7, loc='center left'); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(REPO, 'figs/mu_dependence.png'); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200); print('wrote', out)
    if a.paper:
        shutil.copy(out, a.paper); print('copied to', a.paper)

    # LaTeX table rows
    print('\n% m_S, kappa, eps, mu2, T_n(O->A), T_n(A->B), T_n(O->B), beta1/H, beta2/H, regime')
    def row(r, note=''):
        def f(k, s=1.0, d=3):
            x = fl(r, k)
            return '---' if not np.isfinite(x) else f'{x/s:.{d}f}'
        print(f"{fl(r,'m_S'):.0f} & {r['kappa']} & {r['eps']} & {fl(r,'mu2_mix'):.3g} & {f('T_n_0_to_S',1e3)} & {f('T_n_S_to_v',1e3)} & {f('T_n_0_to_v',1e3)} & {f('betaH_step1',1,1)} & {f('betaH_step2',1,1)} & {note or r['regime']} \\\\")
    for ms in (800, 1200, 1600, 2000):
        v = [r for r in rows if round(fl(r, 'm_S')) == ms and r['kappa'] == '0.1' and r['eps'] == '1e-12']
        v = sorted(v, key=lambda r: fl(r, 'mu2_mix'))
        for r in v[:2]:
            row(r)
    for r in hunt:
        if abs(fl(r, 'm_S') - 478) < 0.5 and r['kappa'] == '0.008' and fl(r, 'mu2_mix') < 130:
            row(r, 'direct $O\\to B$ first')
    row(bench, 'immediate ($A\\to B$ at $T_n(O\\to A)$)')


if __name__ == '__main__':
    main()
