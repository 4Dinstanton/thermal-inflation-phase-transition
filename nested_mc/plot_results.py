#!/usr/bin/env python3
"""Figures for the nested-bubble Monte-Carlo (Sec. III.D-E of the paper).

Reads data/twoStep/nested_mc/scan/{summary.json, fractions/*.npz} and
data/twoStep/nested_mc/gw/gw_*.json; writes figs/nested_mc/mc_*.png and
copies them into the paper figs directory if --paper is given.

    python nested_mc/plot_results.py --paper "paper/twoStepTIPT/.../figs"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from nested_mc.bubbles import Params, analytic_fractions, f_comp_cubic, EIGHT_PI_13  # noqa: E402

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9,
                     'figure.dpi': 130, 'savefig.dpi': 200})
COL = {0.5: 'C0', 0.7: 'C1', 0.9: 'C2'}


def load_scan(path):
    with open(os.path.join(path, 'summary.json')) as fh:
        rows = json.load(fh)
    return rows


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def mean_spread(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, 0.0
    return v.mean(), (v.max() - v.min()) / 2


def fig_fcomp(rows, out):
    base = [r for r in rows if r['params']['r0'] == 0 and abs(r['params']['beta2'] - 2.2) < 1e-9]
    direct = [r for r in rows if r['params']['r0'] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for v1 in sorted({r['params']['v1'] for r in base}):
        sub = [r for r in base if r['params']['v1'] == v1]
        g = group(sub, lambda r: r['params']['deta_star'])
        d = np.array(sorted(g))
        fov = np.array([mean_spread([r['f_overtaken'] for r in g[x]]) for x in d])
        fown = np.array([mean_spread([r['f_comp'] for r in g[x]]) for x in d])
        fob = np.array([mean_spread([r['F_OB'] for r in g[x]]) for x in d])
        c = COL.get(v1, None)
        ax[0].errorbar(d, fov[:, 0], fov[:, 1], marker='o', ms=4, color=c, label=rf'$v_{{w,1}}/v_{{w,2}}={v1}$ (total)')
        ax[0].plot(d, fown[:, 0], ls='--', marker='s', ms=3, color=c, alpha=0.7)
        ax[1].errorbar(d, fob[:, 0], fob[:, 1], marker='o', ms=4, color=c, label=rf'$v_{{w,1}}/v_{{w,2}}={v1}$')
    if direct:
        g = group(direct, lambda r: r['params']['deta_star'])
        d = np.array(sorted(g))
        fob = np.array([mean_spread([r['F_OB'] for r in g[x]]) for x in d])
        ax[1].errorbar(d, fob[:, 0], fob[:, 1], marker='o', ms=5, mfc='none', color=COL[0.7], ls=':', label=r'$v_{w,1}/v_{w,2}=0.7$, $\Gamma_0=\Gamma_2$')
    for a in ax:
        a.axvline(0, color='k', lw=0.6, ls=':')
        a.set_xlabel(r'$\tilde\beta_1(\eta_{*,2}-\eta_{*,1})$')
        a.set_ylim(-0.02, 1.02)
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r'overtaken fraction of first-step wall  $f_{\rm ov}$')
    ax[1].set_ylabel(r'volume converted by composite walls  $F_{OB}$')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[0].text(0.02, 0.04, 'dashed: by own daughters only', transform=ax[0].transAxes, fontsize=8, color='0.3')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'mc_fcomp.png'))
    plt.close(fig)


def fig_fractions(scan_dir, out, v1=0.7, b2=2.2, detas=((3.0, 0.0), (0.0, 0.0), (-2.0, 0.0), (-1.0, 1.0))):
    fig, ax = plt.subplots(1, len(detas), figsize=(4.0 * len(detas), 3.8), sharey=True)
    for a, (d, r0) in zip(ax, detas):
        files = sorted(glob.glob(os.path.join(scan_dir, 'fractions', f'v1_{v1}_v2_1.0_b2_{b2}_deta_{d}_r0_{r0}_L_*_s0.npz')))
        if not files:
            a.set_title(f'lag {d:+}: no data'); continue
        z = np.load(files[0])
        eta, PO, PA, PB = z['eta_grid'], z['PO'], z['PA'], z['PB']
        p = Params(v1=v1, v2=1.0, beta2=b2, deta_star=d, r0=r0)
        an = analytic_fractions(p, eta)
        for key, mc, col in (('PO', PO, 'C0'), ('PA', PA, 'C1'), ('PB', PB, 'C3')):
            a.plot(eta, mc, '.', ms=3, color=col)
            a.plot(eta, an[key], '-', lw=1.3, color=col, label={'PO': r'$P_O$', 'PA': r"$P_{A'}$", 'PB': r"$P_{B'}$"}[key])
        if r0 == 0:
            a.plot(eta, an['PB_sat'], '--', lw=1.1, color='C3', label=r"$P_{B'}$: every $A\!\to\!B$ at once (seeded at $\Gamma_1$, grows at $v_{w,2}$)")
        a.set_ylim(-0.05, 1.05)
        a.set_title(rf'lag $={d:+g}$' + (r', $\Gamma_0=\Gamma_2$ (direct $O\!\to\!B$ on)' if r0 > 0 else r', no direct channel'), fontsize=9)
        a.set_xlabel(r'$\tilde\beta_1(\eta-\eta_{*,1})$')
        a.set_xlim(-4, max(2.5, d + 2.5))
        a.grid(alpha=0.3)
    ax[0].set_ylabel('volume fraction')
    ax[1].legend(loc='upper left', fontsize=6.5)
    ax[0].text(0.03, 0.55, 'points: MC\nlines: Eqs. (P_O)-(P_B)', transform=ax[0].transAxes, fontsize=8, color='0.3')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'mc_fractions.png'))
    plt.close(fig)


def fig_radii(scan_dir, out, v1=0.7, b2=2.2, detas=(3.0, 0.0, -2.0)):
    p = Params(v1=v1, v2=1.0, beta2=b2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, 3, 40)
    for d, col in zip(detas, ('C0', 'C1', 'C3')):
        files = sorted(glob.glob(os.path.join(scan_dir, 'fractions', f'v1_{v1}_v2_1.0_b2_{b2}_deta_{d}_r0_0.0_L_*_s0.npz')))
        if not files:
            continue
        z = np.load(files[0])
        rB, comp = z['rB_death'], z['B_comp']
        fin = np.isfinite(rB)
        # energy carried by an element at collision ∝ r^3 (sigma ∝ r, area ∝ r^2)
        w_all = rB[fin] ** 3
        inner = rB[fin & ~comp] / p.R2
        compo = rB[fin & comp] / p.R1
        e_comp = (rB[fin & comp] ** 3).sum() / w_all.sum()
        if len(inner):
            ax[0].hist(inner, bins=bins, weights=inner ** 3, histtype='step', density=True, color=col, lw=1.4,
                       label=rf'lag ${d:+g}$: $\langle r\rangle_E={np.average(inner, weights=inner**3):.2f}R_{{*,2}}$, {100*(1-e_comp):.0f}% of energy')
        if len(compo) > 20:
            ax[1].hist(compo, bins=bins, weights=compo ** 3, histtype='step', density=True, color=col, lw=1.4,
                       label=rf'lag ${d:+g}$: $\langle r\rangle_E={np.average(compo, weights=compo**3):.2f}R_{{*,1}}$, {100*e_comp:.0f}% of energy')
    ax[0].set_xlabel(r"collision radius of inner $A'\!\to\!B'$ elements  [$R_{*,2}$]")
    ax[1].set_xlabel(r"collision radius of composite $O\!\to\!B'$ elements  [$R_{*,1}$]")
    for a in ax:
        a.set_ylabel('energy-weighted density'); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'mc_radii.png'))
    plt.close(fig)


def fig_gw(gw_file, out, kmax=None):
    d = {}
    for fn in gw_file.split(','):
        with open(fn) as fh:
            dd = json.load(fh)
        for key, val in dd.items():
            if key not in d:
                d[key] = val
    p = None
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    tofloat = lambda row: np.array([np.nan if x is None else x for x in row], float)
    ref = tofloat(d['single']['S_mean']) if 'single' in d else None
    norm = np.nanmax(ref) if ref is not None else 1.0
    order = ['single'] + sorted([n for n in d if n != 'single'], key=lambda n: -float(n.split('_')[1]))
    cols = plt.cm.viridis(np.linspace(0.95, 0.05, len(order)))
    for name, col in zip(order, cols):
        v = d[name]
        kk = np.array(v['kR1']); S = tofloat(v['S_mean'])
        Ss = np.array([tofloat(r) for r in v['S']])
        m = (kk <= (kmax or 1e9)) & np.isfinite(S)
        lab = 'single step ($\\Gamma_2$ only)' if name == 'single' else rf"lag $\tilde\beta_1(\eta_{{*,2}}-\eta_{{*,1}})={float(name.split('_')[1]):+g}$"
        fob = np.mean([s['F_OB'] for s in v['summary']])
        if name != 'single':
            lab += rf',  $F_{{OB}}={fob:.2f}$'
        nm = np.array(v.get('nmode', [100] * len(kk)))
        ax.plot(kk[m], S[m] / norm, '-', color=col, lw=1.6 if name == 'single' else 1.2,
                ls='--' if name == 'single' else '-', label=lab)
        few = nm[m] < 10
        ax.plot(kk[m][~few], (S[m] / norm)[~few], 'o', ms=3.5, color=col)
        ax.plot(kk[m][few], (S[m] / norm)[few], 'o', ms=4, color=col, mfc='none')
        if Ss.shape[0] > 1:
            ax.fill_between(kk[m], np.nanmin(Ss, 0)[m] / norm, np.nanmax(Ss, 0)[m] / norm, color=col, alpha=0.15)
        if p is None:
            pp = v['summary'][0]['params']; p = Params(v1=pp['v1'], v2=pp['v2'], beta2=pp['beta2'])
    ax.axvline(1.0, color='k', ls=':', lw=0.8); ax.text(1.02, 0.02, r'$k=R_{*,1}^{-1}$', fontsize=8, rotation=90, va='bottom')
    ax.axvline(p.R1 / p.R2, color='k', ls=':', lw=0.8); ax.text(p.R1 / p.R2 * 1.02, 0.02, r'$k=R_{*,2}^{-1}$', fontsize=8, rotation=90, va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$k\,R_{*,1}$'); ax.set_ylabel(r'$S(k)\,/\,S^{\rm peak}_{\rm single}$')
    ax.set_title(rf'envelope GW from MC, $v_{{w,1}}/v_{{w,2}}={p.v1}$, $\tilde\beta_2/\tilde\beta_1={p.beta2}$', fontsize=10)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    ax.text(0.02, 0.02, 'open markers: bins with < 10 Fourier modes', transform=ax.transAxes, fontsize=7, color='0.3')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'mc_gw.png'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', default='data/twoStep/nested_mc/scan')
    ap.add_argument('--gw', default='')
    ap.add_argument('--kmax', type=float, default=8.5)
    ap.add_argument('--out', default='figs/nested_mc')
    ap.add_argument('--paper', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_scan(a.scan)
    print(f'{len(rows)} scan rows')
    fig_fcomp(rows, a.out)
    fig_fractions(a.scan, a.out)
    fig_radii(a.scan, a.out)
    if a.gw:
        fig_gw(a.gw, a.out, kmax=a.kmax)
    if a.paper:
        for f in glob.glob(os.path.join(a.out, 'mc_*.png')):
            shutil.copy(f, a.paper)
            print('copied', f, '->', a.paper)
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
