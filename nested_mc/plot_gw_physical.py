#!/usr/bin/env python3
"""Map the Monte-Carlo envelope spectra to today's (f, h^2 Omega) plane.

The MC gives S(k) in units of the first-step bubble scale, k R_{*,1}.  A comoving
mode k today has frequency  f = (k / H_*) f_{H,0} / (2 pi)  with f_{H,0} the
redshifted Hubble rate used in Sec. II.D (twoStepGW._envelope_fpeak), and
k / H_* = (k R_{*,1}) (beta_1/H_*) / ((8 pi)^{1/3} v_{w,1}).  The amplitude is
anchored by setting the MC single-step peak equal to the envelope peak of
Eq. (GW_amp) for beta_2, so that all curves share the normalization of Fig. 3.

    python nested_mc/plot_gw_physical.py --gw a.json,b.json --beta1 385 --beta2 860
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from analysis.twoStepGW import _envelope_amp, _envelope_fpeak, rho_rad, M_PL, GEV_TO_HZ, G_S0, T_CMB_GEV  # noqa: E402

DET = {'LISA': 'C4', 'DECIGO': 'C3', 'BBO': 'C9', 'ET': 'C1', 'aLIGO': 'C2'}


def load(files):
    d = {}
    for fn in files.split(','):
        with open(fn) as fh:
            dd = json.load(fh)
        for k, v in dd.items():
            d.setdefault(k, v)
    return d


def tofloat(row):
    return np.array([np.nan if x is None else x for x in row], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gw', required=True)
    ap.add_argument('--beta1', type=float, default=384.9)
    ap.add_argument('--beta2', type=float, default=860.3)
    ap.add_argument('--V0', type=float, default=2.5e35, help='GeV^4')
    ap.add_argument('--T_reh', type=float, default=100.0)
    ap.add_argument('--g_star', type=float, default=100.0)
    ap.add_argument('--v1', type=float, default=0.7)
    ap.add_argument('--kmax', type=float, default=10.0)
    ap.add_argument('--out', default='figs/nested_mc/mc_gw_physical.png')
    ap.add_argument('--paper', default='')
    ap.add_argument('--cases', default='', help='comma list of case names to plot (default all)')
    ap.add_argument('--title', default='')
    ap.add_argument('--markers', action='store_true', help='draw bin markers (default: lines only)')
    ap.add_argument('--smooth', type=int, default=3, help='running mean over this many log-k bins (0 = off)')
    a = ap.parse_args()

    d = load(a.gw)
    k = np.array(d['single']['kR1'])
    nmode = np.array(d['single'].get('nmode', [100] * len(k)))
    ref = tofloat(d['single']['S_mean'])
    norm = np.nanmax(ref)
    # physical anchors
    rr = rho_rad(a.T_reh, a.g_star)
    R_md = min(1.0, (rr / a.V0) ** (1.0 / 3.0))
    H = math.sqrt(a.V0 / 3.0) / M_PL
    f_H0 = H * GEV_TO_HZ * R_md * (G_S0 / a.g_star) ** (1.0 / 3.0) * T_CMB_GEV / a.T_reh
    Om_peak_single = _envelope_amp(a.beta2, 1.0, a.V0, a.T_reh, a.g_star, 1.0)
    f_peak_env = _envelope_fpeak(a.beta2, a.V0, a.T_reh, a.g_star, 1.0)
    f_of_k = lambda kR1: kR1 * a.beta1 / ((8 * math.pi) ** (1 / 3) * a.v1) * f_H0 / (2 * math.pi)
    print(f"f_H0 = {f_H0:.3e} Hz, envelope single-step peak: f = {f_peak_env:.3g} Hz, h2Omega = {Om_peak_single:.3g}")
    print(f"MC single-step peak at kR1 = {k[np.nanargmax(ref)]:.2f} -> f = {f_of_k(k[np.nanargmax(ref)]):.3g} Hz")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    try:
        from analysis import gwSpectrum as GS
        fgrid = np.geomspace(1e-5, 1e4, 600)
        for name, col, fn in (('LISA', 'C4', GS.sensitivity_LISA), ('DECIGO', 'C3', GS.sensitivity_DECIGO),
                              ('BBO', 'C9', GS.sensitivity_BBO), ('ET', 'C1', GS.sensitivity_ET),
                              ('aLIGO', 'C2', GS.sensitivity_aLIGO)):
            om = np.array([fn(x) for x in fgrid], float)
            ok = np.isfinite(om) & (om > 0)
            ax.plot(fgrid[ok], om[ok], '-.', color=col, lw=1.0, alpha=0.85)
            i = np.argmin(np.where(ok, om, np.inf)); ax.text(fgrid[i], om[i] * 2.5, name, color=col, fontsize=8, ha='center')
    except Exception as e:
        print('sensitivity curves unavailable:', e)
    order = ['single'] + sorted([n for n in d if n != 'single'], key=lambda n: -float(n.split('_')[1]))
    if a.cases:
        keep = a.cases.split(','); order = [n for n in order if n in keep]
    cols = plt.cm.viridis(np.linspace(0.95, 0.05, len(order)))
    for name, col in zip(order, cols):
        v = d[name]; kk = np.array(v['kR1']); S = tofloat(v['S_mean'])
        m = (kk <= a.kmax) & np.isfinite(S)
        f = f_of_k(kk[m]); om = Om_peak_single * S[m] / norm
        if a.smooth and a.smooth > 1:
            w = a.smooth; lo = np.log(om); sm = np.copy(lo)
            for i in range(len(lo)):
                i0, i1 = max(0, i - w // 2), min(len(lo), i + w // 2 + 1); sm[i] = np.mean(lo[i0:i1])
            om = np.exp(sm)
        lab = r'single step ($\Gamma_2$ only)' if name == 'single' else rf"lag ${float(name.split('_')[1]):+g}$"
        ax.plot(f, om, '-', color=col, lw=2.0 if name == 'single' else 1.4, ls='--' if name == 'single' else '-', label=lab)
        j = int(np.nanargmax(om)); print(f'  {name}: peak f = {f[j]:.3g} Hz, h2Omega = {om[j]:.3g}')
        if a.markers:
            few = nmode[m] < 10
            ax.plot(f[~few], om[~few], 'o', color=col, ms=3.5)
            ax.plot(f[few], om[few], 'o', color=col, ms=4, mfc='none')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-4, 1e4); ax.set_ylim(1e-24, 1e-6)
    ax.set_xlabel(r'$f$ [Hz]'); ax.set_ylabel(r'$h^2\Omega_{\rm GW}$')
    ax.set_title(a.title or rf'$V_0^{{1/4}}={a.V0**0.25/1e3:.3g}$ TeV, $T_d={a.T_reh:.0f}$ GeV, $\beta_1/H_*={a.beta1:.0f}$, $\beta_2/H_*={a.beta2:.0f}$', fontsize=9)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=7.5, loc='lower right')
    if a.markers:
        ax.text(0.02, 0.02, 'open markers: bins with < 10 Fourier modes', transform=ax.transAxes, fontsize=7, color='0.3')
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200); print('wrote', a.out)
    if a.paper:
        shutil.copy(a.out, a.paper); print('copied to', a.paper)


if __name__ == '__main__':
    main()
