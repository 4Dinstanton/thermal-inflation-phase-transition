#!/usr/bin/env python3
"""Dense Delta T_n map in the (m_S, mu_mix^2) plane.
T_n(O->A): from the mixed contour scan (function of m_S only).
T_n(A->B) = max(T_tunnel(mu=0), T_roll(m_S, mu^2)) with T_roll found by bisection on the mixed potential.
"""
import argparse, math, os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json as _json
_D = _json.load(open('data/portal/sstep_samebath_shared.json'))
MS_SCAN = np.array([d['m_S'] for d in _D]); TN1_SCAN = np.array([d['Tn1'] for d in _D]); TP1_SCAN = np.array([d['Tperc1'] for d in _D])
V_PHI = (2 * 2.5e35)**0.5 / 1000.0   # m_phi = 1 TeV
T_TUN0 = 1141.5
B1_SCAN = np.array([365., 370., 375., 378., 385., 388.])   # beta_1/H_* vs m_S (mixed scan)


def t_roll(args):
    mS, mu2 = args
    from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI, SET8_V, SET8_V0, s0_from_mS
    s0 = s0_from_mS(1e-12, SET8_V0, mS)
    pot = build_pot(SET8_V0, V_PHI, s0, 1e-12, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], 100.0, phi_thermal='set8', mu2_mix=mu2, S_thermal='set8')
    def rolled(T):
        v = pot.vacua(T); return v.get('A') is None and v.get('A_rolled', False)
    lo, hi = 1000.0, 4500.0
    if not rolled(lo): return float('nan')
    if rolled(hi): return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if rolled(mid): lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=37); ap.add_argument('--ny', type=int, default=33)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--cache', default='data/portal/mixed_contour_mS_mu/dense_Troll_samebath.json')
    ap.add_argument('--mu2min', type=float, default=0.03); ap.add_argument('--mu2max', type=float, default=100.0)
    ap.add_argument('--T_tun0', type=float, default=float('nan'))
    ap.add_argument('--T_dir', type=float, default=1662.0)
    ap.add_argument('--out', default='figs/delta_Tn_contour_mS_mu.png'); ap.add_argument('--paper', default='')
    a = ap.parse_args()
    mS = np.linspace(1000, 2000, a.nx); mu2 = np.geomspace(a.mu2min, a.mu2max, a.ny)
    if os.path.exists(a.cache):
        d = json.load(open(a.cache)); Troll = np.array(d['Troll']); mS = np.array(d['mS']); mu2 = np.array(d['mu2'])
    else:
        jobs = [(m, u) for u in mu2 for m in mS]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(t_roll, jobs, chunksize=8))
        Troll = np.array(res).reshape(len(mu2), len(mS))
        json.dump(dict(mS=mS.tolist(), mu2=mu2.tolist(), Troll=Troll.tolist()), open(a.cache, 'w'))
    MS, MU = np.meshgrid(mS, mu2)
    Tn1 = np.interp(MS, MS_SCAN, TN1_SCAN)
    Tn2 = np.fmax(np.where(np.isfinite(Troll), Troll, 0.0), a.T_tun0)
    dT = (Tn1 - Tn2) / 1e3   # TeV
    # nested: second transition inside the first before it percolates, T_n(A->B) > T_perc(O->A);
    # (T_n1 - T_perc1)/T_n1 from the S-step percolation runs (m_S = 500..1200 GeV)
    PR_MS = np.array([500., 600., 700., 800., 1000., 1200.]); PR = np.array([0.0600, 0.0594, 0.0588, 0.0583, 0.0575, 0.0569])
    Tperc1 = Tn1 * (1 - np.interp(MS, PR_MS, PR))
    # smooth 2-D interpolation of the roll-over surface (log mu^2, m_S) onto a fine grid
    from scipy.interpolate import RectBivariateSpline
    Tr = np.where(np.isfinite(Troll), Troll, 0.0)
    spl = RectBivariateSpline(np.log(mu2), mS, Tr, kx=3, ky=3, s=0)
    mu_f = np.geomspace(mu2[0], mu2[-1], 900); mS_f = np.linspace(mS[0], mS[-1], 900)
    Troll_f = spl(np.log(mu_f), mS_f)
    MSf, MUf = np.meshgrid(mS_f, mu_f)
    Tn1_f = np.interp(MSf, MS_SCAN, TN1_SCAN); Tp1_f = np.interp(MSf, MS_SCAN, TP1_SCAN)
    Tn2_f = np.fmax(Troll_f, a.T_tun0)
    dT_f = (Tn1_f - Tn2_f) / 1e3
    nested = Tn2_f > Tp1_f
    direct = Tp1_f < a.T_dir          # direct O->B channel active before O->A percolates
    plt.rcParams['hatch.color'] = 'w'; plt.rcParams['hatch.linewidth'] = 0.6
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    cf = ax.contourf(MSf, MUf, np.clip(dT_f, 0, None), levels=np.linspace(0, np.nanmax(dT_f), 41), cmap='viridis')
    cb = fig.colorbar(cf, ax=ax, ticks=[0, 0.5, 1.0, 1.5]); cb.set_label(r'$\Delta T_n$ [TeV]')
    ax.contourf(MSf, MUf, np.ma.array(np.ones_like(dT_f), mask=~nested), colors=['none'], hatches=['\\\\'], alpha=0)
    ax.contour(MSf, MUf, (Tn2_f - Tp1_f), levels=[0.0], colors='w', linewidths=1.4)
    ax.contourf(MSf, MUf, np.ma.array(np.ones_like(dT_f), mask=~direct), colors=['0.7'], alpha=0.95)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='none', edgecolor='w', hatch='\\\\', linewidth=1.4, label=r'nested: $A\to B$ bubbles form before $O\to A$ percolates'),
                       Patch(facecolor='0.7', edgecolor='none', label=r'direct $O\to B$ competes')], loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_yscale('log'); ax.set_xlabel(r'$m_S$ [TeV]'); ax.set_ylabel(r'$\mu_{\rm mix}^2$ [TeV$^2$]')
    from analysis.units_tev import log_shift, lin_scale; log_shift(ax.yaxis, -6); lin_scale(ax.xaxis)
    ax.set_title(r'$\kappa=0.1$, $\varepsilon=10^{-12}$', fontsize=10)
    print('direct-channel boundary m_S =', mS_f[np.argmax(~direct[0])] if (~direct[0]).any() else None)
    fig.tight_layout(); fig.savefig(a.out, dpi=200); print('wrote', a.out)
    if a.paper:
        import shutil; shutil.copy(a.out, a.paper); print('copied')


if __name__ == '__main__':
    main()
