#!/usr/bin/env python3
"""n(T), Gamma/H^4, I/0.34, P/0.7 versus T for the two transitions of double thermal inflation
(same definitions as analysis/drawAction.py; H fixed by the vacuum energy of each stage)."""
import argparse, math
import numpy as np
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI
from analysis.portalBounce import scan
from analysis.twoStepBounce import log_gamma3

ap = argparse.ArgumentParser()
ap.add_argument('--m_S', type=float, default=1200.); ap.add_argument('--m_phi', type=float, default=1000.)
ap.add_argument('--s0', type=float, default=1e11); ap.add_argument('--v', type=float, default=1e7)
ap.add_argument('--out', default='figs/doubleti_action_panels.png'); ap.add_argument('--paper', default='')
a = ap.parse_args()
V0 = a.m_S**2 * a.s0**2 / 2; V0p = a.m_phi**2 * a.v**2 / 2
M_PL = 2.435e18
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})


def s3t(V, field):
    if field == 'S':
        pot = build_pot(V, math.sqrt(2 * V) / a.m_phi, a.s0, 1 - 1e-6, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6., 1.)], 100., phi_thermal='set8', S_thermal='set8')
        key, T0 = '0->a', pot.T0_S()
    else:
        pot = build_pot(V, a.v, math.sqrt(2e-12 * V) / a.m_S, 1e-12, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6., 1.)], 100., phi_thermal='set8', S_thermal='set8')
        key, T0 = '0->v', pot.T0_phi()
    Ts = np.geomspace(1.35 * T0, 1.0 * T0, 40)
    res = scan(pot, Ts, refine=14, verbose=False, workers=4, do_2d=False)
    T = np.asarray(res['T'], float); S = np.asarray(res[f'S3/T {key}'], float)
    m = np.isfinite(S) & (S > 0); o = np.argsort(T[m])
    return T[m][o], S[m][o], T0


def compute(V, field):
    T, S, T0 = s3t(V, field)
    H = math.sqrt(V / 3) / M_PL
    Tg = np.linspace(T[0], T[-1], 6000)
    lgh4 = log_gamma3(Tg, np.exp(PchipInterpolator(T, np.log(S))(Tg))) - 4 * math.log(H)
    gh4 = np.exp(np.clip(lgh4, -700, 700))
    rev = slice(None, None, -1)
    n = -cumulative_trapezoid((gh4 / Tg)[rev], Tg[rev], initial=0)[rev]
    I = np.array([4 * np.pi / 3 * np.trapz(gh4[k:] * (Tg[k:] / t - 1)**3 / t, Tg[k:]) for k, t in enumerate(Tg)])
    P = np.exp(-I)
    Tn = Tg[np.argmin(abs(np.log(np.maximum(n, 1e-300))))]
    Tp = Tg[np.argmin(abs(P / 0.7 - 1))]
    Tc1 = Tg[np.argmin(abs(np.log(np.maximum(P / 0.7, 1e-300) / 1e-5)))]
    print(f"{field}: T0={T0:.1f} Tn={Tn:.1f} Tp={Tp:.1f} Tc1={Tc1:.1f}")
    return dict(T=Tg, n=n, gh4=gh4, I=I, P=P, Tn=Tn, Tp=Tp, Tc1=Tc1)


from matplotlib.lines import Line2D
d1 = compute(V0, 'S'); d2 = compute(V0p, 'phi')
def tex(x):
    m, e = f"{x / 1e12:.1e}".split('e'); return m + r"\times10^{" + str(int(e)) + "}"
fig, (ax, bx) = plt.subplots(2, 1, figsize=(15, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.06))
box = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
for d, ls in ((d1, '-'), (d2, '--')):
    x = d['T'] / 1e3
    ax.plot(x, d['n'], color='red', ls=ls, lw=2.2); ax.plot(x, d['gh4'], color='blue', ls=ls, lw=2.2)
    ax.plot(x, d['I'] / 0.34, color='green', ls=ls, lw=2.2); ax.plot(x, d['P'] / 0.7, color='orange', ls=ls, lw=2.2)
    for t, col, lab, y, ha in ((d['Tn'], 'red', r'$T_n$', 3e3, 'left'), (d['Tp'], 'orange', r'$T_p$', 3e2, 'left'), (d['Tc1'], 'black', r'$T_{c_1}$', 3e1, 'right')):
        for axx in (ax, bx):
            axx.axvline(t / 1e3, color=col, ls=':', lw=1.2)
        off = 0.0015 if ha == 'left' else -0.0015
        ax.text(t / 1e3 + off, y, lab, color=col, fontsize=13, ha=ha, va='center', bbox=box, zorder=6)
ax.axhline(1, ls='--', color='k', lw=1.0)
ax.set_yscale('log'); ax.set_ylim(1e-4, 1e4)
leg1 = ax.legend(handles=[Line2D([0], [0], color=c, lw=2.2, label=l) for c, l in (('red', r'$n(T)$'), ('blue', r'$\Gamma/H^4$'), ('green', r'$I/0.34$'), ('orange', r'$P/0.7$'))],
                 fontsize=12, loc='upper center', ncol=4, framealpha=0.95)
ax.add_artist(leg1)
ax.legend(handles=[Line2D([0], [0], color='k', ls='-', lw=2, label=r'stage 1, $O\to A$ ($S$, $V_0=' + tex(V0) + r'$ TeV$^4$)'),
                   Line2D([0], [0], color='k', ls='--', lw=2, label=r"stage 2, $A\to B$ ($\phi$, $V_0'=" + tex(V0p) + r'$ TeV$^4$)')],
          fontsize=11, loc='lower center', framealpha=0.95)
Tx = np.linspace(min(d2['T'][0], d1['T'][0]), max(d1['T'][-1], d2['T'][-1]), 6000)
PO = np.interp(Tx, d1['T'], d1['P'], left=0.0, right=1.0)
PB = 1 - np.interp(Tx, d2['T'], d2['P'], left=0.0, right=1.0); PA = 1 - PO - PB
bx.plot(Tx / 1e3, PO, color='C0', lw=2.4, label=r'$P_O$'); bx.plot(Tx / 1e3, PA, color='C1', lw=2.4, label=r'$P_A$'); bx.plot(Tx / 1e3, PB, color='C3', lw=2.4, label=r'$P_B$')
bx.set_ylim(-0.03, 1.05); bx.set_ylabel('volume fraction', fontsize=13); bx.legend(fontsize=12, loc='center', ncol=3, framealpha=0.95)
lo = d2['Tc1'] - 0.8 * (d2['Tn'] - d2['Tc1']); hi = d1['Tn'] + 1.6 * (d1['Tn'] - d1['Tc1'])
bx.set_xlim(lo / 1e3, hi / 1e3)
bx.set_xlabel(r'$T\,[\mathrm{TeV}]$', fontsize=14)
for axx in (ax, bx): axx.grid(alpha=0.25); axx.tick_params(labelsize=12)
fig.savefig(a.out, dpi=250, bbox_inches='tight'); print('wrote', a.out)
if a.paper:
    import shutil; shutil.copy(a.out, a.paper)
