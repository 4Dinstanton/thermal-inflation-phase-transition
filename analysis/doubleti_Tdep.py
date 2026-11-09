#!/usr/bin/env python3
"""Temperature dependence of nucleation and percolation for the two stages of double thermal inflation.
Top: S3/T and the nucleation condition ln(T^4/H^4) for the S step (V0) and the phi step (V0'),
marking T_c, T_0, T_n.  Bottom: false-vacuum fraction exp(-I(T)) with T_perc (I = 0.34) and T_n."""
import argparse, math, json
import numpy as np
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI
from analysis.portalBounce import scan, analyze
from analysis.twoStepBounce import log_gamma3, PERC_I

ap = argparse.ArgumentParser()
ap.add_argument('--m_S', type=float, default=1200.); ap.add_argument('--m_phi', type=float, default=1000.)
ap.add_argument('--s0', type=float, default=1e11); ap.add_argument('--v', type=float, default=1e7)
ap.add_argument('--out', default='figs/doubleti_Tdep.png'); ap.add_argument('--paper', default='')
a = ap.parse_args()
V0 = a.m_S**2 * a.s0**2 / 2; V0p = a.m_phi**2 * a.v**2 / 2


def channel(V_tot, field):
    """Return T grid, S3/T, H(T), plus T_c, T_0 for one flaton with total vacuum energy V_tot."""
    if field == 'S':
        pot = build_pot(V_tot, math.sqrt(2 * V_tot) / a.m_phi, a.s0, 1 - 1e-6, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6., 1.)], 100., phi_thermal='set8', S_thermal='set8')
        key, T0 = '0->a', pot.T0_S()
        def dV(T):
            sv = pot.slice_S().vacua(T)['true']
            return float(pot.slice_S().V_rel(sv.phi, T)) if sv is not None else 1.0
    else:
        pot = build_pot(V_tot, a.v, math.sqrt(2 * 1e-12 * V_tot) / a.m_S, 1e-12, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6., 1.)], 100., phi_thermal='set8', S_thermal='set8')
        key, T0 = '0->v', pot.T0_phi()
        def dV(T):
            pv = pot.slice_phi(0.0).vacua(T)['true']
            return float(pot.slice_phi(0.0).V_rel(pv.phi, T)) if pv is not None else 1.0
    Ts = np.geomspace(1.6 * T0, 1.0 * T0, 40)
    res = scan(pot, Ts, refine=14, verbose=False, workers=3, do_2d=False)
    T = np.asarray(res['T'], float); S = np.asarray(res[f'S3/T {key}'], float)
    m = np.isfinite(S) & (S > 0); o = np.argsort(T[m]); T, S = T[m][o], S[m][o]
    # T_c: degenerate minima (vacuum energy of the broken phase relative to the origin, at fixed T)
    Tg = np.geomspace(T0, 50 * T0, 200); vals = np.array([dV(t) for t in Tg])
    Tc = np.nan
    sgn = np.where(np.diff(np.sign(vals)))[0]
    if len(sgn): Tc = float(brentq(dV, Tg[sgn[-1]], Tg[sgn[-1] + 1]))
    return pot, T, S, T0, Tc


def nucl_perc(pot, T, S, V_false):
    H = pot.hubble(T, V_false)
    crit = 4 * np.log(T / H)                      # Gamma = H^4  <=>  S3/T = ln(T^4 (S/2pi)^{3/2} / H^4)
    lnG = log_gamma3(T, S); f = lnG - 4 * np.log(H)
    Tn = np.nan
    i = np.where(np.diff(np.sign(f)))[0]
    if len(i): k = i[-1]; Tn = T[k] + (T[k + 1] - T[k]) * (0 - f[k]) / (f[k + 1] - f[k])
    # percolation integral, same discretization as twoStepBounce.percolation_T (ascending T grid)
    keep = S < 1e3
    Tg, Sg = T[keep], S[keep]
    spl = PchipInterpolator(Tg, np.log(Sg)); Tf = np.linspace(Tg[0], Tg[-1], 3000); Sf = np.exp(spl(Tf)); Hf = pot.hubble(Tf, V_false)
    w = np.exp(log_gamma3(Tf, Sf)) / (Hf * Tf**4); invH = 1 / Hf
    cum = np.concatenate([[0], np.cumsum(0.5 * (invH[1:] + invH[:-1]) * np.diff(Tf))])
    I = np.array([4 * np.pi / 3 * np.trapz(w[k:] * (cum[k:] - cum[k])**3, Tf[k:]) for k in range(len(Tf))])
    Pf = np.exp(-I)
    j = np.where(I >= PERC_I)[0]; Tp = Tf[j[-1]] if len(j) else np.nan
    return crit + 1.5 * np.log(S / (2 * np.pi)), Tn, Tf, Pf, Tp


from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(8.2, 5.0)); ax2 = ax.twinx()
summary = {}; u = 1e3
for field, Vf, col, lab in (('S', V0, 'C0', r'stage 1 ($O\to A$, $V_0$)'), ('phi', V0p, 'C3', r"stage 2 ($O\to B$, $V_0'$)")):
    pot, T, S, T0, Tc = channel(Vf, field)
    crit, Tn, Tf, Pf, Tp = nucl_perc(pot, T, S, Vf)
    Tg = np.linspace(T[0], T[-1], 2000); Sg = np.exp(PchipInterpolator(T, np.log(S))(Tg))
    lr = (log_gamma3(Tg, Sg) - 4 * np.log(pot.hubble(Tg, Vf))) / np.log(10)
    ax.plot(Tg / u, 10**lr, '-', color=col, lw=2.2)
    ax2.plot(Tf / u, 1 - Pf, '--', color=col, lw=1.8)
    for t, ls in ((Tn, ':'), (Tp, '-.')):
        ax.axvline(t / u, color=col, lw=0.9, ls=ls)
    ax.text(Tn / u, 1.01, r'$T_n$', color=col, fontsize=9, ha='right', va='bottom', transform=ax.get_xaxis_transform())
    ax.text(Tp / u, 1.01, r'$T_{\rm perc}$', color=col, fontsize=9, ha='left', va='bottom', transform=ax.get_xaxis_transform())
    summary[field] = dict(V=Vf, T0=T0, Tn=Tn, Tperc=Tp)
ax.axhline(1.0, color='k', lw=0.8)
ax.set_yscale('log'); ax.set_ylim(1e-30, 1e30); ax.set_yticks([1e-30, 1e-20, 1e-10, 1, 1e10, 1e20, 1e30]); ax.set_ylabel(r'$\Gamma/H^4$ (solid)')
ax2.set_ylim(-0.02, 1.02); ax2.set_ylabel(r'true-vacuum fraction $1-e^{-I}$ (dashed)')
ax.set_xlim(2.35, 1.55); ax.set_xlabel(r'$T$ [TeV]  (time runs to the right)')
ax.grid(alpha=0.3)
ax.legend(handles=[Line2D([0], [0], color='C0', lw=2, label=r'stage 1: $O\to A$ ($S$, $V_0$)'),
                   Line2D([0], [0], color='C3', lw=2, label=r"stage 2: $O\to B$ ($\phi$, $V_0'$)"),
                   Line2D([0], [0], color='k', ls=':', lw=1, label=r'$T_n$: $\Gamma=H^4$'),
                   Line2D([0], [0], color='k', ls='-.', lw=1, label=r'$T_{\rm perc}$')], fontsize=8, loc='lower left', framealpha=0.95)
fig.tight_layout(); fig.savefig(a.out, dpi=200); print('wrote', a.out); print(json.dumps(summary, indent=1))
if a.paper:
    import shutil; shutil.copy(a.out, a.paper)
