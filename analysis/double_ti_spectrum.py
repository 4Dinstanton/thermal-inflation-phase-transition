#!/usr/bin/env python3
"""Full two-stage GW spectrum of double thermal inflation: stage-1 remnant + stage-2 peak.

Stage 1 (S, V0): produced at H1 = sqrt(V0/3)/M_Pl with beta1/H; S decays promptly ->
radiation at T_RH,S = C V0^{1/4}; RD until V0' dominates at T_start2 = C V0'^{1/4};
N2 = ln(T_start2/T_n2) e-folds of stage-2 vacuum domination (Omega x e^{-4N2}, f x e^{-N2});
phi transition at T_n2; phi-matter until T_d (R_md' = (rho_rad(T_d)/V0')^{1/3}); standard RD.
Stage 2 (phi, V0'): Sec. II.D formula with R_md'.
"""
import argparse, csv, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.twoStepGW import make_step, step_quantities, spectrum, shape, M_PL, GEV_TO_HZ, G_S0, T_CMB_GEV
from analysis import gwSpectrum as GS

G_STAR = 100.0
C = (30 / (math.pi**2 * G_STAR))**0.25
# stage-1 S step (eps -> 1): (V0, T_n1, beta1/H) from the pass-1 bounce runs
S_STEP = {5e25: (2098.0, 900.2), 5e27: (2075.0, 860.7), 5e29: (2052.0, 822.5), 2.5e35: (1981.0, 727.0)}  # m_S = 1.2 TeV, identical bath


def rho_rad(T):
    return math.pi**2 / 30 * G_STAR * T**4


def stage1_quantities(V0, Tn1, b1, V0p, Tn2, T_d, v_w=1.0, kappa=1.0):
    H1 = math.sqrt(V0 / 3) / M_PL
    T_RH = C * V0**0.25; T_start2 = C * V0p**0.25
    N2 = max(math.log(T_start2 / Tn2), 0.0)
    R_md = min(1.0, (rho_rad(T_d) / V0p)**(1 / 3))
    amp = (1.67e-5 / b1**2 * kappa**2 * (100 / G_STAR)**(1 / 3) * 0.11 * v_w**3 / (0.42 + v_w**2)) * math.exp(-4 * N2) * R_md
    a_ratio = (T_start2 / T_RH) * math.exp(-N2) * R_md * (G_S0 / G_STAR)**(1 / 3) * T_CMB_GEV / T_d   # a_1/a_0
    f0 = 0.62 / (1.8 - 0.1 * v_w + v_w**2) * b1 * H1 * GEV_TO_HZ * a_ratio
    return dict(label='stage1', h2Omega_peak=amp, **{'f_peak [Hz]': f0}, N2=N2, T_RH=T_RH, T_start2=T_start2, R_md=R_md, H1=H1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/doubleTI/double_ti_scan.csv')
    ap.add_argument('--T_d', type=float, default=100.0); ap.add_argument('--m_S', type=float, default=1000.0); ap.add_argument('--m_phi', type=float, default=707.1)
    ap.add_argument('--out', default='figs/double_ti_two_stage_spectra.png')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r['T_n2'] not in ('', 'nan')]
    lV = np.array([math.log10(float(r['V0p'])) for r in rows]); Tn2 = np.array([float(r['T_n2']) for r in rows]); b2 = np.array([float(r['betaH2']) for r in rows])
    o = np.argsort(lV); lV, Tn2, b2 = lV[o], Tn2[o], b2[o]
    bench = [(1e12, 1e6, 'teal'), (1e12, 3e6, 'darkviolet'), (1e11, 1e7, 'darkorange'), (1e10, 1e8, 'darkgreen'), (5.89e14, 1e7, 'crimson')]
    f = np.geomspace(1e-5, 1e5, 2400)
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    # label each sensitivity curve next to a point (f0) of the curve itself: (f0, y-factor, x-factor, ha, va)
    DET_AT = {'LISA': (3e-4, 1.0, 1.15, 'left', 'bottom'), 'DECIGO': (8.0, 1.0, 0.75, 'right', 'center'),
              'BBO': (8.0, 1.0, 1.35, 'left', 'center'), 'ET': (1e3, 0.25, 1.1, 'left', 'top'), 'aLIGO': (300., 0.15, 1.0, 'center', 'top')}
    for name, col, fn in (('LISA', 'purple', GS.sensitivity_LISA), ('BBO', 'cyan', GS.sensitivity_BBO), ('DECIGO', 'red', GS.sensitivity_DECIGO), ('ET', 'brown', GS.sensitivity_ET), ('aLIGO', 'forestgreen', GS.sensitivity_aLIGO)):
        S = fn(f); m = np.isfinite(S) & (S > 0)
        ax.plot(f[m], S[m], color=col, lw=1.3, ls='-.')
        f0, yf, xf, ha, va = DET_AT[name]
        y0 = 10**np.interp(np.log10(f0), np.log10(f[m]), np.log10(S[m]))
        ax.text(f0 * xf, y0 * yf, name, fontsize=11, color=('darkcyan' if name == 'BBO' else col), fontweight='bold', ha=ha, va=va)
    print("s0, v | V0, V0' | T_n1, beta1/H | T_n2, beta2/H | N1, N2 | stage-2 peak | stage-1 peak (remnant) | ratio")
    for s0, v, col in bench:
        V0 = a.m_S**2 * s0**2 / 2; V0p = a.m_phi**2 * v**2 / 2
        k = min(S_STEP, key=lambda x: abs(math.log10(x / V0))); Tn1, b1 = S_STEP[k]
        T2 = float(np.interp(math.log10(V0p), lV, Tn2)); bb2 = float(np.interp(math.log10(V0p), lV, b2))
        q2 = step_quantities(make_step('stage2', V0p, T2, bb2, V0p, a.T_d, g_star=G_STAR))
        q1 = stage1_quantities(V0, Tn1, b1, V0p, T2, a.T_d)
        N1 = math.log(C * V0**0.25 / Tn1)
        tot = spectrum(f, q2) + spectrum(f, q1)
        def ex(x):
            e = int(math.floor(math.log10(x))); m = x / 10**e
            return rf"10^{{{e}}}" if abs(m - 1) < 0.05 else rf"{m:.0f}\times10^{{{e}}}"
        lab = rf"$s_0={ex(s0 / 1e3)}$, $v={ex(v / 1e3)}$ TeV:  $N_1+N_2={N1+q1['N2']:.1f}$"
        ax.plot(f, tot, color=col, lw=2.2, label=lab)
        ax.plot(f, spectrum(f, q1), color=col, lw=1.0, ls=':')
        print(f"{s0:.0e} {v:.0e} | {V0:.1e} {V0p:.1e} | {Tn1:.0f} {b1:.0f} | {T2:.0f} {bb2:.0f} | {N1:.1f} {q1['N2']:.1f} | {q2['h2Omega_peak']:.2e} @ {q2['f_peak [Hz]']:.3g} Hz | {q1['h2Omega_peak']:.2e} @ {q1['f_peak [Hz]']:.3g} Hz | {q1['h2Omega_peak']/q2['h2Omega_peak']:.1e}")
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-5, 1e5); ax.set_ylim(1e-25, 1e-6)
    ax.set_xlabel('$f$ [Hz]', fontsize=13); ax.set_ylabel(r'$h^2\Omega_{\rm GW}$', fontsize=13)
    ax.set_title(rf"Double thermal inflation: stage-2 peak + stage-1 remnant (dotted), prompt $S$ decay, $T_d={a.T_d/1e3:g}$ TeV", fontsize=13)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.95); ax.grid(alpha=0.3, which='both'); ax.tick_params(labelsize=11)
    fig.tight_layout(); fig.savefig(a.out, dpi=150); print('wrote', a.out)


if __name__ == '__main__':
    main()
