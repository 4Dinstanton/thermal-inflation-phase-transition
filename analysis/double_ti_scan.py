#!/usr/bin/env python3
"""Double thermal inflation: GW from the low-scale second stage.

Stage 1: large V0 (moduli e-folds), ended by the S transition at T_n1 (Table I).
Stage 2: V0' << V0, ended by the phi transition at T_n2 computed with the
set8 phi bath for each V0' (v = sqrt(2 V0')/m_phi).  GW of stage 2 (Sec. II.D
formula with R_md' built from V0' and T_d).  E-folds N1, N2 and the stage-1
decay requirement are tabulated.

    PYTHONPATH=. python3 analysis/double_ti_scan.py --workers 3
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI
from analysis.portalBounce import scan, analyze
from analysis.twoStepGW import make_step, step_quantities, spectrum
from analysis import gwSpectrum as GS

M_PL = 2.435e18
G_STAR = 100.0


def rho_rad(T):
    return math.pi**2 / 30 * G_STAR * T**4


def T_vac_dom(V):
    """T at which rho_rad = V (start of vacuum domination)."""
    return (30 * V / (math.pi**2 * G_STAR)) ** 0.25


def make_figs(rows, a, N1):
    import shutil
    f = np.geomspace(1e-5, 1e3, 1600)
    # ---- figure 1: second-stage spectra vs detectors (style of the thermal-inflation benchmark figure)
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
    show = [1e16, 1e18, 1e20, 1e22, 1e24, 1e26]
    cols = ['teal', 'darkviolet', 'darkorange', 'darkgreen', 'crimson', 'navy']
    # stage 1 fixed at s0 = 1e12 GeV (1e9 TeV), m_S = 1.2 TeV, T_n1 = 2052 GeV -> N1
    C = (30 / (math.pi**2 * G_STAR))**0.25
    V01 = 1200.0**2 * (1e12)**2 / 2; N1f = math.log(C * V01**0.25 / 2052.0)
    for V, c in zip(show, cols):
        r = next((r for r in rows if abs(r['V0p'] / V - 1) < 0.01 and '_q' in r), None)
        if r is None:
            continue
        ax.plot(f, spectrum(f, r['_q']), color=c, lw=2.2, label=rf"$V_0'=10^{{{int(round(math.log10(V))) - 12}}}$ TeV$^4$")
        from analysis.rs_background import gw_factors
        ka, kf = gw_factors(V01, r['V0p'], r['T_n2'], 0.1, a.T_d)
        if ka > 1e-2:
            q1 = dict(r['_q']); q1['h2Omega_peak'] *= ka; q1['f_peak [Hz]'] *= kf
            ax.plot(f, spectrum(f, q1), color=c, lw=1.6, ls='--')
    # single-stage comparison: one flaton with V0 = 2.5e35 GeV^4 (2.5e23 TeV^4)
    Vs = 2.5e35
    ok = [r for r in rows if np.isfinite(r['T_n2']) and np.isfinite(r['betaH2'])]
    lv = np.log10([r['V0p'] for r in ok]); o = np.argsort(lv)
    Tn_s = float(np.interp(math.log10(Vs), lv[o], np.array([r['T_n2'] for r in ok])[o]))
    b_s = float(np.interp(math.log10(Vs), lv[o], np.array([r['betaH2'] for r in ok])[o]))
    qs = step_quantities(make_step('single', Vs, Tn_s, b_s, Vs, a.T_d, g_star=G_STAR))
    Ns = math.log(C * Vs**0.25 / Tn_s)
    ax.plot(f, spectrum(f, qs), color='k', lw=2.4, ls='--', label=rf'single-stage $\phi$ transition, $V_0=2.5\times10^{{23}}$ TeV$^4$')
    print(f"single stage V0={Vs:.1e}: T_n={Tn_s:.0f} beta/H={b_s:.0f} N={Ns:.2f} peak={qs['h2Omega_peak']:.2e} @ {qs['f_peak [Hz]']:.3g} Hz; N1(s0=1e12 GeV)={N1f:.2f}")
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-5, 1e5); ax.set_ylim(1e-25, 1e-6)
    ax.set_xlabel('$f$ [Hz]', fontsize=13); ax.set_ylabel(r'$h^2\Omega_{\rm GW}$', fontsize=13)
    ax.set_title(rf"Second-stage GW of double thermal inflation, $T_d={a.T_d/1e3:g}$ TeV; solid $r_S=0$, dashed $r_S=0.1$ (stage 1: $s_0=10^9$ TeV)", fontsize=12)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.95); ax.grid(alpha=0.3, which='both'); ax.tick_params(labelsize=11)
    fig.tight_layout(); out1 = os.path.join(a.figs, 'double_ti_gw.png'); fig.savefig(out1, dpi=150); print('wrote', out1)
    # ---- figure 2: e-folds and T_n2, beta/H vs V0'
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sel = [r for r in rows if np.isfinite(r['T_n2']) and r['V0p'] < 1e35]
    V = np.array([r['V0p'] for r in sel])
    ax[0].plot(V, [r['N2'] for r in sel], 'o-', label=r'$N_2$ (stage 2)')
    ax[0].axhline(N1, color='C3', ls='--', label=rf'$N_1$ (stage 1, $T_{{n,1}}={a.T_n1:.0f}$ GeV) $=${N1:.1f}')
    ax[0].plot(V, [r['N_tot'] for r in sel], 's-', color='k', label=r'$N_1+N_2$')
    ax[0].set_xscale('log'); ax[0].set_xlabel(r"$V_0'$ [GeV$^4$]"); ax[0].set_ylabel('e-folds'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(V, [r['betaH2'] for r in sel], 'o-', color='C0'); ax[1].set_ylabel(r'$\beta_2/H_*$', color='C0')
    ax2 = ax[1].twinx(); ax2.plot(V, [r['T_n2'] for r in sel], 's--', color='C3'); ax2.set_ylabel(r'$T_{n,2}$ [GeV]', color='C3')
    ax[1].set_xscale('log'); ax[1].set_xlabel(r"$V_0'$ [GeV$^4$]"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); out2 = os.path.join(a.figs, 'double_ti_efolds.png'); fig.savefig(out2, dpi=200); print('wrote', out2)
    if a.paper:
        shutil.copy(out1, a.paper); shutil.copy(out2, a.paper); print('copied to', a.paper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--V0', type=float, default=2.5e35)
    ap.add_argument('--T_n1', type=float, default=2518.6)
    ap.add_argument('--m_S', type=float, default=1000.0)
    ap.add_argument('--m_phi', type=float, default=707.1067811865476)
    ap.add_argument('--T_d', type=float, default=100.0)
    ap.add_argument('--V0p', type=str, default='1e14,3e14,1e15,3e15,1e16,1e17,1e18,1e19,1e20,1e21,1e22,1e23,1e24,1e25,1e26,1e28,1e30,1e32,2.5e35')
    ap.add_argument('--out', default='data/doubleTI')
    ap.add_argument('--figs', default='figs')
    ap.add_argument('--replot', action='store_true', help='rebuild the figures from the CSV without recomputing bounces')
    ap.add_argument('--paper', default='', help='copy figures to this directory')
    a = ap.parse_args()
    if a.replot:
        rows = []
        for r in csv.DictReader(open(os.path.join(a.out, 'double_ti_scan.csv'))):
            r = {k: (float(v) if v not in ('', 'True', 'False') else v) for k, v in r.items()}
            if np.isfinite(r['T_n2']) and np.isfinite(r['betaH2']):
                r['_q'] = step_quantities(make_step('stage2', r['V0p'], r['T_n2'], r['betaH2'], r['V0p'], a.T_d, g_star=G_STAR))
            N1r = math.log((30 * a.V0 / (math.pi**2 * G_STAR))**0.25 / a.T_n1)
            r['N1'] = N1r; r['N_tot'] = N1r + r['N2']
            rows.append(r)
        make_figs(rows, a, float(rows[0]['N1']))
        return
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
    V0ps = [float(x) for x in a.V0p.split(',')]

    # ---- stage 1 bookkeeping (independent of V0')
    H1 = math.sqrt(a.V0 / 3) / M_PL
    T_start1 = T_vac_dom(a.V0)
    N1 = math.log(T_start1 / a.T_n1)
    s0 = math.sqrt(2 * a.V0) / a.m_S
    Gam_pert = a.m_S**3 / (32 * math.pi * s0**2)      # S -> 2 Goldstones, generic m^3/v^2 law
    print(f"stage 1: V0={a.V0:.2e} T_start1={T_start1:.3e} T_n1={a.T_n1:.0f} N1={N1:.2f}  H1={H1:.2e} GeV  s0={s0:.2e}  Gamma_pert(m^3/32pi v^2)={Gam_pert:.2e} GeV", flush=True)

    rows = []
    for V0p in V0ps:
        t0 = time.time()
        v = math.sqrt(2 * V0p) / a.m_phi
        pot = build_pot(V0p, v, 1e9, 1e-12, 0.3, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], G_STAR, phi_thermal='set8')
        Ts = np.geomspace(2.5 * pot.T0_phi(), 0.3 * pot.T0_phi(), 20)
        try:
            res = scan(pot, Ts, refine=8, verbose=False, workers=a.workers, do_2d=False)
            d = analyze(pot, res, b4=None).get('phi direct (0->v, single-step reference)', {})
            Tn2 = float(d.get('T_n', np.nan)); bH = float(d.get('beta/H', np.nan)); S3T = float(d.get('S3/T(T_n)', np.nan))
        except Exception as e:
            print('  scan failed', V0p, e); Tn2 = bH = S3T = float('nan')
        row = dict(V0p=V0p, v=v, lam_phi=pot.lam_phi, T0_phi=pot.T0_phi(), T_n2=Tn2, betaH2=bH, S3T=S3T,
                   H2=math.sqrt(V0p / 3) / M_PL, vac_dom_at_Tn2=(V0p > rho_rad(Tn2)) if np.isfinite(Tn2) else False)
        row['T_start2'] = T_vac_dom(V0p)
        row['N2'] = math.log(row['T_start2'] / Tn2) if (np.isfinite(Tn2) and row['T_start2'] > Tn2) else 0.0
        row['N1'] = N1; row['N_tot'] = N1 + row['N2']
        # stage-1 decay requirement: S gone before T reaches T_n2 (S-matter era would be very short)
        rhoS_at_Tn2 = a.V0 * (Tn2 / a.T_n1)**3 if np.isfinite(Tn2) else float('nan')
        row['H_SMD_at_Tn2'] = math.sqrt(rhoS_at_Tn2 / 3) / M_PL if np.isfinite(rhoS_at_Tn2) else float('nan')
        row['m1_min_prompt'] = (64 * math.pi * a.V0 * row['H_SMD_at_Tn2'])**0.2 if np.isfinite(row['H_SMD_at_Tn2']) else float('nan')
        # self-consistent phi decay: Gamma = m^3/(32 pi v^2) = m^5/(64 pi V0'), T_d = 0.55 sqrt(Gamma M_Pl), capped by T_RH,max(V0')
        Gam_phi = a.m_phi**5 / (64 * math.pi * V0p)
        row['Gamma_phi'] = Gam_phi
        row['T_d_sc'] = min((90 / (math.pi**2 * G_STAR))**0.25 * math.sqrt(Gam_phi * M_PL), T_vac_dom(V0p))
        row['prompt_phi'] = row['T_d_sc'] >= 0.999 * T_vac_dom(V0p)
        if np.isfinite(Tn2) and np.isfinite(bH) and bH > 0:
            q = step_quantities(make_step('stage2', V0p, Tn2, bH, V0p, a.T_d, g_star=G_STAR))
            row['R_md'] = q['R_md']; row['f_peak'] = q['f_peak [Hz]']; row['h2Omega'] = q['h2Omega_peak']
            q2 = step_quantities(make_step('stage2sc', V0p, Tn2, bH, V0p, row['T_d_sc'], g_star=G_STAR))
            row['R_md_sc'] = q2['R_md']; row['f_peak_sc'] = q2['f_peak [Hz]']; row['h2Omega_sc'] = q2['h2Omega_peak']; row['_q2'] = q2
            f = np.geomspace(1e-5, 1e3, 1600); Om = spectrum(f, q)
            for name in ('LISA', 'DECIGO', 'BBO', 'ET'):
                try:
                    S = getattr(GS, 'sensitivity_' + name)(f)
                    row['max_ratio_' + name] = float(np.nanmax(Om / S))
                except Exception:
                    row['max_ratio_' + name] = float('nan')
            row['_q'] = q
        else:
            row.update(R_md=np.nan, f_peak=np.nan, h2Omega=np.nan, R_md_sc=np.nan, f_peak_sc=np.nan, h2Omega_sc=np.nan)
        row['Delta1_prompt'] = math.exp(3 * N1); row['Delta2_prompt'] = math.exp(3 * row['N2'])
        rows.append(row)
        print(f"V0'={V0p:8.1e}: v={v:.2e} T_n2={Tn2:7.1f} beta/H={bH:6.0f} S3/T={S3T:5.0f} vac-dom={row['vac_dom_at_Tn2']} | R_md={row['R_md']:.1e} f={row['f_peak']:.3g} Hz h2Om={row['h2Omega']:.2e} | N2={row['N2']:.2f} N_tot={row['N_tot']:.2f} | LISA {row.get('max_ratio_LISA',np.nan):.1e} DECIGO {row.get('max_ratio_DECIGO',np.nan):.1e} BBO {row.get('max_ratio_BBO',np.nan):.1e} | m1_min(prompt)={row['m1_min_prompt']:.1e} | T_d,sc={row['T_d_sc']:.2e} ({'prompt' if row['prompt_phi'] else 'late'}) R_md,sc={row['R_md_sc']:.1e} f_sc={row['f_peak_sc']:.3g} h2Om_sc={row['h2Omega_sc']:.2e} | {time.time()-t0:.0f}s", flush=True)

    keys = [k for k in rows[0] if not k.startswith('_')]
    with open(os.path.join(a.out, 'double_ti_scan.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in keys})

    make_figs(rows, a, N1)
    print('DONE')


if __name__ == '__main__':
    main()
