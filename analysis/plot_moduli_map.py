#!/usr/bin/env python3
"""(s0, v) map with the final modulus abundance (note of 27 Oct 2026) as the criterion.
N1 counted from the modulus-dominated onset (Phi_osc = M_Pl), N2 from the second vacuum domination (r_S effect included)."""
import argparse, csv, math, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from analysis.moduli_bound import abundance, nphi_s_osc, s_of_T
from analysis.rs_background import N2 as N2_rs, T_start2
from analysis.units_tev import log_shift
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='data/doubleTI_mphi1000/double_ti_scan.csv')
ap.add_argument('--m_S', type=float, default=1200.); ap.add_argument('--m_phi', type=float, default=1000.); ap.add_argument('--m_mod', type=float, default=1000.)
ap.add_argument('--Tc1', type=float, default=2016.); ap.add_argument('--Td', type=float, default=100.)
ap.add_argument('--b', type=float, default=0.1); ap.add_argument('--bound', type=float, default=1e-15)
ap.add_argument('--rS', type=float, default=0.1)
ap.add_argument('--out', default='figs/double_ti_moduli_map.png'); ap.add_argument('--paper', default='')
a = ap.parse_args()
G = 100.; C = (30 / (math.pi**2 * G))**0.25
rows = [r for r in csv.DictReader(open(a.csv)) if r['T_n2'] not in ('', 'nan')]
lV = np.array([math.log10(float(r['V0p'])) for r in rows]); Tn2 = np.array([float(r['T_n2']) for r in rows])
rD = np.array([float(r['max_ratio_DECIGO']) for r in rows]); rB = np.array([float(r['max_ratio_BBO']) for r in rows])
o = np.argsort(lV); lV, Tn2, rD, rB = lV[o], Tn2[o], rD[o], rB[o]
def thr(ratio):
    lr = np.log10(ratio); i = np.where((lr[:-1] > 0) & (lr[1:] <= 0))[0]
    return 10**np.interp(0.0, [lr[i[0]+1], lr[i[0]]], [lV[i[0]+1], lV[i[0]]]) if len(i) else np.nan
V_D, V_B = thr(rD), thr(rB)

s0 = np.geomspace(1e8, 1e16, 420); v = np.geomspace(1e4, 1e13, 420); S0, VV = np.meshgrid(s0, v)
V1 = a.m_S**2 * S0**2 / 2; V2 = a.m_phi**2 * VV**2 / 2
T2 = np.interp(np.log10(V2), lV, Tn2)
nosc, _ = nphi_s_osc(a.m_mod)
N1 = np.log(V1 / (a.m_mod * nosc * s_of_T(a.Tc1))) / 3            # modulus-dominated first onset (note Eq. 5)
N2_0 = np.clip(np.log(C * V2**0.25 / T2), 0, None)                 # r_S = 0
N2_r = N2_rs(V1, V2, T2, a.rS)                                      # with residual first flaton
res0 = abundance(N1, N2_0, a.Tc1, T2, a.Td, m_phi=a.m_mod, b1=a.b, b2=a.b)
resr = abundance(N1, N2_r, a.Tc1, T2, a.Td, m_phi=a.m_mod, b1=a.b, b2=a.b)
bad = V2 >= V1; norad = VV < math.sqrt(2 * 3.77e14) / a.m_phi; nostage = (N2_r <= 0) & ~bad & ~norad
logn = np.ma.array(np.log10(res0['tot']), mask=bad | norad)

fig, ax = plt.subplots(figsize=(8.8, 6.5))
cf = ax.contourf(S0, VV, np.clip(logn, -21, -8), levels=np.arange(-21, -7.9, 0.5), cmap='viridis_r', extend='max')
cb = fig.colorbar(cf, ax=ax, ticks=np.arange(-21, -7, 2)); cb.set_label(r'$\log_{10}(n_\Phi/s)_{\rm final}$  ($r_S=0$)')
# allowed region: keep the colour scale there, fade everything else
okm = (res0['tot'] < a.bound) & ~bad & ~norad
ax.contourf(S0, VV, np.ma.array(np.ones_like(N1), mask=okm), colors=['w'], alpha=0.55)
ax.contour(S0, VV, logn, levels=[math.log10(a.bound)], colors='w', linewidths=2.4, zorder=6)
# exclusions (distinct tones)
ax.contourf(S0, VV, np.ma.array(np.ones_like(N1), mask=~bad), colors=['0.80'], alpha=1.0)
ax.contour(S0, VV, np.where(bad, 1, 0), levels=[0.5], colors='k', linewidths=0.8)
ax.contourf(S0, VV, np.ma.array(np.ones_like(N1), mask=~norad), colors=['0.45'], alpha=1.0)
ax.text(3e9, 1e11, "$V_0' > V_0$", fontsize=12, color='0.2', ha='center', va='center')
handles = [Line2D([0], [0], color='w', lw=2.4, label=rf'allowed: $n_\Phi/s<10^{{{int(math.log10(a.bound))}}}$ (full colour inside; faded outside)'),
           Patch(facecolor='0.80', edgecolor='k', label=r"$V_0'>V_0$: roles of the flatons exchanged"),
           Patch(facecolor='0.45', label=r"no second vacuum domination: $V_0'<\rho_{\rm rad}(T_{n,2})$")]
# detectors
for Vthr, lab, col in ((V_D, 'DECIGO', 'red'), (V_B, 'BBO', 'cyan')):
    if np.isfinite(Vthr):
        ax.axhline(math.sqrt(2 * Vthr) / a.m_phi, color=col, lw=1.8, ls='-.')
        handles.append(Line2D([0], [0], color=col, lw=1.8, ls='-.', label=f'stage-2 GW reaches {lab} (below the line)'))
ax.axvspan(1e10, 1e12, color='w', alpha=0.18); handles.append(Patch(facecolor='w', edgecolor='0.7', alpha=0.6, label=r'$s_0\in[10^{7},10^{9}]$ TeV'))
ax.legend(handles=handles, loc='upper left', fontsize=7.5, framealpha=0.95)
ax.set_xscale('log'); ax.set_yscale('log'); log_shift(ax.xaxis, -3); log_shift(ax.yaxis, -3)
ax.set_xlabel(r'$s_0$ [TeV]  (stage 1, $V_0=m_S^2 s_0^2/2$, $m_S=1.2$ TeV)'); ax.set_ylabel(r'$v$ [TeV]  (stage 2, $V_0^\prime=m_\phi^2 v^2/2$, $m_\phi=1$ TeV)')
# N1 (modulus count) on top, N2 on right
N1_of = lambda s: np.log(a.m_S**2 * np.asarray(s)**2 / 2 / (a.m_mod * nosc * s_of_T(a.Tc1))) / 3
s_of_N1 = lambda n: np.sqrt(2 * a.m_mod * nosc * s_of_T(a.Tc1) * np.exp(3 * np.asarray(n))) / a.m_S
N2_of = lambda vv: np.log(C * (a.m_phi**2 * np.asarray(vv)**2 / 2)**0.25 / np.interp(np.log10(a.m_phi**2 * np.asarray(vv)**2 / 2), lV, Tn2))
v_of_N2 = lambda n: np.sqrt(2 * (np.exp(np.asarray(n)) * 1800.0 / C)**4) / a.m_phi
tx = ax.secondary_xaxis('top', functions=(lambda s: np.exp(N1_of(np.maximum(s, 1.))), lambda e: s_of_N1(np.log(np.maximum(e, 1e-30)))))
tx.set_xscale('log'); tx.set_xticks(np.exp(np.arange(2, 13))); tx.set_xticklabels([str(k) for k in range(2, 13)]); tx.set_xlabel(r'$N_1$ (from modulus-dominated onset)')
ty = ax.secondary_yaxis('right', functions=(lambda vv: np.exp(N2_of(np.maximum(vv, 1.))), lambda e: v_of_N2(np.log(np.maximum(e, 1e-30)))))
ty.set_yscale('log'); ty.set_yticks(np.exp(np.arange(0, 11))); ty.set_yticklabels([str(k) for k in range(0, 11)]); ty.set_ylabel(r'$N_2$')
ax.set_title(rf'$\Phi_{{\rm osc}}=M_{{\rm Pl}}$, $m_\Phi={a.m_mod/1e3:g}$ TeV, $b_{{1,2}}={a.b:g}$, $T_{{d2}}={a.Td/1e3:g}$ TeV, $T_{{c1}}={a.Tc1/1e3:.2f}$ TeV, $T_{{c2}}=T_{{n,2}}(V_0^\prime)$', fontsize=9, pad=26)
fig.tight_layout(); fig.savefig(a.out, dpi=200); print('wrote', a.out)
if a.paper: shutil.copy(a.out, a.paper)
# allowed ranges
for vv in (1e8, 3e8, 1e9, 1e10, 1e11):
    j = np.argmin(abs(v - vv)); row = okm[j]
    print(f"v={vv/1e3:.0e} TeV: allowed s0 = [{s0[row].min()/1e3:.1e}, {s0[row].max()/1e3:.1e}] TeV" if row.any() else f"v={vv/1e3:.0e} TeV: none")
