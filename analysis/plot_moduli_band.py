import math, numpy as np, matplotlib, shutil
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis.moduli_bound import abundance, M_PL
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
N1 = np.linspace(3, 13, 500); N2 = np.linspace(0, 10, 500); X, Y = np.meshgrid(N1, N2)
cases = [('note: $T_{c1}=2$, $T_{c2}=1$ TeV', 2000., 1000., 'k'),
         ('ours: $T_{c1}=2.0$, $T_{c2}=1.77$ TeV', 2016., 1766., 'C3')]
fig, ax = plt.subplots(figsize=(7.6, 5.4))
r0 = abundance(X, Y, 2016., 1766., 100.)
cf = ax.contourf(X, Y, np.log10(r0['tot']), levels=np.arange(-21, -8, 0.5), cmap='viridis_r')
cb = fig.colorbar(cf, ax=ax); cb.set_label(r'$\log_{10}(n_\Phi/s)_{\rm final}$  (ours)')
handles = []
for lab, Tc1, Tc2, c in cases:
    r = abundance(X, Y, Tc1, Tc2, 100.)
    ax.contour(X, Y, np.log10(r['tot']), levels=[-15.], colors=c, linewidths=2.2)
    ax.contour(X, Y, np.log10(r['pri']), levels=[-15.], colors=c, linewidths=1.2, linestyles='--')
    handles += [Line2D([0], [0], color=c, lw=2.2, label=rf'$n_\Phi/s=10^{{-15}}$, {lab}'),
                Line2D([0], [0], color=c, lw=1.2, ls='--', label='primordial term only')]
# our benchmarks (N1 counted from the modulus-dominated onset: N1 = ln(T_b1/T_c1) with V1 fixed by s0)
C = (30 / (math.pi**2 * 100))**0.25
def N1_mod(V1, Tc1, m_phi=1000.):
    from analysis.moduli_bound import nphi_s_osc, s_of_T
    nosc, _ = nphi_s_osc(m_phi); return math.log(V1 / (m_phi * nosc * s_of_T(Tc1))) / 3
def N2_of(V2, Tc2): return math.log(C * V2**0.25 / Tc2)
for s0, v, mk in ((1e12, 1e6, 'o'), (1e11, 1e7, 's'), (1e10, 1e8, '^'), (1e12, 1e8, 'D')):
    V1 = 1.2e3**2 * s0**2 / 2; V2 = 1e6 * v**2 / 2
    n1, n2 = N1_mod(V1, 2016.), N2_of(V2, 1766.)
    tot = float(abundance(n1, n2, 2016., 1766., 100.)['tot'])
    ax.plot(n1, n2, mk, color='w', mec='k', ms=9, zorder=5)
    ax.annotate(rf'$(10^{{{int(math.log10(s0))-3}}},10^{{{int(math.log10(v))-3}}})$ TeV', (n1, n2), textcoords='offset points', xytext=(7, 4), fontsize=8, color='w')
    print(f"s0={s0:.0e} v={v:.0e}: N1(mod)={n1:.2f} N1(rad)={math.log(C*V1**0.25/2016.):.2f} N2={n2:.2f} n/s={tot:.2e} {'PASS' if tot<1e-15 else 'fail'}")
ax.set_xlabel(r'$N_1$'); ax.set_ylabel(r'$N_2$'); ax.set_xlim(3, 13); ax.set_ylim(0, 10)
ax.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.95)
ax.set_title(r'modulus-dominated first onset, $\Phi_{\rm osc}=M_{\rm Pl}$, $m_\Phi=1$ TeV, $b_{1,2}=0.1$, $T_{d2}=0.1$ TeV', fontsize=9)
fig.tight_layout(); fig.savefig('figs/moduli_band_N1N2.png', dpi=200)
shutil.copy('figs/moduli_band_N1N2.png', 'paper/twoStepTIPT/Double_step_phase_transition_in_thermal_inflation (1)/figs/'); print('wrote')
