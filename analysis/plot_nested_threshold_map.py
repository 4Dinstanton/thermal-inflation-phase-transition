import json, numpy as np, matplotlib, shutil
matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
from scipy.interpolate import RectBivariateSpline
from analysis.units_tev import lin_scale
d = json.load(open('data/portal/nested_threshold/map_mphi1000_samebath.json'))
mS = np.array(d['mS']); le = np.log10(d['eps']); Z = np.log10(np.array(d['mu2'])) - 6.0     # log10(mu^2 / TeV^2)
spl = RectBivariateSpline(le, mS, Z, kx=3, ky=3)
mf = np.linspace(mS[0], mS[-1], 600); ef = np.linspace(le[0], le[-1], 600); Zf = spl(ef, mf); M, E = np.meshgrid(mf, 10**ef)
fig, ax = plt.subplots(figsize=(6.8, 4.9))
cf = ax.contourf(M, E, Zf, levels=np.linspace(-12, -3, 37), cmap='magma', extend='both')
cs = ax.contour(M, E, Zf, levels=np.arange(-11, -3, 1), colors='w', linewidths=0.8)
ax.clabel(cs, fmt=lambda x: rf'$10^{{{int(round(x))}}}$', fontsize=8)
cb = fig.colorbar(cf, ax=ax, ticks=range(-12, -2)); cb.set_label(r'minimum $\mu^2_{\rm mix}$ for nested [TeV$^2$], $\log_{10}$')
ax.set_yscale('log'); ax.set_xlabel(r'$m_S$ [TeV]'); ax.set_ylabel(r'$\varepsilon$'); lin_scale(ax.xaxis)
ax.set_title(r'$m_\phi=1$ TeV, $\kappa=0.1$', fontsize=10)
fig.tight_layout(); fig.savefig('figs/nested_threshold_map.png', dpi=200)
shutil.copy('figs/nested_threshold_map.png', 'paper/twoStepTIPT/Double_step_phase_transition_in_thermal_inflation (1)/figs/'); print('ok')
