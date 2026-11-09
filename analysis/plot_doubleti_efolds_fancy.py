import csv, math, json, shutil
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 12})
rows = [r for r in csv.DictReader(open('data/doubleTI_mphi1000/double_ti_scan.csv')) if r['T_n2'] not in ('', 'nan') and float(r['V0p']) < 1e35]
V = np.array([float(r['V0p']) for r in rows]); Tn = np.array([float(r['T_n2']) for r in rows]); b = np.array([float(r['betaH2']) for r in rows])
o = np.argsort(V); V, Tn, b = V[o], Tn[o], b[o]
C = (30 / (math.pi**2 * 100))**0.25; mphi, mS = 1000., 1200.
rho = lambda T: math.pi**2 / 30 * 100 * T**4
Vmin = rho(Tn[np.argmin(abs(np.log(V) - np.log(1e15)))])          # second vacuum domination needs V0' > rho_rad(T_n2)
ok = V > Vmin
lv = np.log10(V[ok]); fT = PchipInterpolator(lv, Tn[ok]); fb = PchipInterpolator(lv, b[ok])
lvf = np.linspace(lv[0], lv[-1], 400); Vf = 10**lvf
N2 = np.log(C * Vf**0.25 / fT(lvf))
# stage-1 T_n1(V0) at m_S = 1.2 TeV (identical bath)
lV1 = np.log10([5e25, 5e27, 5e29, 2.5e35]); Tn1 = np.array([2098., 2075., 2052., 1981.])
def N1(s0):
    V0 = mS**2 * s0**2 / 2; return math.log(C * V0**0.25 / np.interp(math.log10(V0), lV1, Tn1))
v_of = lambda V0p: np.sqrt(2 * V0p) / mphi; V_of = lambda v: mphi**2 * v**2 / 2

fig, ax0 = plt.subplots(figsize=(7.5, 4.8)); ax = [None, ax0]
# --- right: beta/H and T_n2
a1 = ax[1]; a1b = a1.twinx()
a1.plot(Vf, fb(lvf), '-', color='C0', lw=2.4); a1b.plot(Vf, fT(lvf) / 1e3, '-', color='C3', lw=2.4)
a1.set_xscale('log'); a1.set_xlim(1e14, 1e32)
a1.set_xlabel(r"$V_0'$ [TeV$^4$]"); a1.set_ylabel(r'$\beta_2/H_*$', color='C0'); a1b.set_ylabel(r'$T_{n,2}$ [TeV]', color='C3')
a1.tick_params(axis='y', colors='C0'); a1b.tick_params(axis='y', colors='C3')
for axx in ax[1:]:
    axx.axvspan(1e14, Vmin, color='0.75', lw=0, zorder=0)
    axx.grid(alpha=0.25, which='major')
sec = ax[1].secondary_xaxis('top', functions=(v_of, V_of)); sec.set_xlabel(r'$v$ [TeV]')
from analysis.units_tev import log_shift; log_shift(ax[1].xaxis, -12, numticks=10); log_shift(sec.xaxis, -3)
fig.tight_layout(); fig.savefig('figs/double_ti_efolds.png', dpi=250)
shutil.copy('figs/double_ti_efolds.png', 'paper/twoStepTIPT/Double_step_phase_transition_in_thermal_inflation (1)/figs/'); print('ok, Vmin=%.2e' % Vmin)
