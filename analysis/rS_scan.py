#!/usr/bin/env python3
"""Residual first-stage flaton fraction r_S: background between the two stages and its effect on N2 and the GW.
After stage 1: rho_r = (1-r) V1 (a/a1)^-4, rho_S = r V1 (a/a1)^-3 (no decay), V2 = const.
Stage 2 starts when V2 = rho_r + rho_S; phi nucleates when the radiation temperature reaches T_n2.
GW: Omega_* ∝ (V2/rho_tot)^2 at nucleation; afterwards matter = V2 + rho_S, decaying together at T_d."""
import math, json
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
G = 100.0; C = (30 / (math.pi**2 * G))**0.25; Td = 100.0
rho_r = lambda T: math.pi**2 / 30 * G * T**4
BENCH = [  # (label, s0 GeV, v GeV, T_n1, T_n2)
    (r'$(s_0,v)=(10^9,10^3)$ TeV', 1e12, 1e6, 2052., 1825.),
    (r'$(s_0,v)=(10^8,10^4)$ TeV', 1e11, 1e7, 2075., 1809.),
    (r'$(s_0,v)=(10^7,10^5)$ TeV', 1e10, 1e8, 2098., 1792.)]
mS, mphi = 1200., 1000.


def history(V1, V2, Tn2, r):
    Trh = C * ((1 - r) * V1)**0.25 if r < 1 else 0.0
    # use x = ln(a/a1); radiation temperature T = T_rh e^{-x} (if r=1 the old plasma at ~T_n1 is used)
    T0 = Trh if r < 1 else 2000.
    rr = lambda x: rho_r(T0) * math.exp(-4 * x)
    rs = lambda x: r * V1 * math.exp(-3 * x)
    x_n = math.log(T0 / Tn2)                                   # phi nucleation (radiation T = T_n2)
    f = lambda x: V2 - rr(x) - rs(x)
    x_s = brentq(f, -5, 200) if f(200) > 0 else np.inf          # V2 = rho_r + rho_S
    N2 = max(x_n - x_s, 0.0)
    rho_tot = V2 + rr(x_n) + rs(x_n)
    # after nucleation: matter M = V2 + rho_S(x_n) (a_n/a)^3, radiation R; all decays when total = rho_rad(T_d)
    M0, R0 = V2 + rs(x_n), rr(x_n)
    y_d = brentq(lambda y: M0 * math.exp(-3 * y) + R0 * math.exp(-4 * y) - rho_r(Td), 0, 200)   # y = ln(a_d/a_n)
    Om = (V2 / rho_tot)**2 * rho_tot * math.exp(-4 * y_d) / rho_r(Td)       # Omega_GW at T_d up to common factors
    return dict(N12=x_s if np.isfinite(x_s) else np.nan, N2=N2, frac=V2 / rho_tot, Om=Om, y_d=y_d, x_n=x_n)


rs = np.logspace(-9, 0, 400)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
out = {}
for (lab, s0, v, Tn1, Tn2), c in zip(BENCH, ('C0', 'C1', 'C2')):
    V1 = mS**2 * s0**2 / 2; V2 = mphi**2 * v**2 / 2
    h0 = history(V1, V2, Tn2, 0.0)
    H = [history(V1, V2, Tn2, r) for r in rs]
    N2 = np.array([h['N2'] for h in H]); sup = np.array([h['Om'] / h0['Om'] for h in H])
    rb = (V2 / V1)**0.25
    ax[0].plot(rs, N2, color=c, lw=2.2, label=lab); ax[0].axvline(rb, color=c, ls=':', lw=1.2)
    ax[1].plot(rs, sup, color=c, lw=2.2, label=lab); ax[1].axvline(rb, color=c, ls=':', lw=1.2)
    k = lambda target: rs[np.argmin(abs(N2 - target))]
    out[lab] = dict(N2_0=h0['N2'], N12_0=h0['N12'], rb=rb, r_halfN2=float(k(h0['N2'] / 2)), r_N2zero=float(rs[np.argmax(N2 <= 1e-3)]) if (N2 <= 1e-3).any() else None,
                    sup_at_rb=float(np.interp(rb, rs, sup)), sup_1e_2=float(np.interp(1e-2, rs, sup)), sup_1e_1=float(np.interp(1e-1, rs, sup)))
    print(lab.replace('$', ''), {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in out[lab].items()})
ax[0].set_xscale('log'); ax[0].set_xlabel(r'residual fraction $r_S$'); ax[0].set_ylabel(r'$N_2$'); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'residual fraction $r_S$'); ax[1].set_ylabel(r'$\Omega_{\rm GW}(r_S)/\Omega_{\rm GW}(0)$'); ax[1].grid(alpha=0.25, which='both')
ax[1].set_ylim(1e-6, 2)
for a_ in ax: a_.text(0.02, 0.03, r'dotted: $r_S=(V_2/V_1)^{1/4}$', transform=a_.transAxes, fontsize=9)
fig.tight_layout(); fig.savefig('figs/rS_scan.png', dpi=220); print('wrote figs/rS_scan.png')
json.dump(out, open('data/doubleTI_mphi1000/rS_scan.json', 'w'), indent=1)
