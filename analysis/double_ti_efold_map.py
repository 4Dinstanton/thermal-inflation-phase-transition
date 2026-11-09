#!/usr/bin/env python3
"""E-fold budget of double thermal inflation in the (s0, v) plane.

N1 = ln(T_start1/T_n1), N2 = ln(T_start2/T_n2), T_start = (30 V/pi^2 g*)^{1/4},
V0 = m_S^2 s0^2/2, V0' = m_phi^2 v^2/2.  T_n1 = 2519 GeV (S step, V0-insensitive),
T_n2(V0') interpolated from data/doubleTI/double_ti_scan.csv.  Overlays: the band
10 <= N1+N2 <= 11, the stage-2 detectability (DECIGO / BBO) and the stage-1 VEV window.
"""
import argparse, csv, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G_STAR = 100.0
C = (30 / (math.pi**2 * G_STAR))**0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/doubleTI/double_ti_scan.csv')
    ap.add_argument('--m_S', type=float, default=1000.0); ap.add_argument('--m_phi', type=float, default=707.1)
    ap.add_argument('--T_n1', type=float, default=2518.6)
    ap.add_argument('--out', default='figs/double_ti_efold_map.png')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r['T_n2'] not in ('', 'nan')]
    lV = np.array([math.log10(float(r['V0p'])) for r in rows]); Tn2 = np.array([float(r['T_n2']) for r in rows])
    rD = np.array([float(r['max_ratio_DECIGO']) for r in rows]); rB = np.array([float(r['max_ratio_BBO']) for r in rows])
    o = np.argsort(lV); lV, Tn2, rD, rB = lV[o], Tn2[o], rD[o], rB[o]
    # detectability thresholds in V0' (ratio = 1)
    def thr(ratio):
        lr = np.log10(ratio); i = np.where((lr[:-1] > 0) & (lr[1:] <= 0))[0]
        return 10**np.interp(0.0, [lr[i[0]+1], lr[i[0]]], [lV[i[0]+1], lV[i[0]]]) if len(i) else np.nan
    V_D, V_B = thr(rD), thr(rB)
    s0 = np.geomspace(1e7, 1e15, 400); v = np.geomspace(1e4, 1e13, 400)
    S0, VV = np.meshgrid(s0, v)
    V0 = a.m_S**2 * S0**2 / 2; V0p = a.m_phi**2 * VV**2 / 2
    # T_n1(V0) from the eps->1 S-step runs (m_S = 1 TeV): interpolate in log V0
    lV1 = np.log10([5e25, 5e27, 5e29, 2.5e35]); Tn1 = [2098.0, 2075.0, 2052.0, 1981.0]
    _c = np.polyfit(lV1, Tn1, 1); T1 = np.polyval(_c, np.log10(V0))      # T_n1 is linear in log V1 to <1 GeV
    N1 = np.log(C * V0**0.25 / T1)
    T2 = np.interp(np.log10(V0p), lV, Tn2)
    N2 = np.clip(np.log(C * V0p**0.25 / T2), 0, None)
    from analysis.rs_background import N2 as N2_rs
    N2_r = {r: N2_rs(V0, V0p, T2, r) for r in (0.01, 0.1, 0.5)}
    N_r = {r: N1 + N2_r[r] for r in N2_r}
    N = N1 + N2
    bad = (V0p >= V0)  # ordering: the phi stage must carry less vacuum energy than the S stage
    Nm = np.ma.array(N, mask=bad)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    vmin = math.sqrt(2 * 3.77e14) / a.m_phi       # V0' = rho_rad(T_n2): below, no second vacuum domination
    norad = VV < vmin
    Nm = np.ma.array(N, mask=bad | norad)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    cf = ax.contourf(S0, VV, Nm, levels=np.arange(0, 25, 1), cmap='viridis', alpha=0.9)
    cb = fig.colorbar(cf, ax=ax); cb.set_label(r'$N_1+N_2$')
    ax.contourf(S0, VV, np.ma.array(np.ones_like(N), mask=~((N >= 10) & (N <= 15) & ~bad & ~norad)), colors=['none'], hatches=['///'], alpha=0)
    ax.contourf(S0, VV, np.ma.array(np.ones_like(N), mask=~norad), colors=['0.6'], alpha=0.9)
    ax.contourf(S0, VV, np.ma.array(np.ones_like(N), mask=~bad), colors=['0.75'], alpha=0.9)
    ax.contour(S0, VV, np.where(bad, 1, 0), levels=[0.5], colors='k', linewidths=0.8)
    ax.text(1e8, 3e10, "$V_0\' > V_0$", fontsize=12, color="0.2", ha="center", va="center")
    ax.text(1e11, 1.3e4, r"no second vacuum domination: $V_0' < \rho_{\rm rad}(T_{n,2})$", fontsize=8, color='0.1', ha='center', va='center')
    handles = []
    nostage = (N2_r[0.1] <= 0) & ~bad & ~norad
    ax.contourf(S0, VV, np.ma.array(np.ones_like(N), mask=~nostage), colors=['0.55'], alpha=0.75)
    ax.contour(S0, VV, np.where(nostage, 1, 0), levels=[0.5], colors='0.2', linewidths=0.8)
    handles.append(Patch(facecolor='0.55', alpha=0.75, edgecolor='0.2', label=r'excluded for $r_S=0.1$: $\rho_S(T_{n,2})>V_0^\prime$ (no second stage)'))
    for (r, ls) in ((0.01, '--'), (0.1, '-.'), (0.5, ':')):
        base = bad | norad | (N2_r[r] <= 0)
        ax.contour(S0, VV, np.ma.array(N_r[r], mask=base | nostage), levels=[10.0], colors='k', linewidths=1.4, linestyles=ls, zorder=5)
        ax.contour(S0, VV, np.ma.array(N_r[r], mask=base | ~nostage), levels=[10.0], colors='0.4', linewidths=1.4, linestyles=ls, zorder=5)
        handles.append(Line2D([0], [0], color='k', ls=ls, lw=1.4, label=rf'$N_1+N_2=10$ for $r_S={r:g}$'))
    for Vthr, lab, col in ((V_D, 'DECIGO', 'red'), (V_B, 'BBO', 'cyan')):
        if np.isfinite(Vthr):
            vthr = math.sqrt(2 * Vthr) / a.m_phi
            ax.axhline(vthr, color=col, lw=1.8, ls='-.')
            handles.append(Line2D([0], [0], color=col, lw=1.8, ls='-.', label=f'stage-2 GW reaches {lab} (below the line)'))
    ax.axvspan(1e10, 1e12, color='w', alpha=0.18)
    lv_lines = [8, 10, 15, 20]
    cs = ax.contour(S0, VV, Nm, levels=lv_lines, colors='w', linewidths=2.0, zorder=6)
    # label each line where it crosses s0 = 3e10 GeV (inside the plotted, non-excluded area when possible)
    man = []
    for L in lv_lines:
        j = np.argmin(abs(s0 - {8: 3e9, 10: 3e10, 15: 3e12, 20: 3e14}[L])); col = Nm[:, j]
        k = np.where(np.isfinite(col.filled(np.nan)) & (abs(col.filled(np.nan) - L) < 0.3))[0]
        if len(k): man.append((s0[j], v[k[len(k) // 2]]))
    import matplotlib.patheffects as pe
    for L, (xm, ym) in zip(lv_lines, man):
        ax.text(xm, ym, f'{L}', color='w', fontsize=11, fontweight='bold', ha='center', va='center', zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='0.15')])
    handles += [Patch(facecolor='none', edgecolor='k', hatch='///', label=r'$10\leq N_1+N_2\leq 15$ ($r_S=0$)'),
                Patch(facecolor='w', edgecolor='0.7', alpha=0.6, label=r'$s_0\in[10^{7},10^{9}]$ TeV')]
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.95)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$s_0$ [TeV]  (stage 1, $V_0=m_S^2 s_0^2/2$, $m_S=1.2$ TeV)'); ax.set_ylabel(r'$v$ [TeV]  (stage 2, $V_0^\prime=m_\phi^2 v^2/2$, $m_\phi=1$ TeV)')
    from analysis.units_tev import log_shift; log_shift(ax.xaxis, -3); log_shift(ax.yaxis, -3)
    pass
    N1_of = lambda s: np.log(C * (a.m_S**2 * np.asarray(s)**2 / 2)**0.25 / np.polyval(np.polyfit(lV1, Tn1, 1), np.log10(a.m_S**2 * np.asarray(s)**2 / 2)))
    s_of_N1 = lambda n: np.sqrt(2 * (np.exp(np.asarray(n)) * 2050.0 / C)**4) / a.m_S
    N2_of = lambda vv: np.log(C * (a.m_phi**2 * np.asarray(vv)**2 / 2)**0.25 / np.interp(np.log10(a.m_phi**2 * np.asarray(vv)**2 / 2), lV, Tn2))
    v_of_N2 = lambda n: np.sqrt(2 * (np.exp(np.asarray(n)) * 1800.0 / C)**4) / a.m_phi
    tx = ax.secondary_xaxis('top', functions=(lambda s: np.exp(N1_of(np.maximum(s, 1.0))), lambda e: s_of_N1(np.log(np.maximum(e, 1e-30)))))
    tx.set_xscale('log'); tx.set_xticks(np.exp(np.arange(4, 12))); tx.set_xticklabels([str(k) for k in range(4, 12)]); tx.set_xlabel(r'$N_1$')
    ty = ax.secondary_yaxis('right', functions=(lambda vv: np.exp(N2_of(np.maximum(vv, 1.0))), lambda e: v_of_N2(np.log(np.maximum(e, 1e-30)))))
    ty.set_yscale('log'); ty.set_yticks(np.exp(np.arange(0, 11))); ty.set_yticklabels([str(k) for k in range(0, 11)]); ty.set_ylabel(r'$N_2$')
    fig.tight_layout(); fig.savefig(a.out, dpi=200); print('wrote', a.out)
    print(f"stage-2 detectability thresholds: V0' < {V_D:.2e} (DECIGO, v < {math.sqrt(2*V_D)/a.m_phi:.2e}),  V0' < {V_B:.2e} (BBO, v < {math.sqrt(2*V_B)/a.m_phi:.2e})")
    print("analytic: N1 ≈ ½ ln(s0/m_S) − 1.97,  N2 ≈ ½ ln(v/m_phi) − 1.63  ->  N_tot ≈ ½ ln(s0 v/(m_S m_phi)) − 3.6")
    for s in (1e10, 1e11, 1e12, 1e13):
        for vv in (1e6, 3e6, 1e7, 1e8, 1e9, 1e10):
            i = np.argmin(abs(s0 - s)); j = np.argmin(abs(v - vv))
            print(f"  s0={s:.0e} v={vv:.0e}: V0={a.m_S**2*s**2/2:.1e} V0'={a.m_phi**2*vv**2/2:.1e} N1={N1[j,i]:.1f} N2={N2[j,i]:.1f} N_tot={N[j,i]:.1f}")


if __name__ == '__main__':
    main()
