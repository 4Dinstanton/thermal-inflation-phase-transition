#!/usr/bin/env python3
"""Pass-1 (1D channels only) for several S-bath contents: T_n(O->A), beta_1/H, T_n(O->B), beta_phi/H.
Usage: PYTHONPATH=. python3 analysis/sbath_pass1.py --workers 3
"""
import argparse, json, os, sys, time
import numpy as np
from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI, SET8_V, SET8_V0, s0_from_mS
from analysis.portalBounce import scan, analyze
from potential.portalPotential import make_species

ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=3)
ap.add_argument('--eps', type=float, default=1e-12)
ap.add_argument('--kappa', type=float, default=0.3)
ap.add_argument('--nT', type=int, default=20); ap.add_argument('--refine', type=int, default=8)
ap.add_argument('--out', default='data/portal/sbath_pass1/sbath_pass1.json')
ap.add_argument('--set8S', action='store_true', help='run the Debye-resummed (set8-type, 20+20, g=1.09, mb2_0) S-bath cases instead')
a = ap.parse_args()
cases = [  # (label, m_S, bosons, fermions)
    ('nB6 g1.0',           1000, [(6, 1.0)],    []),
    ('nB20 g1.0',          1000, [(20, 1.0)],   []),
    ('nB20 g1.09',         1000, [(20, 1.09)],  []),
    ('nB20+nF20 g1.09',    1000, [(20, 1.09)],  [(20, 1.09)]),
    ('nB20+nF20 g1.09',    1500, [(20, 1.09)],  [(20, 1.09)]),
    ('nB20+nF20 g1.09',    2000, [(20, 1.09)],  [(20, 1.09)]),
    ('nB20 g1.09',         1500, [(20, 1.09)],  []),
    ('nB6 g1.0',           1500, [(6, 1.0)],    []),
]
if a.set8S:
    cases = [('set8-type S bath', mS, [(6, 1.0)], []) for mS in (1000, 1500, 2000)]
rows = []
for lab, mS, bos, fer in cases:
    s0 = s0_from_mS(a.eps, SET8_V0, mS)
    pot = build_pot(SET8_V0, SET8_V, s0, a.eps, a.kappa, SET8_BOSON_PHI, SET8_FERMION_PHI, bos, 100.0, phi_thermal='set8')
    if fer:
        pot.species_S = list(pot.species_S) + make_species(fer, fermion=True)
    if a.set8S:
        from potential.portalPotential import Set8PhiBath
        pot.S_thermal = 'set8'; pot.set8S = Set8PhiBath(include_cw=False)
    Tmax = 2.5 * max(pot.T0_S(), pot.T0_phi()); Tmin = max(0.02 * min(pot.T0_S(), pot.T0_phi()), 5.0)
    Ts = np.geomspace(Tmax, Tmin, a.nT)
    t0 = time.time()
    res = scan(pot, Ts, refine=a.refine, verbose=False, workers=a.workers, do_2d=False)
    info = analyze(pot, res, b4=None)
    s1 = info.get('S step (0->a)', {}); d = info.get('phi direct (0->v, single-step reference)', {})
    row = dict(label=lab, m_S=mS, c_S=pot.thermal_c(pot.species_S), T0_S=pot.T0_S(), T0_phi=pot.T0_phi(),
               Tn1=s1.get('T_n', np.nan), b1=s1.get('beta/H', np.nan), Tn_dir=d.get('T_n', np.nan), b_dir=d.get('beta/H', np.nan),
               dt=time.time() - t0)
    rows.append(row)
    print(f"{lab:18s} m_S={mS:5d} c_S={row['c_S']:.2f} T0_S={row['T0_S']:7.1f} | T_n(O->A)={row['Tn1']:8.1f} (T_n/T0={row['Tn1']/row['T0_S']:.3f}) beta1/H={row['b1']:8.1f} | direct T_n={row['Tn_dir']:7.1f} beta/H={row['b_dir']:7.1f} | {row['dt']:.0f}s", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rows, open(a.out, 'w'), indent=1, default=float)
print('DONE')
