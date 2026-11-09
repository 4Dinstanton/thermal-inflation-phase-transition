#!/usr/bin/env python3
"""Nested threshold mu^2_th over (m_S, eps): m_phi = 1 TeV, S bath (20, 1.09), kappa = 0.1."""
import json, math, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
D = json.load(open('data/portal/sstep_samebath_shared.json'))
MS = np.array([d['m_S'] for d in D if np.isfinite(d['Tperc1'] or np.nan)])
TP = np.array([d['Tperc1'] for d in D if np.isfinite(d['Tperc1'] or np.nan)])
V0 = 2.5e35; V = math.sqrt(2 * V0) / 1000.0


def rolled(mS, eps, mu2, T):
    from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI
    s0 = math.sqrt(2 * eps * V0) / mS
    pot = build_pot(V0, V, s0, eps, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], 100.0, phi_thermal='set8', mu2_mix=mu2, S_thermal='set8')
    v = pot.vacua(T)
    return v.get('A') is None and v.get('A_rolled', False)


def thr(args):
    mS, eps = args
    Tp = float(np.interp(mS, MS, TP))
    lo, hi = math.log(1e-9), math.log(1e7)
    if not rolled(mS, eps, math.exp(hi), Tp):
        return float('nan')
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if rolled(mS, eps, math.exp(mid), Tp): hi = mid
        else: lo = mid
    return math.exp(hi)


if __name__ == '__main__':
    mS = np.linspace(1050, 2500, 15); eps = np.logspace(-14, -1, 14)
    jobs = [(m, e) for e in eps for m in mS]
    with ProcessPoolExecutor(3) as ex:
        res = np.array(list(ex.map(thr, jobs, chunksize=4))).reshape(len(eps), len(mS))
    json.dump(dict(mS=mS.tolist(), eps=eps.tolist(), mu2=res.tolist()), open('data/portal/nested_threshold/map_mphi1000_samebath.json', 'w'))
    for i, e in enumerate(eps):
        print(f"eps={e:.0e}: " + " ".join(f"{x:.2g}" for x in res[i]))
