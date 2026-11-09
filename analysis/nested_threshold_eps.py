#!/usr/bin/env python3
"""Nested threshold mu^2_th(m_S) for several eps: smallest mu^2 with T_roll >= T_perc(O->A)."""
import json, os, math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

MS_N = np.array([480., 500., 700., 800., 1000., 1200., 1600., 2000.])
TN1_N = np.array([1199.1, 1249.5, 1756.2, 2010.0, 2518.8, 3028.5, 4050.3, 5075.2])
PR_MS = np.array([500., 600., 700., 800., 1000., 1200.]); PR = np.array([0.0600, 0.0594, 0.0588, 0.0583, 0.0575, 0.0569])


def tperc1(mS):
    return np.interp(mS, MS_N, TN1_N) * (1 - np.interp(mS, PR_MS, PR, right=PR[-1] - 0.0003 * (mS - 1200) / 100))


def rolled_at(mS, eps, mu2, T):
    from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI, SET8_V, SET8_V0, s0_from_mS
    s0 = s0_from_mS(eps, SET8_V0, mS)
    pot = build_pot(SET8_V0, SET8_V, s0, eps, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0, 1.0)], 100.0, phi_thermal='set8', mu2_mix=mu2)
    v = pot.vacua(T)
    return v.get('A') is None and v.get('A_rolled', False)


def threshold(args):
    mS, eps = args
    Tp = tperc1(mS)
    lo, hi = math.log(1e-4), math.log(1e4)
    if not rolled_at(mS, eps, math.exp(hi), Tp):
        return float('nan')
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if rolled_at(mS, eps, math.exp(mid), Tp): hi = mid
        else: lo = mid
    return math.exp(hi)


if __name__ == '__main__':
    eps_list = [1e-12, 1e-8, 1e-4, 1e-2]
    mS = np.linspace(520, 2000, 19)
    jobs = [(m, e) for e in eps_list for m in mS]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(threshold, jobs))
    out = {str(e): [res[i * len(mS) + k] for k in range(len(mS))] for i, e in enumerate(eps_list)}
    out['mS'] = mS.tolist()
    os.makedirs('data/portal/nested_threshold', exist_ok=True)
    json.dump(out, open('data/portal/nested_threshold/threshold.json', 'w'), indent=1)
    for e in eps_list:
        print(f"eps={e:.0e}: " + "  ".join(f"{m:.0f}:{t:.3g}" for m, t in zip(mS, out[str(e)])))
