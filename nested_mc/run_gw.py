#!/usr/bin/env python3
"""Run the Monte Carlo for benchmark cases and compute envelope GW spectra.

Cases: single-step reference (channel B only), separated (deta* = +3),
crossover (deta* = 0, +0.5), immediate/nested (deta* = -1, -2).  Same box for
all, so spectra are directly comparable.  Sequential (one process), single
threaded, timestamped progress log.  Results are written after each case so
the run can be resumed.

Usage
-----
    nice -n 15 python nested_mc/run_gw.py --out data/twoStep/nested_mc/gw --v1 0.7
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from nested_mc.bubbles import Params, run, summary, log  # noqa: E402
from nested_mc.gw import envelope_spectrum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/twoStep/nested_mc/gw')
    ap.add_argument('--v1', type=float, default=0.7)
    ap.add_argument('--beta2', type=float, default=2.2)
    ap.add_argument('--L', type=float, default=8.0)
    ap.add_argument('--ndir', type=int, default=64)
    ap.add_argument('--ntime', type=int, default=200)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--detas', default='-2,-1,-0.5,0,0.5,3')
    ap.add_argument('--ngrid', type=int, default=160, help='FFT grid per side')
    ap.add_argument('--kmax', type=float, default=12.0, help='max k R1 kept')
    ap.add_argument('--tag', default='', help='suffix for the output file')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'gw_v1_{a.v1}_b2_{a.beta2}{a.tag}.json')
    results = {}
    if os.path.exists(path):
        with open(path) as fh:
            results = json.load(fh)

    cases = [('single', dict(single_step=True, deta_star=0.0))]
    for d in [float(x) for x in a.detas.split(',')]:
        cases.append((f'deta_{d}', dict(single_step=False, deta_star=d)))
    t0 = time.time()
    for ci, (name, kw) in enumerate(cases, 1):
        if name in results and len(results[name].get('S', [])) >= a.seeds:
            log(f"[gw] {name} already done, skipping")
            continue
        results[name] = dict(kR1=None, S=[], summary=[])
        for s in range(a.seeds):
            p = Params(v1=a.v1, v2=1.0, beta2=a.beta2, L_over_R1=a.L, ndir=a.ndir, seed=s, **kw)
            o = run(p, verbose=True)
            log(f"[gw] {name} seed {s}: spectrum ...")
            g = envelope_spectrum(o, ngrid=a.ngrid, n_time=a.ntime, kR1_max=a.kmax, verbose=True)
            results[name]['kR1'] = g['kR1'].tolist()
            results[name]['nmode'] = g['nmode'].tolist()
            results[name]['S'].append([float(x) if np.isfinite(x) else None for x in g['S']])
            results[name]['summary'].append(summary(o))
            i = int(np.nanargmax(g['S']))
            log(f"[gw] {name} seed {s}: peak S={g['S'][i]:.4g} at kR1={g['kR1'][i]:.2f}")
        Sarr = np.array([[np.nan if x is None else x for x in row] for row in results[name]['S']], float)
        results[name]['S_mean'] = [float(x) if np.isfinite(x) else None for x in np.nanmean(Sarr, axis=0)]
        with open(path, 'w') as fh:
            json.dump(results, fh, indent=1)
        el = time.time() - t0
        log(f"[gw] case {ci}/{len(cases)} done, elapsed {el/60:.1f} min")
    log('[gw] all done')


if __name__ == '__main__':
    main()
