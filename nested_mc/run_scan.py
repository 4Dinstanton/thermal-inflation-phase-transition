#!/usr/bin/env python3
"""Scan the thin-wall Monte Carlo over (v1/v2, deta*, r0, beta2) and store summaries.

Low-footprint: each worker is single-threaded (thread limits set in bubbles.py),
default 2 workers, `nice`d.  Progress is logged with timestamps to stderr and to
<out>/progress.log; results are appended to <out>/summary.json after every run,
so the scan can be interrupted and resumed (finished tags are skipped).

Usage
-----
    nice -n 15 python nested_mc/run_scan.py --out data/twoStep/nested_mc/scan --workers 2
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
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from nested_mc.bubbles import Params, run, summary, save_fractions, log  # noqa: E402

DETAS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
V1S = [0.5, 0.7, 0.9]


def one(args):
    p, out_dir = args
    try:
        os.nice(5)
    except Exception:
        pass
    o = run(p, verbose=True)
    s = summary(o)
    save_fractions(o, os.path.join(out_dir, 'fractions', p.tag() + '.npz'))
    return s


def build_jobs(a):
    jobs = []
    for v1 in V1S:
        for d in DETAS:
            # v1 close to v2: R2 << R1 and the immediate regime produces O(10^4-10^5)
            # B bubbles per box; restrict to lag >= -1 and one seed there.
            if v1 >= 0.9 and d < -1.0:
                continue
            for s in range(1 if v1 >= 0.9 else a.seeds):
                jobs.append(Params(v1=v1, v2=1.0, beta2=a.beta2, deta_star=d, r0=0.0,
                                   L_over_R1=a.L, ndir=a.ndir, seed=s))
    for d in [-1.0, -0.5, 0.0, 0.5, 1.0]:                     # direct channel
        for s in range(a.seeds):
            jobs.append(Params(v1=0.7, v2=1.0, beta2=a.beta2, deta_star=d, r0=1.0,
                               L_over_R1=a.L, ndir=a.ndir, seed=s))
    for b2 in [1.5, 4.0]:                                     # beta2 sensitivity
        for d in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for s in range(a.seeds):
                jobs.append(Params(v1=0.7, v2=1.0, beta2=b2, deta_star=d, r0=0.0,
                                   L_over_R1=a.L, ndir=a.ndir, seed=s))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/twoStep/nested_mc/scan')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--L', type=float, default=8.0)
    ap.add_argument('--ndir', type=int, default=96)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--beta2', type=float, default=2.2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ_path = os.path.join(a.out, 'summary.json')
    results = []
    if os.path.exists(summ_path):
        with open(summ_path) as fh:
            results = json.load(fh)
    done = {r['tag'] for r in results}
    jobs = [p for p in build_jobs(a) if p.tag() not in done]
    log(f"scan: {len(jobs)} runs to do ({len(done)} already done), workers={a.workers}, L={a.L}R1, ndir={a.ndir}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, (p, a.out)): p for p in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                s = f.result()
            except Exception as e:  # keep going
                log(f"FAILED {futs[f].tag()}: {e!r}")
                continue
            results.append(s)
            with open(summ_path, 'w') as fh:
                json.dump(results, fh, indent=1)
            el = time.time() - t0
            log(f"scan progress {n}/{len(jobs)}  elapsed {el/60:.1f} min  ETA {(el/n)*(len(jobs)-n)/60:.1f} min")
    log("scan done")


if __name__ == '__main__':
    main()
