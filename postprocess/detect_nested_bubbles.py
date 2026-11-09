#!/usr/bin/env python3
"""Detect nested bubbles (v inside a inside 0) in two_step lattice snapshots.

Each site is classified by rho = |phi| against the barrier positions of
V(phi, T) (potential/twoStepPotential.py):

    phase 0 (false)        rho <  b1(T)
    phase 1 (intermediate) b1  <= rho < b2(T)
    phase 2 (true, v)      rho >= b2(T)

Connected regions of each phase are labelled with periodic boundaries. A
v-region is *nested* when at least ``--nest_frac`` of its boundary faces touch
the a-phase, *direct* (0 -> v) when most of its faces touch the false vacuum,
and *mixed* otherwise.

Usage:
    python postprocess/detect_nested_bubbles.py data/lattice/two_step/<run>
    python postprocess/detect_nested_bubbles.py --self_test

Outputs in <run>/nested_bubbles/: nested_summary.csv, nested_v_regions.csv,
nested_timeseries.png and a phase slice through the largest nested bubble.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PHASE_NAMES = ("0", "a", "v")
_DIRS = [(ax, s) for ax in range(3) for s in (1, -1)]


# ---------------------------------------------------------------- labelling
def periodic_label(mask):
    """ndimage.label with 6-connectivity and periodic boundaries."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ax in range(3):
        lo = np.take(lab, 0, axis=ax)
        hi = np.take(lab, -1, axis=ax)
        both = (lo > 0) & (hi > 0)
        for i, j in zip(lo[both], hi[both]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, new = np.unique(roots, return_inverse=True)
    return new[lab], len(uniq) - 1


def boundary_contacts(labels, n_labels, phase, target_phase):
    """Per region of ``labels``: number of boundary faces touching each phase.

    Returns an (n_labels + 1, 3) int array indexed by [label, neighbour phase].
    Only faces whose neighbour is *not* ``target_phase`` are counted.
    """
    counts = np.zeros((n_labels + 1, 3), dtype=np.int64)
    inside = labels > 0
    for ax, s in _DIRS:
        nb = np.roll(phase, -s, axis=ax)
        face = inside & (nb != target_phase)
        if not face.any():
            continue
        idx = labels[face] * 3 + nb[face]
        counts += np.bincount(idx, minlength=(n_labels + 1) * 3).reshape(n_labels + 1, 3)
    return counts


def region_centroids(labels, n_labels):
    """Periodic centroids (cells) via the circular mean on each axis."""
    N = labels.shape
    cents = np.zeros((n_labels + 1, 3))
    idx = np.arange(1, n_labels + 1)
    for ax in range(3):
        ang = 2 * np.pi * np.indices(N)[ax] / N[ax]
        c = ndimage.sum(np.cos(ang), labels, idx)
        s = ndimage.sum(np.sin(ang), labels, idx)
        cents[1:, ax] = (np.arctan2(s, c) % (2 * np.pi)) * N[ax] / (2 * np.pi)
    return cents


# ---------------------------------------------------------------- analysis
def thresholds(pot, T, th1=None, th2=None):
    """Barrier positions b1 (0|a) and b2 (a|v) at temperature T."""
    vac = pot.vacua(T)
    b1 = th1 if th1 is not None else (vac["b1"].phi if vac["b1"] else 0.5 * pot.a)
    if th2 is not None:
        b2 = th2
    elif vac["b2"]:
        b2 = vac["b2"].phi
    else:
        b2 = np.sqrt(pot.a * pot.v)
    return float(b1), float(b2)


def classify(rho, b1, b2):
    phase = np.zeros(rho.shape, dtype=np.int8)
    phase[rho >= b1] = 1
    phase[rho >= b2] = 2
    return phase


def analyze_field(rho, b1, b2, dx=1.0, smooth=1.0, nest_frac=0.8, min_cells=8):
    """Classify, label and categorise v-regions. Returns (summary, v_rows, phase, vlab)."""
    if smooth > 0:
        rho = ndimage.gaussian_filter(rho, smooth, mode="wrap")
    phase = classify(rho, b1, b2)
    alab, n_a = periodic_label(phase == 1)
    vlab, n_v = periodic_label(phase == 2)
    vol = np.bincount(vlab.ravel(), minlength=n_v + 1)
    avol = np.bincount(alab.ravel(), minlength=n_a + 1)
    contacts = boundary_contacts(vlab, n_v, phase, 2)
    a_contacts = boundary_contacts(alab, n_a, phase, 1)
    cents = region_centroids(vlab, n_v) if n_v else np.zeros((1, 3))

    # Host a-region of each v-region: the a label most often across its faces.
    host = np.zeros(n_v + 1, dtype=np.int64)
    if n_v and n_a:
        inside = vlab > 0
        pair = np.zeros((n_v + 1, n_a + 1), dtype=np.int64)
        for ax, s in _DIRS:
            nb_lab = np.roll(alab, -s, axis=ax)
            m = inside & (nb_lab > 0)
            np.add.at(pair, (vlab[m], nb_lab[m]), 1)
        host = pair.argmax(axis=1)

    rows = []
    for i in range(1, n_v + 1):
        if vol[i] < min_cells:
            continue
        c0, ca, _ = contacts[i]
        tot = max(c0 + ca, 1)
        fa = ca / tot
        kind = "nested" if fa >= nest_frac else ("direct" if c0 / tot >= nest_frac else "mixed")
        rows.append({
            "label": i, "cells": int(vol[i]),
            "R_eq": float((3 * vol[i] / (4 * np.pi)) ** (1 / 3) * dx),
            "frac_faces_a": float(fa), "frac_faces_0": float(c0 / tot), "kind": kind,
            "host_a_label": int(host[i]) if fa > 0 else 0,
            "host_a_cells": int(avol[host[i]]) if fa > 0 and host[i] > 0 else 0,
            "cx": cents[i, 0], "cy": cents[i, 1], "cz": cents[i, 2],
        })
    big_a = [j for j in range(1, n_a + 1) if avol[j] >= min_cells]
    hosts = {r["host_a_label"] for r in rows if r["kind"] == "nested"}
    summary = {
        "b1": b1, "b2": b2,
        "f0": float(np.mean(phase == 0)), "fa": float(np.mean(phase == 1)), "fv": float(np.mean(phase == 2)),
        "n_a": len(big_a),
        "n_a_touching_0": int(sum(a_contacts[j, 0] > 0 for j in big_a)),
        "n_a_hosting_v": len(hosts),
        "n_v": len(rows),
        "n_v_nested": sum(r["kind"] == "nested" for r in rows),
        "n_v_direct": sum(r["kind"] == "direct" for r in rows),
        "n_v_mixed": sum(r["kind"] == "mixed" for r in rows),
        "R_v_max": max((r["R_eq"] for r in rows), default=0.0),
    }
    return summary, rows, phase, vlab


# ---------------------------------------------------------------- run I/O
def load_potential(run_dir):
    from potential.twoStepPotential import TwoStepPotential

    with open(os.path.join(run_dir, "cl_run_params.json")) as f:
        p = json.load(f)
    if p.get("potential_type") != "two_step":
        sys.exit(f"ERROR: {run_dir} is not a two_step run (potential_type={p.get('potential_type')})")
    pot = TwoStepPotential.from_vacua(V0=p["ts_V0"], a=p["ts_a"], v=p["ts_v"],
                                      delV1_frac=p["ts_delV1_frac"], bosons=((p["nb"], p["ts_g"]),))
    return pot, p


def snapshot_files(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "field_states", "state_step_*.npz")))
    return [f for f in files if "_NaN_debug" not in f]


def load_rho(path):
    d = np.load(path)
    rho = np.asarray(d["rho"] if "rho" in d.files else np.abs(d["phi"]), dtype=np.float64)
    step = int(d["step"]) if "step" in d.files else int(re.search(r"(\d+)", os.path.basename(path)).group(1))
    return rho, step, float(d["time"]), float(d["temperature"])


def plot_slice(phase, vlab, row, title, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    z = int(round(row["cz"])) % phase.shape[2]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = ListedColormap(["#1f3b73", "#f2a541", "#c0392b"])
    im = ax.imshow(phase[:, :, z].T, origin="lower", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax.contour((vlab[:, :, z] == row["label"]).T, levels=[0.5], colors="white", linewidths=1.2)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["0 (false)", "a", "v"])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timeseries(summ, out):
    import matplotlib.pyplot as plt

    t = [s["time"] for s in summ]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, lab in (("f0", "false"), ("fa", "a"), ("fv", "v")):
        axs[0].plot(t, [s[k] for s in summ], label=lab)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-7, 1.5)
    axs[0].set_xlabel("program time")
    axs[0].set_ylabel("volume fraction")
    axs[0].legend()
    for k, lab in (("n_a", "a regions"), ("n_v_nested", "nested v"), ("n_v_direct", "direct 0→v"),
                   ("n_v_mixed", "mixed v")):
        axs[1].plot(t, [s[k] for s in summ], label=lab, marker=".")
    axs[1].set_xlabel("program time")
    axs[1].set_ylabel("count")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(args):
    pot, params = load_potential(args.run_dir)
    files = snapshot_files(args.run_dir)
    if not files:
        sys.exit(f"ERROR: no field_states/state_step_*.npz in {args.run_dir} "
                 "(run tools/export_cl_snapshots.py first)")
    out_dir = os.path.join(args.run_dir, "nested_bubbles")
    os.makedirs(out_dir, exist_ok=True)
    dx = params["dx_phys"]
    summ, vrows, best = [], [], None
    for path in files[:: args.stride]:
        rho, step, t, T = load_rho(path)
        b1, b2 = thresholds(pot, T, args.th1, args.th2)
        s, rows, phase, vlab = analyze_field(rho, b1, b2, dx=dx, smooth=args.smooth,
                                             nest_frac=args.nest_frac, min_cells=args.min_cells)
        s.update(step=step, time=t, T=T)
        summ.append(s)
        for r in rows:
            vrows.append({"step": step, "time": t, **r})
        nested = [r for r in rows if r["kind"] == "nested"]
        if nested:
            top = max(nested, key=lambda r: r["cells"])
            if best is None or top["cells"] > best[0]["cells"]:
                best = (top, phase, vlab, step, t)
        print(f"step {step:>10d} T={T:.3e}  f0={s['f0']:.3f} fa={s['fa']:.3f} fv={s['fv']:.2e}  "
              f"a={s['n_a']} v={s['n_v']} (nested {s['n_v_nested']}, direct {s['n_v_direct']}, "
              f"mixed {s['n_v_mixed']})", flush=True)

    keys = ["step", "time", "T", "b1", "b2", "f0", "fa", "fv", "n_a", "n_a_touching_0", "n_a_hosting_v",
            "n_v", "n_v_nested", "n_v_direct", "n_v_mixed", "R_v_max"]
    with open(os.path.join(out_dir, "nested_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summ)
    if vrows:
        with open(os.path.join(out_dir, "nested_v_regions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(vrows[0].keys()))
            w.writeheader()
            w.writerows(vrows)
    plot_timeseries(summ, os.path.join(out_dir, "nested_timeseries.png"))
    if best is not None:
        row, phase, vlab, step, t = best
        plot_slice(phase, vlab, row,
                   f"step {step}  t={t:.4g}: nested v-region (R_eq={row['R_eq']:.3g} GeV$^{{-1}}$, "
                   f"{100 * row['frac_faces_a']:.0f}% faces on a)",
                   os.path.join(out_dir, f"nested_slice_step_{step:010d}.png"))
    n_nested = sum(s["n_v_nested"] for s in summ)
    print(f"\n{len(summ)} snapshots; snapshots with nested v: "
          f"{sum(s['n_v_nested'] > 0 for s in summ)}; total nested detections: {n_nested}")
    print(f"Wrote {out_dir}")


# ---------------------------------------------------------------- self test
def self_test():
    """Synthetic field: nested (v in a), direct (v in 0), empty a, a wrapping the box edge."""
    N, a, v = 96, 3e6, 3e7
    rng = np.random.default_rng(0)
    rho = np.abs(rng.normal(0, 2e4, (N, N, N)))
    g = np.indices((N, N, N))

    def ball(c, r):
        d2 = sum(np.minimum(np.abs(g[i] - c[i]), N - np.abs(g[i] - c[i])) ** 2 for i in range(3))
        return d2 <= r * r

    rho[ball((30, 30, 30), 18)] = a
    rho[ball((30, 30, 30), 7)] = v                  # nested
    rho[ball((75, 70, 25), 6)] = v                  # direct 0 -> v
    rho[ball((70, 30, 70), 10)] = a                 # a without v
    rho[ball((0, 60, 60), 10)] = a                  # a across x boundary
    rho[ball((2, 60, 60), 4)] = v                   # nested, across boundary
    s, rows, _, _ = analyze_field(rho, 2e5, 1e7, smooth=0.0)
    print(s)
    for r in rows:
        print({k: r[k] for k in ("label", "cells", "kind", "frac_faces_a", "host_a_label")})
    ok = (s["n_a"] == 3 and s["n_v"] == 3 and s["n_v_nested"] == 2 and s["n_v_direct"] == 1
          and s["n_a_hosting_v"] == 2)
    print("SELF TEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--smooth", type=float, default=1.0, help="Gaussian smoothing sigma (cells) before classifying")
    ap.add_argument("--nest_frac", type=float, default=0.8, help="Min fraction of a-faces for 'nested'")
    ap.add_argument("--min_cells", type=int, default=8, help="Ignore regions smaller than this")
    ap.add_argument("--th1", type=float, default=None, help="Override 0|a threshold (GeV)")
    ap.add_argument("--th2", type=float, default=None, help="Override a|v threshold (GeV)")
    ap.add_argument("--stride", type=int, default=1, help="Use every n-th snapshot")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.run_dir:
        ap.error("run_dir required (or --self_test)")
    run(args)


if __name__ == "__main__":
    main()
