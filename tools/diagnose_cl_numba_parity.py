#!/usr/bin/env python3
"""
Compare CosmoLattice vs numba lattice runs: fluctuations, snapshots, T_p readiness.

Usage
-----
    python tools/diagnose_cl_numba_parity.py \\
        data/lattice/set8/256x256x256_T0_1600_..._stochasticrk_V_correct_CL \\
        data/lattice/set8/256x256x256_T0_1600_..._rk2_fused_inline_V_correct
"""
import argparse
import csv
import glob
import json
import os
import re
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR = struct.calcsize("<IIq5d")


def read_raw_max(path):
    with open(path, "rb") as f:
        magic, n, step, t, T, a, H, fStar = struct.unpack("<IIq5d", f.read(HDR))
        phi = np.frombuffer(f.read(4 * n**3), dtype=np.float32).reshape(n, n, n) * fStar
    return dict(step=int(step), T=float(T), a=float(a), max_abs=float(np.max(np.abs(phi))), std=float(phi.std()))


def load_npz_stats(path):
    d = np.load(path)
    phi = d["phi"]
    return dict(
        step=int(d["step"]),
        T=float(d["temperature"]),
        max_abs=float(np.max(np.abs(phi))),
        std=float(phi.std()),
        frac1e4=float(np.mean(np.abs(phi) > 1e4)),
    )


def snapshot_stats(run_dir):
    state_dir = os.path.join(run_dir, "field_states")
    rows = []
    for path in sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz"))):
        rows.append(("npz", os.path.basename(path), load_npz_stats(path)))
    manifest = os.path.join(state_dir, "manifest.csv")
    if os.path.isfile(manifest):
        for line in open(manifest):
            if line.startswith("step,"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            raw = os.path.join(state_dir, parts[-1])
            if os.path.isfile(raw):
                rows.append(("raw", parts[-1], read_raw_max(raw)))
    rows.sort(key=lambda r: r[2]["step"])
    return rows


def blowup_report(run_dir, label, infreq_period=1000):
    """Detect lattice-wide blow-ups and correlate with infrequent spectra cadence."""
    snaps = snapshot_stats(run_dir)
    if not snaps:
        print(f"  {label}: no snapshots")
        return []

    bad = [s for _, _, s in snaps if s["frac1e4"] > 0.1]
    print(f"\n  {label} blow-up analysis ({len(bad)}/{len(snaps)} frames with >10% sites |phi|>1e4 GeV)")
    if bad:
        steps = [s["step"] for s in bad]
        print(f"    steps: {steps[:12]}{'...' if len(steps) > 12 else ''}")
        if infreq_period:
            triggers = sorted(set((s // infreq_period) * infreq_period for s in steps if s >= infreq_period))
            print(f"    likely spectra triggers (every {infreq_period} steps): {triggers[:8]}{'...' if len(triggers)>8 else ''}")

    avg_path = os.path.join(run_dir, "average_scalar_0.txt")
    if os.path.isfile(avg_path):
        d = np.loadtxt(avg_path)
        if d.ndim == 1:
            d = d[None, :]
        fstar = 1.0e15
        phi_rms = np.sqrt(d[:, 3]) * fstar
        spike_rows = np.where(phi_rms > 1000)[0]
        if len(spike_rows):
            print(f"    average_scalar phi_rms spikes at steps: {[int(i * 100) for i in spike_rows[:8]]}")

    return bad


def parse_input_in(run_dir):
    path = os.path.join(run_dir, "input.in")
    if not os.path.isfile(path):
        return {}
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\w+)\s*=\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def infreq_period_from_input(inp):
    dt = float(inp.get("dt", 0.1))
    t_out = float(inp.get("tOutputInfreq", 100))
    return max(1, int(round(t_out / dt)))


def T_p(run_dir):
    path = os.path.join(run_dir, "view3d", "bubble_summary.csv")
    if not os.path.isfile(path):
        return None, None, path
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        if float(r["volume_fraction"]) > 0.99999:
            return float(r["temperature"]), int(float(r["step"])), path
    return None, None, path


def print_run(label, run_dir):
    print(f"\n{'='*60}\n{label}: {os.path.relpath(run_dir, REPO)}\n{'='*60}")
    params = os.path.join(run_dir, "cl_run_params.json")
    if os.path.isfile(params):
        p = json.load(open(params))
        print(f"  steps={p.get('steps')}  phi_threshold={p.get('phi_threshold')}")
    inp = parse_input_in(run_dir)
    for key in ("include_cw", "phi_threshold", "tMax", "tOutputInfreq", "dt", "thermal_noise"):
        if key in inp:
            print(f"  {key} = {inp[key]}")
    snaps = snapshot_stats(run_dir)
    print(f"  snapshots: {len(snaps)}")
    for kind, name, s in snaps[:3]:
        print(f"    [{kind}] {name}: step={s['step']} T={s['T']:.2f} max|phi|={s['max_abs']:.4g} std={s['std']:.4g}")
    tp, st, p = T_p(run_dir)
    if tp is not None:
        print(f"  T_p = {tp:.2f} GeV  (step {st})")
    else:
        print(f"  T_p: not reached  (view3d csv: {os.path.relpath(p, REPO) if p else 'missing'})")


def main():
    ap = argparse.ArgumentParser(description="Diagnose CL vs numba parity")
    ap.add_argument("cl_dir", help="CosmoLattice run directory")
    ap.add_argument("numba_dir", help="Numba reference run directory")
    args = ap.parse_args()

    print_run("CosmoLattice", args.cl_dir)
    print_run("Numba", args.numba_dir)

    cl_inp = parse_input_in(args.cl_dir)
    period = infreq_period_from_input(cl_inp)
    print(f"\n{'='*60}\nBlow-up correlation (tOutputInfreq/dt = {period} steps)\n{'='*60}")
    cl_bad = blowup_report(args.cl_dir, "CL", infreq_period=period)
    nb_bad = blowup_report(args.numba_dir, "Numba", infreq_period=period)

    cl = snapshot_stats(args.cl_dir)
    nb = snapshot_stats(args.numba_dir)
    print(f"\n{'='*60}\nTemperature-matched comparison\n{'='*60}")
    if cl and nb:
        for cs in cl[-5:]:
            s = cs[2]
            near = min(nb, key=lambda r: abs(r[2]["T"] - s["T"]))
            ns = near[2]
            print(
                f"  T~{s['T']:.0f} GeV: CL max|phi|={s['max_abs']:.4g} (step {s['step']})  "
                f"numba max|phi|={ns['max_abs']:.4g} (step {ns['step']})"
            )

    print(f"\n{'='*60}\nDiagnosis\n{'='*60}")
    cl_artificial = cl_bad and period and all(
        abs(s - round(s / period) * period) < 120 or s < 100 for s in (x["step"] for x in cl_bad)
    )
    if cl_artificial:
        print(
            "  CL shows periodic lattice-wide excursions (~100 steps after every\n"
            f"  tOutputInfreq/dt = {period} steps), not localized nucleation bubbles.\n"
            "  Root cause: infrequent power-spectrum measurements FFT fldS/piS in place\n"
            "  without restoring configuration space / ghost cells before evolve.\n"
            "  Fix: refreshFieldsAfterMeasurement() after measurer.measure (patched).\n"
            "  Rebuild: python simulation/run_cosmolattice.py --install && re-run CL."
        )
    elif cl_bad and nb_bad:
        print("  Both runs show large-|phi| excursions; check whether CL timing matches nucleation.")
    elif cl_bad:
        print("  CL snapshots show large-|phi| excursions; inspect step timing vs numba.")
    else:
        print("  No CL blow-up pattern detected in snapshots.")


if __name__ == "__main__":
    main()
