#!/usr/bin/env python3
"""Batch cosmic-string analysis for CosmoLattice NPZ snapshots.

Exports winding (if missing), writes string_summary.csv, and renders 2D/3D
string PNGs using postprocess/revisualize_snapshots.py helpers.

Usage
-----
    python tools/compute_strings_cl.py <run_dir>
    python tools/compute_strings_cl.py <run_dir> --step_min 1000 --step_max 5000
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.winding import compute_winding_number, string_voxel_fraction

# Reuse visualization from revisualize_snapshots
from postprocess.revisualize_snapshots import (
    load_field_state,
    plot_strings_2d,
    plot_strings_3d,
)


def _load_metadata(run_dir):
    meta_path = os.path.join(run_dir, "simulation_metadata.npz")
    if os.path.exists(meta_path):
        return dict(np.load(meta_path, allow_pickle=True))
    return None


def _snapshot_files(state_dir, step_min=None, step_max=None):
    files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    out = []
    for f in files:
        base = os.path.basename(f)
        step = int(base.replace("state_step_", "").replace(".npz", ""))
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        out.append((step, f))
    return out


def ensure_winding_in_npz(npz_path):
    """Add winding/rho/theta to NPZ if phi1/phi2 present but winding missing."""
    data = dict(np.load(npz_path))
    if "phi1" not in data or "phi2" not in data:
        return False
    changed = False
    phi1 = data["phi1"].astype(np.float64)
    phi2 = data["phi2"].astype(np.float64)
    if "winding" not in data:
        data["winding"] = compute_winding_number(phi1, phi2).astype(np.float32)
        changed = True
    if "rho" not in data:
        data["rho"] = np.sqrt(phi1 ** 2 + phi2 ** 2).astype(np.float32)
        changed = True
    if "theta" not in data:
        data["theta"] = np.arctan2(phi2, phi1).astype(np.float32)
        changed = True
    if changed:
        np.savez_compressed(npz_path, **data)
    return True


def process_run(run_dir, step_min=None, step_max=None, skip_plots=False):
    run_dir = os.path.abspath(run_dir)
    state_dir = os.path.join(run_dir, "field_states")
    if not os.path.isdir(state_dir):
        raise FileNotFoundError(f"no field_states/ in {run_dir}")

    metadata = _load_metadata(run_dir)
    n_scalars = int(metadata.get("n_scalars", 1)) if metadata is not None else 1
    if n_scalars < 2:
        print("Run has n_scalars=1 (real scalar); string detection requires phi1+phi2 snapshots.")
        return

    strings_dir = os.path.join(run_dir, "strings")
    strings3d_dir = os.path.join(run_dir, "strings3d")
    os.makedirs(strings_dir, exist_ok=True)
    os.makedirs(strings3d_dir, exist_ok=True)

    summary_path = os.path.join(strings_dir, "string_summary.csv")
    rows = []

    snaps = _snapshot_files(state_dir, step_min, step_max)
    if not snaps:
        print("No NPZ snapshots found. Run tools/export_cl_snapshots.py first.")
        return

    for step, npz_path in snaps:
        if not ensure_winding_in_npz(npz_path):
            continue
        state = load_field_state(npz_path)
        if not state.get("complex") or state.get("winding") is None:
            continue

        winding = np.asarray(state["winding"])
        frac = string_voxel_fraction(winding)

        png2d = os.path.join(strings_dir, f"strings_step_{step:010d}.png")
        png3d = os.path.join(strings3d_dir, f"strings3d_step_{step:010d}.png")

        strings = []
        if not skip_plots:
            strings = plot_strings_2d(state, metadata, png2d)
            plot_strings_3d(state, metadata, png3d)

        n_loops = len(strings) if strings else int(np.sum(np.abs(winding) > 0.5) > 0)
        n_string_vox = int(np.sum(np.abs(winding) > 0.5))
        rows.append({
            "step": step,
            "time": state["time"],
            "temperature": state["temperature"],
            "n_loops": n_loops if strings else "",
            "n_string_voxels": n_string_vox,
            "string_voxel_fraction": frac,
        })
        print(f"  step {step}: {n_string_vox} string voxels ({frac:.2e} fraction)")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "time", "temperature", "n_loops",
                        "n_string_voxels", "string_voxel_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {summary_path} ({len(rows)} snapshots)")


def main():
    ap = argparse.ArgumentParser(description="Cosmic string batch analysis for CL runs")
    ap.add_argument("run_dir", help="CosmoLattice run directory")
    ap.add_argument("--step_min", type=int, default=None)
    ap.add_argument("--step_max", type=int, default=None)
    ap.add_argument("--skip-plots", action="store_true", help="Only write CSV summary")
    args = ap.parse_args()
    process_run(args.run_dir, args.step_min, args.step_max, skip_plots=args.skip_plots)


if __name__ == "__main__":
    main()
