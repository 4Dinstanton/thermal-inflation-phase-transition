#!/usr/bin/env python3
"""3D-only cosmic-string plots from CosmoLattice NPZ snapshots."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from postprocess.revisualize_snapshots import (  # noqa: E402
    load_field_state,
    plot_strings_3d,
    _identify_strings,
)


def _load_metadata(run_dir):
    path = os.path.join(run_dir, "simulation_metadata.npz")
    if os.path.isfile(path):
        return dict(np.load(path, allow_pickle=True))
    return None


def _plot_empty_3d(state, metadata, out_path):
    step = state["step"]
    T_val = state["temperature"]
    mu = 1000.0
    if metadata is not None:
        if "mu" in metadata:
            mu = float(metadata["mu"])
        elif "mphi" in metadata:
            mu = float(metadata["mphi"])
    time_phys = state["time"] / mu
    winding = np.asarray(state["winding"])
    nx, ny, nz = winding.shape

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"Cosmic Strings (3D) | Step {step:,} | t={time_phys:.2e} | T={T_val:.1f}\n"
        f"No strings detected",
        fontsize=11,
        pad=12,
    )
    ax.text2D(
        0.5,
        0.5,
        "No strings",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="0.5",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_run(run_dir, skip_existing=False, empty_placeholders=True):
    run_dir = os.path.abspath(run_dir)
    meta = _load_metadata(run_dir)
    out_dir = os.path.join(run_dir, "strings3d")
    os.makedirs(out_dir, exist_ok=True)

    npzs = sorted(glob.glob(os.path.join(run_dir, "field_states", "state_step_*.npz")))
    if not npzs:
        print(f"No NPZ in {run_dir}/field_states")
        return

    n_ok = n_empty = n_skip = 0
    print(f"3D string analysis: {len(npzs)} snapshots -> {out_dir}")

    for i, npz_path in enumerate(npzs, 1):
        step = int(os.path.basename(npz_path).replace("state_step_", "").replace(".npz", ""))
        out_path = os.path.join(out_dir, f"strings3d_step_{step:010d}.png")
        if skip_existing and os.path.isfile(out_path):
            n_skip += 1
            continue

        state = load_field_state(npz_path)
        if not state.get("complex") or state.get("winding") is None:
            print(f"  [{i}/{len(npzs)}] step {step}: skip (no winding)")
            continue

        winding = np.asarray(state["winding"])
        n_vox = int(np.sum(np.abs(winding) > 0.5))
        labelled, strings = _identify_strings(winding)

        if len(strings) == 0:
            if empty_placeholders:
                _plot_empty_3d(state, meta, out_path)
                n_empty += 1
                print(f"  [{i}/{len(npzs)}] step {step}: empty placeholder")
            else:
                print(f"  [{i}/{len(npzs)}] step {step}: no strings")
            continue

        plot_strings_3d(state, meta, out_path, labelled=labelled, strings=strings)
        n_ok += 1
        print(f"  [{i}/{len(npzs)}] step {step}: {n_vox} voxels, {len(strings)} loops")

    n_png = len(glob.glob(os.path.join(out_dir, "strings3d_step_*.png")))
    print(f"Done: ok={n_ok} empty={n_empty} skipped={n_skip}  PNGs={n_png}")


def main():
    ap = argparse.ArgumentParser(description="3D-only string plots from NPZ")
    ap.add_argument("run_dir")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-empty", action="store_true", help="Skip zero-string placeholders")
    args = ap.parse_args()
    process_run(
        args.run_dir,
        skip_existing=args.skip_existing,
        empty_placeholders=not args.no_empty,
    )


if __name__ == "__main__":
    main()
