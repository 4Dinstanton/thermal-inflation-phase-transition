#!/usr/bin/env python3
"""CosmoLattice HDF5 field snapshots → cosmic-string analysis.

Primary path: read ``field_snapshot.h5`` + ``manifest.csv`` directly (no NPZ).
Optional ``export`` command still writes NPZ for revisualize_snapshots.py.

  **analyze** — winding + ``strings/string_summary.csv`` + optional PNGs
  **export**  — HDF5 → ``state_step_*.npz`` (optional)
  **all**     — export then analyze

Examples
--------
  # Strings from HDF5 only (recommended on KISTI)
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --workers 68

  # With PNGs
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --workers 8

  # Optional NPZ export for revisualize_snapshots.py
  python tools/cl_hdf5_string_pipeline.py export <run_dir> --workers 1
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.cl_field_snapshot_io import (
    estimate_ram_gb,
    load_manifest_rows,
    read_h5_snapshot,
    resolve_h5_path,
    build_h5_time_index,
)
from tools.export_cl_snapshots import load_run_params, write_metadata
from tools.winding import string_voxel_fraction


def _filter_rows(
    rows: List[Dict[str, Any]],
    step_min: Optional[int],
    step_max: Optional[int],
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        step = int(float(row["step"]))
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        out.append(row)
    return out


def _npz_path(state_dir: str, step: int) -> str:
    return os.path.join(state_dir, f"state_step_{step:010d}.npz")


def _save_npz_from_snap(snap: Dict[str, Any], out_path: str, hubble: bool = True) -> None:
    if snap.get("n_scalars", 1) >= 2:
        phi1 = snap["phi1"]
        phi2 = snap["phi2"]
        rho = snap["rho"]
        winding = snap["winding"]
        save_dict = {
            "phi1": phi1.astype(np.float32),
            "phi2": phi2.astype(np.float32),
            "rho": rho.astype(np.float32),
            "theta": snap["theta"].astype(np.float32),
            "winding": winding.astype(np.float32),
            "step": snap["step"],
            "time": snap["time"],
            "temperature": snap["temperature"],
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
        }
    else:
        phi = snap["phi"]
        save_dict = {
            "phi": phi.astype(np.float32),
            "step": snap["step"],
            "time": snap["time"],
            "temperature": snap["temperature"],
            "phi_min": float(phi.min()),
            "phi_max": float(phi.max()),
        }
    if hubble:
        save_dict["scale_factor"] = snap["a"]
        save_dict["hubble"] = snap["H"]
    np.savez_compressed(out_path, **save_dict)


def _export_one(
    args: Tuple[str, Dict[str, Any], str, str, bool, bool],
) -> Tuple[int, str, str]:
    """Worker: (h5_path, row, state_dir, time_index_json, skip_existing, hubble)."""
    h5_path, row, state_dir, _, skip_existing, hubble = args
    step = int(float(row["step"]))
    out_path = _npz_path(state_dir, step)
    if skip_existing and os.path.isfile(out_path):
        return step, "skip", out_path
    try:
        # Rebuild time index in worker (avoid pickling h5py objects)
        time_index = build_h5_time_index(h5_path, "phi_0")
        snap = read_h5_snapshot(h5_path, row, time_index=time_index)
        _save_npz_from_snap(snap, out_path, hubble=hubble)
        return step, "ok", out_path
    except Exception as exc:
        return step, f"fail:{exc}", out_path


def _metrics_from_snap(snap: Dict[str, Any]) -> Dict[str, Any]:
    winding = np.asarray(snap["winding"])
    n_string_vox = int(np.sum(np.abs(winding) > 0.5))
    return {
        "step": snap["step"],
        "time": snap["time"],
        "temperature": snap["temperature"],
        "n_string_voxels": n_string_vox,
        "string_voxel_fraction": string_voxel_fraction(winding),
    }


def _load_or_build_metadata(run_dir: str, n_snapshots: int = 0) -> Dict[str, Any]:
    """Load simulation_metadata.npz or build from cl_run_params.json (no field NPZ)."""
    meta_path = os.path.join(run_dir, "simulation_metadata.npz")
    if os.path.isfile(meta_path):
        return dict(np.load(meta_path, allow_pickle=True))
    params = load_run_params(run_dir)
    if not params:
        return {"mu": 1000.0, "mphi": 1000.0, "n_scalars": 2}
    write_metadata(run_dir, params, n_snapshots)
    return dict(np.load(meta_path, allow_pickle=True))


# Skip expensive loop labeling / 3D scatter above these counts (1024³ early TI).
DENSE_STRING_VOX = 500_000
MAX_3D_STRING_VOX = 80_000


def _analyze_one_h5(
    args: Tuple[str, Dict[str, Any], str, bool, Optional[Dict[str, Any]], Optional[Dict[float, str]]],
) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    h5_path, row, run_dir, do_plots, metadata, time_index = args
    step = int(float(row["step"]))
    try:
        if time_index is None:
            time_index = build_h5_time_index(h5_path, "phi_0")
        snap = read_h5_snapshot(h5_path, row, time_index=time_index)
        metrics = _metrics_from_snap(snap)
        if do_plots:
            from postprocess.revisualize_snapshots import (
                _plot_strings_2d_dense,
                plot_strings_2d,
                plot_strings_3d,
            )
            meta = metadata if metadata is not None else {}
            state = {
                "phi1": snap["phi1"],
                "phi2": snap["phi2"],
                "rho": snap["rho"],
                "theta": snap["theta"],
                "winding": snap["winding"],
                "step": snap["step"],
                "time": snap["time"],
                "temperature": snap["temperature"],
                "complex": True,
            }
            strings_dir = os.path.join(run_dir, "strings")
            strings3d_dir = os.path.join(run_dir, "strings3d")
            os.makedirs(strings_dir, exist_ok=True)
            os.makedirs(strings3d_dir, exist_ok=True)
            png2d = os.path.join(strings_dir, f"strings_step_{step:010d}.png")
            png3d = os.path.join(strings3d_dir, f"strings3d_step_{step:010d}.png")
            n_string_vox = int(metrics["n_string_voxels"])
            if n_string_vox > DENSE_STRING_VOX:
                _plot_strings_2d_dense(state, meta, png2d, n_string_vox)
                metrics["n_loops"] = -1
            else:
                strings = plot_strings_2d(state, meta, png2d)
                metrics["n_loops"] = len(strings) if strings else 0
                if strings and n_string_vox <= MAX_3D_STRING_VOX:
                    plot_strings_3d(
                        state, meta, png3d, labelled=None, strings=strings
                    )
            del state, snap
        else:
            metrics["n_loops"] = ""
            del snap
        gc.collect()
        return step, "ok", metrics
    except Exception as exc:
        gc.collect()
        return step, f"fail:{exc}", None


def export_hdf5(
    run_dir: str,
    workers: int = 1,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    skip_existing: bool = True,
    hubble: bool = True,
) -> int:
    run_dir = os.path.abspath(run_dir)
    state_dir = os.path.join(run_dir, "field_states")
    os.makedirs(state_dir, exist_ok=True)

    rows = _filter_rows(load_manifest_rows(run_dir), step_min, step_max)
    if not rows:
        print("No manifest rows to export.")
        return 0

    h5_path = resolve_h5_path(run_dir, rows)
    N_hint = None
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            if "phi_0" in f and len(f["phi_0"].keys()) > 0:
                k0 = next(iter(f["phi_0"].keys()))
                shp = f["phi_0"][k0].shape
                if len(shp) == 3:
                    N_hint = shp[0]
                elif len(shp) == 1:
                    N_hint = int(round(shp[0] ** (1.0 / 3.0)))
    except Exception:
        pass

    if N_hint:
        ram = estimate_ram_gb(N_hint, n_scalars=2)
        print(f"HDF5: {h5_path}")
        print(f"Grid N≈{N_hint}  est. RAM per worker ≈ {ram:.1f} GB")
        if ram > 12 and workers > 1:
            print(
                f"WARNING: {workers} workers × {ram:.1f} GB may OOM; "
                f"consider --workers 1"
            )

    print(f"Exporting {len(rows)} snapshots -> {state_dir}")
    tasks = [
        (h5_path, row, state_dir, "", skip_existing, hubble)
        for row in rows
    ]

    n_ok = 0
    if workers <= 1:
        for task in tasks:
            step, status, path = _export_one(task)
            print(f"  step {step}: {status} {path}", flush=True)
            if status == "ok" or status == "skip":
                n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_export_one, t): t for t in tasks}
            for fut in as_completed(futures):
                step, status, path = fut.result()
                print(f"  step {step}: {status}", flush=True)
                if status == "ok" or status == "skip":
                    n_ok += 1

    params = load_run_params(run_dir)
    meta_path = write_metadata(run_dir, params, n_ok)
    print(f"Metadata: {meta_path}")
    print(f"Exported/skipped {n_ok}/{len(rows)} snapshots")
    return n_ok


def analyze_hdf5(
    run_dir: str,
    workers: int = 1,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    skip_plots: bool = False,
    metrics_only: bool = False,
    from_npz: bool = False,
) -> int:
    run_dir = os.path.abspath(run_dir)
    strings_dir = os.path.join(run_dir, "strings")
    os.makedirs(strings_dir, exist_ok=True)
    summary_path = os.path.join(strings_dir, "string_summary.csv")

    if from_npz:
        from tools.compute_strings_cl import process_run
        process_run(
            run_dir,
            step_min=step_min,
            step_max=step_max,
            skip_plots=skip_plots or metrics_only,
        )
        return 1

    rows = _filter_rows(load_manifest_rows(run_dir), step_min, step_max)
    if not rows:
        print("No manifest rows to analyze.")
        return 0

    metadata = _load_or_build_metadata(run_dir, len(rows))
    n_scalars = int(metadata.get("n_scalars", int(float(rows[0]["n_scalars"]))))
    if n_scalars < 2:
        print("Run has n_scalars=1; string analysis needs complex (phi1+phi2).")
        return 0

    metrics_rows: List[Dict[str, Any]] = []
    h5_path = resolve_h5_path(run_dir, rows)
    do_plots = not skip_plots and not metrics_only

    N_hint = None
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            if "phi_0" in f and len(f["phi_0"].keys()) > 0:
                k0 = next(iter(f["phi_0"].keys()))
                shp = f["phi_0"][k0].shape
                if len(shp) == 3:
                    N_hint = shp[0]
                elif len(shp) == 1:
                    N_hint = int(round(shp[0] ** (1.0 / 3.0)))
    except Exception:
        pass
    if N_hint:
        ram = estimate_ram_gb(N_hint, n_scalars=2)
        print(f"HDF5: {h5_path}")
        print(f"Grid N≈{N_hint}  est. RAM per worker ≈ {ram:.1f} GB")
        if ram > 12 and workers > 1:
            print(
                f"WARNING: {workers} workers × {ram:.1f} GB may OOM; "
                f"consider fewer workers"
            )

    print(f"Analyzing {len(rows)} HDF5 snapshots (no NPZ export)")
    time_index = build_h5_time_index(h5_path, "phi_0")
    tasks = [
        (h5_path, row, run_dir, do_plots, metadata, time_index)
        for row in rows
    ]

    if workers <= 1:
        for task in tasks:
            step, status, metrics = _analyze_one_h5(task)
            if metrics:
                metrics_rows.append(metrics)
                print(
                    f"  step {step}: {metrics['n_string_voxels']} string voxels "
                    f"({metrics['string_voxel_fraction']:.2e})",
                    flush=True,
                )
            else:
                print(f"  step {step}: {status}", flush=True)
            gc.collect()
    else:
        # Parallel workers cannot share time_index safely; each rebuilds it.
        tasks_par = [
            (h5_path, row, run_dir, do_plots, metadata, None)
            for row in rows
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_analyze_one_h5, t): t for t in tasks_par}
            for fut in as_completed(futures):
                step, status, metrics = fut.result()
                if metrics:
                    metrics_rows.append(metrics)
                    print(
                        f"  step {step}: {metrics['n_string_voxels']} voxels",
                        flush=True,
                    )
                else:
                    print(f"  step {step}: {status}", flush=True)

    metrics_rows.sort(key=lambda r: r["step"])
    fieldnames = [
        "step", "time", "temperature", "n_loops",
        "n_string_voxels", "string_voxel_fraction",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Wrote {summary_path} ({len(metrics_rows)} rows)")
    return len(metrics_rows)


def main():
    ap = argparse.ArgumentParser(
        description="CosmoLattice HDF5 → string analysis (optional NPZ export)",
    )
    ap.add_argument(
        "command",
        choices=["export", "analyze", "all"],
        help="analyze=strings from HDF5; export=NPZ only; all=both",
    )
    ap.add_argument("run_dir", help="CosmoLattice output directory")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel snapshot workers (default 1 for 1024³)")
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Re-export even if state_step_*.npz exists")
    ap.add_argument("--skip-plots", action="store_true",
                    help="Skip 2D/3D string PNGs")
    ap.add_argument("--metrics-only", action="store_true",
                    help="CSV only, no PNGs (still reads HDF5)")
    ap.add_argument("--from-npz", action="store_true",
                    help="Analyze existing NPZ (skip HDF5 read)")
    args = ap.parse_args()

    skip_existing = not args.no_skip_existing
    run_dir = os.path.abspath(args.run_dir)

    if args.command in ("export", "all"):
        export_hdf5(
            run_dir,
            workers=args.workers,
            step_min=args.step_min,
            step_max=args.step_max,
            skip_existing=skip_existing,
        )

    if args.command in ("analyze", "all"):
        analyze_hdf5(
            run_dir,
            workers=args.workers,
            step_min=args.step_min,
            step_max=args.step_max,
            skip_plots=args.skip_plots,
            metrics_only=args.metrics_only,
            from_npz=args.from_npz and args.command == "analyze",
        )


if __name__ == "__main__":
    main()
