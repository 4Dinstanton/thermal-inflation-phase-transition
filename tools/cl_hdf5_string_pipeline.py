#!/usr/bin/env python3
"""CosmoLattice HDF5 field snapshots → cosmic-string analysis.

Primary path: read ``field_snapshot.h5`` + ``manifest.csv`` directly (no NPZ).
Optional ``export`` command still writes NPZ for revisualize_snapshots.py.
Optional ``split`` writes per-step HDF5 under ``field_states/`` for easier transfer.

  **analyze** — winding + ``strings/string_summary.csv`` + optional PNGs
  **split**   — monolith ``field_snapshot.h5`` → ``field_states/snapshot_step_*.h5``
  **export**  — HDF5 → ``state_step_*.npz`` (optional)
  **all**     — export then analyze

Examples
--------
  # Split large monolith for transfer (on KISTI)
  python tools/cl_hdf5_string_pipeline.py split <run_dir>

  # Strings from HDF5 (monolith or per-step files)
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --workers 1

  # CSV only
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --metrics-only
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.cl_field_snapshot_io import (
    build_h5_time_index,
    estimate_ram_gb,
    load_manifest_rows,
    lookup_h5_key,
    per_step_h5_path,
    read_h5_field,
    read_h5_snapshot,
    resolve_h5_path,
    resolve_snapshot_h5,
    write_per_step_h5,
)
from tools.export_cl_snapshots import load_run_params, write_metadata
from tools.winding import string_voxel_fraction


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fmt_s(dt: float) -> str:
    if dt < 60:
        return f"{dt:.1f}s"
    m, s = divmod(dt, 60.0)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60.0)
    return f"{int(h)}h{int(m)}m{s:02.0f}s"


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


# Plot density cuts — fractions of N³ (matched to old 256³ absolute cuts).
DENSE_STRING_FRAC = 500_000 / (256 ** 3)      # ~0.0298
MAX_3D_STRING_FRAC = 80_000 / (256 ** 3)      # ~0.00477
MAX_3D_STRING_ABS = 2_000_000


def _plot_thresholds(N: int) -> Tuple[int, int]:
    """Return (dense_vox_cut, max_3d_vox_cut) for this lattice size."""
    n3 = int(N) ** 3
    dense = max(500_000, int(DENSE_STRING_FRAC * n3))
    max3d = min(MAX_3D_STRING_ABS, max(80_000, int(MAX_3D_STRING_FRAC * n3)))
    return dense, max3d


def _analyze_one_h5(
    args: Tuple[
        str,
        Dict[str, Any],
        str,
        bool,
        Optional[Dict[str, Any]],
        Optional[Dict[float, str]],
        Optional[str],
        int,
        int,
    ],
) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    (
        h5_path,
        row,
        run_dir,
        do_plots,
        metadata,
        time_index,
        source_kind,
        idx,
        n_total,
    ) = args
    step = int(float(row["step"]))
    tag = f"[{idx}/{n_total}] step {step}"
    t0 = time.time()
    try:
        kind = source_kind or "monolith"
        _log(f"  {tag}: start ({kind})  file={os.path.basename(h5_path)}")

        t_load = time.time()
        if time_index is None and kind == "monolith":
            time_index = build_h5_time_index(h5_path, "phi_0")
        snap = read_h5_snapshot(
            h5_path, row, time_index=time_index if kind == "monolith" else None
        )
        _log(
            f"  {tag}: loaded N={snap.get('N')}  "
            f"T={snap['temperature']:.1f}  ({_fmt_s(time.time() - t_load)})"
        )

        t_met = time.time()
        metrics = _metrics_from_snap(snap)
        n_string_vox = int(metrics["n_string_voxels"])
        _log(
            f"  {tag}: winding done  voxels={n_string_vox:,}  "
            f"frac={metrics['string_voxel_fraction']:.3e}  "
            f"({_fmt_s(time.time() - t_met)})"
        )

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
            N = int(snap.get("N", state["winding"].shape[0]))
            dense_cut, max3d_cut = _plot_thresholds(N)

            t_plot = time.time()
            if n_string_vox > dense_cut:
                _log(
                    f"  {tag}: dense winding (>{dense_cut:,}) → light 2D plot"
                )
                _plot_strings_2d_dense(state, meta, png2d, n_string_vox)
                metrics["n_loops"] = -1
                _log(f"  {tag}: 2D PNG done  ({_fmt_s(time.time() - t_plot)})")
            else:
                _log(f"  {tag}: full 2D labeling + plot …")
                strings = plot_strings_2d(state, meta, png2d)
                metrics["n_loops"] = len(strings) if strings else 0
                _log(
                    f"  {tag}: 2D PNG done  loops={metrics['n_loops']}  "
                    f"({_fmt_s(time.time() - t_plot)})"
                )
                if strings and n_string_vox <= max3d_cut:
                    t3 = time.time()
                    _log(f"  {tag}: 3D PNG …")
                    plot_strings_3d(
                        state, meta, png3d, labelled=None, strings=strings
                    )
                    _log(f"  {tag}: 3D PNG done  ({_fmt_s(time.time() - t3)})")
                elif strings:
                    _log(
                        f"  {tag}: skip 3D (voxels={n_string_vox:,} > {max3d_cut:,})"
                    )
            del state, snap
        else:
            metrics["n_loops"] = ""
            del snap

        gc.collect()
        _log(f"  {tag}: OK total {_fmt_s(time.time() - t0)}")
        return step, "ok", metrics
    except Exception as exc:
        gc.collect()
        _log(f"  {tag}: FAIL after {_fmt_s(time.time() - t0)}: {exc}")
        return step, f"fail:{exc}", None


def _split_one(
    args: Tuple[str, Dict[str, Any], str, Optional[Dict[float, str]], bool, int, int],
) -> Tuple[int, str, str]:
    h5_path, row, out_path, time_index, skip_existing, idx, n_total = args
    step = int(float(row["step"]))
    tag = f"[{idx}/{n_total}] step {step}"
    if skip_existing and os.path.isfile(out_path):
        size_gb = os.path.getsize(out_path) / (1024 ** 3)
        _log(f"  {tag}: skip (exists, {size_gb:.2f} GiB)")
        return step, "skip", out_path
    t0 = time.time()
    try:
        t = float(row["t"])
        n_scalars = int(float(row["n_scalars"]))
        if time_index is None:
            time_index = build_h5_time_index(h5_path, "phi_0")
        time_key = lookup_h5_key(t, time_index)
        _log(f"  {tag}: reading phi_0 (t={time_key}) …")
        phi0 = read_h5_field(h5_path, "phi_0", time_key)
        phi1 = None
        if n_scalars >= 2:
            _log(f"  {tag}: reading phi_1 …")
            phi1 = read_h5_field(h5_path, "phi_1", time_key, N=phi0.shape[0])
        _log(f"  {tag}: writing {os.path.basename(out_path)} …")
        write_per_step_h5(out_path, row, phi0, phi1)
        del phi0, phi1
        gc.collect()
        size_gb = os.path.getsize(out_path) / (1024 ** 3)
        _log(
            f"  {tag}: OK  {size_gb:.2f} GiB  ({_fmt_s(time.time() - t0)})"
        )
        return step, "ok", out_path
    except Exception as exc:
        gc.collect()
        _log(f"  {tag}: FAIL: {exc}")
        return step, f"fail:{exc}", out_path


def split_hdf5(
    run_dir: str,
    workers: int = 1,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    skip_existing: bool = True,
) -> int:
    """Split monolith field_snapshot.h5 into field_states/snapshot_step_*.h5."""
    run_dir = os.path.abspath(run_dir)
    state_dir = os.path.join(run_dir, "field_states")
    os.makedirs(state_dir, exist_ok=True)

    rows = _filter_rows(load_manifest_rows(run_dir), step_min, step_max)
    if not rows:
        _log("No manifest rows to split.")
        return 0

    h5_path = resolve_h5_path(run_dir, rows)
    size_gb = os.path.getsize(h5_path) / (1024 ** 3) if os.path.isfile(h5_path) else 0.0
    _log(f"=== split monolith → per-step HDF5 ===")
    _log(f"  source: {h5_path}  ({size_gb:.1f} GiB)")
    _log(f"  dest:   {state_dir}/snapshot_step_XXXXXXXXXX.h5")
    _log(f"  snaps:  {len(rows)}  workers={workers}")

    time_index = build_h5_time_index(h5_path, "phi_0")
    n_total = len(rows)
    tasks = []
    for i, row in enumerate(rows, start=1):
        step = int(float(row["step"]))
        out_path = per_step_h5_path(run_dir, step)
        ti = time_index if workers <= 1 else None
        tasks.append((h5_path, row, out_path, ti, skip_existing, i, n_total))

    n_ok = 0
    t_all = time.time()
    if workers <= 1:
        for task in tasks:
            step, status, path = _split_one(task)
            if status in ("ok", "skip"):
                n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_split_one, t): t for t in tasks}
            for fut in as_completed(futures):
                step, status, path = fut.result()
                if status in ("ok", "skip"):
                    n_ok += 1

    _log(
        f"Split done: {n_ok}/{n_total}  total wall {_fmt_s(time.time() - t_all)}"
    )
    _log(
        "Tip: rsync individual field_states/snapshot_step_*.h5 + "
        "field_states/manifest.csv + cl_run_params.json"
    )
    return n_ok


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
        _log("No manifest rows to export.")
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
        _log(f"HDF5: {h5_path}")
        _log(f"Grid N≈{N_hint}  est. RAM per worker ≈ {ram:.1f} GB")
        if ram > 12 and workers > 1:
            _log(
                f"WARNING: {workers} workers × {ram:.1f} GB may OOM; "
                f"consider --workers 1"
            )

    _log(f"Exporting {len(rows)} snapshots -> {state_dir}")
    tasks = [
        (h5_path, row, state_dir, "", skip_existing, hubble)
        for row in rows
    ]

    n_ok = 0
    if workers <= 1:
        for task in tasks:
            step, status, path = _export_one(task)
            _log(f"  step {step}: {status} {path}")
            if status == "ok" or status == "skip":
                n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_export_one, t): t for t in tasks}
            for fut in as_completed(futures):
                step, status, path = fut.result()
                _log(f"  step {step}: {status}")
                if status == "ok" or status == "skip":
                    n_ok += 1

    params = load_run_params(run_dir)
    meta_path = write_metadata(run_dir, params, n_ok)
    _log(f"Metadata: {meta_path}")
    _log(f"Exported/skipped {n_ok}/{len(rows)} snapshots")
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
        _log("No manifest rows to analyze.")
        return 0

    metadata = _load_or_build_metadata(run_dir, len(rows))
    n_scalars = int(metadata.get("n_scalars", int(float(rows[0]["n_scalars"]))))
    if n_scalars < 2:
        _log("Run has n_scalars=1; string analysis needs complex (phi1+phi2).")
        return 0

    metrics_rows: List[Dict[str, Any]] = []
    do_plots = not skip_plots and not metrics_only

    # Prefer per-step files when available; fall back to monolith.
    monolith_path: Optional[str] = None
    try:
        monolith_path = resolve_h5_path(run_dir, rows)
    except FileNotFoundError:
        monolith_path = None

    sources: List[Tuple[str, str]] = []
    missing = 0
    for row in rows:
        try:
            path, kind = resolve_snapshot_h5(run_dir, row, monolith_path)
            sources.append((path, kind))
        except FileNotFoundError:
            sources.append(("", "missing"))
            missing += 1

    n_per = sum(1 for _, k in sources if k == "per_step")
    n_mono = sum(1 for _, k in sources if k == "monolith")
    _log("=== string analyze ===")
    _log(f"  run_dir: {run_dir}")
    _log(
        f"  snapshots: {len(rows)}  "
        f"(per-step={n_per}, monolith={n_mono}, missing={missing})"
    )
    _log(f"  plots: {'on' if do_plots else 'off (metrics only)'}  workers={workers}")

    N_hint = None
    for path, kind in sources:
        if not path:
            continue
        try:
            import h5py
            with h5py.File(path, "r") as f:
                if "phi_0" not in f:
                    continue
                obj = f["phi_0"]
                if hasattr(obj, "shape") and not hasattr(obj, "keys"):
                    shp = obj.shape
                else:
                    k0 = next(iter(obj.keys()))
                    shp = obj[k0].shape
                if len(shp) == 3:
                    N_hint = int(shp[0])
                elif len(shp) == 1:
                    N_hint = int(round(shp[0] ** (1.0 / 3.0)))
            if N_hint:
                break
        except Exception:
            continue

    if N_hint:
        ram = estimate_ram_gb(N_hint, n_scalars=2)
        dense_cut, max3d_cut = _plot_thresholds(N_hint)
        _log(f"  Grid N≈{N_hint}  est. RAM/snapshot ≈ {ram:.1f} GB")
        if do_plots:
            _log(
                f"  Plot cuts: dense(light 2D) if voxels>{dense_cut:,}; "
                f"3D if voxels≤{max3d_cut:,}"
            )
        if ram > 12 and workers > 1:
            _log(
                f"  WARNING: {workers} workers × {ram:.1f} GB may OOM; "
                f"consider --workers 1"
            )

    time_index: Optional[Dict[float, str]] = None
    if n_mono > 0 and monolith_path and workers <= 1:
        _log(f"  Building monolith time index: {monolith_path}")
        time_index = build_h5_time_index(monolith_path, "phi_0")
        _log(f"  time keys: {len(time_index)}")

    n_total = len(rows)
    tasks = []
    for i, (row, (path, kind)) in enumerate(zip(rows, sources), start=1):
        if kind == "missing":
            _log(f"  [{i}/{n_total}] step {int(float(row['step']))}: MISSING HDF5 — skip")
            continue
        ti = time_index if (kind == "monolith" and workers <= 1) else None
        tasks.append(
            (path, row, run_dir, do_plots, metadata, ti, kind, i, n_total)
        )

    t_all = time.time()
    if workers <= 1:
        for task in tasks:
            step, status, metrics = _analyze_one_h5(task)
            if metrics:
                metrics_rows.append(metrics)
            gc.collect()
    else:
        tasks_par = [
            (path, row, run_dir, do_plots, metadata, None, kind, i, n_total)
            for (path, row, run_dir, do_plots, metadata, _ti, kind, i, n_total) in tasks
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_analyze_one_h5, t): t for t in tasks_par}
            for fut in as_completed(futures):
                step, status, metrics = fut.result()
                if metrics:
                    metrics_rows.append(metrics)

    metrics_rows.sort(key=lambda r: r["step"])
    fieldnames = [
        "step", "time", "temperature", "n_loops",
        "n_string_voxels", "string_voxel_fraction",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    _log(
        f"Wrote {summary_path} ({len(metrics_rows)} rows)  "
        f"wall {_fmt_s(time.time() - t_all)}"
    )
    return len(metrics_rows)


def main():
    ap = argparse.ArgumentParser(
        description="CosmoLattice HDF5 → string analysis / per-step split",
    )
    ap.add_argument(
        "command",
        choices=["export", "analyze", "split", "all"],
        help="analyze=strings; split=per-step h5; export=NPZ; all=export+analyze",
    )
    ap.add_argument("run_dir", help="CosmoLattice output directory")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel snapshot workers (default 1 for 1024³)")
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Re-write even if output file exists")
    ap.add_argument("--skip-plots", action="store_true",
                    help="Skip 2D/3D string PNGs")
    ap.add_argument("--metrics-only", action="store_true",
                    help="CSV only, no PNGs (still reads HDF5)")
    ap.add_argument("--from-npz", action="store_true",
                    help="Analyze existing NPZ (skip HDF5 read)")
    args = ap.parse_args()

    skip_existing = not args.no_skip_existing
    run_dir = os.path.abspath(args.run_dir)

    if args.command == "split":
        split_hdf5(
            run_dir,
            workers=args.workers,
            step_min=args.step_min,
            step_max=args.step_max,
            skip_existing=skip_existing,
        )
        return

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
