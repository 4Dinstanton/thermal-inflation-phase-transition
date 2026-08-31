#!/usr/bin/env python3
"""CosmoLattice HDF5 field snapshots → cosmic-string analysis.

Primary path: read ``field_snapshot.h5`` + ``manifest.csv`` directly (no NPZ).
Optional ``export`` command still writes NPZ for revisualize_snapshots.py.
Optional ``split`` writes per-step HDF5 under ``field_states/`` for easier transfer.

  **analyze**      — winding + network metrics CSV + optional PNGs + timeseries plot
  **plot-network** — re-plot ``strings/string_network_timeseries.png`` from CSV
  **split**        — monolith ``field_snapshot.h5`` → ``field_states/snapshot_step_*.h5``
  **export**       — HDF5 → ``state_step_*.npz`` (optional)
  **all**          — export then analyze

Network metrics (length, ξ, core E, v²) are computed from φ (+ π) for each
slice — CosmoLattice energy dumps (often O(100 GB)) are never opened.

Examples
--------
  # Split large monolith for transfer (on KISTI); add --with-pi for E_kin/v² off-site
  python tools/cl_hdf5_string_pipeline.py split <run_dir> [--with-pi]

  # Strings + network observables from HDF5 (default: step >= 3500, post-percolation)
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --workers 1

  # All manifest snapshots (override default step floor)
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --from-first

  # CSV + network plot only (no 2D/3D PNGs)
  python tools/cl_hdf5_string_pipeline.py analyze <run_dir> --metrics-only
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.cl_field_snapshot_io import (
    build_h5_time_index,
    estimate_ram_gb,
    h5_has_group,
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
from tools.string_network_metrics import (
    NETWORK_CSV_FIELDS,
    compute_network_metrics,
    load_transition_markers,
    plot_network_timeseries,
)

KST = ZoneInfo("Asia/Seoul")
LOG = logging.getLogger("cl_hdf5_string")


class _KSTFormatter(logging.Formatter):
    """Timestamps in Korea Standard Time (Asia/Seoul)."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        dt = datetime.fromtimestamp(record.created, tz=KST)
        if datefmt:
            return dt.strftime(datefmt)
        # Explicit KST label (Asia/Seoul %Z is often "KST")
        return dt.strftime("%Y-%m-%d %H:%M:%S") + " KST"


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure stdout (+ optional file) logging with KST timestamps."""
    LOG.handlers.clear()
    LOG.setLevel(level)
    LOG.propagate = False
    fmt = _KSTFormatter("%(asctime)s | %(levelname)-7s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.flush = sys.stdout.flush  # type: ignore[method-assign]
    LOG.addHandler(sh)
    if log_file:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
        LOG.info("log file: %s", os.path.abspath(log_file))


def _pool_logging_init() -> None:
    """Stdout KST logging in ProcessPool workers (no shared file handle)."""
    setup_logging(log_file=None, level=logging.INFO)


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


# First step for **analyze** / **all** when --step-min is omitted.
# Skips pre-percolation / pre-langoff snapshots on set8-style 1024³ runs.
# Override: --step-min N, --from-first, or analyze_step_min in cl_run_params.json.
ANALYZE_STEP_MIN_DEFAULT = 3500


def resolve_analyze_step_min(
    run_dir: str,
    step_min: Optional[int],
    *,
    from_first: bool = False,
) -> Optional[int]:
    """Step floor for analyze/all. Split/export leave step_min unchanged (no default)."""
    if from_first:
        return None
    if step_min is not None:
        return int(step_min)
    params = load_run_params(run_dir) or {}
    if params.get("analyze_step_min") is not None:
        return int(params["analyze_step_min"])
    return ANALYZE_STEP_MIN_DEFAULT


class _IncrementalSummaryCsv:
    """Append one metrics row per snapshot (flush/fsync) for crash-safe progress."""

    def __init__(
        self,
        path: str,
        fieldnames: List[str],
        *,
        resume: bool = False,
    ) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self._completed: set[int] = set()
        if resume and os.path.isfile(path):
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for row in reader:
                        try:
                            self._completed.add(int(float(row["step"])))
                        except (KeyError, TypeError, ValueError):
                            continue
            self._fp = open(path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._fp, fieldnames=self.fieldnames, extrasaction="ignore"
            )
            LOG.info(
                "CSV resume: %s (%d rows on disk)", path, len(self._completed)
            )
        else:
            self._fp = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._fp, fieldnames=self.fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()
            self._fp.flush()
            LOG.info("CSV opened (incremental): %s", path)

    def has_step(self, step: int) -> bool:
        return int(step) in self._completed

    def append(self, metrics: Dict[str, Any]) -> None:
        self._writer.writerow(metrics)
        self._fp.flush()
        os.fsync(self._fp.fileno())
        step = int(metrics["step"])
        self._completed.add(step)
        LOG.info("  CSV saved step %s  (%d rows total)", step, len(self._completed))

    @property
    def n_rows(self) -> int:
        return len(self._completed)

    def close(self) -> None:
        if self._fp and not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "_IncrementalSummaryCsv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _npz_path(state_dir: str, step: int) -> str:
    return os.path.join(state_dir, f"state_step_{step:010d}.npz")


def _save_npz_from_snap(
    snap: Dict[str, Any], out_path: str, hubble: bool = True
) -> None:
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


def _metrics_from_snap(
    snap: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    *,
    do_loops: bool = True,
    max_label_voxels: int = 32_000_000,
) -> Dict[str, Any]:
    """Winding + network observables (length, ξ, core E, v², loops)."""
    return compute_network_metrics(
        snap,
        params,
        do_loops=do_loops,
        max_label_voxels=max_label_voxels,
    )


def _run_params_for_metrics(
    run_dir: str, metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge cl_run_params.json with simulation_metadata for lattice scales."""
    params = dict(load_run_params(run_dir) or {})
    if metadata:
        for k in ("mphi", "mu", "lam", "dx_phys", "omegaStar", "fStar", "gamma", "Nx"):
            if k in metadata and k not in params:
                try:
                    params[k] = float(np.asarray(metadata[k]).reshape(-1)[0])
                except Exception:
                    params[k] = metadata[k]
    return params


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
DENSE_STRING_FRAC = 500_000 / (256**3)  # ~0.0298
MAX_3D_STRING_FRAC = 80_000 / (256**3)  # ~0.00477
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
        bool,
        bool,
        Dict[str, Any],
        bool,
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
        load_pi,
        do_network_loops,
        run_params,
        paper_plots,
    ) = args
    step = int(float(row["step"]))
    tag = f"[{idx}/{n_total}] step {step}"
    t0 = time.time()
    if not LOG.handlers:
        _pool_logging_init()
    try:
        kind = source_kind or "monolith"
        want_pi = bool(load_pi)
        if want_pi and not (
            h5_has_group(h5_path, "pi_0") and h5_has_group(h5_path, "pi_1")
        ):
            LOG.info(
                f"  {tag}: no pi_* in HDF5 — core E_kin / v² skipped "
                "(use monolith field_snapshot.h5, or re-split with --with-pi)"
            )
            want_pi = False
        LOG.info(
            f"  {tag}: start ({kind})  file={os.path.basename(h5_path)}"
            f"  load_pi={want_pi}"
        )

        t_load = time.time()
        if time_index is None and kind == "monolith":
            time_index = build_h5_time_index(h5_path, "phi_0")
        snap = read_h5_snapshot(
            h5_path,
            row,
            time_index=time_index if kind == "monolith" else None,
            load_pi=want_pi,
        )
        LOG.info(
            f"  {tag}: loaded N={snap.get('N')}  "
            f"T={snap['temperature']:.1f}  ({_fmt_s(time.time() - t_load)})"
        )

        t_met = time.time()
        # Free theta early if we only need metrics (plots need it)
        metrics = _metrics_from_snap(
            snap,
            run_params,
            do_loops=do_network_loops,
            max_label_voxels=_plot_thresholds(int(snap.get("N", 256)))[0],
        )
        n_string_vox = int(metrics["n_string_voxels"])
        LOG.info(
            f"  {tag}: network  voxels={n_string_vox:,}  "
            f"L={metrics['L_comoving']:.3e}  "
            f"xi={metrics['xi_comoving']:.3e}  "
            f"mu_eff={metrics['mu_eff']}  "
            f"v2={metrics['v2_mean']}  "
            f"({_fmt_s(time.time() - t_met)})"
        )

        if do_plots or paper_plots:
            from postprocess.revisualize_snapshots import (
                _plot_strings_2d_dense,
                _plot_strings_3d_dense,
                plot_strings_2d,
                plot_strings_2d_paper,
                plot_strings_3d,
            )

            meta = dict(metadata) if metadata is not None else {}
            if run_params:
                for k in ("analyze_step_min", "percolation_step", "mphi", "lam", "gamma"):
                    if k in run_params and k not in meta:
                        meta[k] = run_params[k]
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

            if do_plots:
                t_plot = time.time()
                if n_string_vox > dense_cut:
                    LOG.info(
                        f"  {tag}: dense winding (>{dense_cut:,}) → "
                        f"light 2D + dense 3D (no loop IDs)"
                    )
                    _plot_strings_2d_dense(state, meta, png2d, n_string_vox)
                    if metrics.get("n_loops", "") == "" or metrics.get("n_loops", -1) < 0:
                        metrics["n_loops"] = -1
                    LOG.info(f"  {tag}: 2D PNG done  ({_fmt_s(time.time() - t_plot)})")
                    t3 = time.time()
                    LOG.info(f"  {tag}: dense 3D PNG (subsampled, no labeling) …")
                    _plot_strings_3d_dense(state, meta, png3d, n_string_vox)
                    LOG.info(f"  {tag}: 3D PNG done  ({_fmt_s(time.time() - t3)})")
                else:
                    LOG.info(f"  {tag}: full 2D labeling + plot …")
                    strings = plot_strings_2d(state, meta, png2d)
                    if not do_network_loops:
                        metrics["n_loops"] = len(strings) if strings else 0
                    LOG.info(
                        f"  {tag}: 2D PNG done  loops={metrics.get('n_loops')}  "
                        f"({_fmt_s(time.time() - t_plot)})"
                    )
                    if strings and n_string_vox <= max3d_cut:
                        t3 = time.time()
                        LOG.info(f"  {tag}: 3D PNG (by loop ID) …")
                        plot_strings_3d(state, meta, png3d, labelled=None, strings=strings)
                        LOG.info(f"  {tag}: 3D PNG done  ({_fmt_s(time.time() - t3)})")
                    elif strings:
                        t3 = time.time()
                        LOG.info(
                            f"  {tag}: voxels={n_string_vox:,} > 3D-label cut "
                            f"{max3d_cut:,} → dense 3D (no loop IDs)"
                        )
                        _plot_strings_3d_dense(state, meta, png3d, n_string_vox)
                        LOG.info(f"  {tag}: 3D PNG done  ({_fmt_s(time.time() - t3)})")
            if paper_plots:
                pub_png = os.path.join(
                    strings_dir, f"strings_pub_step_{step:010d}.png"
                )
                plot_strings_2d_paper(state, meta, pub_png)
                LOG.info(f"  {tag}: paper PNG -> {os.path.basename(pub_png)}")
            del state, snap
        else:
            if metrics.get("n_loops", "") == "":
                metrics["n_loops"] = ""
            del snap

        gc.collect()
        LOG.info(f"  {tag}: OK total {_fmt_s(time.time() - t0)}")
        return step, "ok", metrics
    except Exception as exc:
        gc.collect()
        LOG.error(f"  {tag}: FAIL after {_fmt_s(time.time() - t0)}: {exc}")
        return step, f"fail:{exc}", None


def _split_one(
    args: Tuple[
        str, Dict[str, Any], str, Optional[Dict[float, str]], bool, int, int, bool
    ],
) -> Tuple[int, str, str]:
    h5_path, row, out_path, time_index, skip_existing, idx, n_total, with_pi = args
    step = int(float(row["step"]))
    tag = f"[{idx}/{n_total}] step {step}"
    if not LOG.handlers:
        _pool_logging_init()
    if skip_existing and os.path.isfile(out_path):
        size_gb = os.path.getsize(out_path) / (1024**3)
        LOG.info(f"  {tag}: skip (exists, {size_gb:.2f} GiB)")
        return step, "skip", out_path
    t0 = time.time()
    try:
        t = float(row["t"])
        n_scalars = int(float(row["n_scalars"]))
        if time_index is None:
            time_index = build_h5_time_index(h5_path, "phi_0")
        time_key = lookup_h5_key(t, time_index)
        LOG.info(f"  {tag}: reading phi_0 (t={time_key}) …")
        phi0 = read_h5_field(h5_path, "phi_0", time_key)
        phi1 = None
        if n_scalars >= 2:
            LOG.info(f"  {tag}: reading phi_1 …")
            phi1 = read_h5_field(h5_path, "phi_1", time_key, N=phi0.shape[0])
        pi0 = pi1 = None
        if with_pi:
            LOG.info(f"  {tag}: reading pi_0 …")
            pi0 = read_h5_field(h5_path, "pi_0", time_key, N=phi0.shape[0])
            if n_scalars >= 2:
                LOG.info(f"  {tag}: reading pi_1 …")
                pi1 = read_h5_field(h5_path, "pi_1", time_key, N=phi0.shape[0])
        LOG.info(f"  {tag}: writing {os.path.basename(out_path)} …")
        write_per_step_h5(out_path, row, phi0, phi1, pi0=pi0, pi1=pi1)
        del phi0, phi1, pi0, pi1
        gc.collect()
        size_gb = os.path.getsize(out_path) / (1024**3)
        LOG.info(f"  {tag}: OK  {size_gb:.2f} GiB  ({_fmt_s(time.time() - t0)})")
        return step, "ok", out_path
    except Exception as exc:
        gc.collect()
        LOG.error(f"  {tag}: FAIL: {exc}")
        return step, f"fail:{exc}", out_path


def split_hdf5(
    run_dir: str,
    workers: int = 1,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    skip_existing: bool = True,
    with_pi: bool = False,
) -> int:
    """Split monolith field_snapshot.h5 into field_states/snapshot_step_*.h5."""
    run_dir = os.path.abspath(run_dir)
    state_dir = os.path.join(run_dir, "field_states")
    os.makedirs(state_dir, exist_ok=True)

    rows = _filter_rows(load_manifest_rows(run_dir), step_min, step_max)
    if not rows:
        LOG.info("No manifest rows to split.")
        return 0

    h5_path = resolve_h5_path(run_dir, rows)
    size_gb = os.path.getsize(h5_path) / (1024**3) if os.path.isfile(h5_path) else 0.0
    LOG.info(f"=== split monolith → per-step HDF5 ===")
    LOG.info(f"  source: {h5_path}  ({size_gb:.1f} GiB)")
    LOG.info(f"  dest:   {state_dir}/snapshot_step_XXXXXXXXXX.h5")
    LOG.info(f"  snaps:  {len(rows)}  workers={workers}  with_pi={with_pi}")

    time_index = build_h5_time_index(h5_path, "phi_0")
    n_total = len(rows)
    tasks = []
    for i, row in enumerate(rows, start=1):
        step = int(float(row["step"]))
        out_path = per_step_h5_path(run_dir, step)
        ti = time_index if workers <= 1 else None
        tasks.append((h5_path, row, out_path, ti, skip_existing, i, n_total, with_pi))

    n_ok = 0
    t_all = time.time()
    if workers <= 1:
        for task in tasks:
            step, status, path = _split_one(task)
            if status in ("ok", "skip"):
                n_ok += 1
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_pool_logging_init
        ) as pool:
            futures = {pool.submit(_split_one, t): t for t in tasks}
            for fut in as_completed(futures):
                step, status, path = fut.result()
                if status in ("ok", "skip"):
                    n_ok += 1

    LOG.info(f"Split done: {n_ok}/{n_total}  total wall {_fmt_s(time.time() - t_all)}")
    LOG.info(
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
        LOG.info("No manifest rows to export.")
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
        LOG.info(f"HDF5: {h5_path}")
        LOG.info(f"Grid N≈{N_hint}  est. RAM per worker ≈ {ram:.1f} GB")
        if ram > 12 and workers > 1:
            LOG.warning(
                f"WARNING: {workers} workers × {ram:.1f} GB may OOM; "
                f"consider --workers 1"
            )

    LOG.info(f"Exporting {len(rows)} snapshots -> {state_dir}")
    tasks = [(h5_path, row, state_dir, "", skip_existing, hubble) for row in rows]

    n_ok = 0
    if workers <= 1:
        for task in tasks:
            step, status, path = _export_one(task)
            LOG.info(f"  step {step}: {status} {path}")
            if status == "ok" or status == "skip":
                n_ok += 1
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_pool_logging_init
        ) as pool:
            futures = {pool.submit(_export_one, t): t for t in tasks}
            for fut in as_completed(futures):
                step, status, path = fut.result()
                LOG.info(f"  step {step}: {status}")
                if status == "ok" or status == "skip":
                    n_ok += 1

    params = load_run_params(run_dir)
    meta_path = write_metadata(run_dir, params, n_ok)
    LOG.info(f"Metadata: {meta_path}")
    LOG.info(f"Exported/skipped {n_ok}/{len(rows)} snapshots")
    return n_ok


def analyze_hdf5(
    run_dir: str,
    workers: int = 1,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    skip_plots: bool = False,
    metrics_only: bool = False,
    from_npz: bool = False,
    load_pi: bool = True,
    do_network_loops: bool = True,
    skip_network_plot: bool = False,
    resume_csv: bool = False,
    paper_plots: bool = False,
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
        LOG.info("No manifest rows to analyze.")
        return 0

    metadata = _load_or_build_metadata(run_dir, len(rows))
    run_params = _run_params_for_metrics(run_dir, metadata)
    n_scalars = int(metadata.get("n_scalars", int(float(rows[0]["n_scalars"]))))
    if n_scalars < 2:
        LOG.info("Run has n_scalars=1; string analysis needs complex (phi1+phi2).")
        return 0

    do_plots = not skip_plots and not metrics_only
    fieldnames = list(NETWORK_CSV_FIELDS)

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
    LOG.info("=== string analyze ===")
    LOG.info(f"  run_dir: {run_dir}")
    LOG.info(
        f"  snapshots: {len(rows)}  "
        f"(per-step={n_per}, monolith={n_mono}, missing={missing})"
    )
    LOG.info(
        f"  plots: {'on' if do_plots else 'off (metrics only)'}  "
        f"paper_plots={paper_plots}  "
        f"workers={workers}  load_pi={load_pi}  "
        f"network_loops={do_network_loops}"
    )
    LOG.info(
        "  NOTE: core energies from phi(+pi) only — "
        "does NOT read CosmoLattice E_S_* / potential dumps"
    )

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
        ram = estimate_ram_gb(N_hint, n_scalars=2, with_pi=load_pi)
        dense_cut, max3d_cut = _plot_thresholds(N_hint)
        LOG.info(f"  Grid N≈{N_hint}  est. RAM/snapshot ≈ {ram:.1f} GB")
        if do_plots:
            LOG.info(
                f"  Plot cuts: voxels>{dense_cut:,} → light 2D + dense 3D "
                f"(no loop IDs); ≤{max3d_cut:,} → labeled 2D+3D; "
                f"in between → labeled 2D + dense 3D (no loop IDs)"
            )
        if ram > 12 and workers > 1:
            LOG.warning(
                f"  WARNING: {workers} workers × {ram:.1f} GB may OOM; "
                f"consider --workers 1"
            )

    time_index: Optional[Dict[float, str]] = None
    if n_mono > 0 and monolith_path and workers <= 1:
        LOG.info(f"  Building monolith time index: {monolith_path}")
        time_index = build_h5_time_index(monolith_path, "phi_0")
        LOG.info(f"  time keys: {len(time_index)}")

    n_total = len(rows)
    tasks = []
    for i, (row, (path, kind)) in enumerate(zip(rows, sources), start=1):
        if kind == "missing":
            LOG.info(
                f"  [{i}/{n_total}] step {int(float(row['step']))}: MISSING HDF5 — skip"
            )
            continue
        ti = time_index if (kind == "monolith" and workers <= 1) else None
        step_key = int(float(row["step"]))
        tasks.append(
            (
                path,
                row,
                run_dir,
                do_plots,
                metadata,
                ti,
                kind,
                i,
                n_total,
                load_pi,
                do_network_loops,
                run_params,
                paper_plots,
            )
        )

    t_all = time.time()
    n_written = 0
    with _IncrementalSummaryCsv(
        summary_path, fieldnames, resume=resume_csv
    ) as csvw:
        if resume_csv:
            tasks = [
                t for t in tasks
                if not csvw.has_step(int(float(t[1]["step"])))
            ]
            if not tasks:
                LOG.info("All steps already in CSV — nothing to do.")
                n_written = csvw.n_rows
            else:
                LOG.info("Resume: %d snapshots remaining", len(tasks))

        if tasks:
            if workers <= 1:
                for task in tasks:
                    step, status, metrics = _analyze_one_h5(task)
                    if metrics:
                        csvw.append(metrics)
                        n_written = csvw.n_rows
                    gc.collect()
            else:
                tasks_par = [
                    (
                        path,
                        row,
                        run_dir,
                        do_plots,
                        metadata,
                        None,
                        kind,
                        i,
                        n_total,
                        load_pi,
                        do_network_loops,
                        run_params,
                        paper_plots,
                    )
                    for (
                        path,
                        row,
                        run_dir,
                        do_plots,
                        metadata,
                        _ti,
                        kind,
                        i,
                        n_total,
                        load_pi,
                        do_network_loops,
                        run_params,
                        _pp,
                    ) in tasks
                ]
                with ProcessPoolExecutor(
                    max_workers=workers, initializer=_pool_logging_init
                ) as pool:
                    futures = {
                        pool.submit(_analyze_one_h5, t): t for t in tasks_par
                    }
                    for fut in as_completed(futures):
                        step, status, metrics = fut.result()
                        if metrics:
                            csvw.append(metrics)
                            n_written = csvw.n_rows
        else:
            n_written = csvw.n_rows

    LOG.info(
        f"CSV {summary_path} ({n_written} rows)  "
        f"wall {_fmt_s(time.time() - t_all)}"
    )

    if not skip_network_plot and n_written > 0:
        fig_path = os.path.join(strings_dir, "string_network_timeseries.png")
        try:
            plot_network_timeseries(
                summary_path,
                fig_path,
                title=os.path.basename(run_dir),
                run_dir=run_dir,
            )
            LOG.info(f"Network plot: {fig_path}")
        except Exception as exc:
            LOG.warning(f"Network plot failed: {exc}")

    return n_written


def plot_network(run_dir: str, csv_name: str = "string_summary.csv", *, recompute_markers: bool = False) -> str:
    """Re-plot network timeseries from an existing CSV."""
    run_dir = os.path.abspath(run_dir)
    csv_path = os.path.join(run_dir, "strings", csv_name)
    fig_path = os.path.join(run_dir, "strings", "string_network_timeseries.png")
    markers = load_transition_markers(run_dir, force=recompute_markers)
    plot_network_timeseries(
        csv_path, fig_path, title=os.path.basename(run_dir), run_dir=run_dir, markers=markers
    )
    LOG.info(f"Network plot: {fig_path}")
    return fig_path


def main():
    ap = argparse.ArgumentParser(
        description="CosmoLattice HDF5 → string analysis / per-step split",
    )
    ap.add_argument(
        "command",
        choices=["export", "analyze", "split", "all", "plot-network"],
        help="analyze=strings+network; split=per-step h5; "
        "plot-network=replot CSV; export=NPZ; all=export+analyze",
    )
    ap.add_argument("run_dir", help="CosmoLattice output directory")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel snapshot workers (default 1 for 1024³)",
    )
    ap.add_argument(
        "--step-min",
        type=int,
        default=None,
        help=f"Only snapshots with step >= N. "
        f"Default for analyze/all: {ANALYZE_STEP_MIN_DEFAULT} "
        f"(post-percolation); use --from-first for step 0.",
    )
    ap.add_argument(
        "--from-first",
        action="store_true",
        help="Analyze/export from the first manifest snapshot (step_min=0)",
    )
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-write even if output file exists",
    )
    ap.add_argument("--skip-plots", action="store_true", help="Skip 2D/3D string PNGs")
    ap.add_argument(
        "--metrics-only",
        action="store_true",
        help="CSV only, no PNGs (still reads HDF5)",
    )
    ap.add_argument(
        "--from-npz", action="store_true", help="Analyze existing NPZ (skip HDF5 read)"
    )
    ap.add_argument(
        "--no-pi",
        action="store_true",
        help="Do not load pi_* (saves RAM; skips E_kin and v²)",
    )
    ap.add_argument(
        "--no-network-loops",
        action="store_true",
        help="Skip connected-component loop census (faster on dense windings)",
    )
    ap.add_argument(
        "--skip-network-plot",
        action="store_true",
        help="Do not write strings/string_network_timeseries.png",
    )
    ap.add_argument(
        "--with-pi",
        action="store_true",
        help="When splitting, also write pi_0/pi_1 into per-step HDF5 "
        "(needed for E_kin/v² without the monolith)",
    )
    ap.add_argument(
        "--resume-csv",
        action="store_true",
        help="Append to existing string_summary.csv; skip steps already saved",
    )
    ap.add_argument(
        "--paper-plots",
        action="store_true",
        help="Also write strings_pub_step_*.png (crisp 2×2 PT + string figure)",
    )
    ap.add_argument(
        "--recompute-markers",
        action="store_true",
        help="Re-scan manifest/HDF5 for TIPT/langoff/T_c1 vertical lines on network plot",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="Optional log path (KST timestamps). "
        "Default: <run_dir>/strings/pipeline_<command>.log",
    )
    ap.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not write a log file (stdout only)",
    )
    args = ap.parse_args()

    skip_existing = not args.no_skip_existing
    run_dir = os.path.abspath(args.run_dir)

    # Default step floor applies to analyze/all only (not split / plot-network).
    step_min = args.step_min
    step_max = args.step_max
    if args.command in ("analyze", "all"):
        step_min = resolve_analyze_step_min(
            run_dir, args.step_min, from_first=args.from_first
        )

    log_file = args.log_file
    if log_file is None and not args.no_log_file:
        log_dir = os.path.join(run_dir, "strings")
        log_file = os.path.join(log_dir, f"pipeline_{args.command}.log")
    setup_logging(log_file=None if args.no_log_file else log_file)
    LOG.info(
        "command=%s  run_dir=%s  workers=%s  timezone=KST (Asia/Seoul)",
        args.command,
        run_dir,
        args.workers,
    )
    if args.command in ("analyze", "all"):
        if step_min is not None:
            src = (
                "cli"
                if args.step_min is not None
                else (
                    "cl_run_params"
                    if (load_run_params(run_dir) or {}).get("analyze_step_min")
                    is not None
                    else "default"
                )
            )
            LOG.info("step_min=%s (%s)  step_max=%s", step_min, src, step_max)
        else:
            LOG.info("step_min=none (--from-first)  step_max=%s", step_max)

    if args.command == "plot-network":
        plot_network(run_dir, recompute_markers=args.recompute_markers)
        return

    if args.command == "split":
        split_hdf5(
            run_dir,
            workers=args.workers,
            step_min=step_min,
            step_max=step_max,
            skip_existing=skip_existing,
            with_pi=args.with_pi,
        )
        return

    if args.command in ("export", "all"):
        export_hdf5(
            run_dir,
            workers=args.workers,
            step_min=args.step_min,
            step_max=step_max,
            skip_existing=skip_existing,
        )

    if args.command in ("analyze", "all"):
        analyze_hdf5(
            run_dir,
            workers=args.workers,
            step_min=step_min,
            step_max=step_max,
            skip_plots=args.skip_plots,
            metrics_only=args.metrics_only,
            from_npz=args.from_npz and args.command == "analyze",
            load_pi=not args.no_pi,
            do_network_loops=not args.no_network_loops,
            skip_network_plot=args.skip_network_plot,
            resume_csv=args.resume_csv,
            paper_plots=args.paper_plots,
        )


if __name__ == "__main__":
    main()
