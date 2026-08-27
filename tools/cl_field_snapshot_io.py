#!/usr/bin/env python3
"""Shared I/O for CosmoLattice field snapshots (raw + HDF5).

HDF5 layout (field_snapshot.hpp, snapshot_format=hdf5):
  <run_dir>/field_snapshot.h5
    phi_0/<t>/   dataset shape (Nx, Ny, Nz) program units
    pi_0/<t>/
    phi_1/<t>/   (complex runs)
    pi_1/<t>/

Manifest: field_states/manifest.csv — one row per snapshot; filename column
points to snapshot_*.raw or field_snapshot.h5.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MANIFEST_MIN_FIELDS = 8
PHI_GROUPS = ("phi_0", "phi_1", "pi_0", "pi_1")


def _require_h5py():
    try:
        import h5py  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 field snapshots. "
            "On KISTI: pip install h5py  (or use the module's Python with HDF5)."
        ) from exc
    return h5py


def pretty_time_key(t: float, prec: int = 10) -> str:
    """Match CosmoLattice PrettyToString::get(t, prec) trailing-zero strip."""
    s = f"{t:.{prec}f}"
    while "." in s and s.endswith("0"):
        s = s[:-1]
    if s.endswith("."):
        s = s[:-1]
    return s


def parse_manifest_row(line: str) -> Optional[Dict[str, Any]]:
    """Parse one manifest.csv line (raw or HDF5)."""
    line = line.strip()
    if not line or line.startswith("step,"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < MANIFEST_MIN_FIELDS:
        return None
    filename = parts[-1]
    if not (filename.endswith(".raw") or filename.endswith(".h5")):
        return None
    row: Dict[str, Any] = {
        "step": parts[0],
        "t": parts[1],
        "T": parts[2],
        "a": parts[3],
        "H": parts[4],
        "fStar": parts[5],
        "n_scalars": parts[6],
        "filename": filename,
    }
    if len(parts) >= 10:
        row["expansion_stage"] = parts[7]
        row["rho_m"] = parts[8]
    return row


def load_manifest_rows(run_dir: str) -> List[Dict[str, Any]]:
    manifest_path = os.path.join(run_dir, "field_states", "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"no manifest.csv in {os.path.join(run_dir, 'field_states')}")
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    with open(manifest_path, newline="") as f:
        for line in f:
            row = parse_manifest_row(line)
            if row is None:
                continue
            try:
                step_key = int(float(row["step"]))
            except ValueError:
                continue
            if step_key in seen:
                continue
            seen.add(step_key)
            rows.append(row)
    return rows


def resolve_h5_path(run_dir: str, rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """Locate field_snapshot.h5 for a CosmoLattice run."""
    run_dir = os.path.abspath(run_dir)
    candidates = [
        os.path.join(run_dir, "field_snapshot.h5"),
        os.path.join(run_dir, "field_states", "field_snapshot.h5"),
    ]
    if rows:
        for row in rows:
            fn = row.get("filename", "")
            if fn.endswith(".h5"):
                candidates.insert(0, os.path.join(run_dir, "field_states", fn))
                candidates.insert(0, os.path.join(run_dir, fn))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"no field_snapshot.h5 found under {run_dir} "
        "(expected at run root for snapshot_format=hdf5)"
    )


def _as_3d(arr: np.ndarray, N: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 1:
        n = int(round(arr.size ** (1.0 / 3.0)))
        if n ** 3 != arr.size:
            if N is not None and N ** 3 == arr.size:
                n = N
            else:
                raise ValueError(f"cannot reshape length {arr.size} to cube")
        return arr.reshape((n, n, n), order="C")
    raise ValueError(f"expected 1D or 3D field array, got shape {arr.shape}")


def build_h5_time_index(h5_path: str, group: str = "phi_0") -> Dict[float, str]:
    """Map program time t -> HDF5 dataset name inside group."""
    h5py = _require_h5py()
    index: Dict[float, str] = {}
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            raise KeyError(f"group '{group}' missing in {h5_path}")
        for key in f[group].keys():
            try:
                index[float(key)] = key
            except ValueError:
                continue
    return index


def lookup_h5_key(t: float, time_index: Dict[float, str], tol: float = 1e-6) -> str:
    """Find HDF5 dataset key for manifest time t."""
    key = pretty_time_key(t)
    if key in time_index.values():
        return key
    if t in time_index:
        return time_index[t]
    # nearest
    if not time_index:
        raise KeyError(f"no time datasets in HDF5 for t={t}")
    times = np.array(list(time_index.keys()))
    idx = int(np.argmin(np.abs(times - t)))
    if abs(times[idx] - t) > tol:
        raise KeyError(f"no HDF5 dataset matching t={t} (nearest {times[idx]})")
    return time_index[float(times[idx])]


def read_h5_field(
    h5_path: str,
    group: str,
    time_key: str,
    N: Optional[int] = None,
) -> np.ndarray:
    h5py = _require_h5py()
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            raise KeyError(f"group '{group}' missing in {h5_path}")
        if time_key not in f[group]:
            raise KeyError(f"{group}/{time_key} missing in {h5_path}")
        return _as_3d(f[group][time_key][()], N=N)


def read_h5_snapshot(
    h5_path: str,
    row: Dict[str, Any],
    time_index: Optional[Dict[float, str]] = None,
    *,
    field_dtype=np.float32,
) -> Dict[str, Any]:
    """Read one complex-field snapshot from field_snapshot.h5."""
    from tools.winding import compute_winding_number

    t = float(row["t"])
    step = int(float(row["step"]))
    T = float(row["T"])
    a = float(row["a"])
    H = float(row["H"])
    f_star = float(row["fStar"])
    n_scalars = int(float(row["n_scalars"]))

    if time_index is None:
        time_index = build_h5_time_index(h5_path, "phi_0")
    time_key = lookup_h5_key(t, time_index)

    phi0_prog = read_h5_field(h5_path, "phi_0", time_key)
    N = phi0_prog.shape[0]

    if n_scalars >= 2:
        phi1_gev = np.asarray(phi0_prog, dtype=field_dtype) * np.float32(f_star)
        del phi0_prog
        phi2_prog = read_h5_field(h5_path, "phi_1", time_key, N=N)
        phi2_gev = np.asarray(phi2_prog, dtype=field_dtype) * np.float32(f_star)
        del phi2_prog
        rho = np.sqrt(phi1_gev * phi1_gev + phi2_gev * phi2_gev, dtype=field_dtype)
        winding = compute_winding_number(phi1_gev, phi2_gev, dtype=field_dtype)
        theta = np.arctan2(phi2_gev, phi1_gev, dtype=field_dtype)
        out = {
            "step": step,
            "time": t,
            "temperature": T,
            "a": a,
            "H": H,
            "fStar": f_star,
            "N": N,
            "n_scalars": 2,
            "phi1": phi1_gev,
            "phi2": phi2_gev,
            "rho": rho,
            "theta": theta,
            "winding": winding,
        }
    else:
        phi_gev = np.asarray(phi0_prog, dtype=field_dtype) * np.float32(f_star)
        del phi0_prog
        out = {
            "step": step,
            "time": t,
            "temperature": T,
            "a": a,
            "H": H,
            "fStar": f_star,
            "N": N,
            "n_scalars": 1,
            "phi": phi_gev,
        }
    return out


def estimate_ram_gb(N: int, n_scalars: int = 2, with_winding: bool = True) -> float:
    """Rough peak RAM per snapshot worker (float32 fields + optional PNG path)."""
    n3 = N ** 3
    # phi1, phi2, rho, theta (float32) + winding + HDF5 decode buffer
    bytes_per = 4 * (n_scalars + 2) * n3 + 4 * n3
    if with_winding:
        bytes_per += 8 * n3  # labelled int64 during loop-ID PNGs (worst case)
    return bytes_per / (1024 ** 3)


def list_h5_snapshot_times(h5_path: str, group: str = "phi_0") -> List[float]:
    return sorted(build_h5_time_index(h5_path, group).keys())
