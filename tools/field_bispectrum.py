#!/usr/bin/env python3
"""Equilateral bispectrum + power spectrum from CosmoLattice HDF5 snapshots.

Post-processing only — no simulation re-run.

Algorithm (review)
------------------
**Power spectrum** (same FFT pass as bispectrum):

    δ̃(k) = FFT[δ(x)]     (numpy unnormalised)
    P_raw(k) = ⟨ |δ̃|² / N³ ⟩_{|k|∈shell}     (per FFT mode — IR shells can be O(N³))
    P(k)     = P_raw(k) · n_modes(k) / N³    (matches ``spectra_scalar_*.txt`` col 1)

With δ in **program units** (φ_prog, not φ_GeV).  ``P_raw`` is useful internally;
``P`` is what CosmoLattice writes and what we plot / cross-check.

**Equilateral bispectrum** — Scoccimarro / Jeong *shell-filter* estimator:

    δ_k(x) = IFFT[ δ̃(k) · 𝟙_{|k|∈shell} ]
    P_filt(k) = ⟨ δ_k(x)² ⟩          (= P above for a sharp-k shell)
    B_eq(k)   = ⟨ δ_k(x)³ ⟩            (filtered third moment)

Reduced equilateral bispectrum (Scoccimarro 2000; hierarchical / tree-level):

    Q(k1,k2,k3) = B / [P(k1)P(k2) + P(k1)P(k3) + P(k2)P(k3)]
    Q_eq(k)     = B_eq / (3 P_filt²)     # equilateral limit

Also report skewness of the filtered field:

    skew(k) = ⟨δ_k³⟩ / ⟨δ_k²⟩^{3/2} = B_eq / P_filt^{3/2}

This is a standard fast proxy for equilateral B(k,k,k), **not** the full
triangle sum Σ_{k₁+k₂+k₃=0} ⟨δ̃(k₁)δ̃(k₂)δ̃(k₃)⟩.

**ζ proxy** (default on; needs ``average_energies.txt``):

    A = 1 / [3(ρ+p)] ,   ρ+p = 2 E_K + (2/3) E_G
    P_ζ = A² P ,   B_ζ = A³ B_eq

For ``rho_norm`` this treats dimensionless |Φ| contrast as if it were δρ
in average_energies units (proxy). Reprocess existing CSVs with
``--apply-zeta-to-csvs <dir>``.

**Better field→ζ routes** (subhorizon-safe conceptually):

1. **Via \(t_c\)** (recommended): use |Φ| only to build conversion time —
   ``transition_time_correlators.py`` → \(P_\zeta=H^2 P_{\delta t}\).
2. **Effective clock**: \(\delta t_{\rm eff}=-\delta\mu/\dot\mu\) with
   \(\mu=\sigma(\rho_{\rm norm})\) or similar from the time series, then
   \(P_\zeta=(H/|\dot\mu|)^2 P_\delta\).  Use ``--apply-zeta-clock``.

**Squeezed proxy** (optional): ⟨ δ_soft(x)² · δ_hard(x) ⟩ vs k_hard — a cheap
wall-modulation diagnostic, not the full squeezed B(k_s, k_h, k_h).

Field choices (--field / --fields)
---------------------------------
    rho_norm   |Φ|_prog/φ₀_prog − ⟨|Φ|/φ₀⟩   **default** — PT contrast, O(1)
    phi0_prog  φ₀_prog − ⟨φ₀⟩                 cross-check vs spectra_scalar_0
    phi1_prog  φ₁_prog − ⟨φ₁⟩                 cross-check vs spectra_scalar_1
    theta_bulk θ − ⟨θ⟩_bulk on |Φ|/φ₀ > frac  Goldstone / wall ripples

Use ``--fields rho_norm theta_bulk`` to analyze both in one run (HDF5 loaded once
per snapshot; outputs under ``out_dir/<field>/``).

Usage
-----
    python tools/field_bispectrum.py <run_dir> --times 450 520 581
    python tools/field_bispectrum.py <run_dir> --fields rho_norm theta_bulk --times 450 520 581
    python tools/field_bispectrum.py <run_dir> --all-times --stride 10 --t-min 300
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FIELD_CHOICES = ("rho_norm", "phi0_prog", "phi1_prog", "theta_bulk")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.cl_field_snapshot_io import (  # noqa: E402
    build_h5_time_index,
    load_manifest_rows,
    parse_manifest_row,
    read_h5_field,
    resolve_h5_path,
    resolve_snapshot_h5,
)
from tools.export_cl_snapshots import load_run_params  # noqa: E402

LOG = logging.getLogger("field_bispectrum")

# CosmoLattice fundamental wavenumber: κ = 2π/(N·dx_prog), dx_prog = 1
K_FUND = 2.0 * math.pi


# ---------------------------------------------------------------------------
# manifest / CL spectra I/O
# ---------------------------------------------------------------------------
def _load_manifest_any(run_dir: str) -> List[Dict[str, Any]]:
    try:
        return load_manifest_rows(run_dir)
    except FileNotFoundError:
        pass
    path = os.path.join(run_dir, "manifest.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no manifest.csv under {run_dir}")
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    with open(path, newline="") as f:
        for line in f:
            row = parse_manifest_row(line)
            if row is None:
                continue
            try:
                step = int(float(row["step"]))
            except ValueError:
                continue
            if step in seen:
                continue
            seen.add(step)
            rows.append(row)
    return rows


def nearest_rows(rows: List[Dict[str, Any]], times: Sequence[float]) -> List[Dict[str, Any]]:
    ts = np.array([float(r["t"]) for r in rows], dtype=float)
    out: List[Dict[str, Any]] = []
    used: set[int] = set()
    for t in times:
        i = int(np.argmin(np.abs(ts - float(t))))
        if i in used:
            continue
        used.add(i)
        out.append(rows[i])
    return out


def _load_cl_spectra_blocks(path: str) -> List[np.ndarray]:
    blocks: List[np.ndarray] = []
    cur: List[List[float]] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                if cur:
                    blocks.append(np.asarray(cur, dtype=float))
                    cur = []
                continue
            cur.append([float(x) for x in line.split()])
    if cur:
        blocks.append(np.asarray(cur, dtype=float))
    return blocks


def load_cl_spectra_at_time(
    run_dir: str,
    time: float,
    *,
    scalar_index: int = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (k, P_cl, n_modes_cl) from spectra_scalar_{index}.txt at nearest time."""
    path = os.path.join(run_dir, f"spectra_scalar_{scalar_index}.txt")
    if not os.path.isfile(path):
        return None
    blocks = _load_cl_spectra_blocks(path)
    times_path = os.path.join(run_dir, "average_spectra_times.txt")
    if not os.path.isfile(times_path):
        return None
    times = np.loadtxt(times_path, ndmin=1).astype(float)
    if len(blocks) != len(times):
        n = min(len(blocks), len(times))
        blocks = blocks[:n]
        times = times[:n]
    i = int(np.argmin(np.abs(times - float(time))))
    b = blocks[i]
    return b[:, 0].copy(), b[:, 1].copy(), b[:, 3].copy()


def load_cl_power_at_time(
    run_dir: str,
    time: float,
    *,
    scalar_index: int = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    cl = load_cl_spectra_at_time(run_dir, time, scalar_index=scalar_index)
    if cl is None:
        return None
    return cl[0], cl[1]


def cross_check_cl(
    k: np.ndarray,
    P: np.ndarray,
    cl_k: np.ndarray,
    cl_P: np.ndarray,
) -> Dict[str, float]:
    """Interpolate our CL-style P onto CL k-grid and report agreement."""
    ok = np.isfinite(P) & (P > 0) & np.isfinite(k) & (k > 0)
    if not ok.any():
        return {"n_finite": 0.0, "median_ratio": float("nan"), "max_ratio": float("nan")}
    ok_cl = np.isfinite(cl_P) & (cl_P > 0) & (cl_k > 0)
    if not ok_cl.any():
        return {"n_finite": float(ok.sum()), "median_ratio": float("nan"), "max_ratio": float("nan")}
    P_interp = np.interp(cl_k[ok_cl], k[ok], P[ok], left=np.nan, right=np.nan)
    ratio = P_interp / cl_P[ok_cl]
    fin = np.isfinite(ratio) & (ratio > 0)
    if not fin.any():
        return {"n_finite": float(ok.sum()), "median_ratio": float("nan"), "max_ratio": float("nan")}
    out = {
        "n_finite": float(fin.sum()),
        "median_ratio": float(np.median(ratio[fin])),
        "max_ratio": float(np.max(ratio[fin])),
        "min_ratio": float(np.min(ratio[fin])),
    }
    # Explicit low-k check (first CL bin)
    i0 = int(np.argmin(cl_k[ok_cl]))
    k0 = float(cl_k[ok_cl][i0])
    p0 = float(P_interp[i0]) if np.isfinite(P_interp[i0]) else float("nan")
    out["k_low_cl"] = k0
    out["P_low_mine"] = p0
    out["P_low_cl"] = float(cl_P[ok_cl][i0])
    out["P_low_ratio"] = p0 / cl_P[ok_cl][i0] if cl_P[ok_cl][i0] > 0 else float("nan")
    return out


def _vev_prog_from_params(params: Dict[str, Any], f_star: float) -> float:
    vev = float(params.get("vev", params.get("phi0", f_star)))
    return vev / max(float(f_star), 1e-30)


def _infer_vev_prog(
    params: Dict[str, Any],
    f_star: float,
    rho_prog: Optional[np.ndarray],
) -> Tuple[float, float, Optional[float]]:
    """Return (vev_used, vev_from_params, rho_p95)."""
    vev_params = _vev_prog_from_params(params, f_star)
    if rho_prog is None:
        return vev_params, vev_params, None
    p95 = float(np.percentile(rho_prog.astype(np.float64), 95))
    # During PT the p95 of |Φ| tracks the broken-phase scale better than params.
    if p95 > 1.5 * max(vev_params, 1e-12):
        LOG.warning(
            "vev_prog from params=%.4e but |Φ| p95=%.4e — using snapshot scale",
            vev_params,
            p95,
        )
        return p95, vev_params, p95
    return vev_params, vev_params, p95


def _field_stats(phi0: np.ndarray, phi1: Optional[np.ndarray]) -> Dict[str, float]:
    p0 = phi0.astype(np.float64)
    out = {
        "phi0_mean": float(p0.mean()),
        "phi0_std": float(p0.std()),
        "phi0_min": float(p0.min()),
        "phi0_max": float(p0.max()),
    }
    if phi1 is not None:
        p1 = phi1.astype(np.float64)
        rho = np.sqrt(p0 * p0 + p1 * p1)
        out.update(
            {
                "phi1_mean": float(p1.mean()),
                "phi1_std": float(p1.std()),
                "rho_mean": float(rho.mean()),
                "rho_p95": float(np.percentile(rho, 95)),
                "rho_max": float(rho.max()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# build δ(x) in program units
# ---------------------------------------------------------------------------
def build_delta_field(
    phi0_prog: np.ndarray,
    phi1_prog: Optional[np.ndarray],
    *,
    field: str,
    vev_prog: float,
    bulk_frac: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mean-subtracted fluctuation field for correlators (program units)."""
    p0 = np.asarray(phi0_prog, dtype=np.float32)
    extra: Dict[str, Any] = {"vev_prog": vev_prog, "bulk_frac": bulk_frac}

    if field == "phi0_prog":
        delta = p0 - np.float32(p0.mean())
        extra["cl_scalar_index"] = 0
        return delta, extra

    if phi1_prog is None:
        raise ValueError(f"field={field} requires n_scalars >= 2")

    p1 = np.asarray(phi1_prog, dtype=np.float32)

    if field == "phi1_prog":
        delta = p1 - np.float32(p1.mean())
        extra["cl_scalar_index"] = 1
        return delta, extra

    rho_prog = np.sqrt(p0 * p0 + p1 * p1, dtype=np.float32)
    inv_vev = np.float32(1.0 / max(vev_prog, 1e-30))

    if field == "rho_norm":
        rho_n = rho_prog * inv_vev
        delta = rho_n - np.float32(rho_n.mean())
        extra["cl_scalar_index"] = None  # no direct CL file for |Φ|
        return delta, extra

    if field == "theta_bulk":
        theta = np.arctan2(p1, p0, dtype=np.float32)
        bulk = rho_prog * inv_vev > np.float32(bulk_frac)
        n_bulk = int(bulk.sum())
        extra["bulk_fraction"] = float(n_bulk) / float(p0.size)
        if n_bulk < 8:
            LOG.warning("theta_bulk: only %d bulk voxels", n_bulk)
            return np.zeros_like(p0), extra
        mu = float(theta[bulk].mean())
        delta = np.where(bulk, theta - np.float32(mu), np.float32(0.0)).astype(np.float32)
        extra["cl_scalar_index"] = None
        return delta, extra

    raise ValueError(f"unknown field={field!r}")


def _load_phi_snapshot(
    h5_path: str,
    row: Dict[str, Any],
    time_key: Optional[str],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Load φ₀/φ₁ once; return arrays + shared snapshot meta (no δ yet)."""
    f_star = float(row["fStar"])
    n_scalars = int(float(row["n_scalars"]))
    phi0 = np.asarray(read_h5_field(h5_path, "phi_0", time_key), dtype=np.float32)
    N = int(phi0.shape[0])
    phi1: Optional[np.ndarray] = None
    if n_scalars >= 2:
        phi1 = np.asarray(read_h5_field(h5_path, "phi_1", time_key, N=N), dtype=np.float32)

    fstats = _field_stats(phi0, phi1)
    rho_for_vev: Optional[np.ndarray] = None
    if phi1 is not None:
        rho_for_vev = np.sqrt(
            phi0.astype(np.float64) ** 2 + phi1.astype(np.float64) ** 2
        ).astype(np.float32)
    vev_prog, vev_from_params, _ = _infer_vev_prog(params, f_star, rho_for_vev)
    if rho_for_vev is not None:
        del rho_for_vev

    snap_meta = {
        "N_full": N,
        "step": int(float(row["step"])),
        "time": float(row["t"]),
        "temperature": float(row["T"]),
        "a": float(row["a"]),
        "fStar": f_star,
        "vev_prog": vev_prog,
        "vev_prog_params": vev_from_params,
        **fstats,
    }
    return phi0, phi1, snap_meta


def delta_from_phi(
    phi0: np.ndarray,
    phi1: Optional[np.ndarray],
    snap_meta: Dict[str, Any],
    *,
    field: str = "rho_norm",
    downsample: int = 1,
    bulk_frac: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build mean-subtracted δ from already-loaded φ arrays."""
    vev_prog = float(snap_meta["vev_prog"])
    delta, extra = build_delta_field(
        phi0, phi1, field=field, vev_prog=vev_prog, bulk_frac=bulk_frac
    )
    ds = max(int(downsample), 1)
    if ds > 1:
        delta = np.ascontiguousarray(delta[::ds, ::ds, ::ds])
    meta = {
        **snap_meta,
        "N": int(delta.shape[0]),
        "downsample": ds,
        "field": field,
        "var_delta": float(np.mean(delta.astype(np.float64) ** 2)),
        **extra,
    }
    return delta, meta


def load_delta_from_h5(
    h5_path: str,
    row: Dict[str, Any],
    time_key: Optional[str],
    params: Dict[str, Any],
    *,
    field: str = "rho_norm",
    downsample: int = 1,
    bulk_frac: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    phi0, phi1, snap_meta = _load_phi_snapshot(h5_path, row, time_key, params)
    delta, meta = delta_from_phi(
        phi0, phi1, snap_meta,
        field=field, downsample=downsample, bulk_frac=bulk_frac,
    )
    del phi0
    if phi1 is not None:
        del phi1
    return delta, meta


# ---------------------------------------------------------------------------
# FFT / shells / bispectrum
# ---------------------------------------------------------------------------
def _k_grid(N: int) -> np.ndarray:
    """|k| with k_fund = 2π/N (matches spectra_scalar κ column)."""
    k1d = np.fft.fftfreq(N).astype(np.float64) * K_FUND
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing="ij", sparse=True)
    return np.sqrt(KX * KX + KY * KY + KZ * KZ)


def fft_delta(delta: np.ndarray) -> np.ndarray:
    return np.fft.fftn(np.asarray(delta, dtype=np.float32), axes=(0, 1, 2)).astype(np.complex64)


def shell_edges(N: int, n_bins: int, k_min: Optional[float] = None) -> np.ndarray:
    k_nyq = K_FUND * math.sqrt(3.0) * 0.95
    k0 = k_min if k_min is not None else K_FUND / max(N, 1)
    return np.logspace(math.log10(max(k0, 1e-6)), math.log10(k_nyq), n_bins + 1)


def _shell_mean_power(power: np.ndarray, mask: np.ndarray) -> float:
    v = power[mask]
    v = v[np.isfinite(v)]
    if v.size < 8:
        return float("nan")
    return float(np.mean(v))


# Fork workers inherit these read-only FFT buffers (COW).
_EQ_DK: Optional[np.ndarray] = None
_EQ_KMAG: Optional[np.ndarray] = None
_EQ_POWER: Optional[np.ndarray] = None
_EQ_EDGES: Optional[np.ndarray] = None
_EQ_N3: float = 0.0


def _default_n_workers(N: int, requested: Optional[int] = None) -> int:
    """Cap workers by grid size so scratch IFFTs fit in RAM."""
    cpus = max(int(os.cpu_count() or 1), 1)
    if requested is not None and int(requested) > 0:
        want = int(requested)
    else:
        want = cpus
    # ~3×N³×8 B scratch per worker (shell + real + FFT); leave headroom.
    if N >= 1024:
        cap = 1
    elif N >= 768:
        cap = 2
    elif N >= 512:
        cap = 4
    elif N >= 256:
        cap = 8
    else:
        cap = cpus
    return max(1, min(want, cap, cpus))


def _eq_bin_task(i: int) -> Tuple[int, int, float, float, float]:
    """Shell-filter one equilateral bin. Returns (i, n_modes, P_raw, P_filt, B)."""
    assert _EQ_DK is not None and _EQ_KMAG is not None
    assert _EQ_POWER is not None and _EQ_EDGES is not None
    lo = float(_EQ_EDGES[i])
    hi = float(_EQ_EDGES[i + 1])
    mask = (_EQ_KMAG >= lo) & (_EQ_KMAG < hi)
    n_m = int(mask.sum())
    p_raw = _shell_mean_power(_EQ_POWER, mask)
    if not np.isfinite(p_raw) or n_m < 8:
        return i, n_m, p_raw, float("nan"), float("nan")
    shell = np.zeros_like(_EQ_DK)
    shell[mask] = _EQ_DK[mask]
    real = np.fft.ifftn(shell, axes=(0, 1, 2)).real.astype(np.float64)
    del shell
    p_filt = float(np.mean(real ** 2))
    b = float(np.mean(real ** 3))
    del real
    return i, n_m, p_raw, p_filt, b


def _run_eq_bins_parallel(
    n_bins: int,
    n_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P_raw = np.zeros(n_bins, dtype=np.float64)
    P_filt = np.zeros(n_bins, dtype=np.float64)
    B = np.zeros(n_bins, dtype=np.float64)
    n_modes = np.zeros(n_bins, dtype=np.int64)
    done = 0
    log_every = max(n_bins // 8, 1)
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_eq_bin_task, i) for i in range(n_bins)]
        for fut in as_completed(futs):
            i, n_m, p_raw, p_filt, b = fut.result()
            n_modes[i] = n_m
            P_raw[i] = p_raw
            P_filt[i] = p_filt
            B[i] = b
            done += 1
            if done % log_every == 0 or done == n_bins:
                with np.errstate(divide="ignore", invalid="ignore"):
                    q_i = b / (3.0 * p_filt ** 2) if np.isfinite(p_filt) else float("nan")
                LOG.info(
                    "  bin progress %d/%d  (last i=%d  P=%.3e  B=%.3e  Q=%.3e  n=%d)",
                    done, n_bins, i + 1, p_filt, b, q_i, n_m,
                )
    return P_raw, P_filt, B, n_modes


def analyze_correlators(
    delta: np.ndarray,
    *,
    n_bins: int = 64,
    do_squeezed: bool = True,
    k_soft_max: float = 0.05,
    n_hard_bins: int = 16,
    n_workers: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """One forward FFT → P(k), equilateral B_eq, optional squeezed proxy."""
    global _EQ_DK, _EQ_KMAG, _EQ_POWER, _EQ_EDGES, _EQ_N3

    N = int(delta.shape[0])
    n3 = float(N) ** 3
    nw = _default_n_workers(N, n_workers)
    t0 = time.time()
    delta_k = fft_delta(delta)
    delta_k.flat[0] = 0.0
    LOG.info("  FFT %.1fs (N=%d)", time.time() - t0, N)

    kmag = _k_grid(N)
    edges = shell_edges(N, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # |δ̃|²/N³ : Parseval → mean over all k ≈ Var(δ)
    power = (delta_k.real.astype(np.float64) ** 2 + delta_k.imag.astype(np.float64) ** 2) / n3
    power.flat[0] = 0.0

    _EQ_DK = delta_k
    _EQ_KMAG = kmag
    _EQ_POWER = power
    _EQ_EDGES = edges
    _EQ_N3 = n3

    t1 = time.time()
    if nw <= 1:
        LOG.info("  equilateral shells: %d bins (serial)", n_bins)
        P_raw = np.zeros(n_bins, dtype=np.float64)
        P_filt = np.zeros(n_bins, dtype=np.float64)
        B = np.zeros(n_bins, dtype=np.float64)
        n_modes = np.zeros(n_bins, dtype=np.int64)
        for i in range(n_bins):
            i2, n_m, p_raw, p_filt, b = _eq_bin_task(i)
            n_modes[i2] = n_m
            P_raw[i2] = p_raw
            P_filt[i2] = p_filt
            B[i2] = b
            if (i + 1) % max(n_bins // 8, 1) == 0 or i == n_bins - 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    q_i = b / (3.0 * p_filt ** 2) if np.isfinite(p_filt) else float("nan")
                LOG.info(
                    "  bin %2d/%d  k=%.4f  P=%.3e  B=%.3e  Q_eq=%.3e  n=%d",
                    i + 1, n_bins, centers[i], p_filt, b, q_i, n_m,
                )
    else:
        LOG.info("  equilateral shells: %d bins × %d workers (fork)", n_bins, nw)
        # Keep FFT libs single-threaded so workers don't oversubscribe.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        P_raw, P_filt, B, n_modes = _run_eq_bins_parallel(n_bins, nw)
    LOG.info("  equilateral shells done in %.1fs", time.time() - t1)

    _EQ_DK = _EQ_KMAG = _EQ_POWER = _EQ_EDGES = None

    # CL-style shell average from FFT modes (cross-check vs P_filt)
    P = P_raw * n_modes.astype(np.float64) / n3
    parseval_mean = float(np.mean(power[np.isfinite(power)]))

    # Scoccimarro reduced equilateral bispectrum: Q = B / (3 P²)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = B / (3.0 * P_filt ** 2)
        skew = B / (P_filt ** 1.5)

    eq = {
        "k": centers,
        "k_lo": edges[:-1],
        "k_hi": edges[1:],
        "P": P,
        "P_raw": P_raw,
        "P_filt": P_filt,
        "B_eq": B,
        "Q_eq": Q,
        "skew": skew,
        "n_modes": n_modes.astype(float),
        "parseval_mean_power": np.asarray([parseval_mean]),
    }

    sq: Optional[Dict[str, np.ndarray]] = None
    if do_squeezed:
        soft_mask = (kmag > 0) & (kmag < k_soft_max)
        soft = np.zeros_like(delta_k)
        soft[soft_mask] = delta_k[soft_mask]
        soft_x = np.fft.ifftn(soft, axes=(0, 1, 2)).real.astype(np.float64)
        del soft
        soft2 = soft_x * soft_x
        del soft_x

        h_edges = shell_edges(N, n_hard_bins, k_min=k_soft_max)
        h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])
        Bsq = np.zeros(n_hard_bins, dtype=np.float64)
        P_hard = np.zeros(n_hard_bins, dtype=np.float64)
        nm_h = np.zeros(n_hard_bins, dtype=np.int64)

        for j in range(n_hard_bins):
            mask = (kmag >= h_edges[j]) & (kmag < h_edges[j + 1])
            nm_h[j] = int(mask.sum())
            p_raw = _shell_mean_power(power, mask)
            P_hard[j] = p_raw * nm_h[j] / n3 if nm_h[j] > 0 else float("nan")
            if nm_h[j] < 8:
                Bsq[j] = float("nan")
                continue
            hard = np.zeros_like(delta_k)
            hard[mask] = delta_k[mask]
            hx = np.fft.ifftn(hard, axes=(0, 1, 2)).real.astype(np.float64)
            del hard
            Bsq[j] = float(np.mean(soft2 * hx)) * n3
            del hx

        sq = {
            "k_hard": h_centers,
            "B_squeezed_proxy": Bsq,
            "P_hard": P_hard,
            "n_modes_hard": nm_h.astype(float),
            "k_soft_max": np.asarray([k_soft_max]),
        }

    del delta_k, power, kmag
    return eq, sq


# ---------------------------------------------------------------------------
# CSV / plots / driver
# ---------------------------------------------------------------------------
CSV_FIELDS = (
    "k", "k_lo", "k_hi", "P", "P_filt", "P_raw", "B_eq", "Q_eq", "skew", "n_modes",
    "P_cl", "P_over_Pcl",
    "P_zeta", "B_zeta",
    "k_hard", "B_squeezed_proxy", "P_hard",
)


# ---------------------------------------------------------------------------
# ζ proxy from EOS: ζ = δ / [3(ρ+p)]  ⇒  P_ζ = A² P,  B_ζ = A³ B
# with A = 1/[3(ρ+p)].  For scalar averages in CosmoLattice program units:
#   ρ = E_K + E_G + E_V ,  p = E_K - E_G/3 - E_V ,  ρ+p = 2 E_K + (2/3) E_G
# ---------------------------------------------------------------------------
def load_average_energies(run_dir: str) -> np.ndarray:
    path = os.path.join(run_dir, "average_energies.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no average_energies.txt in {run_dir}")
    return np.loadtxt(path, ndmin=2)


def eos_at_time(
    run_dir: str,
    time: float,
    *,
    rho_plus_p_floor: float = 1.0e-30,
) -> Dict[str, float]:
    """Background ρ, p, A=1/[3(ρ+p)] at nearest average_energies time."""
    en = load_average_energies(run_dir)
    if en.shape[1] < 7:
        raise ValueError(
            f"average_energies.txt has {en.shape[1]} cols; expected "
            "t, E_K0, E_G0, E_K1, E_G1, E_V, E_tot"
        )
    i = int(np.argmin(np.abs(en[:, 0] - float(time))))
    t = float(en[i, 0])
    e_k = float(en[i, 1] + en[i, 3])
    e_g = float(en[i, 2] + en[i, 4])
    e_v = float(en[i, 5])
    e_tot = float(en[i, 6])
    rho = e_k + e_g + e_v
    p = e_k - e_g / 3.0 - e_v
    rpp = 2.0 * e_k + (2.0 / 3.0) * e_g
    rpp_use = max(rpp, float(rho_plus_p_floor))
    A = 1.0 / (3.0 * rpp_use)
    return {
        "t_eos": t,
        "E_K": e_k,
        "E_G": e_g,
        "E_V": e_v,
        "E_tot": e_tot,
        "rho": rho,
        "p": p,
        "rho_plus_p": rpp,
        "rho_plus_p_used": rpp_use,
        "A_zeta": A,
        "w": p / rho if abs(rho) > 0 else float("nan"),
        "floored": float(rpp < float(rho_plus_p_floor)),
    }


def attach_zeta_proxy(
    eq: Dict[str, np.ndarray],
    A: float,
) -> Dict[str, np.ndarray]:
    """In-place add P_zeta, B_zeta using ζ = A δ."""
    A = float(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        eq["P_zeta"] = (A * A) * eq["P"]
        eq["B_zeta"] = (A * A * A) * eq["B_eq"]
    eq["A_zeta"] = np.asarray([A], dtype=np.float64)
    return eq

DIAG_FIELDS = (
    "time", "step", "field", "N", "var_delta", "vev_prog", "vev_prog_params",
    "fStar", "phi0_mean", "phi0_std", "rho_p95",
    "median_P_over_Pcl", "min_P_over_Pcl", "max_P_over_Pcl",
    "P_at_lowest_k", "P_cl_at_lowest_k",
    "A_zeta", "rho", "p", "rho_plus_p", "P_zeta_at_lowest_k",
    "status",
)


def log_snapshot_summary(
    meta: Dict[str, Any],
    eq: Dict[str, np.ndarray],
    xcheck: Dict[str, float],
) -> None:
    """Repeat key diagnostics at end of each snapshot (easy to find in long logs)."""
    parseval = float(eq["parseval_mean_power"][0]) if "parseval_mean_power" in eq else float("nan")
    var_d = float(meta.get("var_delta", float("nan")))
    ok = np.isfinite(eq["P"]) & (eq["P"] > 0)
    k_lo = float(eq["k"][ok][0]) if ok.any() else float("nan")
    p_lo = float(eq["P"][ok][0]) if ok.any() else float("nan")
    pr_lo = float(eq["P_raw"][ok][0]) if ok.any() else float("nan")

    LOG.info("=" * 72)
    LOG.info(
        "SNAPSHOT SUMMARY  t=%.3f  step=%d  field=%s",
        meta["time"],
        meta["step"],
        meta["field"],
    )
    LOG.info(
        "  N=%d  var_delta=%.6e  parseval_mean=%.6e  (should match var_delta)",
        meta["N"],
        var_d,
        parseval,
    )
    LOG.info(
        "  phi0: mean=%.4e  std=%.4e  |  vev_prog=%.4e  fStar=%.4e",
        meta.get("phi0_mean", float("nan")),
        meta.get("phi0_std", float("nan")),
        meta.get("vev_prog", float("nan")),
        meta.get("fStar", float("nan")),
    )
    LOG.info(
        "  lowest bin: k=%.4g  P=%.4e  P_filt=%.4e  B=%.4e  Q_eq=%.4e",
        k_lo,
        p_lo,
        float(eq["P_filt"][ok][0]) if ok.any() else float("nan"),
        float(eq["B_eq"][ok][0]) if ok.any() else float("nan"),
        float(eq["Q_eq"][ok][0]) if ok.any() else float("nan"),
    )
    if xcheck:
        LOG.info(
            "  CL cross-check: median P/P_cl=%.3f  low-k ratio=%.3f  "
            "(k=%.4g  mine=%.4e  cl=%.4e)",
            xcheck.get("median_ratio", float("nan")),
            xcheck.get("P_low_ratio", float("nan")),
            xcheck.get("k_low_cl", float("nan")),
            xcheck.get("P_low_mine", float("nan")),
            xcheck.get("P_low_cl", float("nan")),
        )
    LOG.info("=" * 72)


def _diagnostic_status(meta: Dict[str, Any], xcheck: Dict[str, float]) -> str:
    issues: List[str] = []
    vd = float(meta.get("var_delta", 0.0))
    if not np.isfinite(vd) or vd > 100.0:
        issues.append("var_delta_high")
    med = xcheck.get("median_ratio", float("nan"))
    if meta.get("cl_scalar_index") is not None and np.isfinite(med):
        if med > 5.0 or med < 0.2:
            issues.append("P_cl_mismatch")
    if not issues:
        return "ok"
    return "|".join(issues)


def write_snapshot_sidecar(
    path: str,
    meta: Dict[str, Any],
    xcheck: Dict[str, float],
    eq: Dict[str, np.ndarray],
) -> None:
    """Human-readable diagnostics next to each bispectrum CSV."""
    ok = np.isfinite(eq["P"]) & (eq["P"] > 0)
    payload = {
        **meta,
        "cross_check": xcheck,
        "status": _diagnostic_status(meta, xcheck),
        "parseval_mean_power": float(eq["parseval_mean_power"][0]),
        "P_at_lowest_k": float(eq["P"][ok][0]) if ok.any() else float("nan"),
        "P_raw_at_lowest_k": float(eq["P_raw"][ok][0]) if ok.any() else float("nan"),
        "k_at_lowest_k": float(eq["k"][ok][0]) if ok.any() else float("nan"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_diagnostics_table(out_dir: str, results: List[Dict[str, Any]]) -> str:
    """Summary CSV + text report with field, var(δ), CL cross-check."""
    csv_path = os.path.join(out_dir, "diagnostics.csv")
    rows: List[Dict[str, str]] = []
    lines: List[str] = [
        "Bispectrum diagnostics",
        "=" * 60,
    ]

    for res in results:
        meta = res["meta"]
        xcheck = res.get("cross_check", {})
        eq = res["eq"]
        ok = np.isfinite(eq["P"]) & (eq["P"] > 0)
        p_low = float(eq["P"][ok][0]) if ok.any() else float("nan")
        k_low = float(eq["k"][ok][0]) if ok.any() else float("nan")
        p_cl_low = float("nan")
        if res.get("cl") is not None and ok.any():
            ck, cP = res["cl"]
            p_cl_low = float(np.interp(k_low, ck, cP, left=np.nan, right=np.nan))
        p_zeta_low = float("nan")
        A_z = float("nan")
        rho_b = float("nan")
        p_b = float("nan")
        rpp = float("nan")
        eos = res.get("eos") or {}
        if eos:
            A_z = float(eos.get("A_zeta", float("nan")))
            rho_b = float(eos.get("rho", float("nan")))
            p_b = float(eos.get("p", float("nan")))
            rpp = float(eos.get("rho_plus_p", float("nan")))
            if "P_zeta" in eq and ok.any():
                p_zeta_low = float(eq["P_zeta"][ok][0])
        status = _diagnostic_status(meta, xcheck)
        row = {
            "time": f"{meta['time']:.6f}",
            "step": str(meta["step"]),
            "field": meta["field"],
            "N": str(meta["N"]),
            "var_delta": f"{meta['var_delta']:.6e}",
            "vev_prog": f"{meta.get('vev_prog', float('nan')):.6e}",
            "vev_prog_params": f"{meta.get('vev_prog_params', float('nan')):.6e}",
            "fStar": f"{meta['fStar']:.6e}",
            "phi0_mean": f"{meta.get('phi0_mean', float('nan')):.6e}",
            "phi0_std": f"{meta.get('phi0_std', float('nan')):.6e}",
            "rho_p95": f"{meta.get('rho_p95', float('nan')):.6e}",
            "median_P_over_Pcl": f"{xcheck.get('median_ratio', float('nan')):.6f}",
            "min_P_over_Pcl": f"{xcheck.get('min_ratio', float('nan')):.6f}",
            "max_P_over_Pcl": f"{xcheck.get('max_ratio', float('nan')):.6f}",
            "P_at_lowest_k": f"{p_low:.6e}",
            "P_cl_at_lowest_k": f"{p_cl_low:.6e}",
            "A_zeta": f"{A_z:.6e}",
            "rho": f"{rho_b:.6e}",
            "p": f"{p_b:.6e}",
            "rho_plus_p": f"{rpp:.6e}",
            "P_zeta_at_lowest_k": f"{p_zeta_low:.6e}",
            "status": status,
        }
        rows.append(row)
        lines.extend(
            [
                "",
                f"t={meta['time']:.3f}  step={meta['step']}  field={meta['field']}  status={status}",
                f"  N={meta['N']}  var(δ)={meta['var_delta']:.4e}  vev_prog={meta.get('vev_prog', float('nan')):.4e}",
                f"  φ₀ mean/std=[{meta.get('phi0_mean', float('nan')):.4e}, {meta.get('phi0_std', float('nan')):.4e}]"
                + (
                    f"  |Φ| p95={meta.get('rho_p95', float('nan')):.4e}"
                    if "rho_p95" in meta
                    else ""
                ),
                f"  P(k_min={k_low:.4g})={p_low:.4e}"
                + (f"  P_cl={p_cl_low:.4e}" if np.isfinite(p_cl_low) else ""),
                f"  CL ratio median/min/max="
                f"{xcheck.get('median_ratio', float('nan')):.3f}/"
                f"{xcheck.get('min_ratio', float('nan')):.3f}/"
                f"{xcheck.get('max_ratio', float('nan')):.3f}",
            ]
        )
        if status != "ok":
            lines.append("  ** WARNING: check units/field before using B or Q **")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(DIAG_FIELDS))
        w.writeheader()
        w.writerows(rows)

    txt_path = os.path.join(out_dir, "diagnostics.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    LOG.info("wrote %s", csv_path)
    LOG.info("wrote %s", txt_path)
    return csv_path


def write_csv(
    path: str,
    eq: Dict[str, np.ndarray],
    sq: Optional[Dict[str, np.ndarray]],
    cl_k: Optional[np.ndarray] = None,
    cl_P: Optional[np.ndarray] = None,
    *,
    meta: Optional[Dict[str, Any]] = None,
    xcheck: Optional[Dict[str, float]] = None,
    eos: Optional[Dict[str, float]] = None,
) -> None:
    n = len(eq["k"])
    n2 = len(sq["k_hard"]) if sq is not None else 0
    n_out = max(n, n2)

    P_cl_col = np.full(n, np.nan)
    ratio_col = np.full(n, np.nan)
    if cl_k is not None and cl_P is not None:
        ok = np.isfinite(eq["P"]) & (eq["P"] > 0)
        if ok.any():
            P_cl_col[ok] = np.interp(eq["k"][ok], cl_k, cl_P, left=np.nan, right=np.nan)
            ratio_col[ok] = eq["P"][ok] / P_cl_col[ok]

    with open(path, "w", newline="") as f:
        if meta is not None:
            f.write(f"# field={meta.get('field', '')}\n")
            f.write(f"# time={meta.get('time', '')} step={meta.get('step', '')}\n")
            f.write(f"# var_delta={meta.get('var_delta', '')}\n")
            f.write(f"# vev_prog={meta.get('vev_prog', '')} fStar={meta.get('fStar', '')}\n")
            if xcheck:
                f.write(
                    "# median_P_over_Pcl="
                    f"{xcheck.get('median_ratio', float('nan'))}\n"
                )
        if eos is not None:
            f.write(
                "# zeta_proxy: zeta = A * delta with A = 1/[3(rho+p)]; "
                "P_zeta = A^2 P, B_zeta = A^3 B_eq\n"
            )
            f.write(
                f"# A_zeta={eos.get('A_zeta', float('nan'))} "
                f"rho={eos.get('rho', float('nan'))} "
                f"p={eos.get('p', float('nan'))} "
                f"rho_plus_p={eos.get('rho_plus_p', float('nan'))} "
                f"t_eos={eos.get('t_eos', float('nan'))}\n"
            )
            f.write(
                "# NOTE: for rho_norm this treats dimensionless |Phi| contrast "
                "as if it were delta_rho in average_energies units (proxy).\n"
            )
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for i in range(n_out):
            row = {k: "" for k in CSV_FIELDS}
            if i < n:
                for key in ("k", "k_lo", "k_hi", "P", "P_filt", "P_raw", "B_eq", "Q_eq", "skew", "n_modes"):
                    v = eq[key][i]
                    row[key] = f"{v:.10e}" if np.isfinite(v) else "nan"
                if np.isfinite(P_cl_col[i]):
                    row["P_cl"] = f"{P_cl_col[i]:.10e}"
                    row["P_over_Pcl"] = f"{ratio_col[i]:.6f}"
                if "P_zeta" in eq:
                    v = eq["P_zeta"][i]
                    row["P_zeta"] = f"{v:.10e}" if np.isfinite(v) else "nan"
                if "B_zeta" in eq:
                    v = eq["B_zeta"][i]
                    row["B_zeta"] = f"{v:.10e}" if np.isfinite(v) else "nan"
            if sq is not None and i < n2:
                row["k_hard"] = f"{sq['k_hard'][i]:.10e}"
                v = sq["B_squeezed_proxy"][i]
                row["B_squeezed_proxy"] = f"{v:.10e}" if np.isfinite(v) else "nan"
                v2 = sq["P_hard"][i]
                row["P_hard"] = f"{v2:.10e}" if np.isfinite(v2) else "nan"
            w.writerow(row)


def plot_summary(
    results: List[Dict[str, Any]],
    out_png: str,
    *,
    max_overlay: int = 12,
    min_modes_plot: int = 500,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    def _reliable_mask(eq: Dict[str, np.ndarray]) -> np.ndarray:
        nm = eq.get("n_modes")
        if nm is None:
            return np.isfinite(eq["Q_eq"])
        return np.isfinite(eq["Q_eq"]) & (nm >= min_modes_plot) & np.isfinite(eq["P"]) & (eq["P"] > 0)

    if len(results) > max_overlay:
        ks = results[0]["eq"]["k"]
        ts = np.array([r["meta"]["time"] for r in results], dtype=float)
        Q = np.vstack([r["eq"]["Q_eq"] for r in results])
        P = np.vstack([r["eq"]["P"] for r in results])
        NM = np.vstack([r["eq"]["n_modes"] for r in results])
        Q = np.where(NM >= min_modes_plot, Q, np.nan)

        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
        Pp = np.ma.masked_invalid(np.log10(np.clip(P, 1e-30, None)))
        im0 = axes[0].pcolormesh(ks, ts, Pp, shading="auto", cmap="viridis")
        axes[0].set_xscale("log")
        axes[0].set_xlabel(r"$k$")
        axes[0].set_ylabel(r"$t$")
        axes[0].set_title(r"$\log_{10} P(k,t)$")
        fig.colorbar(im0, ax=axes[0], shrink=0.85)

        Qclip = np.ma.masked_invalid(Q)
        vmax = float(np.nanpercentile(np.abs(Q), 95)) if np.isfinite(Q).any() else 1.0
        vmax = max(vmax, 1e-6)
        im1 = axes[1].pcolormesh(
            ks, ts, Qclip, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel(r"$k$")
        axes[1].set_ylabel(r"$t$")
        axes[1].set_title(r"$Q_{\rm eq}(k,t)$")
        fig.colorbar(im1, ax=axes[1], shrink=0.85)

        qmean = np.nanmean(np.abs(Q), axis=1)
        axes[2].plot(ts, qmean, "C0-", lw=1.4)
        axes[2].set_xlabel(r"$t$")
        axes[2].set_ylabel(r"$\langle|Q_{\rm eq}|\rangle_k$")
        axes[2].set_title("Non-Gaussianity vs time")
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(
            f"Correlators (n={len(results)}) | {results[0]['meta']['field']}",
            fontsize=10,
        )
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.15, 0.9, max(len(results), 1))]

    q_abs_all: List[float] = []
    for res, c in zip(results, colors):
        eq = res["eq"]
        lab = f"t={res['meta']['time']:.0f}"
        m = np.isfinite(eq["P"]) & (eq["P"] > 0)
        axes[0].loglog(eq["k"][m], eq["P"][m], "-", color=c, lw=1.5, label=lab)
        if res.get("cl") is not None:
            ck, cP = res["cl"]
            ok = cP > 0
            axes[0].loglog(
                ck[ok], cP[ok], "o", ms=2, color=c, alpha=0.35, linestyle="none"
            )
        m2 = _reliable_mask(eq)
        axes[1].semilogx(
            eq["k"][m2], eq["Q_eq"][m2],
            "o-", color=c, lw=1.3, ms=3.5, label=lab,
        )
        q_abs_all.extend(np.abs(eq["Q_eq"][m2]).tolist())
        if res.get("sq") is not None:
            sq = res["sq"]
            m3 = np.isfinite(sq["B_squeezed_proxy"])
            if "n_modes_hard" in sq:
                m3 = m3 & (sq["n_modes_hard"] >= min_modes_plot)
            axes[2].semilogx(
                sq["k_hard"][m3], sq["B_squeezed_proxy"][m3],
                "o-", color=c, lw=1.3, ms=3.5, label=lab,
            )

    axes[0].set_xlabel(r"$k$ (program)")
    axes[0].set_ylabel(r"$P(k)$")
    axes[0].set_title("P(k): line=FFT, dots=CL spectra_scalar")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$k$ (program)")
    axes[1].set_ylabel(r"$Q_{\rm eq}$")
    axes[1].set_title(rf"Reduced equilateral $Q$ ($n_{{\rm modes}}\geq{min_modes_plot}$)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    if q_abs_all:
        # Cap y-range so a few noisy bins don't flatten the mid-k structure
        ymax = float(np.nanpercentile(q_abs_all, 98))
        ymax = max(ymax, 1.0)
        axes[1].set_ylim(-ymax, ymax)

    axes[2].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel(r"$k_{\rm hard}$")
    axes[2].set_ylabel("squeezed proxy")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=7)

    fig.suptitle(f"Field correlators | {results[0]['meta']['field']}", fontsize=10)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _write_field_outputs(
    out_dir: str,
    results: List[Dict[str, Any]],
    *,
    run_dir: str,
    fields: Sequence[str],
    bulk_frac: float,
    downsample: int,
    n_bins: int,
    all_times: bool,
    stride: int,
    t_min: Optional[float],
    t_max: Optional[float],
    skipped: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "bispectrum_summary.png")
    plot_summary(results, png)
    LOG.info("wrote %s", png)
    write_diagnostics_table(out_dir, results)
    meta_path = os.path.join(out_dir, "bispectrum_meta.json")
    field_tag = results[0]["meta"]["field"] if results else ",".join(fields)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "field": field_tag,
                "fields": list(fields),
                "bulk_frac": bulk_frac,
                "downsample": downsample,
                "n_bins": n_bins,
                "all_times": all_times,
                "stride": stride,
                "t_min": t_min,
                "t_max": t_max,
                "n_processed": len(results),
                "n_skipped": skipped,
                "snapshots": [r["meta"] for r in results],
                "cross_checks": [r.get("cross_check", {}) for r in results],
                "algorithm": {
                    "P": "P_raw * n_modes/N^3 (matches spectra_scalar col 1)",
                    "P_filt": "⟨δ_k(x)^2⟩ shell-filtered variance",
                    "B_eq": "⟨δ_k(x)^3⟩ shell-filtered third moment",
                    "Q_eq": "B_eq / (3 P_filt^2)  [Scoccimarro reduced equilateral bispectrum]",
                    "skew": "B_eq / P_filt^{3/2}  [filtered-field skewness]",
                    "reference": (
                        "Scoccimarro 2000, Phys. ApJ 544, 597; "
                        "Jeong & Komatsu shell-filter estimators"
                    ),
                },
            },
            f,
            indent=2,
            default=str,
        )


def run(
    run_dir: str,
    *,
    times: Optional[Sequence[float]] = None,
    all_times: bool = False,
    stride: int = 1,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    field: str = "rho_norm",
    fields: Optional[Sequence[str]] = None,
    bulk_frac: float = 0.5,
    downsample: int = 1,
    n_bins: int = 64,
    do_squeezed: bool = True,
    write_zeta: bool = True,
    rho_plus_p_floor: float = 1.0e-30,
    n_workers: Optional[int] = None,
    out_dir: Optional[str] = None,
) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "strings", "bispectrum")
    os.makedirs(out_dir, exist_ok=True)
    params = load_run_params(run_dir) or {}

    field_list = list(fields) if fields else [field]
    for f in field_list:
        if f not in FIELD_CHOICES:
            raise ValueError(f"unknown field={f!r}; choose from {FIELD_CHOICES}")
    # preserve order, drop duplicates
    seen_f: set[str] = set()
    uniq: List[str] = []
    for f in field_list:
        if f not in seen_f:
            seen_f.add(f)
            uniq.append(f)
    field_list = uniq
    multi = len(field_list) > 1
    LOG.info("fields: %s", ", ".join(field_list))

    rows = _load_manifest_any(run_dir)
    rows.sort(key=lambda r: float(r["t"]))
    if t_min is not None:
        rows = [r for r in rows if float(r["t"]) >= t_min]
    if t_max is not None:
        rows = [r for r in rows if float(r["t"]) <= t_max]

    if all_times:
        stride = max(int(stride), 1)
        selected = rows[::stride]
        LOG.info(
            "all-times: %d snapshots (stride=%d)",
            len(selected),
            stride,
        )
    else:
        if not times:
            times = [400.0, 470.0, 520.0, 581.0]
        selected = nearest_rows(rows, times)

    if not selected:
        raise RuntimeError("no snapshots selected")

    try:
        monolith = resolve_h5_path(run_dir, rows)
    except FileNotFoundError:
        monolith = None
    time_index = (
        build_h5_time_index(monolith, "phi_0")
        if monolith and os.path.isfile(monolith)
        else None
    )

    results_by_field: Dict[str, List[Dict[str, Any]]] = {f: [] for f in field_list}
    skipped = 0
    for i_row, row in enumerate(selected, start=1):
        step = int(float(row["step"]))
        t = float(row["t"])
        LOG.info("[%d/%d] step %d  t=%.3f", i_row, len(selected), step, t)
        try:
            h5_path, kind = resolve_snapshot_h5(run_dir, row, monolith_path=monolith)
        except FileNotFoundError as exc:
            LOG.warning("  skip: %s", exc)
            skipped += 1
            continue

        tkey = None
        if kind == "monolith":
            if time_index is None:
                time_index = build_h5_time_index(h5_path, "phi_0")
            from tools.cl_field_snapshot_io import lookup_h5_key

            tkey = lookup_h5_key(t, time_index)

        # Load φ once; analyze each requested field
        phi0, phi1, snap_meta = _load_phi_snapshot(h5_path, row, tkey, params)

        for fld in field_list:
            LOG.info("  --- field=%s ---", fld)
            delta, meta = delta_from_phi(
                phi0, phi1, snap_meta,
                field=fld, downsample=downsample, bulk_frac=bulk_frac,
            )
            if fld == "theta_bulk":
                LOG.info(
                    "  bulk_fraction=%.3f",
                    float(meta.get("bulk_fraction", float("nan"))),
                )

            eq, sq = analyze_correlators(
                delta, n_bins=n_bins, do_squeezed=do_squeezed, n_workers=n_workers,
            )
            del delta

            cl_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
            cl_idx = meta.get("cl_scalar_index")
            if cl_idx is not None:
                cl_pair = load_cl_power_at_time(run_dir, t, scalar_index=int(cl_idx))

            xcheck: Dict[str, float] = {}
            if cl_pair is not None:
                xcheck = cross_check_cl(eq["k"], eq["P"], cl_pair[0], cl_pair[1])
                if xcheck.get("median_ratio", 1.0) > 5 or xcheck.get("median_ratio", 1.0) < 0.2:
                    LOG.warning(
                        "  P(k) disagrees with spectra_scalar_%d — check field/units",
                        cl_idx,
                    )

            log_snapshot_summary(meta, eq, xcheck)

            eos: Optional[Dict[str, float]] = None
            if write_zeta:
                try:
                    eos = eos_at_time(
                        run_dir, t, rho_plus_p_floor=rho_plus_p_floor
                    )
                    attach_zeta_proxy(eq, eos["A_zeta"])
                    LOG.info(
                        "  zeta proxy: A=1/[3(rho+p)]=%.4e  "
                        "rho=%.4e  p=%.4e  rho+p=%.4e  (t_eos=%.3f)%s",
                        eos["A_zeta"],
                        eos["rho"],
                        eos["p"],
                        eos["rho_plus_p"],
                        eos["t_eos"],
                        " [FLOORED]" if eos["floored"] else "",
                    )
                except (FileNotFoundError, ValueError) as exc:
                    LOG.warning("  zeta proxy skipped: %s", exc)
                    eos = None

            field_out = os.path.join(out_dir, fld) if multi else out_dir
            os.makedirs(field_out, exist_ok=True)
            csv_path = os.path.join(field_out, f"bispectrum_t{t:07.1f}_step{step:010d}.csv")
            sidecar_path = csv_path.replace(".csv", ".json")
            write_csv(
                csv_path, eq, sq,
                cl_k=cl_pair[0] if cl_pair else None,
                cl_P=cl_pair[1] if cl_pair else None,
                meta=meta,
                xcheck=xcheck,
                eos=eos,
            )
            side_meta = {**meta}
            if eos is not None:
                side_meta["zeta_eos"] = eos
            write_snapshot_sidecar(sidecar_path, side_meta, xcheck, eq)
            LOG.info("  wrote %s", csv_path)
            results_by_field[fld].append({
                "meta": meta,
                "eq": eq,
                "sq": sq,
                "csv": csv_path,
                "cross_check": xcheck,
                "cl": cl_pair,
                "eos": eos,
            })

        del phi0
        if phi1 is not None:
            del phi1

    any_results = any(results_by_field[f] for f in field_list)
    if not any_results:
        raise RuntimeError("no snapshots processed (HDF5 missing?)")

    for fld in field_list:
        res = results_by_field[fld]
        if not res:
            LOG.warning("no results for field=%s", fld)
            continue
        field_out = os.path.join(out_dir, fld) if multi else out_dir
        _write_field_outputs(
            field_out,
            res,
            run_dir=run_dir,
            fields=[fld],
            bulk_frac=bulk_frac,
            downsample=downsample,
            n_bins=n_bins,
            all_times=all_times,
            stride=stride,
            t_min=t_min,
            t_max=t_max,
            skipped=skipped,
        )

    if multi:
        top_meta = os.path.join(out_dir, "bispectrum_meta.json")
        with open(top_meta, "w") as f:
            json.dump(
                {
                    "run_dir": run_dir,
                    "fields": field_list,
                    "bulk_frac": bulk_frac,
                    "downsample": downsample,
                    "n_bins": n_bins,
                    "subdirs": {fld: os.path.join(out_dir, fld) for fld in field_list},
                    "n_skipped": skipped,
                },
                f,
                indent=2,
            )
        LOG.info("multi-field outputs under %s/{%s}", out_dir, ",".join(field_list))

    return out_dir


def apply_zeta_to_existing_csvs(
    run_dir: str,
    csv_dir: str,
    *,
    rho_plus_p_floor: float = 1.0e-30,
) -> int:
    """Re-write bispectrum_*.csv under csv_dir with P_zeta, B_zeta columns.

    Uses average_energies.txt in run_dir. No HDF5 needed.
    """
    import re

    run_dir = os.path.abspath(run_dir)
    csv_dir = os.path.abspath(csv_dir)
    paths = sorted(
        p for p in os.listdir(csv_dir)
        if p.startswith("bispectrum_t") and p.endswith(".csv")
    )
    if not paths:
        raise FileNotFoundError(f"no bispectrum_t*.csv in {csv_dir}")

    n_done = 0
    summary_rows: List[Dict[str, str]] = []
    for name in paths:
        path = os.path.join(csv_dir, name)
        # parse time from filename bispectrum_t00450.0_step...
        m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
        if not m:
            LOG.warning("skip unparseable name %s", name)
            continue
        t = float(m.group(1))
        eos = eos_at_time(run_dir, t, rho_plus_p_floor=rho_plus_p_floor)
        A = float(eos["A_zeta"])

        # read numeric table (skip # comments)
        with open(path) as f:
            lines = f.readlines()
        header_comments = [ln for ln in lines if ln.startswith("#")]
        data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        if not data_lines:
            continue
        reader = csv.DictReader(data_lines)
        rows = list(reader)
        if not rows or "P" not in rows[0]:
            LOG.warning("skip %s: no P column", name)
            continue

        fieldnames = list(CSV_FIELDS)
        # keep any extra columns that were present
        for k in rows[0].keys():
            if k not in fieldnames:
                fieldnames.append(k)

        out_rows: List[Dict[str, str]] = []
        for r in rows:
            rr = {k: r.get(k, "") for k in fieldnames}
            try:
                P = float(r["P"]) if r.get("P") not in ("", "nan", None) else float("nan")
            except ValueError:
                P = float("nan")
            try:
                B = float(r["B_eq"]) if r.get("B_eq") not in ("", "nan", None) else float("nan")
            except ValueError:
                B = float("nan")
            Pz = (A * A) * P if np.isfinite(P) else float("nan")
            Bz = (A * A * A) * B if np.isfinite(B) else float("nan")
            rr["P_zeta"] = f"{Pz:.10e}" if np.isfinite(Pz) else "nan"
            rr["B_zeta"] = f"{Bz:.10e}" if np.isfinite(Bz) else "nan"
            out_rows.append(rr)

        # drop old zeta comment lines; append fresh ones
        kept = [
            ln for ln in header_comments
            if "zeta_proxy" not in ln and "A_zeta=" not in ln and "NOTE: for rho_norm" not in ln
        ]
        with open(path, "w", newline="") as f:
            for ln in kept:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            f.write(
                "# zeta_proxy: zeta = A * delta with A = 1/[3(rho+p)]; "
                "P_zeta = A^2 P, B_zeta = A^3 B_eq\n"
            )
            f.write(
                f"# A_zeta={eos['A_zeta']} rho={eos['rho']} p={eos['p']} "
                f"rho_plus_p={eos['rho_plus_p']} t_eos={eos['t_eos']}\n"
            )
            f.write(
                "# NOTE: for rho_norm this treats dimensionless |Phi| contrast "
                "as if it were delta_rho in average_energies units (proxy).\n"
            )
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

        # update sidecar json if present
        side = path.replace(".csv", ".json")
        if os.path.isfile(side):
            with open(side) as f:
                meta = json.load(f)
            meta["zeta_eos"] = eos
            with open(side, "w") as f:
                json.dump(meta, f, indent=2, default=str)

        p0 = next((float(r["P"]) for r in rows if r.get("P") not in ("", "nan") and float(r["P"]) > 0), float("nan"))
        summary_rows.append({
            "file": name,
            "t": f"{t:.6f}",
            "A_zeta": f"{A:.6e}",
            "rho": f"{eos['rho']:.6e}",
            "p": f"{eos['p']:.6e}",
            "rho_plus_p": f"{eos['rho_plus_p']:.6e}",
            "P_zeta_low": f"{(A*A)*p0:.6e}" if np.isfinite(p0) else "nan",
        })
        LOG.info(
            "zeta: %s  A=%.4e  rho+p=%.4e  P_zeta(low)~%.4e",
            name, A, eos["rho_plus_p"], (A * A) * p0 if np.isfinite(p0) else float("nan"),
        )
        n_done += 1

    sum_path = os.path.join(csv_dir, "zeta_eos_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "t", "A_zeta", "rho", "p", "rho_plus_p", "P_zeta_low"],
        )
        w.writeheader()
        w.writerows(summary_rows)
    LOG.info("wrote %s (%d files)", sum_path, n_done)
    return n_done


def apply_zeta_clock_to_csvs(
    run_dir: str,
    csv_dir: str,
    *,
    mu_key: str = "var_delta",
) -> int:
    """Field→ζ via effective clock: δt_eff = −δμ/μ̇,  P_ζ = (H/|μ̇|)² P.

    μ(t) is taken from diagnostics.csv (default √var_delta of rho_norm).
    This is the equal-time field route that matches the δN / time-delay
    intuition without claiming superhorizon conservation of the EOS map.

    Preferred field route for PT curvature remains: build t_c from |Φ|
    (transition_time_correlators) → P_ζ = H² P_δt.
    """
    import re

    run_dir = os.path.abspath(run_dir)
    csv_dir = os.path.abspath(csv_dir)
    diag_path = os.path.join(csv_dir, "diagnostics.csv")
    if not os.path.isfile(diag_path):
        raise FileNotFoundError(
            f"need {diag_path} with time series of {mu_key} (or var_delta)"
        )

    times: List[float] = []
    mus: List[float] = []
    with open(diag_path) as f:
        for row in csv.DictReader(f):
            t = float(row["time"])
            if mu_key in row and row[mu_key] not in ("", "nan"):
                raw = float(row[mu_key])
            else:
                raw = float(row["var_delta"])
            # use σ = √var as monotonic-ish conversion tracer during PT
            mu = math.sqrt(max(raw, 0.0))
            times.append(t)
            mus.append(mu)
    if len(times) < 2:
        raise RuntimeError("need ≥2 diagnostics rows to estimate μ̇")

    t_arr = np.asarray(times, dtype=np.float64)
    mu_arr = np.asarray(mus, dtype=np.float64)
    # central differences for μ̇
    mudot = np.gradient(mu_arr, t_arr)

    from tools.transition_time_correlators import hubble_at_time

    paths = sorted(
        p for p in os.listdir(csv_dir)
        if p.startswith("bispectrum_t") and p.endswith(".csv")
    )
    summary: List[Dict[str, str]] = []
    n_done = 0
    for name in paths:
        m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
        if not m:
            continue
        t = float(m.group(1))
        # nearest diagnostics point
        j = int(np.argmin(np.abs(t_arr - t)))
        mu_dot = float(mudot[j])
        hub = hubble_at_time(run_dir, t)
        H = float(hub["H"])
        if abs(mu_dot) < 1e-30:
            LOG.warning("skip %s: μ̇≈0", name)
            continue
        A = H / abs(mu_dot)  # |ζ| ~ H |δt| = H |δμ/μ̇|

        path = os.path.join(csv_dir, name)
        with open(path) as f:
            lines = f.readlines()
        header = [ln for ln in lines if ln.startswith("#")]
        data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        reader = csv.DictReader(data_lines)
        rows = list(reader)
        if not rows or "P" not in rows[0]:
            continue

        fieldnames = list(rows[0].keys())
        for extra in ("P_zeta_clock", "B_zeta_clock", "A_clock", "mu_dot", "H_clock"):
            if extra not in fieldnames:
                fieldnames.append(extra)

        out_rows = []
        p0 = float("nan")
        for r in rows:
            rr = dict(r)
            try:
                P = float(r["P"]) if r.get("P") not in ("", "nan") else float("nan")
            except ValueError:
                P = float("nan")
            try:
                B = float(r["B_eq"]) if r.get("B_eq") not in ("", "nan") else float("nan")
            except ValueError:
                B = float("nan")
            Pz = (A * A) * P if np.isfinite(P) else float("nan")
            Bz = (A * A * A) * B if np.isfinite(B) else float("nan")
            if not np.isfinite(p0) and np.isfinite(P) and P > 0:
                p0 = P
            rr["P_zeta_clock"] = f"{Pz:.10e}" if np.isfinite(Pz) else "nan"
            rr["B_zeta_clock"] = f"{Bz:.10e}" if np.isfinite(Bz) else "nan"
            rr["A_clock"] = f"{A:.10e}"
            rr["mu_dot"] = f"{mu_dot:.10e}"
            rr["H_clock"] = f"{H:.10e}"
            out_rows.append(rr)

        kept = [ln for ln in header if "zeta_clock" not in ln and "A_clock=" not in ln]
        with open(path, "w", newline="") as f:
            for ln in kept:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            f.write(
                "# zeta_clock: delta_t_eff = -delta_mu / mu_dot, "
                "P_zeta_clock = (H/|mu_dot|)^2 P, B_zeta_clock = (H/|mu_dot|)^3 B_eq, "
                "mu=sqrt(var_delta)\n"
            )
            f.write(
                f"# A_clock={A} mu_dot={mu_dot} H={H} t={t} "
                f"mu={mu_arr[j]} channel=effective_clock\n"
            )
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

        summary.append({
            "file": name,
            "t": f"{t:.6f}",
            "A_clock": f"{A:.6e}",
            "mu": f"{mu_arr[j]:.6e}",
            "mu_dot": f"{mu_dot:.6e}",
            "H": f"{H:.6e}",
            "P_zeta_clock_low": f"{(A*A)*p0:.6e}" if np.isfinite(p0) else "nan",
        })
        LOG.info(
            "zeta_clock: %s  A=H/|μ̇|=%.4e  μ̇=%.4e  H=%.4e",
            name, A, mu_dot, H,
        )
        n_done += 1

    sum_path = os.path.join(csv_dir, "zeta_clock_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "t", "A_clock", "mu", "mu_dot", "H", "P_zeta_clock_low"],
        )
        w.writeheader()
        w.writerows(summary)
    LOG.info("wrote %s", sum_path)

    # comparison plot vs δt_c channel if available
    _plot_zeta_channel_comparison(run_dir, csv_dir)
    return n_done


def _find_dtc_csv(run_dir: str, field_csv_dir: str, name: str) -> Optional[str]:
    parent = os.path.dirname(os.path.dirname(field_csv_dir))
    candidates = [
        os.path.join(run_dir, "strings", "transition_correlators", name),
        os.path.join(run_dir, "string_new", "strings", "transition_correlators", name),
        os.path.join(parent, "transition_correlators", name),
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def _extra_clock_csv_dirs(field_csv_dir: str) -> List[str]:
    """Sibling later-time export dirs (e.g. …/bispectrum_time_series_pt_later_time/<field>)."""
    field = os.path.basename(os.path.abspath(field_csv_dir))
    parent = os.path.dirname(os.path.abspath(field_csv_dir))  # …/bispectrum_time_series_pt
    strings = os.path.dirname(parent)
    out: List[str] = []
    for name in (
        "bispectrum_time_series_pt_later_time",
        "bispectrum_time_series_pt_late",
    ):
        cand = os.path.join(strings, name, field)
        if os.path.isdir(cand) and os.path.abspath(cand) != os.path.abspath(field_csv_dir):
            out.append(cand)
    return out


def _plot_p_zeta_clocks_xi(
    field_csv_dir: str,
    clocks: Sequence[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    times_keep: Sequence[float] = (520.0, 580.0, 640.0),
    out_name: str = "P_zeta_channels_compare.png",
    fit_xi_lo: float = 0.04,
    fit_xi_hi: float = 0.50,
) -> None:
    """Field-clock 𝒫_ζ vs ξ only (no δt_c); free IR ξⁿ fit per selected time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    want = {float(t) for t in times_keep}
    selected: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for t, k, pz, _bz, _qz in clocks:
        if min(abs(t - tw) for tw in want) > 0.6:
            continue
        ok = np.isfinite(pz) & (pz > 0) & np.isfinite(k) & (k > 0)
        if not ok.any():
            continue
        selected.append((t, k[ok], pz[ok]))
    selected.sort(key=lambda x: x[0])
    if not selected:
        LOG.warning("no clock snapshots matched times_keep=%s", times_keep)
        return

    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xi_mins: List[float] = []
    xi_maxs: List[float] = []

    for i, (t, k, pz) in enumerate(selected):
        i_pk = int(np.nanargmax(pz))
        k_star = float(k[i_pk])
        if not (np.isfinite(k_star) and k_star > 0):
            continue
        xi = k / k_star
        color = colors[i % len(colors)]
        ax.loglog(
            xi, pz, color=color, lw=1.8,
            label=rf"$t={t:.0f}$  ($k_*={k_star:.3f}$)",
        )
        fit = (xi >= fit_xi_lo) & (xi <= fit_xi_hi) & np.isfinite(pz) & (pz > 0)
        if int(np.count_nonzero(fit)) >= 3:
            # Coefficients from IR window; draw the same power law over full ξ span.
            n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(pz[fit]), 1)
            amp = float(math.exp(log_amp))
            n_fit = float(n_fit)
            xi_lo = float(np.min(xi))
            xi_hi = 7.0  # full plot span (data UV edge is ξ ≲ 8)
            xi_line = np.logspace(math.log10(xi_lo), math.log10(xi_hi), 120)
            ax.loglog(
                xi_line, amp * xi_line ** n_fit, color=color, lw=1.4, ls="--",
                label=rf"$t={t:.0f}$  $\propto\xi^{{{n_fit:.2f}}}$",
            )
        xi_mins.append(float(np.min(xi)))
        xi_maxs.append(float(np.max(xi)))

    ax.axhline(
        2.1e-9, color="0.4", ls=":", lw=1.0,
        label=r"CMB $A_s\simeq 2.1\times 10^{-9}$",
    )
    # UV edge of resolved shells is ξ ≲ 8; cap display at 7 as requested.
    if xi_mins:
        ax.set_xlim(left=min(xi_mins), right=7.0)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlabel(r"$\xi \equiv k/k_* \simeq k\,d_b$")
    ax.set_ylabel(r"$\mathcal{P}_\zeta(\xi)$")
    ax.set_title(r"Field-clock $\mathcal{P}_\zeta$ with IR $\xi^n$ fits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = os.path.join(field_csv_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", out)


def _plot_p_zeta_dtc_xi(
    field_csv_dir: str,
    dtc: Optional[Dict[str, Any]],
    *,
    out_name: str = "P_zeta_dtc_xi.png",
    fit_xi_lo: float = 0.50,
    fit_xi_hi: float = 0.85,
    xi_max: float = 7.0,
) -> None:
    """δt_c channel 𝒫_ζ(ξ) with forced ∝ξ³ guide (Jinno IR prediction)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if dtc is None:
        LOG.warning("no δt_c P_ζ data — skip %s", out_name)
        return

    k = np.asarray(dtc["k"], dtype=np.float64)
    pz = np.asarray(dtc["P"], dtype=np.float64)
    nm = np.asarray(dtc.get("nm", np.ones_like(k)), dtype=np.float64)
    ok = np.isfinite(pz) & (pz > 0) & np.isfinite(k) & (k > 0) & (nm >= 64)
    if not ok.any():
        LOG.warning("δt_c P_ζ has no usable bins — skip %s", out_name)
        return
    k, pz = k[ok], pz[ok]
    i_pk = int(np.nanargmax(pz))
    k_star = float(k[i_pk])
    if not (np.isfinite(k_star) and k_star > 0):
        return
    xi = k / k_star

    H = float(dtc["H"]) if "H" in dtc else float("nan")
    t_ref = float(dtc["t_ref"]) if "t_ref" in dtc else float("nan")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(
        xi, pz, color="C3", lw=2.0,
        label=rf"$\delta t_c$  ($k_*={k_star:.3f}$)",
    )

    # Forced ξ³ guide: amplitude from mid-IR window, drawn across full plot span.
    fit = (xi >= fit_xi_lo) & (xi <= fit_xi_hi) & np.isfinite(pz) & (pz > 0)
    if int(np.count_nonzero(fit)) >= 2:
        amp = float(np.nanmedian(pz[fit] / np.clip(xi[fit] ** 3, 1e-30, None)))
    else:
        amp = float(pz[i_pk])
    xi_lo = float(np.min(xi))
    xi_line = np.logspace(math.log10(xi_lo), math.log10(xi_max), 120)
    ax.loglog(
        xi_line, amp * xi_line ** 3, color="C3", lw=1.4, ls="--",
        label=r"$\propto\xi^3$ (Jinno IR)",
    )

    # Free IR slope (diagnostic — box barely resolves ξ≲1).
    if int(np.count_nonzero(fit)) >= 3:
        n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(pz[fit]), 1)
        amp_n = float(math.exp(log_amp))
        n_fit = float(n_fit)
        ax.loglog(
            xi_line, amp_n * xi_line ** n_fit, color="0.35", lw=1.2, ls=":",
            label=rf"free fit $\propto\xi^{{{n_fit:.2f}}}$",
        )

    ax.set_xlim(left=float(np.min(xi)), right=xi_max)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_xlabel(r"$\xi \equiv k/k_* \simeq k\,d_b$")
    ax.set_ylabel(r"$\mathcal{P}_\zeta(\xi)$")
    title = r"$\delta t_c$ channel: $\mathcal{P}_\zeta=H^2\,\mathcal{P}_{\delta t}$"
    if np.isfinite(H) and np.isfinite(t_ref):
        title += rf"  ($H={H:.2e}$, $t_{{\rm ref}}={t_ref:.0f}$)"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(field_csv_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", out)

    csv_path = dtc.get("csv_path")
    if csv_path and os.path.isfile(csv_path):
        out2 = os.path.join(os.path.dirname(csv_path), out_name)
        try:
            import shutil
            shutil.copy2(out, out2)
            LOG.info("wrote %s", out2)
        except OSError:
            pass


def _bdim_from_B(k: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Dimensionless equilateral ℬ(k) = k⁶/(2π)⁴ B(k,k,k)."""
    return (k ** 6) / ((2.0 * math.pi) ** 4) * B


def _plot_b_zeta_clocks_xi(
    field_csv_dir: str,
    clocks: Sequence[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    times_keep: Sequence[float] = (520.0, 580.0, 640.0),
    out_name: str = "B_zeta_channels_compare.png",
    fit_xi_lo: float = 0.04,
    fit_xi_hi: float = 0.50,
) -> None:
    """Field-clock |ℬ_ζ| vs ξ; free IR ξⁿ fit per selected time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    want = {float(t) for t in times_keep}
    selected: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for t, k, pz, bz, _qz in clocks:
        if min(abs(t - tw) for tw in want) > 0.6:
            continue
        ok = (
            np.isfinite(k) & (k > 0)
            & np.isfinite(pz) & (pz > 0)
            & np.isfinite(bz)
        )
        if not ok.any():
            continue
        selected.append((t, k[ok], pz[ok], bz[ok]))
    selected.sort(key=lambda x: x[0])
    if not selected:
        LOG.warning("no clock snapshots matched times_keep=%s for B", times_keep)
        return

    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xi_mins: List[float] = []
    y_vals: List[float] = []

    for i, (t, k, pz, bz) in enumerate(selected):
        i_pk = int(np.nanargmax(pz))
        k_star = float(k[i_pk])
        if not (np.isfinite(k_star) and k_star > 0):
            continue
        xi = k / k_star
        bdim = np.abs(_bdim_from_B(k, bz))
        color = colors[i % len(colors)]
        m_plot = bdim > 0
        if not m_plot.any():
            continue
        ax.loglog(
            xi[m_plot], bdim[m_plot], color=color, lw=1.8,
            label=rf"$t={t:.0f}$  ($k_*={k_star:.3f}$)",
        )
        fit = (
            (xi >= fit_xi_lo) & (xi <= fit_xi_hi)
            & np.isfinite(bdim) & (bdim > 0)
        )
        if int(np.count_nonzero(fit)) >= 3:
            n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(bdim[fit]), 1)
            amp = float(math.exp(log_amp))
            n_fit = float(n_fit)
            xi_line = np.logspace(math.log10(float(np.min(xi))), math.log10(7.0), 120)
            ax.loglog(
                xi_line, amp * xi_line ** n_fit, color=color, lw=1.4, ls="--",
                label=rf"$t={t:.0f}$  $\propto\xi^{{{n_fit:.2f}}}$",
            )
        xi_mins.append(float(np.min(xi)))
        y_vals.extend(bdim[m_plot].tolist())

    if xi_mins:
        ax.set_xlim(left=min(xi_mins), right=7.0)
    if y_vals:
        ymin = max(min(y_vals) * 0.3, 1e-20)
        ymax = max(y_vals) * 3.0
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\xi \equiv k/k_* \simeq k\,d_b$")
    ax.set_ylabel(r"$|\mathcal{B}_\zeta(\xi)|=\frac{k^6}{(2\pi)^4}|B_\zeta|$")
    ax.set_title(r"Field-clock equilateral $|\mathcal{B}_\zeta|$ with IR $\xi^n$ fits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = os.path.join(field_csv_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", out)


def _plot_b_zeta_dtc_xi(
    field_csv_dir: str,
    dtc: Optional[Dict[str, Any]],
    *,
    out_name: str = "B_zeta_dtc_xi.png",
    fit_xi_lo: float = 0.50,
    fit_xi_hi: float = 0.85,
    xi_max: float = 7.0,
    b_eq_floor: float = 1e-10,
) -> None:
    """δt_c |ℬ_ζ|(ξ) with forced ∝ξ⁶ guide (Jinno IR for white-noise δt)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if dtc is None or not np.isfinite(dtc.get("B", np.array([np.nan]))).any():
        LOG.warning("no δt_c B_ζ data — skip %s", out_name)
        return

    k = np.asarray(dtc["k"], dtype=np.float64)
    B = np.asarray(dtc["B"], dtype=np.float64)
    nm = np.asarray(dtc.get("nm", np.ones_like(k)), dtype=np.float64)
    B_eq = np.asarray(dtc.get("B_eq", B), dtype=np.float64)
    P = np.asarray(dtc.get("P", np.full_like(k, np.nan)), dtype=np.float64)

    # Drop shell-filter numerical zeros (alternate empty shells).
    ok = (
        np.isfinite(B) & np.isfinite(k) & (k > 0) & (nm >= 64)
        & (np.abs(B_eq) > b_eq_floor)
    )
    if not ok.any():
        LOG.warning("δt_c B_ζ has no usable bins — skip %s", out_name)
        return
    k, B, P = k[ok], B[ok], P[ok]
    bdim = np.abs(_bdim_from_B(k, B))

    # k_* from 𝒫_ζ peak when available (same ξ as P_zeta_dtc_xi).
    ok_p = np.isfinite(P) & (P > 0)
    if ok_p.any():
        k_star = float(k[ok_p][int(np.nanargmax(P[ok_p]))])
    else:
        k_star = float(k[int(np.nanargmax(bdim))])
    if not (np.isfinite(k_star) and k_star > 0):
        return
    xi = k / k_star

    H = float(dtc["H"]) if "H" in dtc else float("nan")
    t_ref = float(dtc["t_ref"]) if "t_ref" in dtc else float("nan")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(
        xi, bdim, color="C3", lw=2.0,
        label=rf"$\delta t_c$  ($k_*={k_star:.3f}$)",
    )

    fit = (xi >= fit_xi_lo) & (xi <= fit_xi_hi) & np.isfinite(bdim) & (bdim > 0)
    if int(np.count_nonzero(fit)) >= 2:
        amp6 = float(np.nanmedian(bdim[fit] / np.clip(xi[fit] ** 6, 1e-30, None)))
    else:
        amp6 = float(bdim[int(np.nanargmax(bdim))])
    xi_lo = float(np.min(xi))
    xi_line = np.logspace(math.log10(xi_lo), math.log10(xi_max), 120)
    ax.loglog(
        xi_line, amp6 * xi_line ** 6, color="C3", lw=1.4, ls="--",
        label=r"$\propto\xi^6$ (Jinno IR)",
    )
    if int(np.count_nonzero(fit)) >= 3:
        n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(bdim[fit]), 1)
        amp_n = float(math.exp(log_amp))
        n_fit = float(n_fit)
        ax.loglog(
            xi_line, amp_n * xi_line ** n_fit, color="0.35", lw=1.2, ls=":",
            label=rf"free fit $\propto\xi^{{{n_fit:.2f}}}$",
        )

    ax.set_xlim(left=float(np.min(xi)), right=xi_max)
    ymin = max(float(np.nanmin(bdim)) * 0.3, 1e-25)
    ymax = float(np.nanmax(bdim)) * 30.0
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\xi \equiv k/k_* \simeq k\,d_b$")
    ax.set_ylabel(r"$|\mathcal{B}_\zeta(\xi)|=\frac{k^6}{(2\pi)^4}|B_\zeta|$")
    title = r"$\delta t_c$ channel: $|\mathcal{B}_\zeta|=|( - H)^3|\,\mathcal{B}_{\delta t}$"
    if np.isfinite(H) and np.isfinite(t_ref):
        title += rf"  ($H={H:.2e}$, $t_{{\rm ref}}={t_ref:.0f}$)"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(field_csv_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", out)

    csv_path = dtc.get("csv_path") or dtc.get("b_csv_path")
    if csv_path and os.path.isfile(str(csv_path)):
        out2 = os.path.join(os.path.dirname(str(csv_path)), out_name)
        try:
            import shutil
            shutil.copy2(out, out2)
            LOG.info("wrote %s", out2)
        except OSError:
            pass


def _plot_zeta_channel_comparison(run_dir: str, field_csv_dir: str) -> None:
    """Overlay P_ζ, B_ζ(k,k,k), Q_ζ,eq: δt_c (accumulated) vs field-clock snapshots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import re

    dtc_p_path = _find_dtc_csv(run_dir, field_csv_dir, "P_zeta_dtc.csv")
    dtc_b_path = _find_dtc_csv(run_dir, field_csv_dir, "bispectrum_zeta_dtc.csv")

    clock_dirs = [field_csv_dir] + _extra_clock_csv_dirs(field_csv_dir)
    clock_files: List[Tuple[float, str]] = []
    seen_t: set[float] = set()
    for cdir in clock_dirs:
        for name in sorted(os.listdir(cdir)):
            if not (name.startswith("bispectrum_t") and name.endswith(".csv")):
                continue
            m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
            if not m:
                continue
            t = float(m.group(1))
            if t in seen_t:
                continue
            seen_t.add(t)
            clock_files.append((t, os.path.join(cdir, name)))
    clock_files.sort(key=lambda x: x[0])
    if len(clock_dirs) > 1:
        LOG.info(
            "clock dirs: %s (%d snapshots)",
            ", ".join(os.path.basename(os.path.dirname(d)) for d in clock_dirs),
            len(clock_files),
        )

    def _f(row: Dict[str, str], key: str) -> float:
        v = row.get(key, "nan")
        try:
            return float(v) if v not in ("", "nan", None) else float("nan")
        except ValueError:
            return float("nan")

    # Load clock curves: (t, k, Pz, Bz, Qz)
    clocks: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for t, path in clock_files:
        with open(path) as f:
            rows = list(csv.DictReader(
                [ln for ln in f if ln.strip() and not ln.startswith("#")]
            ))
        if not rows or "P_zeta_clock" not in rows[0]:
            continue
        ks, pz, bz, qz = [], [], [], []
        for r in rows:
            n = _f(r, "n_modes")
            if not (np.isfinite(n) and n >= 64):
                continue
            kk = _f(r, "k")
            # Prefer dimensionless 𝒫 from P_raw when available.
            p_raw = _f(r, "P_raw")
            a = _f(r, "A_clock")
            p_shell_z = _f(r, "P_zeta_clock")
            if np.isfinite(p_raw) and np.isfinite(a):
                p = (kk ** 3) / (2.0 * math.pi ** 2) * (a * a) * p_raw
            else:
                p = p_shell_z
            b = _f(r, "B_zeta_clock")
            q_field = _f(r, "Q_eq")
            # Q_ζ = B_ζ/(3 P_ζ²) = Q_δ / A  for ζ = A δ
            if np.isfinite(a) and a != 0 and np.isfinite(q_field):
                q = q_field / a
            elif np.isfinite(b) and np.isfinite(p) and p > 0:
                q = b / (3.0 * p * p)
            else:
                q = float("nan")
            if not np.isfinite(kk):
                continue
            ks.append(kk)
            pz.append(p)
            bz.append(b)
            qz.append(q)
        if not ks:
            continue
        clocks.append((
            t,
            np.asarray(ks),
            np.asarray(pz),
            np.asarray(bz),
            np.asarray(qz),
        ))

    dtc: Optional[Dict[str, Any]] = None
    if dtc_p_path:
        with open(dtc_p_path) as f:
            rows = list(csv.DictReader(f))
        # Prefer dimensionless 𝒫_ζ (Jinno / CMB); fall back to legacy P_zeta column.
        p_key = "Pdim_zeta" if rows and "Pdim_zeta" in rows[0] else "P_zeta"
        if p_key not in rows[0] and "P_zeta_shell" in rows[0]:
            p_key = "P_zeta_shell"
        dtc = {
            "k": np.array([_f(r, "k") for r in rows]),
            "nm": np.array([_f(r, "n_modes") for r in rows]),
            "P": np.array([_f(r, p_key) for r in rows]),
            "B": np.array([_f(r, "B_zeta") for r in rows]) if "B_zeta" in rows[0] else np.full(len(rows), np.nan),
            "B_eq": np.array([_f(r, "B_eq") for r in rows]) if "B_eq" in rows[0] else np.full(len(rows), np.nan),
            "Q": np.array([_f(r, "Q_zeta") for r in rows]) if "Q_zeta" in rows[0] else np.full(len(rows), np.nan),
            "p_key": p_key,
            "H": _f(rows[0], "H") if "H" in rows[0] else float("nan"),
            "t_ref": _f(rows[0], "t_ref") if "t_ref" in rows[0] else float("nan"),
            "csv_path": dtc_p_path,
        }
    if dtc_b_path:
        with open(dtc_b_path) as f:
            rows = list(csv.DictReader(f))
        if dtc is None:
            dtc = {
                "k": np.array([_f(r, "k") for r in rows]),
                "nm": np.array([_f(r, "n_modes") for r in rows]),
                "P": np.array([_f(r, "P_zeta") for r in rows]),
                "B": np.array([_f(r, "B_zeta") for r in rows]) if "B_zeta" in rows[0] else np.full(len(rows), np.nan),
                "B_eq": np.array([_f(r, "B_eq") for r in rows]) if "B_eq" in rows[0] else np.full(len(rows), np.nan),
                "Q": np.array([_f(r, "Q_zeta") for r in rows]) if "Q_zeta" in rows[0] else np.full(len(rows), np.nan),
                "p_key": "P_zeta",
                "H": _f(rows[0], "H") if "H" in rows[0] else float("nan"),
                "t_ref": _f(rows[0], "t_ref") if "t_ref" in rows[0] else float("nan"),
                "b_csv_path": dtc_b_path,
            }
        else:
            if "B_zeta" in rows[0]:
                dtc["B"] = np.array([_f(r, "B_zeta") for r in rows])
            if "B_eq" in rows[0]:
                dtc["B_eq"] = np.array([_f(r, "B_eq") for r in rows])
            if "Q_zeta" in rows[0]:
                dtc["Q"] = np.array([_f(r, "Q_zeta") for r in rows])
            dtc["b_csv_path"] = dtc_b_path
            if "H" in rows[0] and not np.isfinite(dtc.get("H", float("nan"))):
                dtc["H"] = _f(rows[0], "H")
            if "t_ref" in rows[0] and not np.isfinite(dtc.get("t_ref", float("nan"))):
                dtc["t_ref"] = _f(rows[0], "t_ref")

    cmap = plt.cm.viridis
    n_c = max(len(clocks), 1)

    def _finish(ax, k_mins: List[float], ylabel: str, title: str, out_name: str) -> None:
        if k_mins:
            ax.set_xlim(left=min(k_mins))
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
        fig = ax.figure
        fig.tight_layout()
        out = os.path.join(field_csv_dir, out_name)
        fig.savefig(out, dpi=160)
        plt.close(fig)
        LOG.info("wrote %s", out)

    # ---- P_ζ (selected clocks only, ξ-axis, no δt_c) ----
    _plot_p_zeta_clocks_xi(
        field_csv_dir,
        clocks,
        times_keep=(520.0, 580.0, 640.0),
        out_name="P_zeta_channels_compare.png",
    )
    # ---- P_ζ from δt_c alone, ξ-axis, forced ξ³ ----
    _plot_p_zeta_dtc_xi(field_csv_dir, dtc, out_name="P_zeta_dtc_xi.png")

    # ---- B_ζ (selected clocks only, ξ-axis, no δt_c) ----
    _plot_b_zeta_clocks_xi(
        field_csv_dir,
        clocks,
        times_keep=(520.0, 580.0, 640.0),
        out_name="B_zeta_channels_compare.png",
    )
    # ---- B_ζ from δt_c alone, ξ-axis, forced ξ⁶ ----
    _plot_b_zeta_dtc_xi(field_csv_dir, dtc, out_name="B_zeta_dtc_xi.png")

    # ---- legacy raw |B_ζ| vs k (all clocks + δt_c overlay) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    k_mins = []
    for i, (t, k, _pz, bz, _qz) in enumerate(clocks):
        ok = np.isfinite(bz) & (np.abs(bz) > 0)
        if not ok.any():
            continue
        color = cmap(0.15 + 0.75 * i / max(n_c - 1, 1))
        ax.loglog(
            k[ok], np.abs(bz[ok]), color=color, lw=1.4,
            label=rf"clock $t={t:.0f}$  $|(H/|\dot\mu|)^3 B|$",
        )
        k_mins.append(float(np.min(k[ok])))
    if dtc is not None:
        ok = np.isfinite(dtc["B"]) & (np.abs(dtc["B"]) > 0) & (dtc["nm"] >= 64)
        if ok.any():
            ax.loglog(
                dtc["k"][ok], np.abs(dtc["B"][ok]), color="C3", lw=2.2,
                label=r"$\delta t_c$ / $\delta N$ (accumulated)  $|(-H)^3 B_{\delta t}|$",
            )
            k_mins.append(float(np.min(dtc["k"][ok])))
    _finish(
        ax, k_mins, r"$|B_\zeta(k,k,k)|$",
        r"$|B_\zeta(k,k,k)|$ vs $k$ (raw; see also $\xi$-axis figures)",
        "B_zeta_channels_compare_vs_k.png",
    )

    # signed B_ζ for δt_c alone (readable scale)
    if dtc is not None:
        ok = np.isfinite(dtc["B"]) & (dtc["nm"] >= 64)
        if ok.any():
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            ax.semilogx(dtc["k"][ok], dtc["B"][ok], color="C3", lw=1.8)
            ax.axhline(0.0, color="k", lw=0.7, ls=":")
            ax.set_xlim(left=float(np.min(dtc["k"][ok])))
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$B_\zeta(k,k,k)=(-H)^3 B_{\delta t}$")
            ax.set_title(r"Equilateral $B_\zeta$ from $\delta t_c$ (accumulated $\delta N$)")
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            out = os.path.join(field_csv_dir, "B_zeta_dtc_signed.png")
            fig.savefig(out, dpi=160)
            plt.close(fig)
            LOG.info("wrote %s", out)

    # ---- Q_ζ,eq (mid-k only: IR/UV Q is noise-dominated)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    k_mins = []
    q_for_ylim: List[float] = []
    for i, (t, k, _pz, _bz, qz) in enumerate(clocks):
        ok = np.isfinite(qz) & (k > 0.05) & (k < 2.0)
        if not ok.any():
            continue
        color = cmap(0.15 + 0.75 * i / max(n_c - 1, 1))
        ax.semilogx(
            k[ok], qz[ok], color=color, lw=1.4,
            label=rf"clock $t={t:.0f}$  $Q_{{\rm eq}}/A$",
        )
        k_mins.append(float(np.min(k[ok])))
        q_for_ylim.extend(qz[ok].tolist())
    if dtc is not None:
        ok = np.isfinite(dtc["Q"]) & (dtc["nm"] >= 64) & (dtc["k"] > 0.05) & (dtc["k"] < 2.0)
        if ok.any():
            ax.semilogx(
                dtc["k"][ok], dtc["Q"][ok], color="C3", lw=2.2,
                label=r"$\delta t_c$ / $\delta N$ (accumulated)  $Q_{\zeta,{\rm eq}}$",
            )
            k_mins.append(float(np.min(dtc["k"][ok])))
            q_for_ylim.extend(dtc["Q"][ok].tolist())
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    if q_for_ylim:
        ymax = max(float(np.nanpercentile(np.abs(q_for_ylim), 95)), 1e-6)
        ax.set_ylim(-ymax, ymax)
    _finish(
        ax, k_mins, r"$Q_{\zeta,{\rm eq}}(k)$",
        r"$Q_{\zeta,{\rm eq}}$: accumulated $\delta t_c$ vs field-clock snapshots",
        "Q_zeta_channels_compare.png",
    )



def apply_zeta_isocurvature_to_csvs(
    run_dir: str,
    csv_dir: str,
    *,
    transfer: float = 1.0,
    compare_dtc_dir: Optional[str] = None,
) -> int:
    """Isocurvature S=δθ (theta_bulk) → ζ = T·S → P_ζ = T² P_S, B_ζ = T³ B.

    ``transfer`` T is the linear conversion coefficient after (or during)
    reheating / curvaton-like decay.  T=1 means full conversion ζ=S
    (order-of-magnitude / upper-bound map).  Curvaton-like: T≈r/3 with
    r=3ρ_σ/(4ρ_r+3ρ_σ).

    Does **not** need HDF5 — rewrites existing theta_bulk bispectrum CSVs.
    """
    import re

    run_dir = os.path.abspath(run_dir)
    csv_dir = os.path.abspath(csv_dir)
    T = float(transfer)
    if T == 0.0:
        raise ValueError("iso-transfer T must be nonzero")

    paths = sorted(
        p for p in os.listdir(csv_dir)
        if p.startswith("bispectrum_t") and p.endswith(".csv")
    )
    if not paths:
        raise FileNotFoundError(f"no bispectrum_t*.csv in {csv_dir}")

    summary: List[Dict[str, str]] = []
    n_done = 0
    for name in paths:
        m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
        t = float(m.group(1)) if m else float("nan")
        path = os.path.join(csv_dir, name)
        with open(path) as f:
            lines = f.readlines()
        header = [ln for ln in lines if ln.startswith("#")]
        data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        reader = csv.DictReader(data_lines)
        rows = list(reader)
        if not rows or "P" not in rows[0]:
            continue

        fieldnames = list(rows[0].keys())
        for extra in (
            "P_S", "P_zeta_iso", "B_zeta_iso", "Q_zeta_iso",
            "T_iso", "H_iso",
        ):
            if extra not in fieldnames:
                fieldnames.append(extra)

        hub = None
        try:
            from tools.transition_time_correlators import hubble_at_time
            hub = hubble_at_time(run_dir, t)
            H = float(hub["H"])
        except Exception:
            H = float("nan")

        out_rows = []
        p0 = float("nan")
        for r in rows:
            rr = dict(r)
            try:
                P = float(r["P"]) if r.get("P") not in ("", "nan") else float("nan")
            except ValueError:
                P = float("nan")
            try:
                B = float(r["B_eq"]) if r.get("B_eq") not in ("", "nan") else float("nan")
            except ValueError:
                B = float("nan")
            try:
                Q = float(r["Q_eq"]) if r.get("Q_eq") not in ("", "nan") else float("nan")
            except ValueError:
                Q = float("nan")
            # S ≡ δθ = theta_bulk field; keep P_S alias
            Pz = (T * T) * P if np.isfinite(P) else float("nan")
            Bz = (T ** 3) * B if np.isfinite(B) else float("nan")
            Qz = Q / T if np.isfinite(Q) and T != 0 else float("nan")
            if not np.isfinite(p0) and np.isfinite(P) and P > 0:
                p0 = P
            rr["P_S"] = f"{P:.10e}" if np.isfinite(P) else "nan"
            rr["P_zeta_iso"] = f"{Pz:.10e}" if np.isfinite(Pz) else "nan"
            rr["B_zeta_iso"] = f"{Bz:.10e}" if np.isfinite(Bz) else "nan"
            rr["Q_zeta_iso"] = f"{Qz:.10e}" if np.isfinite(Qz) else "nan"
            rr["T_iso"] = f"{T:.10e}"
            rr["H_iso"] = f"{H:.10e}" if np.isfinite(H) else "nan"
            out_rows.append(rr)

        kept = [
            ln for ln in header
            if "zeta_iso" not in ln and "T_iso=" not in ln and "isocurvature" not in ln
        ]
        with open(path, "w", newline="") as f:
            for ln in kept:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            f.write(
                "# isocurvature: S = delta_theta (theta_bulk), "
                "zeta = T * S,  P_zeta_iso = T^2 P_S, B_zeta_iso = T^3 B_eq, "
                "Q_zeta_iso = Q_eq / T\n"
            )
            f.write(
                f"# T_iso={T} t={t} H={H} "
                f"(T=1: full conversion; curvaton-like T~r/3)\n"
            )
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

        summary.append({
            "file": name,
            "t": f"{t:.6f}",
            "T_iso": f"{T:.6e}",
            "H": f"{H:.6e}",
            "P_S_low": f"{p0:.6e}" if np.isfinite(p0) else "nan",
            "P_zeta_iso_low": f"{(T*T)*p0:.6e}" if np.isfinite(p0) else "nan",
            "var_S_note": "see diagnostics var_delta",
        })
        LOG.info(
            "zeta_iso: %s  T=%.4g  P_S(low)~%.4e  P_zeta=T^2 P_S~%.4e",
            name, T, p0, (T * T) * p0 if np.isfinite(p0) else float("nan"),
        )
        n_done += 1

    sum_path = os.path.join(csv_dir, "zeta_iso_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "t", "T_iso", "H", "P_S_low", "P_zeta_iso_low", "var_S_note"],
        )
        w.writeheader()
        w.writerows(summary)
    LOG.info("wrote %s", sum_path)

    _plot_zeta_iso_comparison(run_dir, csv_dir, compare_dtc_dir=compare_dtc_dir, T=T)
    return n_done


def _plot_zeta_iso_comparison(
    run_dir: str,
    theta_csv_dir: str,
    *,
    compare_dtc_dir: Optional[str] = None,
    T: float = 1.0,
    iso_time: float = 400.0,
) -> None:
    """P_ζ from isocurvature (θ) vs δt_c channel (no aH / no EOS)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import re

    parent = os.path.dirname(os.path.dirname(theta_csv_dir))
    if compare_dtc_dir is None:
        for cand in (
            os.path.join(parent, "transition_correlators"),
            os.path.join(run_dir, "string_new", "strings", "transition_correlators"),
            os.path.join(run_dir, "strings", "transition_correlators"),
        ):
            if os.path.isfile(os.path.join(cand, "P_zeta_dtc.csv")):
                compare_dtc_dir = cand
                break

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    k_mins: List[float] = []

    best = None
    for name in sorted(os.listdir(theta_csv_dir)):
        if not (name.startswith("bispectrum_t") and name.endswith(".csv")):
            continue
        m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
        if not m:
            continue
        t = float(m.group(1))
        if best is None or abs(t - iso_time) < abs(best[0] - iso_time):
            best = (t, os.path.join(theta_csv_dir, name))
    if best is not None:
        t, path = best
        with open(path) as f:
            rows = list(csv.DictReader(
                [ln for ln in f if ln.strip() and not ln.startswith("#")]
            ))
        k, Pz, nm = [], [], []
        for r in rows:
            try:
                kk = float(r["k"])
                n = float(r.get("n_modes", 0))
                pz = float(r["P_zeta_iso"])
            except (KeyError, ValueError):
                continue
            if n < 64 or not (np.isfinite(pz) and pz > 0):
                continue
            k.append(kk)
            Pz.append(pz)
            nm.append(n)
        if k:
            k_a = np.asarray(k)
            Pz_a = np.asarray(Pz)
            ax.loglog(
                k_a, Pz_a, color="C4", lw=1.7,
                label=rf"iso $S=\delta\theta$  $P_\zeta=T^2 P_S$  ($T={T:g}$, $t={t:.0f}$)",
            )
            k_mins.append(float(np.min(k_a)))

    # optional rho clock at same t
    rho_dir = os.path.join(os.path.dirname(theta_csv_dir), "rho_norm")
    if os.path.isdir(rho_dir) and best is not None:
        t_ref = best[0]
        rho_best = None
        for name in sorted(os.listdir(rho_dir)):
            if not (name.startswith("bispectrum_t") and name.endswith(".csv")):
                continue
            m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
            if not m:
                continue
            tt = float(m.group(1))
            if rho_best is None or abs(tt - t_ref) < abs(rho_best[0] - t_ref):
                rho_best = (tt, os.path.join(rho_dir, name))
        if rho_best is not None:
            with open(rho_best[1]) as f:
                rows = list(csv.DictReader(
                    [ln for ln in f if ln.strip() and not ln.startswith("#")]
                ))
            k, Pz = [], []
            for r in rows:
                try:
                    kk = float(r["k"])
                    n = float(r.get("n_modes", 0))
                    pz = float(r["P_zeta_clock"])
                except (KeyError, ValueError):
                    continue
                if n < 64 or not (np.isfinite(pz) and pz > 0):
                    continue
                k.append(kk)
                Pz.append(pz)
            if k:
                k_a = np.asarray(k)
                ax.loglog(
                    k_a, np.asarray(Pz), color="C0", lw=1.4, ls="--",
                    label=rf"field clock $\rho$  $t={rho_best[0]:.0f}$",
                )
                k_mins.append(float(np.min(k_a)))

    if compare_dtc_dir:
        dtc_path = os.path.join(compare_dtc_dir, "P_zeta_dtc.csv")
        if os.path.isfile(dtc_path):
            with open(dtc_path) as f:
                rows = list(csv.DictReader(f))
            k = np.array([float(r["k"]) for r in rows])
            Pz = np.array([
                float(r["P_zeta"]) if r["P_zeta"] not in ("", "nan") else np.nan
                for r in rows
            ])
            nm = np.array([float(r["n_modes"]) for r in rows])
            ok = np.isfinite(Pz) & (Pz > 0) & (nm >= 64)
            if ok.any():
                ax.loglog(
                    k[ok], Pz[ok], color="C3", lw=1.8,
                    label=r"$\delta t_c$  $H^2 P_{\delta t}$",
                )
                k_mins.append(float(np.min(k[ok])))

    if k_mins:
        ax.set_xlim(left=min(k_mins))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P_\zeta(k)$")
    ax.set_title(r"$P_\zeta$: isocurvature ($\theta$) vs other channels")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(theta_csv_dir, "P_zeta_iso_compare.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", out)


def _synthetic_selftest() -> None:
    rng = np.random.default_rng(0)
    N = 64
    g = rng.standard_normal((N, N, N), dtype=np.float32)
    g -= g.mean()
    ng = (g * g - 1.0).astype(np.float32)
    ng -= ng.mean()
    eq_g, _ = analyze_correlators(g, n_bins=12, do_squeezed=False)
    eq_n, _ = analyze_correlators(ng, n_bins=12, do_squeezed=False)
    # IR bins with few modes are noise-dominated; compare well-sampled mid-k.
    well = eq_g["n_modes"] >= 3000
    q_g = float(np.nanmean(np.abs(eq_g["Q_eq"][well])))
    q_n = float(np.nanmean(np.abs(eq_n["Q_eq"][well])))
    s_g = float(np.nanmean(np.abs(eq_g["skew"][well])))
    s_n = float(np.nanmean(np.abs(eq_n["skew"][well])))
    parseval_g = float(eq_g["parseval_mean_power"][0])
    print(f"selftest: parseval={parseval_g:.3e}  mid-k <|Q|> gauss={q_g:.3e} NG={q_n:.3e}")
    print(f"selftest: mid-k <|skew|> gauss={s_g:.3e} NG={s_n:.3e}")
    if not np.isfinite(parseval_g) or not (0.5 < parseval_g < 2.0):
        raise RuntimeError("parseval mean should be O(1) for unit Gaussian")
    if not (q_n > 2 * q_g and s_n > 2 * s_g):
        raise RuntimeError("bispectrum selftest: NG should exceed Gaussian at mid-k")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--times", type=float, nargs="+", default=None)
    ap.add_argument("--all-times", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument(
        "--field",
        choices=FIELD_CHOICES,
        default="rho_norm",
        help="single field (ignored if --fields is set)",
    )
    ap.add_argument(
        "--fields",
        nargs="+",
        choices=FIELD_CHOICES,
        default=None,
        help="one or more fields in one run, e.g. --fields rho_norm theta_bulk "
             "(HDF5 loaded once per snapshot; writes out_dir/<field>/)",
    )
    ap.add_argument("--bulk-frac", type=float, default=0.5,
                    help="theta_bulk: min |Φ|/φ₀ for bulk mask")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument(
        "--n-bins",
        type=int,
        default=64,
        help="number of log-spaced k shells (default 64; try 96–128 for smoother Q(k))",
    )
    ap.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="parallel shell-filter workers (default: auto-capped by N; 1=serial)",
    )
    ap.add_argument("--no-squeezed", action="store_true")
    ap.add_argument(
        "--zeta",
        action="store_true",
        default=True,
        help="write P_zeta=A^2 P, B_zeta=A^3 B with A=1/[3(rho+p)] "
             "from average_energies.txt (default on)",
    )
    ap.add_argument(
        "--no-zeta",
        action="store_true",
        help="disable zeta proxy columns",
    )
    ap.add_argument(
        "--rho-plus-p-floor",
        type=float,
        default=1.0e-30,
        help="floor on (rho+p) to avoid A→∞ in vacuum domination",
    )
    ap.add_argument(
        "--apply-zeta-to-csvs",
        default=None,
        metavar="CSV_DIR",
        help="only reprocess existing bispectrum_*.csv in CSV_DIR "
             "(needs run_dir/average_energies.txt; no HDF5)",
    )
    ap.add_argument(
        "--apply-zeta-clock",
        default=None,
        metavar="CSV_DIR",
        help="field→ζ via effective clock δt=-δμ/μ̇ using diagnostics.csv "
             "(writes P_zeta_clock; preferred equal-time field route)",
    )
    ap.add_argument(
        "--apply-zeta-iso",
        default=None,
        metavar="CSV_DIR",
        help="isocurvature S=δθ (theta_bulk CSVs) → P_zeta_iso = T^2 P_S",
    )
    ap.add_argument(
        "--iso-transfer",
        type=float,
        default=1.0,
        help="T in ζ=T·S (default 1=full conversion; curvaton-like T~r/3)",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if args.selftest:
        _synthetic_selftest()
        # quick EOS/A check
        A = 1.0 / (3.0 * (2.0 * 0.01 + (2.0 / 3.0) * 0.03))
        assert abs(A - 1.0 / (3.0 * 0.04)) < 1e-12
        print("selftest OK")
        return 0

    if not args.run_dir:
        ap.error("run_dir required unless --selftest")

    if args.apply_zeta_to_csvs:
        apply_zeta_to_existing_csvs(
            args.run_dir,
            args.apply_zeta_to_csvs,
            rho_plus_p_floor=args.rho_plus_p_floor,
        )
        return 0

    if args.apply_zeta_clock:
        apply_zeta_clock_to_csvs(args.run_dir, args.apply_zeta_clock)
        return 0

    if args.apply_zeta_iso:
        apply_zeta_isocurvature_to_csvs(
            args.run_dir,
            args.apply_zeta_iso,
            transfer=args.iso_transfer,
        )
        return 0

    run(
        args.run_dir,
        times=args.times,
        all_times=args.all_times,
        stride=args.stride,
        t_min=args.t_min,
        t_max=args.t_max,
        field=args.field,
        fields=args.fields,
        bulk_frac=args.bulk_frac,
        downsample=args.downsample,
        n_bins=args.n_bins,
        do_squeezed=not args.no_squeezed,
        write_zeta=bool(args.zeta) and not args.no_zeta,
        rho_plus_p_floor=args.rho_plus_p_floor,
        n_workers=args.n_workers,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
