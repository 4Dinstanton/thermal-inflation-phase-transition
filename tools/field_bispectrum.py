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
    B_eq(k) = ⟨ δ_k(x)³ ⟩ · N³
    Q_eq(k) = B_eq / P(k)³

This is a standard fast proxy for equilateral B(k,k,k), **not** the full
triangle sum Σ_{k₁+k₂+k₃=0} ⟨δ̃(k₁)δ̃(k₂)δ̃(k₃)⟩. It captures non-Gaussian
cubic statistics on scale k and is appropriate for bubble/wall diagnostics.

**Squeezed proxy** (optional): ⟨ δ_soft(x)² · δ_hard(x) ⟩ vs k_hard — a cheap
wall-modulation diagnostic, not the full squeezed B(k_s, k_h, k_h).

Field choices (--field)
-----------------------
    rho_norm   |Φ|_prog/φ₀_prog − ⟨|Φ|/φ₀⟩   **default** — PT contrast, O(1)
    phi0_prog  φ₀_prog − ⟨φ₀⟩                 cross-check vs spectra_scalar_0
    phi1_prog  φ₁_prog − ⟨φ₁⟩                 cross-check vs spectra_scalar_1
    theta_bulk θ − ⟨θ⟩_bulk on |Φ|/φ₀ > frac  Goldstone / wall ripples

Usage
-----
    python tools/field_bispectrum.py <run_dir> --times 450 520 581
    python tools/field_bispectrum.py <run_dir> --all-times --stride 10 --t-min 300
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    f_star = float(row["fStar"])
    n_scalars = int(float(row["n_scalars"]))
    phi0 = np.asarray(read_h5_field(h5_path, "phi_0", time_key), dtype=np.float32)
    N = phi0.shape[0]
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
    delta, extra = build_delta_field(phi0, phi1, field=field, vev_prog=vev_prog, bulk_frac=bulk_frac)
    del phi0
    if phi1 is not None:
        del phi1
    if rho_for_vev is not None:
        del rho_for_vev

    ds = max(int(downsample), 1)
    if ds > 1:
        delta = np.ascontiguousarray(delta[::ds, ::ds, ::ds])

    meta = {
        "N": int(delta.shape[0]),
        "N_full": int(N),
        "downsample": ds,
        "field": field,
        "step": int(float(row["step"])),
        "time": float(row["t"]),
        "temperature": float(row["T"]),
        "a": float(row["a"]),
        "fStar": f_star,
        "vev_prog_params": vev_from_params,
        "var_delta": float(np.mean(delta.astype(np.float64) ** 2)),
        **fstats,
        **extra,
    }
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


def analyze_correlators(
    delta: np.ndarray,
    *,
    n_bins: int = 32,
    do_squeezed: bool = True,
    k_soft_max: float = 0.05,
    n_hard_bins: int = 16,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """One forward FFT → P(k), equilateral B_eq, optional squeezed proxy."""
    N = int(delta.shape[0])
    n3 = float(N) ** 3
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

    P_raw = np.zeros(n_bins, dtype=np.float64)
    B = np.zeros(n_bins, dtype=np.float64)
    n_modes = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = (kmag >= edges[i]) & (kmag < edges[i + 1])
        n_m = int(mask.sum())
        n_modes[i] = n_m
        P_raw[i] = _shell_mean_power(power, mask)
        if not np.isfinite(P_raw[i]):
            B[i] = float("nan")
            continue

        shell = np.zeros_like(delta_k)
        shell[mask] = delta_k[mask]
        real = np.fft.ifftn(shell, axes=(0, 1, 2)).real.astype(np.float64)
        del shell
        B[i] = float(np.mean(real ** 3)) * n3
        del real
        if (i + 1) % max(n_bins // 8, 1) == 0 or i == n_bins - 1:
            p_cl = P_raw[i] * n_m / n3 if n_m > 0 else float("nan")
            LOG.info(
                "  bin %2d/%d  k=%.4f  P=%.3e  P_raw=%.3e  n=%d  B=%.3e",
                i + 1, n_bins, centers[i], p_cl, P_raw[i], n_m, B[i],
            )

    # CL-style shell average: P(k) = P_raw * n_modes / N³
    P = P_raw * n_modes.astype(np.float64) / n3
    parseval_mean = float(np.mean(power[np.isfinite(power)]))

    with np.errstate(divide="ignore", invalid="ignore"):
        # Q uses P_raw so it stays consistent with the shell-filter B estimator.
        Q = B / (P_raw ** 3)

    eq = {
        "k": centers,
        "k_lo": edges[:-1],
        "k_hi": edges[1:],
        "P": P,
        "P_raw": P_raw,
        "B_eq": B,
        "Q_eq": Q,
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
    "k", "k_lo", "k_hi", "P", "P_raw", "B_eq", "Q_eq", "n_modes",
    "P_cl", "P_over_Pcl",
    "k_hard", "B_squeezed_proxy", "P_hard",
)

DIAG_FIELDS = (
    "time", "step", "field", "N", "var_delta", "vev_prog", "vev_prog_params",
    "fStar", "phi0_mean", "phi0_std", "rho_p95",
    "median_P_over_Pcl", "min_P_over_Pcl", "max_P_over_Pcl",
    "P_at_lowest_k", "P_cl_at_lowest_k", "status",
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
        "  lowest bin: k=%.4g  P=%.4e (CL-style)  P_raw=%.4e",
        k_lo,
        p_lo,
        pr_lo,
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
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for i in range(n_out):
            row = {k: "" for k in CSV_FIELDS}
            if i < n:
                for key in ("k", "k_lo", "k_hi", "P", "P_raw", "B_eq", "Q_eq", "n_modes"):
                    v = eq[key][i]
                    row[key] = f"{v:.10e}" if np.isfinite(v) else "nan"
                if np.isfinite(P_cl_col[i]):
                    row["P_cl"] = f"{P_cl_col[i]:.10e}"
                    row["P_over_Pcl"] = f"{ratio_col[i]:.6f}"
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
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    if len(results) > max_overlay:
        ks = results[0]["eq"]["k"]
        ts = np.array([r["meta"]["time"] for r in results], dtype=float)
        Q = np.vstack([r["eq"]["Q_eq"] for r in results])
        P = np.vstack([r["eq"]["P"] for r in results])

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
        m2 = np.isfinite(eq["Q_eq"])
        axes[1].semilogx(eq["k"][m2], eq["Q_eq"][m2], "-", color=c, lw=1.5, label=lab)
        if res.get("sq") is not None:
            sq = res["sq"]
            m3 = np.isfinite(sq["B_squeezed_proxy"])
            axes[2].semilogx(
                sq["k_hard"][m3], sq["B_squeezed_proxy"][m3], "-", color=c, lw=1.5, label=lab
            )

    axes[0].set_xlabel(r"$k$ (program)")
    axes[0].set_ylabel(r"$P(k)$")
    axes[0].set_title("P(k): line=FFT, dots=CL spectra_scalar")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$k$ (program)")
    axes[1].set_ylabel(r"$Q_{\rm eq}$")
    axes[1].set_title("Reduced equilateral bispectrum")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    axes[2].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel(r"$k_{\rm hard}$")
    axes[2].set_ylabel("squeezed proxy")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=7)

    fig.suptitle(f"Field correlators | {results[0]['meta']['field']}", fontsize=10)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run(
    run_dir: str,
    *,
    times: Optional[Sequence[float]] = None,
    all_times: bool = False,
    stride: int = 1,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    field: str = "rho_norm",
    bulk_frac: float = 0.5,
    downsample: int = 1,
    n_bins: int = 32,
    do_squeezed: bool = True,
    out_dir: Optional[str] = None,
) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "strings", "bispectrum")
    os.makedirs(out_dir, exist_ok=True)
    params = load_run_params(run_dir) or {}

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

    results: List[Dict[str, Any]] = []
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

        delta, meta = load_delta_from_h5(
            h5_path, row, tkey, params,
            field=field, downsample=downsample, bulk_frac=bulk_frac,
        )

        eq, sq = analyze_correlators(delta, n_bins=n_bins, do_squeezed=do_squeezed)
        del delta

        cl_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
        cl_idx = meta.get("cl_scalar_index")
        if cl_idx is not None:
            cl_pair = load_cl_power_at_time(run_dir, t, scalar_index=int(cl_idx))

        xcheck: Dict[str, float] = {}
        if cl_pair is not None:
            xcheck = cross_check_cl(eq["k"], eq["P"], cl_pair[0], cl_pair[1])
            if xcheck.get("median_ratio", 1.0) > 5 or xcheck.get("median_ratio", 1.0) < 0.2:
                LOG.warning("  P(k) disagrees with spectra_scalar_%d — check field/units", cl_idx)

        log_snapshot_summary(meta, eq, xcheck)

        csv_path = os.path.join(out_dir, f"bispectrum_t{t:07.1f}_step{step:010d}.csv")
        sidecar_path = csv_path.replace(".csv", ".json")
        write_csv(
            csv_path, eq, sq,
            cl_k=cl_pair[0] if cl_pair else None,
            cl_P=cl_pair[1] if cl_pair else None,
            meta=meta,
            xcheck=xcheck,
        )
        write_snapshot_sidecar(sidecar_path, meta, xcheck, eq)
        LOG.info("  wrote %s", csv_path)
        LOG.info("  wrote %s", sidecar_path)
        results.append({
            "meta": meta,
            "eq": eq,
            "sq": sq,
            "csv": csv_path,
            "cl": cl_pair,
            "cross_check": xcheck,
        })

    if not results:
        raise RuntimeError("no snapshots processed (HDF5 missing?)")

    png = os.path.join(out_dir, "bispectrum_summary.png")
    plot_summary(results, png)
    LOG.info("wrote %s", png)

    write_diagnostics_table(out_dir, results)

    meta_path = os.path.join(out_dir, "bispectrum_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "field": field,
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
                    "P": "P_raw * n_modes/N^3 in k-shell (matches spectra_scalar col 1)",
                    "P_raw": "mean(|delta_tilde|^2/N^3) per FFT mode in shell",
                    "B_eq": "⟨δ_k(x)³⟩·N³ shell-filter equilateral proxy",
                    "Q_eq": "B_eq / P³  (P uses CL-style normalization)",
                },
            },
            f,
            indent=2,
            default=str,
        )
    return out_dir


def _synthetic_selftest() -> None:
    rng = np.random.default_rng(0)
    N = 64
    g = rng.standard_normal((N, N, N), dtype=np.float32)
    g -= g.mean()
    ng = (g * g - 1.0).astype(np.float32)
    ng -= ng.mean()
    eq_g, _ = analyze_correlators(g, n_bins=12, do_squeezed=False)
    eq_n, _ = analyze_correlators(ng, n_bins=12, do_squeezed=False)
    q_g = float(np.nanmean(np.abs(eq_g["Q_eq"])))
    q_n = float(np.nanmean(np.abs(eq_n["Q_eq"])))
    p_g = float(np.nanmax(eq_g["P"]))
    p_n = float(np.nanmax(eq_n["P"]))
    parseval_g = float(eq_g["parseval_mean_power"][0])
    print(f"selftest: P_max gauss={p_g:.3e} NG={p_n:.3e} parseval={parseval_g:.3e}")
    print(f"selftest: <|Q|> gauss={q_g:.3e} NG={q_n:.3e}")
    if not np.isfinite(parseval_g) or not (0.5 < parseval_g < 2.0):
        raise RuntimeError("parseval mean should be O(1) for unit Gaussian")
    if not (q_n > 2 * q_g):
        raise RuntimeError("bispectrum selftest: NG should exceed Gaussian")


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
        choices=("rho_norm", "phi0_prog", "phi1_prog", "theta_bulk"),
        default="rho_norm",
    )
    ap.add_argument("--bulk-frac", type=float, default=0.5,
                    help="theta_bulk: min |Φ|/φ₀ for bulk mask")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--no-squeezed", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if args.selftest:
        _synthetic_selftest()
        print("selftest OK")
        return 0

    if not args.run_dir:
        ap.error("run_dir required unless --selftest")

    run(
        args.run_dir,
        times=args.times,
        all_times=args.all_times,
        stride=args.stride,
        t_min=args.t_min,
        t_max=args.t_max,
        field=args.field,
        bulk_frac=args.bulk_frac,
        downsample=args.downsample,
        n_bins=args.n_bins,
        do_squeezed=not args.no_squeezed,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
