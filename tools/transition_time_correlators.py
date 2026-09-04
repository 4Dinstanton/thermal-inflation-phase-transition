#!/usr/bin/env python3
"""Lattice estimators for C_F(r;t) and t_c correlators (tex-note style).

Post-processing only — no CosmoLattice re-run, no Poisson / spherical analytics.

Definitions (measured on the grid)
----------------------------------
False-vacuum indicator (matches ``thermal_inflation.h::falseVacuumFraction``):

    F(x,t) = 1[ |Φ|_GeV(x,t) <= φ_esc ] = 1[ ρ_prog <= φ_esc / fStar ]

Connected two-point (equal-time):

    C_F(r;t) = ⟨ δF(x) δF(x+r) ⟩_{|r|=r} ,   δF = F − ⟨F⟩

Conversion / first-crossing time (``--conversion``):

    escape   (default): false vac = ρ_prog ≤ φ_esc/fStar
             (matches CL falseVacuumFraction / langevin-off threshold)
    vev_frac: false vac = ρ_prog ≤ f·φ₀_prog  (φ₀_prog≃1 when fStar=tree VEV)
             — better proxy for bubble / true-vacuum conversion

    t_c(x) = linearly interpolated time when ρ crosses the threshold
             between consecutive snapshots (``--tc-interp``, default on).
             Without interp: first snapshot time with site converted.
             Sites still false at end of scan → ``--fill-unconverted``.

Then for δt = t_c − ⟨t_c⟩:

    C_2(r)      = ⟨ δt(x) δt(x+r) ⟩_{|r|=r}
    P_raw(k)    = ⟨ |δt̃|² / N³ ⟩_shell          (per-mode; ≈ P_3d with dx=1)
    P_δt(k)     = P_raw · n_modes / N³           (CL shell-bin; ∫≈Var)
    𝒫_δt(k)     = k³/(2π²) P_raw                 (dimensionless; Jinno-style)
    𝒫_ζ(k)      = H² 𝒫_δt = k³/(2π²) H² P_raw   (compare to CMB A_s ≃ 2×10⁻⁹)
    C_3^eq(r)   = ⟨ δt(x) δt(x+r₁) δt(x+r₂) ⟩_c
                  over random equilateral triangles of side r
                  (mean-zero → connected = raw third moment)

Note: Jinno’s low-k ∝ ξ³ scaling is for the *dimensionless* 𝒫(k), not CL P_δt.
ξ ≡ k/k_* with k_* the turnover / bubble-separation scale.

Usage
-----
    # C_F at a few times + build t_c and its correlators
    python tools/transition_time_correlators.py <run_dir> \\
        --cf-times 400 450 470 --build-tc --t-min 300 --t-max 520

    # Preferred t_c for bubble/Jinno-style δt_c (VEV fraction + interp):
    python tools/transition_time_correlators.py <run_dir> \\
        --build-tc --tc-only --conversion vev_frac --vev-frac 0.1 \\
        --t-min 300 --t-max 520

    # Reuse a saved t_c map
    python tools/transition_time_correlators.py <run_dir> \\
        --load-tc path/to/tc_map.npz --tc-only

    # Self-test (no HDF5)
    python tools/transition_time_correlators.py --selftest

    # P_ζ(k) = H² P_δt from existing P_dtc.csv
    python tools/transition_time_correlators.py <run_dir> \\
        --apply-zeta-from-dtc <out_dir> [--t-ref-zeta 400]
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
    lookup_h5_key,
    parse_manifest_row,
    read_h5_field,
    resolve_h5_path,
    resolve_snapshot_h5,
)
from tools.export_cl_snapshots import load_run_params  # noqa: E402

LOG = logging.getLogger("transition_correlators")

K_FUND = 2.0 * math.pi  # κ = 2π/(N·dx_prog), dx_prog = 1


# ---------------------------------------------------------------------------
# manifest helpers
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


def filter_rows(
    rows: List[Dict[str, Any]],
    *,
    stride: int = 1,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    rows = sorted(rows, key=lambda r: float(r["t"]))
    if t_min is not None:
        rows = [r for r in rows if float(r["t"]) >= float(t_min)]
    if t_max is not None:
        rows = [r for r in rows if float(r["t"]) <= float(t_max)]
    stride = max(int(stride), 1)
    return rows[::stride]


def resolve_escape_gev(params: Dict[str, Any], override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return float(
        params.get("langevin_off_phi_esc")
        or params.get("expansion_phi_esc")
        or 1.0e4
    )


# ---------------------------------------------------------------------------
# field → F, ρ
# ---------------------------------------------------------------------------
def rho_prog_from_phi(
    phi0: np.ndarray,
    phi1: Optional[np.ndarray],
) -> np.ndarray:
    p0 = np.asarray(phi0, dtype=np.float32)
    if phi1 is None:
        return np.abs(p0)
    p1 = np.asarray(phi1, dtype=np.float32)
    return np.sqrt(p0 * p0 + p1 * p1, dtype=np.float32)


def false_vac_mask(rho_prog: np.ndarray, thr_prog: float) -> np.ndarray:
    """F=1 on false vacuum (|Φ|_prog ≤ thr_prog)."""
    return (np.asarray(rho_prog) <= np.float32(thr_prog)).astype(np.float32)


def conversion_threshold_prog(
    *,
    mode: str,
    esc_gev: float,
    f_star: float,
    vev_frac: float = 0.1,
) -> float:
    """Program-unit threshold: site converts when ρ_prog exceeds this value.

    escape   — CL false-vacuum cut φ_esc/fStar (often ≪1; early roll-off from 0)
    vev_frac — fraction of tree VEV in program units (φ₀_prog ≃ 1 if fStar=φ₀)
    """
    mode = str(mode).lower()
    if mode == "escape":
        return float(esc_gev) / max(float(f_star), 1e-30)
    if mode == "vev_frac":
        if not (0.0 < float(vev_frac) < 1.0):
            raise ValueError(f"vev_frac must be in (0,1), got {vev_frac}")
        return float(vev_frac)
    raise ValueError(f"unknown conversion mode {mode!r} (use escape|vev_frac)")


def interpolate_crossing_times(
    t_prev: float,
    t: float,
    rho_prev: np.ndarray,
    rho: np.ndarray,
    thr: float,
    newly: np.ndarray,
) -> np.ndarray:
    """Linear ρ-crossing times on ``newly`` sites: ρ(t_c)=thr between frames."""
    rp = np.asarray(rho_prev, dtype=np.float64)
    rn = np.asarray(rho, dtype=np.float64)
    denom = rn[newly] - rp[newly]
    good = np.abs(denom) > 1e-30
    alpha = np.zeros(int(np.count_nonzero(newly)), dtype=np.float64)
    alpha[good] = (float(thr) - rp[newly][good]) / denom[good]
    alpha = np.clip(alpha, 0.0, 1.0)
    return (float(t_prev) + alpha * (float(t) - float(t_prev))).astype(np.float32)


def _load_phi(
    h5_path: str,
    row: Dict[str, Any],
    time_key: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_scalars = int(float(row["n_scalars"]))
    phi0 = np.asarray(read_h5_field(h5_path, "phi_0", time_key), dtype=np.float32)
    phi1: Optional[np.ndarray] = None
    if n_scalars >= 2:
        phi1 = np.asarray(
            read_h5_field(h5_path, "phi_1", time_key, N=int(phi0.shape[0])),
            dtype=np.float32,
        )
    return phi0, phi1


def _downsample3(arr: np.ndarray, ds: int) -> np.ndarray:
    ds = max(int(ds), 1)
    if ds == 1:
        return arr
    return np.ascontiguousarray(arr[::ds, ::ds, ::ds])


# ---------------------------------------------------------------------------
# real-space isotropic correlators
# ---------------------------------------------------------------------------
def _periodic_r2_grid(N: int) -> np.ndarray:
    """Squared lattice distance with minimum-image convention (dx=1)."""
    n = np.arange(N, dtype=np.float64)
    n = np.minimum(n, N - n)
    X2 = n[:, None, None] ** 2
    Y2 = n[None, :, None] ** 2
    Z2 = n[None, None, :] ** 2
    return X2 + Y2 + Z2


def autocorrelation_3d(delta: np.ndarray) -> np.ndarray:
    """Circular ⟨δ(x)δ(x+r)⟩ via FFT.  delta should be mean-subtracted."""
    d = np.asarray(delta, dtype=np.float32)
    N3 = float(d.size)
    fk = np.fft.fftn(d, axes=(0, 1, 2))
    corr = np.fft.ifftn(fk * np.conj(fk), axes=(0, 1, 2)).real / N3
    return corr.astype(np.float64)


def radial_bin_field(
    field: np.ndarray,
    *,
    n_bins: int,
    r_max: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Isotropic average of a 3D real-space map vs |r| (lattice units)."""
    N = int(field.shape[0])
    r2 = _periodic_r2_grid(N)
    r = np.sqrt(r2)
    r_hi = float(r_max) if r_max is not None else 0.5 * float(N)
    edges = np.linspace(0.0, r_hi, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    flat_r = r.ravel()
    flat_f = np.asarray(field, dtype=np.float64).ravel()
    flat_w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    for i in range(n_bins):
        m = (flat_r >= edges[i]) & (flat_r < edges[i + 1])
        if flat_w is not None:
            m = m & (flat_w > 0)
        n = int(m.sum())
        counts[i] = n
        if n < 1:
            continue
        mean[i] = float(np.mean(flat_f[m]))
    return {
        "r": centers,
        "r_lo": edges[:-1],
        "r_hi": edges[1:],
        "C": mean,
        "n_pairs": counts,
    }


def connected_2pt_radial(
    field: np.ndarray,
    *,
    n_bins: int = 64,
    r_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """C_2(r) = ⟨δf δf⟩_r for δf = f − ⟨f⟩."""
    f = np.asarray(field, dtype=np.float64)
    delta = (f - f.mean()).astype(np.float32)
    corr = autocorrelation_3d(delta)
    out = radial_bin_field(corr, n_bins=n_bins, r_max=r_max)
    out["var"] = np.full_like(out["r"], float(np.mean(delta.astype(np.float64) ** 2)))
    out["mean"] = np.full_like(out["r"], float(f.mean()))
    return out


def power_spectrum_shells(
    delta: np.ndarray,
    *,
    n_bins: int = 64,
) -> Dict[str, np.ndarray]:
    """CL-style P(k) = ⟨|δ̃|²/N³⟩_shell · n_modes/N³."""
    d = np.asarray(delta, dtype=np.float32)
    N = int(d.shape[0])
    n3 = float(N) ** 3
    fk = np.fft.fftn(d, axes=(0, 1, 2))
    fk.flat[0] = 0.0
    power = (fk.real.astype(np.float64) ** 2 + fk.imag.astype(np.float64) ** 2) / n3
    power.flat[0] = 0.0

    k1d = np.fft.fftfreq(N).astype(np.float64) * K_FUND
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing="ij", sparse=True)
    kmag = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    k_nyq = K_FUND * math.sqrt(3.0) * 0.95
    k0 = K_FUND / max(N, 1)
    edges = np.logspace(math.log10(max(k0, 1e-6)), math.log10(k_nyq), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    P_raw = np.full(n_bins, np.nan)
    n_modes = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        m = (kmag >= edges[i]) & (kmag < edges[i + 1])
        n_m = int(m.sum())
        n_modes[i] = n_m
        if n_m < 8:
            continue
        P_raw[i] = float(np.mean(power[m]))
    P = P_raw * n_modes.astype(np.float64) / n3
    return {
        "k": centers,
        "k_lo": edges[:-1],
        "k_hi": edges[1:],
        "P": P,
        "P_raw": P_raw,
        "n_modes": n_modes,
    }


def _random_equilateral_offsets(
    r: float,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two lattice offsets forming (approx) equilateral triangles of side r."""
    # Base triangle in xy-plane, then random rotation via orthonormal frame.
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.5, math.sqrt(3.0) / 2.0, 0.0])
    # Random orthonormal frames (n, 3, 3)
    z = rng.normal(size=(n, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    # Householder-ish second axis
    tmp = rng.normal(size=(n, 3))
    x = tmp - (tmp * z).sum(axis=1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    y = np.cross(z, x)
    # Rotated edges
    v1 = r * (x * e1[0] + y * e1[1] + z * e1[2])
    v2 = r * (x * e2[0] + y * e2[1] + z * e2[2])
    o1 = np.rint(v1).astype(np.int64)
    o2 = np.rint(v2).astype(np.int64)
    return o1, o2


def c3_eq_monte_carlo(
    delta: np.ndarray,
    r_centers: np.ndarray,
    *,
    n_samp: int = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Equilateral ⟨δ δ δ⟩ vs r by random triangle sampling (periodic)."""
    d = np.asarray(delta, dtype=np.float64)
    N = int(d.shape[0])
    rng = np.random.default_rng(seed)
    out = np.full(len(r_centers), np.nan, dtype=np.float64)
    # Cap samples for tiny grids
    n_samp = int(min(n_samp, max(N**3, 1000)))
    for i, r in enumerate(r_centers):
        if not np.isfinite(r) or r < 1.0:
            continue
        # Skip r that cannot close on the lattice after rounding
        o1, o2 = _random_equilateral_offsets(float(r), n_samp, rng)
        # Drop degenerate / non-equilateral after rounding
        d12 = o1 - o2
        len1 = np.sqrt((o1**2).sum(axis=1))
        len2 = np.sqrt((o2**2).sum(axis=1))
        len12 = np.sqrt((d12**2).sum(axis=1))
        ok = (
            (len1 > 0.5)
            & (len2 > 0.5)
            & (len12 > 0.5)
            & (np.abs(len1 - r) < 0.75)
            & (np.abs(len2 - r) < 0.75)
            & (np.abs(len12 - r) < 0.75)
        )
        if int(ok.sum()) < max(n_samp // 20, 64):
            continue
        o1 = o1[ok]
        o2 = o2[ok]
        n_use = int(o1.shape[0])
        ix = rng.integers(0, N, size=n_use)
        iy = rng.integers(0, N, size=n_use)
        iz = rng.integers(0, N, size=n_use)
        a = d[ix, iy, iz]
        b = d[(ix + o1[:, 0]) % N, (iy + o1[:, 1]) % N, (iz + o1[:, 2]) % N]
        c = d[(ix + o2[:, 0]) % N, (iy + o2[:, 1]) % N, (iz + o2[:, 2]) % N]
        out[i] = float(np.mean(a * b * c))
    return out


# ---------------------------------------------------------------------------
# I/O writers
# ---------------------------------------------------------------------------
def _write_radial_csv(path: str, table: Dict[str, np.ndarray], extra_cols: Optional[Dict[str, np.ndarray]] = None) -> None:
    cols = ["r", "r_lo", "r_hi", "C", "n_pairs"]
    data = {c: table[c] for c in cols if c in table}
    if extra_cols:
        data.update(extra_cols)
    keys = list(data.keys())
    n = len(next(iter(data.values())))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([data[k][i] for k in keys])


def _write_pk_csv(path: str, pk: Dict[str, np.ndarray]) -> None:
    keys = ["k", "k_lo", "k_hi", "P", "P_raw", "n_modes"]
    extra = [k for k in ("P_zeta", "H_used", "A_zeta") if k in pk]
    keys = keys + extra
    n = len(pk["k"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            row = []
            for key in keys:
                v = pk[key]
                arr = np.atleast_1d(v)
                if arr.size == 1:
                    row.append(float(arr[0]))
                else:
                    row.append(arr[i])
            w.writerow(row)


def hubble_at_time(run_dir: str, time: float) -> Dict[str, float]:
    """H from average_scale_factor.txt (cols: t, a, H, H_alt)."""
    path = os.path.join(run_dir, "average_scale_factor.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sf = np.loadtxt(path, ndmin=2)
    i = int(np.argmin(np.abs(sf[:, 0] - float(time))))
    t = float(sf[i, 0])
    a = float(sf[i, 1])
    H = float(sf[i, 2])
    H_alt = float(sf[i, 3]) if sf.shape[1] > 3 else H
    return {"t": t, "a": a, "H": H, "H_alt": H_alt, "aH": a * H}


def write_zeta_from_p_dtc(
    run_dir: str,
    out_dir: str,
    *,
    t_ref: Optional[float] = None,
    p_dtc_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Post-process P_dtc.csv → P_zeta_dtc.csv + plot.  ζ = −H δt_c."""
    out_dir = os.path.abspath(out_dir)
    p_path = p_dtc_path or os.path.join(out_dir, "P_dtc.csv")
    if not os.path.isfile(p_path):
        raise FileNotFoundError(p_path)

    if t_ref is None:
        c2_path = os.path.join(out_dir, "C2_dtc.csv")
        if os.path.isfile(c2_path):
            with open(c2_path) as f:
                row0 = next(csv.DictReader(f))
                t_ref = float(row0.get("tc_mean", "nan"))
        if t_ref is None or not np.isfinite(t_ref):
            meta_path = os.path.join(out_dir, "tc_correlators_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                t_ref = 0.5 * (float(meta["t_first"]) + float(meta["t_last"]))
            else:
                t_ref = 400.0

    hub = hubble_at_time(run_dir, float(t_ref))
    H = float(hub["H"])
    LOG.info(
        "zeta from δt_c: t_ref=%.3f  H=%.6e  aH=%.6e  (sf t=%.3f)",
        t_ref, H, hub["aH"], hub["t"],
    )

    with open(p_path) as f:
        rows = list(csv.DictReader(f))
    k = np.array([float(r["k"]) for r in rows], dtype=np.float64)
    P = np.array([
        float(r["P"]) if r["P"] not in ("", "nan") else np.nan for r in rows
    ])
    # Per-mode shell mean (needed for dimensionless 𝒫 ∝ k³ P_raw)
    if rows and "P_raw" in rows[0]:
        P_raw = np.array([
            float(r["P_raw"]) if r["P_raw"] not in ("", "nan") else np.nan
            for r in rows
        ])
    else:
        P_raw = np.full_like(P, np.nan)
    n_modes = np.array([float(r["n_modes"]) for r in rows], dtype=np.float64)
    k_lo = np.array([float(r["k_lo"]) for r in rows])
    k_hi = np.array([float(r["k_hi"]) for r in rows])
    Pz_shell = (H * H) * P  # CL shell-bin (legacy / Parseval)
    # Cosmology-style dimensionless spectrum (Jinno / CMB convention):
    #   𝒫(k) = k³/(2π²) P_3d(k),  P_3d ≈ P_raw for dx_prog=1
    with np.errstate(divide="ignore", invalid="ignore"):
        Pdim_dtc = (k ** 3) / (2.0 * math.pi ** 2) * P_raw
        Pdim_zeta = (H * H) * Pdim_dtc

    # Physical comoving k [GeV] if dx_phys is known (k_phys = k_prog / dx_phys)
    dx_phys = float("nan")
    try:
        from tools.export_cl_snapshots import load_run_params

        params = load_run_params(run_dir) or {}
        dx_phys = float(params.get("dx_phys", float("nan")))
    except Exception:
        pass
    with np.errstate(divide="ignore", invalid="ignore"):
        k_phys = k / dx_phys if np.isfinite(dx_phys) and dx_phys > 0 else np.full_like(k, np.nan)

    out_csv = os.path.join(out_dir, "P_zeta_dtc.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "k_lo", "k_hi", "k_phys_GeV",
            "P_dtc", "P_raw", "P_zeta_shell",
            "Pdim_dtc", "Pdim_zeta",
            "n_modes", "H", "t_ref", "aH", "dx_phys",
        ])
        for i in range(len(k)):
            def _fmt(v: float) -> object:
                return v if np.isfinite(v) else "nan"

            w.writerow([
                k[i], k_lo[i], k_hi[i], _fmt(k_phys[i]),
                _fmt(P[i]), _fmt(P_raw[i]), _fmt(Pz_shell[i]),
                _fmt(Pdim_dtc[i]), _fmt(Pdim_zeta[i]),
                int(n_modes[i]),
                H, t_ref, hub["aH"],
                dx_phys if np.isfinite(dx_phys) else "nan",
            ])
    LOG.info("wrote %s", out_csv)
    Pz = Pdim_zeta  # plots below use dimensionless by default

    c2_path = os.path.join(out_dir, "C2_dtc.csv")
    c3_path = os.path.join(out_dir, "C3eq_dtc.csv")
    if os.path.isfile(c2_path):
        with open(c2_path) as f:
            c2_rows = list(csv.DictReader(f))
        with open(os.path.join(out_dir, "C2_zeta_dtc.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "r", "r_lo", "r_hi", "C2_dtc", "C2_zeta", "n_pairs", "H", "t_ref",
            ])
            for r in c2_rows:
                C = float(r["C"]) if r["C"] not in ("", "nan") else float("nan")
                w.writerow([
                    r["r"], r["r_lo"], r["r_hi"], C,
                    (H * H) * C if np.isfinite(C) else "nan",
                    r.get("n_pairs", ""), H, t_ref,
                ])
    if os.path.isfile(c3_path):
        with open(c3_path) as f:
            c3_rows = list(csv.DictReader(f))
        A = -H
        with open(os.path.join(out_dir, "C3eq_zeta_dtc.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["r", "r_lo", "r_hi", "C3_dtc", "C3_zeta", "H", "t_ref"])
            for r in c3_rows:
                C = float(r["C"]) if r["C"] not in ("", "nan") else float("nan")
                w.writerow([
                    r["r"], r["r_lo"], r["r_hi"], C,
                    (A ** 3) * C if np.isfinite(C) else "nan",
                    H, t_ref,
                ])

    meta = {
        "channel": "delta_t_c",
        "formula_shell": "P_zeta_shell = H^2 * P_dtc  (CL shell-bin; not CMB convention)",
        "formula_dimless": (
            "Pdim_zeta = k^3/(2 pi^2) * H^2 * P_raw   "
            "(dimensionless; Jinno / CMB A_s convention)"
        ),
        "H": H,
        "t_ref": float(t_ref),
        "aH": hub["aH"],
        "a": hub["a"],
        "dx_phys": dx_phys if np.isfinite(dx_phys) else None,
        "note": (
            "delta-N time-delay channel. Lattice modes are subhorizon "
            "(k_IR >> aH). Compare CMB A_s ≃ 2e-9 to Pdim_zeta, not P_zeta_shell. "
            "Jinno xi^3 is for Pdim; this box only reaches xi_min ~ 0.5 so the "
            "asymptote is barely resolved."
        ),
    }
    with open(os.path.join(out_dir, "P_zeta_dtc_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ok = np.isfinite(P_raw) & (P_raw > 0) & (n_modes >= 64)
        okz = np.isfinite(Pdim_zeta) & (Pdim_zeta > 0) & (n_modes >= 64)

        # Left: dimensionless 𝒫_δt and 𝒫_ζ vs k (program), with ξ top axis
        if okz.any():
            k_ok = k[okz]
            pdim_ok = Pdim_zeta[okz]
            i_pk = int(np.nanargmax(pdim_ok))
            k_star = float(k_ok[i_pk])
            axes[0].loglog(
                k_ok, pdim_ok, color="C3", lw=1.8,
                label=r"$\mathcal{P}_\zeta=\frac{k^3}{2\pi^2}H^2 P_{\rm raw}$",
            )
            axes[0].axhline(
                2.1e-9, color="0.4", ls=":", lw=1.0,
                label=r"CMB $A_s\simeq 2.1\times 10^{-9}$",
            )
            # ξ³ guide through mid-IR bins (if any ξ<1)
            xi = k_ok / k_star
            fit_m = (xi >= 0.5) & (xi <= 0.85)
            if int(np.count_nonzero(fit_m)) >= 3:
                amp = float(np.nanmedian(pdim_ok[fit_m] / np.clip(xi[fit_m] ** 3, 1e-30, None)))
                axes[0].loglog(
                    k_ok, amp * xi ** 3, color="C3", ls="--", lw=1.2,
                    label=rf"$\propto\xi^3$ guide ($k_*={k_star:.3f}$)",
                )
            axes[0].set_xlim(left=float(np.min(k_ok)))
            sec = axes[0].secondary_xaxis(
                "top",
                functions=(lambda kk, ks=k_star: kk / ks, lambda xx, ks=k_star: xx * ks),
            )
            sec.set_xscale("log")
            sec.set_xlabel(r"$\xi\equiv k/k_*$")
        axes[0].set_xlabel(r"$k$ (program, $dx=1$)")
        axes[0].set_ylabel(r"$\mathcal{P}_\zeta(k)$")
        axes[0].set_title(r"Dimensionless $\mathcal{P}_\zeta$ from $\delta t_c$")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=7)

        # Right: legacy CL shell P_ζ vs physical k
        if okz.any() and np.isfinite(k_phys).any():
            axes[1].loglog(
                k_phys[okz], Pz_shell[okz], color="C2", lw=1.5,
                label=r"$H^2 P_{\delta t}^{\rm shell}$ (CL bin)",
            )
            axes[1].loglog(
                k_phys[okz], Pdim_zeta[okz], color="C3", lw=1.8,
                label=r"$\mathcal{P}_\zeta$ (dimensionless)",
            )
            axes[1].axhline(2.1e-9, color="0.4", ls=":", lw=1.0)
            axes[1].set_xlabel(r"$k_{\rm comoving}$ [GeV]  ($=k_{\rm prog}/dx_{\rm phys}$)")
            axes[1].set_xlim(left=float(np.nanmin(k_phys[okz])))
        else:
            axes[1].loglog(k[ok], P[ok], color="C2", lw=1.5, label=r"$P_{\delta t}^{\rm shell}$")
            axes[1].set_xlabel(r"$k$ (program)")
        axes[1].set_ylabel(r"power")
        axes[1].set_title(
            rf"$\zeta=-H\delta t_c$ ($H={H:.3e}$, $t_{{\rm ref}}={t_ref:.0f}$)"
        )
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.suptitle(
            r"Curvature from $\delta t_c$: use $\mathcal{P}_\zeta$ (left) for CMB / Jinno",
            fontsize=10,
        )
        fig.tight_layout()
        png = os.path.join(out_dir, "P_zeta_dtc.png")
        fig.savefig(png, dpi=160)
        plt.close(fig)
        LOG.info("wrote %s", png)
    except ImportError:
        pass

    return meta


def _try_plot_cf(out_dir: str, curves: List[Dict[str, Any]]) -> None:
    if not curves:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in curves:
        t = c["meta"]["time"]
        ax.plot(c["C_F"]["r"], c["C_F"]["C"], lw=1.4, label=rf"$t={t:.0f}$, $\langle F\rangle={c['meta']['F_mean']:.3f}$")
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"$r$ (lattice units)")
    ax.set_ylabel(r"$C_F(r;t)$")
    ax.set_title("False-vacuum indicator correlator")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "C_F_summary.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", path)


def _try_plot_tc(out_dir: str, c2: Dict[str, np.ndarray], c3: np.ndarray, pk: Dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].plot(c2["r"], c2["C"], color="C0", lw=1.4)
    axes[0].axhline(0.0, color="k", lw=0.7, ls=":")
    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$C_2(r)$")
    axes[0].set_title(r"$\delta t_c$ two-point")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(c2["r"], c3, color="C1", lw=1.4)
    axes[1].axhline(0.0, color="k", lw=0.7, ls=":")
    axes[1].set_xlabel(r"$r$")
    axes[1].set_ylabel(r"$C_3^{\rm eq}(r)$")
    axes[1].set_title(r"equilateral 3-pt of $\delta t_c$")
    axes[1].grid(True, alpha=0.3)

    ok = np.isfinite(pk["P"]) & (pk["P"] > 0) & (pk["n_modes"] >= 64)
    axes[2].loglog(pk["k"][ok], pk["P"][ok], color="C2", lw=1.4)
    axes[2].set_xlabel(r"$k$")
    axes[2].set_ylabel(r"$P_{\delta t}(k)$")
    axes[2].set_title(r"power spectrum of $\delta t_c$")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "tc_correlators_summary.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", path)


# ---------------------------------------------------------------------------
# core analysis
# ---------------------------------------------------------------------------
def compute_C_F_for_row(
    run_dir: str,
    row: Dict[str, Any],
    *,
    monolith: str,
    time_index: Dict[float, str],
    esc_gev: float,
    downsample: int,
    n_r_bins: int,
    r_max: Optional[float],
) -> Dict[str, Any]:
    t = float(row["t"])
    f_star = float(row["fStar"])
    esc_prog = float(esc_gev) / max(f_star, 1e-30)
    h5_path, kind = resolve_snapshot_h5(run_dir, row, monolith_path=monolith)
    time_key = lookup_h5_key(t, time_index) if kind == "monolith" else None
    phi0, phi1 = _load_phi(h5_path, row, time_key)
    rho = rho_prog_from_phi(phi0, phi1)
    del phi0
    if phi1 is not None:
        del phi1
    rho = _downsample3(rho, downsample)
    F = false_vac_mask(rho, esc_prog)
    table = connected_2pt_radial(F, n_bins=n_r_bins, r_max=r_max)
    meta = {
        "step": int(float(row["step"])),
        "time": t,
        "temperature": float(row["T"]),
        "a": float(row["a"]),
        "fStar": f_star,
        "escape_phi_GeV": float(esc_gev),
        "escape_phi_prog": esc_prog,
        "downsample": int(downsample),
        "N": int(F.shape[0]),
        "F_mean": float(F.mean()),
        "rho_mean": float(rho.mean()),
        "rho_max": float(rho.max()),
    }
    return {"meta": meta, "C_F": table, "F": F}


def build_tc_map(
    run_dir: str,
    rows: Sequence[Dict[str, Any]],
    *,
    monolith: str,
    time_index: Dict[float, str],
    esc_gev: float,
    downsample: int,
    conversion: str = "escape",
    vev_frac: float = 0.1,
    interpolate: bool = True,
) -> Dict[str, Any]:
    """First-crossing map: scan snapshots in time order.

    Conversion = first time ρ_prog exceeds the mode threshold (see
    ``conversion_threshold_prog``). With ``interpolate=True``, t_c is the
    linear-in-ρ crossing between the last false and first true frames.
    """
    rows = sorted(rows, key=lambda r: float(r["t"]))
    if not rows:
        raise ValueError("no rows to build t_c")

    tc: Optional[np.ndarray] = None
    converted: Optional[np.ndarray] = None
    N = 0
    f_star0 = float(rows[0]["fStar"])
    thr0 = conversion_threshold_prog(
        mode=conversion, esc_gev=esc_gev, f_star=f_star0, vev_frac=vev_frac,
    )
    history: List[Dict[str, Any]] = []
    rho_prev: Optional[np.ndarray] = None
    t_prev: Optional[float] = None
    thr_last = thr0

    ts = [float(r["t"]) for r in rows]
    if len(ts) >= 2:
        med_dt = float(np.median(np.diff(ts)))
        LOG.info(
            "t_c scan: conversion=%s  thr_prog(first)=%.6e  interpolate=%s  "
            "n_snap=%d  Δt_med=%.3g",
            conversion, thr0, interpolate, len(rows), med_dt,
        )
        if med_dt > 5.0:
            LOG.warning(
                "snapshot spacing Δt_med=%.3g is coarse — t_c will be poorly "
                "resolved even with interpolation; prefer denser HDF5 dumps "
                "through the PT",
                med_dt,
            )
    else:
        LOG.info(
            "t_c scan: conversion=%s  thr_prog(first)=%.6e  interpolate=%s  n_snap=%d",
            conversion, thr0, interpolate, len(rows),
        )

    t0 = time.time()
    for i, row in enumerate(rows):
        t = float(row["t"])
        f_star = float(row["fStar"])
        thr = conversion_threshold_prog(
            mode=conversion, esc_gev=esc_gev, f_star=f_star, vev_frac=vev_frac,
        )
        thr_last = thr
        h5_path, kind = resolve_snapshot_h5(run_dir, row, monolith_path=monolith)
        time_key = lookup_h5_key(t, time_index) if kind == "monolith" else None
        phi0, phi1 = _load_phi(h5_path, row, time_key)
        rho = rho_prog_from_phi(phi0, phi1)
        del phi0
        if phi1 is not None:
            del phi1
        rho = _downsample3(rho, downsample)
        F = false_vac_mask(rho, thr)
        if tc is None:
            N = int(F.shape[0])
            tc = np.full((N, N, N), np.nan, dtype=np.float32)
            converted = np.zeros((N, N, N), dtype=bool)

        now_true = rho > np.float32(thr)
        if rho_prev is None:
            newly = (~converted) & now_true
            if newly.any():
                tc[newly] = np.float32(t)
                converted[newly] = True
        else:
            newly = (~converted) & (rho_prev <= np.float32(thr)) & now_true
            if newly.any():
                if interpolate:
                    tc[newly] = interpolate_crossing_times(
                        float(t_prev), t, rho_prev, rho, thr, newly,
                    )
                else:
                    tc[newly] = np.float32(t)
                converted[newly] = True

        n_new = int(newly.sum())
        frac_F = float(F.mean())
        frac_conv = float(converted.mean())
        history.append(
            {
                "t": t,
                "step": int(float(row["step"])),
                "T": float(row["T"]),
                "F_mean": frac_F,
                "converted_frac": frac_conv,
                "n_new": n_new,
                "thr_prog": float(thr),
            }
        )
        if (i + 1) % max(len(rows) // 10, 1) == 0 or i == len(rows) - 1:
            LOG.info(
                "  t_c scan %d/%d  t=%.1f  ⟨F⟩=%.4f  converted=%.4f  new=%d  (%.1fs)",
                i + 1, len(rows), t, frac_F, frac_conv, n_new, time.time() - t0,
            )
        rho_prev = rho
        t_prev = t
        if frac_conv >= 1.0 - 1e-12:
            LOG.info("  all sites converted by t=%.1f — stopping early", t)
            break

    assert tc is not None and converted is not None
    return {
        "tc": tc,
        "converted": converted,
        "N": N,
        "downsample": int(downsample),
        "escape_phi_GeV": float(esc_gev),
        "escape_phi_prog_last": float(thr_last),
        "threshold_prog_last": float(thr_last),
        "conversion": str(conversion),
        "vev_frac": float(vev_frac) if str(conversion) == "vev_frac" else float("nan"),
        "interpolate": bool(interpolate),
        "fStar_first": f_star0,
        "t_first": float(rows[0]["t"]),
        "t_last": float(history[-1]["t"]),
        "converted_frac": float(converted.mean()),
        "history": history,
    }



def fill_unconverted_tc(
    tc: np.ndarray,
    converted: np.ndarray,
    *,
    mode: str,
    t_last: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    out = tc.copy()
    miss = ~converted
    n_miss = int(miss.sum())
    info = {"n_unconverted": float(n_miss), "fill_mode": 0.0}
    if n_miss == 0:
        return out, info
    if mode == "last":
        out[miss] = np.float32(t_last)
    elif mode == "nanmean":
        mu = float(np.nanmean(tc))
        out[miss] = np.float32(mu)
    elif mode == "drop":
        # Keep NaNs; caller must not FFT without cleaning.
        pass
    else:
        raise ValueError(f"unknown fill mode {mode!r}")
    info["fill_value"] = float(np.nanmean(out[miss])) if n_miss and mode != "drop" else float("nan")
    return out, info


def correlators_from_tc(
    tc: np.ndarray,
    *,
    n_r_bins: int,
    n_k_bins: int,
    n_triangles: int,
    r_max: Optional[float],
    seed: int,
    do_k_bispectrum: bool = True,
) -> Dict[str, Any]:
    if not np.isfinite(tc).all():
        raise ValueError("t_c has NaNs — choose --fill-unconverted last|nanmean before correlators")
    c2 = connected_2pt_radial(tc, n_bins=n_r_bins, r_max=r_max)
    delta = (tc.astype(np.float64) - float(tc.mean())).astype(np.float32)
    pk = power_spectrum_shells(delta, n_bins=n_k_bins)
    LOG.info("sampling C_3^eq with %d triangles per r-bin ...", n_triangles)
    t0 = time.time()
    c3 = c3_eq_monte_carlo(delta.astype(np.float64), c2["r"], n_samp=n_triangles, seed=seed)
    LOG.info("  C_3^eq done in %.1fs", time.time() - t0)

    eq_k: Optional[Dict[str, np.ndarray]] = None
    if do_k_bispectrum:
        from tools.field_bispectrum import analyze_correlators

        LOG.info("shell-filter B_eq / Q_eq of δt_c (n_bins=%d) ...", n_k_bins)
        t1 = time.time()
        eq_k, _ = analyze_correlators(
            delta, n_bins=n_k_bins, do_squeezed=False, n_workers=None,
        )
        LOG.info("  k-bispectrum done in %.1fs", time.time() - t1)

    return {
        "C2": c2,
        "C3_eq": c3,
        "P": pk,
        "eq_k": eq_k,
        "delta_var": float(np.mean(delta.astype(np.float64) ** 2)),
    }


def write_bispectrum_dtc_csv(path: str, eq: Dict[str, np.ndarray]) -> None:
    keys = ["k", "k_lo", "k_hi", "P", "P_filt", "P_raw", "B_eq", "Q_eq", "skew", "n_modes"]
    n = len(eq["k"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            row = []
            for key in keys:
                v = eq[key][i]
                row.append(f"{v:.10e}" if np.isfinite(v) else "nan")
            w.writerow(row)


def write_zeta_bispectrum_from_dtc(
    out_dir: str,
    *,
    H: float,
    t_ref: float,
    bispec_path: Optional[str] = None,
) -> Optional[str]:
    """B_ζ = (−H)³ B_eq, Q_ζ = B_ζ/(3 P_ζ²) = Q_eq / (−H) for ζ=−H δt."""
    path = bispec_path or os.path.join(out_dir, "bispectrum_dtc.csv")
    if not os.path.isfile(path):
        return None
    A = -float(H)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = os.path.join(out_dir, "bispectrum_zeta_dtc.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "k_lo", "k_hi", "P_dtc", "P_zeta", "B_eq", "B_zeta",
            "Q_eq", "Q_zeta", "skew", "n_modes", "H", "t_ref",
        ])
        for r in rows:
            def _f(key: str) -> float:
                v = r.get(key, "nan")
                try:
                    return float(v) if v not in ("", "nan", None) else float("nan")
                except ValueError:
                    return float("nan")

            P = _f("P")
            B = _f("B_eq")
            Q = _f("Q_eq")
            Pz = (A * A) * P if np.isfinite(P) else float("nan")
            Bz = (A ** 3) * B if np.isfinite(B) else float("nan")
            # Q_ζ = B_ζ / (3 P_ζ²) = Q / A
            Qz = Q / A if np.isfinite(Q) and A != 0 else float("nan")
            w.writerow([
                r["k"], r["k_lo"], r["k_hi"],
                P if np.isfinite(P) else "nan",
                Pz if np.isfinite(Pz) else "nan",
                B if np.isfinite(B) else "nan",
                Bz if np.isfinite(Bz) else "nan",
                Q if np.isfinite(Q) else "nan",
                Qz if np.isfinite(Qz) else "nan",
                r.get("skew", "nan"),
                r.get("n_modes", ""),
                H, t_ref,
            ])
    LOG.info("wrote %s", out)
    return out


def plot_Q_eq_dtc_vs_rho(
    run_dir: str,
    out_dir: str,
    *,
    rho_csv_dir: Optional[str] = None,
    rho_time: float = 400.0,
    min_modes: int = 500,
) -> Optional[str]:
    """Q_eq(k) of δt_c and rho_norm on one figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import re

    dtc_path = os.path.join(out_dir, "bispectrum_dtc.csv")
    if not os.path.isfile(dtc_path):
        LOG.warning("no bispectrum_dtc.csv — skip Q_eq compare plot")
        return None

    if rho_csv_dir is None:
        for cand in (
            os.path.join(run_dir, "string_new", "strings", "bispectrum_time_series_pt", "rho_norm"),
            os.path.join(run_dir, "strings", "bispectrum_time_series_pt", "rho_norm"),
            os.path.join(os.path.dirname(out_dir), "bispectrum_time_series_pt", "rho_norm"),
        ):
            if os.path.isdir(cand):
                rho_csv_dir = cand
                break
    if rho_csv_dir is None or not os.path.isdir(rho_csv_dir):
        LOG.warning("no rho_norm CSV dir — skip Q_eq compare")
        return None

    def _load_q(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(path) as f:
            rows = list(csv.DictReader(
                [ln for ln in f if ln.strip() and not ln.startswith("#")]
            ))
        k, Q, nm = [], [], []
        for r in rows:
            try:
                kk = float(r["k"])
                qq = float(r["Q_eq"])
                n = float(r.get("n_modes", 0))
            except (KeyError, ValueError):
                continue
            k.append(kk)
            Q.append(qq)
            nm.append(n)
        return np.asarray(k), np.asarray(Q), np.asarray(nm)

    with open(dtc_path) as f:
        # no comment header
        pass
    k_d, Q_d, nm_d = _load_q(dtc_path)

    best = None
    for name in sorted(os.listdir(rho_csv_dir)):
        if not (name.startswith("bispectrum_t") and name.endswith(".csv")):
            continue
        m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
        if not m:
            continue
        t = float(m.group(1))
        if best is None or abs(t - rho_time) < abs(best[0] - rho_time):
            best = (t, os.path.join(rho_csv_dir, name))
    if best is None:
        LOG.warning("no rho_norm bispectrum CSV")
        return None
    t_rho, rho_path = best
    k_r, Q_r, nm_r = _load_q(rho_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ok_d = np.isfinite(Q_d) & (nm_d >= min_modes)
    ok_r = np.isfinite(Q_r) & (nm_r >= min_modes)
    k_mins: List[float] = []
    if ok_d.any():
        ax.semilogx(k_d[ok_d], Q_d[ok_d], color="C3", lw=1.7,
                    label=r"$Q_{\rm eq}(\delta t_c)$")
        k_mins.append(float(np.min(k_d[ok_d])))
    if ok_r.any():
        ax.semilogx(k_r[ok_r], Q_r[ok_r], color="C0", lw=1.7,
                    label=rf"$Q_{{\rm eq}}(\rho_{{\rm norm}}),\ t={t_rho:.0f}$")
        k_mins.append(float(np.min(k_r[ok_r])))
    if k_mins:
        ax.set_xlim(left=min(k_mins))
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$Q_{\rm eq}(k)=B_{\rm eq}/(3P_{\rm filt}^2)$")
    ax.set_title(r"Reduced equilateral bispectrum: $\delta t_c$ vs $\rho_{\rm norm}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    # clip extreme noisy Q
    q_abs = []
    if ok_d.any():
        q_abs.append(np.nanpercentile(np.abs(Q_d[ok_d]), 98))
    if ok_r.any():
        q_abs.append(np.nanpercentile(np.abs(Q_r[ok_r]), 98))
    if q_abs:
        ymax = max(float(max(q_abs)), 1.0)
        ax.set_ylim(-ymax, ymax)
    fig.tight_layout()
    png = os.path.join(out_dir, "Q_eq_dtc_vs_rho_norm.png")
    fig.savefig(png, dpi=160)
    plt.close(fig)
    LOG.info("wrote %s", png)
    return png


def bispectrum_from_tc_map(
    run_dir: str,
    out_dir: str,
    *,
    tc_path: Optional[str] = None,
    n_k_bins: int = 64,
    t_ref_zeta: Optional[float] = None,
    fill_unconverted: str = "last",
    rho_csv_dir: Optional[str] = None,
    downsample: int = 2,
    n_workers: Optional[int] = None,
) -> str:
    """Load tc_map.npz → shell-filter B/Q of δt_c + ζ + Q compare plot.

    ``downsample`` (default 2) keeps every ds-th site for the FFT bispectrum
    so 1024³ fits in memory (512³).
    """
    out_dir = os.path.abspath(out_dir)
    tc_path = tc_path or os.path.join(out_dir, "tc_map.npz")
    z = np.load(tc_path, allow_pickle=True)
    tc = z["tc"]
    converted = z["converted"].astype(bool)
    t_last = float(z["t_last"])
    tc_filled, _ = fill_unconverted_tc(
        tc, converted, mode=fill_unconverted, t_last=t_last,
    )
    ds = max(int(downsample), 1)
    if ds > 1:
        tc_filled = np.ascontiguousarray(tc_filled[::ds, ::ds, ::ds])
        LOG.info("downsample=%d → N=%d for k-bispectrum", ds, tc_filled.shape[0])
    tc_mean = float(tc_filled.mean())
    delta = (tc_filled.astype(np.float64) - tc_mean).astype(np.float32)
    del tc_filled
    from tools.field_bispectrum import analyze_correlators

    LOG.info("computing B_eq/Q_eq of δt_c from %s (N=%d, n_bins=%d)",
             tc_path, delta.shape[0], n_k_bins)
    eq, _ = analyze_correlators(
        delta, n_bins=n_k_bins, do_squeezed=False, n_workers=n_workers,
    )
    del delta
    bsp = os.path.join(out_dir, "bispectrum_dtc.csv")
    write_bispectrum_dtc_csv(bsp, eq)
    LOG.info("wrote %s", bsp)

    if t_ref_zeta is None:
        t_ref_zeta = tc_mean
    hub = hubble_at_time(run_dir, float(t_ref_zeta))
    H = float(hub["H"])
    write_zeta_bispectrum_from_dtc(out_dir, H=H, t_ref=float(t_ref_zeta))

    # B_zeta plot (no aH)
    try:
        import matplotlib.pyplot as plt

        k = eq["k"]
        B = eq["B_eq"]
        nm = eq["n_modes"]
        A = -H
        Bz = (A ** 3) * B
        ok = np.isfinite(B) & (nm >= 64)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        if ok.any():
            axes[0].semilogx(k[ok], B[ok], color="C2", lw=1.5)
            axes[0].set_xlim(left=float(np.min(k[ok])))
            axes[1].semilogx(k[ok], Bz[ok], color="C3", lw=1.5)
            axes[1].set_xlim(left=float(np.min(k[ok])))
        axes[0].axhline(0.0, color="k", lw=0.7, ls=":")
        axes[1].axhline(0.0, color="k", lw=0.7, ls=":")
        axes[0].set_xlabel(r"$k$")
        axes[0].set_ylabel(r"$B_{\rm eq}(\delta t_c)$")
        axes[0].set_title(r"Equilateral $B$ of $\delta t_c$")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[1].set_xlabel(r"$k$")
        axes[1].set_ylabel(r"$B_\zeta=(-H)^3 B_{\rm eq}$")
        axes[1].set_title(rf"$H={H:.3e}$, $t_{{\rm ref}}={t_ref_zeta:.0f}$")
        axes[1].grid(True, which="both", alpha=0.3)
        fig.suptitle(r"Bispectrum: $\delta t_c\to\zeta$", fontsize=10)
        fig.tight_layout()
        png = os.path.join(out_dir, "B_zeta_dtc.png")
        fig.savefig(png, dpi=160)
        plt.close(fig)
        LOG.info("wrote %s", png)
    except ImportError:
        pass

    plot_Q_eq_dtc_vs_rho(
        run_dir, out_dir, rho_csv_dir=rho_csv_dir, rho_time=400.0,
    )
    return bsp


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(
    run_dir: str,
    *,
    cf_times: Optional[Sequence[float]] = None,
    build_tc: bool = False,
    tc_only: bool = False,
    cf_only: bool = False,
    stride: int = 1,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    escape_phi_gev: Optional[float] = None,
    conversion: str = "escape",
    vev_frac: float = 0.1,
    tc_interp: bool = True,
    downsample: int = 1,
    n_r_bins: int = 64,
    n_k_bins: int = 64,
    n_triangles: int = 200_000,
    r_max: Optional[float] = None,
    fill_unconverted: str = "last",
    save_tc: Optional[str] = None,
    load_tc: Optional[str] = None,
    seed: int = 0,
    write_zeta: bool = True,
    t_ref_zeta: Optional[float] = None,
    out_dir: Optional[str] = None,
) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "strings", "transition_correlators")
    os.makedirs(out_dir, exist_ok=True)

    params = load_run_params(run_dir) or {}
    esc_gev = resolve_escape_gev(params, escape_phi_gev)
    LOG.info(
        "escape_phi = %.6g GeV  conversion=%s  vev_frac=%s  tc_interp=%s",
        esc_gev, conversion, vev_frac, tc_interp,
    )

    do_cf = not tc_only
    do_tc = not cf_only
    if cf_times is None and not build_tc and load_tc is None and do_tc:
        build_tc = True

    rows_all = _load_manifest_any(run_dir)
    monolith = resolve_h5_path(run_dir, rows_all)
    time_index = build_h5_time_index(monolith, "phi_0")
    LOG.info("HDF5=%s  manifest rows=%d", monolith, len(rows_all))

    cf_results: List[Dict[str, Any]] = []
    if do_cf and cf_times:
        sel = nearest_rows(rows_all, cf_times)
        for row in sel:
            LOG.info("C_F at requested t~%s → snapshot t=%s step=%s",
                     cf_times, row["t"], row["step"])
            res = compute_C_F_for_row(
                run_dir, row,
                monolith=monolith,
                time_index=time_index,
                esc_gev=esc_gev,
                downsample=downsample,
                n_r_bins=n_r_bins,
                r_max=r_max,
            )
            tag = f"t{res['meta']['time']:07.1f}_step{res['meta']['step']:010d}"
            csv_path = os.path.join(out_dir, f"C_F_{tag}.csv")
            _write_radial_csv(
                csv_path,
                res["C_F"],
                extra_cols={
                    "var_dF": res["C_F"]["var"],
                    "F_mean": res["C_F"]["mean"],
                },
            )
            with open(os.path.join(out_dir, f"C_F_{tag}_meta.json"), "w") as f:
                json.dump(res["meta"], f, indent=2)
            LOG.info(
                "  wrote %s  ⟨F⟩=%.4f  C_F(0)~var=%.4e",
                csv_path, res["meta"]["F_mean"], float(res["C_F"]["var"][0]),
            )
            # drop bulky F before collecting
            cf_results.append({"meta": res["meta"], "C_F": res["C_F"]})
        _try_plot_cf(out_dir, cf_results)

    tc_payload: Optional[Dict[str, Any]] = None
    if do_tc and load_tc:
        LOG.info("loading t_c map from %s", load_tc)
        z = np.load(load_tc, allow_pickle=True)
        tc_payload = {
            "tc": z["tc"],
            "converted": z["converted"].astype(bool),
            "N": int(z["N"]),
            "downsample": int(z["downsample"]),
            "escape_phi_GeV": float(z["escape_phi_GeV"]),
            "t_first": float(z["t_first"]),
            "t_last": float(z["t_last"]),
            "converted_frac": float(z["converted_frac"]),
            "history": json.loads(str(z["history_json"])) if "history_json" in z.files else [],
        }

    if do_tc and tc_payload is None and (build_tc or not cf_times):
        scan_rows = filter_rows(rows_all, stride=stride, t_min=t_min, t_max=t_max)
        if not scan_rows:
            raise RuntimeError("no snapshots in t-range for t_c build")
        LOG.info(
            "building t_c from %d snapshots (t=%.1f … %.1f, stride=%d, ds=%d)",
            len(scan_rows),
            float(scan_rows[0]["t"]),
            float(scan_rows[-1]["t"]),
            stride,
            downsample,
        )
        tc_payload = build_tc_map(
            run_dir, scan_rows,
            monolith=monolith,
            time_index=time_index,
            esc_gev=esc_gev,
            downsample=downsample,
            conversion=conversion,
            vev_frac=vev_frac,
            interpolate=tc_interp,
        )

    if do_tc and tc_payload is not None:
        tc_path = save_tc or os.path.join(out_dir, "tc_map.npz")
        np.savez_compressed(
            tc_path,
            tc=tc_payload["tc"],
            converted=tc_payload["converted"],
            N=tc_payload["N"],
            downsample=tc_payload["downsample"],
            escape_phi_GeV=tc_payload["escape_phi_GeV"],
            threshold_prog_last=tc_payload.get(
                "threshold_prog_last", tc_payload.get("escape_phi_prog_last", float("nan"))
            ),
            conversion=str(tc_payload.get("conversion", "escape")),
            vev_frac=tc_payload.get("vev_frac", float("nan")),
            interpolate=bool(tc_payload.get("interpolate", False)),
            t_first=tc_payload["t_first"],
            t_last=tc_payload["t_last"],
            converted_frac=tc_payload["converted_frac"],
            history_json=json.dumps(tc_payload.get("history", [])),
        )
        LOG.info(
            "wrote %s  converted_frac=%.4f  N=%d",
            tc_path, tc_payload["converted_frac"], tc_payload["N"],
        )
        with open(os.path.join(out_dir, "tc_scan_history.csv"), "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["t", "step", "T", "F_mean", "converted_frac", "n_new", "thr_prog"]
            )
            w.writeheader()
            for h in tc_payload.get("history", []):
                w.writerow(h)

        tc_filled, fill_info = fill_unconverted_tc(
            tc_payload["tc"],
            tc_payload["converted"],
            mode=fill_unconverted,
            t_last=float(tc_payload["t_last"]),
        )
        LOG.info(
            "fill_unconverted=%s  n_miss=%d",
            fill_unconverted, int(fill_info["n_unconverted"]),
        )
        corr = correlators_from_tc(
            tc_filled,
            n_r_bins=n_r_bins,
            n_k_bins=n_k_bins,
            n_triangles=n_triangles,
            r_max=r_max,
            seed=seed,
        )
        _write_radial_csv(
            os.path.join(out_dir, "C2_dtc.csv"),
            corr["C2"],
            extra_cols={"var_dt": corr["C2"]["var"], "tc_mean": corr["C2"]["mean"]},
        )
        _write_radial_csv(
            os.path.join(out_dir, "C3eq_dtc.csv"),
            {
                "r": corr["C2"]["r"],
                "r_lo": corr["C2"]["r_lo"],
                "r_hi": corr["C2"]["r_hi"],
                "C": corr["C3_eq"],
                "n_pairs": np.full_like(corr["C2"]["r"], float(n_triangles)),
            },
        )
        _write_pk_csv(os.path.join(out_dir, "P_dtc.csv"), corr["P"])
        _try_plot_tc(out_dir, corr["C2"], corr["C3_eq"], corr["P"])
        if corr.get("eq_k") is not None:
            write_bispectrum_dtc_csv(
                os.path.join(out_dir, "bispectrum_dtc.csv"), corr["eq_k"]
            )

        if write_zeta:
            t_zeta = t_ref_zeta
            if t_zeta is None:
                t_zeta = float(corr["C2"]["mean"][0])
            try:
                zmeta = write_zeta_from_p_dtc(
                    run_dir, out_dir, t_ref=t_zeta,
                )
                write_zeta_bispectrum_from_dtc(
                    out_dir, H=float(zmeta["H"]), t_ref=float(t_zeta),
                )
                plot_Q_eq_dtc_vs_rho(run_dir, out_dir, rho_time=400.0)
            except FileNotFoundError as exc:
                LOG.warning("P_zeta from δt_c skipped: %s", exc)

        tc_meta = {
            "run_dir": run_dir,
            "escape_phi_GeV": esc_gev,
            "downsample": downsample,
            "fill_unconverted": fill_unconverted,
            "fill_info": fill_info,
            "converted_frac": tc_payload["converted_frac"],
            "t_first": tc_payload["t_first"],
            "t_last": tc_payload["t_last"],
            "N": tc_payload["N"],
            "delta_t_var": corr["delta_var"],
            "n_r_bins": n_r_bins,
            "n_k_bins": n_k_bins,
            "n_triangles": n_triangles,
            "tc_map": tc_path,
            "algorithm": {
                "F": "1[|Φ|_GeV <= φ_esc] = 1[ρ_prog <= φ_esc/fStar]",
                "C_F": "radial ⟨δF δF⟩ via FFT autocorrelation",
                "t_c": "interpolated ρ crossing of conversion threshold (or first true snapshot)",
                "C_2": "radial ⟨δt δt⟩ via FFT autocorrelation",
                "C_3^eq": "MC equilateral ⟨δt δt δt⟩ (mean-zero ⇒ connected)",
                "P_δt": "CL-style shell P = P_raw · n_modes / N³",
                "P_ζ": "H^2 P_δt with ζ = -H δt_c (time-delay / δN channel)",
            },
        }
        with open(os.path.join(out_dir, "tc_correlators_meta.json"), "w") as f:
            json.dump(tc_meta, f, indent=2, default=str)
        LOG.info("wrote correlators under %s", out_dir)

    top_meta = {
        "run_dir": run_dir,
        "escape_phi_GeV": esc_gev,
        "downsample": downsample,
        "cf_times": list(cf_times) if cf_times else None,
        "build_tc": build_tc or (do_tc and load_tc is None),
        "load_tc": load_tc,
        "out_dir": out_dir,
        "n_cf": len(cf_results),
    }
    with open(os.path.join(out_dir, "transition_correlators_meta.json"), "w") as f:
        json.dump(top_meta, f, indent=2, default=str)
    return out_dir


def _synthetic_selftest() -> None:
    rng = np.random.default_rng(1)
    N = 48
    # Smooth Gaussian random field → C_2(0)=var, C_3~0
    g = rng.standard_normal((N, N, N))
    gk = np.fft.fftn(g)
    k1d = np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    filt = np.exp(-0.5 * (k2 * (N / 6.0) ** 2))
    field = np.fft.ifftn(gk * filt).real.astype(np.float64)
    field -= field.mean()
    c2 = connected_2pt_radial(field, n_bins=20, r_max=N / 3)
    var = float(np.mean(field**2))
    if abs(c2["C"][0] - var) / var > 0.05:
        raise RuntimeError(f"C2(0)={c2['C'][0]} vs var={var}")
    c3 = c3_eq_monte_carlo(field, c2["r"], n_samp=80_000, seed=2)
    mid = (c2["r"] > 2) & (c2["r"] < N / 6) & np.isfinite(c3)
    skew_scale = float(np.nanmean(np.abs(c3[mid]))) / (var ** 1.5)
    if skew_scale > 0.15:
        raise RuntimeError(f"Gaussian C3 too large: {skew_scale}")

    # Binary F with known mean
    F = (field > 0).astype(np.float64)
    cf = connected_2pt_radial(F, n_bins=16, r_max=N / 3)
    p = float(F.mean())
    expect0 = p * (1 - p)
    if abs(cf["C"][0] - expect0) / expect0 > 0.05:
        raise RuntimeError(f"CF(0)={cf['C'][0]} vs p(1-p)={expect0}")
    # Interpolated crossing: ρ goes 0→1 between t=10 and t=20, thr=0.4 → t_c=14
    rho_a = np.zeros((4, 4, 4), dtype=np.float32)
    rho_b = np.ones((4, 4, 4), dtype=np.float32)
    newly = np.ones((4, 4, 4), dtype=bool)
    tc_x = interpolate_crossing_times(10.0, 20.0, rho_a, rho_b, 0.4, newly)
    if abs(float(tc_x.mean()) - 14.0) > 1e-5:
        raise RuntimeError(f"interp crossing expected 14, got {tc_x.mean()}")
    thr_e = conversion_threshold_prog(
        mode="escape", esc_gev=5e4, f_star=1e15, vev_frac=0.1,
    )
    thr_v = conversion_threshold_prog(
        mode="vev_frac", esc_gev=5e4, f_star=1e15, vev_frac=0.1,
    )
    if not (thr_e < 1e-9 and abs(thr_v - 0.1) < 1e-12):
        raise RuntimeError(f"threshold sanity failed: escape={thr_e} vev={thr_v}")
    print(
        f"selftest OK: C2(0)/var={c2['C'][0]/var:.4f}  "
        f"⟨|C3|⟩/σ³={skew_scale:.3e}  CF(0)/[p(1-p)]={cf['C'][0]/expect0:.4f}  "
        f"interp_tc={float(tc_x.mean()):.1f}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--cf-times", type=float, nargs="+", default=None,
                    help="program times for equal-time C_F(r;t)")
    ap.add_argument("--build-tc", action="store_true",
                    help="scan snapshots to build first-conversion map t_c(x)")
    ap.add_argument("--tc-only", action="store_true", help="skip C_F")
    ap.add_argument("--cf-only", action="store_true", help="skip t_c correlators")
    ap.add_argument("--stride", type=int, default=1, help="snapshot stride for t_c scan")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--escape-phi-gev", type=float, default=None,
                    help="φ_esc in GeV for conversion=escape "
                         "(default: run params expansion/langevin phi_esc)")
    ap.add_argument(
        "--conversion",
        choices=("escape", "vev_frac"),
        default="escape",
        help="false→true criterion for t_c: escape=CL φ_esc cut; "
             "vev_frac=ρ_prog > f·φ₀_prog (better bubble/true-vac proxy)",
    )
    ap.add_argument(
        "--vev-frac",
        type=float,
        default=0.1,
        help="fraction of program VEV for --conversion vev_frac (default 0.1)",
    )
    ap.add_argument(
        "--tc-interp",
        action="store_true",
        default=True,
        help="linearly interpolate t_c between snapshots (default on)",
    )
    ap.add_argument(
        "--no-tc-interp",
        action="store_true",
        help="disable interpolation: t_c = first true snapshot time",
    )
    ap.add_argument("--downsample", type=int, default=1,
                    help="keep every ds-th site (2 → 512³ from 1024³)")
    ap.add_argument("--n-r-bins", type=int, default=64)
    ap.add_argument("--n-k-bins", type=int, default=64)
    ap.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="parallel shell-filter workers for B_eq (default: auto; 1=serial)",
    )
    ap.add_argument("--n-triangles", type=int, default=200_000,
                    help="MC samples per r-bin for C_3^eq")
    ap.add_argument("--r-max", type=float, default=None,
                    help="max separation in lattice units (default N/2)")
    ap.add_argument(
        "--fill-unconverted",
        choices=("last", "nanmean", "drop"),
        default="last",
        help="how to treat sites still false at end of scan (default: last time)",
    )
    ap.add_argument("--save-tc", default=None, help="path for tc_map.npz")
    ap.add_argument("--load-tc", default=None, help="reuse existing tc_map.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--zeta-from-dtc",
        action="store_true",
        default=True,
        help="after P_dtc, write P_zeta = H^2 P_dtc (default on)",
    )
    ap.add_argument("--no-zeta-from-dtc", action="store_true")
    ap.add_argument(
        "--t-ref-zeta",
        type=float,
        default=None,
        help="time for H(t) in ζ=-H δt_c (default: ⟨t_c⟩)",
    )
    ap.add_argument(
        "--apply-zeta-from-dtc",
        default=None,
        metavar="OUT_DIR",
        help="only convert existing OUT_DIR/P_dtc.csv → P_zeta (no HDF5 scan)",
    )
    ap.add_argument(
        "--apply-bispectrum-from-tc",
        default=None,
        metavar="OUT_DIR",
        help="from OUT_DIR/tc_map.npz compute B_eq/Q_eq of δt_c, B_zeta, "
             "and Q_eq vs rho_norm plot",
    )
    ap.add_argument(
        "--rho-csv-dir",
        default=None,
        help="rho_norm bispectrum dir for Q_eq compare "
             "(default: …/bispectrum_time_series_pt/rho_norm)",
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
        return 0
    if not args.run_dir:
        ap.error("run_dir required unless --selftest")

    if args.apply_zeta_from_dtc:
        zmeta = write_zeta_from_p_dtc(
            args.run_dir,
            args.apply_zeta_from_dtc,
            t_ref=args.t_ref_zeta,
        )
        write_zeta_bispectrum_from_dtc(
            args.apply_zeta_from_dtc,
            H=float(zmeta["H"]),
            t_ref=float(zmeta["t_ref"]),
        )
        return 0

    if args.apply_bispectrum_from_tc:
        bispectrum_from_tc_map(
            args.run_dir,
            args.apply_bispectrum_from_tc,
            n_k_bins=args.n_k_bins,
            t_ref_zeta=args.t_ref_zeta,
            rho_csv_dir=args.rho_csv_dir,
            downsample=args.downsample if args.downsample != 1 else 2,
            n_workers=args.n_workers,
        )
        return 0

    if args.tc_only and args.cf_only:
        ap.error("cannot set both --tc-only and --cf-only")

    run(
        args.run_dir,
        cf_times=args.cf_times,
        build_tc=args.build_tc,
        tc_only=args.tc_only,
        cf_only=args.cf_only,
        stride=args.stride,
        t_min=args.t_min,
        t_max=args.t_max,
        escape_phi_gev=args.escape_phi_gev,
        conversion=args.conversion,
        vev_frac=args.vev_frac,
        tc_interp=bool(args.tc_interp) and not bool(args.no_tc_interp),
        downsample=args.downsample,
        n_r_bins=args.n_r_bins,
        n_k_bins=args.n_k_bins,
        n_triangles=args.n_triangles,
        r_max=args.r_max,
        fill_unconverted=args.fill_unconverted,
        save_tc=args.save_tc,
        load_tc=args.load_tc,
        seed=args.seed,
        write_zeta=bool(args.zeta_from_dtc) and not args.no_zeta_from_dtc,
        t_ref_zeta=args.t_ref_zeta,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
