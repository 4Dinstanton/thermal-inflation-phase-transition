#!/usr/bin/env python3
"""Equilateral (and squeezed) bispectrum from CosmoLattice HDF5 field snapshots.

Computes the equal-time three-point correlator in Fourier space from φ snapshots
**after** the run — no CosmoLattice re-implementation needed.

Definitions
-----------
With δ̃(k) = FFT[δ(x)] (complex, unitary-ish normalisation documented below),

    P(k)  = ⟨|δ̃(k)|²⟩_{|k|∈shell}          (already in spectra_scalar_*.txt)
    B_eq(k) ≈ ⟨ [δ_k(x)]³ ⟩                   (shell-filtered real-space cubic)
    Q_eq(k) = B_eq(k) / [P(k)]³               (dimensionless reduced bispectrum)

where δ_k(x) is the inverse FFT of δ̃ restricted to a thin shell around k
(Scoccimarro / Jeong-style filtered-field estimator). For a Gaussian field
B → 0 and Q → 0; bubble walls / collisions source Q ≠ 0.

Field choices (--field)
-----------------------
    rho     |Φ| − ⟨|Φ|⟩          (bubble/domain contrast; default)
    phi1    φ₁ − ⟨φ₁⟩
    phi2    φ₂ − ⟨φ₂⟩
    complex Φ = φ₁ + i φ₂ then use Re(Φ e^{-iα}) of mean-subtracted |Φ| phase
                stripped amplitude: |Φ|−⟨|Φ|⟩ is usually clearer for PT

Usage
-----
    # selected epochs
    python tools/field_bispectrum.py <run_dir> --times 400 470 520 581

    # every snapshot (writes one CSV each; summary becomes a heatmap)
    python tools/field_bispectrum.py <run_dir> --all-times

    # every 10th snapshot in the PT window, downsampled
    python tools/field_bispectrum.py <run_dir> --all-times --stride 10 \
        --t-min 300 --t-max 640 --downsample 2 --n-bins 24

Outputs (under ``<run_dir>/strings/bispectrum/`` or ``--out-dir``)
    bispectrum_tXXXX.csv   columns: k, P, B_eq, Q_eq, n_modes, ...
    bispectrum_summary.png
    bispectrum_meta.json
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

LOG = logging.getLogger("field_bispectrum")


# ---------------------------------------------------------------------------
# I/O helpers
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
    out = []
    used = set()
    for t in times:
        i = int(np.argmin(np.abs(ts - float(t))))
        if i in used:
            continue
        used.add(i)
        out.append(rows[i])
    return out


def load_delta_field(
    h5_path: str,
    row: Dict[str, Any],
    time_key: Optional[str],
    *,
    field: str = "rho",
    downsample: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return mean-subtracted real field δ(x) as float32, plus meta."""
    f_star = float(row["fStar"])
    n_scalars = int(float(row["n_scalars"]))
    phi0 = np.asarray(read_h5_field(h5_path, "phi_0", time_key), dtype=np.float32)
    N = phi0.shape[0]
    if n_scalars >= 2:
        phi1 = phi0 * np.float32(f_star)
        del phi0
        phi2 = np.asarray(read_h5_field(h5_path, "phi_1", time_key, N=N), dtype=np.float32)
        phi2 *= np.float32(f_star)
    else:
        phi1 = phi0 * np.float32(f_star)
        del phi0
        phi2 = None

    if field == "phi1":
        delta = phi1
    elif field == "phi2":
        if phi2 is None:
            raise ValueError("phi2 requested but n_scalars < 2")
        delta = phi2
    elif field == "rho":
        if phi2 is None:
            delta = np.abs(phi1)
        else:
            delta = np.sqrt(phi1 * phi1 + phi2 * phi2, dtype=np.float32)
    else:
        raise ValueError(f"unknown field={field!r}; use rho|phi1|phi2")

    del phi1
    if phi2 is not None:
        del phi2

    ds = max(int(downsample), 1)
    if ds > 1:
        delta = np.ascontiguousarray(delta[::ds, ::ds, ::ds])
    delta = delta - np.float32(delta.mean())
    meta = {
        "N": int(delta.shape[0]),
        "N_full": int(N),
        "downsample": ds,
        "field": field,
        "step": int(float(row["step"])),
        "time": float(row["t"]),
        "temperature": float(row["T"]),
        "a": float(row["a"]),
    }
    return delta, meta


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def _k_grid(N: int) -> np.ndarray:
    """|k| in program units with dx_prog = 1 (k_fund = 2π/N, CosmoLattice-like)."""
    kx = np.fft.fftfreq(N).astype(np.float32) * np.float32(2.0 * math.pi)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij", sparse=True)
    return np.sqrt(KX * KX + KY * KY + KZ * KZ)


def fft_field(delta: np.ndarray) -> np.ndarray:
    """Forward FFT; return δ̃ with ⟨|δ̃|²⟩ = Var(δ) * N³ (numpy unnormalised)."""
    return np.fft.fftn(delta.astype(np.complex64, copy=False))


def shell_edges(k_nyq: float, n_bins: int, k_min: Optional[float] = None) -> np.ndarray:
    k0 = k_min if k_min is not None else (2.0 * math.pi / 64.0)  # avoid DC bin
    # log-spaced edges up to ~0.9 k_Nyquist
    return np.logspace(math.log10(max(k0, 1e-4)), math.log10(0.9 * k_nyq), n_bins + 1)


def power_and_equilateral_bispectrum(
    delta: np.ndarray,
    *,
    n_bins: int = 32,
    k_min: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Shell-filtered equilateral bispectrum + power spectrum.

    Cost: one forward FFT + ``n_bins`` inverse FFTs of shell-masked transforms.
    Peak RAM ≈ a few × N³ complex64 buffers.
    """
    N = delta.shape[0]
    n3 = float(N) ** 3
    t0 = time.time()
    delta_k = fft_field(delta)
    LOG.info("  FFT done in %.1fs  (N=%d)", time.time() - t0, N)

    kmag = _k_grid(N)
    k_nyq = math.pi  # with k = 2π n/N, max ~ π√3; use π as 1-D Nyquist
    edges = shell_edges(k_nyq, n_bins, k_min=k_min)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Power: bin |δ̃|² / N³  so that Σ_k P ≈ Var(δ)  (Parseval for numpy fft)
    power = (delta_k.real ** 2 + delta_k.imag ** 2).astype(np.float64) / n3
    # leave delta_k for shell masking; work on a copy of amplitudes only when needed

    P = np.zeros(n_bins, dtype=np.float64)
    B = np.zeros(n_bins, dtype=np.float64)
    n_modes = np.zeros(n_bins, dtype=np.int64)
    # exclude DC
    power.flat[0] = 0.0
    delta_k.flat[0] = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (kmag >= lo) & (kmag < hi)
        n_m = int(mask.sum())
        n_modes[i] = n_m
        if n_m < 8:
            P[i] = np.nan
            B[i] = np.nan
            continue
        P[i] = float(power[mask].mean())

        # shell-filtered field
        shell = np.zeros_like(delta_k)
        shell[mask] = delta_k[mask]
        real = np.fft.ifftn(shell).real.astype(np.float64)
        del shell
        # ⟨δ_k³⟩ ; scale so B has consistent units with P³ under numpy FFT
        # δ_k(x) = IFFT(masked δ̃) → mean(δ_k³) * N³  tracks the bispectrum
        # convention: report B such that Q = B / P³ is dimensionless with this P
        B[i] = float(np.mean(real ** 3)) * n3
        del real
        LOG.info(
            "  bin %2d/%d  k=%.4f  n=%d  P=%.3e  B=%.3e",
            i + 1, n_bins, centers[i], n_m, P[i], B[i],
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        Q = B / (P ** 3)
    return {
        "k": centers,
        "k_lo": edges[:-1],
        "k_hi": edges[1:],
        "P": P,
        "B_eq": B,
        "Q_eq": Q,
        "n_modes": n_modes.astype(float),
    }


def squeezed_bispectrum_proxy(
    delta: np.ndarray,
    *,
    k_soft_max: float = 0.05,
    n_hard_bins: int = 16,
) -> Dict[str, np.ndarray]:
    """Cheap squeezed proxy: ⟨δ_soft²(x) · δ_hard(x)⟩ vs k_hard.

    Not a full B(k_s, k_h, k_h), but a useful non-Gaussian diagnostic that
    bubble walls modulate small-scale power.
    """
    N = delta.shape[0]
    n3 = float(N) ** 3
    delta_k = fft_field(delta)
    delta_k.flat[0] = 0.0
    kmag = _k_grid(N)

    soft_mask = (kmag > 0) & (kmag < k_soft_max)
    soft = np.zeros_like(delta_k)
    soft[soft_mask] = delta_k[soft_mask]
    soft_x = np.fft.ifftn(soft).real.astype(np.float64)
    del soft
    soft2 = soft_x * soft_x
    del soft_x

    edges = shell_edges(math.pi, n_hard_bins, k_min=k_soft_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Bsq = np.zeros(n_hard_bins, dtype=np.float64)
    P_hard = np.zeros(n_hard_bins, dtype=np.float64)
    n_modes = np.zeros(n_hard_bins, dtype=np.int64)
    power = (delta_k.real ** 2 + delta_k.imag ** 2).astype(np.float64) / n3

    for i in range(n_hard_bins):
        mask = (kmag >= edges[i]) & (kmag < edges[i + 1])
        n_modes[i] = int(mask.sum())
        if n_modes[i] < 8:
            Bsq[i] = np.nan
            P_hard[i] = np.nan
            continue
        P_hard[i] = float(power[mask].mean())
        hard = np.zeros_like(delta_k)
        hard[mask] = delta_k[mask]
        hard_x = np.fft.ifftn(hard).real.astype(np.float64)
        del hard
        Bsq[i] = float(np.mean(soft2 * hard_x)) * n3
        del hard_x

    return {
        "k_hard": centers,
        "B_squeezed_proxy": Bsq,
        "P_hard": P_hard,
        "n_modes_hard": n_modes.astype(float),
        "k_soft_max": np.asarray([k_soft_max]),
    }


# ---------------------------------------------------------------------------
# driver + plot
# ---------------------------------------------------------------------------
CSV_FIELDS = (
    "k", "k_lo", "k_hi", "P", "B_eq", "Q_eq", "n_modes",
    "k_hard", "B_squeezed_proxy", "P_hard",
)


def write_csv(path: str, eq: Dict[str, np.ndarray], sq: Optional[Dict[str, np.ndarray]]) -> None:
    n = len(eq["k"])
    n2 = len(sq["k_hard"]) if sq is not None else 0
    n_out = max(n, n2)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for i in range(n_out):
            row = {k: "" for k in CSV_FIELDS}
            if i < n:
                for key in ("k", "k_lo", "k_hi", "P", "B_eq", "Q_eq", "n_modes"):
                    row[key] = f"{eq[key][i]:.10e}"
            if sq is not None and i < n2:
                row["k_hard"] = f"{sq['k_hard'][i]:.10e}"
                row["B_squeezed_proxy"] = f"{sq['B_squeezed_proxy'][i]:.10e}"
                row["P_hard"] = f"{sq['P_hard'][i]:.10e}"
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

    # Many snapshots → heatmap of Q(k,t) + time series of mean |Q|
    if len(results) > max_overlay:
        ks = results[0]["eq"]["k"]
        ts = np.array([r["meta"]["time"] for r in results], dtype=float)
        Q = np.vstack([r["eq"]["Q_eq"] for r in results])
        P = np.vstack([r["eq"]["P"] for r in results])

        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)

        # P(k,t)
        Pp = np.ma.masked_invalid(np.log10(np.clip(P, 1e-40, None)))
        im0 = axes[0].pcolormesh(ks, ts, Pp, shading="auto", cmap="viridis")
        axes[0].set_xscale("log")
        axes[0].set_xlabel(r"$k$")
        axes[0].set_ylabel(r"$t$")
        axes[0].set_title(r"$\log_{10} P(k,t)$")
        fig.colorbar(im0, ax=axes[0], shrink=0.85)

        # Q(k,t)
        Qclip = np.ma.masked_invalid(Q)
        vmax = float(np.nanpercentile(np.abs(Q), 95)) if np.isfinite(Q).any() else 1.0
        vmax = max(vmax, 1e-12)
        im1 = axes[1].pcolormesh(
            ks, ts, Qclip, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel(r"$k$")
        axes[1].set_ylabel(r"$t$")
        axes[1].set_title(r"$Q_{\rm eq}(k,t)$")
        fig.colorbar(im1, ax=axes[1], shrink=0.85)

        # mean |Q| vs t
        qmean = np.nanmean(np.abs(Q), axis=1)
        axes[2].plot(ts, qmean, "C0-", lw=1.4)
        axes[2].set_xlabel(r"$t$")
        axes[2].set_ylabel(r"$\langle|Q_{\rm eq}|\rangle_k$")
        axes[2].set_title("Non-Gaussianity vs time")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Field correlators (all snapshots, n={len(results)})  |  "
            + results[0]["meta"]["field"],
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
    axes[0].set_title("Power spectrum (from same FFT)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$k$ (program)")
    axes[1].set_ylabel(r"$Q_{\rm eq}=B_{\rm eq}/P^3$")
    axes[1].set_title("Reduced equilateral bispectrum")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)

    axes[2].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel(r"$k_{\rm hard}$ (program)")
    axes[2].set_ylabel(r"squeezed proxy")
    axes[2].set_title(r"Squeezed proxy $\langle\delta_{\rm soft}^2\delta_{\rm hard}\rangle$")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=7)

    fig.suptitle(
        "Field correlators from HDF5  |  "
        + (results[0]["meta"]["field"] if results else ""),
        fontsize=10,
    )
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
    field: str = "rho",
    downsample: int = 1,
    n_bins: int = 32,
    do_squeezed: bool = True,
    out_dir: Optional[str] = None,
) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "strings", "bispectrum")
    os.makedirs(out_dir, exist_ok=True)

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
            "all-times: %d snapshots (stride=%d, window t∈[%s,%s])",
            len(selected),
            stride,
            f"{float(selected[0]['t']):.1f}" if selected else "—",
            f"{float(selected[-1]['t']):.1f}" if selected else "—",
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
        LOG.info("[%d/%d] === step %d  t=%.3f ===", i_row, len(selected), step, t)
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

        delta, meta = load_delta_field(
            h5_path, row, tkey, field=field, downsample=downsample
        )
        LOG.info(
            "  field=%s  N=%d (full %d)  var=%.4e",
            field, meta["N"], meta["N_full"], float(delta.var()),
        )
        eq = power_and_equilateral_bispectrum(delta, n_bins=n_bins)
        sq = squeezed_bispectrum_proxy(delta) if do_squeezed else None
        del delta

        csv_path = os.path.join(out_dir, f"bispectrum_t{t:07.1f}_step{step:010d}.csv")
        write_csv(csv_path, eq, sq)
        LOG.info("  wrote %s", csv_path)
        results.append({"meta": meta, "eq": eq, "sq": sq, "csv": csv_path})

    if not results:
        raise RuntimeError("no snapshots processed (HDF5 missing?)")

    png = os.path.join(out_dir, "bispectrum_summary.png")
    plot_summary(results, png)
    LOG.info("wrote %s", png)

    meta_path = os.path.join(out_dir, "bispectrum_meta.json")
    summary = {
        "run_dir": run_dir,
        "field": field,
        "downsample": downsample,
        "n_bins": n_bins,
        "all_times": all_times,
        "stride": stride,
        "t_min": t_min,
        "t_max": t_max,
        "n_processed": len(results),
        "n_skipped_missing_h5": skipped,
        "snapshots": [r["meta"] for r in results],
        "csvs": [r["csv"] for r in results],
        "png": png,
        "notes": [
            "B_eq from shell-filtered ⟨δ_k(x)³⟩ estimator (equilateral).",
            "Q_eq = B_eq / P³; Gaussian ⇒ Q≈0.",
            "Squeezed proxy is ⟨δ_soft² δ_hard⟩, not the full B(k_s,k_h,k_h).",
            "k uses program units with dx_prog=1 (k_fund=2π/N_grid_used).",
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_dir


def _synthetic_selftest() -> None:
    """Quick non-Gaussian sanity check on a small grid."""
    rng = np.random.default_rng(0)
    N = 64
    g = rng.standard_normal((N, N, N)).astype(np.float32)
    # local chi-squared-like non-Gaussianity
    ng = (g * g - 1.0).astype(np.float32)
    ng -= ng.mean()
    eq_g = power_and_equilateral_bispectrum(g - g.mean(), n_bins=12)
    eq_n = power_and_equilateral_bispectrum(ng, n_bins=12)
    q_g = np.nanmean(np.abs(eq_g["Q_eq"]))
    q_n = np.nanmean(np.abs(eq_n["Q_eq"]))
    print(f"selftest: ⟨|Q|⟩_gauss={q_g:.3e}  ⟨|Q|⟩_NG={q_n:.3e}")
    if not (q_n > 3 * q_g):
        raise RuntimeError("bispectrum selftest failed: NG should exceed Gaussian")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=None,
        help="Program times (nearest snapshot). Ignored if --all-times.",
    )
    ap.add_argument(
        "--all-times",
        action="store_true",
        help="Process every manifest snapshot (optionally strided / windowed)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="With --all-times, keep every N-th snapshot (default 1 = all)",
    )
    ap.add_argument("--t-min", type=float, default=None, help="Only t >= this")
    ap.add_argument("--t-max", type=float, default=None, help="Only t <= this")
    ap.add_argument("--field", choices=("rho", "phi1", "phi2"), default="rho")
    ap.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every d-th point (d=2 → 512³ from 1024³)",
    )
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--no-squeezed", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true", help="Run synthetic NG check and exit")
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
        downsample=args.downsample,
        n_bins=args.n_bins,
        do_squeezed=not args.no_squeezed,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
