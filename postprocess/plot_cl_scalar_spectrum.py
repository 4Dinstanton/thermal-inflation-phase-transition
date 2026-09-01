#!/usr/bin/env python3
"""Plot CosmoLattice scalar field power spectra P(k) from spectra_scalar_*.txt.

These are the equal-time Fourier correlators
    P_i(k) ∝ ⟨|φ̃_i(k)|²⟩
written by CosmoLattice (not the GW spectrum).

Usage
-----
    python postprocess/plot_cl_scalar_spectrum.py <run_dir>
    python postprocess/plot_cl_scalar_spectrum.py <run_dir> --times 400 470 581
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_blocks(path: str) -> List[np.ndarray]:
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


def _load_times(run_dir: str, n_blocks: int) -> np.ndarray:
    for name in ("average_spectra_times.txt", "spectra_times.txt"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            t = np.loadtxt(p, ndmin=1).astype(float)
            if t.size == n_blocks:
                return t
            if t.size > n_blocks:
                return t[:n_blocks]
    return np.arange(n_blocks, dtype=float)


def _load_markers(run_dir: str) -> Dict[str, Optional[float]]:
    out = {"tipt_start": None, "langoff": None, "tc1": None}
    for rel in (
        "strings/transition_markers.json",
        "string_new/strings/transition_markers.json",
    ):
        p = os.path.join(run_dir, rel)
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            raw = json.load(f)
        for k in out:
            e = raw.get(k) or {}
            if e.get("t") is not None:
                out[k] = float(e["t"])
        break
    return out


def pick_indices(
    times: np.ndarray,
    *,
    targets: Optional[Sequence[float]] = None,
    markers: Optional[Dict[str, Optional[float]]] = None,
) -> List[Tuple[int, str]]:
    if targets:
        return [
            (int(np.argmin(np.abs(times - float(t)))), f"t={float(t):.0f}")
            for t in targets
        ]
    picks: List[Tuple[float, str]] = []
    if markers:
        for key, lab in (
            ("tipt_start", "TIPT start"),
            ("langoff", "Langevin off"),
            ("tc1", r"$T_{c_1}$"),
        ):
            if markers.get(key) is not None:
                picks.append((float(markers[key]), lab))
    # always include a mid-PT and late point
    picks.append((float(np.median(times[times > 0])) if np.any(times > 0) else 0.0, "mid"))
    picks.append((float(times[-1]), "last"))
    # GW-ish peak epoch often ~580 for this setup
    picks.append((581.0, "GW peak~581"))

    # unique nearest indices
    seen = set()
    out: List[Tuple[int, str]] = []
    for t, lab in picks:
        i = int(np.argmin(np.abs(times - t)))
        if i in seen:
            continue
        seen.add(i)
        out.append((i, f"{lab} (t={times[i]:.0f})"))
    out.sort(key=lambda x: x[0])
    return out


def plot_spectra(
    run_dir: str,
    *,
    times_req: Optional[Sequence[float]] = None,
    out_path: Optional[str] = None,
    combine: bool = True,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = os.path.abspath(run_dir)
    path0 = os.path.join(run_dir, "spectra_scalar_0.txt")
    path1 = os.path.join(run_dir, "spectra_scalar_1.txt")
    if not os.path.isfile(path0):
        raise FileNotFoundError(path0)

    blocks0 = _load_blocks(path0)
    blocks1 = _load_blocks(path1) if os.path.isfile(path1) else None
    times = _load_times(run_dir, len(blocks0))
    markers = _load_markers(run_dir)
    picks = pick_indices(times, targets=times_req, markers=markers)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.15, 0.9, len(picks))]

    for ax, weight, title in (
        (axes[0], lambda k, p: p, r"$P(k)$"),
        (axes[1], lambda k, p: k * k * p, r"$k^2 P(k)$"),
    ):
        for (i, lab), c in zip(picks, colors):
            b = blocks0[i]
            k, p0 = b[:, 0], b[:, 1]
            if combine and blocks1 is not None and i < len(blocks1):
                p = p0 + blocks1[i][:, 1]
                tag = r"$P_0+P_1$"
            else:
                p = p0
                tag = r"$P_0$"
            y = weight(k, p)
            m = (k > 0) & (y > 0) & np.isfinite(y)
            ax.loglog(k[m], y[m], color=c, lw=1.5, label=lab)
        ax.set_xlabel(r"$k$ (program)")
        ax.set_ylabel(title)
        ax.set_title(title + (f"  ({tag})" if picks else ""))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    short = os.path.basename(run_dir)
    if len(short) > 70:
        short = short[:30] + "…" + short[-35:]
    fig.suptitle(
        f"Scalar correlator spectrum  |  {short}",
        fontsize=10,
    )

    out_dir = os.path.join(run_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_path or os.path.join(out_dir, "scalar_power_spectrum.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=None,
        help="Program times to plot (nearest snapshot). Default: markers + late.",
    )
    ap.add_argument(
        "--no-combine",
        action="store_true",
        help="Plot only spectra_scalar_0 (default: P0+P1)",
    )
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)
    path = plot_spectra(
        args.run_dir,
        times_req=args.times,
        out_path=args.out,
        combine=not args.no_combine,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
