#!/usr/bin/env python3
"""Plot CosmoLattice on-lattice GW spectra from spectra_gws.txt / energy_gws.txt.

CosmoLattice writes GW power spectra on infrequent measurement steps to:
  <run_dir>/spectra_gws.txt   — binned P_GW(k) vs program wavenumber
  <run_dir>/energy_gws.txt    — t, spectral energy, energy * rho

Usage
-----
    python postprocess/plot_cl_gw_spectrum.py <run_dir>
    python postprocess/plot_cl_gw_spectrum.py <run_dir> --out figs/gw_spectrum.png
"""
import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_run_params(run_dir):
    path = os.path.join(run_dir, "cl_run_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def parse_spectra_gws(path):
    """Parse spectra_gws.txt into list of (k_centers, values) blocks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GW spectrum file not found: {path}")

    blocks = []
    current_k = []
    current_v = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_k:
                    blocks.append((np.array(current_k), np.array(current_v)))
                    current_k, current_v = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                k = float(parts[0])
                val = float(parts[1])
            except ValueError:
                continue
            current_k.append(k)
            current_v.append(val)

    if current_k:
        blocks.append((np.array(current_k), np.array(current_v)))
    return blocks


def parse_energy_gws(path):
    """Parse energy_gws.txt -> arrays (t, E_spec, E_rho)."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def plot_gw_spectrum(run_dir, out_path=None):
    run_dir = os.path.abspath(run_dir)
    params = _load_run_params(run_dir)
    mphi = float(params.get("mphi", 1000.0))
    T0 = float(params.get("T0", 7350.0))

    spectra_path = os.path.join(run_dir, "spectra_gws.txt")
    energy_path = os.path.join(run_dir, "energy_gws.txt")
    blocks = parse_spectra_gws(spectra_path)
    energy = parse_energy_gws(energy_path)

    if not blocks:
        raise RuntimeError(f"No GW spectrum blocks in {spectra_path}")

    n_blocks = len(blocks)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.viridis
    for i, (k, pgw) in enumerate(blocks):
        color = cmap(i / max(n_blocks - 1, 1))
        label = f"block {i + 1}"
        if energy is not None and i < len(energy[0]):
            t_prog = energy[0][i]
            T_est = T0 / max(1.0, t_prog / mphi) if params.get("no_hubble") else None
            if T_est is not None:
                label = f"t={t_prog:.2g}, T~{T_est:.0f} GeV"
        axes[0].loglog(k[k > 0], np.maximum(pgw[k > 0], 1e-40), color=color, alpha=0.8, label=label)

    axes[0].set_xlabel(r"$k$ (program units)")
    axes[0].set_ylabel(r"$P_{\mathrm{GW}}(k)$")
    axes[0].set_title("CosmoLattice GW spectrum evolution")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, which="both", alpha=0.3)

    if energy is not None:
        t, e_spec, e_rho = energy
        axes[1].semilogy(t, np.maximum(e_spec, 1e-40), "b-", label=r"$E_{\mathrm{spec}}$")
        axes[1].semilogy(t, np.maximum(e_rho, 1e-40), "r--", label=r"$E_{\mathrm{spec}} \times \rho$")
        axes[1].set_xlabel("t (program time)")
        axes[1].set_ylabel("GW spectral energy")
        axes[1].set_title("GW energy vs time")
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "energy_gws.txt not found", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.suptitle(os.path.basename(run_dir), fontsize=10)
    fig.tight_layout()

    if out_path is None:
        out_dir = os.path.join(run_dir, "figs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gw_spectrum.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} ({n_blocks} spectrum snapshots)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot CosmoLattice GW spectra")
    ap.add_argument("run_dir", help="CosmoLattice output directory")
    ap.add_argument("--out", default=None, help="Output PNG path")
    args = ap.parse_args()
    plot_gw_spectrum(args.run_dir, out_path=args.out)


if __name__ == "__main__":
    main()
