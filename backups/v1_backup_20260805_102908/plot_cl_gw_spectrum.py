#!/usr/bin/env python3
"""Plot CosmoLattice GW spectra and convert to h²Ω_GW(f) today.

CosmoLattice writes (technical note):
  spectra_gws.txt  — κ, Ω_GW(κ) = (1/ρ̃_tot) dρ̃_GW/dlogκ , #modes
  energy_gws.txt   — t, E_spec, E_spec * ρ̃_tot

Because StochasticRK uses *prescribed* H(T,ΔV), ρ̃_tot ≠ ρ̃_c. To get the
critical-density normalization multiply by ρ̃_tot / ρ̃_c.

Thermal-inflation redshift (EGLPS / Caprini-style):
  R_md = (π² g_* T_rh⁴ / (30 V_TI))^{1/3}   # MD dilution a_*/a_rh
  a_rh/a_0 from T_rh → today
  h²Ω_0 = F_GW * Ω_crit,* * R_md
  f_0   = (k_comov / 2π) * R_md * (a_rh/a_0)   [Hz]

Usage
-----
    python postprocess/plot_cl_gw_spectrum.py <run_dir>
    python postprocess/plot_cl_gw_spectrum.py <run_dir> --physical --T_rh 1000
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

M_PL = 2.4e18  # reduced Planck mass [GeV]
T_CMB_GEV = 2.7255 * 8.617333262145e-14
G_S0 = 3.91
HBAR_GEV_S = 6.582119569e-25  # ℏ [GeV·s]
GEV_TO_HZ = 1.0 / HBAR_GEV_S


def _load_run_params(run_dir):
    path = os.path.join(run_dir, "cl_run_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_input_in(run_dir):
    """Best-effort parse of CosmoLattice input.in key=value pairs."""
    path = os.path.join(run_dir, "input.in")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k] = v
    return out


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


def parse_scale_factor(run_dir):
    """average_scale_factor.txt -> t, a, H (best-effort column guess)."""
    path = os.path.join(run_dir, "average_scale_factor.txt")
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0]
    a = data[:, 1]
    # Prefer last column as H when 4 columns (t, a, adot, H)
    H = data[:, -1] if data.shape[1] >= 3 else np.full_like(a, np.nan)
    return t, a, H


def _delV_from_params(params, inp):
    if "delV" in inp:
        return float(inp["delV"])
    gamma = float(params.get("gamma", inp.get("gamma", 4.1667e-4)))
    mphi = float(params.get("mphi", inp.get("mphi", 1000.0)))
    phi0 = gamma * M_PL
    lam = mphi * mphi / (phi0 * phi0)
    return 0.25 * lam * phi0**4


def _fstar_omega(params, inp):
    gamma = float(params.get("gamma", inp.get("gamma", 4.1667e-4)))
    mphi = float(params.get("mphi", inp.get("mphi", 1000.0)))
    fStar = gamma * M_PL
    return fStar, mphi


def select_production_index(energy, blocks):
    """Pick snapshot with largest positive E_spec*|rho| proxy (E_rho)."""
    if energy is None:
        return len(blocks) - 1
    t, e_spec, e_rho = energy
    n = min(len(blocks), len(t))
    best = 0
    best_val = -np.inf
    for i in range(n):
        # Prefer late positive growth: use |E_rho| when E_spec>0
        val = e_rho[i] if e_spec[i] > 0 else -np.inf
        if val > best_val:
            best_val = val
            best = i
    return best


def convert_to_today(
    kappa,
    omega_over_rho_tot,
    *,
    rho_tot_prog,
    a_star,
    H_star_phys,
    omega_star,
    fStar,
    V_TI,
    T_rh,
    g_star,
):
    """Map one CL spectrum block to (f_Hz, h2_Omega_GW) today.

    Parameters
    ----------
    kappa : program wavenumber column from spectra_gws.txt
    omega_over_rho_tot : CL Ω_GW = ρ_GW/ρ_tot (second column)
    rho_tot_prog : Energies::rho in program units at that dump
    a_star : scale factor at production
    H_star_phys : Hubble [GeV] at production (prescribed)
    omega_star, fStar : CosmoLattice scales [GeV]
    V_TI, T_rh, g_star : TI vacuum energy, reheating T, g_*
    """
    # Critical density in program units: ρ_c = 3 H² Mpl² / (fStar² ωStar²)
    rho_c_phys = 3.0 * H_star_phys**2 * M_PL**2
    rho_c_prog = rho_c_phys / (fStar**2 * omega_star**2)
    if rho_c_prog <= 0 or not np.isfinite(rho_c_prog):
        raise RuntimeError("Invalid rho_c for GW conversion")

    # CL note: multiply by ρ_tot/ρ_c when expansion is prescribed
    rho_tot = max(rho_tot_prog, 1e-300)
    omega_crit = omega_over_rho_tot * (rho_tot / rho_c_prog)

    # Comoving k [GeV]: κ_prog * ω_*  (κ = k_phys/ω_* at a≈1 lattice units)
    k_comov = np.asarray(kappa, dtype=float) * omega_star

    R_md = (math.pi**2 * g_star * T_rh**4 / (30.0 * V_TI)) ** (1.0 / 3.0)
    a_rh_over_a0 = (G_S0 / g_star) ** (1.0 / 3.0) * T_CMB_GEV / T_rh
    # a_*/a_0 = (a_*/a_rh)*(a_rh/a_0) ≈ R_md * a_rh/a_0
    # (R_md matches energy-equivalent MD dilution used in gwSpectrum.gw_thermal_inflation)
    a_star_over_a0 = R_md * a_rh_over_a0

    f_hz = (k_comov / (2.0 * math.pi)) * a_star_over_a0 * GEV_TO_HZ

    F_GW = 1.67e-5 * (100.0 / g_star) ** (1.0 / 3.0)
    h2_omega = F_GW * omega_crit * R_md

    meta = {
        "R_md": R_md,
        "F_GW": F_GW,
        "rho_tot_prog": rho_tot,
        "rho_c_prog": rho_c_prog,
        "rho_tot_over_rho_c": rho_tot / rho_c_prog,
        "a_star": a_star,
        "H_star_phys": H_star_phys,
        "a_star_over_a0": a_star_over_a0,
    }
    return f_hz, h2_omega, meta


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
            else:
                label = f"t={t_prog:.2g}"
        mask = k > 0
        axes[0].loglog(
            k[mask], np.maximum(np.abs(pgw[mask]), 1e-40),
            color=color, alpha=0.8, label=label,
        )

    axes[0].set_xlabel(r"$\kappa$ (program units)")
    axes[0].set_ylabel(r"$\Omega_{\mathrm{GW}}(\kappa)/\tilde\rho_{\mathrm{tot}}$ (CL)")
    axes[0].set_title("CosmoLattice GW spectrum evolution")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, which="both", alpha=0.3)

    if energy is not None:
        t, e_spec, e_rho = energy
        axes[1].semilogy(t, np.maximum(np.abs(e_spec), 1e-40), "b-", label=r"$E_{\mathrm{spec}}$")
        axes[1].semilogy(t, np.maximum(np.abs(e_rho), 1e-40), "r--", label=r"$E_{\mathrm{spec}} \times \rho$")
        axes[1].set_xlabel("t (program time)")
        axes[1].set_ylabel("GW spectral energy")
        axes[1].set_title("GW energy vs time")
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.3)
    else:
        axes[1].text(
            0.5, 0.5, "energy_gws.txt not found",
            ha="center", va="center", transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()

    fig.suptitle(os.path.basename(run_dir), fontsize=10)
    fig.tight_layout()

    if out_path is None:
        out_dir = os.path.join(run_dir, "figs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gw_spectrum.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} ({n_blocks} spectrum snapshots)")
    return out_path


def plot_physical_omega(
    run_dir,
    out_path=None,
    T_rh=None,
    snapshot=None,
    g_star=106.75,
):
    run_dir = os.path.abspath(run_dir)
    params = _load_run_params(run_dir)
    inp = _load_input_in(run_dir)
    blocks = parse_spectra_gws(os.path.join(run_dir, "spectra_gws.txt"))
    energy = parse_energy_gws(os.path.join(run_dir, "energy_gws.txt"))
    sf = parse_scale_factor(run_dir)
    if not blocks:
        raise RuntimeError("No GW spectrum blocks")

    V_TI = _delV_from_params(params, inp)
    fStar, omega_star = _fstar_omega(params, inp)
    if T_rh is None:
        T_rh = float(params.get("T_rh", 0.0) or 0.0)
    if T_rh <= 0:
        T_rh = float(params.get("T0", inp.get("T0", 1000.0))) * 0.8
        print(f"WARNING: T_rh not set; defaulting to 0.8*T0 = {T_rh:g} GeV")

    idx = snapshot if snapshot is not None else select_production_index(energy, blocks)
    idx = int(np.clip(idx, 0, len(blocks) - 1))
    kappa, omega_cl = blocks[idx]

    # ρ_tot from energy_gws: E_rho / E_spec
    if energy is not None and idx < len(energy[0]) and abs(energy[1][idx]) > 0:
        rho_tot = energy[2][idx] / energy[1][idx]
        t_star = energy[0][idx]
    else:
        rho_tot = 0.25  # false-vac ΔV in program units (shifted tree)
        t_star = float("nan")
        print("WARNING: could not infer rho_tot from energy_gws; using 0.25")

    # a(t), H(t)
    if sf is not None:
        t_sf, a_sf, H_sf = sf
        a_star = float(np.interp(t_star, t_sf, a_sf)) if np.isfinite(t_star) else float(a_sf[-1])
        H_prog = float(np.interp(t_star, t_sf, H_sf)) if np.isfinite(t_star) else float(H_sf[-1])
        # average_scale_factor H is often H/ω_* (program); convert to GeV
        H_phys = H_prog * omega_star if H_prog < 1.0 else H_prog
    else:
        a_star = 1.0
        H_phys = math.sqrt(V_TI / (3.0 * M_PL**2))
        print("WARNING: no average_scale_factor.txt; using H=H_TI vacuum")

    # Prefer prescribed H from vacuum+radiation at T≈T0/a
    T0 = float(params.get("T0", inp.get("T0", 1230.0)))
    T_star = T0 / max(a_star, 1e-30)
    chig2 = 30.0 / (math.pi**2 * g_star)
    H_prescribed = math.sqrt((T_star**4 / chig2 + V_TI) / (3.0 * M_PL**2))
    # During/after nucleation ΔV may still dominate; use max for conservative ρ_c
    H_use = max(H_phys, H_prescribed) if np.isfinite(H_phys) else H_prescribed

    f_hz, h2_omega, meta = convert_to_today(
        kappa,
        omega_cl,
        rho_tot_prog=rho_tot,
        a_star=a_star,
        H_star_phys=H_use,
        omega_star=omega_star,
        fStar=fStar,
        V_TI=V_TI,
        T_rh=T_rh,
        g_star=g_star,
    )

    mask = (f_hz > 0) & np.isfinite(h2_omega) & (h2_omega > 0)
    f_hz, h2_omega = f_hz[mask], h2_omega[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(f_hz, h2_omega, "b-", lw=2, label=f"CL snapshot {idx}" + (f" (t={t_star:g})" if np.isfinite(t_star) else ""))
    ax.set_xlabel(r"$f$ [Hz] (today)")
    ax.set_ylabel(r"$h^2\Omega_{\mathrm{GW}}(f)$")
    ax.set_title("Lattice GW → today (TI MD dilution)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Annotate peak
    if len(h2_omega):
        ip = int(np.argmax(h2_omega))
        ax.plot(f_hz[ip], h2_omega[ip], "ro", ms=5)
        ax.annotate(
            f"peak\n$f$={f_hz[ip]:.3g} Hz\n$h^2\\Omega$={h2_omega[ip]:.3g}",
            xy=(f_hz[ip], h2_omega[ip]),
            xytext=(0.55, 0.55),
            textcoords="axes fraction",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.4"),
        )

    text = (
        f"$T_{{\\mathrm{{rh}}}}$={T_rh:g} GeV\n"
        f"$R_{{\\mathrm{{md}}}}$={meta['R_md']:.3e}\n"
        f"$\\tilde\\rho_{{\\mathrm{{tot}}}}/\\tilde\\rho_c$={meta['rho_tot_over_rho_c']:.3e}\n"
        f"$a_*$={meta['a_star']:.4g}"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    if out_path is None:
        out_dir = os.path.join(run_dir, "figs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gw_omega_today.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Save numeric spectrum
    npz_path = os.path.splitext(out_path)[0] + ".npz"
    np.savez(
        npz_path,
        f_hz=f_hz,
        h2_omega=h2_omega,
        kappa=kappa[mask] if mask.shape == kappa.shape else kappa,
        snapshot=idx,
        T_rh=T_rh,
        V_TI=V_TI,
        **{k: np.asarray(v) for k, v in meta.items()},
    )

    print(f"Wrote {out_path}")
    print(f"Wrote {npz_path}")
    print(f"  snapshot={idx}  T_rh={T_rh:g}  R_md={meta['R_md']:.4e}")
    print(f"  rho_tot/rho_c={meta['rho_tot_over_rho_c']:.4e}  H_*={H_use:.4e} GeV")
    if len(h2_omega):
        print(f"  peak: f={f_hz[ip]:.4e} Hz  h²Ω={h2_omega[ip]:.4e}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot CosmoLattice GW spectra")
    ap.add_argument("run_dir", help="CosmoLattice output directory")
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument(
        "--physical",
        action="store_true",
        help="Convert selected snapshot to h²Ω_GW(f) today with TI MD dilution",
    )
    ap.add_argument("--T_rh", type=float, default=None,
                    help="Reheating temperature [GeV] for redshift (default: from cl_run_params)")
    ap.add_argument("--snapshot", type=int, default=None,
                    help="Spectrum block index (default: max E_spec*ρ)")
    ap.add_argument("--g_star", type=float, default=106.75)
    args = ap.parse_args()

    if args.physical:
        plot_physical_omega(
            args.run_dir,
            out_path=args.out,
            T_rh=args.T_rh,
            snapshot=args.snapshot,
            g_star=args.g_star,
        )
    else:
        plot_gw_spectrum(args.run_dir, out_path=args.out)


if __name__ == "__main__":
    main()
