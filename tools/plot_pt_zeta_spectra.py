#!/usr/bin/env python3
"""ξ-axis plots for 𝒫_ζ (δt_c) and 𝒫_ρ (field) with CMB A_s reference."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

AS_CMB = 2.1e-9
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def _load_p_zeta_dtc_csv(path: str) -> Dict[str, Any]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    k = np.array([float(r["k"]) for r in rows], dtype=np.float64)
    p_key = "Pdim_zeta" if "Pdim_zeta" in rows[0] else "P_zeta"
    pz = np.array([
        float(r[p_key]) if r.get(p_key, "") not in ("", "nan") else float("nan")
        for r in rows
    ], dtype=np.float64)
    nm = np.array([float(r["n_modes"]) for r in rows], dtype=np.float64)
    H = float(rows[0].get("H", "nan"))
    t_ref = float(rows[0].get("t_ref", "nan"))
    return {"k": k, "pz": pz, "nm": nm, "H": H, "t_ref": t_ref, "path": path}


def _load_rho_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(
            [ln for ln in f if ln.strip() and not ln.startswith("#")]
        ))
    k = np.array([float(r["k"]) for r in rows], dtype=np.float64)
    pr = np.array([float(r["P_raw"]) for r in rows], dtype=np.float64)
    nm = np.array([float(r["n_modes"]) for r in rows], dtype=np.float64)
    prho = (k ** 3) / (2.0 * math.pi ** 2) * pr
    return k, prho, nm


def _find_clock_csv(clock_dir: str, t: float) -> Optional[str]:
    if not os.path.isdir(clock_dir):
        return None
    for name in sorted(os.listdir(clock_dir)):
        if not (name.startswith("bispectrum_t") and name.endswith(".csv")):
            continue
        m = re.search(r"bispectrum_t([0-9.+-]+)_step", name)
        if m and abs(float(m.group(1)) - t) < 0.6:
            return os.path.join(clock_dir, name)
    return None


def _fit_line(
    xi: np.ndarray,
    y: np.ndarray,
    *,
    fit_lo: float,
    fit_hi: float,
    xi_lo: float,
    xi_hi: float,
) -> Optional[Tuple[float, float, float]]:
    fit = (
        (xi >= fit_lo) & (xi <= fit_hi)
        & np.isfinite(y) & (y > 0)
    )
    if int(np.count_nonzero(fit)) < 3:
        return None
    n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(y[fit]), 1)
    return float(n_fit), float(math.exp(log_amp)), float(n_fit)


def plot_p_zeta_dtc_xi(
    dtc_dir: str,
    *,
    label: str = r"$\delta t_c$",
    out_name: str = "P_zeta_dtc_xi.png",
    fit_xi_lo: float = 0.50,
    fit_xi_hi: float = 0.85,
    xi_max: float = 7.0,
    y_lo: float = 1e-9,
    y_hi: float = 1e-1,
    show_xi3: bool = True,
) -> str:
    import matplotlib.pyplot as plt

    data = _load_p_zeta_dtc_csv(os.path.join(dtc_dir, "P_zeta_dtc.csv"))
    ok = np.isfinite(data["pz"]) & (data["pz"] > 0) & (data["nm"] >= 64)
    k, pz = data["k"][ok], data["pz"][ok]
    i_pk = int(np.nanargmax(pz))
    k_star = float(k[i_pk])
    xi = k / k_star

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(
        xi, pz, color="C3", lw=2.0,
        label=rf"{label}  ($k_*={k_star:.3g}$)",
    )

    xi_line = np.logspace(math.log10(float(np.min(xi))), math.log10(xi_max), 120)
    fit = (xi >= fit_xi_lo) & (xi <= fit_xi_hi) & (pz > 0)
    if show_xi3 and int(np.count_nonzero(fit)) >= 2:
        amp3 = float(np.nanmedian(pz[fit] / np.clip(xi[fit] ** 3, 1e-30, None)))
        ax.loglog(
            xi_line, amp3 * xi_line ** 3, color="C3", lw=1.4, ls="--",
            label=r"$\propto\xi^3$ (Jinno IR)",
        )
    if int(np.count_nonzero(fit)) >= 3:
        n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(pz[fit]), 1)
        amp_n = float(math.exp(log_amp))
        ax.loglog(
            xi_line, amp_n * xi_line ** float(n_fit), color="0.35", lw=1.2, ls=":",
            label=rf"free fit $\propto\xi^{{{float(n_fit):.2f}}}$",
        )

    ax.axhline(AS_CMB, color="0.4", ls=":", lw=1.2, label=rf"CMB $A_s={AS_CMB:.1e}$")
    ax.set_xlim(left=float(np.min(xi)), right=xi_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"$\xi \equiv k/k_* \simeq k\,d_b$")
    ax.set_ylabel(r"$\mathcal{P}_\zeta(\xi)$")
    title = r"$\mathcal{P}_\zeta = \frac{k^3}{2\pi^2}H^2 P_{\rm raw}(\delta t_c)$"
    if np.isfinite(data["H"]) and np.isfinite(data["t_ref"]):
        title += rf"  ($H={data['H']:.2e}$, $t_{{\rm ref}}={data['t_ref']:.0f}$)"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = os.path.join(dtc_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_p_zeta_dtc_compare(
    dirs: Sequence[Tuple[str, str]],
    out_path: str,
    *,
    xi_max: float = 7.0,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, (dtc_dir, label) in enumerate(dirs):
        data = _load_p_zeta_dtc_csv(os.path.join(dtc_dir, "P_zeta_dtc.csv"))
        ok = np.isfinite(data["pz"]) & (data["pz"] > 0) & (data["nm"] >= 64)
        k, pz = data["k"][ok], data["pz"][ok]
        k_star = float(k[int(np.nanargmax(pz))])
        xi = k / k_star
        color = COLORS[i % len(COLORS)]
        ax.loglog(xi, pz, color=color, lw=2.0, label=label)
        fit = (xi >= 0.5) & (xi <= 0.85) & (pz > 0)
        if int(np.count_nonzero(fit)) >= 3:
            n_fit, _ = np.polyfit(np.log(xi[fit]), np.log(pz[fit]), 1)
            ax.plot([], [], color=color, ls="--", lw=1.2,
                    label=rf"{label}  $\propto\xi^{{{float(n_fit):.2f}}}$")

    ax.axhline(AS_CMB, color="0.4", ls=":", lw=1.2, label=rf"CMB $A_s={AS_CMB:.1e}$")
    ax.set_xlim(right=xi_max)
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel(r"$\xi \equiv k/k_*$")
    ax.set_ylabel(r"$\mathcal{P}_\zeta(\xi)$")
    ax.set_title(r"$\delta t_c$ channel: old vs new $t_c$ map")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_p_rho_xi(
    clock_dir: str,
    *,
    times: Sequence[float] = (520.0, 580.0, 640.0),
    out_name: str = "P_rho_xi_compare.png",
    fit_xi_lo: float = 0.04,
    fit_xi_hi: float = 0.50,
    xi_max: float = 7.0,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xi_mins: List[float] = []

    for i, t in enumerate(times):
        path = _find_clock_csv(clock_dir, t)
        if path is None:
            continue
        k, prho, nm = _load_rho_csv(path)
        ok = np.isfinite(prho) & (prho > 0) & (nm >= 64)
        k, prho = k[ok], prho[ok]
        k_star = float(k[int(np.nanargmax(prho))])
        xi = k / k_star
        color = COLORS[i % len(COLORS)]
        ax.loglog(
            xi, prho, color=color, lw=1.8,
            label=rf"$t={t:.0f}$  ($k_*={k_star:.3g}$)",
        )
        fit = (xi >= fit_xi_lo) & (xi <= fit_xi_hi) & (prho > 0)
        if int(np.count_nonzero(fit)) >= 3:
            n_fit, log_amp = np.polyfit(np.log(xi[fit]), np.log(prho[fit]), 1)
            amp = float(math.exp(log_amp))
            xi_line = np.logspace(math.log10(float(np.min(xi))), math.log10(xi_max), 120)
            ax.loglog(
                xi_line, amp * xi_line ** float(n_fit), color=color, lw=1.4, ls="--",
                label=rf"$t={t:.0f}$  $\propto\xi^{{{float(n_fit):.2f}}}$",
            )
        xi_mins.append(float(np.min(xi)))

    ax.axhline(AS_CMB, color="0.4", ls=":", lw=1.0,
               label=rf"CMB $A_s$ (ref.)")
    if xi_mins:
        ax.set_xlim(left=min(xi_mins), right=xi_max)
    ax.set_ylim(1e-9, 1e-1)
    ax.set_xlabel(r"$\xi \equiv k/k_* \simeq k\,d_b$")
    ax.set_ylabel(r"$\mathcal{P}_\rho(\xi)=\frac{k^3}{2\pi^2}P_{\rm raw}$")
    ax.set_title(r"Field $\rho_{\rm norm}$: dimensionless power with IR $\xi^n$ fits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = os.path.join(clock_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_amplitude_diagnostic(dtc_dir: str, run_dir: str, out_name: str = "P_zeta_amplitude_diagnostic.png") -> str:
    """Bar-style diagnostic: peak 𝒫_ζ vs CMB and implied H or σ_δt for match."""
    import matplotlib.pyplot as plt

    data = _load_p_zeta_dtc_csv(os.path.join(dtc_dir, "P_zeta_dtc.csv"))
    ok = np.isfinite(data["pz"]) & (data["pz"] > 0) & (data["nm"] >= 64)
    peak = float(np.max(data["pz"][ok]))
    H = float(data["H"])
    with open(os.path.join(dtc_dir, "P_dtc.csv")) as f:
        rows = list(csv.DictReader(f))
    p_key = "P_dtc" if rows and "P_dtc" in rows[0] else "P"
    var = float(np.nansum([
        float(r[p_key]) for r in rows if r.get(p_key, "") not in ("", "nan")
    ]))
    sig = math.sqrt(max(var, 0.0))
    Hsig = H * sig
    ok_idx = np.where(ok)[0]
    with open(os.path.join(dtc_dir, "P_zeta_dtc.csv")) as f:
        zrows = list(csv.DictReader(f))
    pdim_dtc = float(zrows[int(ok_idx[int(np.argmax(data["pz"][ok]))])]["Pdim_dtc"])
    H_need = math.sqrt(AS_CMB / pdim_dtc) if pdim_dtc > 0 else float("nan")
    sig_need2 = sig * math.sqrt(AS_CMB / peak) if peak > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["peak $\\mathcal{P}_\\zeta$", "CMB $A_s$", "$H\\sigma_{\\delta t}$"]
    vals = [peak, AS_CMB, Hsig]
    colors = ["C3", "0.45", "C0"]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("amplitude")
    ax.set_title(
        rf"Amplitude check ($H={H:.2e}$, $\sigma_{{\delta t}}={sig:.2f}$)\n"
        rf"peak$/A_s={peak/AS_CMB:.0f}$; "
        rf"match $A_s$ $\Rightarrow$ $H_{{\rm need}}\sim{H_need:.2e}$ or $\sigma_{{\rm need}}\sim{sig_need2:.2f}$"
    )
    fig.tight_layout()
    out = os.path.join(dtc_dir, out_name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="CosmoLattice run directory")
    ap.add_argument(
        "--dtc-dir",
        default="string_new/strings/transition_correlators_new",
        help="relative path under run_dir with P_zeta_dtc.csv",
    )
    ap.add_argument(
        "--dtc-dir-old",
        default="string_new/strings/transition_correlators",
        help="optional old δt_c dir for comparison",
    )
    ap.add_argument(
        "--clock-dir",
        default="string_new/strings/bispectrum_time_series_pt/rho_norm",
    )
    ap.add_argument("--times", type=float, nargs="+", default=[520, 580, 640])
    args = ap.parse_args(argv)

    run = os.path.abspath(args.run_dir)
    dtc = os.path.join(run, args.dtc_dir)
    clock = os.path.join(run, args.clock_dir)

    outs = []
    outs.append(plot_p_zeta_dtc_xi(dtc))
    outs.append(plot_amplitude_diagnostic(dtc, run))
    outs.append(plot_p_rho_xi(clock, times=args.times))

    old = os.path.join(run, args.dtc_dir_old)
    if os.path.isfile(os.path.join(old, "P_zeta_dtc.csv")):
        cmp_path = os.path.join(dtc, "P_zeta_dtc_compare_old_new.png")
        outs.append(plot_p_zeta_dtc_compare(
            [(old, "escape / coarse $t_c$"), (dtc, "vev_frac + interp $t_c$")],
            cmp_path,
        ))
        plot_p_zeta_dtc_xi(old, label=r"$\delta t_c$ (old)", out_name="P_zeta_dtc_xi_old.png")

    for p in outs:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
