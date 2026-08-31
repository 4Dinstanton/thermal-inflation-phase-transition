#!/usr/bin/env python3
"""Estimate cosmic-string Gμ and a simple analytic h²Ω_GW(f) from string_summary.csv.

Order-of-magnitude / upper-envelope tool — **not** a lattice measurement of
string GW.

What it does
------------
1. Read ``string_summary.csv``.
2. Estimate μ_eff after T_c1 (default t ≥ 470).
3. Convert to Gμ = μ / M_Pl² (M_Pl = 2.4×10¹⁸ GeV).
4. Build a simple analytic stochastic GW spectrum:
   - NG/BOS-like upper envelope (optimistic for local NG strings)
   - TI/global suppressed curve (extra factor; Goldstones + cusp annihilation
     dominate for flaton / global U(1) strings — Brandenberger & Favero)
5. Optionally overlay the CosmoLattice **phase-transition** GW spectrum.

Usage
-----
    python tools/estimate_string_gw.py \\
        data/lattice/set8/<run>/string_new/strings

    python tools/estimate_string_gw.py \\
        data/lattice/set8/<run>/string_new/strings \\
        --run-dir data/lattice/set8/<run> --T-rh 1000 --t-min 470

Outputs
-------
    <strings_dir>/string_gw_estimate.png
    <strings_dir>/string_gw_estimate.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

M_PL = 2.4e18
G_NEWTON = 1.0 / (M_PL * M_PL)
OMEGA_R_H2 = 4.15e-5
GAMMA_GW = 50.0
C_BOS = 0.05


def _float(row: Dict[str, str], *keys: str) -> float:
    lower = {k.lower(): v for k, v in row.items()}
    for key in keys:
        v = row.get(key, lower.get(key.lower(), ""))
        if v in ("", "nan", "None", None):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return float("nan")


def resolve_csv(strings_dir: Path) -> Path:
    for name in ("string_summary.csv", "string_network_summary.csv", "summary.csv"):
        p = strings_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No string_summary.csv under {strings_dir}")


def load_string_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def estimate_mu_eff(
    rows: List[Dict[str, str]],
    *,
    t_min: Optional[float] = None,
    step_min: Optional[int] = None,
) -> Dict[str, Any]:
    selected = []
    for r in rows:
        t = _float(r, "time")
        step = _float(r, "step")
        mu = _float(r, "mu_eff")
        if not np.isfinite(mu) or mu <= 0:
            e = _float(r, "E_tot_core")
            L = _float(r, "L_comoving")
            if np.isfinite(e) and np.isfinite(L) and L > 0:
                mu = e / L
            else:
                continue
        if t_min is not None and (not np.isfinite(t) or t < t_min):
            continue
        if step_min is not None and (not np.isfinite(step) or int(step) < step_min):
            continue
        selected.append(
            {
                "t": t,
                "step": step,
                "T": _float(r, "temperature"),
                "mu_eff": mu,
                "L_comoving": _float(r, "L_comoving"),
                "xi_comoving": _float(r, "xi_comoving"),
            }
        )
    if not selected:
        raise RuntimeError("No usable rows for μ_eff (check --t-min / CSV).")

    mus = np.asarray([s["mu_eff"] for s in selected], dtype=float)
    mu_med = float(np.median(mus))
    gmu = mu_med * G_NEWTON
    return {
        "n_rows": len(selected),
        "t_min_used": float(selected[0]["t"]),
        "t_max_used": float(selected[-1]["t"]),
        "mu_eff_median_GeV2": mu_med,
        "mu_eff_mean_GeV2": float(np.mean(mus)),
        "mu_eff_last_GeV2": float(mus[-1]),
        "Gmu": gmu,
        "log10_Gmu": float(math.log10(gmu)) if gmu > 0 else float("nan"),
        "rows": selected,
    }


def bos_like_plateau(gmu: float, *, gamma: float = GAMMA_GW, c: float = C_BOS) -> float:
    """Optimistic NG/BOS-like radiation-era plateau: h²Ω ∼ C Ω_r h² (Gμ)^{3/2}/√Γ."""
    if gmu <= 0:
        return 0.0
    return float(c * OMEGA_R_H2 * (gmu**1.5) / math.sqrt(gamma))


def analytic_string_spectrum(
    f_hz: np.ndarray,
    gmu: float,
    *,
    f_peak_hz: float,
    gamma: float = GAMMA_GW,
    ti_suppress: float = 1.0,
) -> np.ndarray:
    """Broken-power-law analytic string GW template (schematic)."""
    amp = bos_like_plateau(gmu, gamma=gamma) * float(ti_suppress)
    x = np.asarray(f_hz, dtype=float) / max(f_peak_hz, 1e-30)
    x = np.clip(x, 1e-30, None)
    s_ir = x**1.5
    s_uv = x ** (-1.0 / 3.0)
    shape = (s_ir ** (-2) + s_uv ** (-2)) ** (-0.5)
    shape = shape / max(float(np.nanmax(shape)), 1e-30)
    shape = shape * np.exp(-((x / 30.0) ** 2))
    return amp * shape


def estimate_f_peak_hz(
    *,
    T_rh: float,
    g_star: float = 100.0,
    V_TI: float = 2.5e27,
    alpha: float = 0.1,
) -> float:
    """Rough peak frequency today near TI → RD reheating (α = loop-size param)."""
    T_CMB = 2.7255 * 8.617333262145e-14
    G_S0 = 3.91
    HBAR = 6.582119569e-25
    GEV_TO_HZ = 1.0 / HBAR

    R_md = (math.pi**2 * g_star * T_rh**4 / (30.0 * max(V_TI, 1.0))) ** (1.0 / 3.0)
    a_rh_a0 = (G_S0 / g_star) ** (1.0 / 3.0) * T_CMB / T_rh
    a_star_a0 = R_md * a_rh_a0
    H_star = (T_rh * T_rh) / M_PL
    f_star_gev = H_star / (2.0 * math.pi * max(alpha, 1e-4))
    return float(f_star_gev * a_star_a0 * GEV_TO_HZ)


def try_load_lattice_pt_spectrum(
    run_dir: Path,
    *,
    T_rh: float,
    g_star: float = 100.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """CosmoLattice PT GW → (f_Hz, h²Ω, meta), following plot_physical_omega."""
    try:
        from postprocess.plot_cl_gw_spectrum import (
            _delV_from_params,
            _fstar_omega,
            _load_input_in,
            _load_run_params,
            convert_to_today,
            parse_energy_gws,
            parse_scale_factor,
            parse_spectra_gws,
            resolve_gw_paths,
            select_production_index,
        )
    except Exception as exc:
        print(f"[warn] cannot import plot_cl_gw_spectrum: {exc}")
        return None

    spectra_path, energy_path = resolve_gw_paths(str(run_dir))
    if spectra_path is None:
        print(f"[warn] no spectra_gws in {run_dir}")
        return None
    blocks = parse_spectra_gws(spectra_path)
    energy = parse_energy_gws(energy_path)
    if not blocks:
        return None

    params = _load_run_params(str(run_dir))
    inp = _load_input_in(str(run_dir))
    V_TI = float(_delV_from_params(params, inp))
    fStar, omega_star = _fstar_omega(params, inp)
    g_star = float(params.get("g_star", inp.get("g_star", g_star)))

    idx = int(select_production_index(energy, blocks))
    kappa, omega = blocks[idx]

    if energy is not None and idx < len(energy[0]) and abs(energy[1][idx]) > 0:
        rho_tot = float(energy[2][idx] / energy[1][idx])
        t_star = float(energy[0][idx])
    else:
        rho_tot = 0.25
        t_star = float("nan")

    sf = parse_scale_factor(str(run_dir))
    if sf is not None:
        t_sf, a_sf, H_sf = sf
        a_star = (
            float(np.interp(t_star, t_sf, a_sf))
            if np.isfinite(t_star)
            else float(a_sf[-1])
        )
        H_prog = (
            float(np.interp(t_star, t_sf, H_sf))
            if np.isfinite(t_star)
            else float(H_sf[-1])
        )
        H_phys = H_prog * float(omega_star) if H_prog < 1.0 else H_prog
    else:
        a_star = 1.0
        H_phys = math.sqrt(max(V_TI, 0.0) / (3.0 * M_PL**2))

    T0 = float(params.get("T0", inp.get("T0", 1230.0)))
    T_star = T0 / max(a_star, 1e-30)
    chi_g2 = 30.0 / (math.pi**2 * g_star)
    H_prescribed = math.sqrt((T_star**4 / chi_g2 + V_TI) / (3.0 * M_PL**2))
    H_use = max(H_phys, H_prescribed) if np.isfinite(H_phys) else H_prescribed

    try:
        f_hz, h2, meta = convert_to_today(
            kappa,
            omega,
            rho_tot_prog=rho_tot,
            a_star=a_star,
            H_star_phys=H_use,
            omega_star=float(omega_star),
            fStar=float(fStar),
            V_TI=V_TI,
            T_rh=T_rh,
            g_star=g_star,
        )
    except Exception as exc:
        print(f"[warn] lattice conversion failed: {exc}")
        return None

    mask = (f_hz > 0) & np.isfinite(h2) & (h2 > 0)
    meta = dict(meta)
    meta["block_index"] = idx
    meta["V_TI"] = float(V_TI)
    meta["H_use_GeV"] = float(H_use)
    return np.asarray(f_hz[mask]), np.asarray(h2[mask]), meta


def plot_estimate(
    out_png: Path,
    *,
    f_hz: np.ndarray,
    h2_ng: np.ndarray,
    h2_ti: np.ndarray,
    gmu: float,
    f_peak: float,
    ti_suppress: float,
    lattice: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.loglog(
        f_hz,
        h2_ng,
        "C0-",
        lw=2.0,
        label=rf"NG/BOS-like upper env. ($G\mu={gmu:.2e}$)",
    )
    ax.loglog(
        f_hz,
        h2_ti,
        "C3--",
        lw=1.8,
        label=rf"TI/global suppressed ($\times {ti_suppress:g}$)",
    )
    if lattice is not None:
        fl, hl = lattice
        ax.loglog(fl, hl, "C2-", lw=1.4, alpha=0.85, label="Lattice PT (CosmoLattice)")

    ax.axvline(
        f_peak,
        color="0.4",
        ls=":",
        lw=1.0,
        label=rf"$f_{{\rm peak}}\sim{f_peak:.2e}$ Hz",
    )
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$h^2\Omega_{\mathrm{GW}}(f)$")
    ax.set_xlim(1e-10, 1e3)
    pos = h2_ti[h2_ti > 0]
    ymin = max(1e-25, 0.1 * float(np.nanmin(pos))) if len(pos) else 1e-25
    ymax = max(1e-8, 10 * float(np.nanmax(h2_ng)))
    if lattice is not None and len(lattice[1]):
        ymax = max(ymax, 10 * float(np.nanmax(lattice[1])))
        ymin = min(ymin, 0.1 * float(np.nanmin(lattice[1][lattice[1] > 0])))
    ax.set_ylim(ymin, ymax)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    if title:
        short = title if len(title) <= 90 else title[:40] + "…" + title[-45:]
        ax.set_title(short, fontsize=9)
    ax.text(
        0.02,
        0.02,
        "Analytic string GW = order-of-magnitude only.\n"
        "NG curve is optimistic for global/TI flaton strings.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "strings_dir",
        help="Directory with string_summary.csv (e.g. <run>/string_new/strings)",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="CosmoLattice run root (optional lattice PT GW overlay)",
    )
    ap.add_argument("--t-min", type=float, default=470.0)
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--T-rh", type=float, default=1000.0)
    ap.add_argument("--V-TI", type=float, default=2.5e27)
    ap.add_argument("--g-star", type=float, default=100.0)
    ap.add_argument(
        "--ti-suppress",
        type=float,
        default=1e-2,
        help="Extra suppression for TI/global vs NG (default 1e-2)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Loop-size parameter α∼L/(ξ) for f_peak (default 0.1)",
    )
    ap.add_argument("-o", "--out-dir", default=None)
    args = ap.parse_args(argv)

    strings_dir = Path(args.strings_dir).resolve()
    csv_path = resolve_csv(strings_dir)
    rows = load_string_csv(csv_path)
    est = estimate_mu_eff(rows, t_min=args.t_min, step_min=args.step_min)

    f_peak = estimate_f_peak_hz(
        T_rh=args.T_rh,
        g_star=args.g_star,
        V_TI=args.V_TI,
        alpha=args.alpha,
    )
    gmu = float(est["Gmu"])
    f_hz = np.logspace(-10, 3, 800)
    h2_ng = analytic_string_spectrum(f_hz, gmu, f_peak_hz=f_peak, ti_suppress=1.0)
    h2_ti = analytic_string_spectrum(
        f_hz, gmu, f_peak_hz=f_peak, ti_suppress=args.ti_suppress
    )

    lattice = None
    lattice_meta: Dict[str, Any] = {}
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None:
        cand = strings_dir.parent.parent
        if (cand / "spectra_gws.txt").is_file() or (cand / "energy_gws.txt").is_file():
            run_dir = cand
    if run_dir is not None:
        loaded = try_load_lattice_pt_spectrum(
            run_dir, T_rh=args.T_rh, g_star=args.g_star
        )
        if loaded is not None:
            fl, hl, lattice_meta = loaded
            lattice = (fl, hl)
            print(f"  lattice PT overlay: {len(fl)} bins from {run_dir.name}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else strings_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "string_gw_estimate.png"
    out_json = out_dir / "string_gw_estimate.json"

    plot_estimate(
        out_png,
        f_hz=f_hz,
        h2_ng=h2_ng,
        h2_ti=h2_ti,
        gmu=gmu,
        f_peak=f_peak,
        ti_suppress=args.ti_suppress,
        lattice=lattice,
        title=f"{strings_dir.parent.name}/{strings_dir.name}",
    )

    summary = {
        "csv": str(csv_path),
        "t_min": args.t_min,
        "mu_eff_median_GeV2": est["mu_eff_median_GeV2"],
        "mu_eff_mean_GeV2": est["mu_eff_mean_GeV2"],
        "Gmu": gmu,
        "log10_Gmu": est["log10_Gmu"],
        "Gmu_from_pi_phi0_sq": math.pi * (1.0e15) ** 2 * G_NEWTON,
        "bos_like_plateau_h2Omega": bos_like_plateau(gmu),
        "ti_suppressed_plateau_h2Omega": bos_like_plateau(gmu) * args.ti_suppress,
        "ti_suppress": args.ti_suppress,
        "f_peak_Hz": f_peak,
        "alpha_loop": args.alpha,
        "T_rh_GeV": args.T_rh,
        "V_TI_GeV4": args.V_TI,
        "n_rows_used": est["n_rows"],
        "t_window": [est["t_min_used"], est["t_max_used"]],
        "lattice_overlay": lattice is not None,
        "lattice_meta": {
            k: float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
            for k, v in lattice_meta.items()
        },
        "caveats": [
            "NG/BOS curve is an optimistic upper envelope for local NG strings.",
            "Thermal-inflation / global U(1) strings mainly lose energy to "
            "Goldstone radiation and cusp annihilation (Brandenberger & Favero); "
            "GW from loops is expected to be strongly suppressed vs NG.",
            "This run is not yet in a clear scaling regime; μ_eff is a post-T_c1 proxy.",
            "Lattice CosmoLattice GW (if shown) is dominated by the phase transition, "
            "not a pure string network signal.",
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print("=== string GW estimate ===")
    print(f"  CSV:        {csv_path}")
    print(
        f"  window:     t∈[{est['t_min_used']:.1f}, {est['t_max_used']:.1f}]  "
        f"({est['n_rows']} rows)"
    )
    print(f"  μ_eff:      {est['mu_eff_median_GeV2']:.3e} GeV²  (median)")
    print(f"  Gμ:         {gmu:.3e}   (log10 = {est['log10_Gmu']:.2f})")
    print(f"  NG plateau: {summary['bos_like_plateau_h2Omega']:.3e}  (h²Ω)")
    print(
        f"  TI×{args.ti_suppress:g}:  "
        f"{summary['ti_suppressed_plateau_h2Omega']:.3e}  (h²Ω)"
    )
    print(f"  f_peak:     {f_peak:.3e} Hz  (α={args.alpha:g})")
    print(f"  wrote:      {out_png}")
    print(f"  wrote:      {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
