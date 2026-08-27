#!/usr/bin/env python3
"""Cosmic-string network observables from lattice φ (and optional π).

Designed for CosmoLattice thermal-inflation global U(1) runs (N up to 1024³).

**Does not read** CosmoLattice energy-snapshot HDF5 (``E_S_*`` / potential /
gradient dumps can be O(100 GB) per series). Kinetic / gradient / tree
potential energies on string cores are accumulated from ``phi_*`` + ``pi_*``
for the **current** time slice only.

Observables (global-string / one-scale style)
---------------------------------------------
- Comoving length ``L_com`` from plaquette winding
- Mean separation ``ξ = V^{1/2} / L^{1/2}`` (i.e. ``sqrt(Volume/L)``)
- Core kinetic / gradient / tree-potential energy and ``μ_eff = E_tot/L``
- Mean-square velocity estimator on cores
- Optional connected-component loop census (voxel lengths)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Default tree potential (matches Potential.zeroTemperaturePotential.V_tree
# with complex magnitude |Φ|): V = (λ/4)|Φ|^4 − (m²/2)|Φ|^2
_DEFAULT_MPHI = 1000.0
_DEFAULT_LAM = 1.0e-6  # overridden from cl_run_params when present


def _as_f32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def string_core_mask(winding: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.abs(winding) > threshold


def comoving_string_length(
    winding: np.ndarray,
    dx_com: float,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Total comoving string length from site-summed plaquette windings.

    ``compute_winding_number`` deposits |ν| from XY+XZ+YZ plaquettes onto
    sites. We use ``L = dx * Σ |W|`` (all sites), and also report the masked
    voxel count length ``L_vox = dx * N_vox`` as a coarser proxy.

    Returns
    -------
    L_winding, L_voxel
    """
    w = np.asarray(winding)
    L_w = float(dx_com) * float(np.sum(np.abs(w), dtype=np.float64))
    n_vox = int(np.sum(np.abs(w) > threshold))
    L_vox = float(dx_com) * float(n_vox)
    return L_w, L_vox


def mean_separation(volume: float, length: float) -> float:
    """ξ = sqrt(V / L); returns nan if L<=0."""
    if length <= 0.0 or volume <= 0.0:
        return float("nan")
    return float(np.sqrt(volume / length))


def tree_potential_density(rho: np.ndarray, mphi: float, lam: float) -> np.ndarray:
    """V_tree(|Φ|) with vacuum subtracted: V(|Φ|) − V(v), v² = m²/λ."""
    r2 = np.asarray(rho, dtype=np.float64) ** 2
    v2 = (mphi * mphi) / max(lam, 1e-30)
    # V = (λ/4) r^4 − (m²/2) r² ; V(v) = − m⁴/(4λ)
    V = 0.25 * lam * r2 * r2 - 0.5 * mphi * mphi * r2
    Vmin = -0.25 * (mphi**4) / max(lam, 1e-30)
    return V - Vmin


def _phidot_from_pi(
    pi_prog: np.ndarray,
    f_star: float,
    omega_star: float,
    a: float,
) -> np.ndarray:
    """Physical ∂_t φ from CosmoLattice program π.

    Convention (see cosmolattice_ext_v2 README): 
    ``π = a³ φ̇_tilde / ω_*`` with ``φ_tilde = φ_phys / f_*``,
    so ``φ̇_phys = π * f_* * ω_* / a³``.
    """
    a = max(float(a), 1e-30)
    scale = float(f_star) * float(omega_star) / (a ** 3)
    return np.asarray(pi_prog, dtype=np.float32) * np.float32(scale)


def sum_kinetic_on_mask(
    pi1: np.ndarray,
    pi2: np.ndarray,
    mask: np.ndarray,
    f_star: float,
    omega_star: float,
    a: float,
    dV: float,
) -> float:
    """∫ ½(φ̇₁²+φ̇₂²) dV over mask."""
    phidot1 = _phidot_from_pi(pi1, f_star, omega_star, a)
    e = 0.5 * np.asarray(phidot1, dtype=np.float64) ** 2
    del phidot1
    phidot2 = _phidot_from_pi(pi2, f_star, omega_star, a)
    e += 0.5 * np.asarray(phidot2, dtype=np.float64) ** 2
    del phidot2
    return float(e[mask].sum() * dV)


def sum_gradient_on_mask(
    phi1: np.ndarray,
    phi2: np.ndarray,
    mask: np.ndarray,
    dx_com: float,
    a: float,
    dV: float,
) -> float:
    """∫ ½ |∇φ|² / a² dV over mask (physical spatial gradient, FRW).

    Forward differences, one axis at a time (peak +~1 field of RAM).
    """
    phi1 = _as_f32(phi1)
    phi2 = _as_f32(phi2)
    inv_dx2 = 1.0 / (float(dx_com) ** 2)
    pref = 0.5 / (max(float(a), 1e-30) ** 2)
    total = 0.0
    for ax in range(3):
        d1 = np.roll(phi1, -1, axis=ax) - phi1
        d2 = np.roll(phi2, -1, axis=ax) - phi2
        # e_dens contribution for this axis (float64 accumulate on mask only)
        dens = (np.asarray(d1, dtype=np.float64) ** 2
                + np.asarray(d2, dtype=np.float64) ** 2)
        del d1, d2
        total += float(dens[mask].sum())
        del dens
    return pref * inv_dx2 * total * dV


def sum_tree_pot_on_mask(
    rho: np.ndarray,
    mask: np.ndarray,
    mphi: float,
    lam: float,
    dV: float,
) -> float:
    """∫ [V_tree(|Φ|)−V_min] dV on mask — tree only, not full thermal V."""
    rho_c = np.asarray(rho, dtype=np.float64)[mask]
    if rho_c.size == 0:
        return 0.0
    return float(tree_potential_density(rho_c, mphi, lam).sum() * dV)


def velocity_estimator(
    e_kin: float,
    e_grad: float,
) -> float:
    """Lagrangian-style ⟨v²⟩ ≈ E_kin / (E_kin + E_grad) on cores."""
    denom = e_kin + e_grad
    if denom <= 0.0:
        return float("nan")
    return float(e_kin / denom)


def loop_length_stats(
    winding: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 20,
    max_label_voxels: int = 32_000_000,
) -> Dict[str, Any]:
    """Connected-component census; skipped if too many string voxels."""
    w = np.asarray(winding)
    n_vox = int(np.sum(np.abs(w) > threshold))
    out: Dict[str, Any] = {
        "n_loops": -1,
        "top_loop_voxels": "",
        "long_string_voxel_fraction": float("nan"),
        "max_loop_voxels": 0,
    }
    if n_vox == 0:
        out["n_loops"] = 0
        out["long_string_voxel_fraction"] = float("nan")
        return out
    if n_vox > max_label_voxels:
        return out

    from scipy.ndimage import label as ndimage_label

    mask = np.abs(w) > threshold
    labelled, n_loops = ndimage_label(mask)
    if n_loops == 0:
        out["n_loops"] = 0
        return out
    counts = np.bincount(labelled.ravel(), minlength=n_loops + 1)[1:]
    order = np.argsort(counts)[::-1]
    top = counts[order[:top_n]]
    # "Long" ≈ components longer than mean (rough formation diagnostic)
    mean_len = float(counts.mean())
    long_frac = float(counts[counts >= max(mean_len, 1.0)].sum() / max(n_vox, 1))
    out["n_loops"] = int(n_loops)
    out["top_loop_voxels"] = ";".join(str(int(x)) for x in top)
    out["long_string_voxel_fraction"] = long_frac
    out["max_loop_voxels"] = int(counts.max())
    del labelled
    return out


def resolve_lattice_scales(params: Optional[Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, float]:
    """dx_com, omegaStar, mphi, lam, fStar, a from run params + manifest row."""
    params = params or {}
    f_star = float(row.get("fStar", params.get("fStar", params.get("f_star", 1.0))))
    a = float(row.get("a", 1.0))
    mphi = float(params.get("mphi", params.get("mu", _DEFAULT_MPHI)))
    omega = float(params.get("omegaStar", params.get("omega_star", mphi)))
    dx_phys = float(params.get("dx_phys", 1e-3))
    # CosmoLattice program dx = omegaStar * dx_phys; comoving physical dx = dx_phys
    # (fields already converted to GeV via fStar). Use physical comoving spacing.
    dx_com = dx_phys
    lam = float(params.get("lambda", params.get("lam", _DEFAULT_LAM)))
    # Some JSON stores lambdaSix / epsilon; tree quartic often "lambda"
    if "lambda" not in params and "lam" not in params:
        # set8-style: try nested or gamma sets
        for k in ("lam_phi", "lambda_phi"):
            if k in params:
                lam = float(params[k])
                break
    return {
        "dx_com": dx_com,
        "dx_phys": dx_phys,
        "omega_star": omega,
        "mphi": mphi,
        "lam": lam,
        "f_star": f_star,
        "a": a,
    }


def compute_network_metrics(
    snap: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    *,
    threshold: float = 0.5,
    do_loops: bool = True,
    max_label_voxels: int = 32_000_000,
    top_n_loops: int = 20,
) -> Dict[str, Any]:
    """Compute network metrics for one snapshot dict from ``read_h5_snapshot``.

    Expected keys: phi1, phi2, rho, winding, N, a, H, time, temperature, fStar.
    Optional: pi1, pi2 (program units) for kinetic / velocity.
    """
    scales = resolve_lattice_scales(params, {
        "fStar": snap.get("fStar", 1.0),
        "a": snap.get("a", 1.0),
    })
    winding = snap["winding"]
    N = int(snap.get("N", winding.shape[0]))
    dx = scales["dx_com"]
    a = float(snap.get("a", scales["a"]))
    H = float(snap.get("H", 0.0))
    t = float(snap.get("time", snap.get("t", 0.0)))
    dV = dx ** 3
    volume = (N * dx) ** 3

    n_vox = int(np.sum(np.abs(winding) > threshold))
    L_w, L_vox = comoving_string_length(winding, dx, threshold=threshold)
    xi_w = mean_separation(volume, L_w)
    xi_vox = mean_separation(volume, L_vox)
    L_phys = L_w * a  # physical length if a is scale factor

    out: Dict[str, Any] = {
        "step": int(snap["step"]),
        "time": t,
        "temperature": float(snap["temperature"]),
        "a": a,
        "H": H,
        "N": N,
        "n_string_voxels": n_vox,
        "string_voxel_fraction": float(n_vox) / float(N ** 3),
        "L_comoving": L_w,
        "L_voxel": L_vox,
        "L_physical": L_phys,
        "xi_comoving": xi_w,
        "xi_voxel": xi_vox,
        "xi_over_t": (xi_w / t) if t > 0 else float("nan"),
        "xi_H": (xi_w * H) if H > 0 else float("nan"),
        "E_kin_core": float("nan"),
        "E_grad_core": float("nan"),
        "E_pot_core": float("nan"),
        "E_tot_core": float("nan"),
        "mu_eff": float("nan"),
        "v2_mean": float("nan"),
        "n_loops": "",
        "top_loop_voxels": "",
        "long_string_voxel_fraction": float("nan"),
        "max_loop_voxels": "",
        "has_pi": False,
    }

    mask = string_core_mask(winding, threshold)
    if n_vox > 0:
        e_grad = sum_gradient_on_mask(
            snap["phi1"], snap["phi2"], mask, dx, a, dV
        )
        e_pot = sum_tree_pot_on_mask(
            snap["rho"], mask, scales["mphi"], scales["lam"], dV
        )
        e_kin = float("nan")
        v2 = float("nan")
        if snap.get("pi1") is not None and snap.get("pi2") is not None:
            out["has_pi"] = True
            e_kin = sum_kinetic_on_mask(
                snap["pi1"],
                snap["pi2"],
                mask,
                scales["f_star"],
                scales["omega_star"],
                a,
                dV,
            )
            v2 = velocity_estimator(e_kin, e_grad)
        e_tot = e_grad + e_pot + (0.0 if not np.isfinite(e_kin) else e_kin)
        out["E_grad_core"] = e_grad
        out["E_pot_core"] = e_pot
        out["E_kin_core"] = e_kin
        out["E_tot_core"] = e_tot
        out["mu_eff"] = (e_tot / L_w) if L_w > 0 else float("nan")
        out["v2_mean"] = v2

    if do_loops:
        loops = loop_length_stats(
            winding,
            threshold=threshold,
            top_n=top_n_loops,
            max_label_voxels=max_label_voxels,
        )
        out["n_loops"] = loops["n_loops"]
        out["top_loop_voxels"] = loops["top_loop_voxels"]
        out["long_string_voxel_fraction"] = loops["long_string_voxel_fraction"]
        out["max_loop_voxels"] = loops["max_loop_voxels"]

    return out


NETWORK_CSV_FIELDS: Sequence[str] = (
    "step",
    "time",
    "temperature",
    "a",
    "H",
    "n_string_voxels",
    "string_voxel_fraction",
    "n_loops",
    "L_comoving",
    "L_voxel",
    "L_physical",
    "xi_comoving",
    "xi_voxel",
    "xi_over_t",
    "xi_H",
    "E_kin_core",
    "E_grad_core",
    "E_pot_core",
    "E_tot_core",
    "mu_eff",
    "v2_mean",
    "long_string_voxel_fraction",
    "max_loop_voxels",
    "top_loop_voxels",
    "has_pi",
)


def plot_network_timeseries(
    csv_path: str,
    out_path: str,
    *,
    title: Optional[str] = None,
) -> str:
    """Plot network observables vs time from ``string_summary.csv``."""
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"no rows in {csv_path}")

    def col(name: str) -> np.ndarray:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(name, "nan")))
            except (TypeError, ValueError):
                vals.append(np.nan)
        return np.asarray(vals, dtype=np.float64)

    t = col("time")
    order = np.argsort(t)
    t = t[order]

    def y(name: str) -> np.ndarray:
        return col(name)[order]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=12)

    ax = axes[0, 0]
    ax.plot(t, y("L_comoving"), "C0-", lw=1.5, label=r"$L$ (winding)")
    ax.plot(t, y("L_voxel"), "C0--", lw=1.0, alpha=0.7, label=r"$L$ (voxels)")
    ax.set_ylabel(r"$L$ comoving")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, y("xi_comoving"), "C1-", lw=1.5, label=r"$\xi$")
    ax.set_ylabel(r"$\xi=\sqrt{V/L}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, y("xi_over_t"), "C2-", lw=1.5, label=r"$\xi/t$")
    ax.plot(t, y("xi_H"), "C2--", lw=1.0, label=r"$\xi H$")
    ax.set_ylabel("scaling proxies")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, y("E_grad_core"), "C3-", lw=1.2, label=r"$E_\mathrm{grad}$")
    ax.plot(t, y("E_pot_core"), "C4-", lw=1.2, label=r"$E_\mathrm{pot}^\mathrm{tree}$")
    ekin = y("E_kin_core")
    if np.isfinite(ekin).any():
        ax.plot(t, ekin, "C5-", lw=1.2, label=r"$E_\mathrm{kin}$")
    ax.plot(t, y("E_tot_core"), "k-", lw=1.4, label=r"$E_\mathrm{tot}$")
    ax.set_ylabel("core energy")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, y("mu_eff"), "C6-", lw=1.5, label=r"$\mu_\mathrm{eff}=E/L$")
    ax.set_ylabel(r"$\mu_\mathrm{eff}$")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    v2 = y("v2_mean")
    nloops = y("n_loops")
    ax.plot(t, v2, "C7-", lw=1.5, label=r"$\langle v^2\rangle$")
    ax.set_ylabel(r"$\langle v^2\rangle$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    # only plot finite loop counts (>=0)
    ok = np.isfinite(nloops) & (nloops >= 0)
    if ok.any():
        ax2.plot(t[ok], nloops[ok], "C8--", lw=1.0, label=r"$N_\mathrm{loops}$")
        ax2.set_ylabel(r"$N_\mathrm{loops}$")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

    for ax in axes[1, :]:
        ax.set_xlabel(r"$t$ (program)")
    for ax in axes[0, :]:
        ax.set_xlabel(r"$t$ (program)")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
