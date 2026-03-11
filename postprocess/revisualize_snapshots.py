#!/usr/bin/env python
"""
Revisualize saved field states with FIXED colorbar settings.

IMPORTANT: Uses the SAME colorbar range for ALL snapshots!
This is critical for detecting rare bubble nucleation events.

Usage:
    python postprocess/revisualize_snapshots.py <simulation_directory>

Example:
    python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350_dt_1e-05_a_0.001_interval_500000
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection
from scipy.ndimage import label as ndimage_label
import glob
import os
import re
import sys


def load_metadata(sim_dir):
    """Load simulation metadata, searching parent directories if needed."""
    # Search current dir and up to 3 parent levels (handles detailed_replay subdirs)
    search_dir = os.path.abspath(sim_dir)
    for _ in range(4):
        metadata_file = os.path.join(search_dir, "simulation_metadata.npz")
        if os.path.exists(metadata_file):
            print(f"  Metadata found: {metadata_file}")
            return np.load(metadata_file)
        search_dir = os.path.dirname(search_dir)

    print(f"Warning: Metadata file not found (searched up to 3 parent dirs)")
    print(f"  Using hardcoded physics defaults: mphi=1000, lam=1e-16")
    return None


def _extract_step_from_filename(path):
    """Extract step number from state_step_NNNN.npz filename."""
    base = os.path.basename(path)
    m = re.search(r"state_step_(\d+)", base)
    return int(m.group(1)) if m else None


def _filter_by_step_range(state_files, step_min=None, step_max=None):
    """Keep only files whose step number is within [step_min, step_max]."""
    if step_min is None and step_max is None:
        return state_files
    filtered = []
    for f in state_files:
        s = _extract_step_from_filename(f)
        if s is None:
            continue
        if step_min is not None and s < step_min:
            continue
        if step_max is not None and s > step_max:
            continue
        filtered.append(f)
    return filtered


def load_field_state(state_file):
    """Load a single field state (supports both real-field and complex-field snapshots)."""
    data = np.load(state_file)
    is_complex = "phi1" in data

    if is_complex:
        phi1 = data["phi1"]
        phi2 = data["phi2"]
        rho = data["rho"] if "rho" in data else np.sqrt(phi1.astype(np.float64)**2 + phi2.astype(np.float64)**2).astype(np.float32)
        theta = data["theta"] if "theta" in data else np.arctan2(phi2.astype(np.float64), phi1.astype(np.float64)).astype(np.float32)
        winding = data["winding"] if "winding" in data else None
        return {
            "complex": True,
            "phi1": phi1, "phi2": phi2,
            "rho": rho, "theta": theta,
            "winding": winding,
            "phi": rho,  # backward compat: use rho as the scalar field
            "pi": None,
            "step": int(data["step"]),
            "time": float(data["time"]),
            "temperature": float(data["temperature"]),
            "phi_min": float(data["rho_min"]) if "rho_min" in data else float(rho.min()),
            "phi_max": float(data["rho_max"]) if "rho_max" in data else float(rho.max()),
            "zn_active": float(data["zn_active"]) if "zn_active" in data else 0.0,
        }

    return {
        "complex": False,
        "phi": data["phi"],
        "pi": data["pi"] if "pi" in data else None,
        "step": int(data["step"]),
        "time": float(data["time"]),
        "temperature": float(data["temperature"]),
        "phi_min": (
            float(data["phi_min"]) if "phi_min" in data else float(data["phi"].min())
        ),
        "phi_max": (
            float(data["phi_max"]) if "phi_max" in data else float(data["phi"].max())
        ),
    }


def _plot_2d_slice(
    ax, phi_2d, vmin, vmax, cmap, cbar_label, fig, escape_phi=None, phi_raw=None
):
    """Helper: draw a single 2D slice with colorbar.

    If escape_phi is set, sites with |phi| > escape_phi are overlaid
    in solid red (positive) or blue (negative).
    """
    im = ax.imshow(phi_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    if escape_phi is not None and phi_raw is not None:
        ny_s, nx_s = phi_raw.shape
        overlay = np.zeros((ny_s, nx_s, 4), dtype=np.float32)
        mask_pos = phi_raw > escape_phi
        mask_neg = phi_raw < -escape_phi
        overlay[mask_pos] = [1.0, 0.0, 0.0, 1.0]  # red
        overlay[mask_neg] = [0.0, 0.0, 1.0, 1.0]  # blue
        if np.any(mask_pos) or np.any(mask_neg):
            ax.imshow(overlay, origin="lower", extent=im.get_extent())

    return im


def _find_surface_voxels(mask):
    """Return indices of True voxels that border at least one False neighbour."""
    nx, ny, nz = mask.shape
    interior = (
        mask[:-2, 1:-1, 1:-1]
        & mask[2:, 1:-1, 1:-1]
        & mask[1:-1, :-2, 1:-1]
        & mask[1:-1, 2:, 1:-1]
        & mask[1:-1, 1:-1, :-2]
        & mask[1:-1, 1:-1, 2:]
    )
    surface = mask.copy()
    surface[1:-1, 1:-1, 1:-1] &= ~interior
    return np.argwhere(surface)


def _identify_bubbles(mask, phi, dx_phys=None):
    """Label connected components in *mask* and measure each bubble.

    Returns a list of dicts sorted by volume (largest first):
        centroid : (cx, cy, cz) in lattice coordinates
        volume   : number of voxels
        r_eff    : effective radius  (3V/4π)^{1/3}  in lattice sites
        r_phys   : effective radius in physical units (if dx_phys given)
        r_rms    : RMS distance of voxels from centroid (lattice sites)
        sign     : +1 or -1 (dominant sign of phi inside the bubble)
        peak     : max |phi| inside the bubble
    """
    labelled, n_clusters = ndimage_label(mask)
    if n_clusters == 0:
        return []

    bubbles = []
    for cid in range(1, n_clusters + 1):
        coords = np.argwhere(labelled == cid)
        vol = len(coords)
        centroid = coords.mean(axis=0)

        r_eff = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        r_rms = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

        phi_vals = phi[labelled == cid]
        sign = 1 if np.mean(phi_vals) >= 0 else -1
        peak = float(np.max(np.abs(phi_vals)))

        entry = dict(
            centroid=centroid,
            volume=vol,
            r_eff=r_eff,
            r_rms=r_rms,
            sign=sign,
            peak=peak,
        )
        if dx_phys is not None:
            entry["r_phys"] = r_eff * dx_phys
        bubbles.append(entry)

    bubbles.sort(key=lambda b: b["volume"], reverse=True)
    return bubbles


def plot_3d_bubbles(
    phi,
    metadata,
    state_info,
    output_file,
    escape_phi,
    elev=25,
    azim=135,
    max_points=200_000,
    surface_only=True,
    dx_phys=None,
):
    """Render 3D bubble surfaces as a matplotlib scatter plot.

    Positive-excursion sites (phi > escape_phi) are shown in red,
    negative (phi < -escape_phi) in blue.  Only surface voxels
    (those touching a below-threshold neighbour) are drawn by default.
    """
    if phi.ndim != 3:
        print(f"  [view3d] Skipping non-3D field (ndim={phi.ndim})")
        return None

    if metadata is not None and "vev" in metadata:
        mu = float(metadata["mu"])
    elif metadata is not None and "mphi" in metadata:
        mu = float(metadata["mphi"])
    else:
        mu = 1000.0

    if dx_phys is None and metadata is not None and "dx_phys" in metadata:
        dx_phys = float(metadata["dx_phys"])

    mask_pos = phi > escape_phi
    mask_neg = phi < -escape_phi
    mask_any = mask_pos | mask_neg

    bubbles = _identify_bubbles(mask_any, phi, dx_phys=dx_phys)
    n_pos = int(np.sum(mask_pos))
    n_neg = int(np.sum(mask_neg))

    step = state_info["step"]
    time_val = state_info["time"] / mu
    T_val = state_info["temperature"]
    n_bub = len(bubbles)
    nx, ny, nz = phi.shape

    if surface_only:
        pts_pos = _find_surface_voxels(mask_pos)
        pts_neg = _find_surface_voxels(mask_neg)
    else:
        pts_pos = np.argwhere(mask_pos)
        pts_neg = np.argwhere(mask_neg)

    if len(pts_pos) > max_points:
        idx = np.random.choice(len(pts_pos), max_points, replace=False)
        pts_pos = pts_pos[idx]
    if len(pts_neg) > max_points:
        idx = np.random.choice(len(pts_neg), max_points, replace=False)
        pts_neg = pts_neg[idx]

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Adaptive marker size: larger when few voxels, still visible when many
    total_pts = n_pos + n_neg
    marker_size = max(2.0, min(20.0, 2000.0 / max(1, total_pts)))

    if len(pts_pos) > 0:
        ax.scatter(
            pts_pos[:, 0],
            pts_pos[:, 1],
            pts_pos[:, 2],
            c="red",
            s=marker_size,
            alpha=0.7,
            linewidths=0,
            label=f"+bubble ({n_pos} vox)",
            depthshade=True,
        )
    if len(pts_neg) > 0:
        ax.scatter(
            pts_neg[:, 0],
            pts_neg[:, 1],
            pts_neg[:, 2],
            c="blue",
            s=marker_size,
            alpha=0.7,
            linewidths=0,
            label=f"−bubble ({n_neg} vox)",
            depthshade=True,
        )

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.set_title(
        f"3D bubbles  |  Step {step:,}  |  t={time_val:.2e}  |  T={T_val:.1f}\n"
        f"|φ| > {escape_phi:.1f}   clusters: {n_bub}  "
        f"voxels(+{n_pos}/−{n_neg})",
        fontsize=10,
        pad=12,
    )
    if n_pos > 0 or n_neg > 0:
        ax.legend(loc="upper right", fontsize=8, markerscale=3)

    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return bubbles


def _write_bubble_csv(csv_path, bubbles, step, time_val, temperature, dx_phys):
    """Write per-step CSV with one row per bubble."""
    with open(csv_path, "w") as f:
        has_phys = dx_phys is not None
        hdr = "bubble_id,sign,centroid_x,centroid_y,centroid_z,volume,r_eff,r_rms"
        if has_phys:
            hdr += ",r_phys,r_rms_phys"
        hdr += ",peak,step,time,temperature\n"
        f.write(hdr)
        for bi, b in enumerate(bubbles):
            cx, cy, cz = b["centroid"]
            sign_str = "+" if b["sign"] > 0 else "-"
            row = (
                f"{bi+1},{sign_str},{cx:.2f},{cy:.2f},{cz:.2f},"
                f"{b['volume']},{b['r_eff']:.4f},{b['r_rms']:.4f}"
            )
            if has_phys:
                row += f",{b['r_eff'] * dx_phys:.6e},{b['r_rms'] * dx_phys:.6e}"
            row += f",{b['peak']:.4e},{step},{time_val:.8e},{temperature:.2f}\n"
            f.write(row)


# =====================================================================
# Cosmic string detection and visualization
# =====================================================================

def _identify_strings(winding, threshold=0.5):
    """Label connected string voxels (|winding| > threshold) and measure each loop.

    Returns (labelled_array, list_of_string_dicts) sorted by length (longest first).
    Each dict: loop_id, n_voxels, centroid, winding_sign, max_winding.
    """
    mask = np.abs(winding) > threshold
    labelled, n_loops = ndimage_label(mask)
    if n_loops == 0:
        return labelled, []

    strings = []
    for lid in range(1, n_loops + 1):
        coords = np.argwhere(labelled == lid)
        n_vox = len(coords)
        centroid = coords.mean(axis=0)
        w_vals = winding[labelled == lid]
        sign = 1 if np.mean(w_vals) >= 0 else -1
        strings.append(dict(
            loop_id=lid,
            n_voxels=n_vox,
            centroid=centroid,
            winding_sign=sign,
            max_winding=float(np.max(np.abs(w_vals))),
        ))

    strings.sort(key=lambda s: s["n_voxels"], reverse=True)
    return labelled, strings


def plot_strings_2d(state, metadata, output_file):
    """6-panel 2D visualization of complex field with cosmic strings highlighted.

    Panels: rho, theta, winding, string loops (colored by ID),
            rho histogram, and string length bar chart.
    """
    rho = np.asarray(state["rho"], dtype=np.float64)
    theta = np.asarray(state["theta"], dtype=np.float64)
    winding = np.asarray(state["winding"], dtype=np.float64)
    step = state["step"]
    T_val = state["temperature"]

    mu = 1000.0
    if metadata is not None:
        if "mu" in metadata:
            mu = float(metadata["mu"])
        elif "mphi" in metadata:
            mu = float(metadata["mphi"])
    time_phys = state["time"] / mu

    nx, ny, nz = rho.shape
    zmid = nz // 2

    labelled, strings = _identify_strings(winding)
    n_total_vox = int(np.sum(np.abs(winding) > 0.5))
    n_loops = len(strings)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 0: rho slice
    rho_sl = rho[:, :, zmid]
    im0 = axes[0, 0].imshow(rho_sl.T, origin="lower", cmap="viridis")
    axes[0, 0].set_title(r"$\rho = |\Phi|$  (z-midplane)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # Panel 1: theta slice
    theta_sl = theta[:, :, zmid]
    im1 = axes[0, 1].imshow(theta_sl.T, origin="lower", cmap="hsv",
                             vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title(r"$\theta = \arg(\Phi)$  (z-midplane)")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # Panel 2: winding number slice
    wind_sl = winding[:, :, zmid]
    wmax = max(float(np.max(np.abs(wind_sl))), 0.1)
    im2 = axes[0, 2].imshow(wind_sl.T, origin="lower", cmap="RdBu_r",
                             vmin=-wmax, vmax=wmax)
    axes[0, 2].set_title(f"Winding  (|W|>0.5 total: {n_total_vox})")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # Panel 3: string loops colored by ID (z-midplane)
    loop_sl = labelled[:, :, zmid]
    loop_display = np.zeros((nx, ny, 4), dtype=np.float32)
    if n_loops > 0:
        loop_cmap = plt.cm.get_cmap("tab20", max(n_loops, 1))
        for si, s_info in enumerate(strings):
            lid = s_info["loop_id"]
            mask_2d = loop_sl == lid
            if np.any(mask_2d):
                c = loop_cmap(si % 20)
                loop_display[mask_2d] = c
    axes[1, 0].imshow(np.zeros((nx, ny)), origin="lower", cmap="gray_r",
                      vmin=0, vmax=1, alpha=0.3)
    axes[1, 0].imshow(np.transpose(loop_display, (1, 0, 2)), origin="lower")
    axes[1, 0].set_title(f"String loops by ID  ({n_loops} distinct loops)")

    # Panel 4: rho histogram
    rho_flat = rho.flatten()
    axes[1, 1].hist(rho_flat, bins=100, density=True, color="steelblue", alpha=0.8)
    axes[1, 1].set_xlabel(r"$\rho = |\Phi|$")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title(r"$\rho$ histogram")

    # Panel 5: string length bar chart (top 20)
    if n_loops > 0:
        top_n = min(20, n_loops)
        top_strings = strings[:top_n]
        loop_ids = [f"#{s['loop_id']}" for s in top_strings]
        lengths = [s["n_voxels"] for s in top_strings]
        bar_colors = [plt.cm.tab20(i % 20) for i in range(top_n)]
        axes[1, 2].barh(loop_ids[::-1], lengths[::-1], color=bar_colors[::-1])
        axes[1, 2].set_xlabel("Voxels (string length)")
        axes[1, 2].set_title(f"Top {top_n} string loops")
    else:
        axes[1, 2].text(0.5, 0.5, "No strings detected", ha="center", va="center",
                        transform=axes[1, 2].transAxes, fontsize=14, color="gray")
        axes[1, 2].set_title("String lengths")

    fig.suptitle(
        f"Step {step:,} | t={time_phys:.4e} | T={T_val:.1f} | "
        f"Strings: {n_loops} loops, {n_total_vox} voxels",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return strings


def plot_strings_3d(state, metadata, output_file,
                    elev=25, azim=135, max_points=300_000):
    """3D scatter plot of cosmic string voxels, colored by loop ID."""
    winding = np.asarray(state["winding"], dtype=np.float64)
    step = state["step"]
    T_val = state["temperature"]

    mu = 1000.0
    if metadata is not None:
        if "mu" in metadata:
            mu = float(metadata["mu"])
        elif "mphi" in metadata:
            mu = float(metadata["mphi"])
    time_phys = state["time"] / mu

    nx, ny, nz = winding.shape
    labelled, strings = _identify_strings(winding)
    n_loops = len(strings)
    n_total_vox = int(np.sum(np.abs(winding) > 0.5))

    if n_loops == 0:
        print(f"    [3d-strings] No strings at step {step}")
        return strings

    loop_cmap = plt.cm.get_cmap("tab20", max(n_loops, 1))

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    total_pts_plotted = 0
    for si, s_info in enumerate(strings):
        lid = s_info["loop_id"]
        coords = np.argwhere(labelled == lid)
        n_vox = len(coords)

        if total_pts_plotted + n_vox > max_points and total_pts_plotted > 0:
            frac = max(0.1, (max_points - total_pts_plotted) / n_vox)
            idx = np.random.choice(n_vox, int(n_vox * frac), replace=False)
            coords = coords[idx]

        col = loop_cmap(si % 20)
        ms = max(1.0, min(8.0, 5000.0 / max(1, n_total_vox)))
        sign_str = "+" if s_info["winding_sign"] > 0 else "−"
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=[col], s=ms, alpha=0.8, linewidths=0, depthshade=True,
            label=f"#{lid} ({sign_str}, {n_vox} vox)" if si < 10 else None,
        )
        total_pts_plotted += len(coords)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([nx, ny, nz])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"Cosmic Strings (3D) | Step {step:,} | t={time_phys:.2e} | T={T_val:.1f}\n"
        f"{n_loops} loops, {n_total_vox} voxels",
        fontsize=11, pad=12)
    if n_loops <= 10:
        ax.legend(loc="upper right", fontsize=7, markerscale=4)

    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return strings


def _write_string_csv(csv_path, strings, step, time_val, temperature):
    """Write per-step CSV with one row per string loop."""
    with open(csv_path, "w") as f:
        f.write("loop_id,winding_sign,n_voxels,centroid_x,centroid_y,centroid_z,"
                "max_winding,step,time,temperature\n")
        for s in strings:
            cx, cy, cz = s["centroid"]
            sign_str = "+" if s["winding_sign"] > 0 else "-"
            f.write(f"{s['loop_id']},{sign_str},{s['n_voxels']},"
                    f"{cx:.2f},{cy:.2f},{cz:.2f},{s['max_winding']:.4f},"
                    f"{step},{time_val:.8e},{temperature:.2f}\n")


def revisualize_strings(sim_dir, elev=25, azim=135, step_min=None, step_max=None):
    """Generate 2D and 3D cosmic string visualizations for every snapshot."""
    metadata = load_metadata(sim_dir)

    state_dir = os.path.join(sim_dir, "field_states")
    if not os.path.exists(state_dir):
        print(f"Error: Field states directory not found: {state_dir}")
        return

    state_files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    state_files = [f for f in state_files if "_NaN_debug" not in f]
    state_files = _filter_by_step_range(state_files, step_min, step_max)
    if not state_files:
        print(f"Error: No field state files found in {state_dir}")
        return

    output_dir = os.path.join(sim_dir, "strings")
    output_3d = os.path.join(output_dir, "3d")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_3d, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Cosmic string visualization")
    print(f"  Snapshots : {len(state_files)}")
    print(f"  Output 2D : {output_dir}")
    print(f"  Output 3D : {output_3d}")
    print(f"  CSV       : {csv_dir}")
    print()

    summary_path = os.path.join(output_dir, "string_summary.csv")
    summary_rows = []

    for i, state_file in enumerate(state_files):
        state = load_field_state(state_file)
        step = state["step"]

        if not state.get("complex", False) or state.get("winding") is None:
            if i == 0:
                print("  Warning: snapshot has no winding data (not a complex-field run)")
            continue

        winding = np.asarray(state["winding"], dtype=np.float64)
        n_string_vox = int(np.sum(np.abs(winding) > 0.5))

        # 2D panel
        out_2d = os.path.join(output_dir, f"strings_step_{step:010d}.png")
        strings = plot_strings_2d(state, metadata, out_2d)
        n_loops = len(strings)

        # 3D view (only if strings exist)
        if n_loops > 0:
            out_3d = os.path.join(output_3d, f"strings3d_step_{step:010d}.png")
            plot_strings_3d(state, metadata, out_3d, elev=elev, azim=azim)

            csv_path = os.path.join(csv_dir, f"strings_step_{step:010d}.csv")
            mu = 1000.0
            if metadata is not None and "mu" in metadata:
                mu = float(metadata["mu"])
            elif metadata is not None and "mphi" in metadata:
                mu = float(metadata["mphi"])
            _write_string_csv(csv_path, strings, step, state["time"],
                              state["temperature"])

        top_lens = [s["n_voxels"] for s in strings[:3]] if strings else []
        top_str = ", ".join(str(v) for v in top_lens) if top_lens else "-"
        summary_rows.append(
            f"{step},{state['time']:.8e},{state['temperature']:.2f},"
            f"{n_loops},{n_string_vox},{top_str}\n")

        print(f"  [{i+1}/{len(state_files)}] step {step:>10,}  "
              f"loops={n_loops:>4d}  voxels={n_string_vox:>7d}")

    with open(summary_path, "w") as f:
        f.write("step,time,temperature,n_loops,n_string_voxels,top_loop_sizes\n")
        for row in summary_rows:
            f.write(row)

    print(f"\nDone – string visualizations saved to {output_dir}")
    print(f"  Summary CSV : {summary_path}")


def revisualize_3d(
    sim_dir,
    escape_phi,
    elev=25,
    azim=135,
    surface_only=True,
    step_min=None,
    step_max=None,
):
    """Generate 3D bubble renders for every snapshot that has excursions."""
    metadata = load_metadata(sim_dir)

    # Extract dx_phys once so it's available for console output and CSV
    dx_phys = None
    if metadata is not None and "dx_phys" in metadata:
        dx_phys = float(metadata["dx_phys"])

    state_dir = os.path.join(sim_dir, "field_states")
    if not os.path.exists(state_dir):
        print(f"Error: Field states directory not found: {state_dir}")
        return

    state_files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    state_files = [f for f in state_files if "_NaN_debug" not in f]
    state_files = _filter_by_step_range(state_files, step_min, step_max)
    if len(state_files) == 0:
        print(f"Error: No field state files found in {state_dir}")
        return

    output_dir = os.path.join(sim_dir, "view3d")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"3D bubble visualization")
    print(f"  Threshold : |phi| > {escape_phi:.1f}")
    print(f"  Surface   : {'on' if surface_only else 'off'}")
    print(
        f"  dx_phys   : {dx_phys}" if dx_phys else "  dx_phys   : not found in metadata"
    )
    print(f"  View      : elev={elev}, azim={azim}")
    print(f"  Snapshots : {len(state_files)}")
    print(f"  Output    : {output_dir}")
    print(f"  CSV       : {csv_dir}")
    print()

    # Summary CSV: one row per snapshot that has bubbles
    summary_path = os.path.join(output_dir, "bubble_summary.csv")
    summary_hdr = "step,time,temperature,n_bubbles,n_pos,n_neg"
    if dx_phys is not None:
        summary_hdr += ",r_max_phys,r_mean_phys"
    summary_hdr += ",r_max_lattice,r_mean_lattice,total_volume\n"
    summary_rows = []

    rendered = 0
    for i, state_file in enumerate(state_files):
        state = load_field_state(state_file)
        phi = state["phi"]

        has_bubbles = np.any(np.abs(phi) > escape_phi)
        if not has_bubbles:
            continue

        out_path = os.path.join(output_dir, f"bubble3d_step_{state['step']:010d}.png")
        bubbles = plot_3d_bubbles(
            phi,
            metadata,
            state,
            out_path,
            escape_phi=escape_phi,
            elev=elev,
            azim=azim,
            surface_only=surface_only,
            dx_phys=dx_phys,
        )
        rendered += 1

        if bubbles is None:
            bubbles = []
        n_bub = len(bubbles)

        # --- Per-step CSV ---
        if n_bub > 0:
            csv_path = os.path.join(csv_dir, f"bubbles_step_{state['step']:010d}.csv")
            _write_bubble_csv(
                csv_path,
                bubbles,
                state["step"],
                state["time"],
                state["temperature"],
                dx_phys,
            )

        # --- Summary row ---
        if n_bub > 0:
            n_pos_bub = sum(1 for b in bubbles if b["sign"] > 0)
            n_neg_bub = n_bub - n_pos_bub
            r_all = [b["r_eff"] for b in bubbles]
            r_max = max(r_all)
            r_mean = np.mean(r_all)
            tot_vol = sum(b["volume"] for b in bubbles)
            row = (
                f"{state['step']},{state['time']:.8e},{state['temperature']:.2f},"
                f"{n_bub},{n_pos_bub},{n_neg_bub}"
            )
            if dx_phys is not None:
                row += f",{r_max * dx_phys:.6e},{r_mean * dx_phys:.6e}"
            row += f",{r_max:.4f},{r_mean:.4f},{tot_vol}\n"
            summary_rows.append(row)

        # --- Console output ---
        bub_info = ""
        if n_bub > 0:
            top = bubbles[:3]
            parts = []
            for b in top:
                s = "+" if b["sign"] > 0 else "-"
                r_part = f"r={b['r_eff']:.1f}"
                if dx_phys is not None:
                    r_part += f"({b['r_eff'] * dx_phys:.2e})"
                parts.append(f"{s}{r_part}")
            bub_info = f"  {n_bub} clusters: {', '.join(parts)}"

        print(f"  [{i+1}/{len(state_files)}] step {state['step']:>10,}{bub_info}")

    # Write summary CSV
    with open(summary_path, "w") as f:
        f.write(summary_hdr)
        for row in summary_rows:
            f.write(row)

    print(f"\nDone – {rendered} 3D frames saved to {output_dir}")
    print(f"  Per-step CSV : {csv_dir}/")
    print(f"  Summary CSV  : {summary_path}")


def plot_snapshot_fixed(
    phi,
    metadata,
    state_info,
    output_file,
    vmin,
    vmax,
    cbar_label,
    colorbar_mode,
    cmap="coolwarm",
    high_precision_time=False,
    escape_phi=None,
):
    """
    Create a snapshot plot with FIXED colorbar range.

    All snapshots use the same vmin/vmax for consistency!
    Supports both 2D (Nx, Ny) and 3D (Nx, Ny, Nz) fields.
    For 3D, shows three orthogonal midplane slices.
    """
    if metadata is not None and "vev" in metadata:
        vev = float(metadata["vev"])
        mu = float(metadata["mu"])
    elif metadata is not None and "mphi" in metadata and "lam" in metadata:
        mphi_val = float(metadata["mphi"])
        lam_val = float(metadata["lam"])
        vev = np.sqrt(mphi_val**2 / lam_val)
        mu = mphi_val
    else:
        _mphi_default = 1000.0
        _lam_default = 1e-16
        vev = np.sqrt(_mphi_default**2 / _lam_default)
        mu = _mphi_default

    if colorbar_mode == "normalized":
        phi_plot = phi / vev
    else:
        phi_plot = phi

    time_val = state_info["time"] / mu
    T_val = state_info["temperature"]
    step = state_info["step"]
    t_fmt = f"{time_val:.6f}" if high_precision_time else f"{time_val:.2e}"

    is_3d = phi.ndim == 3

    if is_3d:
        nx, ny, nz = phi.shape
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        slices = [
            (
                phi_plot[:, :, nz // 2],
                phi[:, :, nz // 2],
                f"z = {nz // 2} (xy-plane)",
                "x",
                "y",
            ),
            (
                phi_plot[:, ny // 2, :],
                phi[:, ny // 2, :],
                f"y = {ny // 2} (xz-plane)",
                "x",
                "z",
            ),
            (
                phi_plot[nx // 2, :, :],
                phi[nx // 2, :, :],
                f"x = {nx // 2} (yz-plane)",
                "y",
                "z",
            ),
        ]

        for ax, (slc, raw_slc, title, xlabel, ylabel) in zip(axes, slices):
            _plot_2d_slice(
                ax,
                slc,
                vmin,
                vmax,
                cmap,
                cbar_label,
                fig,
                escape_phi=escape_phi,
                phi_raw=raw_slc,
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)

        esc_str = f" | esc={escape_phi:.0f}" if escape_phi is not None else ""
        fig.suptitle(
            f"Step {step:,} | t={t_fmt} | T={T_val:.1f} | "
            f"$\\phi$: [{phi.min():.2e}, {phi.max():.2e}]{esc_str}",
            fontsize=12,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        _plot_2d_slice(
            ax,
            phi_plot,
            vmin,
            vmax,
            cmap,
            cbar_label,
            fig,
            escape_phi=escape_phi,
            phi_raw=phi,
        )
        ax.set_title(
            f"Step {step:,} | t={t_fmt} | T={T_val:.1f}\n"
            f"$\\phi$ range: [{phi.min():.2e}, {phi.max():.2e}]",
            fontsize=10,
        )
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def revisualize_all(
    sim_dir, colorbar_mode="normalized", cmap="coolwarm", escape_phi=None,
    step_min=None, step_max=None,
):
    """
    Revisualize all snapshots with FIXED colorbar range.

    The same colorbar range is used for ALL snapshots to enable
    detection of rare bubble nucleation events.
    Works for both regular simulation directories and detailed_replay subdirs.
    """

    is_replay = "detailed_replay" in os.path.abspath(sim_dir)
    if is_replay:
        print("Detected detailed_replay directory -> high-precision time labels")

    # Load metadata
    metadata = load_metadata(sim_dir)
    if metadata is None:
        print("Warning: Running without metadata (using hardcoded defaults)")

    # Find all field states
    state_dir = os.path.join(sim_dir, "field_states")
    if not os.path.exists(state_dir):
        print(f"Error: Field states directory not found: {state_dir}")
        print("Make sure the simulation has run with field state saving enabled.")
        return

    state_files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    state_files = [f for f in state_files if "_NaN_debug" not in f]
    state_files = _filter_by_step_range(state_files, step_min, step_max)
    if len(state_files) == 0:
        print(f"Error: No field state files found in {state_dir}")
        return

    print(f"Found {len(state_files)} field states to revisualize")
    print(f"Colorbar mode: {colorbar_mode}")
    print(f"Colormap: {cmap}")
    if escape_phi is not None:
        print(f"Escape highlight: |phi| > {escape_phi:.1f} (red=+, blue=-)")

    # Calculate VEV (derive from mphi/lam if vev not saved directly)
    if metadata is not None and "vev" in metadata:
        vev = float(metadata["vev"])
        mphi = float(metadata["mphi"])
        mu = float(metadata["mu"])
    elif metadata is not None and "mphi" in metadata and "lam" in metadata:
        mphi = float(metadata["mphi"])
        lam_val = float(metadata["lam"])
        vev = np.sqrt(mphi**2 / lam_val)
        mu = mphi
    else:
        mphi = 1000.0
        lam_val = 1e-16
        vev = np.sqrt(mphi**2 / lam_val)
        mu = mphi

    print(f"\nPhysics parameters:")
    print(f"  VEV (v) = {vev:.2e}")
    print(f"  mphi = {mphi:.2e}")

    # CRITICAL: Calculate GLOBAL colorbar range
    # This is FIXED for all snapshots to detect rare events!
    print(
        f"\nCalculating GLOBAL colorbar range for all {len(state_files)} snapshots..."
    )

    if colorbar_mode == "normalized":
        # Fixed range based on physics (φ/v)
        vmin, vmax = -1.5, 1.5
        cbar_label = r"$\phi / v$"
        print(f"  Mode: Normalized (φ/v)")
        print(f"  Range: [{vmin}, {vmax}] (FIXED)")
        print(f"  → False vacuum (φ≈0) appears as φ/v≈0 (green)")
        print(f"  → True vacuum (φ≈±v) appears as φ/v≈±1 (red/blue)")

    elif colorbar_mode == "vev_based":
        # Fixed range based on VEV
        vmin, vmax = -1.2 * vev, 1.2 * vev
        cbar_label = r"$\phi$"
        print(f"  Mode: VEV-based")
        print(f"  Range: [{vmin:.2e}, {vmax:.2e}] (FIXED)")

    elif colorbar_mode == "auto":
        # Scan ALL snapshots to find global min/max
        print(f"  Mode: Auto (scanning all snapshots...)")
        global_min = float("inf")
        global_max = float("-inf")
        for state_file in state_files:
            state = load_field_state(state_file)
            global_min = min(global_min, state["phi_min"])
            global_max = max(global_max, state["phi_max"])
        # Add padding
        phi_range = max(abs(global_min), abs(global_max))
        vmin, vmax = -phi_range * 1.1, phi_range * 1.1
        cbar_label = r"$\phi$"
        print(f"  Global field range: [{global_min:.2e}, {global_max:.2e}]")
        print(f"  Padded range: [{vmin:.2e}, {vmax:.2e}] (FIXED)")

    elif colorbar_mode == "symmetric":
        # Sample snapshots for percentile
        print(f"  Mode: Symmetric (sampling snapshots...)")
        all_abs_phi = []
        sample_files = state_files[:: max(1, len(state_files) // 20)]  # Sample 20
        for state_file in sample_files:
            state = load_field_state(state_file)
            all_abs_phi.extend(np.abs(state["phi"]).flatten())
        phi_range = np.percentile(all_abs_phi, 99.5)
        vmin, vmax = -phi_range, phi_range
        cbar_label = r"$\phi$"
        print(f"  99.5th percentile: {phi_range:.2e}")
        print(f"  Range: [{vmin:.2e}, {vmax:.2e}] (FIXED)")

    else:  # fixed or custom
        vmin, vmax = -10000000, 10000000
        cbar_label = r"$\phi$"
        print(f"  Mode: Fixed")
        print(f"  Range: [{vmin}, {vmax}] (FIXED)")
        print(f"  WARNING: This may not show bubbles if v >> 1000!")

    # Create output directory
    output_dir = os.path.join(sim_dir, f"revisualized_{colorbar_mode}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"\n{'='*70}")
    print(f"IMPORTANT: ALL snapshots will use the SAME colorbar range!")
    print(f"  vmin = {vmin:.2e}")
    print(f"  vmax = {vmax:.2e}")
    print(f"This fixed scale is essential for detecting rare bubble nucleation.")
    print(f"{'='*70}\n")

    # Process each state with FIXED colorbar
    for i, state_file in enumerate(state_files):
        state = load_field_state(state_file)

        output_file = os.path.join(
            output_dir, f"snapshot_step_{state['step']:010d}.png"
        )

        # Use GLOBAL FIXED range for all snapshots
        plot_snapshot_fixed(
            state["phi"],
            metadata,
            state,
            output_file,
            vmin=vmin,
            vmax=vmax,
            cbar_label=cbar_label,
            colorbar_mode=colorbar_mode,
            cmap=cmap,
            high_precision_time=is_replay,
            escape_phi=escape_phi,
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(state_files):
            print(f"Progress: {i+1}/{len(state_files)} snapshots")

    print(f"\n{'='*70}")
    print(f"All snapshots saved to: {output_dir}")
    print(f"\nColorbar info:")
    print(f"  Range used: [{vmin:.2e}, {vmax:.2e}]")
    print(f"  ALL {len(state_files)} snapshots use this SAME range")
    print(f"  → Rare events will stand out dramatically!")
    print(f"{'='*70}")


def compare_colorbar_modes(sim_dir, step_to_plot=None):
    """Compare different colorbar modes for a specific snapshot."""

    # Load metadata
    metadata = load_metadata(sim_dir)

    # Find field states
    state_dir = os.path.join(sim_dir, "field_states")
    state_files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))

    if len(state_files) == 0:
        print(f"Error: No field state files found")
        return

    # Choose which snapshot to compare
    if step_to_plot is None:
        # Use middle snapshot
        state_file = state_files[len(state_files) // 2]
    else:
        # Find specific step
        matching = [f for f in state_files if f"step_{step_to_plot:010d}" in f]
        if not matching:
            print(f"Error: Step {step_to_plot} not found")
            return
        state_file = matching[0]

    state = load_field_state(state_file)
    phi = state["phi"]

    if metadata is not None and "vev" in metadata:
        vev = float(metadata["vev"])
        mu = float(metadata["mu"])
    elif metadata is not None and "mphi" in metadata and "lam" in metadata:
        mphi_val = float(metadata["mphi"])
        lam_val = float(metadata["lam"])
        vev = np.sqrt(mphi_val**2 / lam_val)
        mu = mphi_val
    else:
        _mphi_default = 1000.0
        _lam_default = 1e-16
        vev = np.sqrt(_mphi_default**2 / _lam_default)
        mu = _mphi_default

    # For 3D fields, take the z-midplane slice for comparison
    is_3d = phi.ndim == 3
    phi_2d = phi[:, :, phi.shape[2] // 2] if is_3d else phi
    dim_label = " (z-midplane)" if is_3d else ""

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    modes = [
        ("normalized", -1.5, 1.5, r"$\phi / v$", phi_2d / vev),
        ("vev_based", -1.2 * vev, 1.2 * vev, r"$\phi$", phi_2d),
        (
            "auto",
            -max(abs(phi.min()), abs(phi.max())) * 1.1,
            max(abs(phi.min()), abs(phi.max())) * 1.1,
            r"$\phi$",
            phi_2d,
        ),
        ("fixed", -1000, 1000, r"$\phi$", phi_2d),
    ]

    for ax, (mode, vmin, vmax, cbar_label, phi_plot) in zip(axes.flat, modes):
        im = ax.imshow(phi_plot, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        ax.set_title(
            f"Mode: {mode}{dim_label}\n"
            f"Range: [{vmin:.2e}, {vmax:.2e}]\n"
            f"φ actual: [{phi.min():.2e}, {phi.max():.2e}]",
            fontsize=10,
        )

    fig.suptitle(
        f"Colorbar Comparison - Step {state['step']}\n"
        f"t={state['time']/mu:.2e}, T={state['temperature']:.1f}\n"
        f"VEV = {vev:.2e}",
        fontsize=14,
    )
    fig.tight_layout()

    output_file = os.path.join(sim_dir, "colorbar_comparison.png")
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"\n{'='*70}")
    print(f"Comparison saved to: {output_file}")
    print(f"{'='*70}")
    print(f"\nRecommendation for bubble nucleation:")
    print(f"  → Use 'normalized' mode (φ/v with [-1.5, 1.5])")
    print(f"  → False vacuum appears as 0 (green)")
    print(f"  → True vacuum appears as ±1 (red/blue)")
    print(f"  → Bubbles will be VERY visible!")


def print_usage():
    """Print usage information."""
    print(__doc__)
    print("\nColorbar modes:")
    print("  normalized  - φ/v scale (RECOMMENDED for bubble detection!)")
    print("  vev_based   - Fixed range ±1.2*VEV")
    print("  auto        - Global data range from all snapshots")
    print("  symmetric   - Global percentile-based")
    print("  fixed       - Fixed [-1000, 1000] (not recommended)")
    print("\nExamples:")
    print("  # Revisualize with normalized colorbar (RECOMMENDED)")
    print("  python postprocess/revisualize_snapshots.py data/lattice/set6/T0_7350_...")
    print("")
    print("  # Use VEV-based scaling")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --mode vev_based")
    print("")
    print("  # Compare different modes first")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --compare")
    print("")
    print("  # Use different colormap")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --cmap seismic")
    print("")
    print("  # Highlight sites exceeding escape point (red=+, blue=-)")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --escape_phi 3000")
    print("")
    print("  # 3D bubble rendering (requires --escape_phi)")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --view3d --escape_phi 3000")
    print(
        "  python postprocess/revisualize_snapshots.py <sim_dir> --view3d --escape_phi 3000 --elev 30 --azim 120"
    )
    print(
        "  python postprocess/revisualize_snapshots.py <sim_dir> --view3d --escape_phi 3000 --no_surface"
    )
    print("")
    print("  # Cosmic string visualization (complex-field snapshots)")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --strings")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --strings --elev 30 --azim 120")
    print("")
    print("  # Process only a range of steps (works with all modes)")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --strings --step_min 500000")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --strings --step_min 500000 --step_max 1000000")
    print("  python postprocess/revisualize_snapshots.py <sim_dir> --view3d --escape_phi 3000 --step_max 200000")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    sim_dir = sys.argv[1]

    if not os.path.exists(sim_dir):
        print(f"Error: Directory not found: {sim_dir}")
        sys.exit(1)

    # Parse arguments
    mode = "auto"
    cmap = "coolwarm"
    compare_mode = False
    escape_phi = None
    view3d = False
    strings_mode = False
    elev = 25
    azim = 135
    surface_only = True
    step_min = None
    step_max = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--mode":
            mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--cmap":
            cmap = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--compare":
            compare_mode = True
            i += 1
        elif sys.argv[i] == "--escape_phi":
            escape_phi = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--view3d":
            view3d = True
            i += 1
        elif sys.argv[i] == "--strings":
            strings_mode = True
            i += 1
        elif sys.argv[i] == "--elev":
            elev = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--azim":
            azim = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no_surface":
            surface_only = False
            i += 1
        elif sys.argv[i] == "--step_min":
            step_min = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--step_max":
            step_max = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--help":
            print_usage()
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            print_usage()
            sys.exit(1)

    if step_min is not None or step_max is not None:
        print(f"Step range: [{step_min or 'start'} .. {step_max or 'end'}]")

    if strings_mode:
        revisualize_strings(sim_dir, elev=elev, azim=azim,
                            step_min=step_min, step_max=step_max)
    elif view3d:
        if escape_phi is None:
            print("Error: --view3d requires --escape_phi <value>")
            sys.exit(1)
        revisualize_3d(
            sim_dir,
            escape_phi=escape_phi,
            elev=elev,
            azim=azim,
            surface_only=surface_only,
            step_min=step_min,
            step_max=step_max,
        )
    elif compare_mode:
        compare_colorbar_modes(sim_dir)
    else:
        revisualize_all(sim_dir, colorbar_mode=mode, cmap=cmap, escape_phi=escape_phi,
                        step_min=step_min, step_max=step_max)


if __name__ == "__main__":
    main()
