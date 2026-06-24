#!/usr/bin/env python
"""
Side-by-side lattice simulation snapshots with field evolution and
false-vacuum volume fraction.

Each column: 4 snapshots (2x2 3D) + [⟨φ⟩ | false-vacuum fraction] strip.
Renders each panel individually then composites via PIL.

Usage:
    python postprocess/plot_bubble_comparison_with_frac.py SIM1_DIR SIM2_DIR [options]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import re
import sys
import argparse
import tempfile

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "axes.unicode_minus": False,
})

sys.path.insert(0, os.path.dirname(__file__))
from revisualize_snapshots import _find_surface_voxels


def load_snapshot(path):
    d = np.load(path)
    return {
        "phi": d["phi"].astype(np.float64),
        "step": int(d["step"]),
        "temperature": float(d["temperature"]),
        "time": float(d["time"]),
    }


def pick_closest_file(state_dir, target_step):
    files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    best, best_diff = None, 1e18
    for f in files:
        m = re.search(r"state_step_(\d+)", os.path.basename(f))
        if m:
            s = int(m.group(1))
            if abs(s - target_step) < best_diff:
                best_diff = abs(s - target_step)
                best = f
    return best


def auto_select_steps(state_dir, escape_phi, n=4):
    files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    steps, fracs = [], []
    for f in files:
        d = np.load(f)
        phi = d["phi"].astype(np.float64)
        steps.append(int(d["step"]))
        fracs.append(np.mean(np.abs(phi) > escape_phi))
    fracs = np.array(fracs)
    steps = np.array(steps)
    targets = [0.001, 0.02, 0.10, 0.35]
    return [int(steps[np.argmin(np.abs(fracs - t))]) for t in targets]


def render_single_3d(phi, escape_phi, T_val, out_png,
                     elev=25, azim=135, dpi=150):
    mask_pos = phi > escape_phi
    mask_neg = phi < -escape_phi
    n_pos = int(np.sum(mask_pos))
    n_neg = int(np.sum(mask_neg))
    total_pts = n_pos + n_neg

    pts_pos = _find_surface_voxels(mask_pos)
    pts_neg = _find_surface_voxels(mask_neg)

    max_pts = 500_000
    if len(pts_pos) > max_pts:
        pts_pos = pts_pos[np.random.choice(len(pts_pos), max_pts, replace=False)]
    if len(pts_neg) > max_pts:
        pts_neg = pts_neg[np.random.choice(len(pts_neg), max_pts, replace=False)]

    marker_size = max(2.0, min(20.0, 2000.0 / max(1, total_pts)))

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    if len(pts_pos) > 0:
        ax.scatter(pts_pos[:, 0], pts_pos[:, 1], pts_pos[:, 2],
                   c="red", s=marker_size, alpha=0.7, linewidths=0,
                   depthshade=True)
    if len(pts_neg) > 0:
        ax.scatter(pts_neg[:, 0], pts_neg[:, 1], pts_neg[:, 2],
                   c="blue", s=marker_size, alpha=0.7, linewidths=0,
                   depthshade=True)

    nx, ny, nz = phi.shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x", fontsize=24, labelpad=10)
    ax.set_ylabel("y", fontsize=24, labelpad=10)
    ax.set_zlabel("z", fontsize=24, labelpad=10)
    ax.tick_params(labelsize=18, pad=5)
    ax.set_title(rf"$T = {T_val/1000:.3f}$  TeV",
                 fontsize=32, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.02)
    plt.close(fig)


def render_evolution(state_dir, out_png, xlim=None, show_ylabel=True, dpi=150):
    """Render ⟨φ⟩ evolution vs step (narrower width)."""
    files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    steps, rms_vals = [], []
    for f in files:
        d = np.load(f)
        phi = d["phi"].astype(np.float64)
        steps.append(int(d["step"]))
        rms_vals.append(np.sqrt(np.mean(phi**2)))
    steps = np.array(steps)
    rms_vals = np.array(rms_vals)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.plot(steps, rms_vals / 1000, color="navy", lw=4.0,
            label=r"$\sqrt{\langle\phi^2_{\rm lat}\rangle}$")
    ax.set_xlabel(r"Step", fontsize=30)
    if show_ylabel:
        ax.set_ylabel(r"$\sqrt{\langle\phi^2_{\rm lat}\rangle}$ [TeV]", fontsize=26)
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=24, loc="upper left")
    if xlim:
        ax.set_xlim(*xlim)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)


def render_false_vacuum_frac(state_dir, escape_phi, Tc1_GeV, out_png,
                             show_ylabel=True, dpi=150, max_step=None):
    """Render false-vacuum volume fraction vs temperature."""
    files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    temps, fracs_false = [], []
    for f in files:
        d = np.load(f)
        step = int(d["step"])
        if max_step is not None and step > max_step:
            continue
        phi = d["phi"].astype(np.float64)
        T = float(d["temperature"])
        frac_true = np.mean(np.abs(phi) > escape_phi)
        temps.append(T)
        fracs_false.append(1.0 - frac_true)
    temps = np.array(temps)
    fracs_false = np.array(fracs_false)

    order = np.argsort(temps)
    temps = temps[order]
    fracs_false = fracs_false[order]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.plot(temps / 1000, fracs_false, color="darkgreen", lw=4.0,
            label=r"$f_{\rm false}$")
    if Tc1_GeV is not None:
        ax.axvline(Tc1_GeV / 1000, ls="--", color="red", lw=3.0,
                   label=rf"$T_{{c_1}} = {Tc1_GeV/1000:.3f}$ TeV")
    ax.set_xlabel(r"$T$ [TeV]", fontsize=30)
    if show_ylabel:
        ax.set_ylabel(r"False vacuum fraction", fontsize=26)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=20, loc="upper right")
    ax.invert_xaxis()
    # Note: we do not set xlim manually here so it autoscales to the data within max_step.
    
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)


def render_subtitle(text, width_px, out_png, dpi=180, fontsize=38):
    fig_w = width_px / dpi
    fig = plt.figure(figsize=(fig_w, 1.0), facecolor="white")
    fig.text(0.5, 0.5, text, ha="center", va="center",
             fontsize=fontsize, fontweight="bold")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side lattice snapshots with false-vacuum fraction")
    parser.add_argument("sim1", help="Simulation 1 directory (left = Set B)")
    parser.add_argument("sim2", help="Simulation 2 directory (right = Set C)")
    parser.add_argument("--escape_phi", type=float, default=10000)
    parser.add_argument("--steps1", type=int, nargs=4, default=None)
    parser.add_argument("--steps2", type=int, nargs=4, default=None)
    parser.add_argument("--Tc1_B", type=float, default=1157.6,
                        help="T_c1 for Set B in GeV (default: 1157.6)")
    parser.add_argument("--Tc1_C", type=float, default=1511.5,
                        help="T_c1 for Set C in GeV (default: 1511.5)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    escape_phi = args.escape_phi
    sim_dirs = [args.sim1, args.sim2]
    step_overrides = [args.steps1, args.steps2]
    Tc1_vals = [args.Tc1_B, args.Tc1_C]

    tmpdir = tempfile.mkdtemp(prefix="bubble_comp_frac_")
    panel_files = {}

    for col_idx, (sim_dir, steps_ov) in enumerate(zip(sim_dirs, step_overrides)):
        state_dir = os.path.join(sim_dir, "field_states")

        if steps_ov is not None:
            step_list = steps_ov
        else:
            step_list = auto_select_steps(state_dir, escape_phi, n=4)

        print(f"\n  Column {col_idx}: {os.path.basename(sim_dir)}")
        print(f"  Steps: {step_list}")

        for i, target_step in enumerate(step_list):
            snap_file = pick_closest_file(state_dir, target_step)
            snap = load_snapshot(snap_file)
            frac = np.mean(np.abs(snap["phi"]) > escape_phi)
            print(f"    [{i}] step={snap['step']:6d}  T={snap['temperature']:.1f}  "
                  f"frac={frac:.4f}")

            png = os.path.join(tmpdir, f"c{col_idx}_s{i}.png")
            render_single_3d(snap["phi"], escape_phi, snap["temperature"],
                             png, dpi=args.dpi)
            panel_files[(col_idx, i)] = png

        ev_png = os.path.join(tmpdir, f"ev{col_idx}.png")
        render_evolution(state_dir, ev_png,
                         xlim=(0, 6000),
                         show_ylabel=True,
                         dpi=args.dpi)
        panel_files[(col_idx, "ev")] = ev_png

        frac_png = os.path.join(tmpdir, f"frac{col_idx}.png")
        render_false_vacuum_frac(state_dir, escape_phi, Tc1_vals[col_idx],
                                 frac_png,
                                 show_ylabel=True,
                                 dpi=args.dpi,
                                 max_step=6000)
        panel_files[(col_idx, "frac")] = frac_png

    subtitles = [
        r"Set B : $V_0 = 2.5 \times 10^{35},\; n_b = n_f = 20$",
        r"Set C : $V_0 = 2.5 \times 10^{35},\; n_b = 0,\; n_f = 20$",
    ]

    imgs = {}
    for key, path in panel_files.items():
        imgs[key] = Image.open(path)

    sample = imgs[(0, 0)]
    pw, ph = sample.size

    gap_inner = 5
    gap_col = pw // 12
    gap_row = 10
    gap_ev = 15
    gap_ev_frac = 10

    col_w = 2 * pw + gap_inner

    sub_pngs = []
    for i, txt in enumerate(subtitles):
        sp = os.path.join(tmpdir, f"sub{i}.png")
        render_subtitle(txt, col_w, sp, dpi=args.dpi)
        sub_pngs.append(sp)
    sub_imgs = [Image.open(p) for p in sub_pngs]
    sub_h = max(s.size[1] for s in sub_imgs)

    ev_sample = imgs[(0, "ev")]
    frac_sample = imgs[(0, "frac")]
    ew, eh = ev_sample.size
    fw, fh = frac_sample.size

    half_col_w = (col_w - gap_ev_frac) // 2
    ev_scaled_h = int(eh * half_col_w / ew)
    frac_scaled_h = int(fh * half_col_w / fw)
    bottom_h = max(ev_scaled_h, frac_scaled_h)

    total_w = 2 * col_w + gap_col
    total_h = sub_h + 2 * (ph + gap_row) + bottom_h + gap_ev

    canvas = Image.new("RGB", (total_w, total_h), "white")

    for col_idx in range(2):
        x_base = col_idx * (col_w + gap_col)

        si = sub_imgs[col_idx]
        sx = x_base + (col_w - si.size[0]) // 2
        canvas.paste(si, (sx, 0))

        for i in range(4):
            row_i, col_i = divmod(i, 2)
            x = x_base + col_i * (pw + gap_inner)
            y = sub_h + row_i * (ph + gap_row)
            canvas.paste(imgs[(col_idx, i)], (x, y))

        ev_img = imgs[(col_idx, "ev")]
        ev_resized = ev_img.resize((half_col_w, ev_scaled_h), Image.LANCZOS)
        ev_x = x_base
        ev_y = sub_h + 2 * (ph + gap_row) + gap_ev
        canvas.paste(ev_resized, (ev_x, ev_y))

        frac_img = imgs[(col_idx, "frac")]
        frac_resized = frac_img.resize((half_col_w, frac_scaled_h), Image.LANCZOS)
        frac_x = x_base + half_col_w + gap_ev_frac
        frac_y = sub_h + 2 * (ph + gap_row) + gap_ev
        canvas.paste(frac_resized, (frac_x, frac_y))

    if args.output:
        out_path = args.output
    else:
        os.makedirs("figs/bubble_comparison", exist_ok=True)
        out_path = "figs/bubble_comparison/lattice_two_sets_with_frac.png"

    cw, ch = canvas.size
    target_ratio = 4.0 / 3.0
    current_ratio = cw / ch
    if current_ratio > target_ratio:
        new_h = int(cw / target_ratio)
        padded = Image.new("RGB", (cw, new_h), "white")
        padded.paste(canvas, (0, (new_h - ch) // 2))
        canvas = padded
    elif current_ratio < target_ratio:
        new_w = int(ch * target_ratio)
        padded = Image.new("RGB", (new_w, ch), "white")
        padded.paste(canvas, ((new_w - cw) // 2, 0))
        canvas = padded

    canvas.save(out_path)
    print(f"\nSaved: {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)  "
          f"[{canvas.size[0]}x{canvas.size[1]} px]")

    for p in panel_files.values():
        os.remove(p)
    for p in sub_pngs:
        os.remove(p)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
