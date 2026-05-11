#!/usr/bin/env python
"""
Plot the lattice-averaged field evolution sqrt(<phi^2>) vs time/step.

Produces a figure similar to DJS (2024) Fig. 4 (right panel),
showing the RMS field value on the lattice as a function of simulation
time, with the zero-temperature VEV as a reference line.

Usage:
    python postprocess/plot_field_evolution.py <simulation_directory> [options]

Options:
    --x_axis  step|time|time_phys   X-axis variable (default: step)
    --normalize                     Plot phi_rms / VEV instead of raw value
    --show_temp                     Overlay temperature on secondary y-axis
    --show_mean                     Also plot |<phi>| (mean field)
    --show_max                      Also plot max|phi|
    --step_min N                    Only include steps >= N
    --step_max N                    Only include steps <= N
    --output FILE                   Output filename (default: auto)

Examples:
    python postprocess/plot_field_evolution.py data/lattice/set7/...
    python postprocess/plot_field_evolution.py data/lattice/set7/... --normalize --show_temp
    python postprocess/plot_field_evolution.py data/lattice/set7/... --x_axis time_phys
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import sys
import argparse


def load_metadata(sim_dir):
    search_dir = os.path.abspath(sim_dir)
    for _ in range(4):
        metadata_file = os.path.join(search_dir, "simulation_metadata.npz")
        if os.path.exists(metadata_file):
            return dict(np.load(metadata_file))
        search_dir = os.path.dirname(search_dir)
    return None


def extract_step(path):
    m = re.search(r"state_step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Plot lattice field evolution")
    parser.add_argument("sim_dir", help="Simulation directory")
    parser.add_argument(
        "--x_axis",
        choices=["step", "time", "time_phys"],
        default="step",
        help="X-axis: step number, rescaled time, or physical time (default: step)",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize by VEV"
    )
    parser.add_argument(
        "--show_temp", action="store_true", help="Overlay temperature"
    )
    parser.add_argument(
        "--show_mean", action="store_true", help="Also plot |<phi>|"
    )
    parser.add_argument(
        "--show_max", action="store_true", help="Also plot max|phi|"
    )
    parser.add_argument(
        "--show_energy", action="store_true",
        help="Plot energy diagnostics (requires --diag_energy in simulation)",
    )
    parser.add_argument("--step_min", type=int, default=None)
    parser.add_argument("--step_max", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output file")
    args = parser.parse_args()

    sim_dir = args.sim_dir
    if not os.path.isdir(sim_dir):
        print(f"Error: directory not found: {sim_dir}")
        sys.exit(1)

    metadata = load_metadata(sim_dir)
    mu = float(metadata["mu"]) if metadata and "mu" in metadata else 1000.0
    vev = float(metadata["vev"]) if metadata and "vev" in metadata else None
    if vev is None and metadata and "mphi" in metadata and "lam" in metadata:
        vev = np.sqrt(float(metadata["mphi"]) ** 2 / float(metadata["lam"]))

    state_dir = os.path.join(sim_dir, "field_states")
    if not os.path.isdir(state_dir):
        print(f"Error: field_states not found in {sim_dir}")
        sys.exit(1)

    state_files = sorted(glob.glob(os.path.join(state_dir, "state_step_*.npz")))
    state_files = [f for f in state_files if "_NaN_debug" not in f]

    if args.step_min is not None or args.step_max is not None:
        filtered = []
        for f in state_files:
            s = extract_step(f)
            if s is None:
                continue
            if args.step_min is not None and s < args.step_min:
                continue
            if args.step_max is not None and s > args.step_max:
                continue
            filtered.append(f)
        state_files = filtered

    if not state_files:
        print("Error: no state files found")
        sys.exit(1)

    print(f"Loading {len(state_files)} snapshots...")

    steps_arr = []
    times_arr = []
    temps_arr = []
    phi_rms_arr = []
    phi_mean_arr = []
    phi_absmax_arr = []
    e_kin_arr = []
    e_grad_arr = []
    e_pot_arr = []
    has_energy = False

    for i, sf in enumerate(state_files):
        d = np.load(sf)
        phi = d["phi"].astype(np.float64)

        steps_arr.append(int(d["step"]))
        times_arr.append(float(d["time"]))
        temps_arr.append(float(d["temperature"]))
        phi_rms_arr.append(np.sqrt(np.mean(phi ** 2)))
        phi_mean_arr.append(np.abs(np.mean(phi)))
        phi_absmax_arr.append(np.max(np.abs(phi)))

        if "E_kin" in d:
            has_energy = True
            e_kin_arr.append(float(d["E_kin"]))
            e_grad_arr.append(float(d["E_grad"]))
            e_pot_arr.append(float(d["E_pot"]))
        else:
            e_kin_arr.append(np.nan)
            e_grad_arr.append(np.nan)
            e_pot_arr.append(np.nan)

        if (i + 1) % 20 == 0 or i + 1 == len(state_files):
            print(f"  [{i+1}/{len(state_files)}]")

    steps_arr = np.array(steps_arr)
    times_arr = np.array(times_arr)
    temps_arr = np.array(temps_arr)
    phi_rms = np.array(phi_rms_arr)
    phi_mean = np.array(phi_mean_arr)
    phi_absmax = np.array(phi_absmax_arr)
    e_kin = np.array(e_kin_arr)
    e_grad = np.array(e_grad_arr)
    e_pot = np.array(e_pot_arr)

    if args.x_axis == "step":
        x = steps_arr
        xlabel = "Step"
    elif args.x_axis == "time":
        x = times_arr
        xlabel = r"$\tilde{t}$ (rescaled)"
    else:
        x = times_arr / mu
        xlabel = r"$t_{\rm phys}$ [GeV$^{-1}$]"

    norm = vev if (args.normalize and vev) else 1.0
    norm_label = r" / $v$" if args.normalize else ""

    # --- Figure ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        x,
        phi_rms / norm,
        color="navy",
        lw=2,
        label=r"$\sqrt{\langle\phi^2_{\rm lat}\rangle}$",
    )

    if args.show_mean:
        ax1.plot(
            x,
            phi_mean / norm,
            color="steelblue",
            lw=1.5,
            ls="--",
            label=r"$|\langle\phi_{\rm lat}\rangle|$",
        )

    if args.show_max:
        ax1.plot(
            x,
            phi_absmax / norm,
            color="gray",
            lw=1,
            ls=":",
            alpha=0.7,
            label=r"$\max|\phi|$",
        )

    if vev is not None:
        ax1.axhline(
            vev / norm,
            color="red",
            ls="dotted",
            lw=2,
            label=r"$v = \phi_{\rm VEV}$" + (f" = {vev:.2e}" if not args.normalize else ""),
        )

    ax1.set_xlabel(xlabel, fontsize=13)
    if args.normalize:
        ax1.set_ylabel(
            r"$\sqrt{\langle\phi^2_{\rm lat}\rangle}\,/\,v$", fontsize=13
        )
        ax1.set_ylim(-0.05, 1.15)
        ax1.yaxis.get_major_formatter().set_useOffset(False)
        ax1.yaxis.get_major_formatter().set_scientific(False)
    else:
        ax1.set_ylabel(
            r"$\sqrt{\langle\phi^2_{\rm lat}\rangle}$ [GeV]", fontsize=13
        )

    if args.show_temp:
        ax2 = ax1.twinx()
        ax2.plot(x, temps_arr, color="orangered", lw=1.2, ls="-.", alpha=0.6)
        ax2.set_ylabel(r"$T$ [GeV]", fontsize=12, color="orangered")
        ax2.tick_params(axis="y", labelcolor="orangered")

    ax1.legend(loc="best", fontsize=11, framealpha=0.9)

    T_range_str = f"$T$: {temps_arr[0]:.0f} $\\to$ {temps_arr[-1]:.0f} GeV"
    step_range_str = f"Steps: {steps_arr[0]:,} – {steps_arr[-1]:,}"
    info = f"{step_range_str}   |   {T_range_str}"
    if metadata and "integrator" in metadata:
        info += f"   |   {metadata['integrator']}"
    ax1.set_title(info, fontsize=10, color="gray")

    ax1.tick_params(labelsize=11)
    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(sim_dir, "figs")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_normalized" if args.normalize else ""
        out_path = os.path.join(out_dir, f"field_evolution_{args.x_axis}{suffix}.png")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # --- Energy diagnostics figure ---
    if args.show_energy:
        if not has_energy:
            print("\nWarning: no energy data found in snapshots.")
            print("  Run simulation with --diag_energy to enable.")
        else:
            dx_phys = (
                float(metadata["dx_phys"])
                if metadata and "dx_phys" in metadata
                else 0.001
            )

            fig_e, (ax_e1, ax_e2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

            e_tot = e_kin + e_grad + e_pot
            ax_e1.plot(x, e_kin, color="red", lw=1.5, label=r"$E_{\rm kin}$")
            ax_e1.plot(x, e_grad, color="blue", lw=1.5, label=r"$E_{\rm grad}$")
            ax_e1.plot(x, e_pot, color="green", lw=1.5, label=r"$E_{\rm pot}$ (tree)")
            ax_e1.plot(x, e_tot, color="black", lw=2, ls="--", label=r"$E_{\rm tot}$")
            ax_e1.set_ylabel("Energy density per site", fontsize=12)
            ax_e1.legend(fontsize=10, ncol=2)
            ax_e1.set_title("Energy diagnostics", fontsize=11, color="gray")

            e_kin_expect = mu ** 2 * temps_arr / (2.0 * dx_phys ** 3)
            equip_ratio = np.where(e_kin_expect > 0, e_kin / e_kin_expect, np.nan)
            ax_e2.plot(x, equip_ratio, color="navy", lw=2)
            ax_e2.axhline(1.0, color="red", ls="dotted", lw=1.5, label="Equipartition")
            ax_e2.set_xlabel(xlabel, fontsize=13)
            ax_e2.set_ylabel(
                r"$E_{\rm kin}\,/\,(\mu^2 T / 2\Delta x^3)$", fontsize=12
            )
            ax_e2.legend(fontsize=10)
            ax_e2.set_ylim(0, max(2.0, np.nanmax(equip_ratio) * 1.1))

            fig_e.tight_layout()
            e_out = os.path.join(
                os.path.dirname(out_path),
                f"energy_diagnostics_{args.x_axis}.png",
            )
            fig_e.savefig(e_out, dpi=200, bbox_inches="tight")
            plt.close(fig_e)
            print(f"Saved: {e_out}")


if __name__ == "__main__":
    main()
