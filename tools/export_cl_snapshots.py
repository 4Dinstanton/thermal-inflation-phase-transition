#!/usr/bin/env python3
"""
Convert CosmoLattice raw field snapshots to numba-compatible NPZ files.

Reads field_states/manifest.csv + snapshot_*.raw written by field_snapshot.hpp,
produces field_states/state_step_{step:010d}.npz and simulation_metadata.npz
for postprocess/revisualize_snapshots.py.

For two-component (complex-field) snapshots, also computes winding density via
tools/winding.py so --strings mode works out of the box.

Usage
-----
    python tools/export_cl_snapshots.py <run_directory>
    python tools/export_cl_snapshots.py <run_directory> --keep-raw
"""
import argparse
import csv
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.winding import compute_winding_number

SNAPSHOT_MAGIC = 0x464C5048  # 'FLPH' phi only
SNAPSHOT_MAGIC_PI = 0x464C5049  # 'FLPI' phi + pi
SNAPSHOT_MAGIC_PI2 = 0x464C5032  # 'FLP2' phi1, phi2, pi1, pi2

HEADER_FMT = "<IIq5d"
HEADER2_FMT = "<IIq5dI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
HEADER2_SIZE = struct.calcsize(HEADER2_FMT)


def read_raw_snapshot(path):
    with open(path, "rb") as f:
        hdr_peek = f.read(4)
        if len(hdr_peek) < 4:
            raise ValueError(f"truncated file: {path}")
        magic = struct.unpack("<I", hdr_peek)[0]
        f.seek(0)

        if magic == SNAPSHOT_MAGIC_PI2:
            buf = f.read(HEADER2_SIZE)
            if len(buf) != HEADER2_SIZE:
                raise ValueError(f"truncated header: {path}")
            magic, n, step, t, T, a, H, fStar, n_scalars = struct.unpack(HEADER2_FMT, buf)
            n_comp = int(n_scalars)
        else:
            buf = f.read(HEADER_SIZE)
            if len(buf) != HEADER_SIZE:
                raise ValueError(f"truncated header: {path}")
            magic, n, step, t, T, a, H, fStar = struct.unpack(HEADER_FMT, buf)
            n_comp = 1

        if magic not in (SNAPSHOT_MAGIC, SNAPSHOT_MAGIC_PI, SNAPSHOT_MAGIC_PI2):
            raise ValueError(f"bad magic {magic:#x} in {path}")

        n3 = n * n * n

        def read_field():
            payload = f.read(4 * n3)
            if len(payload) != 4 * n3:
                raise ValueError(f"truncated field payload in {path}")
            return np.frombuffer(payload, dtype=np.float32).reshape((n, n, n), order="C")

        if n_comp >= 2:
            fld1 = read_field()
            fld2 = read_field()
            pi1_prog = read_field()
            pi2_prog = read_field()
            phi1_gev = fld1.astype(np.float64) * fStar
            phi2_gev = fld2.astype(np.float64) * fStar
            rho = np.sqrt(phi1_gev ** 2 + phi2_gev ** 2)
            winding = compute_winding_number(phi1_gev, phi2_gev)
            return {
                "step": int(step),
                "time": float(t),
                "temperature": float(T),
                "a": float(a),
                "H": float(H),
                "fStar": float(fStar),
                "N": int(n),
                "n_scalars": 2,
                "phi1": phi1_gev,
                "phi2": phi2_gev,
                "rho": rho,
                "theta": np.arctan2(phi2_gev, phi1_gev),
                "winding": winding,
                "pi1": pi1_prog.astype(np.float64) * fStar,
                "pi2": pi2_prog.astype(np.float64) * fStar,
            }

        fld = read_field()
        phi_gev = fld.astype(np.float64) * fStar
        out = {
            "step": int(step),
            "time": float(t),
            "temperature": float(T),
            "a": float(a),
            "H": float(H),
            "fStar": float(fStar),
            "phi": phi_gev,
            "N": int(n),
            "n_scalars": 1,
        }
        if magic == SNAPSHOT_MAGIC_PI:
            pi_prog = read_field()
            out["pi"] = pi_prog.astype(np.float64) * fStar
        return out


def load_run_params(run_dir):
    params_path = os.path.join(run_dir, "cl_run_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            return json.load(f)
    return {}


def write_metadata(run_dir, params, n_snapshots):
    mphi = float(params.get("mphi", 1000.0))
    gamma = float(params.get("gamma", 4.1667e-4))
    M_PL = 2.4e18
    phi0 = gamma * M_PL
    lam = params.get("lam")
    if lam is None:
        lam = mphi * mphi / (phi0 * phi0)
    vev = params.get("vev", np.sqrt(mphi ** 2 / lam))

    meta = {
        "Nx": int(params.get("Nx", 64)),
        "Ny": int(params.get("Ny", params.get("Nx", 64))),
        "Nz": int(params.get("Nz", params.get("Nx", 64))),
        "dx_phys": float(params.get("dx_phys", 1e-3)),
        "dt_phys": float(params.get("dt_phys", 1e-4)),
        "mphi": mphi,
        "lam": float(lam),
        "eta_phys": float(params.get("eta_phys", params.get("T0", 7350))),
        "nb": float(params.get("nb", 20)),
        "nf": float(params.get("nf", 20)),
        "bosonCoupling": float(params.get("boson_coupling", 1.09)),
        "fermionCoupling": float(params.get("fermion_coupling", 1.09)),
        "T0": float(params.get("T0", 7350)),
        "cooling_rate": float(params.get("cooling_rate", 0.0)),
        "Nt": int(params.get("Nt", 0)),
        "steps": int(params.get("steps", 0)),
        "total_time": float(params.get("total_time", params.get("tMax", 0))),
        "mu": mphi,
        "dx": mphi * float(params.get("dx_phys", 1e-3)),
        "dt": mphi * float(params.get("dt_phys", 1e-4)),
        "eta": float(params.get("eta_phys", params.get("T0", 7350))) / mphi,
        "vev": float(vev),
        "integrator": params.get("integrator", "stochasticrk_CL"),
        "counterterm": params.get("counterterm", "none"),
        "potential_type": params.get("potential_type", "V_correct"),
        "no_hubble": bool(params.get("no_hubble", False)),
        "no_scale_factor": False,
        "n_snapshots": n_snapshots,
        "n_scalars": int(params.get("n_scalars", 1)),
        "zn_order": float(params.get("zn_order", 0)),
        "zn_strength": float(params.get("zn_strength", 0)),
        "with_gws": bool(params.get("with_gws", False)),
        "source": "cosmolattice",
    }
    out = os.path.join(run_dir, "simulation_metadata.npz")
    np.savez(out, **meta)
    return out


MANIFEST_FIELDS = ("step", "t", "T", "a", "H", "fStar", "n_scalars", "filename")


def _parse_manifest_row(line):
    """Parse one manifest.csv line (with or without header row)."""
    line = line.strip()
    if not line or line.startswith("step,"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < len(MANIFEST_FIELDS):
        return None
    return dict(zip(MANIFEST_FIELDS, parts))


def export_run(run_dir, keep_raw=False, hubble=True):
    state_dir = os.path.join(run_dir, "field_states")
    manifest = os.path.join(state_dir, "manifest.csv")
    if not os.path.isdir(state_dir):
        raise FileNotFoundError(f"no field_states/ in {run_dir}")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"no manifest.csv in {state_dir}")

    params = load_run_params(run_dir)
    n_exported = 0
    seen_steps = set()

    with open(manifest, newline="") as f:
        lines = list(f)

    rows = []
    for line in lines:
        row = _parse_manifest_row(line)
        if row is None:
            continue
        step_key = int(float(row["step"]))
        if step_key in seen_steps:
            continue
        seen_steps.add(step_key)
        rows.append(row)

    print(f"Exporting {len(rows)} unique snapshots from {run_dir}")
    for i, row in enumerate(rows):
        raw_name = row["filename"].strip()
        raw_path = os.path.join(state_dir, raw_name)
        if not os.path.exists(raw_path):
            print(f"  [{i+1}/{len(rows)}] skip missing {raw_name}")
            continue
        out_probe_step = int(float(row["step"]))
        out_probe = os.path.join(state_dir, f"state_step_{out_probe_step:010d}.npz")
        if os.path.exists(out_probe):
            print(f"  [{i+1}/{len(rows)}] skip existing state_step_{out_probe_step:010d}.npz", flush=True)
            n_exported += 1
            continue
        print(f"  [{i+1}/{len(rows)}] {raw_name} ...", flush=True)
        try:
            snap = read_raw_snapshot(raw_path)
        except Exception as exc:
            print(f"      FAILED ({exc})", flush=True)
            continue
        step = snap["step"]
        out = os.path.join(state_dir, f"state_step_{step:010d}.npz")

        if snap.get("n_scalars", 1) >= 2:
            phi1 = snap["phi1"]
            phi2 = snap["phi2"]
            rho = snap["rho"]
            winding = snap["winding"]
            save_dict = {
                "phi1": phi1.astype(np.float32),
                "phi2": phi2.astype(np.float32),
                "rho": rho.astype(np.float32),
                "theta": snap["theta"].astype(np.float32),
                "winding": winding.astype(np.float32),
                "step": step,
                "time": snap["time"],
                "temperature": snap["temperature"],
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
            }
            if "pi1" in snap:
                save_dict["pi1"] = snap["pi1"].astype(np.float32)
                save_dict["pi2"] = snap["pi2"].astype(np.float32)
        else:
            phi = snap["phi"]
            save_dict = {
                "phi": phi.astype(np.float32),
                "step": step,
                "time": snap["time"],
                "temperature": snap["temperature"],
                "phi_min": float(phi.min()),
                "phi_max": float(phi.max()),
            }
            if "pi" in snap:
                save_dict["pi"] = snap["pi"].astype(np.float32)

        if hubble and not params.get("no_hubble", False):
            save_dict["scale_factor"] = snap["a"]
            save_dict["hubble"] = snap["H"]

        np.savez_compressed(out, **save_dict)
        n_exported += 1
        n_str = int(np.sum(np.abs(save_dict.get("winding", np.array([]))) > 0.5)) if "winding" in save_dict else 0
        print(
            f"      -> state_step_{step:010d}.npz  "
            f"T={snap['temperature']:.2f}  "
            f"rho_max={float(save_dict.get('rho_max', save_dict.get('phi_max', 0))):.3e}"
            + (f"  string_vox={n_str}" if "winding" in save_dict else ""),
            flush=True,
        )
        if not keep_raw:
            os.remove(raw_path)

    meta_path = write_metadata(run_dir, params, n_exported)
    return n_exported, meta_path


def main():
    ap = argparse.ArgumentParser(description="Export CosmoLattice raw snapshots to numba NPZ")
    ap.add_argument("run_dir", help="CosmoLattice output directory")
    ap.add_argument("--keep-raw", action="store_true", help="Keep .raw files after export")
    ap.add_argument("--no-hubble-meta", action="store_true",
                    help="Do not write scale_factor/hubble into NPZ")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    n, meta = export_run(run_dir, keep_raw=args.keep_raw, hubble=not args.no_hubble_meta)
    print(f"Exported {n} snapshots -> {os.path.join(run_dir, 'field_states')}")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
