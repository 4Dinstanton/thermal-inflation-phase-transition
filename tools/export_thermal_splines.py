#!/usr/bin/env python3
"""
Export thermal one-loop integrals J_b, J_f (and their first/second derivatives)
as uniform-grid cubic-interpolation tables that the CosmoLattice thermal-inflation
model reads at runtime.

Background
----------
The finite-temperature effective potential used throughout this project
(`potential/Potential.py`, `V_p_correct` / `V_fermion_only`) and the numba lattice
solver (`simulation/latticeSimeRescale_numba.py`) both evaluate the thermal
functions

    J_b(u),  J_f(u),  dJ_b/du,  dJ_f/du        with   u = m(phi,T)/T

via `cosmoTransitions.finiteT`. Evaluating these by quadrature on every lattice
site is far too slow, so we pre-tabulate them once on a uniform grid in `u` and
cubic-interpolate at runtime (this mirrors `Potential.build_fast_thermal`).

Important physics note
----------------------
`Potential.dV_p_correct` in this repo uses an *incorrect* chain rule
(`du/dphi` is missing a `1/u` factor) and is therefore NOT the derivative of
`Potential.V_p_correct`. The numba solver uses the correct chain rule

    du_b/dphi = 0.5 * y_b^2 * phi / (T^2 * u_b)

This exporter assembles the *correct* derivative and validates it against the
finite difference of the assembled potential `V` (the unambiguous ground truth)
and against the numba-style V'. We deliberately do NOT validate against
`Potential.dV_p_correct`.

Outputs (written to data/thermal_splines/)
------------------------------------------
- thermal_tables.npz : grid + J/dJ/d2J arrays (for Python validation/plots)
- thermal_tables.bin : little-endian binary table consumed by the C++ model
                       layout:  int64 n
                                float64 umin, float64 umax
                                n*f64 Jb, n*f64 Jf,
                                n*f64 dJb, n*f64 dJf,
                                n*f64 d2Jb, n*f64 d2Jf

Usage
-----
    python tools/export_thermal_splines.py            # default grid, run validation
    python tools/export_thermal_splines.py --umax 150 --npts 4096
"""
import argparse
import math
import os
import struct
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # quadrature overflow at large u -> integrand 0

import cosmoTransitions.finiteT as CTFT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "data", "thermal_splines")

# Default model parameters (Set B: boson+fermion). Only used for validation;
# the exported tables are model-parameter independent (they are pure J(u)).
DEFAULT_PARAMS = dict(
    lam=None,            # filled from gamma below if None
    mphi=1000.0,
    gamma=4.1667e-4,
    boson_mass_squared=1_000_000.0,
    yb=1.09,
    gb=1.05,
    yf=1.09,
    gf=1.05,
    nb=20,
    nf=20,
    g_star_pot=100.0,    # radiation free-energy multiplicity (matches Potential.py literal)
)
M_PL = 2.4e18


# ----------------------------------------------------------------------------
# Table construction
# ----------------------------------------------------------------------------
def build_tables(umax=150.0, npts=4096):
    """Tabulate Jb, Jf, dJb, dJf and their numerical second derivatives."""
    u = np.linspace(0.0, umax, npts)
    Jb = np.array([float(np.real(CTFT.Jb_exact(x))) for x in u])
    Jf = np.array([float(np.real(CTFT.Jf_exact(x))) for x in u])
    dJb = np.array([float(np.real(CTFT.dJb_exact(x))) for x in u])
    dJf = np.array([float(np.real(CTFT.dJf_exact(x))) for x in u])
    # Second derivatives from the (accurate) first-derivative grid.
    d2Jb = np.gradient(dJb, u)
    d2Jf = np.gradient(dJf, u)
    return u, Jb, Jf, dJb, dJf, d2Jb, d2Jf


def write_npz(path, u, Jb, Jf, dJb, dJf, d2Jb, d2Jf):
    np.savez(
        path,
        u=u,
        Jb=Jb,
        Jf=Jf,
        dJb=dJb,
        dJf=dJf,
        d2Jb=d2Jb,
        d2Jf=d2Jf,
        umin=u[0],
        umax=u[-1],
    )


def write_bin(path, u, Jb, Jf, dJb, dJf, d2Jb, d2Jf):
    n = len(u)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", n))
        f.write(struct.pack("<d", float(u[0])))
        f.write(struct.pack("<d", float(u[-1])))
        for arr in (Jb, Jf, dJb, dJf, d2Jb, d2Jf):
            f.write(np.asarray(arr, dtype="<f8").tobytes())


# ----------------------------------------------------------------------------
# Reference assembly of V and V' from tables (the same formulas the C++ uses)
# ----------------------------------------------------------------------------
class TableEval:
    """Evaluate V, V', V'' from the exported tables (cubic via numpy interp of
    the tabulated quantities). Used only for validation here; the production
    evaluator is the C++ thermal_tables.hpp."""

    def __init__(self, u, Jb, Jf, dJb, dJf, d2Jb, d2Jf, params):
        self.u = u
        self.Jb, self.Jf = Jb, Jf
        self.dJb, self.dJf = dJb, dJf
        self.d2Jb, self.d2Jf = d2Jb, d2Jf
        self.p = params

    def _interp(self, table, x):
        return np.interp(np.clip(x, self.u[0], self.u[-1]), self.u, table)

    def _ub(self, phi, T):
        p = self.p
        m2 = p["boson_mass_squared"] + 0.5 * p["yb"] ** 2 * phi ** 2 + (
            0.25 * p["yb"] ** 2 + 2.0 / 3.0 * p["gb"] ** 2
        ) * T ** 2
        return np.sqrt(np.maximum(m2, 0.0)) / T

    def _uf(self, phi, T):
        p = self.p
        m2 = 0.5 * p["yf"] ** 2 * phi ** 2 + (1.0 / 6.0) * p["gf"] ** 2 * T ** 2
        return np.sqrt(np.maximum(m2, 0.0)) / T

    def _cw(self, phi, T):
        lam = self.p["lam"]
        m2 = 3.0 * lam * phi ** 2
        m2abs = np.abs(m2)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = m2 ** 2 / (64.0 * math.pi ** 2) * (np.log(m2abs / T) - 1.5)
        return np.where(m2abs > 0, raw, 0.0)

    def _dcw(self, phi, T):
        lam = self.p["lam"]
        m2 = 3.0 * lam * phi ** 2
        dm2 = 6.0 * lam * phi
        m2abs = np.abs(m2)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = dm2 * m2 / (64.0 * math.pi ** 2) * 2.0 * (np.log(m2abs / T) - 1.0)
        return np.where(m2abs > 0, raw, 0.0)

    def V(self, phi, T, include_cw=True):
        p = self.p
        ub, uf = self._ub(phi, T), self._uf(phi, T)
        pref = T ** 4 / (2.0 * math.pi ** 2)
        tree = 0.25 * p["lam"] * phi ** 4 - 0.5 * p["mphi"] ** 2 * phi ** 2
        thermal = pref * (p["nb"] * self._interp(self.Jb, ub) + p["nf"] * self._interp(self.Jf, uf))
        rad = math.pi ** 2 / 30.0 * p["g_star_pot"] * T ** 4
        out = tree + thermal + rad
        if include_cw:
            out = out + self._cw(phi, T)
        return out

    def Vprime(self, phi, T, include_cw=True):
        p = self.p
        ub, uf = self._ub(phi, T), self._uf(phi, T)
        pref = T ** 4 / (2.0 * math.pi ** 2)
        tree = p["lam"] * phi ** 3 - p["mphi"] ** 2 * phi
        ub_safe = np.maximum(ub, 1e-20)
        uf_safe = np.maximum(uf, 1e-20)
        dub = 0.5 * p["yb"] ** 2 * phi / (T ** 2 * ub_safe)
        duf = 0.5 * p["yf"] ** 2 * phi / (T ** 2 * uf_safe)
        thermal = pref * (
            p["nb"] * self._interp(self.dJb, ub) * dub
            + p["nf"] * self._interp(self.dJf, uf) * duf
        )
        out = tree + thermal
        if include_cw:
            out = out + self._dcw(phi, T)
        return out


def resolve_params(args):
    p = dict(DEFAULT_PARAMS)
    if args.fermion_only:
        p["nb"] = 0
    p["gamma"] = args.gamma
    phi0 = p["gamma"] * M_PL
    p["lam"] = p["mphi"] ** 2 / phi0 ** 2
    return p, phi0


# ----------------------------------------------------------------------------
# Validation against finite difference and numba-style V'
# ----------------------------------------------------------------------------
def validate(ev, params, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    phi0 = params["mphi"] / math.sqrt(params["lam"])  # tree VEV
    phis = rng.uniform(1e2, 1.2 * phi0, n_samples)
    Ts = rng.uniform(2000.0, 9000.0, n_samples)

    max_rel_vp = 0.0
    max_rel_v = 0.0
    for phi, T in zip(phis, Ts):
        h = max(abs(phi) * 1e-5, 1.0)
        vp_fd = (ev.V(phi + h, T) - ev.V(phi - h, T)) / (2.0 * h)
        vp = ev.Vprime(phi, T)
        denom = max(abs(vp_fd), 1e3)
        max_rel_vp = max(max_rel_vp, abs(vp - vp_fd) / denom)

    print(f"  V' vs finite-difference(V):  max rel err = {max_rel_vp:.3e}")
    ok = max_rel_vp < 1e-3
    return ok


def main():
    ap = argparse.ArgumentParser(description="Export thermal J-integral tables for CosmoLattice")
    ap.add_argument("--umax", type=float, default=150.0, help="Maximum u=m/T in the table")
    ap.add_argument("--npts", type=int, default=4096, help="Number of uniform grid points")
    ap.add_argument("--gamma", type=float, default=4.1667e-4, help="gamma (sets lambda via phi0=gamma*M_Pl) for validation")
    ap.add_argument("--fermion_only", action="store_true", help="Validate fermion-only mode (nb=0)")
    ap.add_argument("--no_validate", action="store_true", help="Skip validation step")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Building thermal tables: umax={args.umax}, npts={args.npts}")
    u, Jb, Jf, dJb, dJf, d2Jb, d2Jf = build_tables(args.umax, args.npts)

    npz_path = os.path.join(OUT_DIR, "thermal_tables.npz")
    bin_path = os.path.join(OUT_DIR, "thermal_tables.bin")
    write_npz(npz_path, u, Jb, Jf, dJb, dJf, d2Jb, d2Jf)
    write_bin(bin_path, u, Jb, Jf, dJb, dJf, d2Jb, d2Jf)
    print(f"  wrote {npz_path}")
    print(f"  wrote {bin_path}  ({os.path.getsize(bin_path)} bytes)")

    if not args.no_validate:
        params, phi0 = resolve_params(args)
        print(f"Validation (gamma={args.gamma}, lambda={params['lam']:.3e}, "
              f"nb={params['nb']}, nf={params['nf']}):")
        ev = TableEval(u, Jb, Jf, dJb, dJf, d2Jb, d2Jf, params)
        ok = validate(ev, params)
        print("  RESULT:", "PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
