#!/usr/bin/env python3
"""Map γ = φ₀/M_Pl to a lattice output set name under data/lattice/<set>/.

Known sets keep historical names. Anything else becomes set_gamma_<value>
so a γ scan never overwrites set8.

Usage:
    python simulation/gamma_sets.py --gamma 1e-5
    python simulation/gamma_sets.py --list
"""
from __future__ import annotations

import argparse
import math
import sys

# Reduced Planck mass used in thermal_inflation.h / run_cosmolattice.py
M_PL = 2.4e18
MPHI_DEFAULT = 1000.0

# Historical lattice / paper tags. set8 is the 1024³ / 256³ γ you already ran.
KNOWN_SETS: dict[str, float] = {
    "set8": 4.1667e-4,   # φ₀ = 1e15 GeV, V₀ ~ 2.5e35 GeV⁴
    "setA": 4.1667e-8,   # φ₀ = 1e11 GeV, V₀ ~ 2.5e27 GeV⁴
}

SET8_GAMMA = KNOWN_SETS["set8"]


def gammas_close(a: float, b: float, rel: float = 1e-4) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= rel


def format_gamma_tag(gamma: float) -> str:
    """Filesystem-safe compact γ, e.g. 4.1667e-4, 1e-5."""
    s = f"{gamma:.4g}".replace("+", "")
    return s


def set_name_for_gamma(gamma: float) -> str:
    """Return the data/lattice/<name> folder for this γ."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    for name, g in KNOWN_SETS.items():
        if gammas_close(gamma, g):
            return name
    return f"set_gamma_{format_gamma_tag(gamma)}"


def gamma_for_set(name: str) -> float | None:
    return KNOWN_SETS.get(name)


def v0_of_gamma(gamma: float, mphi: float = MPHI_DEFAULT) -> float:
    """Tree V₀ = m² φ₀² / 4 = m² (γ M_Pl)² / 4."""
    phi0 = gamma * M_PL
    return 0.25 * mphi * mphi * phi0 * phi0


def lambda_of_gamma(gamma: float, mphi: float = MPHI_DEFAULT) -> float:
    phi0 = gamma * M_PL
    return (mphi * mphi) / (phi0 * phi0)


def resolve_param_set(gamma: float, param_set: str | None, auto: bool) -> str:
    """Pick output set. auto or param_set='auto' → from γ.

    If param_set is the default set8 but γ is not set8's γ, switch to the
    γ-derived name so a scan cannot clobber set8.
    """
    if auto or param_set in (None, "", "auto"):
        return set_name_for_gamma(gamma)
    if param_set == "set8" and not gammas_close(gamma, SET8_GAMMA):
        mapped = set_name_for_gamma(gamma)
        print(
            f"NOTE: --gamma={gamma:g} is not set8 (γ={SET8_GAMMA:g}); "
            f"writing to data/lattice/{mapped}/ instead of set8. "
            f"Pass --param_set {param_set} --force_param_set to override.",
            file=sys.stderr,
        )
        return mapped
    return param_set


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--set", dest="set_name", default=None, help="Look up γ for a known set")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list or (args.gamma is None and args.set_name is None):
        print(f"{'set':<16} {'gamma':>12} {'phi0 [GeV]':>14} {'V0 [GeV^4]':>14}")
        for name, g in KNOWN_SETS.items():
            print(f"{name:<16} {g:12.4e} {g * M_PL:14.4e} {v0_of_gamma(g):14.4e}")
        print("unlisted γ → set_gamma_<value>")
        return
    if args.set_name:
        g = gamma_for_set(args.set_name)
        if g is None:
            sys.exit(f"unknown set {args.set_name!r}; known: {list(KNOWN_SETS)}")
        print(g)
        return
    name = set_name_for_gamma(args.gamma)
    print(name)
    print(
        f"gamma={args.gamma:g}  phi0={args.gamma * M_PL:.4e} GeV  "
        f"lambda={lambda_of_gamma(args.gamma):.4e}  V0={v0_of_gamma(args.gamma):.4e} GeV^4",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
