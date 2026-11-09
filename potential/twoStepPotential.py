"""Two-step (0 -> a -> v) thermal-inflation flaton potential.

Tree level (paper Eq. 2 with the tadpole set to zero):

    V_tree(phi) = V0 + c2 phi^2 + c3 phi^3 + c4 phi^4 + c6 phi^6
                = V0 - m0^2/2 phi^2 + mu3/3 phi^3 - lam/4 phi^4 + phi^6/(6 Lambda^2)

The four coefficients are fixed by the vacuum data

    V_tree(a) = V0 - delV1,   V_tree(v) = 0,   V_tree'(a) = V_tree'(v) = 0,

which is a linear system in (c2, c3, c4, c6). The thermal part (paper Eq. 4)

    dV_T(phi, T) = T^4/(2 pi^2) [ sum_B n_B J_B(g_B^2 phi^2/T^2)
                                + sum_F n_F J_F(g_F^2 phi^2/T^2) ]

traps the field at phi = 0 at high T and decouples at phi >~ T/g.

The odd term is implemented as c3 |phi|^3, i.e. V depends on |phi| (phi is the
radial mode, or a real field with that Z2-symmetric extension). With an analytic
c3 phi^3 a real field has a much deeper minimum at phi < 0 that the field reaches
through the same thermal barrier, and the 0 -> a -> v sequence never happens.

All quantities are in GeV. ``V_rel`` returns V - V0 - dV_T(0, T), i.e. the
potential measured from the thermal false vacuum, so the huge constant V0
never enters a subtraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from cosmoTransitions import finiteT as CTFT

M_PL = 2.435e18  # reduced Planck mass (GeV), as in the paper


@dataclass
class Species:
    """Thermal-bath species with field-dependent mass m = g * phi."""

    n: float  # degrees of freedom (positive for both bosons and fermions)
    g: float  # coupling to phi
    fermion: bool = False


@dataclass
class Extremum:
    phi: float
    V_rel: float
    kind: str  # "min" or "max"


@dataclass
class TwoStepPotential:
    V0: float
    a: float
    v: float
    delV1: float
    species: List[Species] = field(default_factory=list)
    g_star: float = 100.0  # relativistic dof of the bath (Hubble / rho_rad)

    def __post_init__(self):
        if not (0.0 < self.a < self.v):
            raise ValueError(f"need 0 < a < v, got a={self.a:g}, v={self.v:g}")
        if not (0.0 < self.delV1 < self.V0):
            raise ValueError(
                f"need 0 < delV1 < V0, got delV1={self.delV1:g}, V0={self.V0:g}"
            )
        self.c2, self.c3, self.c4, self.c6 = solve_coefficients(
            self.V0, self.a, self.v, self.delV1
        )

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_vacua(
        cls,
        V0: float,
        a: float,
        v: float,
        delV1: Optional[float] = None,
        delV1_frac: Optional[float] = None,
        bosons: Sequence[Tuple[float, float]] = ((10.0, 1.0),),
        fermions: Sequence[Tuple[float, float]] = (),
        g_star: float = 100.0,
    ) -> "TwoStepPotential":
        """Build from V0, a, v and delV1 (absolute) or delV1_frac = delV1 / V0.

        ``bosons`` / ``fermions`` are (n_dof, g) pairs with m = g * phi.
        """
        if (delV1 is None) == (delV1_frac is None):
            raise ValueError("give exactly one of delV1 or delV1_frac")
        if delV1 is None:
            delV1 = delV1_frac * V0
        species = [Species(n, g, False) for n, g in bosons]
        species += [Species(n, g, True) for n, g in fermions]
        return cls(V0=V0, a=a, v=v, delV1=delV1, species=species, g_star=g_star)

    # ------------------------------------------------------- named parameters
    @property
    def delV2(self) -> float:
        return self.V0 - self.delV1

    @property
    def m0_sq(self) -> float:
        return -2.0 * self.c2

    @property
    def mu3(self) -> float:
        return 3.0 * self.c3

    @property
    def lam(self) -> float:
        return -4.0 * self.c4

    @property
    def Lambda(self) -> float:
        return float(np.sqrt(1.0 / (6.0 * self.c6))) if self.c6 > 0 else float("nan")

    @property
    def thermal_c(self) -> float:
        """c in the high-T thermal mass c T^2 phi^2 / 2 (bosons g^2 n/12, fermions g^2 n/24)."""
        return sum(s.n * s.g**2 / (24.0 if s.fermion else 12.0) for s in self.species)

    @property
    def T0_spinodal(self) -> float:
        """T below which phi=0 is tachyonic in the high-T expansion: sqrt(m0^2/c)."""
        c = self.thermal_c
        return float(np.sqrt(self.m0_sq / c)) if c > 0 and self.m0_sq > 0 else float("nan")

    # ------------------------------------------------------------- tree level
    def V_tree_rel(self, phi):
        """V_tree - V0 (exact, no V0 cancellation)."""
        phi = np.asarray(phi, dtype=float)
        p2 = phi**2
        return p2 * (self.c2 + np.abs(phi) * self.c3 + p2 * (self.c4 + p2 * self.c6))

    def V_tree(self, phi):
        return self.V0 + self.V_tree_rel(phi)

    def dV_tree(self, phi):
        phi = np.asarray(phi, dtype=float)
        p2 = phi * phi
        return phi * (2 * self.c2 + 3 * self.c3 * np.abs(phi) + 4 * self.c4 * p2 + 6 * self.c6 * p2 * p2)

    def d2V_tree(self, phi):
        phi = np.asarray(phi, dtype=float)
        p2 = phi * phi
        return 2 * self.c2 + 6 * self.c3 * np.abs(phi) + 12 * self.c4 * p2 + 30 * self.c6 * p2 * p2

    def tree_extrema(self) -> List[float]:
        """Positive real roots of V_tree'/phi (a, b, v, ...), sorted."""
        roots = np.roots([6 * self.c6, 0.0, 4 * self.c4, 3 * self.c3, 2 * self.c2])
        real = roots[np.abs(roots.imag) <= 1e-9 * np.abs(roots).max()].real
        return sorted(float(r) for r in real if r > 0)

    def validate_tree(self) -> dict:
        """Check the intended shape: max at 0, min at a, barrier b in (a, v), min at v."""
        ext = self.tree_extrema()
        between = [r for r in ext if self.a * (1 + 1e-9) < r < self.v * (1 - 1e-9)]
        b = between[0] if len(between) == 1 else float("nan")
        out = {
            "signs (c2<0, c3>0, c4<0, c6>0)": (
                self.c2 < 0 and self.c3 > 0 and self.c4 < 0 and self.c6 > 0
            ),
            "origin is tree max": self.c2 < 0,
            "a is min": float(self.d2V_tree(self.a)) > 0,
            "v is min": float(self.d2V_tree(self.v)) > 0,
            "single barrier b in (a,v)": len(between) == 1,
            "b": b,
            "V(b) - V0": float(self.V_tree_rel(b)) if np.isfinite(b) else float("nan"),
        }
        out["ok"] = all(
            out[k]
            for k in ("origin is tree max", "a is min", "v is min", "single barrier b in (a,v)")
        )
        return out

    # ---------------------------------------------------------------- thermal
    def dV_T_rel(self, phi, T: float):
        """dV_T(phi, T) - dV_T(0, T) (>= 0)."""
        phi = np.asarray(phi, dtype=float)
        out = np.zeros_like(phi)
        if T <= 0:
            return out
        pref = T**4 / (2 * np.pi**2)
        for s in self.species:
            J = CTFT.Jf_spline if s.fermion else CTFT.Jb_spline
            out = out + pref * s.n * (J((s.g * phi / T) ** 2) - J(0.0))
        return out

    def dV_T0(self, T: float) -> float:
        """dV_T(0, T): the (phi-independent) free energy of the bath species."""
        if T <= 0:
            return 0.0
        pref = T**4 / (2 * np.pi**2)
        return float(
            sum(pref * s.n * (CTFT.Jf_spline(0.0) if s.fermion else CTFT.Jb_spline(0.0))
                for s in self.species)
        )

    def ddV_T(self, phi, T: float):
        phi = np.asarray(phi, dtype=float)
        out = np.zeros_like(phi)
        if T <= 0:
            return out
        for s in self.species:
            J = CTFT.Jf_spline if s.fermion else CTFT.Jb_spline
            # d/dphi [T^4/(2pi^2) n J(g^2 phi^2/T^2)] = n g^2 T^2 phi J'(theta) / pi^2
            out = out + s.n * s.g**2 * T**2 * phi * J((s.g * phi / T) ** 2, n=1) / np.pi**2
        return out

    def d2dV_T(self, phi, T: float):
        phi = np.asarray(phi, dtype=float)
        out = np.zeros_like(phi)
        if T <= 0:
            return out
        for s in self.species:
            J = CTFT.Jf_spline if s.fermion else CTFT.Jb_spline
            th = (s.g * phi / T) ** 2
            out = out + s.n * s.g**2 * T**2 / np.pi**2 * (J(th, n=1) + 2 * th * J(th, n=2))
        return out

    # ------------------------------------------------------------------ total
    def V_rel(self, phi, T: float):
        """V(phi, T) - V(0, T)."""
        return self.V_tree_rel(phi) + self.dV_T_rel(phi, T)

    def V(self, phi, T: float):
        return self.V0 + self.dV_T0(T) + self.V_rel(phi, T)

    def dV(self, phi, T: float):
        return self.dV_tree(phi) + self.ddV_T(phi, T)

    def d2V(self, phi, T: float):
        return self.d2V_tree(phi) + self.d2dV_T(phi, T)

    # --------------------------------------------------------------- extrema
    def extrema(self, T: float, n_grid: int = 4000) -> List[Extremum]:
        """All extrema of V(., T) on [0, 1.3 v], refined with brentq on dV.

        The grid is log-spaced so the thermal core (phi ~ T/g), a and v are all
        resolved even when they are separated by many decades.
        """
        lo = min([self.a * 1e-6] + [T / s.g * 1e-3 for s in self.species if s.g > 0 and T > 0])
        grid = np.concatenate([[0.0], np.geomspace(lo, 1.3 * self.v, n_grid)])
        d = self.dV(grid, T)
        ext: List[Extremum] = []
        # phi = 0 is always stationary (V is even in m, tree has no tadpole)
        kind0 = "min" if float(self.d2V(0.0, T)) > 0 else "max"
        ext.append(Extremum(0.0, 0.0, kind0))
        s = np.sign(d[1:])
        for i in np.nonzero(s[:-1] * s[1:] < 0)[0]:
            x0, x1 = grid[i + 1], grid[i + 2]
            r = brentq(lambda x: float(self.dV(x, T)), x0, x1, xtol=1e-14 * x1, rtol=1e-13)
            kind = "min" if s[i] < 0 < s[i + 1] else "max"
            ext.append(Extremum(r, float(self.V_rel(r, T)), kind))
        return ext

    def vacua(self, T: float) -> dict:
        """Locate the false (0), intermediate (~a) and true (~v) minima at T.

        Returns phi and V_rel for each that exists, plus the barrier tops
        between consecutive minima. Missing pieces are None.
        """
        ext = self.extrema(T)
        mins = [e for e in ext if e.kind == "min"]
        out = {"false": None, "inter": None, "true": None, "b1": None, "b2": None, "b02": None}
        if ext[0].kind == "min":
            out["false"] = ext[0]
        nonzero = [m for m in mins if m.phi > 0]
        if nonzero:
            true = min(nonzero, key=lambda m: abs(np.log(m.phi / self.v)))
            if abs(np.log(true.phi / self.v)) < np.log(3.0):
                out["true"] = true
            inter_cands = [m for m in nonzero if m is not true and m.phi < (true.phi if out["true"] else np.inf)]
            if inter_cands:
                out["inter"] = min(inter_cands, key=lambda m: abs(np.log(m.phi / self.a)))

        def top(lo, hi):
            maxs = [e for e in ext if e.kind == "max" and lo < e.phi < hi]
            return max(maxs, key=lambda e: e.V_rel) if maxs else None

        if out["false"] and out["inter"]:
            out["b1"] = top(0.0, out["inter"].phi)
        if out["inter"] and out["true"]:
            out["b2"] = top(out["inter"].phi, out["true"].phi)
        if out["false"] and out["true"]:
            out["b02"] = top(0.0, out["true"].phi)
        return out

    # ---------------------------------------------------------- cosmology
    def rho_rad(self, T):
        return np.pi**2 / 30.0 * self.g_star * np.asarray(T, dtype=float) ** 4

    def hubble(self, T, V_false: float):
        return np.sqrt((self.rho_rad(T) + V_false) / (3.0 * M_PL**2))

    # ------------------------------------------------------------- summary
    def summary(self) -> str:
        val = self.validate_tree()
        lines = [
            "TwoStepPotential",
            f"  V0 = {self.V0:.4e} GeV^4  (V0^1/4 = {self.V0**0.25:.4e} GeV)",
            f"  a  = {self.a:.4e} GeV,  v = {self.v:.4e} GeV,  a/v = {self.a / self.v:.3e}",
            f"  delV1 = {self.delV1:.4e} (= {self.delV1 / self.V0:.3e} V0),"
            f"  delV2 = {self.delV2:.4e} (= {self.delV2 / self.V0:.3e} V0)",
            f"  c2 = {self.c2:.4e}  ->  m0^2   = {self.m0_sq:.4e} GeV^2  (m0 = {np.sqrt(abs(self.m0_sq)):.4e})",
            f"  c3 = {self.c3:.4e}  ->  mu3    = {self.mu3:.4e} GeV",
            f"  c4 = {self.c4:.4e}  ->  lambda = {self.lam:.4e}",
            f"  c6 = {self.c6:.4e}  ->  Lambda = {self.Lambda:.4e} GeV",
            f"  tree barrier b = {val['b']:.4e} GeV,  V(b)-V0 = {val['V(b) - V0']:.4e}",
            f"  tree shape ok: {val['ok']}  (strict paper signs: {val['signs (c2<0, c3>0, c4<0, c6>0)']})",
            f"  thermal c = {self.thermal_c:.4g},  T0 = sqrt(m0^2/c) = {self.T0_spinodal:.4e} GeV",
        ]
        for s in self.species:
            kind = "F" if s.fermion else "B"
            lines.append(f"  species {kind}: n={s.n:g}, g={s.g:g}  (a*g = {self.a * s.g:.3e} GeV)")
        return "\n".join(lines)


@dataclass
class SingleStepPotential(TwoStepPotential):
    """Reference single-step flaton V0 - m^2 phi^2/2 + phi^6/(6 Lambda^2) with V(v) = 0.

    Same V0, v and bath species as a two-step model but no intermediate minimum,
    so 0 -> v is the only transition. ``a`` and ``delV1`` are set to v and 0.
    """

    def __post_init__(self):
        if not (self.v > 0 and self.V0 > 0):
            raise ValueError("need v > 0 and V0 > 0")
        self.a, self.delV1 = self.v, 0.0
        self.c2 = -1.5 * self.V0 / self.v**2
        self.c3 = self.c4 = 0.0
        self.c6 = 0.5 * self.V0 / self.v**6

    @classmethod
    def from_vacuum(cls, V0: float, v: float, bosons=((10.0, 1.0),), fermions=(),
                    g_star: float = 100.0) -> "SingleStepPotential":
        species = [Species(n, g, False) for n, g in bosons]
        species += [Species(n, g, True) for n, g in fermions]
        return cls(V0=V0, a=v, v=v, delV1=0.0, species=species, g_star=g_star)

    def validate_tree(self) -> dict:
        return {"signs (c2<0, c3>0, c4<0, c6>0)": False,
                "origin is tree max": True, "v is min": True, "b": float("nan"),
                "V(b) - V0": float("nan"), "ok": True}


def delV1_edge_frac(a_over_v: float, tol: float = 1e-15) -> float:
    """Smallest delV1/V0 for which phi = a is still a tree-level minimum.

    At the edge a merges with the a -> v barrier b (inflection point), so the
    a -> v barrier is shallowest just above it. Independent of V0 and v.
    """

    def is_min(d):
        k2, k3, k4, k6 = (np.array(solve_coefficients(1.0, a_over_v, 1.0, d)))
        x = a_over_v
        return 2 * k2 + 6 * k3 * x + 12 * k4 * x**2 + 30 * k6 * x**4 > 0

    lo, hi = 1e-300, 1.0 - 1e-12
    if is_min(lo):
        return 0.0
    if not is_min(hi):
        return float("nan")
    for _ in range(4000):
        mid = math.sqrt(lo * hi) if hi / lo > 4 else 0.5 * (lo + hi)
        if is_min(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * hi:
            break
    return hi


def solve_coefficients(V0: float, a: float, v: float, delV1: float) -> Tuple[float, float, float, float]:
    """Solve V(a)=V0-delV1, V(v)=0, V'(a)=V'(v)=0 for (c2, c3, c4, c6).

    Solved in units phi = v x, V = V0 u so the 4x4 system is well conditioned
    even when a/v and delV1/V0 span many decades.
    """
    e = a / v
    d = delV1 / V0
    # unknowns k = (k2, k3, k4, k6) with V/V0 = 1 + k2 x^2 + k3 x^3 + k4 x^4 + k6 x^6
    A = np.array(
        [
            [2.0, 3.0 * e, 4.0 * e**2, 6.0 * e**4],  # V'(a)/a = 0
            [2.0, 3.0, 4.0, 6.0],                    # V'(v)/v = 0
            [e**2, e**3, e**4, e**6],                # V(a) - V0 = -delV1
            [1.0, 1.0, 1.0, 1.0],                    # V(v) - V0 = -V0
        ]
    )
    rhs = np.array([0.0, 0.0, -d, -1.0])
    # Row-scale the a-equations: they are O(e^2) and O(e^0) otherwise.
    scale = np.array([1.0, 1.0, 1.0 / e**2, 1.0])
    k = np.linalg.solve(A * scale[:, None], rhs * scale)
    k2, k3, k4, k6 = k
    return (V0 * k2 / v**2, V0 * k3 / v**3, V0 * k4 / v**4, V0 * k6 / v**6)


if __name__ == "__main__":
    pot = TwoStepPotential.from_vacua(V0=1e28, a=3e9, v=1e11, delV1_frac=1e-6)
    print(pot.summary())
    for name, phi, target in [("0", 0.0, pot.V0), ("a", pot.a, pot.V0 - pot.delV1), ("v", pot.v, 0.0)]:
        print(f"  check V({name}) = {pot.V_tree(phi):.6e} (target {target:.6e}),"
              f" V'({name}) = {float(pot.dV_tree(phi)):.3e}")
