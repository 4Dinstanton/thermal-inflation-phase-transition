"""Portal potential with U(1)-preserving bilinear mixing (paper Oct 2026).

Extends :class:`PortalPotential` by the real-radial form of

    -μ²_mix (ϕ* S + S* ϕ)  →  V_mix = -μ²_mix φ S   (phase-aligned, φ,S ≥ 0)

as in Eq. (2) of “Sequential False Vacuum Decay and Multi-Peaked Gravitational
Waves at the End of Thermal Inflation”.  Nonzero μ²_mix shifts the intermediate
vacuum to A′ = (φ_off, s_T) with

    φ_off(T) ≃ μ²_mix s_T / M²_φ,A(T)

(Eq. 5), lowers the A→B saddle, and raises the effective first-step latent heat
ε_eff (Eq. 9) — allowing two-step tunneling at larger ε that would otherwise
remain stuck in A.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq

from potential.portalPotential import PortalPotential


@dataclass
class PortalMixedPotential(PortalPotential):
    """PortalPotential + bilinear mixing ``mu2_mix`` [GeV²].

    Parameters
    ----------
    mu2_mix :
        Real-field mixing coefficient in V ⊃ −μ²_mix φ S (GeV²).  Prefer
        ``rmix`` for scans: μ²_mix = rmix · m_S² with rmix ≪ 1.
    rmix :
        If set (and ``mu2_mix`` is 0/None), sets ``mu2_mix = rmix * m_S²`` after
        the parent ``__post_init__``.
    """

    mu2_mix: float = 0.0
    rmix: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.rmix is not None and (self.mu2_mix is None or self.mu2_mix == 0.0):
            self.mu2_mix = float(self.rmix) * self.m_S2
        self.mu2_mix = float(self.mu2_mix or 0.0)
        if self.mu2_mix < 0:
            raise ValueError("mu2_mix must be ≥ 0 (phase-aligned convention)")
        # refresh latent heats; ε_eff evaluated at T=0 A′ below
        self._refresh_delV_eff(T=0.0)

    @property
    def mixed(self) -> bool:
        return self.mu2_mix > 0.0

    def _refresh_delV_eff(self, T: float = 0.0):
        """Update delV1/delV2 from ε_eff at the displaced intermediate min."""
        if not self.mixed:
            self.delV1 = self.eps * self.V0
            self.delV2 = (1.0 - self.eps) * self.V0
            self.eps_eff = self.eps
            return
        vac = self.vacua(T)
        if vac["A"] is None or vac["B"] is None:
            self.eps_eff = self.eps
            self.delV1 = self.eps * self.V0
            self.delV2 = (1.0 - self.eps) * self.V0
            return
        V_A = float(vac["A"][2])
        V_B = float(vac["B"][2])
        # V_rel is shifted so V(O)∼0 at high T; use differences
        dV1 = max(0.0, -V_A)  # O→A latent (O at ~0)
        dV_tot = max(dV1 + max(0.0, V_A - V_B), self.V0 * 1e-30)
        # Prefer paper Eq. (9) when A′ is close to (0,s0)
        self.eps_eff = float(np.clip(dV1 / self.V0, 0.0, 0.999))
        self.delV1 = self.eps_eff * self.V0
        self.delV2 = (1.0 - self.eps_eff) * self.V0
        _ = dV_tot

    # ------------------------------------------------------------- tree
    def V_tree_rel(self, phi, S):
        base = super().V_tree_rel(phi, S)
        if not self.mixed:
            return base
        phi = np.asarray(phi, float)
        S = np.asarray(S, float)
        return base - self.mu2_mix * phi * S

    def grad_tree(self, phi, S):
        dphi, dS = super().grad_tree(phi, S)
        if not self.mixed:
            return dphi, dS
        phi = np.asarray(phi, float)
        S = np.asarray(S, float)
        return dphi - self.mu2_mix * S, dS - self.mu2_mix * phi

    def V_delta(self, phi, S, T: float, S_A: float, phi_A: float = 0.0):
        """V(φ,S,T) − V(φ_A, S_A, T) without huge absolute subtraction."""
        p = np.asarray(phi, float)
        s = np.asarray(S, float)
        pA, sA = float(phi_A), float(S_A)
        # tree relative to A
        tree = (self.V_tree_rel(p, s) - self.V_tree_rel(pA, sA))
        dS = self._dVT(self.species_S, s, T) - self._dVT(self.species_S, sA, T)
        if self.phi_thermal == "set8":
            dphi = (self.set8.V_th_rel(p, T, s) + self.set8.V_cw_rel(p, T)
                    - self.set8.V_th_rel(pA, T, sA) - self.set8.V_cw_rel(pA, T))
        else:
            dphi = (self._dVT(self.species_phi, p, T)
                    - self._dVT(self.species_phi, pA, T))
        return tree + dphi + dS

    def phi_off(self, s_T: float, T: float) -> float:
        """Analytic flaton offset at the intermediate S condensate (Eq. 5)."""
        if not self.mixed or s_T <= 0:
            return 0.0
        M2 = self.mass2_phi(0.0, s_T, T)
        if not (M2 > 0):
            return 0.0
        return float(self.mu2_mix * s_T / M2)

    def mass2_phi(self, phi, S, T):
        # curvature along φ; mixing is linear so Hessian φφ unchanged
        return super().mass2_phi(phi, S, T)

    def mass2_S(self, phi, S, T):
        return super().mass2_S(phi, S, T)

    # ------------------------------------------------------------ vacua
    def vacua(self, T: float) -> dict:
        if not self.mixed:
            return super().vacua(T)

        out = {"O": None, "A": None, "B": None}
        mO_p, mO_s = self.mass2_phi(0.0, 0.0, T), self.mass2_S(0.0, 0.0, T)
        # O stable if thermal curvatures dominate the mixing det
        # Mφ M_S > μ⁴_mix  (paper: M²_φ,O M²_S,O > μ⁴_mix)
        if mO_p > 0 and mO_s > 0 and mO_p * mO_s > self.mu2_mix**2:
            out["O"] = (0.0, 0.0, float(self.V_rel(0.0, 0.0, T)))

        # Seed A′ from unmixed S VEV + analytic φ_off, then polish.
        # Cap φ near the set8 thermal barrier (~few TeV): a larger analytic
        # φ_off means A′ has already rolled past the barrier into the B basin.
        sv = self.slice_S().vacua(T)["true"]
        if sv is not None:
            s_seed = float(sv.phi)
            p_seed = self.phi_off(s_seed, T)
            phi_cap = float(min(max(5.0 * max(T, 100.0), 2.0e3), 8.0e3))
            out["s_T"] = s_seed
            out["phi_off"] = p_seed
            # A_rolled: the S condensate exists but the tilt has removed the
            # φ-minimum, so the intermediate phase rolls classically to B
            # (immediate A→B), as opposed to "no A phase at all".
            out["A_rolled"] = True
            if self.mass2_phi(0.0, s_seed, T) > 0 and p_seed < phi_cap:
                # Find ∂V/∂φ = 0 on [0, φ_cap] at fixed s_seed
                p_hi = min(phi_cap, max(5.0 * max(p_seed, 1.0), 500.0))
                p1 = self._root_phi_at_S(s_seed, T, p_seed, p_hi)
                if p1 is not None:
                    A = self._polish_min(p1, s_seed, T,
                                         bounds=((0.0, p_hi),
                                                 (0.1 * s_seed, 1.5 * s_seed)))
                    if A is None:
                        A = (p1, s_seed, float(self.V_rel(p1, s_seed, T)))
                    if A is not None and self._is_local_min(A[0], A[1], T):
                        out["A"] = A
                        out["s_T"] = A[1]
                        out["phi_off"] = A[0]
                        out["A_rolled"] = False
        else:
            out["s_T"] = float("nan")
            out["phi_off"] = 0.0

        # Seed B′ from unmixed φ VEV + small S_ind
        pv = self.slice_phi(0.0).vacua(T)["true"]
        if pv is not None:
            v_seed = float(pv.phi)
            # S_ind ≃ μ²_mix v / M²_S,B
            M2S = self.mass2_S(v_seed, 0.0, T)
            s_ind = (self.mu2_mix * v_seed / M2S) if M2S > 0 else 0.0
            s_ind = float(np.clip(s_ind, 0.0, min(0.1 * self.s0, 1.0e-3 * v_seed)))
            B = self._polish_min(v_seed, s_ind, T,
                                 bounds=((0.5 * v_seed, 1.5 * v_seed),
                                         (0.0, max(1.0, min(0.2 * self.s0,
                                                            1.0e-2 * v_seed)))))
            if B is not None and self._is_local_min(B[0], B[1], T, for_B=True):
                out["B"] = B
                out["v_T"] = B[0]
                out["S_ind"] = B[1]
            else:
                out["v_T"] = v_seed
                out["S_ind"] = s_ind
                if self.mass2_S(v_seed, 0.0, T) > 0:
                    out["B"] = (v_seed, 0.0, float(self.V_rel(v_seed, 0.0, T)))
        else:
            out["v_T"] = float("nan")
            out["S_ind"] = 0.0
        return out

    def _root_phi_at_S(self, S: float, T: float, p_seed: float, p_hi: float):
        """Solve ∂V/∂φ(φ, S, T) = 0 near p_seed within [0, p_hi]."""
        def gp(p):
            g, _ = self.grad(p, S, T)
            return float(np.asarray(g).ravel()[0])

        # The φ-gradient at fixed S starts negative (−μ² S), crosses zero upward
        # at A′ (the minimum), then again downward at the thermal barrier top.
        # Take the FIRST upward zero crossing scanning from φ = 0: a bracket
        # [0, p_hi] can straddle both roots and brentq / minimize_scalar then
        # land on the barrier top or in the B basin (the old "stuck in A").
        p0 = float(np.clip(p_seed, 0.0, p_hi))
        try:
            g0 = gp(p0)
            if abs(g0) < 1.0:
                return p0
            grid = np.unique(np.concatenate([[0.0],
                                             np.geomspace(1e-3, p_hi, 400)]))
            gv = np.array([gp(p) for p in grid])
            up = np.nonzero((gv[:-1] < 0) & (gv[1:] >= 0))[0]
            if len(up) == 0:
                return None  # no minimum along φ at this S: A′ has rolled over
            i = int(up[0])
            return float(brentq(gp, grid[i], grid[i + 1], xtol=1e-8))
        except Exception:
            return None

    def _is_local_min(self, phi, S, T, for_B: bool = False) -> bool:
        """Require positive curvatures and small gradient (true critical point)."""
        try:
            gp, gs = self.grad(phi, S, T)
            gp = float(np.asarray(gp).ravel()[0])
            gs = float(np.asarray(gs).ravel()[0])
        except Exception:
            return False
        m2p = float(self.mass2_phi(phi, S, T))
        m2s = float(self.mass2_S(phi, S, T))
        if not (m2p > 0 and m2s > 0):
            return False
        # φ-direction must be a true critical point (set8 + mix are stiff in S)
        gp_cut = 1e-3 * m2p * max(abs(phi), abs(T), 1.0)
        if abs(gp) > max(gp_cut, 1.0):
            return False
        if for_B:
            gs_cut = 1e-3 * m2s * max(abs(S), abs(T), 1.0)
            if abs(gs) > max(gs_cut, 1.0):
                return False
        return True

    def _polish_min(self, phi0, S0, T, bounds):
        def f(x):
            return float(self.V_rel(x[0], x[1], T))

        try:
            r = minimize(f, x0=np.array([phi0, S0], float),
                         bounds=bounds, method="L-BFGS-B",
                         options={"ftol": 1e-14, "gtol": 1e-10})
        except Exception:
            return None
        if not r.success:
            return None
        p, s = float(r.x[0]), float(r.x[1])
        return (p, s, float(self.V_rel(p, s, T)))

    def summary(self) -> str:
        base = super().summary()
        extra = (f"  bilinear mix: mu2_mix = {self.mu2_mix:.4e} GeV²"
                 f"  (rmix = mu2_mix/m_S² = {self.mu2_mix / self.m_S2:.4e})"
                 if self.mixed else "  bilinear mix: off (mu2_mix = 0)")
        if self.mixed:
            vac0 = self.vacua(0.0)
            extra += (f"\n  T=0 A′ φ_off = {vac0.get('phi_off', float('nan')):.4e}, "
                      f"eps_eff ≈ {getattr(self, 'eps_eff', self.eps):.4e}")
        return base + "\n" + extra
