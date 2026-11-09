#!/usr/bin/env python3
"""Thin-wall Monte Carlo for nested bubbles in a two-step first-order transition.

Geometry-only simulation: bubbles are spheres that nucleate as a Poisson
process in comoving space / conformal time and expand at constant velocity.
No field equations are solved. The model has three nucleation channels

    A : O -> A   rate  G1(eta) = (b1^4 / 8 pi v1^3) exp[b1 (eta - eta1*)]  in O only
    B : A -> B   rate  G2(eta) = (b2^4 / 8 pi v2^3) exp[b2 (eta - eta2*)]  in A only
    D : O -> B   rate  r0 * G2(eta)                                           in O only

with b_i the comoving inverse durations.  The normalisation is such that
I_i(eta_i*) = 1 for an unobstructed single-step transition (Guth-Weinberg).

Units: b1 = 1 (conformal time in units of 1/b1), lengths in units of 1/b1,
velocities in units of c.  The mean A-bubble separation is
R1 = (8 pi)^(1/3) v1 / b1, and R2 = (8 pi)^(1/3) v2 / b2.

Phase at (x, eta): B if inside any B/D sphere, else A if inside any A sphere,
else O.  A B sphere keeps expanding at v2 after it leaves its parent A sphere;
the part of its surface outside every A sphere is a composite O->B wall.

Wall worldlines: for each sphere we follow N_dir directions on the unit
sphere.  An A-wall element dies when it enters another A sphere (collision)
or a B sphere (overtaken).  A B-wall element dies when it enters another B
sphere.  Death times are found in closed form (linear for equal speeds,
quadratic otherwise).

Resource notes: single-threaded numpy (thread limits are set below before the
numpy import); peak memory per run ~ ndir x (neighbours) x 8 bytes x few, i.e.
tens of MB.  Run several seeds with a small process pool only.

Author: generated with Claude (Anthropic) for H. Kim, Oct 2026.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import cKDTree

EIGHT_PI_13 = (8.0 * math.pi) ** (1.0 / 3.0)


def log(msg: str):
    print(time.strftime("%H:%M:%S"), f"[pid {os.getpid()}]", msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    v1: float = 0.6            # outer (O->A) wall velocity
    v2: float = 1.0            # inner / composite (->B) wall velocity
    beta2: float = 2.2         # b2 / b1
    deta_star: float = 0.5     # eta2* - eta1*  (in units of 1/b1)
    r0: float = 0.0            # direct O->B channel, rate = r0 * G2 in O regions
    L_over_R1: float = 8.0     # box size in units of R1
    ndir: int = 96             # wall directions per sphere
    nsample: int = 16000       # sample points for volume fractions
    dbin: float = 0.02         # time-bin width (1/b1) for nucleation thinning
    seed: int = 0
    single_step: bool = False  # reference: only channel B, nucleating everywhere
    stop_frac: float = 1e-4    # stop the run when P_O + P_A below this
    chan_frac: float = 1e-3    # stop generating candidates of a channel once its host phase
                               # fraction is below this (acceptance ∝ host fraction; the
                               # exponentially growing candidate stream would otherwise
                               # be spent on rejections)

    @property
    def R1(self) -> float:
        return EIGHT_PI_13 * self.v1

    @property
    def R2(self) -> float:
        return EIGHT_PI_13 * self.v2 / self.beta2

    @property
    def L(self) -> float:
        return self.L_over_R1 * self.R1

    @property
    def G1star(self) -> float:
        return 1.0 / (8.0 * math.pi * self.v1 ** 3)

    @property
    def G2star(self) -> float:
        return self.beta2 ** 4 / (8.0 * math.pi * self.v2 ** 3)

    def chi_nest(self) -> float:
        """Eq. (chi_nest) of the paper with (beta1/H) dT/T -> b1 * deta_star."""
        v1, v2 = self.v1, self.v2
        if v2 <= v1:
            return float("inf")
        return (4 * v2 - 3 * v1) / (4 * EIGHT_PI_13 * (v2 - v1)) * self.deta_star

    def tag(self) -> str:
        s = "single" if self.single_step else f"deta_{self.deta_star}"
        return f"v1_{self.v1}_v2_{self.v2}_b2_{self.beta2}_{s}_r0_{self.r0}_L_{self.L_over_R1}_s{self.seed}"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def min_image(d: np.ndarray, L: float) -> np.ndarray:
    return d - L * np.round(d / L)


def sample_times(rng, b, eta_star, ea, eb, n):
    """Inverse-CDF sample of n times from rate ∝ exp[b (eta - eta_star)] on [ea, eb]."""
    u = rng.random(n)
    lo = math.exp(b * (ea - eta_star))
    hi = math.exp(b * (eb - eta_star))
    return eta_star + np.log(lo + u * (hi - lo)) / b


# ----------------------------------------------------------------------------
# nucleation with thinning
# ----------------------------------------------------------------------------
def nucleate(p: Params, rng) -> dict:
    """Generate accepted bubbles.  kind: 0 = A, 1 = B (A->B), 2 = D (direct O->B)."""
    L, V = p.L, p.L ** 3
    b1, b2 = 1.0, p.beta2
    eta1s, eta2s = 0.0, p.deta_star

    cA, tA = [], []
    cB, tB, kB, parB = [], [], [], []

    eta = (eta2s - 8.0 / b2) if p.single_step else (eta1s - 8.0)
    eta_max = max(eta1s, eta2s) + 12.0

    sample = rng.random((p.nsample, 3)) * L
    tA_arr = np.full(p.nsample, np.inf)
    tB_arr = np.full(p.nsample, np.inf)
    treeS = cKDTree(sample, boxsize=L)

    def update_arrivals(centers, times, v, arr):
        for c, t in zip(centers, times):
            idx = treeS.query_ball_point(c, min(0.5 * L, 6.0 * p.R1))
            if not idx:
                continue
            idx = np.asarray(idx)
            d = np.linalg.norm(min_image(sample[idx] - c, L), axis=1)
            arr[idx] = np.minimum(arr[idx], t + d / v)

    nbin = 0
    t0 = time.time()
    while eta < eta_max:
        ea, eb = eta, eta + p.dbin
        PO = np.mean((tA_arr > ea) & (tB_arr > ea))
        PB = np.mean(tB_arr <= ea)
        PA = 1.0 - PO - PB
        if (not p.single_step) and (PO + PA < p.stop_frac) and len(cB) > 0:
            break
        if p.single_step and PO < p.stop_frac and len(cB) > 0:
            break
        nbin += 1
        if nbin % 100 == 0:
            log(f"  nucleate eta={ea:+.2f} P_O={PO:.3f} P_A={PA:.3f} P_B={PB:.3f} NA={len(cA)} NB={len(cB)} ({time.time()-t0:.0f}s)")

        cand = []
        if not p.single_step and PO > p.chan_frac:
            nA = rng.poisson(V * p.G1star / b1 * (math.exp(b1 * (eb - eta1s)) - math.exp(b1 * (ea - eta1s))))
            if nA:
                cand.append((sample_times(rng, b1, eta1s, ea, eb, nA), rng.random((nA, 3)) * L, np.zeros(nA, int)))
            if p.r0 > 0:
                nD = rng.poisson(p.r0 * V * p.G2star / b2 * (math.exp(b2 * (eb - eta2s)) - math.exp(b2 * (ea - eta2s))))
                if nD:
                    cand.append((sample_times(rng, b2, eta2s, ea, eb, nD), rng.random((nD, 3)) * L, np.full(nD, 2)))
        if (p.single_step and PO > p.chan_frac) or ((not p.single_step) and PA > p.chan_frac):
            nB = rng.poisson(V * p.G2star / b2 * (math.exp(b2 * (eb - eta2s)) - math.exp(b2 * (ea - eta2s))))
            if nB:
                cand.append((sample_times(rng, b2, eta2s, ea, eb, nB), rng.random((nB, 3)) * L, np.ones(nB, int)))
        if cand:
            ct = np.concatenate([c[0] for c in cand])
            cx = np.concatenate([c[1] for c in cand])
            ck = np.concatenate([c[2] for c in cand])
            order = np.argsort(ct)
            ct, cx, ck = ct[order], cx[order], ck[order]

            arrA = np.asarray(cA) if cA else np.zeros((0, 3))
            tAa = np.asarray(tA) if tA else np.zeros(0)
            arrB = np.asarray(cB) if cB else np.zeros((0, 3))
            tBa = np.asarray(tB) if tB else np.zeros(0)
            treeA = cKDTree(arrA, boxsize=L) if len(arrA) else None
            treeB = cKDTree(arrB, boxsize=L) if len(arrB) else None
            rmaxA = min(p.v1 * (eb - tAa.min()), 0.5 * L) if len(tAa) else 0.0
            rmaxB = min(p.v2 * (eb - tBa.min()), 0.5 * L) if len(tBa) else 0.0
            newA_c, newA_t = [], []
            newB_c, newB_t, newB_k, newB_p = [], [], [], []

            for t, x, k in zip(ct, cx, ck):
                inB = False
                if treeB is not None:
                    idx = treeB.query_ball_point(x, rmaxB)
                    if idx:
                        idx = np.asarray(idx)
                        d = np.linalg.norm(min_image(arrB[idx] - x, L), axis=1)
                        inB = bool(np.any(d < p.v2 * (t - tBa[idx])))
                if not inB and newB_c:
                    d = np.linalg.norm(min_image(np.asarray(newB_c) - x, L), axis=1)
                    inB = bool(np.any(d < p.v2 * (t - np.asarray(newB_t))))
                if inB:
                    continue
                inA, parent = False, -1
                if not p.single_step:
                    best = np.inf
                    if treeA is not None:
                        idx = treeA.query_ball_point(x, rmaxA)
                        if idx:
                            idx = np.asarray(idx)
                            d = np.linalg.norm(min_image(arrA[idx] - x, L), axis=1)
                            m = d < p.v1 * (t - tAa[idx])
                            if np.any(m):
                                inA = True
                                arr_t = tAa[idx][m] + d[m] / p.v1
                                j = np.argmin(arr_t)
                                best, parent = arr_t[j], int(idx[m][j])
                    if newA_c:
                        d = np.linalg.norm(min_image(np.asarray(newA_c) - x, L), axis=1)
                        m = d < p.v1 * (t - np.asarray(newA_t))
                        if np.any(m):
                            inA = True
                            arr_t = np.asarray(newA_t)[m] + d[m] / p.v1
                            j = np.argmin(arr_t)
                            if arr_t[j] < best:
                                best, parent = arr_t[j], len(cA) + int(np.flatnonzero(m)[j])
                if k == 0:
                    if not inA:
                        newA_c.append(x); newA_t.append(t)
                elif k == 1:
                    if p.single_step or inA:
                        newB_c.append(x); newB_t.append(t); newB_k.append(1); newB_p.append(parent)
                else:
                    if not inA:
                        newB_c.append(x); newB_t.append(t); newB_k.append(2); newB_p.append(-1)

            if newA_c:
                update_arrivals(newA_c, newA_t, p.v1, tA_arr)
                cA.extend(newA_c); tA.extend(newA_t)
            if newB_c:
                update_arrivals(newB_c, newB_t, p.v2, tB_arr)
                cB.extend(newB_c); tB.extend(newB_t); kB.extend(newB_k); parB.extend(newB_p)
        eta = eb

    return dict(
        cA=np.asarray(cA).reshape(-1, 3), tA=np.asarray(tA, float),
        cB=np.asarray(cB).reshape(-1, 3), tB=np.asarray(tB, float),
        kB=np.asarray(kB, int), parB=np.asarray(parB, int),
        sample=sample, tA_arr=tA_arr, tB_arr=tB_arr, eta_end=eta,
    )


# ----------------------------------------------------------------------------
# wall-element death times (broadcast over (ndir, M); no array replication)
# ----------------------------------------------------------------------------
def _death_equal_speed(d2, dn, Delta, v):
    """tau at which a wall element (speed v) enters a sphere of the same speed.

    d2 : (M,) |c_self - c_k|^2 ; dn : (nd,M) (c_self - c_k).n ; Delta : (M,) t_self - t_k.
    f(tau) = d2 - v^2 Delta^2 + 2 tau v (dn - v Delta) ; root where slope < 0.
    """
    f0 = (d2 - v * v * Delta * Delta)[None, :]
    slope = 2.0 * v * (dn - v * Delta[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(slope < 0, -f0 / slope, np.inf)
    tau[tau < 0] = np.inf
    return tau


def _death_faster_sphere(d2, dn, tj, v1, v2):
    """tau at which an A-wall element (speed v1) enters a B sphere (speed v2 > v1).

    g(tau) = (v1^2 - v2^2) tau^2 + 2 (v1 dn + v2^2 tj) tau + d2 - v2^2 tj^2, tj = t_B - t_A.
    Downward parabola, positive at tau0 = max(0, tj): death at the larger root.
    """
    a = v1 * v1 - v2 * v2
    b = 2.0 * (v1 * dn + (v2 * v2 * tj)[None, :])
    c = (d2 - v2 * v2 * tj * tj)[None, :]
    disc = b * b - 4 * a * c
    with np.errstate(invalid='ignore'):
        tau = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    tau0 = np.maximum(0.0, tj)[None, :]
    return np.where(tau < tau0, np.inf, tau)


def wall_deaths(p: Params, nuc: dict, dirs: np.ndarray) -> dict:
    """Death time, killer type and (for B) composite flag for every wall element."""
    L = p.L
    cA, tA, cB, tB, parB = nuc['cA'], nuc['tA'], nuc['cB'], nuc['tB'], nuc['parB']
    NA, NB, nd = len(cA), len(cB), len(dirs)
    treeA = cKDTree(cA, boxsize=L) if NA else None
    treeB = cKDTree(cB, boxsize=L) if NB else None

    A_tau = np.full((NA, nd), np.inf)
    A_kill = np.full((NA, nd), -1, int)
    B_tau = np.full((NB, nd), np.inf)
    B_comp = np.zeros((NB, nd), bool)

    def neighbours(tree, c, rcut):
        return np.asarray(tree.query_ball_point(c, min(rcut, 0.5 * L)), int)

    t0 = time.time()
    for i in range(NA):
        rc = 4.0 * p.R1
        for _ in range(4):
            tau_best = np.full(nd, np.inf)
            kill = np.full(nd, -1, int)
            idx = neighbours(treeA, cA[i], rc)
            idx = idx[idx != i]
            if len(idx):
                d = min_image(cA[i] - cA[idx], L)
                tau = _death_equal_speed(np.einsum('ij,ij->i', d, d), dirs @ d.T, tA[i] - tA[idx], p.v1)
                tmin = tau.min(axis=1)
                m = tmin < tau_best
                tau_best[m] = tmin[m]; kill[m] = 0
            if treeB is not None:
                idxb = neighbours(treeB, cA[i], rc * max(1.0, p.v2 / p.v1))
                if len(idxb):
                    d = min_image(cA[i] - cB[idxb], L)
                    tau = _death_faster_sphere(np.einsum('ij,ij->i', d, d), dirs @ d.T, tB[idxb] - tA[i], p.v1, p.v2)
                    j = np.argmin(tau, axis=1)
                    tmin = tau[np.arange(nd), j]
                    m = tmin < tau_best
                    tau_best[m] = tmin[m]
                    own = parB[idxb[j]] == i
                    kill[m & own] = 1
                    kill[m & ~own] = 2
            if np.all(np.isfinite(tau_best)):
                break
            rc *= 2.0
        A_tau[i], A_kill[i] = tau_best, kill
        if (i + 1) % 500 == 0:
            log(f"  A walls {i+1}/{NA} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    for j in range(NB):
        rc = 4.0 * p.R2
        tau_best = np.full(nd, np.inf)
        for _ in range(5):
            idx = neighbours(treeB, cB[j], rc)
            idx = idx[idx != j]
            if len(idx):
                d = min_image(cB[j] - cB[idx], L)
                tau = _death_equal_speed(np.einsum('ij,ij->i', d, d), dirs @ d.T, tB[j] - tB[idx], p.v2)
                tau_best = tau.min(axis=1)
            if np.all(np.isfinite(tau_best)):
                break
            rc *= 2.0
        B_tau[j] = tau_best
        if treeA is not None and NA:
            fin = np.isfinite(tau_best)
            comp = np.zeros(nd, bool)
            if fin.any():
                pos = (cB[j] + p.v2 * tau_best[fin, None] * dirs[fin]) % L
                tdeath = tB[j] + tau_best[fin]
                rmaxA = min(p.v1 * (tdeath.max() - tA.min()), 0.5 * L)
                hits = treeA.query_ball_point(pos, rmaxA)
                cf = np.ones(fin.sum(), bool)
                for a, ia in enumerate(hits):
                    if ia:
                        ia = np.asarray(ia)
                        dd = np.linalg.norm(min_image(cA[ia] - pos[a], L), axis=1)
                        if np.any(dd < p.v1 * (tdeath[a] - tA[ia])):
                            cf[a] = False
                comp[fin] = cf
            B_comp[j] = comp
        else:
            B_comp[j] = True
        if (j + 1) % 1000 == 0:
            log(f"  B walls {j+1}/{NB} ({time.time()-t0:.0f}s)")
    return dict(A_tau=A_tau, A_kill=A_kill, B_tau=B_tau, B_comp=B_comp)


# ----------------------------------------------------------------------------
# analytic reference curves
# ----------------------------------------------------------------------------
def analytic_fractions(p: Params, eta: np.ndarray) -> dict:
    """P_O, P_A, P_B from Eqs. (P_O), (I_2_cond), (P_B_P_A); direct channel via r0."""
    b1, b2 = 1.0, p.beta2
    I1 = np.exp(b1 * eta)
    I0 = p.r0 * np.exp(b2 * (eta - p.deta_star))
    PO = np.exp(-I1 - I0)
    e2 = np.linspace(eta.min() - 10.0 / b2, eta.max(), 4000)
    G2 = p.G2star * np.exp(b2 * (e2 - p.deta_star))
    PO2 = np.exp(-np.exp(b1 * e2) - p.r0 * np.exp(b2 * (e2 - p.deta_star)))
    I2 = np.zeros_like(eta)
    for n, e in enumerate(eta):
        m = e2 <= e
        I2[n] = np.trapz(G2[m] * (1 - PO2[m]) * 4 * math.pi / 3 * p.v2 ** 3 * (e - e2[m]) ** 3, e2[m])
    PB = 1.0 - np.exp(-I2 - I0)
    # "one daughter per parent" (saturated) limit: every A bubble converts to B
    # right after nucleation, so B bubbles are seeded at the A rate and expand
    # at v2:  I_sat = (v2/v1)^3 I1.
    PB_sat = 1.0 - np.exp(-(p.v2 / p.v1) ** 3 * I1 - I0)
    return dict(PO=PO, PA=1.0 - PO - PB, PB=PB, PB_sat=PB_sat, I1=I1, I2=I2)


def f_comp_cubic(chi: float) -> float:
    return math.exp(-chi ** 3)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run(p: Params, verbose: bool = True) -> dict:
    t0 = time.time()
    log(f"start {p.tag()}")
    rng = np.random.default_rng(p.seed)
    dirs = fibonacci_sphere(p.ndir)
    nuc = nucleate(p, rng)
    t1 = time.time()
    log(f"  nucleation done NA={len(nuc['cA'])} NB={len(nuc['cB'])} ({t1-t0:.0f}s); wall deaths ...")
    dth = wall_deaths(p, nuc, dirs)
    t2 = time.time()

    NA, NB = len(nuc['cA']), len(nuc['cB'])
    if NA:
        kill = dth['A_kill'].ravel()
        f_comp = float(np.mean(kill == 1))
        f_ext = float(np.mean(kill == 2))
        f_coll = float(np.mean(kill == 0))
        rA = (p.v1 * dth['A_tau']).ravel()
    else:
        f_comp = f_ext = f_coll = float('nan'); rA = np.zeros(0)
    rB = (p.v2 * dth['B_tau']).ravel()
    comp = dth['B_comp'].ravel()
    fin = np.isfinite(rB)
    tmin = min(nuc['tA'].min() if NA else np.inf, nuc['tB'].min())
    eta_grid = np.linspace(tmin - 0.2, nuc['eta_end'] + 0.5, 200)
    tAa, tBa = nuc['tA_arr'], nuc['tB_arr']
    PO = np.array([np.mean((tAa > e) & (tBa > e)) for e in eta_grid])
    PB = np.array([np.mean(tBa <= e) for e in eta_grid])
    PA = 1 - PO - PB
    F_OB = float(np.mean(tBa < tAa))
    if NA and NB:
        counts = np.bincount(nuc['parB'][nuc['parB'] >= 0], minlength=NA)
        daughters_mean = float(counts.mean())
    else:
        daughters_mean = float('nan')

    out = dict(
        params=asdict(p), chi_nest=p.chi_nest(), tag=p.tag(),
        NA=NA, NB=NB, NB_direct=int(np.sum(nuc['kB'] == 2)),
        f_comp=f_comp, f_ext=f_ext, f_coll=f_coll, f_overtaken=f_comp + f_ext if NA else float('nan'),
        F_OB=F_OB, daughters_per_parent=daughters_mean,
        rA_death_mean=float(np.mean(rA[np.isfinite(rA)])) if len(rA) else float('nan'),
        rB_comp_death_mean=float(np.mean(rB[fin & comp])) if np.any(fin & comp) else float('nan'),
        rB_inner_death_mean=float(np.mean(rB[fin & ~comp])) if np.any(fin & ~comp) else float('nan'),
        frac_B_wall_composite=float(np.mean(comp[fin])) if np.any(fin) else float('nan'),
        eta_grid=eta_grid, PO=PO, PA=PA, PB=PB,
        rA_death=rA, rB_death=rB, B_comp=comp,
        nuc=nuc, deaths=dth, dirs=dirs,
        t_nucleate=t1 - t0, t_deaths=t2 - t1,
    )
    if verbose:
        log(f"done {p.tag()} | NA={NA} NB={NB} chi={p.chi_nest():.3f} "
            f"f_own={f_comp:.3f} f_ext={f_ext:.3f} F_OB={F_OB:.3f} "
            f"<rB_comp>/R1={out['rB_comp_death_mean']/p.R1:.2f} <rB_in>/R2={out['rB_inner_death_mean']/p.R2:.2f} "
            f"| {t1-t0:.0f}s + {t2-t1:.0f}s")
    return out


def summary(out: dict) -> dict:
    keys = ['params', 'tag', 'chi_nest', 'NA', 'NB', 'NB_direct', 'f_comp', 'f_ext', 'f_coll', 'f_overtaken',
            'F_OB', 'daughters_per_parent', 'rA_death_mean', 'rB_comp_death_mean', 'rB_inner_death_mean',
            'frac_B_wall_composite', 't_nucleate', 't_deaths']
    return {k: out[k] for k in keys}


def save_fractions(out: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, eta_grid=out['eta_grid'], PO=out['PO'], PA=out['PA'], PB=out['PB'],
                        rA_death=out['rA_death'], rB_death=out['rB_death'], B_comp=out['B_comp'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v1', type=float, default=0.6)
    ap.add_argument('--v2', type=float, default=1.0)
    ap.add_argument('--beta2', type=float, default=2.2)
    ap.add_argument('--deta', type=float, default=0.5)
    ap.add_argument('--r0', type=float, default=0.0)
    ap.add_argument('--L', type=float, default=8.0)
    ap.add_argument('--ndir', type=int, default=96)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--single', action='store_true')
    a = ap.parse_args()
    p = Params(v1=a.v1, v2=a.v2, beta2=a.beta2, deta_star=a.deta, r0=a.r0,
               L_over_R1=a.L, ndir=a.ndir, seed=a.seed, single_step=a.single)
    o = run(p)
    print(json.dumps(summary(o), indent=1))
