#!/usr/bin/env python3
"""Envelope-approximation gravitational-wave spectrum from the thin-wall Monte-Carlo.

Follows Huber & Konstandin (2008) / Jinno & Takimoto (2017): the source is the
transverse-traceless part of the wall stress tensor of all *uncollided* wall
elements.  Only ->B walls carry energy (eps -> 0 limit): a wall element of a
sphere of radius r carries surface energy sigma = Delta V * r / 3, with
Delta V = V0 both for inner A->B elements and for composite O->B elements.

In the Minkowski limit (transition duration << 1/H) the GW energy per solid
angle and log-wavenumber is (Weinberg)

    dE / dOmega_k dln k  ∝  k^3  Lambda_{ij,lm} T*_ij(k, w=k) T_lm(k, w=k),
    T_ij(k, w) = int d eta e^{i w eta} sum_elements  dA sigma n_i n_j e^{-i k.x}.

Implementation (grid/FFT): at each time step the alive elements deposit
dA sigma n_i n_j onto a periodic grid (nearest grid point), the six
components are FFT'd, and the time integral with e^{i|k| eta} is accumulated
per Fourier cell.  At the end the TT contraction is formed for every k-vector
and averaged over |k| shells, which gives a low-variance estimate of S(k).
The NGP window is deconvolved.

Output: dimensionless shape S(k) = k^3 <Lambda T T*>_shell / (V Delta V^2), in
units with b1 = 1; ratios between runs in the same box are what matter.
"""
from __future__ import annotations

import math
import time

import numpy as np

from nested_mc.bubbles import log


def envelope_spectrum(out: dict, ngrid: int = 160, n_time: int = 200,
                      kR1_max: float = 12.0, nbins: int = 18, verbose: bool = True) -> dict:
    p = out['params']
    R1 = (8 * math.pi) ** (1 / 3) * p['v1']
    L = p['L_over_R1'] * R1
    v2 = p['v2']
    cB, tB = out['nuc']['cB'], out['nuc']['tB']
    B_tau = out['deaths']['B_tau']
    dirs = out['dirs']
    NB, nd = B_tau.shape
    dOmega = 4 * math.pi / nd
    if NB == 0:
        raise ValueError("no B bubbles")

    death_all = (tB[:, None] + B_tau)
    fin = np.isfinite(death_all)
    # end the time grid when 99.9 % of the wall energy (∝ r^3 at death) has collided;
    # a few stray elements with no neighbour would otherwise stretch the grid
    w_death = (v2 * B_tau[fin]) ** 3
    order = np.argsort(death_all[fin])
    cum = np.cumsum(w_death[order]) / w_death.sum()
    t_end = death_all[fin][order][np.searchsorted(cum, 0.999)]
    t_beg = tB.min()
    kmax_abs = kR1_max / ((8 * math.pi) ** (1 / 3) * p['v1'])
    n_time = max(n_time, int(np.ceil((t_end - t_beg) * kmax_abs / 0.3)) + 1)
    etas = np.linspace(t_beg, t_end, n_time)
    deta = etas[1] - etas[0]
    if verbose:
        log(f"  [gw] time grid: eta in [{t_beg:.2f}, {t_end:.2f}], {n_time} steps, k_max*deta = {kmax_abs*deta:.2f}")
    N = ngrid
    dx = L / N

    # k grid (rfft along last axis)
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).astype(np.float64)
    kmax = kR1_max / R1
    sel = (kmag > 0) & (kmag <= kmax)
    # keep only needed cells to save memory in the accumulation
    idx = np.nonzero(sel)
    km = kmag[idx]
    khat = np.stack([KX[idx], KY[idx], KZ[idx]], axis=1) / km[:, None]
    # NGP window deconvolution
    W = (np.sinc(KX[idx] * dx / (2 * np.pi)) * np.sinc(KY[idx] * dx / (2 * np.pi)) * np.sinc(KZ[idx] * dx / (2 * np.pi)))
    phase_w = np.exp(1j * km * 0.0)  # placeholder shape
    comps = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    acc = np.zeros((6, len(km)), np.complex128)

    birth = tB[:, None]
    death = tB[:, None] + B_tau
    t0 = time.time()
    for it, eta in enumerate(etas):
        alive = (birth <= eta) & (eta < death)
        if not alive.any():
            continue
        jj, aa = np.nonzero(alive)
        r = v2 * (eta - tB[jj])
        x = (cB[jj] + r[:, None] * dirs[aa]) % L
        n = dirs[aa]
        w = (r ** 3 / 3.0) * dOmega
        cell = np.floor(x / dx).astype(np.int64) % N
        flat = (cell[:, 0] * N + cell[:, 1]) * N + cell[:, 2]
        ph = np.exp(1j * km * eta) * deta
        for c, (a, b) in enumerate(comps):
            grid = np.bincount(flat, weights=w * n[:, a] * n[:, b], minlength=N ** 3).reshape(N, N, N)
            F = np.fft.rfftn(grid)
            acc[c] += F[idx] * ph
        if verbose and it % 50 == 0:
            log(f"  [gw] eta={eta:.2f} alive={len(jj)} ({time.time()-t0:.0f}s)")

    # TT contraction: Lambda T T* = T_ij T*_ij - 2 |T_ij khat_j|^2 + |khat T khat|^2
    T = np.zeros((3, 3, len(km)), np.complex128)
    for c, (a, b) in enumerate(comps):
        T[a, b] = acc[c] / W
        T[b, a] = acc[c] / W
    TT = np.einsum('ijk,ijk->k', T, np.conj(T)).real
    Tk = np.einsum('ijk,kj->ik', T, khat)               # (3, nk)
    TkTk = np.einsum('ik,ik->k', Tk, np.conj(Tk)).real
    kTk = np.einsum('ki,ik->k', khat, Tk)
    lam = TT - 2 * TkTk + np.abs(kTk) ** 2
    S_cell = km ** 3 * lam / L ** 3

    # shell average in log bins of kR1
    kR = km * R1
    edges = np.geomspace(kR.min() * 0.999, kR1_max, nbins + 1)
    which = np.digitize(kR, edges) - 1
    S = np.full(nbins, np.nan); kc = np.full(nbins, np.nan); nmode = np.zeros(nbins, int)
    for b in range(nbins):
        m = which == b
        if m.sum() >= 3:
            S[b] = S_cell[m].mean(); kc[b] = np.exp(np.log(kR[m]).mean()); nmode[b] = m.sum()
    # bins with too few modes are NaN; bin centres are the nominal log-bin centres
    kc = np.where(np.isfinite(kc), kc, np.sqrt(edges[:-1] * edges[1:]))
    return dict(kR1=kc, S=S, nmode=nmode, etas=etas, L=L, R1=R1)
