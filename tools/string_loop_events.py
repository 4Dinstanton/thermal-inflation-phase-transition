#!/usr/bin/env python3
"""Detect loop-collapse (cusp/kink annihilation) events and radiation in 3D.

Reads CosmoLattice ``phi_*`` (+ optional ``pi_*``) HDF5 snapshots, labels string
loops from the plaquette winding, tracks them between consecutive snapshots, and
records:

**Loop tracking / collapse events**
  - per-loop voxel count, winding length, periodic centroid, radius of gyration
  - matching between consecutive snapshots (periodic, within ``c·Δt`` + drift)
  - *collapse events*: a loop present at step N and gone at step N+1, with its
    last-known size, shrink rate ``dR/dt`` (≈1 means relativistic collapse) and
    velocity at last sighting

**Kink / cusp proxies** (per loop, from the winding vector)
  The plaquette winding vector **W** points along the string, so smoothing it
  over a small ball gives a genuine unit tangent û without any curve tracing.
  - ``coherence`` = |Σ û| / count over the ball: 1 for a straight segment (any
    orientation), lower at a kink, →0 where the string folds back on itself
  - ``kink_frac``: fraction of core voxels whose turning angle over ``--kink-sep``
    exceeds ``--kink-deg`` (NaN for loops smaller than ``--kink-sep``, where a
    loop's own curvature is indistinguishable from a kink)
  - ``cusp_frac``: fraction with low coherence *and* local ``v² > --cusp-v2``,
    i.e. the string doubling back while moving relativistically

**Radiation budget** (the "where did the energy go" part)
  Gradient energy is split into the two physical channels for a global U(1) string,
      ½|∇Φ|² = ½(∇ρ)²  +  ½ρ²(∇θ)²
                radial      Goldstone
  and each is summed over three regions: ``core`` (|W|>0.5), ``shell``
  (dilated core, the near field) and ``bulk`` (everything else = radiation).
  Loop annihilation should show core energy dropping while **bulk Goldstone**
  energy rises.

  Each collapse event additionally gets a *local* probe: the same sphere of
  radius ``--probe-radius`` around the loop's last position, measured on the
  snapshot before and the snapshot after it vanished. A loop that annihilated
  into Goldstone radiation leaves ``probe_gold_after > probe_gold_before``;
  one that merely drifted out of the tracker does not.

Caveats
-------
A lattice field simulation does not have literal Nambu–Goto cusps. What is
measured here are *proxies*: relativistic, low-coherence core regions plus the
loop-disappearance statistics. Treat ``cusp_frac`` as a candidate counter, not a
cusp detection. Loops can also leave the census by reconnecting with a neighbour
rather than annihilating — the local energy probe is what separates the two.

Usage
-----
    python tools/string_loop_events.py <run_dir> --step-min 3800

    # cheaper: skip the per-voxel tangent/kink stats and the energy pass
    python tools/string_loop_events.py <run_dir> --no-kinks --skip-budget

Outputs (under ``<run_dir>/strings/events/``)
    loop_tracks.csv       one row per (step, loop)
    loop_events.csv       one row per collapse event, incl. the energy probe
    radiation_budget.csv  one row per step (energy channels)
    string_loop_events.png
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.cl_field_snapshot_io import (  # noqa: E402
    build_h5_time_index,
    h5_has_group,
    load_manifest_rows,
    parse_manifest_row,
    read_h5_snapshot,
    resolve_h5_path,
    resolve_snapshot_h5,
)
from tools.export_cl_snapshots import load_run_params  # noqa: E402
from tools.string_network_metrics import (  # noqa: E402
    resolve_lattice_scales,
    tree_potential_density,
)

LOG = logging.getLogger("string_loop_events")

WIND_THRESHOLD = 0.5

TRACK_FIELDS = (
    "step", "time", "temperature", "a", "loop_id",
    "n_voxels", "L_comoving", "R_gyr_vox", "extent_vox",
    "cx", "cy", "cz",
    "v2_mean", "v2_p95", "v2_max",
    "coherence_mean", "kink_frac", "cusp_frac",
    "matched_prev_id", "dR_dt",
    "probe_kin", "probe_gold", "probe_rad", "probe_pot",
)

EVENT_FIELDS = (
    "event", "step", "time", "temperature",
    "loop_id", "n_voxels", "L_comoving", "R_gyr_vox",
    "v2_mean", "v2_p95", "kink_frac", "cusp_frac",
    "shrink_rate_dR_dt", "lifetime_steps", "cx", "cy", "cz",
    "probe_radius_vox",
    "probe_kin_before", "probe_gold_before", "probe_rad_before", "probe_pot_before",
    "probe_kin_after", "probe_gold_after", "probe_rad_after", "probe_pot_after",
    "probe_gold_gain", "probe_total_change",
)

BUDGET_FIELDS = (
    "step", "time", "temperature", "a",
    "n_string_voxels", "n_loops",
    "E_kin_core", "E_grad_rad_core", "E_grad_gold_core",
    "E_kin_shell", "E_grad_rad_shell", "E_grad_gold_shell",
    "E_kin_bulk", "E_grad_rad_bulk", "E_grad_gold_bulk",
    "E_pot_core", "E_pot_bulk", "E_total",
    "gold_over_radial_bulk", "has_pi",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _wrap_pi(d: np.ndarray) -> np.ndarray:
    """Wrap phase differences into (-π, π]."""
    return np.arctan2(np.sin(d), np.cos(d))


def _plaquette(th00: np.ndarray, th10: np.ndarray,
               th11: np.ndarray, th01: np.ndarray) -> np.ndarray:
    """Winding of one oriented plaquette, in units of 2π."""
    tot = (_wrap_pi(th10 - th00) + _wrap_pi(th11 - th10)
           + _wrap_pi(th01 - th11) + _wrap_pi(th00 - th01))
    return tot * (1.0 / (2.0 * math.pi))


# Winding-vector components: W_x from the (y,z) plaquette, W_y from (z,x),
# W_z from (x,y). ``tools.winding.compute_winding_number`` returns the signed
# *sum* of the three orientations, which cancels for segments running diagonally
# and so fragments a single loop into pieces. Keeping the components separate
# makes |W| orientation independent, and W itself is the local string tangent.
_PLAQUETTE_AXES = ((1, 2), (2, 0), (0, 1))
_UNIT = np.eye(3, dtype=int)


def _slab_corner(ts: np.ndarray, L: int, d: np.ndarray) -> np.ndarray:
    """Haloed slab shifted by unit offsets ``d`` (each 0 or 1), trimmed to L planes."""
    a = ts[:, :, d[2]: d[2] + L]
    if d[0]:
        a = np.roll(a, -1, axis=0)
    if d[1]:
        a = np.roll(a, -1, axis=1)
    return a


def winding_magnitude(theta: np.ndarray, *, slab: int = 16) -> np.ndarray:
    """|W| = sqrt(W_x²+W_y²+W_z²), computed slab-wise to bound peak RAM."""
    theta = np.asarray(theta)
    N = theta.shape[0]
    mag2 = np.zeros(theta.shape, dtype=np.float32)
    for z0 in range(0, N, slab):
        z1 = min(z0 + slab, N)
        L = z1 - z0
        zi = list(range(z0, z1)) + [z1 % N]  # +1 halo plane, periodic
        ts = np.asarray(theta[:, :, zi], dtype=np.float32)

        base = _slab_corner(ts, L, np.zeros(3, dtype=int))
        acc = np.zeros_like(base)
        for ax_a, ax_b in _PLAQUETTE_AXES:
            ea, eb = _UNIT[ax_a], _UNIT[ax_b]
            w = _plaquette(
                base,
                _slab_corner(ts, L, ea),
                _slab_corner(ts, L, ea + eb),
                _slab_corner(ts, L, eb),
            )
            acc += w * w
            del w
        mag2[:, :, z0:z1] = acc
        del ts, base, acc
    return np.sqrt(mag2, out=mag2)


def winding_vector_at_coords(theta: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """(n,3) winding vector at the given voxels — the local string tangent."""
    N = theta.shape[0]
    n = len(coords)
    out = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return out
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]

    def th(di: int, dj: int, dk: int) -> np.ndarray:
        return np.asarray(
            theta[(i + di) % N, (j + dj) % N, (k + dk) % N], dtype=np.float64
        )

    base = th(0, 0, 0)
    for comp, (ax_a, ax_b) in enumerate(_PLAQUETTE_AXES):
        ea, eb = _UNIT[ax_a], _UNIT[ax_b]
        out[:, comp] = _plaquette(
            base, th(*ea), th(*(ea + eb)), th(*eb)
        )
    return out


def periodic_centroid(coords: np.ndarray, N: int) -> np.ndarray:
    """Circular mean of integer voxel coords on a periodic cube."""
    ang = coords.astype(np.float64) * (2.0 * math.pi / N)
    c = np.arctan2(np.sin(ang).mean(axis=0), np.cos(ang).mean(axis=0))
    return np.mod(c, 2.0 * math.pi) * (N / (2.0 * math.pi))


def periodic_delta(a: np.ndarray, b: np.ndarray, N: int) -> np.ndarray:
    d = a - b
    return d - N * np.round(d / N)


def periodic_radius_of_gyration(coords: np.ndarray, N: int) -> Tuple[float, float]:
    """(R_gyr, max extent) in voxels, minimum-image relative to the centroid."""
    if len(coords) == 0:
        return float("nan"), float("nan")
    c = periodic_centroid(coords, N)
    d = periodic_delta(coords.astype(np.float64), c[None, :], N)
    r2 = np.sum(d * d, axis=1)
    return float(np.sqrt(r2.mean())), float(np.sqrt(r2.max()))


def phidot_scale(f_star: float, omega_star: float, a: float) -> float:
    """φ̇_phys = π_prog × fStar × ωStar / a³ (CosmoLattice convention)."""
    return float(f_star) * float(omega_star) / (max(float(a), 1e-30) ** 3)


# ---------------------------------------------------------------------------
# per-voxel quantities on the string cores (cheap: only ~N_core lookups)
# ---------------------------------------------------------------------------
def local_v2_on_coords(
    snap: Dict[str, Any],
    coords: np.ndarray,
    scales: Dict[str, float],
) -> Optional[np.ndarray]:
    """Per-core-voxel v² ≈ e_kin / (e_kin + e_grad), using neighbour lookups."""
    if snap.get("pi1") is None or snap.get("pi2") is None:
        return None
    phi1 = snap["phi1"]
    phi2 = snap["phi2"]
    N = phi1.shape[0]
    a = float(snap.get("a", 1.0))
    dx = float(scales["dx_com"])
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]

    scale = phidot_scale(scales["f_star"], scales["omega_star"], a)
    d1 = np.asarray(snap["pi1"][i, j, k], dtype=np.float64) * scale
    d2 = np.asarray(snap["pi2"][i, j, k], dtype=np.float64) * scale
    e_kin = 0.5 * (d1 * d1 + d2 * d2)

    p1c = np.asarray(phi1[i, j, k], dtype=np.float64)
    p2c = np.asarray(phi2[i, j, k], dtype=np.float64)
    g2 = np.zeros_like(e_kin)
    for axis in range(3):
        ii, jj, kk = i.copy(), j.copy(), k.copy()
        if axis == 0:
            ii = (i + 1) % N
        elif axis == 1:
            jj = (j + 1) % N
        else:
            kk = (k + 1) % N
        d_p1 = np.asarray(phi1[ii, jj, kk], dtype=np.float64) - p1c
        d_p2 = np.asarray(phi2[ii, jj, kk], dtype=np.float64) - p2c
        g2 += d_p1 * d_p1 + d_p2 * d_p2
    e_grad = 0.5 * g2 / (dx * dx * max(a, 1e-30) ** 2)

    denom = e_kin + e_grad
    out = np.full_like(e_kin, np.nan)
    ok = denom > 0
    out[ok] = e_kin[ok] / denom[ok]
    return out


def _ball_offsets(r_min: float, r_max: float) -> np.ndarray:
    """Integer lattice offsets with r_min < |d| <= r_max (self excluded)."""
    r = int(math.ceil(r_max))
    g = np.arange(-r, r + 1)
    d = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    n2 = np.sum(d * d, axis=1)
    keep = (n2 > r_min * r_min) & (n2 <= r_max * r_max)
    return d[keep]


class _VoxelIndex:
    """O(log n) periodic lookup 'is this voxel a string core, and which index?'.

    Uses sorted flat keys rather than an N³ index array, so cost is set by the
    number of core voxels, not by the lattice size.
    """

    def __init__(self, coords: np.ndarray, N: int):
        self.N = N
        self.coords = coords
        keys = np.ravel_multi_index(coords.T, (N, N, N))
        self.order = np.argsort(keys)
        self.keys_sorted = keys[self.order]

    def lookup(self, offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(index of the neighbour at +offset, validity mask) for every voxel."""
        shifted = np.ravel_multi_index(
            ((self.coords + offset) % self.N).T, (self.N,) * 3
        )
        pos = np.searchsorted(self.keys_sorted, shifted)
        np.clip(pos, 0, len(self.keys_sorted) - 1, out=pos)
        valid = self.keys_sorted[pos] == shifted
        return self.order[pos], valid


def string_geometry(
    coords: np.ndarray,
    wvec: np.ndarray,
    N: int,
    *,
    smooth_radius: float = 3.0,
    kink_sep: float = 6.0,
) -> Dict[str, np.ndarray]:
    """Smoothed tangent, coherence and turning angle for every core voxel.

    The plaquette winding vector **W** points along the string, but on a single
    voxel it is quantised to a few lattice directions: a straight *diagonal*
    string alternates between x- and y-faces, so a naive ``|ΣW|/Σ|W|`` would
    read ``‖t‖₂/‖t‖₁`` — as low as 0.58 — for a perfectly straight segment.

    So the tangent is built in two stages. First **W** is summed over a ball of
    radius ``smooth_radius`` and normalised, which averages the quantisation away
    and yields a genuine unit tangent û. Then

        coherence = |Σ û| / count

    over the same ball, which is 1 for any straight segment regardless of its
    orientation, drops at a kink, and →0 where the string doubles back on itself
    (the lattice analogue of a cusp).

    ``turn_deg`` is the angle between û at a voxel and at its neighbours roughly
    ``kink_sep`` away, i.e. the bend of the curve on that scale.
    """
    n = len(coords)
    out = {
        "tangent": np.zeros((n, 3)),
        "coherence": np.full(n, np.nan),
        "turn_deg": np.full(n, np.nan),
    }
    if n == 0:
        return out

    index = _VoxelIndex(coords, N)
    ball = _ball_offsets(0.0, smooth_radius)
    neighbours = [index.lookup(off) for off in ball]

    tang = wvec.copy()
    for nb, ok in neighbours:
        tang += np.where(ok[:, None], wvec[nb], 0.0)

    tnorm = np.linalg.norm(tang, axis=1)
    good = tnorm > 0
    unit = np.zeros_like(tang)
    unit[good] = tang[good] / tnorm[good, None]
    out["tangent"] = unit

    # Stage 2: coherence of the unit tangents over the same ball. W carries a
    # globally consistent orientation along a string, so anti-parallel tangents
    # mean the string genuinely folds back on itself and must be allowed to
    # cancel here — that cancellation is the cusp signature.
    acc = unit.copy()
    cnt = good.astype(np.float64)
    for nb, ok in neighbours:
        use = ok & good & good[nb]
        acc += np.where(use[:, None], unit[nb], 0.0)
        cnt += use
    has = good & (cnt > 0)
    out["coherence"][has] = np.linalg.norm(acc[has], axis=1) / cnt[has]

    cos_sum = np.zeros(n)
    cos_cnt = np.zeros(n)
    for off in _ball_offsets(smooth_radius, kink_sep):
        nb, ok = index.lookup(off)
        ok = ok & good & good[nb]
        if not ok.any():
            continue
        c = np.abs(np.sum(unit * unit[nb], axis=1))
        cos_sum += np.where(ok, c, 0.0)
        cos_cnt += ok
    has_t = cos_cnt > 0
    mean_cos = np.clip(cos_sum[has_t] / cos_cnt[has_t], -1.0, 1.0)
    out["turn_deg"][has_t] = np.degrees(np.arccos(mean_cos))
    return out


def loop_kink_cusp(
    turn_deg: np.ndarray,
    coherence: np.ndarray,
    v2: Optional[np.ndarray],
    *,
    kink_deg: float = 35.0,
    cusp_v2: float = 0.6,
    coherence_cut: float = 0.6,
) -> Dict[str, float]:
    """Aggregate per-voxel geometry into per-loop kink/cusp fractions."""
    out = {
        "coherence_mean": float("nan"),
        "kink_frac": float("nan"),
        "cusp_frac": float("nan"),
    }
    ok_c = np.isfinite(coherence)
    if ok_c.any():
        out["coherence_mean"] = float(np.mean(coherence[ok_c]))
    ok_t = np.isfinite(turn_deg)
    if ok_t.any():
        out["kink_frac"] = float(np.mean(turn_deg[ok_t] > kink_deg))
    if v2 is not None:
        # cusp candidate: string doubling back *and* moving relativistically
        ok = ok_c & np.isfinite(v2)
        if ok.any():
            cusp = (coherence[ok] < coherence_cut) & (np.asarray(v2)[ok] > cusp_v2)
            out["cusp_frac"] = float(np.mean(cusp))
    return out


# ---------------------------------------------------------------------------
# energy channels (slab-wise so peak RAM stays ~ one slab, not one more box)
# ---------------------------------------------------------------------------
def energy_channels(
    snap: Dict[str, Any],
    core_mask: np.ndarray,
    shell_mask: np.ndarray,
    scales: Dict[str, float],
    *,
    slab: int = 16,
) -> Dict[str, float]:
    """Kinetic / radial-gradient / Goldstone-gradient / potential per region.

    The gradient energy splits exactly into the two physical channels

        ½|∇Φ|² = ½(∇ρ)² + ½ρ²(∇θ)²

    which, in terms of the Cartesian components, is the Lagrange identity

        (φ₁∂φ₁ + φ₂∂φ₂)²/ρ²  +  (φ₁∂φ₂ − φ₂∂φ₁)²/ρ²  =  (∂φ₁)² + (∂φ₂)²
              radial                    Goldstone

    Working from φ₁, φ₂ rather than θ keeps the split exact and sidesteps the
    2π branch cut. Inside the core (ρ→0) the split is meaningless, so those
    voxels are booked entirely to the radial channel; the *sum* stays exact.

    The Goldstone channel is the one that drains a **global** string network.
    """
    phi1 = snap["phi1"]
    phi2 = snap["phi2"]
    N = phi1.shape[0]
    a = float(snap.get("a", 1.0))
    dx = float(scales["dx_com"])
    dV = dx**3
    pref = 0.5 / ((dx * dx) * (max(a, 1e-30) ** 2))

    mphi = float(scales["mphi"])
    lam = float(scales["lam"])
    v_vev = mphi / math.sqrt(max(lam, 1e-30))
    rho2_floor = (1e-6 * v_vev) ** 2

    acc = {
        f"{q}_{reg}": 0.0
        for q in ("E_kin", "E_grad_rad", "E_grad_gold", "E_pot")
        for reg in ("core", "shell", "bulk")
    }

    has_pi = snap.get("pi1") is not None and snap.get("pi2") is not None
    kin_scale = phidot_scale(scales["f_star"], scales["omega_star"], a) if has_pi else 0.0

    for z0 in range(0, N, slab):
        z1 = min(z0 + slab, N)
        zc = slice(z0, z1)
        zi = list(range(z0, z1)) + [z1 % N]  # +1 halo plane along z (periodic)
        p1s = np.asarray(phi1[:, :, zi], dtype=np.float64)
        p2s = np.asarray(phi2[:, :, zi], dtype=np.float64)
        a1 = p1s[:, :, :-1]
        a2 = p2s[:, :, :-1]

        rho2 = a1 * a1 + a2 * a2
        core_like = rho2 < rho2_floor
        inv_rho2 = 1.0 / np.maximum(rho2, rho2_floor)

        g_rad = np.zeros_like(a1)
        g_gold = np.zeros_like(a1)
        for axis in (0, 1, 2):
            if axis == 2:
                d1 = p1s[:, :, 1:] - a1
                d2 = p2s[:, :, 1:] - a2
            else:
                d1 = np.roll(a1, -1, axis=axis) - a1
                d2 = np.roll(a2, -1, axis=axis) - a2
            tot = d1 * d1 + d2 * d2
            rad = (a1 * d1 + a2 * d2) ** 2 * inv_rho2
            np.copyto(rad, tot, where=core_like)
            g_rad += rad
            g_gold += tot - rad
            del d1, d2, tot, rad
        g_rad *= pref
        g_gold *= pref
        del inv_rho2, core_like

        e_pot = tree_potential_density(np.sqrt(rho2), mphi, lam)
        del rho2

        if has_pi:
            q1 = np.asarray(snap["pi1"][:, :, zc], dtype=np.float64) * kin_scale
            q2 = np.asarray(snap["pi2"][:, :, zc], dtype=np.float64) * kin_scale
            e_kin = 0.5 * (q1 * q1 + q2 * q2)
            del q1, q2
        else:
            e_kin = None

        m_core = core_mask[:, :, zc]
        m_shell = shell_mask[:, :, zc] & ~m_core
        m_bulk = ~(m_core | m_shell)

        for reg, m in (("core", m_core), ("shell", m_shell), ("bulk", m_bulk)):
            if not m.any():
                continue
            acc[f"E_grad_rad_{reg}"] += float(g_rad[m].sum()) * dV
            acc[f"E_grad_gold_{reg}"] += float(g_gold[m].sum()) * dV
            acc[f"E_pot_{reg}"] += float(e_pot[m].sum()) * dV
            if e_kin is not None:
                acc[f"E_kin_{reg}"] += float(e_kin[m].sum()) * dV

        del p1s, p2s, g_rad, g_gold, e_pot
        if e_kin is not None:
            del e_kin

    acc["E_total"] = float(sum(acc.values()))
    acc["has_pi"] = has_pi
    return acc


PROBE_CHANNELS = ("kin", "gold", "rad", "pot")


def probe_energy(
    snap: Dict[str, Any],
    center: np.ndarray,
    radius: float,
    scales: Dict[str, float],
    *,
    offsets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Energy channels inside one periodic sphere — the local 'did it radiate?' probe.

    Sampling a fixed sphere around a loop before and after it vanishes shows
    whether its core energy reappeared as radiation in the same place.
    """
    phi1 = snap["phi1"]
    phi2 = snap["phi2"]
    N = phi1.shape[0]
    a = float(snap.get("a", 1.0))
    dx = float(scales["dx_com"])
    dV = dx**3
    mphi = float(scales["mphi"])
    lam = float(scales["lam"])
    v_vev = mphi / math.sqrt(max(lam, 1e-30))
    rho2_floor = (1e-6 * v_vev) ** 2

    if offsets is None:
        offsets = _ball_offsets(0.0, radius)
    c = np.rint(np.asarray(center, dtype=np.float64)).astype(np.int64)
    pts = (offsets + c[None, :]) % N
    i, j, k = pts[:, 0], pts[:, 1], pts[:, 2]

    a1 = np.asarray(phi1[i, j, k], dtype=np.float64)
    a2 = np.asarray(phi2[i, j, k], dtype=np.float64)
    rho2 = a1 * a1 + a2 * a2
    core_like = rho2 < rho2_floor
    inv_rho2 = 1.0 / np.maximum(rho2, rho2_floor)

    g_rad = np.zeros_like(a1)
    g_gold = np.zeros_like(a1)
    for axis in range(3):
        sh = pts.copy()
        sh[:, axis] = (sh[:, axis] + 1) % N
        d1 = np.asarray(phi1[sh[:, 0], sh[:, 1], sh[:, 2]], dtype=np.float64) - a1
        d2 = np.asarray(phi2[sh[:, 0], sh[:, 1], sh[:, 2]], dtype=np.float64) - a2
        tot = d1 * d1 + d2 * d2
        rad = (a1 * d1 + a2 * d2) ** 2 * inv_rho2
        np.copyto(rad, tot, where=core_like)
        g_rad += rad
        g_gold += tot - rad
    pref = 0.5 / ((dx * dx) * (max(a, 1e-30) ** 2))

    out = {
        "rad": float(g_rad.sum()) * pref * dV,
        "gold": float(g_gold.sum()) * pref * dV,
        "pot": float(tree_potential_density(np.sqrt(rho2), mphi, lam).sum()) * dV,
        "kin": 0.0,
    }
    if snap.get("pi1") is not None and snap.get("pi2") is not None:
        s = phidot_scale(scales["f_star"], scales["omega_star"], a)
        q1 = np.asarray(snap["pi1"][i, j, k], dtype=np.float64) * s
        q2 = np.asarray(snap["pi2"][i, j, k], dtype=np.float64) * s
        out["kin"] = float((0.5 * (q1 * q1 + q2 * q2)).sum()) * dV
    return out


def dilate_coords(coords: np.ndarray, N: int, radius: int) -> np.ndarray:
    """Boolean near-field mask built by scattering a ball around each core voxel."""
    mask = np.zeros((N, N, N), dtype=bool)
    if len(coords) == 0:
        return mask
    r = int(radius)
    offs = [
        (dx, dy, dz)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dz in range(-r, r + 1)
        if dx * dx + dy * dy + dz * dz <= r * r
    ]
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
    for dx, dy, dz in offs:
        mask[(i + dx) % N, (j + dy) % N, (k + dz) % N] = True
    return mask


# ---------------------------------------------------------------------------
# per-snapshot loop census
# ---------------------------------------------------------------------------
def analyze_snapshot_loops(
    snap: Dict[str, Any],
    scales: Dict[str, float],
    *,
    min_voxels: int = 6,
    max_loops: int = 4000,
    do_kinks: bool = True,
    smooth_radius: float = 3.0,
    kink_sep: float = 6.0,
    kink_deg: float = 35.0,
    cusp_v2: float = 0.6,
    legacy_winding: bool = False,
    slab: int = 16,
    max_label_voxels: int = 40_000_000,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, int]:
    """Label loops and measure per-loop properties.

    Returns (loops, core_mask, core_coords, n_string_voxels).
    """
    from scipy.ndimage import label as ndimage_label

    theta = snap["theta"]
    N = int(snap.get("N", theta.shape[0]))
    dx = float(scales["dx_com"])

    if legacy_winding:
        wmag = np.abs(snap["winding"])
    else:
        wmag = winding_magnitude(theta, slab=slab)
    core_mask = wmag > WIND_THRESHOLD
    n_vox = int(core_mask.sum())
    if n_vox == 0:
        return [], core_mask, np.zeros((0, 3), dtype=np.int64), 0
    if n_vox > max_label_voxels:
        # Right after the transition the whole box is string; labelling (and the
        # near-field shell) would blow up. The energy budget still works, so keep
        # going with an empty census rather than materialising the coordinates.
        LOG.warning("  %d string voxels > label cap — skipping loop census", n_vox)
        return [], core_mask, np.zeros((0, 3), dtype=np.int64), n_vox

    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity
    labelled, n_lab = ndimage_label(core_mask, structure=structure)
    coords_all = np.argwhere(core_mask)
    ii, jj, kk = coords_all[:, 0], coords_all[:, 1], coords_all[:, 2]
    labs = labelled[ii, jj, kk]
    w_all = np.asarray(wmag[ii, jj, kk], dtype=np.float64)
    del labelled

    v2_all = local_v2_on_coords(snap, coords_all, scales)
    if do_kinks:
        wvec = winding_vector_at_coords(theta, coords_all)
        geom = string_geometry(
            coords_all, wvec, N,
            smooth_radius=smooth_radius,
            kink_sep=kink_sep,
        )
        del wvec
    else:
        geom = None

    order = np.argsort(labs, kind="stable")
    labs_s = labs[order]
    coords_s = coords_all[order]
    w_s = w_all[order]
    v2_s = v2_all[order] if v2_all is not None else None
    turn_s = geom["turn_deg"][order] if geom else None
    coh_s = geom["coherence"][order] if geom else None
    bounds = np.searchsorted(labs_s, np.arange(1, n_lab + 1), side="left")
    bounds = np.append(bounds, len(labs_s))

    counts = np.diff(bounds)
    big = np.argsort(counts)[::-1][:max_loops]

    loops: List[Dict[str, Any]] = []
    for li in big:
        lo, hi = bounds[li], bounds[li + 1]
        if hi - lo < min_voxels:
            continue
        c = coords_s[lo:hi]
        rg, ext = periodic_radius_of_gyration(c, N)
        v2 = v2_s[lo:hi] if v2_s is not None else None
        entry: Dict[str, Any] = {
            "label": int(li + 1),
            "n_voxels": int(hi - lo),
            "L_comoving": float(w_s[lo:hi].sum() * dx),
            "R_gyr_vox": rg,
            "extent_vox": ext,
            "centroid": periodic_centroid(c, N),
            "v2_mean": float(np.nanmean(v2)) if v2 is not None else float("nan"),
            "v2_p95": float(np.nanpercentile(v2, 95)) if v2 is not None else float("nan"),
            "v2_max": float(np.nanmax(v2)) if v2 is not None else float("nan"),
            "coherence_mean": float("nan"),
            "kink_frac": float("nan"),
            "cusp_frac": float("nan"),
        }
        if geom is not None:
            entry.update(
                loop_kink_cusp(
                    turn_s[lo:hi], coh_s[lo:hi], v2,
                    kink_deg=kink_deg,
                    cusp_v2=cusp_v2,
                )
            )
            # Below ~kink_sep the loop's own curvature is indistinguishable from
            # a kink, so the turning-angle statistic carries no information.
            if not (rg > kink_sep):
                entry["kink_frac"] = float("nan")
        loops.append(entry)

    return loops, core_mask, coords_all, n_vox


def match_loops(
    prev: List[Dict[str, Any]],
    cur: List[Dict[str, Any]],
    N: int,
    *,
    max_move_vox: float,
    size_ratio: float = 6.0,
) -> Dict[int, int]:
    """Greedy nearest-centroid matching cur→prev (indices into each list)."""
    if not prev or not cur:
        return {}
    pc = np.array([p["centroid"] for p in prev])
    cc = np.array([c["centroid"] for c in cur])
    pn = np.array([p["n_voxels"] for p in prev], dtype=np.float64)
    cn = np.array([c["n_voxels"] for c in cur], dtype=np.float64)

    d = periodic_delta(cc[:, None, :], pc[None, :, :], N)
    dist = np.sqrt(np.sum(d * d, axis=2))
    ratio = np.maximum(cn[:, None] / np.maximum(pn[None, :], 1.0),
                       pn[None, :] / np.maximum(cn[:, None], 1.0))
    dist = np.where((dist <= max_move_vox) & (ratio <= size_ratio), dist, np.inf)

    out: Dict[int, int] = {}
    used_prev: set[int] = set()
    flat = np.argsort(dist, axis=None)
    for f in flat:
        ci, pi = np.unravel_index(f, dist.shape)
        if not np.isfinite(dist[ci, pi]):
            break
        if ci in out or pi in used_prev:
            continue
        out[int(ci)] = int(pi)
        used_prev.add(int(pi))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _load_manifest_any(run_dir: str) -> List[Dict[str, Any]]:
    """Manifest rows from ``field_states/manifest.csv`` or the run root."""
    try:
        return load_manifest_rows(run_dir)
    except FileNotFoundError:
        pass
    path = os.path.join(run_dir, "manifest.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no manifest.csv in {run_dir} or its field_states/")
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    with open(path, newline="") as f:
        for line in f:
            row = parse_manifest_row(line)
            if row is None:
                continue
            try:
                step = int(float(row["step"]))
            except ValueError:
                continue
            if step in seen:
                continue
            seen.add(step)
            rows.append(row)
    return rows


def run(
    run_dir: str,
    *,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    out_dir: Optional[str] = None,
    load_pi: bool = True,
    do_kinks: bool = True,
    shell_radius: int = 3,
    min_voxels: int = 6,
    max_loops: int = 4000,
    smooth_radius: float = 3.0,
    kink_sep: float = 6.0,
    kink_deg: float = 35.0,
    cusp_v2: float = 0.6,
    probe_radius: float = 10.0,
    legacy_winding: bool = False,
    slab: int = 16,
    skip_budget: bool = False,
) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "strings", "events")
    os.makedirs(out_dir, exist_ok=True)

    params = load_run_params(run_dir) or {}
    rows = _load_manifest_any(run_dir)
    rows.sort(key=lambda r: float(r["t"]))
    if step_min is not None:
        rows = [r for r in rows if int(float(r["step"])) >= step_min]
    if step_max is not None:
        rows = [r for r in rows if int(float(r["step"])) <= step_max]
    if not rows:
        raise RuntimeError("No manifest rows in the requested step range")

    monolith: Optional[str]
    try:
        monolith = resolve_h5_path(run_dir, rows)
    except FileNotFoundError:
        monolith = None
    time_index = (
        build_h5_time_index(monolith, "phi_0")
        if monolith and os.path.isfile(monolith)
        else None
    )

    tracks_path = os.path.join(out_dir, "loop_tracks.csv")
    events_path = os.path.join(out_dir, "loop_events.csv")
    budget_path = os.path.join(out_dir, "radiation_budget.csv")

    f_tracks = open(tracks_path, "w", newline="")
    f_events = open(events_path, "w", newline="")
    f_budget = open(budget_path, "w", newline="")
    w_tracks = csv.DictWriter(f_tracks, fieldnames=list(TRACK_FIELDS))
    w_events = csv.DictWriter(f_events, fieldnames=list(EVENT_FIELDS))
    w_budget = csv.DictWriter(f_budget, fieldnames=list(BUDGET_FIELDS))
    w_tracks.writeheader()
    w_events.writeheader()
    w_budget.writeheader()

    prev_loops: List[Dict[str, Any]] = []
    prev_t: Optional[float] = None
    prev_step: Optional[int] = None
    prev_birth: Dict[int, int] = {}
    next_track_id = 1

    try:
        for i, row in enumerate(rows, start=1):
            step = int(float(row["step"]))
            t = float(row["t"])
            t0 = time.time()
            try:
                h5_path, kind = resolve_snapshot_h5(run_dir, row, monolith_path=monolith)
            except FileNotFoundError:
                LOG.warning("[%d/%d] step %d: no HDF5 — skip", i, len(rows), step)
                continue

            want_pi = load_pi and h5_has_group(h5_path, "pi_0") and h5_has_group(h5_path, "pi_1")
            snap = read_h5_snapshot(
                h5_path,
                row,
                time_index=time_index if kind == "monolith" else None,
                load_pi=want_pi,
            )
            scales = resolve_lattice_scales(params, row)
            N = int(snap.get("N", snap["winding"].shape[0]))

            loops, core_mask, core_coords, n_vox = analyze_snapshot_loops(
                snap, scales,
                min_voxels=min_voxels,
                max_loops=max_loops,
                do_kinks=do_kinks,
                smooth_radius=smooth_radius,
                kink_sep=kink_sep,
                kink_deg=kink_deg,
                cusp_v2=cusp_v2,
                legacy_winding=legacy_winding,
                slab=slab,
            )
            # theta/rho are only needed for the census; free them before the
            # energy pass, which works straight off phi1/phi2.
            snap.pop("theta", None)
            snap.pop("rho", None)
            snap.pop("winding", None)
            LOG.info(
                "[%d/%d] step %d  t=%.1f  voxels=%d  loops=%d  (%.1fs)",
                i, len(rows), step, t, n_vox, len(loops), time.time() - t0,
            )

            # --- energy channels -------------------------------------------------
            if not skip_budget:
                shell_mask = dilate_coords(core_coords, N, shell_radius)
                ch = energy_channels(snap, core_mask, shell_mask, scales, slab=slab)
                del shell_mask
                gold_b = ch.get("E_grad_gold_bulk", 0.0)
                rad_b = ch.get("E_grad_rad_bulk", 0.0)
                w_budget.writerow(
                    {
                        "step": step, "time": t,
                        "temperature": float(row["T"]), "a": float(row["a"]),
                        "n_string_voxels": n_vox, "n_loops": len(loops),
                        "E_kin_core": ch["E_kin_core"],
                        "E_grad_rad_core": ch["E_grad_rad_core"],
                        "E_grad_gold_core": ch["E_grad_gold_core"],
                        "E_kin_shell": ch["E_kin_shell"],
                        "E_grad_rad_shell": ch["E_grad_rad_shell"],
                        "E_grad_gold_shell": ch["E_grad_gold_shell"],
                        "E_kin_bulk": ch["E_kin_bulk"],
                        "E_grad_rad_bulk": rad_b,
                        "E_grad_gold_bulk": gold_b,
                        "E_pot_core": ch["E_pot_core"],
                        "E_pot_bulk": ch["E_pot_bulk"],
                        "E_total": ch["E_total"],
                        "gold_over_radial_bulk": (gold_b / rad_b) if rad_b > 0 else "",
                        "has_pi": ch["has_pi"],
                    }
                )
                f_budget.flush()

            # --- track / events --------------------------------------------------
            dt = (t - prev_t) if prev_t is not None else float("nan")
            # A loop cannot move further than c·Δt between snapshots. In program
            # units dx_prog = ωStar·dx_phys, so the causal bound in voxels is
            # Δt_prog / dx_prog; allow 1.5× for centroid jitter as loops reconnect.
            dx_prog = max(float(scales["dx_com"]) * float(scales["omega_star"]), 1e-30)
            ct_vox = abs(dt) / dx_prog if np.isfinite(dt) else 0.0
            max_move = float(np.clip(1.5 * ct_vox, 8.0, N / 4.0))

            probe_offsets = _ball_offsets(0.0, probe_radius)
            for lp in loops:
                lp["probe"] = probe_energy(
                    snap, lp["centroid"], probe_radius, scales, offsets=probe_offsets
                )

            m = match_loops(prev_loops, loops, N, max_move_vox=max_move) if prev_loops else {}
            for ci, lp in enumerate(loops):
                pi = m.get(ci)
                if pi is not None:
                    tid = prev_loops[pi]["track_id"]
                    dR = (
                        (lp["R_gyr_vox"] - prev_loops[pi]["R_gyr_vox"]) / dt
                        if np.isfinite(dt) and dt != 0
                        else float("nan")
                    )
                else:
                    tid = next_track_id
                    next_track_id += 1
                    prev_birth[tid] = step
                    dR = float("nan")
                lp["track_id"] = tid
                cen = lp["centroid"]
                w_tracks.writerow(
                    {
                        "step": step, "time": t,
                        "temperature": float(row["T"]), "a": float(row["a"]),
                        "loop_id": tid,
                        "n_voxels": lp["n_voxels"],
                        "L_comoving": lp["L_comoving"],
                        "R_gyr_vox": lp["R_gyr_vox"],
                        "extent_vox": lp["extent_vox"],
                        "cx": cen[0], "cy": cen[1], "cz": cen[2],
                        "v2_mean": lp["v2_mean"],
                        "v2_p95": lp["v2_p95"],
                        "v2_max": lp["v2_max"],
                        "coherence_mean": lp["coherence_mean"],
                        "kink_frac": lp["kink_frac"],
                        "cusp_frac": lp["cusp_frac"],
                        "matched_prev_id": (
                            prev_loops[pi]["track_id"] if pi is not None else ""
                        ),
                        "dR_dt": dR,
                        **{f"probe_{c}": lp["probe"][c] for c in PROBE_CHANNELS},
                    }
                )

            matched_prev = set(m.values())
            for pi, lp in enumerate(prev_loops):
                if pi in matched_prev:
                    continue
                cen = lp["centroid"]
                tid = lp["track_id"]
                born = prev_birth.get(tid, prev_step if prev_step is not None else -1)
                shrink = (
                    -lp["R_gyr_vox"] / dt if np.isfinite(dt) and dt != 0 else float("nan")
                )
                # Same sphere, same place, one snapshot later: where the energy went.
                before = lp.get("probe", {})
                after = probe_energy(
                    snap, cen, probe_radius, scales, offsets=probe_offsets
                )
                gold_gain = after["gold"] - before.get("gold", float("nan"))
                total_change = sum(after[c] for c in PROBE_CHANNELS) - sum(
                    before.get(c, float("nan")) for c in PROBE_CHANNELS
                )
                w_events.writerow(
                    {
                        "probe_radius_vox": probe_radius,
                        **{f"probe_{c}_before": before.get(c, "") for c in PROBE_CHANNELS},
                        **{f"probe_{c}_after": after[c] for c in PROBE_CHANNELS},
                        "probe_gold_gain": gold_gain,
                        "probe_total_change": total_change,
                        "event": "collapse",
                        "step": step, "time": t,
                        "temperature": float(row["T"]),
                        "loop_id": tid,
                        "n_voxels": lp["n_voxels"],
                        "L_comoving": lp["L_comoving"],
                        "R_gyr_vox": lp["R_gyr_vox"],
                        "v2_mean": lp["v2_mean"],
                        "v2_p95": lp["v2_p95"],
                        "kink_frac": lp["kink_frac"],
                        "cusp_frac": lp["cusp_frac"],
                        "shrink_rate_dR_dt": shrink,
                        "lifetime_steps": (
                            (prev_step - born) if prev_step is not None else ""
                        ),
                        "cx": cen[0], "cy": cen[1], "cz": cen[2],
                    }
                )
            f_tracks.flush()
            f_events.flush()

            prev_loops = loops
            prev_t = t
            prev_step = step
            del snap, core_mask, core_coords
    finally:
        f_tracks.close()
        f_events.close()
        f_budget.close()

    LOG.info("Wrote %s", tracks_path)
    LOG.info("Wrote %s", events_path)
    LOG.info("Wrote %s", budget_path)

    png = os.path.join(out_dir, "string_loop_events.png")
    try:
        plot_events(out_dir, png)
        LOG.info("Wrote %s", png)
    except Exception as exc:
        LOG.warning("plot failed: %s", exc)

    if not skip_budget:
        try:
            print_dissipation(dissipation_summary(out_dir))
        except Exception as exc:
            LOG.warning("dissipation summary failed: %s", exc)
    return out_dir


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _col(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def dissipation_summary(
    out_dir: str,
    *,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Energy released by the string network, estimated three independent ways.

    All ``E_*`` columns are physical energy *densities* integrated over the
    **comoving** volume, i.e. ``E_csv = E_phys / a³``. For a network of fixed
    comoving length ``a²·E_core`` is constant, and for free massless radiation
    ``a⁴·E_bulk`` is constant, so those factors are removed before differencing
    — otherwise expansion alone looks like dissipation.

    The three estimates are:

    ``events``   Σ over collapse events of μ_eff·L, the string energy carried by
                 each loop that vanished. Counts only annihilation.
    ``core``     drop in the total core energy across the window. Counts every
                 way the network loses length, including shrinking without
                 collapsing.
    ``goldstone`` rise in bulk Goldstone energy. This is the radiation side of
                 the ledger, but it also picks up any non-string phase gradients,
                 so treat it as an upper bound.

    Agreement between ``events`` and ``core`` means loop collapse dominates the
    loss. ``goldstone`` matching them means the budget closes.
    """
    tracks = _read_csv(os.path.join(out_dir, "loop_tracks.csv"))
    events = _read_csv(os.path.join(out_dir, "loop_events.csv"))
    budget = _read_csv(os.path.join(out_dir, "radiation_budget.csv"))
    if not budget:
        raise RuntimeError(f"no radiation_budget.csv in {out_dir} (ran with --skip-budget?)")

    def in_window(t: float) -> bool:
        return (t_min is None or t >= t_min) and (t_max is None or t <= t_max)

    # --- per-step network state -------------------------------------------
    L_by_step: Dict[int, float] = {}
    nvox_by_step: Dict[int, float] = {}
    for r in tracks:
        try:
            s = int(float(r["step"]))
        except (TypeError, ValueError):
            continue
        L_by_step[s] = L_by_step.get(s, 0.0) + float(r.get("L_comoving") or 0.0)
        nvox_by_step[s] = nvox_by_step.get(s, 0.0) + float(r.get("n_voxels") or 0.0)

    steps: List[Dict[str, float]] = []
    for r in budget:
        t = float(r["time"])
        if not in_window(t):
            continue
        s = int(float(r["step"]))
        a = float(r["a"])
        core = sum(
            float(r[k] or 0.0)
            for k in ("E_kin_core", "E_grad_rad_core", "E_grad_gold_core", "E_pot_core")
        )
        n_str = float(r["n_string_voxels"] or 0.0)
        L = L_by_step.get(s, float("nan"))
        steps.append(
            {
                "step": s,
                "time": t,
                "a": a,
                "E_core": core,
                "E_core_scaled": core * a * a,
                "E_gold_bulk_scaled": float(r["E_grad_gold_bulk"] or 0.0) * a**4,
                "L_total": L,
                "mu_eff": core / L if L and np.isfinite(L) and L > 0 else float("nan"),
                "coverage": nvox_by_step.get(s, 0.0) / n_str if n_str > 0 else float("nan"),
            }
        )
    if len(steps) < 2:
        raise RuntimeError("need at least two snapshots in the window")
    steps.sort(key=lambda d: d["time"])
    mu_at_step = {int(d["step"]): d["mu_eff"] for d in steps}
    a_at_step = {int(d["step"]): d["a"] for d in steps}
    prev_step_of = {
        int(steps[i]["step"]): int(steps[i - 1]["step"]) for i in range(1, len(steps))
    }

    # --- 1) sum over collapse events --------------------------------------
    e_events = 0.0
    n_events = 0
    n_priced = 0
    n_relativistic = 0
    n_small = 0
    for r in events:
        t = float(r["time"])
        if not in_window(t):
            continue
        n_events += 1
        rg = float(r.get("R_gyr_vox") or "nan")
        if np.isfinite(rg) and rg < 3.0:
            n_small += 1
        sr = float(r.get("shrink_rate_dR_dt") or "nan")
        if np.isfinite(sr) and abs(sr) > 0.5:
            n_relativistic += 1
        s = int(float(r["step"]))
        # the loop was last seen (and last measured) one snapshot earlier
        mu = mu_at_step.get(prev_step_of.get(s, s), float("nan"))
        a_prev = a_at_step.get(prev_step_of.get(s, s), 1.0)
        L = float(r.get("L_comoving") or "nan")
        if np.isfinite(mu) and np.isfinite(L):
            e_events += mu * L * a_prev * a_prev
            n_priced += 1

    # --- 2) core-energy drop, 3) Goldstone rise ---------------------------
    first, last = steps[0], steps[-1]
    e_core_drop = first["E_core_scaled"] - last["E_core_scaled"]
    e_gold_rise = last["E_gold_bulk_scaled"] - first["E_gold_bulk_scaled"]

    coverage = [d["coverage"] for d in steps if np.isfinite(d["coverage"])]
    summary = {
        "window": {"t_first": first["time"], "t_last": last["time"],
                   "n_snapshots": len(steps)},
        "note": "energies are a^n-corrected comoving integrals; see docstring",
        "E_diss_from_events": e_events,
        "E_core_drop": e_core_drop,
        "E_gold_bulk_rise": e_gold_rise,
        "events_over_core_drop": (
            e_events / e_core_drop if e_core_drop else float("nan")
        ),
        "gold_over_core_drop": (
            e_gold_rise / e_core_drop if e_core_drop else float("nan")
        ),
        "n_events": n_events,
        "n_events_priced": n_priced,
        "n_events_relativistic": n_relativistic,
        "n_events_below_3vox": n_small,
        "frac_events_below_3vox": n_small / n_events if n_events else float("nan"),
        "mu_eff_median_GeV2": float(
            np.nanmedian([d["mu_eff"] for d in steps])
        ),
        "E_core_first": first["E_core_scaled"],
        "E_core_last": last["E_core_scaled"],
        "census_coverage_median": float(np.median(coverage)) if coverage else float("nan"),
    }

    path = os.path.join(out_dir, "dissipation_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def print_dissipation(s: Dict[str, Any]) -> None:
    w = s["window"]
    print("\n=== string energy dissipation ===")
    print(f"  window        t ∈ [{w['t_first']:.1f}, {w['t_last']:.1f}]  "
          f"({w['n_snapshots']} snapshots)")
    print(f"  mu_eff        {s['mu_eff_median_GeV2']:.3e} GeV²  (median)")
    print(f"  core energy   {s['E_core_first']:.4e} -> {s['E_core_last']:.4e}")
    print("  --- energy released, three ways ---")
    print(f"  from events   {s['E_diss_from_events']:.4e}   "
          f"({s['n_events_priced']}/{s['n_events']} events priced)")
    print(f"  core drop     {s['E_core_drop']:.4e}")
    print(f"  Goldstone     {s['E_gold_bulk_rise']:.4e}   (upper bound)")
    print(f"  events/core   {s['events_over_core_drop']:.3f}   "
          f"gold/core {s['gold_over_core_drop']:.3f}")
    print("  --- health checks ---")
    print(f"  relativistic collapses (|dR/dt|>0.5): {s['n_events_relativistic']}"
          f"/{s['n_events']}")
    print(f"  collapses at R_gyr < 3 vox:           {s['n_events_below_3vox']}"
          f"/{s['n_events']}  "
          f"({100 * s['frac_events_below_3vox']:.0f}% — resolution limited)")
    cov = s["census_coverage_median"]
    if np.isfinite(cov) and cov < 0.9:
        print(f"  WARNING: loop census covers only {100 * cov:.0f}% of string "
              "voxels; raise --max-loops or lower --min-voxels")


def plot_events(out_dir: str, out_png: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = _read_csv(os.path.join(out_dir, "loop_tracks.csv"))
    events = _read_csv(os.path.join(out_dir, "loop_events.csv"))
    budget = _read_csv(os.path.join(out_dir, "radiation_budget.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)

    # 1) collapse events per snapshot
    ax = axes[0, 0]
    if events:
        te = _col(events, "time")
        uniq, cnt = np.unique(te[np.isfinite(te)], return_counts=True)
        ax.step(uniq, cnt, where="mid", color="C3")
        ax.set_ylabel("collapse events / snapshot")
    ax.set_xlabel(r"$t$ (program)")
    ax.set_title("Loop disappearance rate", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2) size spectrum of collapsing loops
    ax = axes[0, 1]
    if events:
        nv = _col(events, "n_voxels")
        nv = nv[np.isfinite(nv) & (nv > 0)]
        if len(nv):
            ax.hist(nv, bins=np.logspace(0, max(1.1, np.log10(nv.max())), 30),
                    color="C0", alpha=0.85)
            ax.set_xscale("log")
    ax.set_xlabel("loop size at last sighting [voxels]")
    ax.set_ylabel("count")
    ax.set_title("Collapsing-loop size spectrum", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3) local energy probe: did the core energy reappear as radiation?
    ax = axes[0, 2]
    if events:
        gb = _col(events, "probe_gold_before")
        ga = _col(events, "probe_gold_after")
        ok = np.isfinite(gb) & np.isfinite(ga) & (gb > 0)
        if ok.any():
            ax.loglog(gb[ok], ga[ok], "o", ms=3, alpha=0.5, color="C2")
            lim = [min(gb[ok].min(), ga[ok].min()), max(gb[ok].max(), ga[ok].max())]
            ax.plot(lim, lim, "k--", lw=0.8, label="no change")
            frac = float(np.mean(ga[ok] > gb[ok]))
            ax.set_title(
                f"Goldstone energy at collapse site\n({100 * frac:.0f}% gained)",
                fontsize=10,
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("before collapse")
        ax.set_ylabel("after collapse")
    ax.grid(True, alpha=0.3, which="both")

    # 4) energy channels
    ax = axes[1, 0]
    if budget:
        tb = _col(budget, "time")
        for key, c, lab in (
            ("E_kin_core", "C0", r"$E_{\rm kin}$ core"),
            ("E_grad_gold_core", "C1", r"$E_{\rm gold}$ core"),
            ("E_grad_gold_bulk", "C2", r"$E_{\rm gold}$ bulk"),
            ("E_grad_rad_bulk", "C3", r"$E_{\rm radial}$ bulk"),
        ):
            y = _col(budget, key)
            if np.isfinite(y).any():
                ax.plot(tb, np.abs(y), color=c, lw=1.4, label=lab)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_xlabel(r"$t$ (program)")
    ax.set_ylabel("energy")
    ax.set_title("Core vs radiated energy", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 5) Goldstone / radial ratio in the bulk
    ax = axes[1, 1]
    if budget:
        tb = _col(budget, "time")
        ratio = _col(budget, "gold_over_radial_bulk")
        if np.isfinite(ratio).any():
            ax.plot(tb, ratio, "C4-", lw=1.5)
            ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (program)")
    ax.set_ylabel(r"$E_{\rm gold}/E_{\rm radial}$ (bulk)")
    ax.set_title("Radiation channel (bulk)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 6) kink fraction / linearity vs time
    ax = axes[1, 2]
    if tracks:
        tt = _col(tracks, "time")
        kf = _col(tracks, "kink_frac")
        coh = _col(tracks, "coherence_mean")
        cf = _col(tracks, "cusp_frac")
        uniq = np.unique(tt[np.isfinite(tt)])
        if len(uniq):
            def mean_at(vals):
                return [
                    np.nanmean(v) if np.isfinite(v).any() else np.nan
                    for v in (vals[tt == u] for u in uniq)
                ]

            kf_m, coh_m, cf_m = mean_at(kf), mean_at(coh), mean_at(cf)
            ax.plot(uniq, kf_m, "C3-", lw=1.5, label="kink frac")
            ax.plot(uniq, coh_m, "C0--", lw=1.3, label="coherence")
            ax.plot(uniq, cf_m, "C1:", lw=1.5, label="cusp frac")
            ax.legend(fontsize=8)
    ax.set_xlabel(r"$t$ (program)")
    ax.set_title("Kink statistics (loop-averaged)", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("String loop collapse / kink / radiation diagnostics", fontsize=11)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dir", help="CosmoLattice run directory (has field_states/)")
    ap.add_argument("--step-min", type=int, default=None)
    ap.add_argument("--step-max", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-pi", action="store_true", help="Skip pi_* (no v², no E_kin)")
    ap.add_argument("--no-kinks", action="store_true", help="Skip per-voxel PCA stats")
    ap.add_argument("--skip-budget", action="store_true",
                    help="Skip the (slower) energy-channel pass")
    ap.add_argument("--shell-radius", type=int, default=3,
                    help="Near-field shell thickness in voxels (default 3)")
    ap.add_argument("--min-voxels", type=int, default=6)
    ap.add_argument("--max-loops", type=int, default=4000)
    ap.add_argument("--smooth-radius", type=float, default=3.0,
                    help="Ball radius (voxels) for smoothing the winding tangent")
    ap.add_argument("--kink-sep", type=float, default=6.0,
                    help="Separation (voxels) over which the turning angle is taken")
    ap.add_argument("--kink-deg", type=float, default=35.0,
                    help="Turning angle above which a voxel counts as a kink")
    ap.add_argument("--cusp-v2", type=float, default=0.6,
                    help="Local v² above which a low-coherence voxel is a cusp candidate")
    ap.add_argument("--probe-radius", type=float, default=10.0,
                    help="Radius (voxels) of the local energy probe at collapse sites")
    ap.add_argument("--legacy-winding", action="store_true",
                    help="Use the summed scalar winding instead of |W| from the "
                         "orientation-resolved winding vector")
    ap.add_argument("--slab", type=int, default=16,
                    help="z-planes per energy-pass slab; peak RAM ~ 12·N²·slab·8 B")
    ap.add_argument("--plot-only", action="store_true",
                    help="Only re-plot from existing CSVs in --out-dir")
    ap.add_argument("--summarize", action="store_true",
                    help="Only recompute the dissipation summary from existing CSVs")
    ap.add_argument("--window", type=float, nargs=2, metavar=("T_MIN", "T_MAX"),
                    default=None, help="Restrict --summarize to this time window")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    run_dir = os.path.abspath(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "strings", "events")
    if args.plot_only:
        png = plot_events(out_dir, os.path.join(out_dir, "string_loop_events.png"))
        print(f"wrote {png}")
        return 0

    if args.summarize:
        lo, hi = args.window if args.window else (None, None)
        print_dissipation(dissipation_summary(out_dir, t_min=lo, t_max=hi))
        return 0

    run(
        run_dir,
        step_min=args.step_min,
        step_max=args.step_max,
        out_dir=args.out_dir,
        load_pi=not args.no_pi,
        do_kinks=not args.no_kinks,
        shell_radius=args.shell_radius,
        min_voxels=args.min_voxels,
        max_loops=args.max_loops,
        smooth_radius=args.smooth_radius,
        kink_sep=args.kink_sep,
        kink_deg=args.kink_deg,
        cusp_v2=args.cusp_v2,
        probe_radius=args.probe_radius,
        legacy_winding=args.legacy_winding,
        slab=args.slab,
        skip_budget=args.skip_budget,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
