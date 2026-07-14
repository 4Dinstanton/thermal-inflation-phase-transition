#!/usr/bin/env python3
"""Plaquette winding-number density for global U(1) cosmic strings (phi1 + i phi2).

Ported from simulation/latticeSimComplex_numba.py::compute_winding_number.
Vectorized with np.roll for 256^3-scale snapshots.
"""
import numpy as np


def _wrap_phase_diff(d):
    """Map phase difference into (-pi, pi]."""
    return np.arctan2(np.sin(d), np.cos(d))


def _plaquette_winding_arr(th00, th10, th11, th01):
    """Vectorized plaquette winding for four corner phase arrays."""
    total = (
        _wrap_phase_diff(th10 - th00)
        + _wrap_phase_diff(th11 - th10)
        + _wrap_phase_diff(th01 - th11)
        + _wrap_phase_diff(th00 - th01)
    )
    return total / (2.0 * np.pi)


def compute_winding_number(phi1, phi2, out=None):
    """Compute winding number density on a 3D periodic lattice.

    Parameters
    ----------
    phi1, phi2 : array-like, shape (Nx, Ny, Nz)
        Real components of the complex scalar field (GeV or any consistent units).
    out : ndarray, optional
        Pre-allocated output array; created if None.

    Returns
    -------
    winding : ndarray, shape (Nx, Ny, Nz)
        Winding number per site (integer when a string pierces the cell).
    """
    phi1 = np.asarray(phi1, dtype=np.float64)
    phi2 = np.asarray(phi2, dtype=np.float64)
    if phi1.shape != phi2.shape or phi1.ndim != 3:
        raise ValueError("phi1 and phi2 must be same-shaped 3D arrays")

    theta = np.arctan2(phi2, phi1)
    th_ip = np.roll(theta, -1, axis=0)
    th_jp = np.roll(theta, -1, axis=1)
    th_kp = np.roll(theta, -1, axis=2)
    th_ip_jp = np.roll(th_ip, -1, axis=1)
    th_ip_kp = np.roll(th_ip, -1, axis=2)
    th_jp_kp = np.roll(th_jp, -1, axis=2)

    # XY, XZ, YZ plaquettes (same convention as latticeSimComplex_numba)
    w = (
        _plaquette_winding_arr(theta, th_ip, th_ip_jp, th_jp)
        + _plaquette_winding_arr(theta, th_ip, th_ip_kp, th_kp)
        + _plaquette_winding_arr(theta, th_jp, th_jp_kp, th_kp)
    )

    if out is None:
        return w
    out[...] = w
    return out


def string_voxel_fraction(winding, threshold=0.5):
    """Fraction of lattice sites with |winding| > threshold."""
    winding = np.asarray(winding)
    return float(np.mean(np.abs(winding) > threshold))
