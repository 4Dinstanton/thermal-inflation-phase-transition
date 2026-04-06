"""
Local wrapper around cosmoTransitions.pathDeformation.fullTunneling
that exposes ``extend_to_minima`` as a parameter (default **False**).

The upstream ``fullTunneling`` hard-codes ``extend_to_minima=True``,
which causes ``SplinePath`` to search for the tree-level VEV along
the tunneling path.  For very small quartic coupling λ the VEV is
astronomically large and the resulting arrays consume tens of GB.

Usage
-----
    from tunneling_utils import fullTunneling
    result = fullTunneling(
        path_pts=..., V=..., dV=...,
        extend_to_minima=False,   # <-- the whole point
        ...                       # same kwargs as CTPD.fullTunneling
    )
"""

import numpy as np
from collections import namedtuple

import cosmoTransitions.pathDeformation as CTPD
from cosmoTransitions.pathDeformation import (
    SplinePath,
    Deformation_Spline,
    DeformationError,
)
from cosmoTransitions import tunneling1D


def fullTunneling(
    path_pts,
    V,
    dV,
    maxiter=20,
    fixEndCutoff=0.03,
    save_all_steps=False,
    verbose=False,
    callback=None,
    callback_data=None,
    V_spline_samples=100,
    tunneling_class=tunneling1D.SingleFieldInstanton,
    tunneling_init_params=None,
    tunneling_findProfile_params=None,
    deformation_class=Deformation_Spline,
    deformation_init_params=None,
    deformation_deform_params=None,
    extend_to_minima=False,
):
    """Drop-in replacement for ``CTPD.fullTunneling`` with controllable
    ``extend_to_minima`` (default False)."""
    if tunneling_init_params is None:
        tunneling_init_params = {}
    if tunneling_findProfile_params is None:
        tunneling_findProfile_params = {}
    if deformation_init_params is None:
        deformation_init_params = {}
    if deformation_deform_params is None:
        deformation_deform_params = {}

    assert maxiter > 0
    pts = np.asanyarray(path_pts)
    saved_steps = []
    deformation_init_params["save_all_steps"] = save_all_steps

    for num_iter in range(1, maxiter + 1):
        if verbose:
            print("Starting tunneling step %i" % num_iter)
        path = SplinePath(
            pts, V, dV,
            V_spline_samples=V_spline_samples,
            extend_to_minima=extend_to_minima,
        )
        if V_spline_samples is not None:
            tobj = tunneling_class(
                0.0, path.L, path.V, path.dV, path.d2V,
                **tunneling_init_params,
            )
        else:
            tobj = tunneling_class(
                0.0, path.L, path.V, path.dV, None,
                **tunneling_init_params,
            )
        profile1D = tobj.findProfile(**tunneling_findProfile_params)
        phi, dphi = profile1D.Phi, profile1D.dPhi
        phi, dphi = tobj.evenlySpacedPhi(
            phi, dphi, npoints=len(phi), fixAbs=False
        )
        dphi[0] = dphi[-1] = 0.0

        pts = path.pts(phi)
        deform_obj = deformation_class(
            pts, dphi, dV, **deformation_init_params
        )
        if callback and not callback(path, tobj, profile1D, callback_data):
            break
        try:
            converged = deform_obj.deformPath(**deformation_deform_params)
        except DeformationError as err:
            print(err.args[0])
            converged = False
        pts = deform_obj.phi
        if save_all_steps:
            saved_steps.append(deform_obj.phi_list)
        if converged and deform_obj.num_steps < 2:
            break
    else:
        if verbose:
            print("Reached maxiter in fullTunneling. No convergence.")

    deform_obj = deformation_class(pts, dphi, dV, **deformation_init_params)
    F, dV_forces = deform_obj.forces()
    F_max = np.max(np.sqrt(np.sum(F * F, -1)))
    dV_max = np.max(np.sqrt(np.sum(dV_forces * dV_forces, -1)))
    fRatio = F_max / dV_max

    rtuple = namedtuple(
        "fullTunneling_rval", "profile1D Phi action fRatio saved_steps"
    )
    Phi = path.pts(profile1D.Phi)
    action = tobj.findAction(profile1D)
    return rtuple(profile1D, Phi, action, fRatio, saved_steps)
