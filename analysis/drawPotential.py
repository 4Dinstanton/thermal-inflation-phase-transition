import sympy as sp
import numpy as np
import scipy
from scipy import integrate, interpolate
import scipy.optimize as opt

import matplotlib.pyplot as plt
import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import sys as _sys, os as _os

_sys.path.insert(
    0,
    _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"
    ),
)
import Potential as p
from tunneling_utils import fullTunneling

potential_flag = "fermion_only"


def draw_VVT(phi, VT, T):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    (l,) = ax[0].plot(phi, VT.V_tree(phi), label="T=0", color="blue")
    ax1 = ax[0].twinx()
    (ll,) = ax1.plot(phi, VT.V(phi), label=f"T={T}", color="red")
    ax1.set_ylim([min(VT.V(phi)) - 0.1, (VT.V(np.array([0]).reshape(-1, 1))) + 0.1])
    ax1.set_ylabel(r"$V_T$")
    ax1.yaxis.label.set_color("red")
    ax1.tick_params(axis="y", colors="red")
    ax1.plot(phi, VT.V_tree(phi) + VT.V_T(phi))
    ax[0].yaxis.label.set_color("blue")
    ax[0].tick_params(axis="y", colors="blue")
    leg = [l]
    leg.append(ll)
    labels = [le.get_label() for le in leg]
    ax[0].legend(leg, labels)
    phi = np.linspace(-2 * VT.v, 2 * VT.v, 200).reshape(-1, 1)
    ax[0].set_title("V depending on temperature")
    ax[0].set_ylabel(r"$V(T=0)$")
    (lll,) = ax[1].plot(phi, VT.V_T(phi), label=r"$J_b$", color="green")
    lleg = [lll]
    lab = [lll.get_label() for lll in lleg]
    ax[1].set_xlabel(r"$phi$")
    ax[1].set_ylabel(r"$J_b(\dfrac{m_{\phi}^2}{T^2})$")
    ax[1].set_title(f"Thermal contribution at T={T}")
    print(lleg, lab)
    ax[1].legend(lleg, lab)
    fig.tight_layout()
    # plt.savefig("figs/meeting/V-VT", dpi=300)
    plt.show()


def draw_VT_temp_array(phi, VT, temp_arr, title="toy_model_VT"):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 8))
    leg = []
    T_dict = {}
    phi_zero = np.zeros((1, phi.shape[1])) if phi.ndim > 1 else np.array([0.0])
    for T in temp_arr:
        T = round(T, 3)
        VT.update_T(T)
        if potential_flag == "fermion_only":
            V_at_zero = VT.V_p_fermion_only(phi_zero).item()
            V_shifted = VT.V_p_fermion_only(phi) - V_at_zero
        else:
            V_at_zero = VT.V_p_correct(phi_zero).item()
            V_shifted = VT.V_p_correct(phi) - V_at_zero

        (ll,) = ax1.plot(phi / 1000, V_shifted, label=f"T={T}")
        T_dict.update({T: V_shifted})
        # ax1.set_ylim([min(VT.V(phi))-0.1, (VT.V(np.array([0]).reshape(-1,1))) + 0.1])
        # ax1.set_ylabel(r"$V_T - V_{T_0}$")
        # ax1.yaxis.label.set_color('red')
        # ax1.tick_params(axis='y', colors='red')
        # ax.yaxis.label.set_color('blue')
        # ax.tick_params(axis='y', colors='blue')
        leg.append(ll)
        labels = [le.get_label() for le in leg]
        # ax.legend(leg, labels)
        # ax.set_title('V depending on temperature')
        # ax.set_ylabel(r'$V(T=0)$')
        # fig.tight_layout()
    ax1.set_title("V depending on temperature")
    ax1.set_ylabel(r"$V_T - V_{T_0}$")
    # plt.xlim([1e7, 1.5 * 1e7])
    plt.legend()
    plt.xlabel(r"$\phi (\text{TeV})$")
    # ax1.set_xlim([-4,4])
    # ax1.set_ylim([-5,-0])
    plt.savefig(f"figs/finiteTemp/{title}", dpi=300)
    plt.show()


def format_e(n):
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "E" + a.split("E")[1]


def _V_at(VT, phi_val, pot_flag="fermion_only"):
    """Evaluate the potential at a single field value."""
    X = np.array([[phi_val]])
    if pot_flag == "fermion_only":
        return VT.V_p_fermion_only(X).item()
    return VT.V_p_correct(X).item()


def _V_second_derivative_at_origin(VT, pot_flag="fermion_only", h=1.0):
    """V''(0) via central finite difference."""
    return (
        _V_at(VT, h, pot_flag)
        - 2.0 * _V_at(VT, 0.0, pot_flag)
        + _V_at(VT, -h, pot_flag)
    ) / h**2


def find_barrier_temperature(
    VT, pot_flag="fermion_only", T_lo=100.0, T_hi=50000.0, n_coarse=500, fd_step=1.0
):
    """
    Find T_sp where V''(0) = 0 (barrier near origin disappears).

    Above T_sp the origin is a local minimum (barrier exists);
    below T_sp the curvature is negative (no barrier, no tunneling).

    Returns T_sp or None if no sign change is found.
    """
    T_arr = np.linspace(T_lo, T_hi, n_coarse)
    d2V_vals = np.empty(n_coarse)
    for i, T in enumerate(T_arr):
        VT.update_T(T)
        d2V_vals[i] = _V_second_derivative_at_origin(VT, pot_flag, h=fd_step)

    sign_changes = np.where(np.diff(np.sign(d2V_vals)))[0]
    if len(sign_changes) == 0:
        print(f"  [find_barrier_temperature] No sign change in [{T_lo}, {T_hi}]")
        print(f"  V''(0) range: [{d2V_vals.min():.6e}, {d2V_vals.max():.6e}]")
        return None

    idx = sign_changes[0]

    def _d2V_at_T(T):
        VT.update_T(T)
        return _V_second_derivative_at_origin(VT, pot_flag, h=fd_step)

    T_sp = opt.brentq(_d2V_at_T, T_arr[idx], T_arr[idx + 1], xtol=1e-4)
    return T_sp


def _compute_tv_fv(TEMP, epsil):
    """Deterministic true/false vacuum for a given temperature.

    tv is set to ~10× the escape-point scale (which is O(T)).
    This avoids cosmoTransitions' extend_to_minima allocating arrays
    all the way to the tree-level VEV = sqrt(m²/λ), which explodes
    for very small λ.
    """
    fv = 0 if epsil == 0 else None
    tv = max(10.0 * TEMP, 50000.0)
    return tv, fv


def _make_clipped_potential(V_func, dV_func, phi_cutoff, wall_k=1.0):
    """Wrap V and dV with a quadratic wall beyond phi_cutoff.

    This creates a local minimum near phi_cutoff so that
    cosmoTransitions' extend_to_minima stops there instead of
    extending the path all the way to the tree-level VEV.
    The wall has no effect on the tunneling physics since
    phi_escape << phi_cutoff.
    """

    def V_clipped(X):
        val = V_func(X)
        phi = X[..., 0]
        excess = np.abs(phi) - phi_cutoff
        mask = excess > 0
        if np.any(mask):
            val = np.array(val, dtype=float, copy=True)
            val[mask] += 0.5 * wall_k * excess[mask] ** 2
        return val

    def dV_clipped(X):
        val = np.array(dV_func(X), dtype=float, copy=True)
        phi = X[..., 0]
        excess = np.abs(phi) - phi_cutoff
        mask = excess > 0
        if np.any(mask):
            val[mask] += wall_k * excess[mask] * np.sign(phi[mask])
        return val

    return V_clipped, dV_clipped


def _tunneling_worker(args):
    """Worker for one temperature: builds own VT, runs fullTunneling."""
    TEMP, param_dict, epsil, spline_arrays, pot_flag = args
    VT_w = p.finiteTemperaturePotential(param_dict)
    VT_w.update_T(TEMP)
    VT_w.set_fast_thermal_from_arrays(*spline_arrays)

    tv, fv = _compute_tv_fv(TEMP, epsil)
    if fv is None:
        _, fv = VT_w.find_new_minima()

    if pot_flag == "fermion_only":
        V_raw = VT_w.V_fermion_only
        dV_raw = VT_w.dV_p_fermion_only
    else:
        V_raw = VT_w.V_correct
        dV_raw = VT_w.dV_p_correct

    phi_cutoff = max(15.0 * TEMP, 50000.0)
    mphi = param_dict["mphi"]
    wall_k = 100.0 * mphi**2
    V_func, dV_func = _make_clipped_potential(V_raw, dV_raw, phi_cutoff, wall_k)

    try:
        tunneling_result = CTPD.fullTunneling(
            path_pts=np.array([[tv], [fv]]),
            V=V_func,
            dV=dV_func,
            maxiter=1,
            V_spline_samples=200,
            tunneling_init_params=dict(alpha=2),
            tunneling_findProfile_params=dict(
                xtol=0.00001, phitol=0.00001, rmin=0.00001, npoints=200
            ),
            deformation_class=CTPD.Deformation_Spline,
        )
    except Exception as e:
        print(f"  T={TEMP:.1f}  tunneling FAILED (T_c2 / no barrier): {e}")
        return TEMP, 0.0, 0.0, 0.0, np.array([]), np.array([])

    print("tunneling done!")

    S3 = tunneling_result.action
    S3_T = S3 / TEMP
    _R = tunneling_result.profile1D.R
    _Phi = tv - tunneling_result.profile1D.Phi
    _phi_mid = 0.5 * (_Phi[0] + _Phi[-1])
    r_c = (
        np.interp(_phi_mid, _Phi[::-1], _R[::-1])
        if _Phi[0] > _Phi[-1]
        else np.interp(_phi_mid, _Phi, _R)
    )
    phi_esc = tv - _Phi[0]

    print(
        f"  T={TEMP:.1f}  tv={tv:.1f}  fv={fv:.1f}  "
        f"S3/T={S3_T:.4f}  r_c={r_c:.4e}  phi_esc={phi_esc:.4e}"
    )
    return TEMP, S3_T, r_c, phi_esc, _R.copy(), _Phi.copy()


def gaussian_model(r, phi_fv, phi_tv, r_c, sigma):
    """Gaussian ansatz: phi(r) = phi_fv + (phi_tv - phi_fv) * exp(-(r-r_c)^2 / (2*sigma^2))"""
    return phi_fv + (phi_tv - phi_fv) * np.exp(-0.5 * ((r - r_c) / sigma) ** 2)


def fit_gaussian_bounce(R, Phi):
    """Fit the bounce profile phi(r) to a Gaussian form.

    Returns (popt, pcov, success) where popt = [phi_fv, phi_tv, r_c, sigma].
    """
    if len(R) < 5:
        return None, None, False

    phi_fv = Phi[-1]
    phi_tv = Phi[0]
    r_half = R[np.argmin(np.abs(Phi - 0.5 * (phi_tv + phi_fv)))]
    sigma0 = max(r_half * 0.5, 1e-10)

    try:
        popt, pcov = opt.curve_fit(
            gaussian_model,
            R,
            Phi,
            p0=[phi_fv, phi_tv, 0.0, sigma0],
            maxfev=10000,
        )
        return popt, pcov, True
    except Exception:
        return None, None, False


def plot_bounce_profile(R, Phi, T, output_dir, param_set, pot_flag, S3_T=None):
    """Plot phi(r) from the bounce solution with a Gaussian fit overlay.

    Saves one figure per temperature.
    """
    if len(R) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    ax.plot(
        R, Phi, "o-", color="navy", markersize=2, lw=1.5, label=r"$\phi(r)$ (bounce)"
    )

    popt, _, success = fit_gaussian_bounce(R, Phi)
    if success:
        r_fine = np.linspace(R.min(), R.max(), 500)
        phi_fit = gaussian_model(r_fine, *popt)
        ax.plot(r_fine, phi_fit, "--", color="crimson", lw=1.8, label="Gaussian fit")

        residuals = Phi - gaussian_model(R, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Phi - np.mean(Phi)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fit_text = (
            rf"$\phi_{{fv}} = {popt[0]:.2f}$"
            + "\n"
            + rf"$\phi_{{tv}} = {popt[1]:.2f}$"
            + "\n"
            + rf"$r_c = {popt[2]:.4e}$"
            + "\n"
            + rf"$\sigma = {popt[3]:.4e}$"
            + "\n"
            + rf"$R^2 = {r_squared:.6f}$"
        )
        ax.text(
            0.97,
            0.97,
            fit_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
        )
        is_gaussian = "YES" if r_squared > 0.95 else "NO"
        print(f"  T={T:.2f}  Gaussian fit R²={r_squared:.6f}  → {is_gaussian}")
    else:
        ax.text(
            0.97,
            0.97,
            "Gaussian fit failed",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            color="red",
        )
        print(f"  T={T:.2f}  Gaussian fit FAILED")

    title = f"Bounce profile at T = {T:.2f} GeV"
    if S3_T is not None and S3_T > 0:
        title += rf"  ($S_3/T = {S3_T:.2f}$)"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$r$ [GeV$^{-1}$]", fontsize=12)
    ax.set_ylabel(r"$\phi(r)$ [GeV]", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _os.makedirs(output_dir, exist_ok=True)
    fname = f"bounce_phi_r_T_{T:.2f}_{param_set}_{pot_flag}.png"
    fig_path = _os.path.join(output_dir, fname)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"    Saved: {fig_path}")


if __name__ == "__main__":
    import time as _time

    lam = 1e-24
    v = 1e6
    mphi = 1000
    epsil = 0
    lambdaSix = 0
    T = 0.6

    print("D", 4 * lam**2 - 5 * lambdaSix * mphi**2)

    # set 6 : thermal coupling g= 1.05, lambda = 1.09
    # ser 7 : thermal coupling g= 1.31, lambda = 1.35

    bosonMassSquared = 1000000
    bosonCoupling = 1.09
    bosonGaugeCoupling = 1.05
    fermionCoupling = 1.09
    fermionGaugeCoupling = 1.05
    nb = 20
    nf = 20
    param_set = "set8"

    param = {
        param_set: {
            "lambda": lam,
            "mphi": mphi,
            "epsilon": epsil,
            "lambdaSix": lambdaSix,
            "bosonMassSquared": bosonMassSquared,
            "bosonCoupling": bosonCoupling,
            "bosonGaugeCoupling": bosonGaugeCoupling,
            "fermionCoupling": fermionCoupling,
            "fermionGaugeCoupling": fermionGaugeCoupling,
            "nb": nb,
            "nf": nf,
        }
    }

    VT = p.finiteTemperaturePotential(param[param_set])
    VT.update_T(T)
    VT.build_fast_thermal(x_max=150.0, n_pts=4096)

    phi = np.linspace(-2 * VT.v, 2 * VT.v, 200).reshape(-1, 1)

    # --- Find T_sp (barrier disappearance temperature) ---
    print("\n--- Finding barrier disappearance temperature T_sp ---")
    T_sp = find_barrier_temperature(
        VT, pot_flag=potential_flag, T_lo=100.0, T_hi=10000.0, n_coarse=500, fd_step=1.0
    )
    if T_sp is not None:
        print(f"  T_sp = {T_sp:.4f} GeV  ({T_sp/1e3:.4f} TeV)")
        TEMP_START = T_sp
    else:
        print("  T_sp not found, falling back to manual start = 6730 GeV")
        TEMP_START = 6730.0

    TEMP_RANGE = 300.0
    N_TEMPS = 30
    TEMP_LIST = np.linspace(TEMP_START, TEMP_START + TEMP_RANGE, N_TEMPS)
    print(
        f"  Temperature scan: [{TEMP_START:.2f}, {TEMP_START + TEMP_RANGE:.2f}] GeV, "
        f"{N_TEMPS} points"
    )

    window = 0.000000000002
    phi = np.linspace(-window * VT.v, window * VT.v, 500).reshape(-1, 1)
    N_TEMPS_PLOT = 5
    TEMP_LIST_PLOT = np.linspace(TEMP_START, TEMP_START + TEMP_RANGE, N_TEMPS_PLOT)
    draw_VT_temp_array(
        phi, VT, TEMP_LIST_PLOT, title=f"T-S_{param_set}_{potential_flag}"
    )

    # window = 0.00000002
    # phi = np.linspace(-window * VT.v, window * VT.v, 500).reshape(-1, 1)
    # VT.update_T(TEMP_LIST[0])

    # draw_VVT(phi, VT, TEMP_PLOT)

    # TEMP_LIST = np.arange(0.75, 0.78, 0.001)
    # VT.update_T(0.74)

    # tv, fv = VT.find_new_minima()
    # print(tv, fv)

    # plt.plot(phi, VT.V(phi))
    # plt.plot(phi[:-1], np.diff(VT.V_T(phi), n=1))
    # plt.plot(phi, VT.dJb_exact2(VT.mphi2(phi)/VT.T**2)* (VT._dm2dphi_boson(phi) / VT.T **2))
    # print(min(VT.V(phi)))
    # plt.ylim([(VT.V(np.array([0]).reshape(-1,1)))-0.05, (VT.V(np.array([0]).reshape(-1,1))) + 0.05])

    # plt.show()

    epsil_arr = np.arange(10, 1.7, -0.01)

    spline_arrays = VT._fast_arrays
    N_WORKERS = min(len(TEMP_LIST), multiprocessing.cpu_count())
    N_WORKERS = 1
    print(
        f"\n--- Running tunneling for {len(TEMP_LIST)} temperatures "
        f"with {N_WORKERS} workers ---"
    )
    print(TEMP_LIST)

    _t_start = _time.time()

    worker_args = [
        (TEMP, param[param_set], epsil, spline_arrays, potential_flag)
        for TEMP in TEMP_LIST
    ]

    S3_T_list = [None] * len(TEMP_LIST)
    S4_list = []
    r_c_list = [None] * len(TEMP_LIST)
    phi_esc_list = [None] * len(TEMP_LIST)
    bounce_profiles = [None] * len(TEMP_LIST)

    temp_to_idx = {float(T): i for i, T in enumerate(TEMP_LIST)}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_tunneling_worker, a): a[0] for a in worker_args}
        for fut in as_completed(futures):
            TEMP, S3_T, r_c, phi_esc, _R, _Phi = fut.result()
            idx = temp_to_idx[float(TEMP)]
            S3_T_list[idx] = S3_T
            r_c_list[idx] = r_c
            phi_esc_list[idx] = phi_esc
            bounce_profiles[idx] = (_R, _Phi)

    _t_end = _time.time()
    print(f"\n--- All tunneling done in {_t_end - _t_start:.1f}s ---")

    # --- Plot bounce profiles phi(r) with Gaussian fit ---
    bounce_dir = f"figs/bounce_profiles/{param_set}"
    print(f"\n--- Plotting bounce profiles → {bounce_dir} ---")
    for i, TEMP in enumerate(TEMP_LIST):
        if bounce_profiles[i] is None:
            continue
        _R, _Phi = bounce_profiles[i]
        if len(_R) == 0:
            continue
        plot_bounce_profile(
            _R,
            _Phi,
            TEMP,
            bounce_dir,
            param_set,
            potential_flag,
            S3_T=S3_T_list[i],
        )
    print("--- Bounce profile plots done ---\n")

    """
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    l, = ax.plot(TEMP_LIST, S4_list, label=r'$S_4$', color='blue')
    ax.set_xlabel('T')
    ax1 = ax.twinx()
    ll, = ax1.plot(TEMP_LIST, S3_T_list, label=r'$S_3/T$', color='red')
    ax1.set_ylabel(r'$S_3/T$')
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('blue')
    ax.tick_params(axis='y', colors='blue')
    leg = [l]
    leg.append(ll)
    labels = [le.get_label() for le in leg]
    ax.set_ylabel(r'$S_4$')
    ax.legend(leg, labels)
    fig.tight_layout()
    #plt.savefig("figs/meeting/V-VT", dpi=300)
    plt.savefig(f"figs/finiteTemp/T-S_{param_set}_p2", dpi=300)
    """
    import pandas as pd

    df = pd.DataFrame()
    df["T"] = TEMP_LIST
    # df["S4"] = S4_list
    df["S3/T"] = S3_T_list
    df["r_c"] = r_c_list
    df["phi_esc"] = phi_esc_list
    import os
    import json

    csv_path = f"data/tunneling/{param_set}"
    os.makedirs(csv_path, exist_ok=True)
    df.to_csv(
        f"{csv_path}/T-S_param_{param_set}_lambdaSix_{format_e(lambdaSix)}_{potential_flag}.csv"
    )
    json.dump(param[param_set], open(f"{csv_path}/parameters.json", "w"))
    plt.show()
    print(S3_T_list)
    print(S4_list)
