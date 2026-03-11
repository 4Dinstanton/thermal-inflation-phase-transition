import sympy as sp
import numpy as np
import scipy
from scipy import integrate, interpolate

import matplotlib.pyplot as plt
import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"))
import Potential as p

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


def _compute_tv_fv(TEMP, epsil):
    """Deterministic true/false vacuum for a given temperature."""
    fv = 0 if epsil == 0 else None
    if TEMP < 10000:
        tv = 100000
    elif TEMP < 12000:
        tv = 120000 * TEMP / 10000
    elif TEMP < 60000:
        tv = 1000000 * TEMP / 12000
    else:
        tv = 3500000 * TEMP / 12000
    return tv, fv


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
        V_func = VT_w.V_fermion_only
        dV_func = VT_w.dV_p_fermion_only
    else:
        V_func = VT_w.V_correct
        dV_func = VT_w.dV_p_correct

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
    print("tunneling done!")

    S3 = tunneling_result.action
    S3_T = S3 / TEMP
    _R = tunneling_result.profile1D.R
    _Phi = tunneling_result.profile1D.Phi
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
    return TEMP, S3_T, r_c, phi_esc


if __name__ == "__main__":
    import time as _time

    lam = 1e-16
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
    param_set = "set6"

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
        }
    }

    VT = p.finiteTemperaturePotential(param[param_set])
    VT.update_T(T)
    VT.build_fast_thermal(x_max=150.0, n_pts=4096)

    phi = np.linspace(-2 * VT.v, 2 * VT.v, 200).reshape(-1, 1)
    TEMP_PLOT = 0.7
    VT.update_T(TEMP_PLOT)
    TEMP_LIST = np.arange(0.72, 0.85, 0.015)
    TEMP_LIST = np.linspace(4868, 5100, 30)
    # TEMP_LIST = np.linspace(8000, 12_000_000, 5)
    VT.update_T(TEMP_LIST[0])

    # print("v?", VT.v)
    # tv, fv = VT.find_new_minima()
    # print("tv", VT.tv)
    # print("tv?", tv)
    # print("fv?", fv)
    # TEMP_LIST = np.arange(100, 300, 25)
    window = 0.00000002

    TEMP_LIST = np.linspace(4868, 5000, 5)
    TEMP_LIST = np.linspace(6730, 7000, 5)
    phi = np.linspace(-window * VT.v, window * VT.v, 500).reshape(-1, 1)
    # print(-window* VT.v)
    # VT.V_p(phi)
    # plt.plot(phi, VT.V_p(phi))
    # plt.show()
    # print(ho)

    # print(len(phi))
    # print(ho)
    # draw_VT_temp_array(
    #    phi,
    #    VT,
    #    TEMP_LIST,
    #    f"TI_param_{param_set}_lambdaSix_{format_e(lambdaSix)}_{potential_flag}",
    # )
    # print(ho)
    TEMP_LIST = np.linspace(6730, 9000, 30)
    TEMP_LIST = np.linspace(6730, 7000, 30)
    # TEMP_LIST = np.linspace(4868, 5100, 30)
    # TEMP_LIST = np.linspace(4868, 200000, 30)
    # print(ho)

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

    temp_to_idx = {float(T): i for i, T in enumerate(TEMP_LIST)}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_tunneling_worker, a): a[0] for a in worker_args}
        for fut in as_completed(futures):
            TEMP, S3_T, r_c, phi_esc = fut.result()
            idx = temp_to_idx[float(TEMP)]
            S3_T_list[idx] = S3_T
            r_c_list[idx] = r_c
            phi_esc_list[idx] = phi_esc

    _t_end = _time.time()
    print(f"\n--- All tunneling done in {_t_end - _t_start:.1f}s ---")

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
