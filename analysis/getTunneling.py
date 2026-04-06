import sympy as sp
import numpy as np
import scipy
from scipy import integrate, interpolate

import matplotlib.pyplot as plt
import cosmoTransitions.finiteT as CTFT
import cosmoTransitions.pathDeformation as CTPD
import math

import sys as _sys, os as _os

_sys.path.insert(
    0,
    _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "potential"
    ),
)

import Potential as p
from tunneling_utils import fullTunneling


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
    # l, = ax.plot(phi, VT.V_tree(phi), label='T=0', color='blue')
    # ax1 = ax.twinx()
    leg = []
    for T in temp_arr:
        T = round(T, 3)
        VT.update_T(T)
        (ll,) = ax1.plot(phi / 1000, VT.V_p(phi), label=f"T={T}")
        # ax1.set_ylim([min(VT.V(phi))-0.1, (VT.V(np.array([0]).reshape(-1,1))) + 0.1])
        ax1.set_ylabel(r"$V_T$")
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
    ax1.set_ylabel(r"$V_T(\phi, T)$")
    plt.legend()
    plt.xlabel(r"$\phi (\text{TeV})$")
    # ax1.set_xlim([-4,4])
    # ax1.set_ylim([-5,-0])
    plt.savefig(f"figs/finiteTemp/{title}", dpi=300)
    plt.show()


def format_e(n):
    a = "%E" % n
    return a.split("E")[0].rstrip("0").rstrip(".") + "E" + a.split("E")[1]


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
    }
}


VT = p.finiteTemperaturePotential(param[param_set])
T = 7450
VT.update_T(T)

COUPLING_LIST = np.arange(1.07, 1.40, 0.005)


epsil_arr = np.arange(10, 1.7, -0.01)
S3_T_list = []
S4_list = []
r_c_list = []  # Critical bubble radius
phi_esc_list = []  # Escape point (field value at bubble center)
print(COUPLING_LIST)
for COUP in COUPLING_LIST:
    param[param_set]["bosonCoupling"] = COUP
    param[param_set]["fermionCoupling"] = COUP
    # TEMP = 12_000_000
    print(COUP)
    VT.update_params(param[param_set])
    # window = 0.0000000025
    # phi = np.linspace(0, window * VT.v, 500).reshape(-1, 1)
    # print(-window* VT.v)
    # VT.V_p(phi)
    # plt.plot(phi, VT.V_p(phi))
    # plt.show()
    tv, fv = VT.find_new_minima()
    if tv < 0.01:
        tv = prev_tv
    # print("V diff?", VT.V(np.array([fv]).reshape(-1,1)) - VT.V(np.array([tv]).reshape(-1,1)))
    if epsil == 0:
        fv = 0
    tv = max(10.0 * T, 20000.0)

    V_tv = VT.V(np.array([tv]).reshape(-1, 1))
    V_fv = VT.V(np.array([fv]).reshape(-1, 1))
    print("True vacuum : {:.4f}, False vacuum : {:.4f}".format(tv, fv))
    print(
        "V_fv : {:.4f}, V_tv : {:.4f}, V_fv - V_tv : {:.4f}".format(
            V_tv[0], V_fv[0], V_fv[0] - V_tv[0]
        )
    )
    tunneling_result = fullTunneling(
        path_pts=np.array(
            [
                [tv],
                [fv],
            ]
        ),
        V=VT.V,
        dV=VT.dV_p,
        maxiter=1,
        V_spline_samples=200,
        tunneling_init_params=dict(alpha=2),
        tunneling_findProfile_params=dict(
            xtol=0.00001, phitol=0.00001, rmin=0.00001, npoints=200
        ),
        deformation_class=CTPD.Deformation_Spline,
        extend_to_minima=False,
    )
    print(tunneling_result.profile1D.Phi[0])
    print(tunneling_result.profile1D.Phi[-1])
    print(tunneling_result.action / T)
    S3_T_list.append(tunneling_result.action / T)
    S3 = tunneling_result.action

    # Critical bubble radius: where φ crosses midpoint between center and boundary
    _R = tunneling_result.profile1D.R
    _Phi = tunneling_result.profile1D.Phi
    _phi_mid = 0.5 * (_Phi[0] + _Phi[-1])
    r_c = (
        np.interp(_phi_mid, _Phi[::-1], _R[::-1])
        if _Phi[0] > _Phi[-1]
        else np.interp(_phi_mid, _Phi, _R)
    )
    phi_esc = _Phi[0]  # Field value at bubble center
    r_c_list.append(r_c)
    phi_esc_list.append(phi_esc)
    print(f"r_c: {r_c:.4e}, phi_esc: {phi_esc:.4e}")
    """
    tunneling_result = CTPD.fullTunneling(
        path_pts=np.array(
            [
                [tv],
                [fv],
            ]
        ),
        V=VT.V,
        dV=VT.dV_p,
        maxiter=1,
        V_spline_samples=200,
        tunneling_init_params=dict(alpha=3),
        tunneling_findProfile_params=dict(
            xtol=0.00001, phitol=0.00001, rmin=0.00001, npoints=300),
        deformation_class=CTPD.Deformation_Spline,
    )
    S4_list.append(tunneling_result.action)
    """
    print(f"S_3 : {S3}, S_3/T : {S3_T_list[-1]}")

    prev_tv = tv
    # plt.plot(epsil_arr, B_arr)
    # plt.xlabel(r"$\epsilon$")
    # plt.ylabel(r"$\tilde{B}$")
    # plt.title(r"$\epsilon - \tilde{B}$")
    # plt.savefig("figs/zeroTemp/e-B", dpi=300)
    # plt.show()


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
df["lambda"] = COUPLING_LIST
# df["S4"] = S4_list
df["S3/T"] = S3_T_list
df["r_c"] = r_c_list  # Critical bubble radius
df["phi_esc"] = phi_esc_list  # Escape point (field at bubble center)
import os
import json

csv_path = f"data/tunneling/{param_set}"
os.makedirs(csv_path, exist_ok=True)
df.to_csv(f"{csv_path}/T-S_param_{param_set}_coupling_scan.csv")
json.dump(param[param_set], open(f"{csv_path}/parameters.json", "w"))
plt.show()
print(S3_T_list)
print(S4_list)
