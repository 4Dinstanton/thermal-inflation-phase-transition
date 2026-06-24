import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt

# We need to compute V_eff. We can import it from the potential module
sys.path.insert(0, "potential")
from Potential import finiteTemperaturePotential

M_PL = 2.4e18
gamma = 4.1667e-4  # Set B
g = 1.05
y_c = 1.09
nb = 20
nf = 20
mphi = 1000.0
phi0 = gamma * M_PL
lam = mphi**2 / phi0**2

param = {
    "lambda": lam,
    "mphi": mphi,
    "epsilon": 0,
    "lambdaSix": 0,
    "bosonMassSquared": 1_000_000,
    "bosonCoupling": y_c,
    "bosonGaugeCoupling": g,
    "fermionCoupling": y_c,
    "fermionGaugeCoupling": g,
    "nb": nb,
    "nf": nf,
}
VT = finiteTemperaturePotential(param)
del_V = 1e36 / 4  # Vacuum energy (2.5e35)


def main(sim_dir):
    files = sorted(glob.glob(os.path.join(sim_dir, "field_states", "state_step_*.npz")))

    # physical parameters
    # The directory name has the parameters, e.g. dx_0.001
    dx_phys = 1e-3
    mu = mphi
    dx = mu * dx_phys
    inv_dx2 = 1.0 / (dx**2)

    eps_br_list = []
    temps = []
    steps = []
    rho_lat_list = []
    rho_phi_bg_list = []
    rho_bg_total_list = []

    for f in files:
        d = np.load(f)
        if "pi" not in d.keys():
            continue

        step = int(d["step"])
        T = float(d["temperature"])
        a_current = float(d["scale_factor"])
        inv_a2 = 1.0 / (a_current**2)

        phi = d["phi"].astype(np.float64)
        pi = d["pi"].astype(np.float64)

        # 1. Kinetic energy density
        # pi is rescaled as pi_lat = dot_phi_phys / mu, so dot_phi_phys^2 = mu^2 * pi_lat^2
        E_kin_phys = np.mean(0.5 * pi**2) * mu**2

        # 2. Gradient energy density
        phi_xp = np.roll(phi, -1, axis=0)
        phi_yp = np.roll(phi, -1, axis=1)
        phi_zp = np.roll(phi, -1, axis=2)
        grad_sq = (phi_xp - phi) ** 2 + (phi_yp - phi) ** 2 + (phi_zp - phi) ** 2
        # grad_sq * inv_dx2 is (grad_lat phi_lat)^2. Phys grad = mu * grad_lat
        E_grad_phys = np.mean(0.5 * inv_a2 * grad_sq * inv_dx2) * mu**2

        # 3. Potential energy density
        # Compute V_eff for each site. This might be slow for 256^3.
        # Let's use the fast_thermal method!
        VT.update_T(T)
        VT.build_fast_thermal(x_max=150.0, n_pts=4096)

        phi_flat = phi.flatten()
        chunk_size = 1000000
        V_sum = 0.0
        for i in range(0, len(phi_flat), chunk_size):
            chunk = phi_flat[i : i + chunk_size]
            X = chunk.reshape(-1, 1)
            # Use the fast potential (it automatically uses splines if build_fast_thermal was called)
            V_chunk = VT.V_correct(X) + del_V
            V_sum += np.sum(V_chunk)
        E_pot_phys = V_sum / len(phi_flat)

        rho_lat = E_kin_phys + E_grad_phys + E_pot_phys
        print(E_kin_phys, E_grad_phys, E_pot_phys, rho_lat)

        # 4. Background density
        # The background assumes the field is at the false vacuum (phi = 0)
        rho_phi_bg = VT.V_correct(np.zeros((1, 1)))[0] + del_V
        # print(rho_phi_bg, del_V)

        g_star = 106.75
        rho_rad = (np.pi**2 / 30.0) * g_star * T**4
        # The background energy driving Hubble expansion is radiation + vacuum energy
        rho_bg_total = rho_rad + rho_phi_bg

        eps_br = abs(rho_lat - rho_phi_bg) / del_V

        temps.append(T)
        steps.append(step)
        eps_br_list.append(eps_br)
        rho_lat_list.append(rho_lat)
        rho_phi_bg_list.append(rho_phi_bg)
        rho_bg_total_list.append(rho_bg_total)

        print(
            f"Step {step:5d} (T={T:.1f}): rho_lat = {rho_lat:.4e}, rho_phi_bg = {rho_phi_bg:.4e} => eps_br = {eps_br:.4e}"
        )

    # Save to CSV
    import csv

    csv_path = "figs/backreaction_check.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "temperature", "rho_lat", "rho_phi_bg", "rho_bg_total", "eps_br"]
        )
        for i in range(len(steps)):
            writer.writerow(
                [
                    steps[i],
                    temps[i],
                    rho_lat_list[i],
                    rho_phi_bg_list[i],
                    rho_bg_total_list[i],
                    eps_br_list[i],
                ]
            )
    print(f"Saved {csv_path}")

    plt.figure(figsize=(8, 6))
    plt.plot(temps, eps_br_list, "o-", color="navy", lw=2)
    plt.xlabel("Temperature [GeV]", fontsize=14)
    plt.ylabel(r"$\epsilon_{\rm br}$", fontsize=14)
    plt.title("Backreaction Validity Check", fontsize=16)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figs/backreaction_check.png", dpi=200)
    print("Saved figs/backreaction_check.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python check_backreaction.py <sim_dir>")
