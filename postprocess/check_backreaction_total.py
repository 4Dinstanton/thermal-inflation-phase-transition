import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "potential")
from Potential import finiteTemperaturePotential

M_PL = 2.4e18
gamma = 4.1667e-4
g = 1.05
y_c = 1.09
nb = 20
nf = 20
mphi = 1000.0
phi0 = gamma * M_PL
lam = mphi**2 / phi0**2

param = {
    "lambda": lam, "mphi": mphi, "epsilon": 0, "lambdaSix": 0,
    "bosonMassSquared": 1_000_000, "bosonCoupling": y_c, "bosonGaugeCoupling": g,
    "fermionCoupling": y_c, "fermionGaugeCoupling": g, "nb": nb, "nf": nf,
}
VT = finiteTemperaturePotential(param)
del_V = 1e36 / 4

def main(sim_dir):
    files = sorted(glob.glob(os.path.join(sim_dir, "field_states", "state_step_*.npz")))
    
    mu = mphi
    dx = mu * 1e-3
    inv_dx2 = 1.0 / (dx**2)
    eta_phys = 1230.0

    temps = []
    steps = []
    times = []
    rho_lat_list = []
    E_kin_list = []
    
    # Pass 1: Gather data
    for f in files:
        d = np.load(f)
        if "pi" not in d.keys(): continue

        step = int(d["step"])
        T = float(d["temperature"])
        time = float(d["time"]) / mu
        a_current = float(d["scale_factor"])
        inv_a2 = 1.0 / (a_current**2)

        phi = d["phi"].astype(np.float64)
        pi = d["pi"].astype(np.float64)

        E_kin_phys = np.mean(0.5 * pi**2) * mu**2
        
        phi_xp = np.roll(phi, -1, axis=0)
        phi_yp = np.roll(phi, -1, axis=1)
        phi_zp = np.roll(phi, -1, axis=2)
        grad_sq = (phi_xp - phi) ** 2 + (phi_yp - phi) ** 2 + (phi_zp - phi) ** 2
        E_grad_phys = np.mean(0.5 * inv_a2 * grad_sq * inv_dx2) * mu**2

        VT.update_T(T)
        VT.build_fast_thermal(x_max=150.0, n_pts=4096)
        phi_flat = phi.flatten()
        V_sum = 0.0
        chunk_size = 1000000
        for i in range(0, len(phi_flat), chunk_size):
            chunk = phi_flat[i : i + chunk_size].reshape(-1, 1)
            V_chunk = VT.V_p_correct(chunk) + del_V
            V_sum += np.sum(V_chunk)
        E_pot_phys = V_sum / len(phi_flat)

        rho_lat = E_kin_phys + E_grad_phys + E_pot_phys
        
        temps.append(T)
        steps.append(step)
        times.append(time)
        rho_lat_list.append(rho_lat)
        E_kin_list.append(E_kin_phys)

    temps = np.array(temps)
    times = np.array(times)
    E_kin_list = np.array(E_kin_list)
    rho_lat_list = np.array(rho_lat_list)
    
    # Pass 2: Integrate injected energy
    # d(rho_inj) = 2 * eta * E_kin * dt
    # Actually, injected energy redshifts as a^-4. 
    # d(rho_rad * a^4) = 2 * eta * E_kin * a^4 * dt
    # But since a changes by only 2%, simple integration is a good proxy.
    # Let's do the proper redshift tracking:
    # a(t) = a0 * exp(H t). We know T = T0 / a. So a = T0 / T
    a_vals = temps[0] / temps
    rho_inj = np.zeros_like(E_kin_list)
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        # (rho_inj * a^4)_new = (rho_inj * a^4)_old + 2 * eta * E_kin * a^4 * dt
        a_mid = 0.5 * (a_vals[i] + a_vals[i-1])
        ekin_mid = 0.5 * (E_kin_list[i] + E_kin_list[i-1])
        source = eta_phys * ekin_mid * 2.0
        
        val_old = rho_inj[i-1] * a_vals[i-1]**4
        val_new = val_old + source * a_mid**4 * dt
        rho_inj[i] = val_new / a_vals[i]**4

    eps_br_original = np.abs(rho_lat_list - del_V) / del_V
    eps_br_total = np.abs(rho_lat_list + rho_inj - del_V) / del_V

    for i in range(len(steps)):
        print(f"Step {steps[i]:5d} (T={temps[i]:.1f}): rho_lat={rho_lat_list[i]:.2e}, rho_inj={rho_inj[i]:.2e}, eps_orig={eps_br_original[i]*100:.1f}%, eps_tot={eps_br_total[i]*100:.1f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(temps, eps_br_original * 100, '--', color='red', lw=2, label=r'Field Only ($\rho_\phi^{\rm lat}$)')
    plt.plot(temps, eps_br_total * 100, '-', color='navy', lw=2, label=r'Total Energy ($\rho_\phi^{\rm lat} + \rho_{\rm inj}$)')
    plt.xlabel('Temperature [GeV]', fontsize=14)
    plt.ylabel(r'Backreaction Error $\epsilon_{\rm br}$ [%]', fontsize=14)
    plt.title('Validity Check of Prescribed Background', fontsize=16)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('figs/backreaction_check_total.png', dpi=200)
    print("Saved figs/backreaction_check_total.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
