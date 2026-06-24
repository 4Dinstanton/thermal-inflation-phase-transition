import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt

def main(sim_dir):
    files = sorted(glob.glob(os.path.join(sim_dir, "field_states", "state_step_*.npz")))
    
    mu = 1000.0
    temps = []
    steps = []
    times = []
    
    for f in files:
        d = np.load(f)
        if "pi" not in d.keys(): continue

        step = int(d["step"])
        T = float(d["temperature"])
        time = float(d["time"]) / mu
        
        temps.append(T)
        steps.append(step)
        times.append(time)

    temps = np.array(temps)
    times = np.array(times)
    
    # Calculate Hubble time
    M_PL = 2.4e18
    del_V = 2.5e35
    H = np.sqrt(del_V / (3.0 * M_PL**2))
    t_H = 1.0 / H

    # Calculate fraction of Hubble time
    delta_t = times - times[0]
    hubble_fraction = delta_t / t_H * 100.0 # in percent
    
    # Calculate scale factor change
    a_change = (np.exp(H * delta_t) - 1.0) * 100.0 # in percent

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(temps, hubble_fraction, '-', color='navy', lw=3)
    plt.xlabel('Temperature [GeV]', fontsize=14)
    plt.ylabel(r'$\Delta t / H^{-1}$ [%]', fontsize=14)
    plt.title('Transition Time vs Hubble Time', fontsize=14)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(temps, a_change, '-', color='darkgreen', lw=3)
    plt.xlabel('Temperature [GeV]', fontsize=14)
    plt.ylabel(r'$\Delta a / a$ [%]', fontsize=14)
    plt.title('Scale Factor Change', fontsize=14)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/validity_alternative.png', dpi=200)
    print(f"H = {H:.4f} GeV, t_H = {t_H:.2f} GeV^-1")
    print(f"Max transition time = {delta_t[-1]:.4f} GeV^-1")
    print("Saved figs/validity_alternative.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
