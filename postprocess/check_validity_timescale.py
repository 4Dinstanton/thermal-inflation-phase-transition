import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt

def main(sim_dir):
    files = sorted(glob.glob(os.path.join(sim_dir, "field_states", "state_step_*.npz")))
    
    mu = 1000.0
    temps = []
    steps = []
    times = []
    fracs = []
    
    for f in files:
        d = np.load(f)
        if "pi" not in d.keys(): continue

        step = int(d["step"])
        T = float(d["temperature"])
        time = float(d["time"]) / mu
        phi = d["phi"]
        
        frac_false = np.mean(np.abs(phi) <= 10000.0)
        
        temps.append(T)
        steps.append(step)
        times.append(time)
        fracs.append(frac_false)

    temps = np.array(temps)
    times = np.array(times)
    fracs = np.array(fracs)
    
    # Calculate Hubble time
    M_PL = 2.4e18
    del_V = 2.5e35
    H = np.sqrt(del_V / (3.0 * M_PL**2))
    t_H = 1.0 / H

    a_change = (np.exp(H * times) - 1.0) * 100.0 # relative to start

    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'darkgreen'
    ax1.set_xlabel('Temperature [GeV]', fontsize=14)
    ax1.set_ylabel('False Vacuum Fraction', color=color, fontsize=14)
    ax1.plot(temps, fracs, '-', color=color, lw=3)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()  
    color = 'navy'
    ax2.set_ylabel(r'$\Delta a / a$ [%] (Scale Factor Growth)', color=color, fontsize=14)  
    ax2.plot(temps, a_change - a_change[1], '--', color=color, lw=3)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Transition Duration vs Cosmological Expansion', fontsize=16)
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/validity_timescale.png', dpi=200)
    print("Saved figs/validity_timescale.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
