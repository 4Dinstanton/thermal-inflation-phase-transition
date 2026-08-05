// Standalone test for thermal_tables.hpp (no CosmoLattice dependency).
// Build:
//   g++ -std=c++14 -O2 -I cosmolattice_ext/models \
//       cosmolattice_ext/tests/test_thermal_tables.cpp -o /tmp/test_thermal_tables
// Run:
//   /tmp/test_thermal_tables data/thermal_splines/thermal_tables.bin
//
// Prints V, V', V'' at a few (phi, T) points so they can be diffed against the
// Python reference (tools/export_thermal_splines.py : TableEval).

#include "thermal_tables.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::string bin = argc > 1 ? argv[1] : "data/thermal_splines/thermal_tables.bin";
    ThermalInflation::ThermalTables::Params p;
    // Set B (boson+fermion), gamma = 4.1667e-4 -> lambda = mphi^2/(gamma*MPl)^2
    const double MPL = 2.4e18;
    const double gamma = 4.1667e-4;
    p.mphi = 1000.0;
    p.lam = p.mphi * p.mphi / ((gamma * MPL) * (gamma * MPL));
    p.includeCW = true;
    ThermalInflation::ThermalTables tab(bin, p);

    const double phis[] = {5.0e3, 2.0e4, 5.0e4};
    const double Ts[] = {7350.0, 5000.0};
    for (double T : Ts) {
        for (double phi : phis) {
            std::printf("T=%.1f phi=%.4e  V=%.8e  Vp=%.8e  Vpp=%.8e\n",
                        T, phi, tab.V(phi, T), tab.Vprime(phi, T), tab.Vsecond(phi, T));
        }
    }
    return 0;
}
