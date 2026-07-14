// Standalone V' audit: print Vprime(phi,T) for CL thermal_tables.hpp
// Build: g++ -std=c++14 -O2 -I cosmolattice_ext/models \
//        cosmolattice_ext/tests/vprime_audit.cpp -o /tmp/vprime_audit
// Run: /tmp/vprime_audit data/thermal_splines/thermal_tables.bin [include_cw:0|1]

#include "thermal_tables.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::string bin = argc > 1 ? argv[1] : "data/thermal_splines/thermal_tables.bin";
    const bool include_cw = (argc > 2) ? (std::string(argv[2]) != "0") : false;

    ThermalInflation::ThermalTables::Params p;
    const double MPL = 2.4e18;
    const double gamma = 4.1667e-4;
    p.mphi = 1000.0;
    p.lam = p.mphi * p.mphi / ((gamma * MPL) * (gamma * MPL));
    p.mb2_0 = 1e6;
    p.yb = 1.09; p.gb = 1.05;
    p.yf = 1.09; p.gf = 1.05;
    p.nb = 20.0; p.nf = 20.0;
    p.gStarPot = 100.0;
    p.includeCW = include_cw;

    ThermalInflation::ThermalTables tab(bin, p);

    const double Ts[] = {1230.0, 1172.22};
    const double phis[] = {
        0, 100, 200, 300, 400, 470, 500, 600, 700, 800, 900, 1000,
        1200, 1500, 2000, 2646, 5000, 10000, 48000
    };
    for (double T : Ts) {
        std::printf("T=%.4f include_cw=%d\n", T, include_cw ? 1 : 0);
        for (double phi : phis) {
            std::printf("  phi=%12.4f  Vp=% .8e  Vpp=% .8e  ub=% .6f\n",
                        phi, tab.Vprime(phi, T), tab.Vsecond(phi, T), tab.uBoson(phi, T));
        }
    }
    return 0;
}
