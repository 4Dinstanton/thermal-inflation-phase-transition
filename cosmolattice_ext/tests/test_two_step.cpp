// Standalone check of the two_step tree + thermal pieces in thermal_tables.hpp.
// Build:
//   g++ -std=c++14 -O2 -I cosmolattice_ext/models \
//       cosmolattice_ext/tests/test_two_step.cpp -o /tmp/test_two_step
// Run (arguments from potential/twoStepPotential.py, see tests/check_two_step_cpp.py):
//   /tmp/test_two_step <table.bin> V0 v k2 k3 k4 k6 g nb T phi1 [phi2 ...]
//
// Prints phi, V(phi,T)-V(0,T), V'(phi,T), V''(phi,T) per line.

#include "thermal_tables.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 12) {
        std::fprintf(stderr, "usage: %s table V0 v k2 k3 k4 k6 g nb T phi...\n", argv[0]);
        return 1;
    }
    using P = ThermalInflation::ThermalTables::Params;
    P p;
    p.treeType = P::TreeType::TwoStep;
    p.V0 = std::atof(argv[2]);
    p.vVac = std::atof(argv[3]);
    p.k2 = std::atof(argv[4]);
    p.k3 = std::atof(argv[5]);
    p.k4 = std::atof(argv[6]);
    p.k6 = std::atof(argv[7]);
    const double g = std::atof(argv[8]);
    p.nb = std::atof(argv[9]);
    p.nf = 0.0;
    p.yb = std::sqrt(2.0) * g;
    p.gb = 0.0;
    p.mb2_0 = 0.0;
    p.includeCW = false;
    const double T = std::atof(argv[10]);
    ThermalInflation::ThermalTables tab(argv[1], p);
    const double V00 = tab.V(0.0, T);
    for (int i = 11; i < argc; ++i) {
        const double phi = std::atof(argv[i]);
        std::printf("%.17g %.17g %.17g %.17g\n", phi, tab.V(phi, T) - V00,
                    tab.Vprime(phi, T), tab.Vsecond(phi, T));
    }
    return 0;
}
