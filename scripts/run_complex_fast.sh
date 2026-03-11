#!/bin/bash

# Run complex scalar field lattice simulation with optimal threading settings
# Usage: ./run_complex_fast.sh [options]
#
# Examples:
#   ./run_complex_fast.sh
#   ./run_complex_fast.sh --Nx 128 --Ny 128 --T0 7405
#   ./run_complex_fast.sh --zn_order 6 --zn_strength 1e20 --zn_turn_on_T 7300
#   ./run_complex_fast.sh --integrator rk2_fused_inline --init_rho 0.01
#
# New options for complex field (in addition to all original options):
#   --zn_order INT       Z_N symmetry breaking order (0 = pure U(1))
#   --zn_strength FLOAT  Strength of cos(N*theta) breaking term (GeV^4)
#   --zn_turn_on_T FLOAT Temperature below which Z_N activates (0 = always on)
#   --init_rho FLOAT     Initial radial fluctuation amplitude (default: 0.01)

# Detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=$(nproc)
fi

echo "=========================================="
echo "Complex Scalar Field Lattice Simulation"
echo "Setting up optimal Numba threading"
echo "Detected CPU cores: $NCORES"
echo "=========================================="

export NUMBA_NUM_THREADS=$NCORES

python simulation/latticeSimComplex_numba.py "$@"
