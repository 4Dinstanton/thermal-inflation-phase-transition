#!/bin/bash

# Run lattice simulation with optimal threading settings
# Usage: ./run_lattice_fast.sh [options]
#
# Examples:
#   ./run_lattice_fast.sh
#   ./run_lattice_fast.sh --Nx 512 --Ny 512 --T0 8000
#   ./run_lattice_fast.sh --Nt 50000000 --steps 50000
#
# Options (passed through to Python):
#   --Nx INT       Lattice size in x (default: 256)
#   --Ny INT       Lattice size in y (default: 256)
#   --Nt INT       Total timesteps (default: 100000000)
#   --T0 FLOAT     Initial temperature in GeV (default: 7350)
#   --steps INT    Snapshot interval (default: 100000)

# Detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NCORES=$(sysctl -n hw.ncpu)
else
    # Linux
    NCORES=$(nproc)
fi

echo "=========================================="
echo "Setting up optimal Numba threading"
echo "Detected CPU cores: $NCORES"
echo "=========================================="

# Set Numba threading
export NUMBA_NUM_THREADS=$NCORES

# Optional: force TBB threading layer (usually faster)
# export NUMBA_THREADING_LAYER=tbb

# Run simulation, passing all CLI arguments through
python simulation/latticeSimeRescale_numba.py "$@"
