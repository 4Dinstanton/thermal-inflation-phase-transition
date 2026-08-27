#!/bin/bash
# Sequential 256³ γ scan. Each γ lands in its own data/lattice/<set>/.
#
# Default gammas: set8 (4.1667e-4), setA (4.1667e-8), plus 1e-5 and 1e-3.
# Override with:
#   GAMMAS="4.1667e-4 1e-5 1e-3" bash simulation/run_tipt_gamma_scan.sh
#
# Dry-run all (only write input.in):
#   DRY_RUN=1 bash simulation/run_tipt_gamma_scan.sh
#
# Build once then scan:
#   BUILD_ONCE=1 bash simulation/run_tipt_gamma_scan.sh
#
# Later for 1024³ (one γ at a time recommended):
#   NX=1024 NP=32 GAMMAS="1e-5" bash simulation/run_tipt_gamma_scan.sh

set -euo pipefail
cd "$(dirname "$0")/.."

NX="${NX:-256}"
NP="${NP:-8}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_ONCE="${BUILD_ONCE:-0}"
# space-separated γ list
GAMMAS="${GAMMAS:-4.1667e-4 1e-5 1e-3 4.1667e-8}"

if [ "$BUILD_ONCE" = "1" ]; then
  echo "=== build thermal_inflation once ==="
  BUILD=1 DRY_RUN=1 NX=32 NP=1 GAMMA=4.1667e-4 \
    bash simulation/run_tipt_gamma.sh
fi

echo "=== γ scan  Nx=$NX  np=$NP  gammas=[$GAMMAS] ==="
for g in $GAMMAS; do
  echo
  echo "########## GAMMA=$g ##########"
  GAMMA="$g" NX="$NX" NP="$NP" DRY_RUN="$DRY_RUN" BUILD=0 \
    bash simulation/run_tipt_gamma.sh
done

echo
echo "=== scan done ==="
echo "Outputs under data/lattice/{set8,setA,set_gamma_*}/"
python3 simulation/gamma_sets.py --list
