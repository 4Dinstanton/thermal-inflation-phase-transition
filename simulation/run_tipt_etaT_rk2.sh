#!/bin/bash
# TIPT CosmoLattice v1: eta ~ T, complex scalar (U(1) strings),
# fused / fdt / nonfused RK2.
#
# ou is v2-only; these jobs stay on v1.
#   rk2_fused    4-pass RK2, same z both half-steps (Numba rk2_fused, full FDT)
#   fdt          independent-z fused RK2, half-kicks *sqrt(2)
#   nonfused_rk2 2-pass RK2, one full-dt kick of sigma (Numba rk2_nonfused)
#
# --n_scalars 2 for global U(1) strings. v1 has no in-simulation winding;
# export snapshots then tools/compute_strings_cl.py (or revisualize --strings).
#
# --eta_follows_T => eta_phys(t)=T(t) because eta_phys defaults to T0.
# tMax is sized for a ~100 GeV bath drop (T=T0/a, H_TI ≈ 0.120 GeV):
#   T0=1600 -> tMax 540;  T0=1230 -> tMax 710.
#
#   bash simulation/run_tipt_etaT_rk2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
COMMON=(
  --eta_follows_T
  --n_scalars 2
  --with_gws
  --Nx 256
  --steps 4000
  --phi_threshold 50000
  --steps_dense 20
  --param_set set8
  --mpi --np 8
)

echo "=== rebuild thermal_inflation (v1; needed for eta~T + nonfused_rk2) ==="
"$PY" simulation/run_cosmolattice.py --install --build --mpi \
  --Nx 32 --tMax 1 --no_snapshots --dry_run

run_one() {
  local scheme="$1"
  shift
  echo "=== $scheme  $* ==="
  "$PY" simulation/run_cosmolattice.py \
    --stochastic_scheme "$scheme" \
    "${COMMON[@]}" \
    "$@"
}

# T=1600, fermions only (nf=20, nb=0); drop ~100 GeV
run_one rk2_fused    --T0 1600 --tMax 540 --potential_type fermion_only --nb 0 --nf 20
run_one fdt          --T0 1600 --tMax 540 --potential_type fermion_only --nb 0 --nf 20
run_one nonfused_rk2 --T0 1600 --tMax 540 --potential_type fermion_only --nb 0 --nf 20

# T=1230, bosons+fermions (nb=nf=20); drop ~100 GeV
run_one rk2_fused    --T0 1230 --tMax 710 --potential_type V_correct --nb 20 --nf 20
run_one fdt          --T0 1230 --tMax 710 --potential_type V_correct --nb 20 --nf 20
run_one nonfused_rk2 --T0 1230 --tMax 710 --potential_type V_correct --nb 20 --nf 20
