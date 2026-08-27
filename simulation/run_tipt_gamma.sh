#!/bin/bash
# Lattice TIPT run at a chosen γ → data/lattice/<set>/.
#
# Default: 256³, same physics as the set8 langoff / staged / GW 1024 run
# (V_correct, complex, eta~T, langoff, T_rh=1000, with GWs).
#
# Set naming (see simulation/gamma_sets.py):
#   γ = 4.1667e-4  →  set8
#   γ = 4.1667e-8  →  setA
#   other γ        →  set_gamma_<value>   (never overwrites set8)
#
# Examples
# --------
#   # set8 γ (writes under data/lattice/set8/)
#   bash simulation/run_tipt_gamma.sh
#
#   # new γ → data/lattice/set_gamma_1e-5/...
#   GAMMA=1e-5 bash simulation/run_tipt_gamma.sh
#
#   # paper set A
#   GAMMA=4.1667e-8 bash simulation/run_tipt_gamma.sh
#
#   # dry run (write input.in + print mpirun only)
#   DRY_RUN=1 GAMMA=1e-5 bash simulation/run_tipt_gamma.sh
#
#   # later 1024³ (company Xeon / KISTI — not a laptop)
#   NX=1024 NP=32 GAMMA=1e-5 bash simulation/run_tipt_gamma.sh
#
#   # scan several γ at 256³
#   bash simulation/run_tipt_gamma_scan.sh
#
# Env overrides: GAMMA NX NP T0 TMAX T_RH TIPT_POT PARAM_SET FORCE_PARAM_SET
#                DRY_RUN BUILD SNAP_FMT_OVERRIDE PYTHON

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="${PYTHON:-python3}"

GAMMA="${GAMMA:-4.1667e-4}"
NX="${NX:-256}"
NP="${NP:-8}"
TMAX="${TMAX:-1000}"
T_RH="${T_RH:-1000}"
TIPT_POT="${TIPT_POT:-V_correct}"
DRY_RUN="${DRY_RUN:-0}"
BUILD="${BUILD:-0}"
PARAM_SET="${PARAM_SET:-auto}"
FORCE_PARAM_SET="${FORCE_PARAM_SET:-0}"

case "$TIPT_POT" in
  fermion_only)
    # Only default T0 if the caller did not set it.
    T0="${T0:-1600}"
    POT_ARGS=(--potential_type fermion_only --nb 0 --nf 20)
    ;;
  V_correct|boson_fermion)
    TIPT_POT=V_correct
    T0="${T0:-1230}"
    POT_ARGS=(--potential_type V_correct --nb 20 --nf 20)
    ;;
  *)
    echo "ERROR: TIPT_POT=$TIPT_POT  (fermion_only | V_correct)"
    exit 1
    ;;
esac

if [ "$PARAM_SET" = "auto" ]; then
  PARAM_SET="$("$PY" simulation/gamma_sets.py --gamma "$GAMMA" 2>/dev/null | head -1)"
fi

echo "=== TIPT γ run ==="
echo "  gamma=$GAMMA  param_set=$PARAM_SET  Nx=$NX  np=$NP  T0=$T0  tMax=$TMAX"
echo "  out: data/lattice/${PARAM_SET}/"
"$PY" simulation/gamma_sets.py --gamma "$GAMMA" >/dev/null

EXTRA=()
if [ "$DRY_RUN" = "1" ]; then
  EXTRA+=(--dry_run)
fi
if [ "$BUILD" = "1" ]; then
  EXTRA+=(--install --build)
fi
if [ "$FORCE_PARAM_SET" = "1" ]; then
  EXTRA+=(--force_param_set)
fi

# 256³: raw snapshots are OK; 1024³: keep hdf5 slabs.
SNAP_FMT=hdf5
if [ "$NX" -le 256 ]; then
  SNAP_FMT="${SNAP_FMT_OVERRIDE:-raw}"
fi

"$PY" simulation/run_cosmolattice.py \
  --mpi --np "$NP" \
  --Nx "$NX" \
  --tMax "$TMAX" \
  --dx_phys 1e-3 --dt_phys 1e-4 \
  --T0 "$T0" \
  --gamma "$GAMMA" \
  --param_set "$PARAM_SET" \
  "${POT_ARGS[@]}" \
  --eta_follows_T \
  --n_scalars 2 \
  --with_gws \
  --expansion_mode staged \
  --expansion_f_switch 0.01 \
  --expansion_phi_esc 5e4 \
  --T_rh "$T_RH" \
  --langevin_off_after_nucleation \
  --langevin_off_f_switch 0.01 \
  --langevin_off_phi_esc 5e4 \
  --phi_threshold 50000 \
  --steps 4000 \
  --steps_dense 100 \
  --tOutputFreq 1 \
  --tOutputInfreq 1 \
  --tOutputRareFreq 20 \
  --backup_steps 1000 \
  --snapshot_format "$SNAP_FMT" \
  "${EXTRA[@]}"

echo "done $(date)  set=$PARAM_SET  gamma=$GAMMA"
echo "  → data/lattice/${PARAM_SET}/"
