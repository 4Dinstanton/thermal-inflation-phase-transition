#!/bin/bash
#
# Cosmic-string analysis from field_snapshot.h5 on a fat Xeon node
# (Intel Xeon Gold 6126, 48 cores, ~376 GiB RAM) — no KNL / no SRU required.
#
# Run interactively (login / analysis node), e.g.:
#
#   ssh <kisti-host>
#   cd /scratch/a2136a01/PhaseTransition
#   bash simulation/run_strings_hdf5_xeon.sh \
#       data/lattice/set8/<1024_run_dir>
#
# Or with env vars:
#
#   RUN_DIR=data/lattice/set8/<run> WORKERS=12 \
#     bash simulation/run_strings_hdf5_xeon.sh
#
# Optional PBS (if you regain SRU on SKL queue):
#   #PBS -N strings-hdf5-xeon
#   #PBS -q norm_skl
#   #PBS -A inhouse
#   #PBS -l select=1:ncpus=48:mpiprocs=1:ompthreads=1
#   #PBS -l walltime=48:00:00
#   #PBS -j oe
#   qsub -v RUN_DIR=data/lattice/set8/<run> simulation/run_strings_hdf5_xeon.sh
#
# Outputs under RUN_DIR:
#   strings/string_summary.csv
#   strings/strings_step_*.png
#   strings3d/strings3d_step_*.png
#
# RAM guide (1024³, ~20 GB / worker):
#   WORKERS=12  → ~240 GB   (recommended default on 376 GiB)
#   WORKERS=16  → ~320 GB   (aggressive)
#   WORKERS=1–4 → safest if OOM recurs
#
set -euo pipefail

REPO="${REPO:-/scratch/a2136a01/PhaseTransition}"

# First CLI arg overrides RUN_DIR if set
if [ "${1:-}" != "" ] && [ "${1#-}" = "$1" ]; then
  RUN_DIR="$1"
  shift
fi
RUN_DIR="${RUN_DIR:-}"
# Optional env: STEP_MIN overrides pipeline default (analyze only)
STEP_MIN="${STEP_MIN:-}"   # empty → pipeline uses ANALYZE_STEP_MIN_DEFAULT (3500)
STEP_MAX="${STEP_MAX:-}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"       # 0 = write PNGs
METRICS_ONLY="${METRICS_ONLY:-0}"   # 1 = CSV only
INSTALL_DEPS="${INSTALL_DEPS:-1}"   # 1 = pip install missing packages into venv

# Default workers for 376 GiB Xeon: leave ~100 GB free for OS / HDF5 cache
NCPU=$( (command -v nproc >/dev/null && nproc) || echo 48 )
if [ -z "${WORKERS:-}" ]; then
  if [ -n "${PBS_NCPUS:-}" ]; then
    WORKERS="$PBS_NCPUS"
  else
    WORKERS=12
  fi
fi
# Cap at available cores
if [ "$WORKERS" -gt "$NCPU" ]; then
  WORKERS="$NCPU"
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LIVE_DIR="${STRINGS_LIVE_DIR:-/scratch/a2136a01/strings_live}"
mkdir -p "$LIVE_DIR" "$REPO/kisti_log"
JOBTAG="${PBS_JOBID:-$(hostname)-$$}"
export STRINGS_LIVE_LOG="$LIVE_DIR/${JOBTAG}.log"
ln -sfn "$STRINGS_LIVE_LOG" "$REPO/kisti_log/latest_strings.log"

# Tee to live log (works interactively and under PBS)
exec > >(tee -a "$STRINGS_LIVE_LOG") 2>&1

echo "=== strings HDF5 (Xeon / 376 GiB) ==="
echo "  live log: $STRINGS_LIVE_LOG"
echo "  host=$(hostname)  date=$(date)"
echo "  ncpu=$NCPU  WORKERS=$WORKERS  REPO=$REPO"

# --- Modules (Nurion SKL-style; soft-fail on non-module hosts) ---
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  # Prefer Skylake; fall back if craype name differs
  module load craype-x86-skylake 2>/dev/null \
    || module load craype-x86-rome 2>/dev/null \
    || true
  module load gcc/8.3.0 2>/dev/null || module load gcc 2>/dev/null || true
  module load python/3.9.5 2>/dev/null \
    || module load python/3.9 2>/dev/null \
    || module load python/3.8 2>/dev/null \
    || module load python 2>/dev/null || true
  # Serial HDF5 is enough for h5py reads (no MPI needed for analyze)
  module load hdf5/1.10.2 2>/dev/null \
    || module load hdf5-parallel/1.10.2 2>/dev/null \
    || module load hdf5 2>/dev/null || true
  module list 2>&1 || true
else
  echo "NOTE: no 'module' command — using system Python/HDF5"
fi

cd "$REPO" || { echo "ERROR: no repo at $REPO"; exit 1; }

if [ -z "$RUN_DIR" ]; then
  echo "ERROR: set RUN_DIR"
  echo "  bash simulation/run_strings_hdf5_xeon.sh data/lattice/set8/<run_dir>"
  echo "  RUN_DIR=data/lattice/set8/<run> bash simulation/run_strings_hdf5_xeon.sh"
  exit 1
fi

if [[ "$RUN_DIR" = /* ]]; then
  RUN_ABS="$RUN_DIR"
else
  RUN_ABS="$REPO/$RUN_DIR"
fi

if [ ! -d "$RUN_ABS" ]; then
  echo "ERROR: run directory not found: $RUN_ABS"
  exit 1
fi

# --- Python venv (user-writable; no root / no SRU) ---
VENV_DIR="${VENV_DIR:-$REPO/.venv_strings}"
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Load a python module first."
  exit 1
fi

echo "Python: $($PY --version 2>&1)  ($($PY -c 'import sys; print(sys.executable)'))"

if [ "$INSTALL_DEPS" = "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv: $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY=python
  pip install --upgrade pip setuptools wheel >/dev/null
  # Prefer binary wheels (much faster than building h5py against system HDF5)
  REQ="$REPO/requirements.txt"
  if [ -f "$REQ" ]; then
    echo "Installing from $REQ ..."
    pip install -r "$REQ"
  else
    echo "Ensuring numpy h5py matplotlib scipy (no requirements.txt) ..."
    pip install --upgrade "numpy" "h5py" "matplotlib" "scipy" "pillow"
  fi
else
  if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PY=python
  fi
fi

echo "Checking imports..."
"$PY" - <<'PY'
import numpy, h5py, matplotlib, scipy
print(f"  numpy={numpy.__version__}  h5py={h5py.__version__}"
      f"  matplotlib={matplotlib.__version__}  scipy={scipy.__version__}")
print(f"  h5py HDF5 lib = {h5py.version.hdf5_version}")
PY

# --- Input checks ---
MANIFEST="$RUN_ABS/field_states/manifest.csv"
H5_FOUND=""
for h in \
  "$RUN_ABS/field_snapshot.h5" \
  "$RUN_ABS/field_states/field_snapshot.h5"
do
  if [ -f "$h" ]; then
    H5_FOUND="$h"
    break
  fi
done

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: missing $MANIFEST"
  exit 1
fi
if [ -z "$H5_FOUND" ]; then
  echo "ERROR: no field_snapshot.h5 under $RUN_ABS"
  echo "  looked for: $RUN_ABS/field_snapshot.h5"
  echo "              $RUN_ABS/field_states/field_snapshot.h5"
  exit 1
fi

N_MANIFEST=$(grep -cve '^\s*$' "$MANIFEST" 2>/dev/null || echo 0)
echo "manifest rows (approx): $N_MANIFEST"
echo "HDF5: $H5_FOUND"
ls -lh "$H5_FOUND"

MEM_GB=""
if [ -r /proc/meminfo ]; then
  MEM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  MEM_GB=$(( MEM_KB / 1024 / 1024 ))
  echo "host RAM: ${MEM_GB} GiB"
fi

if echo "$RUN_ABS" | grep -q '1024x1024x1024'; then
  EST=$(( WORKERS * 20 ))
  echo "1024³ RAM estimate: WORKERS=$WORKERS × ~20 GB ≈ ${EST} GB"
  if [ -n "$MEM_GB" ] && [ "$EST" -gt $(( MEM_GB * 9 / 10 )) ]; then
    SAFE=$(( MEM_GB / 25 ))
    [ "$SAFE" -lt 1 ] && SAFE=1
    echo "WARNING: estimated peak exceeds ~90% of RAM."
    echo "         Auto-capping WORKERS $WORKERS → $SAFE"
    WORKERS="$SAFE"
  fi
fi

PIPE_ARGS=(
  tools/cl_hdf5_string_pipeline.py analyze "$RUN_ABS"
  --workers "$WORKERS"
)

if [ -n "$STEP_MIN" ]; then
  PIPE_ARGS+=(--step-min "$STEP_MIN")
fi
if [ -n "$STEP_MAX" ]; then
  PIPE_ARGS+=(--step-max "$STEP_MAX")
fi
if [ "$SKIP_PLOTS" = "1" ]; then
  PIPE_ARGS+=(--skip-plots)
fi
if [ "$METRICS_ONLY" = "1" ]; then
  PIPE_ARGS+=(--metrics-only)
fi

echo "=== run: $PY ${PIPE_ARGS[*]} ==="
"$PY" "${PIPE_ARGS[@]}"

echo "=== outputs ==="
SUMMARY="$RUN_ABS/strings/string_summary.csv"
if [ -f "$SUMMARY" ]; then
  echo "string_summary.csv:"
  wc -l "$SUMMARY"
  head -n 3 "$SUMMARY"
fi
PNG2D=$(find "$RUN_ABS/strings" -maxdepth 1 -name 'strings_step_*.png' 2>/dev/null | wc -l || echo 0)
PNG3D=$(find "$RUN_ABS/strings3d" -maxdepth 1 -name 'strings3d_step_*.png' 2>/dev/null | wc -l || echo 0)
echo "strings PNGs: $PNG2D (2D)  $PNG3D (3D)"
echo "done $(date)"
