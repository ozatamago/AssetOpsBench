#!/bin/bash
set -euo pipefail

# ---- Conda ----
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

# ---- Locations inside container ----
TRACK_OUT="/home/track1_result"
TRAJFM_SRC="/home/TrajFM"

# TrajFM options
FM_MODEL_ID="${FM_MODEL_ID:-12}"

# Persist outputs on host via bind mount under /home/track1_result
OUT_ROOT="${OUT_ROOT:-${TRACK_OUT}/trajfm_outputs}"

echo "[trajfm] TRACK_OUT=$TRACK_OUT"
echo "[trajfm] TRAJFM_SRC=$TRAJFM_SRC"
echo "[trajfm] OUT_ROOT=$OUT_ROOT"
echo "[trajfm] FM_MODEL_ID=$FM_MODEL_ID"

# ---- Sanity checks ----
if [ ! -d "$TRAJFM_SRC" ]; then
  echo "[trajfm] ERROR: $TRAJFM_SRC not found."
  exit 1
fi

run_trajfm () {
  local traj_dir="$1"
  local workdir="$2"
  local tag="$3"

  echo "============================================================"
  echo "[trajfm] Running TrajFM for: $tag"
  echo "[trajfm] traj_directory=$traj_dir"
  echo "[trajfm] workdir=$workdir"
  echo "============================================================"

  if [ ! -d "$traj_dir" ]; then
    echo "[trajfm] ERROR: trajectory directory not found: $traj_dir"
    exit 1
  fi

  mkdir -p "$workdir"
  cd "$workdir"

  CMD=(
    python "${TRAJFM_SRC}/failure_mode_extractor.py"
    --traj_directory "$traj_dir"
    --model_id "$FM_MODEL_ID"
    --out_dir "$workdir"
  )

  echo "[trajfm] ${CMD[*]}"
  "${CMD[@]}"
}

# ---- Run BASE and BASE_RAX ----
TAGS=("BASE" "BASE_RAX")
TRAJ_DIRS=(
  "${TRACK_OUT}/trajectory/[BASE]Model_16"
  "${TRACK_OUT}/trajectory/[BASE_RAX]Model_16"
)
WORKDIRS=(
  "${OUT_ROOT}/base_new/work"
  "${OUT_ROOT}/base_rax_new/work"
)

for i in "${!TAGS[@]}"; do
  run_trajfm "${TRAJ_DIRS[$i]}" "${WORKDIRS[$i]}" "${TAGS[$i]}"
done

echo "[trajfm] Done."
echo "[trajfm] BASE outputs:     ${OUT_ROOT}/base_new/work"
echo "[trajfm] BASE_RAX outputs: ${OUT_ROOT}/base_rax_new/work"