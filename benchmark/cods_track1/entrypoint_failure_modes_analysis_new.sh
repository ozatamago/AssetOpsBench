#!/bin/bash
set -euo pipefail

# ---- GLOBAL CONFIG (defaults; can be overridden by env) ----
GENERATE_STEPS_ONLY="${GENERATE_STEPS_ONLY:-True}"   # keep "True" because your CLI currently uses "--generate_steps_only True"
LLM_MODELS="${LLM_MODELS:-6}"   # e.g. "38" or "38,39,40" or "38 39 40"
RUN_TRACK="/home/run_track_1.py"

# ==== PATHS ====
JSONL_FILE="/home/scenarios/all_utterance.jsonl"

# ==== Conda ====
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

: "${GHE_USER:?GHE_USER is not set. Check compose env_file (.env.local).}"
: "${GHE_TOKEN:?GHE_TOKEN is not set. Check compose env_file (.env.local).}"

python -m pip show agent_hive || true
# ===== Patch reactxen files from IBM/ReActXen commit 97606e8 =====
PATCH_COMMIT="97606e87e0ee94e6c23af72fbf55e26246aae200"
PATCH_BASE="https://raw.githubusercontent.com/IBM/ReActXen/${PATCH_COMMIT}/src/reactxen"

# (module_spec | raw_url) のペアを列挙（commit 97606e8 の変更対象2ファイル） :contentReference[oaicite:1]{index=1}
PATCH_TARGETS=(
  "reactxen.utils.model_inference|${PATCH_BASE}/utils/model_inference.py"
  "reactxen.agents.react.agents|${PATCH_BASE}/agents/react/agents.py"
)

for item in "${PATCH_TARGETS[@]}"; do
  MODULE="${item%%|*}"
  PATCH_URL="${item#*|}"

  TARGET_PATH="$(python - <<PY
import importlib.util
spec = importlib.util.find_spec("${MODULE}")
print(spec.origin if spec and spec.origin else "")
PY
)"

  if [ -z "$TARGET_PATH" ]; then
    echo "[patch] ERROR: Could not locate ${MODULE} in this env." >&2
    python - <<PY
import ${MODULE%.*} as pkg
print("module base:", pkg, "file:", getattr(pkg, "__file__", None))
PY
    exit 1
  fi

  echo "[patch] MODULE=${MODULE}"
  echo "[patch] PATCH_URL=${PATCH_URL}"
  echo "[patch] TARGET_PATH=$TARGET_PATH"

  cp -v "$TARGET_PATH" "${TARGET_PATH}.bak.$(date +%Y%m%d%H%M%S)" || true

  python - <<PY "$PATCH_URL" "$TARGET_PATH"
import sys, urllib.request
url, dst = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, dst)
print("[patch] updated:", dst)
PY
done
python -m pip show reactxen || true
python -m pip show fmsr_agent || true
python -m pip show iotagent || true
python -m pip show tsfmagent || true

python -m pip install -qU "psycopg[binary]>=3.1" "ibm-watsonx-ai>=1.4.0"

# ---- Run exactly the specified command ----
# track1_result/trajectory/[ReAct_CD][oracle_verify_recovery]Model_16/Q_10_trajectory.json
  # --trajectory_path "/home/track1_result/trajectory/[ReAct_CD][oracle_verify_recovery]Model_16/Q_1001_trajectory.json" \
  # --trajectory_path "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16/Q_9_trajectory.json" \

python /home/failure_mode_analysis.py \
  --trajectory_path "/home/track1_result/trajectory/[ReAct_CD][oracle_verify_recovery]Model_16/Q_424_trajectory.json" \
  --cache_dir "/home/track1_result/trajefm_outputs_new/fma_cache/[ReAct_CD][oracle_verify_recovery]Model_16" \
  --output_dir "/home/track1_result/trajefm_outputs_new/fma_report/[ReAct_CD][oracle_verify_recovery]Model_16" \
  --llm_model "20" \
  --overwrite_stage_prefixes stage4_

# python /home/failure_mode_analysis.py \
#   --trajectory_dir "/home/track1_result/trajectory/[ReAct_CD][oracle_verify_recovery]Model_16" \
#   --cache_dir "/home/track1_result/trajefm_outputs_new/fma_cache/[ReAct_CD][oracle_verify_recovery]Model_16" \
#   --output_dir "/home/track1_result/trajefm_outputs_new/fma_report/[ReAct_CD][oracle_verify_recovery]Model_16" \
#   --llm_model "20" \
#   # --overwrite_stage_cache \
#   # --overwrite_verification

# python /home/failure_mode_analysis.py \
#   --trajectory_dir "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16" \
#   --cache_dir "/home/track1_result/trajefm_outputs_new/fma_cache/[ReAct_CD][no_verify]Model_16" \
#   --output_dir "/home/track1_result/trajefm_outputs_new/fma_report/[ReAct_CD][no_verify]Model_16" \
#   --llm_model "20" \
#   --overwrite_stage_cache \
#   --overwrite_verification

# /Users/yusuke/Desktop/Program/codabench/AssetOpsBench/benchmark/cods_track1/track1_result/trajectory/[BASE]Model_16
# python /home/failure_mode_analysis.py \
#   --trajectory_dir "/home/track1_result/trajectory/[BASE]Model_16" \
#   --cache_dir "/home/track1_result/trajefm_outputs_new/fma_cache/[BASE]Model_16" \
#   --output_dir "/home/track1_result/trajefm_outputs_new/fma_report/[BASE]Model_16" \
#   --llm_model "20" \
#   --overwrite_stage_cache \
#   --overwrite_verification