#!/bin/bash
set -euo pipefail

# ---- GLOBAL CONFIG (defaults; can be overridden by env) ----
GENERATE_STEPS_ONLY="${GENERATE_STEPS_ONLY:-False}"   # keep "True" because your CLI currently uses "--generate_steps_only True"
LLM_MODELS="${LLM_MODELS:-20, 22, 23, 38}"   # e.g. "38" or "38,39,40" or "38 39 40"
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

emit_ids() {
  if command -v jq >/dev/null 2>&1; then
    jq -r 'select(.id != null) | .id' "$JSONL_FILE"
  else
    python - <<'PY' "$JSONL_FILE"
import json, sys
p = sys.argv[1]
with open(p,'r',encoding='utf-8') as f:
  for line in f:
    line=line.strip()
    if not line: continue
    try: obj=json.loads(line)
    except Exception: continue
    i=obj.get("id")
    if i is not None: print(i)
PY
  fi
}

# list up IDs
echo "Scenario IDs:"
emit_ids

# ---- run all utterances at once (CSV string) ----
IDS_CSV="$(emit_ids | paste -sd, -)"

if [ -z "$IDS_CSV" ]; then
  echo "No scenario IDs found in $JSONL_FILE" >&2
  exit 1
fi

# split into array (support comma-separated; also works for a single value)
MODELS_STR="${LLM_MODELS//,/ }"
read -r -a LLM_MODEL_LIST <<< "$MODELS_STR"

# utterance count (same as before)
UTT_COUNT="$(echo "$IDS_CSV" | awk -F, '{print NF}')"

for LLM_MODEL in "${LLM_MODEL_LIST[@]}"; do
  echo ">>> Running ALL utterances at once: count=$UTT_COUNT (LLM_MODEL=$LLM_MODEL, GENERATE_STEPS_ONLY=$GENERATE_STEPS_ONLY)"

  python "$RUN_TRACK" \
    --utterance_ids "$IDS_CSV" \
    --generate_steps_only "$GENERATE_STEPS_ONLY" \
    --llm_model "$LLM_MODEL"
done

tail -f /dev/null
