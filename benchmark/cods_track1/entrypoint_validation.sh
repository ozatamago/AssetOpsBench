#!/bin/bash
set -euo pipefail

# ---- GLOBAL CONFIG (defaults; can be overridden by env) ----
GENERATE_STEPS_ONLY="${GENERATE_STEPS_ONLY:-True}"   # keep "True" because your CLI currently uses "--generate_steps_only True"
LLM_MODEL="${LLM_MODEL:-3}"                         # your --llm_model value
RUN_TRACK="/home/run_track_1.py"

# ==== PATHS ====
JSONL_FILE="/home/scenarios/all_utterance.jsonl"

# ==== Conda ====
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

: "${GHE_USER:?GHE_USER is not set. Check compose env_file (.env.local).}"
: "${GHE_TOKEN:?GHE_TOKEN is not set. Check compose env_file (.env.local).}"

python -m pip show agent_hive || true
# ===== Patch reactxen/utils/model_inference.py from IBM/ReActXen (public GitHub) =====
PATCH_URL="https://raw.githubusercontent.com/IBM/ReActXen/main/src/reactxen/utils/model_inference.py"

TARGET_PATH="$(python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("reactxen.utils.model_inference")
print(spec.origin if spec and spec.origin else "")
PY
)"

if [ -z "$TARGET_PATH" ]; then
  echo "[patch] ERROR: Could not locate reactxen.utils.model_inference in this env." >&2
  python -c "import reactxen; import inspect; print('reactxen:', reactxen, 'file:', getattr(reactxen,'__file__',None))"
  exit 1
fi

echo "[patch] TARGET_PATH=$TARGET_PATH"
cp -v "$TARGET_PATH" "${TARGET_PATH}.bak.$(date +%Y%m%d%H%M%S)" || true

python - <<PY "$PATCH_URL" "$TARGET_PATH"
import sys, urllib.request
url, dst = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, dst)
print("[patch] updated:", dst)
PY
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

echo ">>> Running ALL utterances at once: count=$(echo "$IDS_CSV" | awk -F, '{print NF}') (LLM_MODEL=$LLM_MODEL, GENERATE_STEPS_ONLY=$GENERATE_STEPS_ONLY)"

python "$RUN_TRACK" \
  --utterance_ids "$IDS_CSV" \
  --generate_steps_only "$GENERATE_STEPS_ONLY" \
  --llm_model "$LLM_MODEL"

tail -f /dev/null
