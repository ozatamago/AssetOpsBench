#!/bin/bash
set -euo pipefail

# ---- GLOBAL CONFIG (defaults; can be overridden by env) ----
GENERATE_STEPS_ONLY="${GENERATE_STEPS_ONLY:-True}"   # keep "True" because your CLI currently uses "--generate_steps_only True"
LLM_MODEL="${LLM_MODEL:-16}"                         # your --llm_model value
RUN_TRACK="/home/run_track_1.py"

# ==== PATHS ====
JSONL_FILE="/home/scenarios/all_utterance.jsonl"

# ==== Conda ====
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

export GHE_USER="willysk"
export GHE_TOKEN="REDACTED"

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

# run them sequentially (no batching, no skipping)
while IFS= read -r ID; do
  [ -n "$ID" ] || continue
  echo ">>> Running utterance_id=$ID (LLM_MODEL=$LLM_MODEL, GENERATE_STEPS_ONLY=$GENERATE_STEPS_ONLY)"
  if ! python "$RUN_TRACK" \
      --utterance_ids "$ID" \
      --generate_steps_only "$GENERATE_STEPS_ONLY" \
      --llm_model "$LLM_MODEL"
  then
    echo "!!!! run_track_1_validation.py failed for id=$ID" >&2
  fi
done < <(emit_ids)


tail -f /dev/null



# #!/bin/bash
# # Activate conda env
# source /opt/conda/etc/profile.d/conda.sh
# conda activate assetopsbench

# export GHE_USER="willysk"
# export GHE_TOKEN="REDACTED"


# which python
# python --version
# python -m pip show agent_hive
# # ===== Patch reactxen/utils/model_inference.py from IBM/ReActXen (public GitHub) =====
# PATCH_URL="https://raw.githubusercontent.com/IBM/ReActXen/main/src/reactxen/utils/model_inference.py"

# TARGET_PATH="$(python - <<'PY'
# import importlib.util
# spec = importlib.util.find_spec("reactxen.utils.model_inference")
# print(spec.origin if spec and spec.origin else "")
# PY
# )"

# if [ -z "$TARGET_PATH" ]; then
#   echo "[patch] ERROR: Could not locate reactxen.utils.model_inference in this env." >&2
#   python -c "import reactxen; import inspect; print('reactxen:', reactxen, 'file:', getattr(reactxen,'__file__',None))"
#   exit 1
# fi

# echo "[patch] TARGET_PATH=$TARGET_PATH"
# cp -v "$TARGET_PATH" "${TARGET_PATH}.bak.$(date +%Y%m%d%H%M%S)" || true

# python - <<PY "$PATCH_URL" "$TARGET_PATH"
# import sys, urllib.request
# url, dst = sys.argv[1], sys.argv[2]
# urllib.request.urlretrieve(url, dst)
# print("[patch] updated:", dst)
# PY

# python -c "import reactxen.utils.model_inference as m; print('[patch] import OK:', m.__file__)"
# python -m pip show reactxen
# python -m pip show fmsr_agent
# python -m pip show iotagent
# python -m pip show tsfmagent

# # Run the entire thing
# # python /home/run_track_1.py --utterance_ids "1,2,3,4,5,6,7,8,9,10,11,12,41,42,43,44,45,46,47,48" --generate_steps_only True
# # python /home/run_track_1.py \
# #   --utterance_ids "1,2,3,4,5,6,7,8,9,10,11,12,41,42,43,44,45,46,47,48" \
# #   --generate_steps_only True
# python /home/run_track_1.py --utterance_ids 1 --generate_steps_only True 

# # Keep the container alive
# tail -f /dev/null




