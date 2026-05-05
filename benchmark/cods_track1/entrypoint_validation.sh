#!/bin/bash
set -euo pipefail

# ---- GLOBAL CONFIG (defaults; can be overridden by env) ----
GENERATE_STEPS_ONLY="${GENERATE_STEPS_ONLY:-False}"   # keep "True" because your CLI currently uses "--generate_steps_only True"
LLM_MODELS="${LLM_MODELS:-16, 20, 22, 23, 38}"   # e.g. "38" or "38,39,40" or "38 39 40"
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

python /home/run_track_1.py --utterance_ids "1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,4001,4002,4003,4004,4005,4006,4007,4008,4009,40010,40011,40012,40013,40014,40015,40016,40017,40018,40019,40020,40021,40022,40023,40024,40025,40026,40027,40028,40029,40030,40031,40032,40033,40034,40035,40036,40037,40038,40039,40040,40041,40042,40043,40044,40045,40046,40047,40048,40049,40050,40051,40052,40053,40054,40055,40056,40057,40058,40059,40060,40061,40062,40063,40064,40065,40066,40067,40068,40069,40070,40071,40072,40073,40074,40075,40076,40077,40078,40079,40080,40081,40082,40083,40084,40085,40086,40087,40088,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2101,2102,2103,2104,2105,2106,2107" --generate_steps_only "True"


tail -f /dev/null
