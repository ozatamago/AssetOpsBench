#!/bin/bash
set -euo pipefail

# ==== PATHS ====
JSONL_FILE="/home/scenarios/all_utterance.jsonl"   # もう使っていないが、残しておいてもOK

RESULT_DIR="/home/track1_result/"
PLAN_DIR="${RESULT_DIR}plan/"
TRAJECTORY_DIR="${RESULT_DIR}trajectory/"
mkdir -p "$PLAN_DIR" "$TRAJECTORY_DIR"

# ==== Conda ====
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

python -m pip show agent_hive || true
python -m pip show reactxen || true
python -m pip show fmsr_agent || true
python -m pip show iotagent || true
python -m pip show tsfmagent || true

python -m pip install -qU "psycopg[binary]>=3.1" "ibm-watsonx-ai>=1.4.0"

echo "========== Running utterance_ids 1 =========="
# echo "========== Running utterance_ids 1–12 and 41-48 =========="
# python /home/run_track_1.py --utterance_ids "1,2,3,4,5,6,7,8,9,10,11,12,41,42,43,44,45,46,47,48"
python /home/run_track_1.py --utterance_ids 8

echo "All runs finished."
tail -f /dev/null


#!/bin/bash
# Activate conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

which python
python --version
python -m pip show agent_hive
python -m pip show reactxen
python -m pip show fmsr_agent
python -m pip show iotagent
python -m pip show tsfmagent

python -m pip install -qU "psycopg[binary]>=3.1"

# python /home/auto_scoring.py
# python /home/run_track_1.py --utterance_ids "2"
python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/simulator_agent.py
# python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/critic_agent.py
# python /home/summarize.py Q_4
# python /home/summarize.py Q_6
# python /home/summarize.py Q_42
# python /home/summarize.py Q_101
# python /home/summarize.py Q_102
# python /home/summarize.py Q_201
# python /home/summarize.py Q_218
# python /home/summarize.py Q_219
# python /home/summarize.py Q_401
# python /home/summarize.py Q_410

# Run the entire thing
# 1 - 141
# python /home/run_track_1.py --utterance_ids 1

# Keep the container alive
tail -f /dev/null


#!/bin/bash
set -euo pipefail

# ==== PATHS ====
JSONL_FILE="/home/scenarios/all_utterance.jsonl"

RESULT_DIR="/home/track1_result/"
PLAN_DIR="${RESULT_DIR}plan/"
TRAJECTORY_DIR="${RESULT_DIR}trajectory/"
mkdir -p "$PLAN_DIR" "$TRAJECTORY_DIR"

BATCH_SIZE=10

# ==== Conda ====
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

python -m pip show agent_hive || true
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

extract_num() { grep -oE '[0-9]+' <<<"$1" | head -1 || true; }
has_trajectory() { [[ -n "$1" ]] && [[ -f "${TRAJECTORY_DIR}/Q_${1}_trajectory.json" ]]; }

PENDING_IDS=()
while IFS= read -r ID; do
  [[ -n "$ID" ]] || continue
  NUM="$(extract_num "$ID")"
  if has_trajectory "$NUM"; then
    echo ">>> Skip (exists): ${TRAJECTORY_DIR}/Q_${NUM}_trajectory.json [id=$ID]"
    continue
  fi
  PENDING_IDS+=("$ID")
done < <(emit_ids)

TOTAL="${#PENDING_IDS[@]}"
echo "Total pending: $TOTAL"

if (( TOTAL == 0 )); then
  echo "All done. Nothing to run."
  tail -f /dev/null
  exit 0
fi

batch=1
for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
  end=$(( i + BATCH_SIZE )); (( end > TOTAL )) && end=$TOTAL
  echo "========== Batch ${batch}: [$((i+1))..$end] / $TOTAL =========="

  for ((j=i; j<end; j++)); do
    ID="${PENDING_IDS[j]}"
    echo ">>> Running utterance_id=$ID"
    if ! python /home/run_track_1.py --utterance_ids "$ID"; then
      echo "!!!! run_track_1.py failed for id=$ID" >&2
    fi
  done

  ((batch++))
done

tail -f /dev/null




