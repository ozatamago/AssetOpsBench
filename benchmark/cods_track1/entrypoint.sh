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

# （確認だけ。無くてもOK）
python -m pip show agent_hive || true
python -m pip show reactxen || true
python -m pip show fmsr_agent || true
python -m pip show iotagent || true
python -m pip show tsfmagent || true

# 便利パッケージ（任意）
python -m pip install -qU "psycopg[binary]>=3.1" "ibm-watsonx-ai>=1.4.0"

# ==== ID 列挙 ====
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

# ==== PENDING を集める（既存はスキップ）====
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

# ==== 10件ずつ実行 ====
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

# コンテナを起動状態に維持（ログ確認用）
tail -f /dev/null


# #!/bin/bash
# # Activate conda env
# source /opt/conda/etc/profile.d/conda.sh
# conda activate assetopsbench

# which python
# python --version
# python -m pip show agent_hive
# python -m pip show reactxen
# python -m pip show fmsr_agent
# python -m pip show iotagent
# python -m pip show tsfmagent

# python -m pip install -qU "psycopg[binary]>=3.1"

# # Run the entire thing
# # 1 - 141
# python /home/run_track_1.py --utterance_ids 1

# # Keep the container alive
# tail -f /dev/null

