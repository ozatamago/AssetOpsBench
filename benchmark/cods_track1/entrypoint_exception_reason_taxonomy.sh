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

# # python /home/run_track_1.py --utterance_ids "1,2,3,4,5,6,7,8,9,10,11,12,41,42,43,44,45,46,47,48,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,4001,4002,4003,4004,4005,4006,4007,4008,4009,40010,40011,40012,40013,40014,40015,40016,40017,40018,40019,40020,40021,40022,40023,40024,40025,40026,40027,40028,40029,40030,40031,40032,40033,40034,40035,40036,40037,40038,40039,40040,40041,40042,40043,40044,40045,40046,40047,40048,40049,40050,40051,40052,40053,40054,40055,40056,40057,40058,40059,40060,40061,40062,40063,40064,40065,40066,40067,40068,40069,40070,40071,40072,40073,40074,40075,40076,40077,40078,40079,40080,40081,40082,40083,40084,40085,40086,40087,40088,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2101,2102,2103,2104,2105,2106,2107" 
# python /home/exception_reason_taxonomy.py \
#   --input_dir "/home/track1_result/trajectory/[BASE_RAX]Model_16" \
#   --output_dir "/home/track1_result/exception_reason_taxonomy/[BASE_RAX]Model_16" \
#   --model_id 20 \
#   --debug true \
#   # --max_cases 5

# python /home/build_plan_taxonomies_all_in_one.py \
#   --plan_dir "/home/track1_result/plan/[ReAct_CD][no_verify]Model_16" \
#   --output_dir "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16" \
#   --model_id 20 \
#   --batch_size 10 \
#   --merge_group_size 5

# python /home/build_frequency_model.py \
#   --plan_dir "/home/track1_result/plan/[ReAct_CD][no_verify]Model_16" \
#   --trajectory_dir "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/common_exception_taxonomy.json" \
#   --output_dir "/home/track1_result/frequency_model/[ReAct_CD][no_verify]Model_16" \
#   --model_id 20 \
#   --alpha 1.0 \
#   --llm_temperature 0.0 \
#   --logs_max_chars 2500 \
#   --debug

# python /home/build_frequency_model.py \
#   --plan_dir "/home/track1_result/plan/[RAX_CD][no_verify]Model_16" \
#   --trajectory_dir "/home/track1_result/trajectory/[RAX_CD][no_verify]Model_16" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/common_exception_taxonomy.json" \
#   --output_dir "/home/track1_result/frequency_model/[RAX_CD][no_verify]Model_16" \
#   --model_id 20 \
#   --alpha 1.0 \
#   --llm_temperature 0.0 \
#   --logs_max_chars 2500 \
#   --debug

# python /home/verification_f1_score.py \
#   --builder_script "/home/build_frequency_model.py" \
#   --plan_dir "/home/track1_result/plan/[ReAct_CD][allocation_only]Model_16" \
#   --frequency_model "/home/track1_result/frequency_model/[ReAct_CD][no_verify]Model_16/frequency_model_with_risk.json" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/common_exception_taxonomy.json" \
#   --output_examples "/home/track1_result/eval/merged_node_signature_and_verifier_allocation_examples.json" \
#   --output_metrics "/home/track1_result/eval/verifier_allocation_metrics.json" \
#   --output_errors "/home/track1_result/eval/verifier_allocation_errors.json" \
#   --cache_json "/home/track1_result/eval/llm_cache_eval.json" \
#   --min_support 5

# python /home/compare_annotated_vs_allocation.py \
#   --annotated_dir "/home/track1_result/plan/[ReAct_CD][annotated_only]Model_16" \
#   --allocation_dir "/home/track1_result/plan/[ReAct_CD][allocation_only]Model_16" \
#   --output_examples "/home/track1_result/eval/annotated_vs_allocation_examples.json" \
#   --output_metrics "/home/track1_result/eval/annotated_vs_allocation_metrics.json"

# python /home/build_node_signature_table.py \
#   --builder_script "/home/build_frequency_model.py" \
#   --plan_dir "/home/track1_result/plan/[ReAct_CD][no_verify]Model_16" \
#   --trajectory_dir "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/common_exception_taxonomy.json" \
#   --existing_node_signature_table "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table.json" \
#   --output_json "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table_v2.json" \
#   --output_updates_json "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_outcome_failure_updates.json" \
#   --output_errors "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_errors_v2.json" \
#   --output_summary "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_summary_v2.json" \
#   --cache_json "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_cache_v2.json"

# python /home/build_node_signature_table.py \
#   --builder_script "/home/build_frequency_model.py" \
#   --plan_dir "/home/track1_result/plan/[RAX_CD][no_verify]Model_16" \
#   --trajectory_dir "/home/track1_result/trajectory/[RAX_CD][no_verify]Model_16" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/common_exception_taxonomy.json" \
#   --output_json "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_table_v2.json" \
#   --output_updates_json "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_outcome_failure_updates.json" \
#   --output_errors "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_errors_v2.json" \
#   --output_summary "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_summary_v2.json" \
#   --cache_json "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_cache_v2.json"

# python /home/build_node_signature_table.py \
#   --builder_script "/home/build_frequency_model.py" \
#   --plan_dir "/home/track1_result/plan/[ReAct_CD][no_verify]Model_16" \
#   --trajectory_dir "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/stratified_exception_taxonomy.json" \
#   --existing_node_signature_table "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table_v2.json" \
#   --output_json "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table_v4.json" \
#   --output_updates_json "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_outcome_failure_updates_v4.json" \
#   --output_errors "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_errors_v4.json" \
#   --output_summary "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_summary_v4.json" \
#   --cache_json "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_cache_v4.json"

# python /home/build_node_signature_table.py \
#   --builder_script "/home/build_frequency_model.py" \
#   --plan_dir "/home/track1_result/plan/[RAX_CD][no_verify]Model_16" \
#   --trajectory_dir "/home/track1_result/trajectory/[RAX_CD][no_verify]Model_16" \
#   --task_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/task_taxonomy/final_task_taxonomy.json" \
#   --contract_taxonomy_json "/home/track1_result/plan_taxonomy/[ReAct_CD][no_verify]Model_16/node_contract_taxonomy/final_node_contract_taxonomy.json" \
#   --exception_taxonomy "/home/track1_result/exception_reason_taxonomy/stratified_exception_taxonomy.json" \
#   --existing_node_signature_table "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_table_v2.json" \
#   --output_json "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_table_v4.json" \
#   --output_updates_json "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_outcome_failure_updates_v4.json" \
#   --output_errors "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_errors_v4.json" \
#   --output_summary "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_summary_v4.json" \
#   --cache_json "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_cache_v4.json"

# python /home/eval_module2.py \
#   --node_signature_table "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table_v2.json" \
#   --annotated_plan_dir "/home/track1_result/plan/[ReAct_CD][annotated_only_prior_1]Model_16" \
#   --output_examples "/home/track1_result/eval/[ReAct_CD][no_verify]Model_16/module2_examples.json" \
#   --output_metrics "/home/track1_result/eval/[ReAct_CD][no_verify]Model_16/module2_metrics.json"

# python /home/eval_module2.py \
#   --node_signature_table "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_table_v2.json" \
#   --annotated_plan_dir "/home/track1_result/plan/[RAX_CD][annotated_only_prior_1]Model_16" \
#   --output_examples "/home/track1_result/eval/[RAX_CD][no_verify]Model_16/module2_examples.json" \
#   --output_metrics "/home/track1_result/eval/[RAX_CD][no_verify]Model_16/module2_metrics.json"

# python /home/eval_module3.py \
#   --node_signature_table "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table.json" \
#   --allocation_plan_dir "/home/track1_result/plan/[ReAct_CD][allocation_only]Model_16" \
#   --output_examples "/home/track1_result/eval/module3_examples.json" \
#   --output_metrics "/home/track1_result/eval/module3_metrics.json"

# python /home/build_frequency_model_with_risk_from_node_table.py \
#   --node-signature-table "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table_v4.json" \
#   --weights-json "/home/track1_result/frequency_model/[ReAct_CD][no_verify]Model_16/weights.json" \
#   --output-json "/home/track1_result/frequency_model/[ReAct_CD][no_verify]Model_16/frequency_model_with_risk.json" \
#   --alpha 1.0 \
#   --oracle-quantile 0.80 \
#   --observed-label-source "failure_reason_labels" \
#   --expected-label-source "expected_exception_labels" \
#   --fallback-expected-label-source "failure_reason_labels" \
#   --dep-key-mode "both"

# python /home/build_frequency_model_with_risk_from_node_table.py \
#   --node-signature-table "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_table_v4.json" \
#   --weights-json "/home/track1_result/frequency_model/[RAX_CD][no_verify]Model_16/weights.json" \
#   --output-json "/home/track1_result/frequency_model/[RAX_CD][no_verify]Model_16/frequency_model_with_risk.json" \
#   --alpha 1.0 \
#   --oracle-quantile 0.80 \
#   --observed-label-source "failure_reason_labels" \
#   --expected-label-source "expected_exception_labels" \
#   --fallback-expected-label-source "failure_reason_labels" \
#   --dep-key-mode "both"

# python /home/verifier_oracle_threshold_sweep.py \
#   --node-signature-table "/home/track1_result/node_signature_table/[ReAct_CD][no_verify]Model_16/node_signature_table_v4.json" \
#   --frequency-model "/home/track1_result/frequency_model/[ReAct_CD][no_verify]Model_16/frequency_model_with_risk.json" \
#   --output-dir "/home/track1_result/verifier_oracle/[ReAct_CD][no_verify]Model_16_resweep" \
#   --mode resweep \
#   --risk-field assigned_risk_weighted \
#   --target-miss-rate 0.10 \
#   --expected-label-source expected_exception_labels \
#   --fallback-expected-label-source failure_reason_labels

# python /home/verifier_oracle_threshold_sweep.py \
#   --node-signature-table "/home/track1_result/node_signature_table/[RAX_CD][no_verify]Model_16/node_signature_table_v4.json" \
#   --frequency-model "/home/track1_result/frequency_model/[RAX_CD][no_verify]Model_16/frequency_model_with_risk.json" \
#   --output-dir "/home/track1_result/verifier_oracle/[RAX_CD][no_verify]Model_16_resweep" \
#   --mode resweep \
#   --risk-field assigned_risk_weighted \
#   --target-miss-rate 0.10 \
#   --expected-label-source expected_exception_labels \
#   --fallback-expected-label-source failure_reason_labels

# python /home/verification_f1_score.py \
#   --annotated_plan_dir "/home/track1_result/plan/[ReAct_CD][annotated_only_w_few_shot_1]Model_16" \
#   --allocation_plan_dir "/home/track1_result/plan/[ReAct_CD][allocation_only_w_few_shot_1]Model_16" \
#   --oracle_rows_json "/home/track1_result/verifier_oracle/[ReAct_CD][no_verify]Model_16_resweep/oracle_rows_with_threshold.json" \
#   --output_dir "/home/track1_result/verification_f1/[ReAct_CD][annotated_only_w_few_shot_1]_with_module2_split_Model_16"

# python /home/verification_f1_score.py \
#   --annotated_plan_dir "/home/track1_result/plan/[RAX_CD][annotated_only_prior_1]Model_16" \
#   --allocation_plan_dir "/home/track1_result/plan/[RAX_CD][allocation_only_modularize_3]Model_16" \
#   --oracle_rows_json "/home/track1_result/verifier_oracle/[RAX_CD][no_verify]Model_16_resweep/oracle_rows_with_threshold.json" \
#   --output_dir "/home/track1_result/verification_f1/[RAX_CD][allocation_only_modularize_3]_with_module2_split_Model_16"

python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/verification_agent.py \
  --trajectory_dir "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16" \
  --output_path "/home/track1_result/verification_oracle/react_diagnosis_only_cache_v3.json" \
  --output_mode diagnosis_only

  # --trajectory_path "/home/track1_result/trajectory/[ReAct_CD][no_verify]Model_16/Q_619_trajectory.json" \

# python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/verification_agent.py \
#   --diagnosis_cache_path "/home/track1_result/verification_oracle/react_diagnosis_only_cache.json" \
#   --output_dir "/home/track1_result/recovery_only/react" \
#   --output_mode recovery_only

# python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/verification_agent.py \
#   --diagnosis_cache_path "/home/track1_result/verification_oracle/reactxen_diagnosis_only_cache.json" \
#   --output_dir "/home/track1_result/recovery_only/reactxen" \
#   --output_mode recovery_only

# python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/verification_agent.py \
  # --trajectory_dir "/home/track1_result/trajectory/[RAX_CD][no_verify_common_base_plan_w_ReAct]Model_16" \
  # --output_path "/home/track1_result/verification_oracle/reactxen_diagnosis_only_cache.json" \
  # --output_mode diagnosis_only

# python /opt/conda/envs/assetopsbench/lib/python3.12/site-packages/agent_hive/workflows/verification_agent.py \
#   --trajectory_dir "/home/track1_result/trajectory/[RAFA_CD][no_verify_common_base_plan_w_ReAct]Model_16" \
#   --output_path "/home/track1_result/verification_oracle/rafa_diagnosis_only_cache.json" \
#   --output_mode diagnosis_only