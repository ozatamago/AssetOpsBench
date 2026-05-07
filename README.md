## Reproducing the Trajectory-Level FMA and Verifier-Oracle Pipeline

This section describes how to reproduce the main analysis pipeline used in the paper, including plan/trajectory generation, node-signature construction, risk estimation, and verifier-oracle threshold sweeping.

### 1. Launch the customized Track-1 environment

From the repository root:

```bash
cd benchmark/cods_track1
docker compose up --build
```

This launches the customized AssetOpsBench Track-1 environment. The main outputs are written under:
```bash
benchmark/cods_track1/track1_result/
```

Inside the container, the main scripts used in the paper are mounted at:
```bash
/home/run_track_1.py
/home/build_node_signature_table.py
/home/build_frequency_model_with_risk_from_node_table.py
/home/verifier_oracle_threshold_sweep.py
/home/auto_scoring.py
```

### 2. Generate plans and trajectories
Run the Track-1 workflow with a comma-separated list of scenario IDs:
```bash
python /home/run_track_1.py \
  --utterance_ids "1,2,3,4,5" \
  --generate_steps_only False
  ```

- --utterance_ids: comma-separated list of scenario IDs
- --generate_steps_only: if True, stop after plan generation; if False, execute the full Plan-Execute workflow


### 3. Build the node-signature table
Construct a node-signature table from generated plans and trajectories:
```bash
python /home/build_node_signature_table.py \
  --builder_script /home/build_frequency_model.py \
  --plan_dir /path/to/track1_result/plan/<planner_condition>/<model_id> \
  --trajectory_dir /path/to/track1_result/trajectory/<planner_condition>/<model_id> \
  --task_taxonomy_json /path/to/merged_task_taxonomy.json \
  --contract_taxonomy_json /path/to/merged_node_contract_taxonomy.json \
  --exception_taxonomy /path/to/exception_reason_taxonomy.json \
  --output_json /path/to/node_signature_table_v2.json \
  --output_errors /path/to/node_signature_errors.json \
  --output_summary /path/to/node_signature_summary.json
```

### 4. Build the frequency model with risk

Estimate p(Y | state, label) and derive risk values from the node-signature table:

```bash
python /home/build_frequency_model_with_risk_from_node_table.py \
  --node-signature-table /path/to/node_signature_table_v2.json \
  --weights-json /path/to/weights.json \
  --output-json /path/to/frequency_model_with_risk.json \
  --alpha 1.0 \
  --oracle-quantile 0.80 \
  --observed-label-source failure_reason_labels \
  --expected-label-source expected_exception_labels \
  --fallback-expected-label-source failure_reason_labels \
  --dep-key-mode both
```

Arguments:
```bash
--node-signature-table: node-signature table JSON
--weights-json: outcome weights JSON
--output-json: output frequency/risk model JSON
--alpha: additive smoothing parameter
--oracle-quantile: quantile used to derive stored oracle flags
--observed-label-source: label source for estimating p(Y | state, label)
--expected-label-source: label source for assembling signature-level risk
--fallback-expected-label-source: fallback label field if the expected source is absent
--dep-key-mode: one of both, bucket_only, or raw_only
```

### 5. Attach oracle decisions or resweep thresholds

Use the stored oracle flags already present in the risk table:

```bash
python /home/verifier_oracle_threshold_sweep.py \
  --node-signature-table /path/to/node_signature_table_v2.json \
  --frequency-model /path/to/frequency_model_with_risk.json \
  --output-dir /path/to/oracle_eval_dir \
  --mode use_existing_oracle \
  --oracle-field oracle_verify_weighted_q80 \
  --expected-label-source expected_exception_labels \
  --fallback-expected-label-source failure_reason_labels
```
Or resweep thresholds directly from assigned risk:
```bash
python /home/verifier_oracle_threshold_sweep.py \
  --node-signature-table /path/to/node_signature_table_v2.json \
  --frequency-model /path/to/frequency_model_with_risk.json \
  --output-dir /path/to/oracle_eval_dir \
  --mode resweep \
  --risk-field assigned_risk_weighted \
  --target-miss-rate 0.10 \
  --harmful-outcomes P,N,E \
  --expected-label-source expected_exception_labels \
  --fallback-expected-label-source failure_reason_labels
```
Arguments:
```bash
--mode: use_existing_oracle or resweep
--oracle-field: boolean oracle field in the risk table, used when mode=use_existing_oracle
--risk-field: one of assigned_risk_weighted or assigned_risk_simple, used when mode=resweep
--target-miss-rate: upper bound on miss rate for threshold resweep
--harmful-outcomes: comma-separated outcomes treated as oracle positives
```