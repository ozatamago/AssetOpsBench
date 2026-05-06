# SPIN Artifact on AssetOpsBench

## Overview
This repository contains the code and saved artifacts used to reproduce the main results of the paper on SPIN, a planning wrapper for tool-using LLM agents that combines executable DAG validation with prefix-based execution control.

The repository supports:
- deterministic recomputation of the reported tables and figures from saved artifacts
- end-to-end replay of benchmark trajectories through Docker Compose
- failure mode analysis replay and figure generation

## Repository Structure
- `benchmark/cods_track1/`: benchmark configuration, Docker Compose files, and experiment outputs
- `src/agent_hive/`: main implementation of planning and execution workflows
- `infra/postgres/`: PostgreSQL-related infrastructure
- `benchmark/cods_track1/track1_result/trajectory/`: saved execution trajectories
- `benchmark/cods_track1/track1_result/exp/`: experiment summaries
- `benchmark/cods_track1/track1_result/trajfm_outputs/`: failure mode analysis outputs

Key implementation files include:
- `src/agent_hive/workflows/track1_planning_baseline.py`
- `src/agent_hive/workflows/track1_planning_spin.py`
- `src/agent_hive/workflows/critic_agent.py`
- `src/agent_hive/workflows/simulator_agent.py`
- `src/agent_hive/workflows/validate_plan_text.py`

## Setup
Clone the repository and switch to the target branch:

```bash
git clone https://github.com/ozatamago/AssetOpsBench.git
cd AssetOpsBench
git checkout UACap10
```

To run end-to-end replay, create:

```bash
benchmark/cods_track1/.env.local
```

## Reproducing Main Results

### 1. Deterministic table recomputation from saved artifacts
The reported tables can be recomputed directly from saved trajectories and experiment summaries without re-running the benchmark or calling external APIs.

Example:
```bash
python3 make_table_2.py \
  --trajectory_root "./benchmark/cods_track1/track1_result/trajectory" \
  --exp_root "./benchmark/cods_track1/track1_result/exp" \
  --model "Model_16" \
  --tags "BASE,SPIN,SPIN_wo_sim,SPIN_wo_cri" \
  --out_dir "./benchmark/cods_track1/track1_result/tables3" \
  --debug
```

### 2. End-to-end trajectory replay
```bash
docker compose -f benchmark/cods_track1/docker-compose.yml up -d --build
```
### 3. Failure mode analysis replay
```bash
docker compose -f benchmark/cods_track1/docker-compose.yml exec -T assetopsbench \
  bash /home/entrypoint_failure_modes_analysis.sh
```

### 4. Figure generation
```bash
python make_fma_figure.py
```

## Outputs

### Main outputs are written under:

- benchmark/cods_track1/track1_result/trajectory/
- benchmark/cods_track1/track1_result/exp/
- benchmark/cods_track1/track1_result/trajfm_outputs/

### Generated FMA figures include:

- fma_full_rate.pdf
- fma_category_rate.pdf

### Notes
Table and figure recomputation from saved artifacts is deterministic.
End-to-end replay depends on external model/API calls and is not guaranteed to be bitwise identical across runs.
Exact replay may break if external provider-side model routing or API mappings change.