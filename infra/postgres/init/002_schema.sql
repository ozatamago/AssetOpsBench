-- ============================================================
-- 002_schema.sql (integrated)
-- Purpose: Planning rounds + Query runs + Full Trajectory DB
-- Notes:
--   - Uses pgcrypto (UUID), pgvector (HNSW), pg_trgm (optional)
--   - FTS via generated tsvector columns (GIN)
--   - JSONB kept at every level + JSONPath-ready GIN
--   - Trajectory top-level = traj_docs; query_runs now links to it
-- ============================================================

-- ---------- Extensions ----------
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector (HNSW index)
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram (optional, fuzzy/JP help)

-- ============================================================
-- 0) Planning / bundle tables (existing concept, unchanged)
-- ============================================================

-- Per-round plan (validated/refined DAG text -> normalized DAG JSON)
CREATE TABLE IF NOT EXISTS plan_dag_rounds (
  plan_id      UUID        NOT NULL DEFAULT gen_random_uuid(),
  scenario_id  INTEGER     NOT NULL,
  llm          TEXT        NOT NULL,
  round_t      INTEGER     NOT NULL CHECK (round_t >= 0),
  prompt       TEXT        NOT NULL,
  answer       TEXT        NOT NULL,
  dag          JSONB       NOT NULL CHECK (dag ? 'nodes' AND dag ? 'edges'),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (plan_id, round_t)
);

CREATE INDEX IF NOT EXISTS idx_plan_dag_rounds_scenario ON plan_dag_rounds (scenario_id);
CREATE INDEX IF NOT EXISTS idx_plan_dag_rounds_dag_gin
  ON plan_dag_rounds USING GIN (dag jsonb_path_ops);

-- ============================================================
-- 1) Trajectory top-level (one row per JSON file)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_docs (
  doc_id      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  json_id     INTEGER     NOT NULL,      -- JSON "id"
  text        TEXT        NOT NULL,      -- JSON "text" (user question)
  raw_json    JSONB       NOT NULL,      -- full original JSON document
  source_path TEXT,                      -- optional: original file path
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Vector & FTS
  text_vec    vector(1536),              -- embedding of "text"
  tsv_all     tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(text,'')) ||
                jsonb_to_tsvector('simple', raw_json, '["string","numeric","boolean"]')
              ) STORED
);

CREATE INDEX IF NOT EXISTS ix_traj_docs_tsv_gin    ON traj_docs USING GIN (tsv_all);
CREATE INDEX IF NOT EXISTS ix_traj_docs_vec_hnsw   ON traj_docs USING hnsw (text_vec vector_l2_ops);
CREATE INDEX IF NOT EXISTS ix_traj_docs_raw_gin    ON traj_docs USING GIN (raw_json jsonb_path_ops);

-- ============================================================
-- 2) Query runs (bundle per user query) – now links to traj_docs
-- ============================================================
CREATE TABLE IF NOT EXISTS query_runs (
  run_id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario_id       INTEGER     NOT NULL,
  plan_id           UUID        NOT NULL,
  text              TEXT        NOT NULL,   -- user question for this run
  trajectory_path   TEXT        CHECK (trajectory_path ~ '\.json$'),
  trajectory_doc_id UUID        NULL REFERENCES traj_docs(doc_id), -- FK to fully parsed json
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (scenario_id, plan_id)
);


CREATE INDEX IF NOT EXISTS idx_query_runs_scenario ON query_runs (scenario_id);
CREATE INDEX IF NOT EXISTS idx_query_runs_trajdoc  ON query_runs (trajectory_doc_id);

-- ============================================================
-- 3) Top-level view (bundle: text + rounds + trajectory pointers)
-- ============================================================

CREATE OR REPLACE VIEW v_query_bundle AS
SELECT
  qr.run_id,
  qr.scenario_id,
  qr.text,
  (
    SELECT jsonb_agg(
             jsonb_build_object(
               'round_t',    r.round_t,
               'llm',        r.llm,
               'prompt',     r.prompt,
               'answer',     r.answer,
               'dag',        r.dag,
               'created_at', r.created_at
             )
             ORDER BY r.round_t
           )
    FROM plan_dag_rounds r
    WHERE r.plan_id = qr.plan_id
      AND r.round_t >= 1
  ) AS plan_dag_rounds,
  qr.trajectory_doc_id AS trajectory_doc_id,
  qr.trajectory_path    AS trajectory_path,
  qr.created_at
FROM query_runs qr;

-- ============================================================
-- 4) Trajectory: top-level array "trajectory[]" → tasks
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_tasks (
  task_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id           UUID        NOT NULL REFERENCES traj_docs(doc_id) ON DELETE CASCADE,
  task_number      INTEGER     NOT NULL,
  task_description TEXT        NOT NULL,
  agent_name       TEXT        NOT NULL,
  response         TEXT        NOT NULL,
  final_answer     TEXT        NOT NULL,
  raw_task_json    JSONB       NOT NULL,

  -- NEW: Critic / evaluation status for this task
  -- Examples: 'Accomplished', 'Partially accomplished', 'Not accomplished', 'Unknown'
  status           TEXT        NOT NULL DEFAULT 'Unknown',

  -- Vector & FTS
  task_vec         vector(1536),
  tsv_task         tsvector GENERATED ALWAYS AS (
                     to_tsvector(
                       'simple',
                       coalesce(task_description,'') || ' ' ||
                       coalesce(agent_name,'')       || ' ' ||
                       coalesce(response,'')         || ' ' ||
                       coalesce(final_answer,'')
                     )
                   ) STORED,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE INDEX IF NOT EXISTS ix_traj_tasks_doc_num   ON traj_tasks (doc_id, task_number);
CREATE INDEX IF NOT EXISTS ix_traj_tasks_tsv_gin   ON traj_tasks USING GIN (tsv_task);
CREATE INDEX IF NOT EXISTS ix_traj_tasks_vec_hnsw  ON traj_tasks USING hnsw (task_vec vector_l2_ops);
CREATE INDEX IF NOT EXISTS ix_traj_tasks_raw_gin   ON traj_tasks USING GIN (raw_task_json jsonb_path_ops);

-- ============================================================
-- 5) Trajectory logs (1:1 with each task)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_logs (
  log_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id        UUID        NOT NULL REFERENCES traj_tasks(task_id) ON DELETE CASCADE,
  type           TEXT,
  task           TEXT,
  environment    TEXT,
  system_prompt  TEXT,
  demonstration  TEXT,
  scratchpad     TEXT,
  endstate       TEXT,
  raw_logs_json  JSONB       NOT NULL,

  -- Vector & FTS
  log_vec        vector(1536),
  tsv_log        tsvector GENERATED ALWAYS AS (
                   to_tsvector('simple',
                     coalesce(type,'') || ' ' ||
                     coalesce(task,'') || ' ' ||
                     coalesce(environment,'') || ' ' ||
                     coalesce(system_prompt,'') || ' ' ||
                     coalesce(demonstration,'') || ' ' ||
                     coalesce(scratchpad,'') || ' ' ||
                     coalesce(endstate,'')
                   )
                 ) STORED,

  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_logs_task       ON traj_logs (task_id);
CREATE INDEX IF NOT EXISTS ix_traj_logs_tsv_gin    ON traj_logs USING GIN (tsv_log);
CREATE INDEX IF NOT EXISTS ix_traj_logs_vec_hnsw   ON traj_logs USING hnsw (log_vec vector_l2_ops);
CREATE INDEX IF NOT EXISTS ix_traj_logs_raw_gin    ON traj_logs USING GIN (raw_logs_json jsonb_path_ops);

-- ============================================================
-- 5-a) logs.trajectroy_log[] (per-step details)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_log_steps (
  step_id                        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  log_id                         UUID        NOT NULL REFERENCES traj_logs(log_id) ON DELETE CASCADE,
  step                           INTEGER,
  raw_llm_thought_output         TEXT,
  raw_llm_action_output          TEXT,
  raw_observation_output         TEXT,
  raw_llm_output                 TEXT,
  thought                        TEXT,
  action                         TEXT,
  action_input                   TEXT,
  observation                    TEXT,
  state                          TEXT,
  is_loop_detected               BOOLEAN,
  additional_scratchpad_feedback TEXT,
  step_trajectory_file_name      TEXT,
  step_metric_file_name          TEXT,
  step_trajectory_json           JSONB,
  step_metric_json               JSONB,

  -- Vector & FTS
  step_vec                       vector(1536),
  tsv_step                       tsvector GENERATED ALWAYS AS (
                                    to_tsvector('simple',
                                      coalesce(thought,'') || ' ' ||
                                      coalesce(action,'')  || ' ' ||
                                      coalesce(action_input,'') || ' ' ||
                                      coalesce(observation,'') || ' ' ||
                                      coalesce(state,'') || ' ' ||
                                      coalesce(raw_llm_thought_output,'') || ' ' ||
                                      coalesce(raw_llm_action_output,'')  || ' ' ||
                                      coalesce(raw_observation_output,'') || ' ' ||
                                      coalesce(raw_llm_output,'')
                                    )
                                  ) STORED,

  created_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_log_steps_log_idx  ON traj_log_steps (log_id, step);
CREATE INDEX IF NOT EXISTS ix_traj_log_steps_tsv_gin  ON traj_log_steps USING GIN (tsv_step);
CREATE INDEX IF NOT EXISTS ix_traj_log_steps_vec_hnsw ON traj_log_steps USING hnsw (step_vec vector_l2_ops);

-- ============================================================
-- 5-b) logs.history[] (message history)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_log_history (
  history_id  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  log_id      UUID        NOT NULL REFERENCES traj_logs(log_id) ON DELETE CASCADE,
  idx         INTEGER     NOT NULL,
  role        TEXT,
  content     TEXT,
  agent       TEXT,
  is_demo     BOOLEAN,

  -- Vector & FTS
  history_vec vector(1536),
  tsv_history tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(content,''))
              ) STORED,

  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_log_history_log_idx  ON traj_log_history (log_id, idx);
CREATE INDEX IF NOT EXISTS ix_traj_log_history_tsv_gin  ON traj_log_history USING GIN (tsv_history);
CREATE INDEX IF NOT EXISTS ix_traj_log_history_vec_hnsw ON traj_log_history USING hnsw (history_vec vector_l2_ops);

-- ============================================================
-- 5-c) logs.trajectory[] (inner trajectory steps)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_log_inner_trajectory (
  inner_traj_id  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  log_id         UUID        NOT NULL REFERENCES traj_logs(log_id) ON DELETE CASCADE,
  idx            INTEGER     NOT NULL,
  thought        TEXT,
  action         TEXT,
  observation    TEXT,

  -- Vector & FTS
  inner_traj_vec vector(1536),
  tsv_inner_traj tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple',
                      coalesce(thought,'') || ' ' ||
                      coalesce(action,'')  || ' ' ||
                      coalesce(observation,'')
                    )
                  ) STORED,

  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_log_inner_traj_log_idx  ON traj_log_inner_trajectory (log_id, idx);
CREATE INDEX IF NOT EXISTS ix_traj_log_inner_traj_tsv_gin  ON traj_log_inner_trajectory USING GIN (tsv_inner_traj);
CREATE INDEX IF NOT EXISTS ix_traj_log_inner_traj_vec_hnsw ON traj_log_inner_trajectory USING hnsw (inner_traj_vec vector_l2_ops);

-- ============================================================
-- 6) info.model_stats (per task)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_info_model_stats (
  stats_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id         UUID        NOT NULL REFERENCES traj_tasks(task_id) ON DELETE CASCADE,
  tokens_sent     INTEGER,
  tokens_received INTEGER,
  api_calls       INTEGER,
  total_cost      NUMERIC,
  instance_cost   NUMERIC,
  raw_info_json   JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_info_task ON traj_info_model_stats (task_id);

-- ============================================================
-- 7) reviews / reflections (per task arrays → rows)
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_reviews (
  review_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id     UUID        NOT NULL REFERENCES traj_tasks(task_id) ON DELETE CASCADE,
  idx         INTEGER     NOT NULL,
  text        TEXT        NOT NULL,

  -- Vector & FTS
  review_vec  vector(1536),
  tsv_review  tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(text,''))
              ) STORED,

  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_reviews_task_idx   ON traj_reviews (task_id, idx);
CREATE INDEX IF NOT EXISTS ix_traj_reviews_tsv_gin    ON traj_reviews USING GIN (tsv_review);
CREATE INDEX IF NOT EXISTS ix_traj_reviews_vec_hnsw   ON traj_reviews USING hnsw (review_vec vector_l2_ops);

CREATE TABLE IF NOT EXISTS traj_reflections (
  reflection_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  task_id         UUID        NOT NULL REFERENCES traj_tasks(task_id) ON DELETE CASCADE,
  idx             INTEGER     NOT NULL,
  text            TEXT        NOT NULL,

  -- Vector & FTS
  reflection_vec  vector(1536),
  tsv_reflection  tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(text,''))
                  ) STORED,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_traj_reflections_task_idx ON traj_reflections (task_id, idx);
CREATE INDEX IF NOT EXISTS ix_traj_reflections_tsv_gin  ON traj_reflections USING GIN (tsv_reflection);
CREATE INDEX IF NOT EXISTS ix_traj_reflections_vec_hnsw ON traj_reflections USING hnsw (reflection_vec vector_l2_ops);


CREATE TABLE IF NOT EXISTS dag_trajectory_score (
    score_id       BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    plan_id        UUID    NOT NULL,
    doc_id         UUID    NOT NULL REFERENCES traj_docs (doc_id) ON DELETE CASCADE,

    num_edit       INTEGER,
    correct        NUMERIC(4,3) CHECK (correct >= 0 AND correct <= 1),
    num_partially  INTEGER CHECK (num_partially >= 0),
    num_not        INTEGER CHECK (num_not >= 0),
    error_analysis TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_score_round_doc UNIQUE (plan_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_score_round ON dag_trajectory_score (plan_id);
CREATE INDEX IF NOT EXISTS idx_score_doc   ON dag_trajectory_score (doc_id);


-- ============================================================
-- 8) Task-level summaries (1 row per (doc_id, task_id))
-- ============================================================

CREATE TABLE IF NOT EXISTS traj_task_summaries (
  summary_id      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Identity / joins
  doc_id          UUID        NOT NULL REFERENCES traj_docs(doc_id) ON DELETE CASCADE,
  task_id         UUID        NOT NULL REFERENCES traj_tasks(task_id) ON DELETE CASCADE,
  task_number     INTEGER,  -- optional, mirrors traj_tasks.task_number for convenience

  -- Core task information (denormalized for easy retrieval)
  user_question   TEXT        NOT NULL,  -- copy of traj_docs.text
  task_description TEXT       NOT NULL,  -- copy of traj_tasks.task_description
  agent_name      TEXT        NOT NULL,  -- copy of traj_tasks.agent_name
  final_answer    TEXT,                  -- copy or shortened version of traj_tasks.final_answer
  status          TEXT,                  -- e.g. 'Accomplished', 'Partially accomplished', 'Not accomplished'
  review          TEXT,                  -- parsed / concatenated review text for this task

  -- Human-readable summary
  summary         TEXT        NOT NULL,  -- short natural-language summary of this task

  -- Vector & FTS for summary-level search
  summary_vec     vector(1536),          -- embedding of `summary` (same dim as other *_vec columns)
  tsv_summary     tsvector GENERATED ALWAYS AS (
                    to_tsvector(
                      'simple',
                      coalesce(user_question,'')   || ' ' ||
                      coalesce(task_description,'') || ' ' ||
                      coalesce(agent_name,'')       || ' ' ||
                      coalesce(summary,'')
                    )
                  ) STORED,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- Enforce "one summary per task"
  CONSTRAINT uq_traj_task_summaries_task UNIQUE (doc_id, task_id)
);

-- Join / lookup by doc_id + task_id
CREATE INDEX IF NOT EXISTS ix_traj_task_summaries_doc_task
  ON traj_task_summaries (doc_id, task_id);

-- Fast lookup by doc_id + task_number (often used in ingestion / debugging)
CREATE INDEX IF NOT EXISTS ix_traj_task_summaries_doc_num
  ON traj_task_summaries (doc_id, task_number);

-- Full-text search over summaries
CREATE INDEX IF NOT EXISTS ix_traj_task_summaries_tsv_gin
  ON traj_task_summaries USING GIN (tsv_summary);

-- Vector similarity search over summary embeddings
CREATE INDEX IF NOT EXISTS ix_traj_task_summaries_vec_hnsw
  ON traj_task_summaries
  USING hnsw (summary_vec vector_l2_ops);
