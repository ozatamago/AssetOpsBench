from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from huggingface_hub import login

login(os.getenv("HF_APIKEY", None))

from agent_hive.task import Task
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
    iot_task_examples,
)
from agent_hive.tools.tsfm import (
    tsfm_tools,
    tsfm_fewshots,
    tsfm_agent_name,
    tsfm_agent_description,
    tsfm_task_examples,
)
from agent_hive.tools.wo import (
    wo_agent_description,
    wo_agent_name,
    wo_fewshots,
    wo_tools,
    wo_task_examples,
)
from agent_hive.agents.react_reflect_agent import ReactReflectAgent
from agent_hive.logger import get_custom_logger
from agent_hive.agents.wo_agent import WorderOrderAgent
from agent_hive.workflows.track1_planning import NewPlanningWorkflow

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

import warnings

warnings.filterwarnings("ignore")

from auto_scoring import auto_score_dag_trajectory

RESULT_DIR = "/home/track1_result/"
PLAN_DIR = RESULT_DIR + "plan/"
TRAJECTORY_DIR = RESULT_DIR + "trajectory/"
SUMMARY_DIR = RESULT_DIR + "summary/"

human_scored_triplets= """
Q6
dag: 
{
  "ok": true,
  "final_plan": " \n#Task1: List the assets at the MAIN site\n#Agent1: IoT Data Download\n#Dependency1: None\n#ExpectedOutput1: List of assets at the MAIN site\n#Task2: Verify if Chiller 9 is present in the list of assets at the MAIN site\n#Agent2: IoT Data Download\n#Dependency2: #S1\n#ExpectedOutput2: Confirmation that Chiller 9 is among the assets listed\n#Task3: Retrieve asset details for Chiller 9 at the MAIN site\n#Agent3: IoT Data Download\n#Dependency3: #S2\n#ExpectedOutput3: Asset details for Chiller 9",
  "scenario_id": 6,
  "generated_at": "2025-10-19T15:58:17.539122Z"
}
trajectory: 
{
  "text": "Get the asset details for Chiller 9 at the MAIN site.",
  "tasks": [
    {
      "task_description": "List assets at MAIN site",
      "agent_name": "IoT Data Download",
      "final_answer": "Question</response>",
      "review": "Task Status: Accomplished | executed 'assets' tool successfully | avoid redundant actions</response>\n```<response>executed 'assets' tool successfully | avoid redundant actions</response>```"
    },
    {
      "task_description": "Check if Chiller 9 is in MAIN site assets",
      "agent_name": "IoT Data Download",
      "final_answer": "MAIN</response>",
      "review": "Task Status: Accomplished | read JSON file correctly | none</response>"
    },
    {
      "task_description": "Fetch Chiller 9 asset details from MAIN.",
      "agent_name": "IoT Data Download",
      "final_answer": "MAIN</response>",
      "review": "Task Status: Partially Accomplished | identified relevant JSON file | extract Chiller 9 details from JSON file."
    }
  ]
}
score: 
Q6
- num_edit: 0
- correct: 0.4
- num_partially: 1
- num_not: 0
- error_analysis: Symptom 1: Final answer lacks required details
The response returns only a file path and even appends stray boilerplate (“Here’s the rewritten response… Question”), instead of extracting and presenting Chiller 9’s actual fields (e.g., id, name, site, metadata). Fix: After obtaining the file path, call jsonreader, filter for “Chiller 9”, and surface the concrete fields in the final answer. Keep the final answer clean and free of internal notes. Symptom 2: Action-input contamination A tool call was issued with site_name=MAIN ## Step 5: Analyze the result ..., causing unknown site "MAIN ## Step 5...". This is classic argument leakage from thoughts/explanations into the action payload. Fix: Enforce strict tool-call hygiene: arguments must be structured inputs only (no commentary). Many tool-use guides emphasize separating “reasoning” from “tool arguments” and keeping final outputs free of scaffolding text. Symptom 3: Redundant & brittle calls The agent repeats the assets call after already getting a valid file, then errors. Fix: Cache observations and avoid repeated identical calls unless inputs change. Transition immediately from “list” → “read/parse” → “filter” → “final answer.” This reduces surface area for failures and aligns with robust ReAct-style loops. Symptom 4: Improper output handling The final message includes scaffolding text (“Question”) and references to internal formatting, which is a form of improper output handling in agent pipelines. Fix: Post-process the final answer to strip scaffolding and ensure only the user-facing result remains (a practice echoed in LLM agent risk/best-practice write-ups). 

Q410
dag: 
{
  "ok": true,
  "final_plan": " \n#Task1: Retrieve the events of equipment CWC04009 for the first week of June 2020.\n#Agent1: IoT Data Download\n#Dependency1: None\n#ExpectedOutput1: A JSON file containing the events of equipment CWC04009 for the first week of June 2020.\n#Task2: Analyze the retrieved events and provide a summary based on the event group for work order event, alert, and anomaly.\n#Agent2: WorkOrder Agent\n#Dependency2: #S1\n#ExpectedOutput2: A summary in JSON format containing the count of work order events, alerts, and anomalies for equipment CWC04009 for the first week of June 2020.\n#Task3: Verify if the summary contains the required information for work order event, alert, and anomaly.\n#Agent3: IoT Data Download\n#Dependency3: #S2\n#ExpectedOutput3: Confirmation if the summary contains the required information.",
  "scenario_id": 410,
  "generated_at": "2025-10-19T19:06:59.745523Z"
}
trajectory: 
{
  "text": "Get all the events of equipment CWC04009 for the first week of June of 2020 and provide a summary based on the event group for work order event, alert, and anomaly.",
  "tasks": [
    {
      "task_description": "Fetch CWC04009 events for June 1-7 2020.",
      "agent_name": "IoT Data Download",
      "final_answer": "MAIN</response>",
      "review": "Task Status: Accomplished | followed logical sequence | provide detailed error handling</response>\n```<response>followed logical sequence | provide detailed error handling</response>```"
    },
    {
      "task_description": "Group work order events, alerts, and anomalies by category.",
      "agent_name": "WorkOrder Agent",
      "final_answer": "{\"CWC04009\": {\"ALERT\": 6}} </response>\nbecomes \n{\"CWC04009\": {\"ALERT\": 6}}",
      "review": "Task Status: Accomplished | retrieved events correctly | none</response>\n<response>retrieved events correctly | none</response>"
    },
    {
      "task_description": "Check work order event alert and anomaly details in summary.",
      "agent_name": "IoT Data Download",
      "final_answer": "",
      "review": "Task Status: Not Accomplished | failed to verify summary | ensure summary file is generated or available</response>\n</to-solve>"
    }
  ]
}
score: 
Q410
num_edit: 1
correct: 0.0
num_partially: 0
num_not: 1
error_analysis: Task 1 skipped asset ID/site reconciliation, prematurely assumed “absent at MAIN,” and failed to generate the event JSON. Task 2 aggregated only ALERTs (6 items); WO/ANOM were missing, and the result is inconsistent with Task 1’s conclusion. Task 3 was not executed (no validation performed). Add a preliminary task for “equipment ID / alias / site resolution,” ensure consistency by analyzing from a single data source, then have the verification agent check for requirement compliance.
"""

import os
from typing import Any, Callable, Iterable, Mapping, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

VectorFn = Optional[Callable[[str], list[float]]]

def _as_vector_literal(vec: list[float]) -> str:
    """
    pgvector リテラル表現: [v1,v2,...] を返す。
    INSERT/UPDATE 時に `CAST(%s AS vector(1536))` で使う。
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

from typing import Any, Mapping, Optional, Sequence, Callable
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

def save_full_trajectory(
    db_url: str,
    output: Mapping[str, Any],
    *,
    source_path: Optional[str] = None,          # Q_*.json のファイルパス（任意）
    run_id: Optional[str] = None,               # 既存 query_runs を紐付け更新したい場合（任意）
    plan_id: Optional[str] = None,              # 上に同じ（どちらか分かればOK）
    scenario_id: Optional[int] = None,          # plan_id と併用して一意に更新したい場合（任意）
    embed_text: Optional[Callable[[str], Sequence[float]]] = None,  # 文字列→ベクトル(1536)
) -> str:
    """
    `output = {"id": int, "text": str, "trajectory": [...]}` を
      - traj_docs (+ text_vec 任意)
      - traj_tasks
      - traj_logs
      - traj_log_steps
      - traj_log_history
      - traj_log_inner_trajectory
      - traj_info_model_stats
      - traj_reviews
      - traj_reflections
    に INSERT。run_id / plan_id があれば query_runs を UPDATE し、
    trajectory_doc_id / trajectory_path を埋める。

    戻り値: 作成した traj_docs.doc_id（文字列）
    """
    json_id = int(output.get("id", 0))
    text = str(output.get("text", "") or "")
    traj_list = output.get("trajectory") or []

    # psycopg3: connection/context を抜けたら接続クローズ、transaction() でトランザクション境界
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # ---------- 1) traj_docs ----------
                cur.execute(
                    """
                    INSERT INTO traj_docs (json_id, text, raw_json, source_path)
                    VALUES (%s, %s, %s, %s)
                    RETURNING doc_id
                    """,
                    (json_id, text, Json(output), source_path),
                )
                doc_id = cur.fetchone()["doc_id"]

                # ベクトル更新（任意）
                if embed_text:
                    try:
                        vec = embed_text(text)
                    except Exception:
                        vec = None
                    if vec:
                        cur.execute(
                            "UPDATE traj_docs SET text_vec = CAST(%s AS vector(1536)) WHERE doc_id = %s",
                            (_as_vector_literal(vec), doc_id),
                        )

                # ---------- 2) query_runs 紐付け（任意） ----------
                if run_id:
                    cur.execute(
                        """
                        UPDATE query_runs
                        SET trajectory_doc_id = COALESCE(trajectory_doc_id, %s),
                            trajectory_path    = COALESCE(trajectory_path, %s)
                        WHERE run_id = %s
                        """,
                        (doc_id, source_path, run_id),
                    )
                elif plan_id:
                    if scenario_id is not None:
                        cur.execute(
                            """
                            UPDATE query_runs
                            SET trajectory_doc_id = COALESCE(trajectory_doc_id, %s),
                                trajectory_path    = COALESCE(trajectory_path, %s)
                            WHERE plan_id = %s AND scenario_id = %s
                            """,
                            (doc_id, source_path, plan_id, scenario_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE query_runs
                            SET trajectory_doc_id = COALESCE(trajectory_doc_id, %s),
                                trajectory_path    = COALESCE(trajectory_path, %s)
                            WHERE plan_id = %s
                            """,
                            (doc_id, source_path, plan_id),
                        )

                # ---------- 3) trajectory[] → 各テーブル ----------
                for task in traj_list:
                    task_number      = task.get("task_number")
                    task_description = task.get("task_description") or ""
                    agent_name       = task.get("agent_name") or ""
                    response         = task.get("response") or ""
                    final_answer     = task.get("final_answer") or ""
                    raw_task_json    = task

                    # 3-a) traj_tasks
                    cur.execute(
                        """
                        INSERT INTO traj_tasks
                          (doc_id, task_number, task_description, agent_name, response, final_answer, raw_task_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING task_id
                        """,
                        (doc_id, task_number, task_description, agent_name, response, final_answer, Json(raw_task_json)),
                    )
                    task_id = cur.fetchone()["task_id"]

                    if embed_text:
                        try:
                            tv = embed_text(" ".join([task_description, agent_name, response, final_answer]).strip())
                        except Exception:
                            tv = None
                        if tv:
                            cur.execute(
                                "UPDATE traj_tasks SET task_vec = CAST(%s AS vector(1536)) WHERE task_id = %s",
                                (_as_vector_literal(tv), task_id),
                            )

                    # 3-b) logs (1:1)
                    logs = task.get("logs") or {}
                    cur.execute(
                        """
                        INSERT INTO traj_logs
                          (task_id, type, task, environment, system_prompt,
                           demonstration, scratchpad, endstate, raw_logs_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING log_id
                        """,
                        (
                            task_id,
                            logs.get("type"),
                            logs.get("task"),
                            logs.get("environment"),
                            logs.get("system_prompt"),
                            logs.get("demonstration"),
                            logs.get("scratchpad"),
                            logs.get("endstate"),
                            Json(logs),
                        ),
                    )
                    log_id = cur.fetchone()["log_id"]

                    if embed_text:
                        try:
                            lv = embed_text(
                                " ".join([
                                    str(logs.get("type","")), str(logs.get("task","")), str(logs.get("environment","")),
                                    str(logs.get("system_prompt","")), str(logs.get("demonstration","")),
                                    str(logs.get("scratchpad","")), str(logs.get("endstate",""))
                                ]).strip()
                            )
                        except Exception:
                            lv = None
                        if lv:
                            cur.execute(
                                "UPDATE traj_logs SET log_vec = CAST(%s AS vector(1536)) WHERE log_id = %s",
                                (_as_vector_literal(lv), log_id),
                            )

                    # 3-b-1) logs.trajectory_log[] / trajectroy_log[] → traj_log_steps
                    steps = logs.get("trajectory_log") or logs.get("trajectroy_log") or []
                    if steps:
                        step_params = []
                        for st in steps:
                            step_params.append((
                                log_id,
                                st.get("step"),
                                st.get("raw_llm_thought_output"),
                                st.get("raw_llm_action_output"),
                                st.get("raw_observation_output"),
                                st.get("raw_llm_output"),
                                st.get("thought"),
                                st.get("action"),
                                st.get("action_input"),
                                st.get("observation"),
                                st.get("state"),
                                st.get("is_loop_detected"),
                                st.get("additional_scratchpad_feedback"),
                                st.get("step_trajectory_file_name"),
                                st.get("step_metric_file_name"),
                                Json(st.get("step_trajectory_json")) if st.get("step_trajectory_json") is not None else None,
                                Json(st.get("step_metric_json")) if st.get("step_metric_json") is not None else None,
                            ))
                        cur.executemany(
                            """
                            INSERT INTO traj_log_steps
                              (log_id, step, raw_llm_thought_output, raw_llm_action_output, raw_observation_output,
                               raw_llm_output, thought, action, action_input, observation, state, is_loop_detected,
                               additional_scratchpad_feedback, step_trajectory_file_name, step_metric_file_name,
                               step_trajectory_json, step_metric_json)
                            VALUES
                              (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            step_params,
                        )

                    # 3-b-2) logs.history[] → executemany
                    history = logs.get("history") or []
                    if history:
                        hist_params = []
                        for idx, h in enumerate(history):
                            hist_params.append((
                                log_id,
                                h.get("idx", idx),
                                h.get("role"),
                                h.get("content"),
                                h.get("agent"),
                                h.get("is_demo"),
                            ))
                        cur.executemany(
                            """
                            INSERT INTO traj_log_history
                              (log_id, idx, role, content, agent, is_demo)
                            VALUES (%s,%s,%s,%s,%s,%s)
                            """,
                            hist_params,
                        )

                    # 3-b-3) logs.trajectory[] (inner) → executemany
                    inner_traj = logs.get("trajectory") or []
                    if inner_traj:
                        inner_params = []
                        for idx, it in enumerate(inner_traj):
                            inner_params.append(
                                (log_id, idx, it.get("thought"), it.get("action"), it.get("observation"))
                            )
                        cur.executemany(
                            """
                            INSERT INTO traj_log_inner_trajectory
                              (log_id, idx, thought, action, observation)
                            VALUES (%s,%s,%s,%s,%s)
                            """,
                            inner_params,
                        )

                    # 3-c) info.model_stats
                    info = task.get("info") or {}
                    model_stats = info.get("model_stats") or {}
                    if model_stats:
                        cur.execute(
                            """
                            INSERT INTO traj_info_model_stats
                              (task_id, tokens_sent, tokens_received, api_calls, total_cost, instance_cost, raw_info_json)
                            VALUES (%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                task_id,
                                model_stats.get("tokens_sent"),
                                model_stats.get("tokens_received"),
                                model_stats.get("api_calls"),
                                model_stats.get("total_cost"),
                                model_stats.get("instance_cost"),
                                Json(info),
                            ),
                        )

                    # 3-d) reviews / reflections
                    for i, rv in enumerate(task.get("reviews") or []):
                        cur.execute(
                            "INSERT INTO traj_reviews (task_id, idx, text) VALUES (%s,%s,%s)",
                            (task_id, i, rv if isinstance(rv, str) else str(rv)),
                        )
                    for i, rf in enumerate(task.get("reflections") or []):
                        cur.execute(
                            "INSERT INTO traj_reflections (task_id, idx, text) VALUES (%s,%s,%s)",
                            (task_id, i, rf if isinstance(rf, str) else str(rf)),
                        )

        # with conn.transaction() 正常終了で自動 commit、with conn で自動 close
        return str(doc_id)

import psycopg
from psycopg.rows import dict_row

def save_dag_traj_score(
    db_url: str,
    *,
    plan_id: str,
    doc_id: str,
    score: dict,  # {num_edit:int, correct:float[0..1], num_partially:int, num_not:int, error_analysis:str}
) -> dict:
    """
    dag_trajectory_score へ UPSERT。保存した行を dict で返す。
    """
    # --- normalize & guardrails ---
    num_edit      = int(score.get("num_edit", 0) or 0)
    correct       = float(score.get("correct", 0.0) or 0.0)
    correct       = 0.0 if correct < 0.0 else (1.0 if correct > 1.0 else correct)  # NUMERIC(4,3), CHECK 0..1
    num_partially = int(score.get("num_partially", 0) or 0)
    num_not       = int(score.get("num_not", 0) or 0)
    error_analysis = (score.get("error_analysis") or "")[:20000]

    sql = """
    INSERT INTO dag_trajectory_score
      (plan_id, doc_id, num_edit, correct, num_partially, num_not, error_analysis)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (plan_id, doc_id)
    DO UPDATE SET
      num_edit      = EXCLUDED.num_edit,
      correct       = EXCLUDED.correct,
      num_partially = EXCLUDED.num_partially,
      num_not       = EXCLUDED.num_not,
      error_analysis= EXCLUDED.error_analysis
    RETURNING score_id, plan_id, doc_id, num_edit, correct, num_partially, num_not, error_analysis, created_at;
    """

    params = (
        plan_id, doc_id, num_edit, correct, num_partially, num_not, error_analysis
    )

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.commit()
    return row


def load_scenarios(utterance_ids):
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    train_ds = ds["train"]
    df = train_ds.to_pandas()

    filtered_df = df[df["id"].isin(utterance_ids)]

    return filtered_df.to_dict(orient="records")


def run_planning_workflow(
        question, qid, llm_model=16, generate_steps_only=False
):
    iot_r_agent = ReactReflectAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
        task_examples=iot_task_examples,
        reflect_step=1,
    )

    fmsr_r_agent = ReactReflectAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
        reflect_step=1,
    )

    tsfm_rr_agent = ReactReflectAgent(
        name=tsfm_agent_name,
        description=tsfm_agent_description,
        tools=tsfm_tools,
        llm=llm_model,
        few_shots=tsfm_fewshots,
        task_examples=tsfm_task_examples,
        reflect_step=1,
    )
    
    wo_rr_agent = WorderOrderAgent(
        name=wo_agent_name,
        description=wo_agent_description,
        tools=wo_tools,
        llm=llm_model,
        few_shots=wo_fewshots,
        reflect_step=1,
        task_examples=wo_task_examples,
    )

    task = Task(
        description=question,
        expected_output="",
        agents=[iot_r_agent, fmsr_r_agent, tsfm_rr_agent, wo_rr_agent],
    )

    wf = NewPlanningWorkflow(
        tasks=[task],
        llm=llm_model,
    )

    if generate_steps_only:
        os.makedirs(PLAN_DIR, exist_ok=True)

        return wf.generate_steps(
            save_plan=True,
            saved_plan_filename=RESULT_DIR + f"Model_{llm_model}_Q_{qid}_plan",
        )
    
    history, run_id, plan_id = wf.run(qid=qid)

    return history, run_id, plan_id

def run(utterances, generate_steps_only=False):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)

    for utterance in utterances:
        logger.info("=" * 10)
        logger.info(f"ID: {utterance['id']}, Task: {utterance['text']}")
        dag_file = f"{PLAN_DIR}Q_{utterance['id']}_finalplan.json"
        trajectory_file = f"{TRAJECTORY_DIR}Q_{utterance['id']}_trajectory.json"

        ans, run_id, plan_id = run_planning_workflow(
            utterance["text"],
            utterance["id"],
            generate_steps_only=generate_steps_only,
        )

        if generate_steps_only:
            continue

        output = {"id": utterance["id"], "text": utterance["text"], "trajectory": ans}

        doc_id = save_full_trajectory(
            db_url=os.environ["DATABASE_URL"],
            output=output,                             
            source_path=trajectory_file,                
            run_id=run_id,                        
            plan_id=plan_id,                     
            scenario_id=0,
            embed_text=None,                  
        )
        print("saved doc_id:", doc_id)

        with open(trajectory_file, "w") as f:
            json.dump(output, f, indent=4)

        score_result = auto_score_dag_trajectory(
            dag_path=dag_file,
            traj_path=trajectory_file,
            human_scored_triplets=human_scored_triplets,
            model_id=16,
        )
        print(json.dumps(score_result, ensure_ascii=False, indent=2))

        saved_row = save_dag_traj_score(
            db_url=os.environ["DATABASE_URL"],
            plan_id=plan_id,
            doc_id=doc_id,
            score=score_result["Q"],   # ← num_edit / correct / num_partially / num_not / error_analysis
        )
        print("saved score_id:", saved_row["score_id"])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_ids", type=str, default="1,106")
    parser.add_argument("--generate_steps_only", type=bool, default=False)

    args = parser.parse_args()
    utterance_ids = [int(uid.strip()) for uid in args.utterance_ids.split(",")]
    utterances = load_scenarios(utterance_ids)

    run(
        utterances,
        generate_steps_only=args.generate_steps_only,
    )