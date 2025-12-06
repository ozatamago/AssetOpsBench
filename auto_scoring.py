# auto_scoring.py
# Purpose:
#   (2) 自動スコアリング関数：human-scored * 10 + 対象 DAG/trajectory（要約）を投げて
#       {num_edit, correct, num_partially, num_not, error_analysis} を返す
#
# Requirements satisfied:
#   i) 既存の要約関数で trajectory を短縮（summarize_pipeline を想定）
#   ii) {human-scored triplet * 10} + {DAG, summarized trajectory} を 1 つの Prompt に
#   iii) LLM は {num_edit, correct, error_analysis} だけ JSON で出力
#        -> num_partially, num_not は error_analysis から regex 集計

import os
import re
import json
from typing import List, Dict, Any, Optional

from reactxen.utils.model_inference import watsonx_llm

# 既存のサマリ関数（あなたが作った 1 の関数群）を利用
# summarize_pipeline.py などに置いてある想定です
try:
    from summarize import extract_trajectory_fields, summarise_all_fields
except ImportError:
    # フォールバック（最小限）：text/first-task の抜粋だけ
    def extract_trajectory_fields(json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = []
        for t in data.get("trajectory", []):
            logs = t.get("logs", {}) or {}
            review = ""
            rv = logs.get("reviews")
            if isinstance(rv, list) and rv: review = rv[0]
            elif isinstance(rv, str): review = rv
            tasks.append({
                "task_description": t.get("task_description",""),
                "agent_name": t.get("agent_name",""),
                "final_answer": logs.get("final_answer",""),
                "review": review,
            })
        return {"text": data.get("text",""), "tasks": tasks}

    def summarise_all_fields(extracted: dict, model_id: int = 16) -> dict:
        # 何もしない（最小フォールバック）
        return extracted

# --------- helpers ---------
def _read_json_minified(path: str, max_chars: int = 6000) -> str:
    """DAG など巨大 JSON をプロンプトに載せるために minify ＆上限カット。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s[:max_chars]

def _summarize_trajectory_for_scoring(traj_path: str, model_id: int = 16) -> dict:
    """(i) 既存の要約関数を用いて trajectory を要約（text はそのまま）。"""
    extracted = extract_trajectory_fields(traj_path)
    summary = summarise_all_fields(extracted, model_id=model_id)
    return summary

# ---- replace your existing _format_examples_for_prompt with: ----
def _format_examples_for_prompt(human_scored: list) -> str:
    """
    Expect normalized list of dicts with keys: dag, trajectory_summary, score{num_edit, correct, error_analysis}
    """
    examples = _normalize_human_scored_triplets(human_scored)
    if not examples:
        raise ValueError("human_scored_triplets is empty or malformed")

    blocks = []
    for ex in examples[:10]:
        dag = ex.get("dag", "")
        traj = ex.get("trajectory_summary", "")
        sc = ex.get("score", {}) or {}
        sc_json = json.dumps({
            "num_edit": sc.get("num_edit", 0),
            "correct": sc.get("correct", 0),
            "error_analysis": sc.get("error_analysis", "")
        }, ensure_ascii=False)
        block = (
            "<example>\n"
            "<dag>\n" + str(dag) + "\n</dag>\n"
            "<trajectory>\n" + str(traj) + "\n</trajectory>\n"
            "<gold_score_json>\n" + sc_json + "\n</gold_score_json>\n"
            "</example>\n"
        )
        blocks.append(block)
    return "<examples>\n" + "".join(blocks) + "</examples>"

def _build_scoring_prompt(
    dag_json_min: str,
    traj_summary: dict,
    human_scored_examples_block: str
) -> str:
    return (
        "You are an impartial evaluator for planning DAGs before execution.\n"
        "Judge the candidate DAG using the summarized trajectory context.\n"
        "Follow the rubric:\n"
        "- num_edit: Non-negative integer; minimal edits required to make the DAG executable and aligned with the task.\n"
        "- correct: Non-negative integer; count of fully correct steps/decisions (not partial).\n"
        "- error_analysis: A short bullet list, one issue per line. Each line MUST start with exactly one of\n"
        "  [OK] or [PARTIAL] or [NOT], followed by a brief reason (≤ 15 words).\n"
        "Return ONLY a JSON object with keys: num_edit, correct, error_analysis (string with multiple lines).\n"
        "Do NOT include num_partially or num_not in the JSON (they will be computed downstream).\n"
        "No preamble, no explanation, no markdown.\n\n"
        + human_scored_examples_block + "\n\n"
        "<candidate>\n"
        "<dag_json>\n" + dag_json_min + "\n</dag_json>\n"
        "<trajectory_summary>\n" + json.dumps(traj_summary, ensure_ascii=False) + "\n</trajectory_summary>\n"
        "</candidate>\n\n"
        "Respond with ONLY this JSON (no extra keys, no trailing text):\n"
        '{ "num_edit": int (num_edit can take negative value), "correct": float (from 0.0 to 1.0), "error_analysis": "<lines starting with [OK]/[PARTIAL]/[NOT]>" }'
    )

def _parse_llm_json(s: str) -> Dict[str, Any]:
    """LLM からの JSON を堅牢にパース（前後のノイズ除去）。"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise
import re
from typing import Dict, Any, Iterable

# 事前コンパイル（繰り返し呼ぶなら高速）
_PATTERNS = {
    "num_partially": re.compile(r"Task Status:\s*Partially Accomplished\b"),
    "num_not":       re.compile(r"Task Status:\s*Not Accomplished\b"),
    # 必要なら達成件数もカウント可能
    # "num_accomplished": re.compile(r"Task Status:\s*Accomplished\b"),
}

def _iter_review_strings(obj: Any) -> Iterable[str]:
    """
    traj_summary（dict / list / str）を再帰的に走査し、文字列だけ取り出す。
    'review' キーを優先しつつ、他の文字列も保険で拾う。
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "review" and isinstance(v, str):
                yield v
            else:
                yield from _iter_review_strings(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_review_strings(x)
    # それ以外（int/None等）は無視

def _count_tags(review_source: Any) -> Dict[str, int]:
    """
    review_source が dict（traj_summary）でも str でも OK。
    全ての文字列を結合してから厳密一致でカウント。
    """
    corpus = "\n".join(_iter_review_strings(review_source))
    return {key: len(pat.findall(corpus)) for key, pat in _PATTERNS.items()}



# ---- add this near the top of auto_scoring.py ----
def _normalize_human_scored_triplets(human_scored_triplets) -> list[dict]:
    """
    Accepts:
      - list[dict]                  (ideal)
      - list[str JSON], str JSON    (parse with json.loads)
      - path to .json / .jsonl      (load; .jsonl -> per-line JSON)
      - plain str                   (coerce into minimal dict)
    Returns: list[dict] with keys: dag, trajectory_summary, score
    """
    def _to_list(obj):
        return obj if isinstance(obj, list) else [obj]

    data = human_scored_triplets

    # If it's a path or a raw JSON string
    if isinstance(data, str):
        if os.path.exists(data):
            # file path
            if data.lower().endswith(".jsonl"):
                items = []
                with open(data, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            # skip malformed line
                            continue
                data = items
            else:
                with open(data, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        # if file is plain text JSON lines
                        f.seek(0)
                        data = [json.loads(line) for line in f if line.strip()]
        else:
            # try to parse as JSON string
            try:
                data = json.loads(data)
            except Exception:
                # fall back to a single minimal dict carrying the text
                data = [{"dag": "", "trajectory_summary": "",
                         "score": {"num_edit": 0, "correct": 0, "error_analysis": data}}]

    out: list[dict] = []
    for item in _to_list(data):
        # if item is a JSON string, parse to dict
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                out.append({"dag": "", "trajectory_summary": "",
                            "score": {"num_edit": 0, "correct": 0, "error_analysis": item}})
                continue

        if isinstance(item, dict):
            # normalize keys (case-insensitive)
            lower = {str(k).lower(): v for k, v in item.items()}
            dag  = lower.get("dag", "")
            traj = lower.get("trajectory_summary", lower.get("trajectory", ""))
            score = lower.get("score", {})
            # score might be stringified JSON
            if isinstance(score, str):
                try:
                    score = json.loads(score)
                except Exception:
                    score = {"num_edit": 0, "correct": 0, "error_analysis": score}
            # ensure minimal fields exist
            score = {
                "num_edit": int(score.get("num_edit", 0)) if isinstance(score, dict) else 0,
                "correct": int(score.get("correct", 0)) if isinstance(score, dict) else 0,
                "error_analysis": (score.get("error_analysis", "") if isinstance(score, dict) else "")
            }
            out.append({"dag": dag, "trajectory_summary": traj, "score": score})
        # non-dict, non-str -> skip silently
    return out

# --------- main scoring function (②) ---------
def auto_score_dag_trajectory(
    dag_path: str,
    traj_path: str,
    human_scored_triplets: List[Dict[str, Any]],
    model_id: int = 16,
) -> Dict[str, Any]:
    
    traj_summary = _summarize_trajectory_for_scoring(traj_path, model_id=model_id)
    dag_min = _read_json_minified(dag_path, max_chars=6000)
    ex_block = _format_examples_for_prompt(human_scored_triplets)

    prompt = _build_scoring_prompt(dag_min, traj_summary, ex_block)
    print(f"prompt: {prompt}")

    llm_text = (watsonx_llm(prompt, model_id=model_id) or {}).get("generated_text", "").strip()
    print(f"llm_text: {llm_text}")

    try:
        parsed = _parse_llm_json(llm_text)
        print(f"parsed: {parsed}")
    except Exception as e:
        parsed = {"num_edit": 0, "correct": 0, "error_analysis": ""}

    tags = _count_tags(traj_summary)

    result = {
        "Q": {
            "num_edit": int(parsed.get("num_edit", 0) or 0),
            "correct": int(parsed.get("correct", 0) or 0),
            "num_partially": int(tags["num_partially"]),
            "num_not": int(tags["num_not"]),
            "error_analysis": parsed.get("error_analysis", ""),
        },
        "raw_llm": llm_text,
    }
    return result

# RESULT_DIR = "/home/track1_result/"
# PLAN_DIR = RESULT_DIR + "plan/"
# TRAJECTORY_DIR = RESULT_DIR + "trajectory/"
# SUMMARY_DIR = RESULT_DIR + "summary/"

# dag_path=f"{PLAN_DIR}Q_410_finalplan.json"
# traj_path=f"{TRAJECTORY_DIR}Q_410_trajectory.json"
# human_scored_triplets= """
# Q6
# dag: 
# {
#   "ok": true,
#   "final_plan": " \n#Task1: List the assets at the MAIN site\n#Agent1: IoT Data Download\n#Dependency1: None\n#ExpectedOutput1: List of assets at the MAIN site\n#Task2: Verify if Chiller 9 is present in the list of assets at the MAIN site\n#Agent2: IoT Data Download\n#Dependency2: #S1\n#ExpectedOutput2: Confirmation that Chiller 9 is among the assets listed\n#Task3: Retrieve asset details for Chiller 9 at the MAIN site\n#Agent3: IoT Data Download\n#Dependency3: #S2\n#ExpectedOutput3: Asset details for Chiller 9",
#   "scenario_id": 6,
#   "generated_at": "2025-10-19T15:58:17.539122Z"
# }
# trajectory: 
# {
#   "text": "Get the asset details for Chiller 9 at the MAIN site.",
#   "tasks": [
#     {
#       "task_description": "List assets at MAIN site",
#       "agent_name": "IoT Data Download",
#       "final_answer": "Question</response>",
#       "review": "Task Status: Accomplished | executed 'assets' tool successfully | avoid redundant actions</response>\n```<response>executed 'assets' tool successfully | avoid redundant actions</response>```"
#     },
#     {
#       "task_description": "Check if Chiller 9 is in MAIN site assets",
#       "agent_name": "IoT Data Download",
#       "final_answer": "MAIN</response>",
#       "review": "Task Status: Accomplished | read JSON file correctly | none</response>"
#     },
#     {
#       "task_description": "Fetch Chiller 9 asset details from MAIN.",
#       "agent_name": "IoT Data Download",
#       "final_answer": "MAIN</response>",
#       "review": "Task Status: Partially Accomplished | identified relevant JSON file | extract Chiller 9 details from JSON file."
#     }
#   ]
# }
# score: 
# Q6
# - num_edit: 0
# - correct: 0.4
# - num_partially: 1
# - num_not: 0
# - error_analysis: Symptom 1: Final answer lacks required details
# The response returns only a file path and even appends stray boilerplate (“Here’s the rewritten response… Question”), instead of extracting and presenting Chiller 9’s actual fields (e.g., id, name, site, metadata). Fix: After obtaining the file path, call jsonreader, filter for “Chiller 9”, and surface the concrete fields in the final answer. Keep the final answer clean and free of internal notes. Symptom 2: Action-input contamination A tool call was issued with site_name=MAIN ## Step 5: Analyze the result ..., causing unknown site "MAIN ## Step 5...". This is classic argument leakage from thoughts/explanations into the action payload. Fix: Enforce strict tool-call hygiene: arguments must be structured inputs only (no commentary). Many tool-use guides emphasize separating “reasoning” from “tool arguments” and keeping final outputs free of scaffolding text. Symptom 3: Redundant & brittle calls The agent repeats the assets call after already getting a valid file, then errors. Fix: Cache observations and avoid repeated identical calls unless inputs change. Transition immediately from “list” → “read/parse” → “filter” → “final answer.” This reduces surface area for failures and aligns with robust ReAct-style loops. Symptom 4: Improper output handling The final message includes scaffolding text (“Question”) and references to internal formatting, which is a form of improper output handling in agent pipelines. Fix: Post-process the final answer to strip scaffolding and ensure only the user-facing result remains (a practice echoed in LLM agent risk/best-practice write-ups). 

# Q410
# dag: 
# {
#   "ok": true,
#   "final_plan": " \n#Task1: Retrieve the events of equipment CWC04009 for the first week of June 2020.\n#Agent1: IoT Data Download\n#Dependency1: None\n#ExpectedOutput1: A JSON file containing the events of equipment CWC04009 for the first week of June 2020.\n#Task2: Analyze the retrieved events and provide a summary based on the event group for work order event, alert, and anomaly.\n#Agent2: WorkOrder Agent\n#Dependency2: #S1\n#ExpectedOutput2: A summary in JSON format containing the count of work order events, alerts, and anomalies for equipment CWC04009 for the first week of June 2020.\n#Task3: Verify if the summary contains the required information for work order event, alert, and anomaly.\n#Agent3: IoT Data Download\n#Dependency3: #S2\n#ExpectedOutput3: Confirmation if the summary contains the required information.",
#   "scenario_id": 410,
#   "generated_at": "2025-10-19T19:06:59.745523Z"
# }
# trajectory: 
# {
#   "text": "Get all the events of equipment CWC04009 for the first week of June of 2020 and provide a summary based on the event group for work order event, alert, and anomaly.",
#   "tasks": [
#     {
#       "task_description": "Fetch CWC04009 events for June 1-7 2020.",
#       "agent_name": "IoT Data Download",
#       "final_answer": "MAIN</response>",
#       "review": "Task Status: Accomplished | followed logical sequence | provide detailed error handling</response>\n```<response>followed logical sequence | provide detailed error handling</response>```"
#     },
#     {
#       "task_description": "Group work order events, alerts, and anomalies by category.",
#       "agent_name": "WorkOrder Agent",
#       "final_answer": "{\"CWC04009\": {\"ALERT\": 6}} </response>\nbecomes \n{\"CWC04009\": {\"ALERT\": 6}}",
#       "review": "Task Status: Accomplished | retrieved events correctly | none</response>\n<response>retrieved events correctly | none</response>"
#     },
#     {
#       "task_description": "Check work order event alert and anomaly details in summary.",
#       "agent_name": "IoT Data Download",
#       "final_answer": "",
#       "review": "Task Status: Not Accomplished | failed to verify summary | ensure summary file is generated or available</response>\n</to-solve>"
#     }
#   ]
# }
# score: 
# Q410
# num_edit: 1
# correct: 0.0
# num_partially: 0
# num_not: 1
# error_analysis: Task 1 skipped asset ID/site reconciliation, prematurely assumed “absent at MAIN,” and failed to generate the event JSON. Task 2 aggregated only ALERTs (6 items); WO/ANOM were missing, and the result is inconsistent with Task 1’s conclusion. Task 3 was not executed (no validation performed). Add a preliminary task for “equipment ID / alias / site resolution,” ensure consistency by analyzing from a single data source, then have the verification agent check for requirement compliance.
# """
# score = auto_score_dag_trajectory(dag_path, traj_path, human_scored_triplets, model_id=16)
# print(json.dumps(score, ensure_ascii=False, indent=2))
