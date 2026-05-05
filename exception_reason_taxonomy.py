#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# Config
# ============================================================

TARGET_STATUSES = {
    "partially accomplished",
    "not accomplished",
    "error",
}

DEFAULT_MODEL_ID = 20
DEFAULT_BATCH_SIZE = 10
DEFAULT_MERGE_GROUP_SIZE = 5

MAX_TASK_DESC_CHARS = 1800
MAX_RESPONSE_CHARS = 2200
MAX_FINAL_ANSWER_CHARS = 2200
MAX_REVIEW_CHARS = 2200

LLM_MAX_RETRIES = 3
LLM_RETRY_SLEEP = 2.0


# ============================================================
# Basic utils
# ============================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u0000", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        safe_row = {}
        for k in fieldnames:
            v = row.get(k, "")
            if isinstance(v, (list, dict)):
                safe_row[k] = json.dumps(v, ensure_ascii=False)
            else:
                safe_row[k] = v
        writer.writerow(safe_row)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def group_list(xs: List[Any], group_size: int) -> List[List[Any]]:
    return [xs[i:i + group_size] for i in range(0, len(xs), group_size)]


def load_processed_case_ids(jsonl_path: Path) -> set:
    ids = set()
    for row in load_jsonl(jsonl_path):
        cid = row.get("case_id")
        if cid:
            ids.add(cid)
    return ids


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = normalize_text(x).lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.rglob("*.json")):
        if p.is_file():
            yield p


# ============================================================
# Review parsing
# ============================================================

def normalize_status(status: str) -> str:
    s = normalize_text(status).lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_review_status(review: str) -> str:
    r = normalize_text(review)

    m = re.search(r"Task Status:\s*([A-Za-z _-]+)", r, flags=re.I)
    if m:
        return normalize_status(m.group(1))

    first_line = r.splitlines()[0] if r else ""
    first_line_norm = normalize_status(first_line)
    for s in ["accomplished", "partially accomplished", "not accomplished", "error"]:
        if first_line_norm.startswith(s):
            return s

    full_norm = normalize_status(r)
    for s in ["partially accomplished", "not accomplished", "error", "accomplished"]:
        if s in full_norm:
            return s

    return ""


def extract_review_reasoning(review: str) -> str:
    r = normalize_text(review)
    m = re.search(
        r"Reasoning:\s*(.*?)(?:Suggestions for Improvement:|$)",
        r,
        flags=re.I | re.S,
    )
    if m:
        return normalize_text(m.group(1))
    return r


def looks_like_context_length_error(text: str) -> bool:
    t = normalize_text(text).lower()
    patterns = [
        "context length",
        "input length",
        "maximum context length",
        "too many tokens",
        "token limit",
        "exceeds the maximum",
        "prompt is too long",
        "input is too long",
        "request too large",
        "max tokens",
    ]
    return any(p in t for p in patterns)


def collect_reviews(task: Dict[str, Any]) -> List[str]:
    task_reviews = task.get("reviews")
    if isinstance(task_reviews, list) and task_reviews:
        return [normalize_text(x) for x in task_reviews if normalize_text(x)]

    logs = task.get("logs") or {}
    log_reviews = logs.get("reviews")
    if isinstance(log_reviews, list) and log_reviews:
        return [normalize_text(x) for x in log_reviews if normalize_text(x)]

    return []


# ============================================================
# Scenario iteration
# ============================================================

def iter_scenarios_from_json_obj(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict) and "trajectory" in obj:
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "trajectory" in item:
                yield item


# ============================================================
# Debug scan
# ============================================================

DEBUG_FILE_SCAN_FIELDS = [
    "source_file",
    "parse_ok",
    "top_level_type",
    "scenario_count",
    "trajectory_scenario_count",
    "task_count",
    "review_count",
    "status_histogram",
    "example_statuses",
    "example_review_prefixes",
    "notes",
]

DEBUG_STATUS_HIST_FIELDS = [
    "status",
    "count",
]

FAILURE_CASE_FIELDNAMES = [
    "case_id",
    "source_file",
    "scenario_id",
    "scenario_text",
    "task_number",
    "agent_name",
    "status",
    "task_description",
    "response",
    "final_answer",
    "review",
    "review_reasoning",
    "unmet_requirements",
    "reason_summary",
    "root_cause_category",
    "planner_branch_worth_adding",
    "verification_signal",
    "recommended_branch_pattern",
    "llm_used",
]

TAXONOMY_FIELDNAMES = [
    "taxonomy_id",
    "round_index",
    "group_index",
    "category_name",
    "definition",
    "typical_statuses",
    "failure_signals",
    "unmet_requirement_patterns",
    "planner_response",
    "branch_condition_template",
    "example_case_ids",
]


def scan_input_dir_debug(
    input_dir: Path,
    output_dir: Path,
    debug: bool = False,
    debug_sample_limit: int = 50,
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    file_scan_csv = output_dir / "debug_file_scan.csv"
    status_hist_csv = output_dir / "debug_status_histogram.csv"
    sample_reviews_jsonl = output_dir / "debug_sample_reviews.jsonl"
    report_json = output_dir / "debug_report.json"

    for p in [file_scan_csv, status_hist_csv, sample_reviews_jsonl]:
        if p.exists():
            p.unlink()

    all_json_files = list(iter_json_files(input_dir))
    global_status_counter = Counter()
    global_notes_counter = Counter()

    total_files = 0
    parse_failed_files = 0
    total_scenarios = 0
    total_trajectory_scenarios = 0
    total_tasks = 0
    total_reviews = 0
    sample_review_count = 0

    if debug:
        print(f"[debug] input_dir={input_dir}", flush=True)
        print(f"[debug] found {len(all_json_files)} json files", flush=True)
        for p in all_json_files[:10]:
            print(f"[debug] file: {p}", flush=True)

    for file_path in all_json_files:
        total_files += 1
        row = {
            "source_file": str(file_path),
            "parse_ok": True,
            "top_level_type": "",
            "scenario_count": 0,
            "trajectory_scenario_count": 0,
            "task_count": 0,
            "review_count": 0,
            "status_histogram": {},
            "example_statuses": [],
            "example_review_prefixes": [],
            "notes": [],
        }

        try:
            data = load_json(file_path)
        except Exception as e:
            parse_failed_files += 1
            row["parse_ok"] = False
            row["notes"] = [f"json_parse_error: {type(e).__name__}: {e}"]
            append_csv(file_scan_csv, row, DEBUG_FILE_SCAN_FIELDS)
            global_notes_counter["json_parse_error"] += 1
            if debug:
                print(f"[debug][parse-fail] {file_path}: {type(e).__name__}: {e}", flush=True)
            continue

        row["top_level_type"] = type(data).__name__

        if isinstance(data, dict):
            row["scenario_count"] = 1
            total_scenarios += 1
        elif isinstance(data, list):
            row["scenario_count"] = len(data)
            total_scenarios += len(data)
        else:
            row["notes"].append("top_level_not_dict_or_list")
            global_notes_counter["top_level_not_dict_or_list"] += 1

        local_status_counter = Counter()
        example_statuses = []
        example_review_prefixes = []

        scenario_iter = list(iter_scenarios_from_json_obj(data))
        row["trajectory_scenario_count"] = len(scenario_iter)
        total_trajectory_scenarios += len(scenario_iter)

        if len(scenario_iter) == 0:
            row["notes"].append("no_trajectory_scenarios_found")
            global_notes_counter["no_trajectory_scenarios_found"] += 1

        for scenario in scenario_iter:
            trajectory = scenario.get("trajectory")
            if not isinstance(trajectory, list):
                row["notes"].append("trajectory_not_list")
                global_notes_counter["trajectory_not_list"] += 1
                continue

            row["task_count"] += len(trajectory)
            total_tasks += len(trajectory)

            for task in trajectory:
                reviews = collect_reviews(task)
                row["review_count"] += len(reviews)
                total_reviews += len(reviews)

                if not reviews:
                    global_notes_counter["task_without_reviews"] += 1

                for review in reviews:
                    status = extract_review_status(review)
                    if status:
                        local_status_counter[status] += 1
                        global_status_counter[status] += 1
                        if status not in example_statuses:
                            example_statuses.append(status)
                    else:
                        local_status_counter["<unparsed>"] += 1
                        global_status_counter["<unparsed>"] += 1
                        global_notes_counter["unparsed_review_status"] += 1

                    prefix = normalize_text(review)[:240]
                    if prefix and len(example_review_prefixes) < 3:
                        example_review_prefixes.append(prefix)

                    if sample_review_count < debug_sample_limit:
                        append_jsonl(
                            sample_reviews_jsonl,
                            {
                                "source_file": str(file_path),
                                "scenario_id": scenario.get("id", ""),
                                "task_number": task.get("task_number", ""),
                                "agent_name": normalize_text(task.get("agent_name", "")),
                                "task_description": normalize_text(task.get("task_description", "")),
                                "status_parsed": status,
                                "review_prefix": prefix,
                                "review_full": review,
                            },
                        )
                        sample_review_count += 1

        row["status_histogram"] = dict(local_status_counter)
        row["example_statuses"] = example_statuses
        row["example_review_prefixes"] = example_review_prefixes

        if row["review_count"] == 0:
            row["notes"].append("no_reviews_found_in_file")
            global_notes_counter["no_reviews_found_in_file"] += 1

        append_csv(file_scan_csv, row, DEBUG_FILE_SCAN_FIELDS)

        if debug:
            print(
                f"[debug][scan] file={file_path.name} "
                f"scenarios={row['trajectory_scenario_count']} "
                f"tasks={row['task_count']} "
                f"reviews={row['review_count']} "
                f"statuses={dict(local_status_counter)} "
                f"notes={row['notes']}",
                flush=True,
            )

    for status, count in global_status_counter.items():
        append_csv(status_hist_csv, {"status": status, "count": count}, DEBUG_STATUS_HIST_FIELDS)

    debug_report = {
        "input_dir": str(input_dir),
        "total_json_files": total_files,
        "parse_failed_files": parse_failed_files,
        "total_scenarios_seen": total_scenarios,
        "total_trajectory_scenarios": total_trajectory_scenarios,
        "total_tasks": total_tasks,
        "total_reviews": total_reviews,
        "global_status_histogram": dict(global_status_counter),
        "global_notes_histogram": dict(global_notes_counter),
        "target_statuses": sorted(TARGET_STATUSES),
        "matching_target_status_count": sum(global_status_counter.get(s, 0) for s in TARGET_STATUSES),
    }
    write_json(report_json, debug_report)

    if debug:
        print("[debug] scan summary:")
        print(json.dumps(debug_report, indent=2, ensure_ascii=False), flush=True)

    return debug_report


# ============================================================
# Failure case extraction
# ============================================================

def iter_failure_cases(input_dir: Path, debug: bool = False) -> Iterable[Dict[str, Any]]:
    for file_path in iter_json_files(input_dir):
        try:
            data = load_json(file_path)
        except Exception as e:
            if debug:
                print(f"[debug][iter_failure_cases][parse-fail] {file_path}: {type(e).__name__}: {e}", flush=True)
            yield {
                "case_id": f"{file_path.stem}__file_error",
                "source_file": str(file_path),
                "scenario_id": "",
                "scenario_text": "",
                "task_number": "",
                "agent_name": "",
                "status": "error",
                "task_description": "",
                "response": "",
                "final_answer": "",
                "review": f"Error: failed to parse json file. {type(e).__name__}: {e}",
                "review_reasoning": f"Error: failed to parse json file. {type(e).__name__}: {e}",
            }
            continue

        scenario_count = 0
        for scenario in iter_scenarios_from_json_obj(data):
            scenario_count += 1
            scenario_id = scenario.get("id", "")
            scenario_text = normalize_text(scenario.get("text", ""))

            trajectory = scenario.get("trajectory") or []
            if not isinstance(trajectory, list):
                if debug:
                    print(f"[debug][iter_failure_cases] scenario {scenario_id} trajectory is not list", flush=True)
                continue

            for task in trajectory:
                task_number = task.get("task_number", "")
                agent_name = normalize_text(task.get("agent_name", ""))
                task_description = normalize_text(task.get("task_description", ""))
                response = normalize_text(task.get("response", ""))
                final_answer = normalize_text(task.get("final_answer", ""))
                reviews = collect_reviews(task)

                for ridx, review in enumerate(reviews, start=1):
                    status = extract_review_status(review)
                    if status not in TARGET_STATUSES:
                        continue

                    yield {
                        "case_id": f"{scenario_id}_t{task_number}_r{ridx}",
                        "source_file": str(file_path),
                        "scenario_id": scenario_id,
                        "scenario_text": scenario_text,
                        "task_number": task_number,
                        "agent_name": agent_name,
                        "status": status,
                        "task_description": task_description,
                        "response": response,
                        "final_answer": final_answer,
                        "review": review,
                        "review_reasoning": extract_review_reasoning(review),
                    }

        if debug and scenario_count == 0:
            print(f"[debug][iter_failure_cases] no valid trajectory scenarios in {file_path}", flush=True)


# ============================================================
# Watsonx adapter
# ============================================================

def raw_watsonx_generate(prompt: str, model_id: int = DEFAULT_MODEL_ID) -> str:
    """
    Prefer the project's existing watsonx wrapper.
    Expected shapes:
    - plain string
    - {"generated_text": "..."}
    - {"results": [{"generated_text": "..."}]}
    """
    try:
        from reactxen.utils.model_inference import watsonx_llm
    except Exception as e:
        raise RuntimeError(
            "Could not import watsonx_llm from reactxen.utils.model_inference. "
            f"Import error: {type(e).__name__}: {e}"
        )

    ret = watsonx_llm(prompt, model_id=model_id)

    if isinstance(ret, str):
        return normalize_text(ret)

    if isinstance(ret, dict):
        if "generated_text" in ret and ret["generated_text"] is not None:
            return normalize_text(ret["generated_text"])

        results = ret.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and first.get("generated_text") is not None:
                return normalize_text(first["generated_text"])

    raise RuntimeError(
        "watsonx_llm returned an unsupported response shape: "
        f"{type(ret).__name__} -> {repr(ret)[:500]}"
    )

# ============================================================
# Robust JSON extraction helpers
# ============================================================

def iter_fenced_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    blocks = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I):
        block = normalize_text(m.group(1))
        if block:
            blocks.append(block)
    return blocks


def iter_balanced_object_candidates(text: str) -> List[str]:
    """
    text中の全ての balanced {...} 候補をできるだけ拾う。
    最初の一個だけでなく複数候補を見る。
    """
    text = normalize_text(text)
    candidates = []

    for start in range(len(text)):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        quote_char = None

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                    quote_char = None
                continue

            if ch in ['"', "'"]:
                in_string = True
                quote_char = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = normalize_text(text[start:i + 1])
                    if candidate:
                        candidates.append(candidate)
                    break

    # 長いもの優先
    candidates = sorted(set(candidates), key=len, reverse=True)
    return candidates


def looks_like_target_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    expected_keys = {
        "status",
        "unmet_requirements",
        "reason_summary",
        "root_cause_category",
        "planner_branch_worth_adding",
        "verification_signal",
        "recommended_branch_pattern",
    }
    if expected_keys.intersection(set(obj.keys())):
        return True

    taxonomy_keys = {
        "taxonomy_name",
        "categories",
    }
    if taxonomy_keys.intersection(set(obj.keys())):
        return True

    return False


def try_parse_json_strict(candidate: str) -> Dict[str, Any]:
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def try_parse_python_literal(candidate: str) -> Dict[str, Any]:
    obj = ast.literal_eval(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed Python literal is not a dict.")
    return obj


def parse_best_object_without_repair(text: str, debug: bool = False) -> Dict[str, Any]:
    """
    1. 全文 strict JSON
    2. fenced block 群
    3. balanced object 群
    の順で試し、target schema に最も近いものを返す。
    """
    text = normalize_text(text)
    parse_errors = []

    # 1) full text
    try:
        obj = try_parse_json_strict(text)
        if looks_like_target_schema(obj):
            return obj
    except Exception as e:
        parse_errors.append(f"strict_full: {type(e).__name__}: {e}")

    # 2) fenced blocks
    for block in iter_fenced_blocks(text):
        try:
            obj = try_parse_json_strict(block)
            if looks_like_target_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"strict_fenced: {type(e).__name__}: {e}")

        try:
            obj = try_parse_python_literal(block)
            if looks_like_target_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"py_fenced: {type(e).__name__}: {e}")

    # 3) balanced object candidates
    for candidate in iter_balanced_object_candidates(text):
        try:
            obj = try_parse_json_strict(candidate)
            if looks_like_target_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"strict_candidate: {type(e).__name__}: {e}")

        try:
            obj = try_parse_python_literal(candidate)
            if looks_like_target_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"py_candidate: {type(e).__name__}: {e}")

    if debug:
        print("[debug][json-extract] parse errors sample:", parse_errors[:8], flush=True)
        print("[debug][json-extract] raw head:", repr(text[:500]), flush=True)

    raise ValueError("Failed to parse model output as target JSON/Python-literal object.")

# ============================================================
# Robust JSON extraction / repair
# ============================================================

def strip_code_fences(text: str) -> str:
    text = normalize_text(text)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return text


def find_balanced_json_object(text: str) -> Optional[str]:
    text = normalize_text(text)
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    quote_char = None

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
                quote_char = None
            continue

        if ch in ['"', "'"]:
            in_string = True
            quote_char = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def try_parse_json_strict(candidate: str) -> Dict[str, Any]:
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def try_parse_python_literal(candidate: str) -> Dict[str, Any]:
    obj = ast.literal_eval(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed Python literal is not a dict.")
    return obj


def parse_object_without_repair(text: str, debug: bool = False) -> Dict[str, Any]:
    original = normalize_text(text)

    try:
        return try_parse_json_strict(original)
    except Exception as e1:
        if debug:
            print(f"[debug][json-extract] strict full parse failed: {type(e1).__name__}: {e1}", flush=True)

    fenced = strip_code_fences(original)
    if fenced != original:
        try:
            return try_parse_json_strict(fenced)
        except Exception as e2:
            if debug:
                print(f"[debug][json-extract] fenced parse failed: {type(e2).__name__}: {e2}", flush=True)

    candidate = find_balanced_json_object(fenced)
    if candidate:
        try:
            return try_parse_json_strict(candidate)
        except Exception as e3:
            if debug:
                print(f"[debug][json-extract] strict candidate parse failed: {type(e3).__name__}: {e3}", flush=True)
                print("[debug][json-extract] candidate head:", repr(candidate[:300]), flush=True)

        try:
            return try_parse_python_literal(candidate)
        except Exception as e4:
            if debug:
                print(f"[debug][json-extract] python-literal parse failed: {type(e4).__name__}: {e4}", flush=True)

    raise ValueError("Failed to parse model output as JSON/Python-literal object.")


def coerce_to_json_via_repair_prompt(bad_text: str, model_id: int, debug: bool = False) -> Dict[str, Any]:
    repair_prompt = f"""
Convert the following text into ONE valid JSON object.

Rules:
- Output JSON only.
- No markdown.
- No code fences.
- Use double quotes for all property names and all string values.
- Do not add any explanation before or after the JSON.
- Preserve all recoverable fields.
- If a value is unclear, preserve it as a string.
- Return exactly one JSON object.

Text:
{bad_text}
""".strip()

    repaired = raw_watsonx_generate(repair_prompt, model_id=model_id)
    repaired = normalize_text(repaired)

    if debug:
        print("[debug][json-repair] repaired raw head:", repr(repaired[:300]), flush=True)

    if not repaired:
        raise ValueError("Repair prompt returned empty text.")

    return parse_best_object_without_repair(repaired, debug=debug)


def extract_first_json_object(text: str, model_id: int, debug: bool = False) -> Dict[str, Any]:
    original = normalize_text(text)

    if debug:
        print("[debug][json-extract] raw head:", repr(original[:300]), flush=True)

    if not original:
        raise ValueError("Model output is empty.")

    try:
        return parse_best_object_without_repair(original, debug=debug)
    except Exception as e:
        if debug:
            print(f"[debug][json-extract] non-repair parse failed: {type(e).__name__}: {e}", flush=True)

    return coerce_to_json_via_repair_prompt(original, model_id=model_id, debug=debug)


def call_watsonx_json(prompt: str, model_id: int = DEFAULT_MODEL_ID, debug: bool = False) -> Dict[str, Any]:
    last_err = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            raw = raw_watsonx_generate(prompt, model_id=model_id)
            raw = normalize_text(raw)

            if debug:
                print(f"[debug][watsonx] attempt={attempt} raw head={repr(raw[:300])}", flush=True)

            if not raw:
                raise ValueError("raw_watsonx_generate returned empty text.")

            return extract_first_json_object(raw, model_id=model_id, debug=debug)

        except Exception as e:
            last_err = e
            if debug:
                print(f"[debug][watsonx] attempt={attempt} error={type(e).__name__}: {e}", flush=True)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP)

    raise RuntimeError(f"watsonx json generation failed after retries: {last_err}")

# ============================================================
# LLM prompts
# ============================================================

def build_failure_summary_prompt(case: Dict[str, Any]) -> str:
    return f"""
You are analyzing why a task was judged as Partially accomplished, Not accomplished, or Error.

Your job:
1. Compare the task requirement against the response, final_answer, and review.
2. Identify what requirement was not satisfied.
3. Summarize the likely root cause.
4. Produce planner-facing information useful for designing verification branches.

Return JSON only.
Return exactly one JSON object starting with "{{" and ending with "}}".
Do not use markdown.
Do not use code fences.
Do not add any text before or after the JSON.

Output schema:
{{
  "status": "<partially accomplished | not accomplished | error>",
  "unmet_requirements": ["<short phrase>", "..."],
  "reason_summary": "<one concise paragraph>",
  "root_cause_category": "<short category name>",
  "planner_branch_worth_adding": true,
  "verification_signal": "<what a verification step should check>",
  "recommended_branch_pattern": "<short branch pattern>"
}}

Case:
- task_description: {json.dumps(clip_text(case["task_description"], MAX_TASK_DESC_CHARS), ensure_ascii=False)}
- response: {json.dumps(clip_text(case["response"], MAX_RESPONSE_CHARS), ensure_ascii=False)}
- final_answer: {json.dumps(clip_text(case["final_answer"], MAX_FINAL_ANSWER_CHARS), ensure_ascii=False)}
- review: {json.dumps(clip_text(case["review"], MAX_REVIEW_CHARS), ensure_ascii=False)}

Guidelines:
- Focus on the missing or violated requirement.
- If the main problem is wrong entity, wrong variable, wrong time range, or wrong scope, say that explicitly.
- If the answer terminated before evidence sufficiency was established, say so.
- If the answer is partially correct but missing a required artifact, say so.
- Keep the category name short and reusable across cases.
- planner_branch_worth_adding should be true if this failure could realistically justify a verification or clarification branch.
""".strip()


def build_taxonomy_batch_prompt(case_batch: List[Dict[str, Any]]) -> str:
    compact_cases = []
    for c in case_batch:
        compact_cases.append({
            "case_id": c["case_id"],
            "status": c["status"],
            "task_description": c["task_description"],
            "unmet_requirements": c.get("unmet_requirements", []),
            "reason_summary": c.get("reason_summary", ""),
            "root_cause_category": c.get("root_cause_category", ""),
            "verification_signal": c.get("verification_signal", ""),
            "recommended_branch_pattern": c.get("recommended_branch_pattern", ""),
        })

    return f"""
You are creating a planner-facing taxonomy of reasons why tasks were judged as Partially accomplished, Not accomplished, or Error.

Input: a batch of failure cases.

Your goals:
1. Group overlapping failure reasons into reusable categories.
2. Define categories in a planner-usable way.
3. Emphasize categories that justify explicit verification or clarification branches.

Return JSON only.
Return exactly one JSON object starting with "{{" and ending with "}}".
Do not use markdown.
Do not use code fences.
Do not add any text before or after the JSON.

Output schema:
{{
  "taxonomy_name": "partial_not_accomplished_reason_taxonomy",
  "categories": [
    {{
      "category_name": "<short category>",
      "definition": "<1-2 sentence definition>",
      "typical_statuses": ["<status>", "..."],
      "failure_signals": ["<signal>", "..."],
      "unmet_requirement_patterns": ["<pattern>", "..."],
      "planner_response": "<how the planner should respond>",
      "branch_condition_template": "<if ... then ...>",
      "example_case_ids": ["<case_id>", "..."]
    }}
  ]
}}

Cases:
{json.dumps(compact_cases, ensure_ascii=False, indent=2)}

Guidelines:
- Prefer reusable categories over overly specific ones.
- Merge categories that differ only in wording.
- Include category types such as missing constraint, wrong scope, missing artifact, failed verification, premature termination, format contamination, tool/runtime error, context-length failure, ambiguity not resolved, or other genuinely recurring patterns if supported by the cases.
- planner_response should be concrete and branch-oriented.
""".strip()


def build_taxonomy_merge_prompt(taxonomies: List[Dict[str, Any]]) -> str:
    return f"""
You are merging several planner-facing taxonomies of failure reasons into one unified taxonomy.

Goals:
- Merge overlapping categories.
- Remove near-duplicates.
- Keep category names stable and reusable.
- Preserve planner-facing branch guidance.

Return JSON only.
Return exactly one JSON object starting with "{{" and ending with "}}".
Do not use markdown.
Do not use code fences.
Do not add any text before or after the JSON.

Output schema:
{{
  "taxonomy_name": "merged_partial_not_accomplished_reason_taxonomy",
  "categories": [
    {{
      "category_name": "<short category>",
      "definition": "<1-2 sentence definition>",
      "typical_statuses": ["<status>", "..."],
      "failure_signals": ["<signal>", "..."],
      "unmet_requirement_patterns": ["<pattern>", "..."],
      "planner_response": "<how the planner should respond>",
      "branch_condition_template": "<if ... then ...>",
      "example_case_ids": ["<case_id>", "..."]
    }}
  ]
}}

Input taxonomies:
{json.dumps(taxonomies, ensure_ascii=False, indent=2)}

Guidelines:
- Prefer fewer, broader, still-actionable categories.
- If two categories differ only lexically, merge them.
- Keep context-length and runtime-error categories distinct from semantic failure categories.
- Keep category definitions planner-facing, not evaluator-facing.
""".strip()

# ============================================================
# Rule-based fallback summarization
# ============================================================

def fallback_summary_from_case(case: Dict[str, Any], llm_error: str = "") -> Dict[str, Any]:
    text = " ".join([
        normalize_text(case.get("task_description", "")),
        normalize_text(case.get("response", "")),
        normalize_text(case.get("final_answer", "")),
        normalize_text(case.get("review", "")),
    ]).lower()

    status = case.get("status", "")

    # Error
    if status == "error":
        return {
            "unmet_requirements": ["Task could not be evaluated or completed due to execution error."],
            "reason_summary": "Fallback summary used because LLM JSON extraction failed. The case is treated as an execution or context-length related error unless a more specific cause is visible in the review.",
            "root_cause_category": "Execution or context-length failure",
            "planner_branch_worth_adding": True,
            "verification_signal": "Check whether intermediate payload size, runtime status, and required tool outputs are valid before downstream processing.",
            "recommended_branch_pattern": "if execution_error_or_payload_too_large then branch to execution_guard_or_chunking_step",
            "llm_used": False,
            "llm_failure": llm_error,
        }

    # Missing data
    if any(p in text for p in [
        "file not found", "no such file", "dataset not found", "missing data",
        "could not locate", "could not read", "unreadable", "never retrieved"
    ]):
        return {
            "unmet_requirements": ["Required dataset or file was missing, unreadable, or not retrieved."],
            "reason_summary": "The task could not be completed because required data was unavailable or was not accessed before analysis.",
            "root_cause_category": "Missing Data Availability",
            "planner_branch_worth_adding": True,
            "verification_signal": "Check that the required dataset or file exists and is readable before analysis.",
            "recommended_branch_pattern": "if data_file_missing_or_unreadable then branch to data_acquisition_check",
            "llm_used": False,
            "llm_failure": llm_error,
        }

    # Missing tool execution
    if any(p in text for p in [
        "did not call", "failed to call", "no tool output", "tool was not used",
        "only described", "api call was not made", "not executed", "invalid file path"
    ]):
        return {
            "unmet_requirements": ["Required tool or API execution was missing or invalid."],
            "reason_summary": "The task required a concrete tool or API execution, but the agent did not execute it correctly or did not use the output.",
            "root_cause_category": "Missing Tool Execution",
            "planner_branch_worth_adding": True,
            "verification_signal": "Check that the required tool was actually called with valid arguments and produced output.",
            "recommended_branch_pattern": "if tool_output_absent_or_invalid then branch to tool_execution_verification",
            "llm_used": False,
            "llm_failure": llm_error,
        }

    # Missing final artifact
    if any(p in text for p in [
        "final response missing", "no final answer", "artifact absent",
        "work order not generated", "output missing", "no evidence of api call"
    ]):
        return {
            "unmet_requirements": ["Expected final artifact or final answer was not produced."],
            "reason_summary": "The task did not produce the required final deliverable, so completion could not be verified.",
            "root_cause_category": "Missing Final Artifact",
            "planner_branch_worth_adding": True,
            "verification_signal": "Check that the required output artifact is present before marking the step complete.",
            "recommended_branch_pattern": "if required_artifact_missing then branch to output_presence_check",
            "llm_used": False,
            "llm_failure": llm_error,
        }

    # Insufficient evidence
    if any(p in text for p in [
        "without evidence", "insufficient evidence", "not supported by data",
        "generic suggestion", "not based on analysis", "unsupported conclusion"
    ]):
        return {
            "unmet_requirements": ["Answer was not supported by sufficient evidence from data or tools."],
            "reason_summary": "The agent gave a conclusion, but it was not adequately justified by analysis results or retrieved evidence.",
            "root_cause_category": "Insufficient Evidence",
            "planner_branch_worth_adding": True,
            "verification_signal": "Check that the answer includes concrete supporting evidence before downstream use.",
            "recommended_branch_pattern": "if answer_lacks_evidence then branch to evidence_verification",
            "llm_used": False,
            "llm_failure": llm_error,
        }

    # Default fallback
    return {
        "unmet_requirements": ["One or more task requirements were not satisfied."],
        "reason_summary": "Fallback summary used because LLM JSON extraction failed. The review indicates that the step did not satisfy its local task requirements.",
        "root_cause_category": "Unclassified Requirement Failure",
        "planner_branch_worth_adding": True,
        "verification_signal": "Check local task completion against required entities, evidence, and output presence.",
        "recommended_branch_pattern": "if local_task_requirement_not_met then branch to requirement_specific_verification",
        "llm_used": False,
        "llm_failure": llm_error,
    }

# ============================================================
# Failure summarization
# ============================================================

def summarize_failure_case(case: Dict[str, Any], model_id: int, debug: bool = False) -> Dict[str, Any]:
    status = case["status"]
    joined = " ".join([
        case["task_description"],
        case["response"],
        case["final_answer"],
        case["review"],
    ])

    if status == "error":
        if looks_like_context_length_error(joined):
            reason_summary = (
                "The failure is most likely caused by input or context length limitation "
                "while extracting or passing JSON evidence to the model."
            )
        else:
            reason_summary = (
                "The failure is treated as likely caused by input or context length limitation "
                "while extracting or passing JSON evidence, unless a more specific runtime cause is available."
            )

        return {
            **case,
            "unmet_requirements": ["Task could not be evaluated or completed due to execution error."],
            "reason_summary": reason_summary,
            "root_cause_category": "Context or input length failure",
            "planner_branch_worth_adding": True,
            "verification_signal": "Check whether the evidence payload or intermediate JSON exceeds safe input size before downstream processing.",
            "recommended_branch_pattern": "if evidence_payload_too_large then branch to chunking_or_filtering_step",
            "llm_used": False,
        }

    prompt = build_failure_summary_prompt(case)

    try:
        llm_obj = call_watsonx_json(prompt, model_id=model_id, debug=debug)
        return {
            **case,
            "unmet_requirements": llm_obj.get("unmet_requirements", []),
            "reason_summary": llm_obj.get("reason_summary", ""),
            "root_cause_category": llm_obj.get("root_cause_category", ""),
            "planner_branch_worth_adding": llm_obj.get("planner_branch_worth_adding", True),
            "verification_signal": llm_obj.get("verification_signal", ""),
            "recommended_branch_pattern": llm_obj.get("recommended_branch_pattern", ""),
            "llm_used": True,
            "llm_raw_summary_obj": llm_obj,
        }
    except Exception as e:
        if debug:
            print(f"[debug][summary-fallback] case_id={case.get('case_id')} error={type(e).__name__}: {e}", flush=True)

        fallback = fallback_summary_from_case(case, llm_error=f"{type(e).__name__}: {e}")
        return {
            **case,
            **fallback,
        }

# ============================================================
# Taxonomy building / merging
# ============================================================

def build_local_taxonomy_from_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for c in cases:
        key = normalize_text(c.get("root_cause_category", "")) or "Unclassified Requirement Failure"
        groups.setdefault(key, []).append(c)

    categories = []
    for category_name, members in groups.items():
        typical_statuses = sorted(set(normalize_text(m.get("status", "")) for m in members if normalize_text(m.get("status", ""))))
        failure_signals = sorted(set(normalize_text(m.get("verification_signal", "")) for m in members if normalize_text(m.get("verification_signal", ""))))
        unmet_patterns = []
        for m in members:
            for x in m.get("unmet_requirements", []) or []:
                x = normalize_text(x)
                if x and x not in unmet_patterns:
                    unmet_patterns.append(x)

        planner_response = normalize_text(members[0].get("verification_signal", "")) or "Verify the missing local requirement before proceeding."
        branch_condition_template = normalize_text(members[0].get("recommended_branch_pattern", "")) or "if local_requirement_not_met then branch to verification_step"

        categories.append({
            "category_name": category_name,
            "definition": f"Locally grouped fallback category for cases labeled as {category_name}.",
            "typical_statuses": typical_statuses,
            "failure_signals": failure_signals[:5],
            "unmet_requirement_patterns": unmet_patterns[:5],
            "planner_response": planner_response,
            "branch_condition_template": branch_condition_template,
            "example_case_ids": [m["case_id"] for m in members[:5]],
        })

    return {
        "taxonomy_name": "partial_not_accomplished_reason_taxonomy",
        "categories": categories,
    }

def build_taxonomy_for_batch(
    cases: List[Dict[str, Any]],
    model_id: int,
    batch_index: int,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt = build_taxonomy_batch_prompt(cases)

    try:
        taxonomy = call_watsonx_json(prompt, model_id=model_id, debug=debug)
    except Exception as e:
        if debug:
            print(f"[debug][taxonomy-fallback] batch={batch_index} error={type(e).__name__}: {e}", flush=True)
        taxonomy = build_local_taxonomy_from_cases(cases)

    taxonomy["_meta"] = {
        "taxonomy_id": f"batch_{batch_index:04d}",
        "round_index": 0,
        "group_index": batch_index,
        "source_case_ids": [c["case_id"] for c in cases],
    }

    out_path = output_dir / "taxonomy_batches" / f"taxonomy_batch_{batch_index:04d}.json"
    write_json(out_path, taxonomy)
    return taxonomy

def build_local_merged_taxonomy(taxonomies: List[Dict[str, Any]]) -> Dict[str, Any]:
    category_map: Dict[str, Dict[str, Any]] = {}

    for tx in taxonomies:
        for cat in tx.get("categories", []):
            name = normalize_text(cat.get("category_name", "")) or "Unclassified Requirement Failure"
            if name not in category_map:
                category_map[name] = {
                    "category_name": name,
                    "definition": normalize_text(cat.get("definition", "")),
                    "typical_statuses": [],
                    "failure_signals": [],
                    "unmet_requirement_patterns": [],
                    "planner_response": normalize_text(cat.get("planner_response", "")),
                    "branch_condition_template": normalize_text(cat.get("branch_condition_template", "")),
                    "example_case_ids": [],
                }

            dst = category_map[name]
            for k in ["typical_statuses", "failure_signals", "unmet_requirement_patterns", "example_case_ids"]:
                for v in cat.get(k, []) or []:
                    v = normalize_text(v)
                    if v and v not in dst[k]:
                        dst[k].append(v)

            if not dst["definition"] and normalize_text(cat.get("definition", "")):
                dst["definition"] = normalize_text(cat.get("definition", ""))

            if not dst["planner_response"] and normalize_text(cat.get("planner_response", "")):
                dst["planner_response"] = normalize_text(cat.get("planner_response", ""))

            if not dst["branch_condition_template"] and normalize_text(cat.get("branch_condition_template", "")):
                dst["branch_condition_template"] = normalize_text(cat.get("branch_condition_template", ""))

    return {
        "taxonomy_name": "merged_partial_not_accomplished_reason_taxonomy",
        "categories": list(category_map.values()),
    }


def merge_taxonomy_group(
    taxonomies: List[Dict[str, Any]],
    round_index: int,
    group_index: int,
    model_id: int,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt = build_taxonomy_merge_prompt(taxonomies)

    try:
        merged = call_watsonx_json(prompt, model_id=model_id, debug=debug)
    except Exception as e:
        if debug:
            print(f"[debug][merge-fallback] round={round_index} group={group_index} error={type(e).__name__}: {e}", flush=True)
        merged = build_local_merged_taxonomy(taxonomies)

    src_ids = []
    for t in taxonomies:
        meta = t.get("_meta", {})
        tid = meta.get("taxonomy_id")
        if tid:
            src_ids.append(tid)

    merged["_meta"] = {
        "taxonomy_id": f"round_{round_index:02d}_group_{group_index:04d}",
        "round_index": round_index,
        "group_index": group_index,
        "source_taxonomy_ids": src_ids,
    }

    out_path = (
        output_dir
        / f"taxonomy_merge_round_{round_index:02d}"
        / f"taxonomy_round_{round_index:02d}_group_{group_index:04d}.json"
    )
    write_json(out_path, merged)
    return merged


def flatten_taxonomy_to_rows(taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    meta = taxonomy.get("_meta", {})
    taxonomy_id = meta.get("taxonomy_id", "")
    round_index = meta.get("round_index", "")
    group_index = meta.get("group_index", "")

    for cat in taxonomy.get("categories", []):
        rows.append({
            "taxonomy_id": taxonomy_id,
            "round_index": round_index,
            "group_index": group_index,
            "category_name": cat.get("category_name", ""),
            "definition": cat.get("definition", ""),
            "typical_statuses": cat.get("typical_statuses", []),
            "failure_signals": cat.get("failure_signals", []),
            "unmet_requirement_patterns": cat.get("unmet_requirement_patterns", []),
            "planner_response": cat.get("planner_response", ""),
            "branch_condition_template": cat.get("branch_condition_template", ""),
            "example_case_ids": cat.get("example_case_ids", []),
        })
    return rows


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    model_id: int,
    batch_size: int,
    merge_group_size: int,
    max_cases: Optional[int] = None,
    debug: bool = False,
    debug_sample_limit: int = 50,
    skip_llm: bool = False,
) -> None:
    ensure_dir(output_dir)

    debug_report = scan_input_dir_debug(
        input_dir=input_dir,
        output_dir=output_dir,
        debug=debug,
        debug_sample_limit=debug_sample_limit,
    )

    failure_jsonl = output_dir / "failure_case_summaries.jsonl"
    failure_csv = output_dir / "failure_case_summaries.csv"
    taxonomy_jsonl = output_dir / "taxonomy_rows.jsonl"
    taxonomy_csv = output_dir / "taxonomy_rows.csv"

    processed_case_ids = load_processed_case_ids(failure_jsonl)

    matching_count = debug_report.get("matching_target_status_count", 0)
    if debug:
        print(f"[debug] matching target failure statuses = {matching_count}", flush=True)

    if matching_count == 0:
        print("[done] no failing cases found.", flush=True)
        print(f"[debug] see: {output_dir / 'debug_report.json'}", flush=True)
        return

    extracted_count = 0
    summarized_cases = []

    for case in iter_failure_cases(input_dir, debug=debug):
        case_id = case["case_id"]
        if case_id in processed_case_ids:
            if debug:
                print(f"[debug][skip-existing] {case_id}", flush=True)
            continue

        if skip_llm:
            summary = {
                **case,
                "unmet_requirements": [],
                "reason_summary": "LLM summarization skipped.",
                "root_cause_category": "",
                "planner_branch_worth_adding": True,
                "verification_signal": "",
                "recommended_branch_pattern": "",
                "llm_used": False,
            }
        else:
            summary = summarize_failure_case(case, model_id=model_id, debug=debug)

        append_jsonl(failure_jsonl, summary)
        append_csv(failure_csv, summary, FAILURE_CASE_FIELDNAMES)

        summarized_cases.append(summary)
        extracted_count += 1

        print(f"[failure-summary] {case_id} -> {summary.get('root_cause_category', '')}", flush=True)

        if max_cases is not None and extracted_count >= max_cases:
            break

    all_cases = load_jsonl(failure_jsonl)
    if not all_cases:
        print("[done] no failing cases found after filtering existing cases.", flush=True)
        print(f"[debug] see: {output_dir / 'debug_report.json'}", flush=True)
        return

    if skip_llm:
        print("[done] skip_llm=True, finished extraction/debug only.", flush=True)
        return

    case_batches = chunk_list(all_cases, batch_size)
    batch_taxonomies = []

    for batch_idx, batch in enumerate(case_batches, start=1):
        taxonomy = build_taxonomy_for_batch(
            cases=batch,
            model_id=model_id,
            batch_index=batch_idx,
            output_dir=output_dir,
            debug=debug,
        )
        batch_taxonomies.append(taxonomy)

        for row in flatten_taxonomy_to_rows(taxonomy):
            append_jsonl(taxonomy_jsonl, row)
            append_csv(taxonomy_csv, row, TAXONOMY_FIELDNAMES)

        print(
            f"[taxonomy-batch] batch={batch_idx} size={len(batch)} categories={len(taxonomy.get('categories', []))}",
            flush=True,
        )

    current_taxonomies = batch_taxonomies
    round_index = 1

    while len(current_taxonomies) > 1:
        next_round = []
        groups = group_list(current_taxonomies, merge_group_size)

        for group_index, group in enumerate(groups, start=1):
            if len(group) == 1:
                next_round.append(group[0])
                continue

            merged = merge_taxonomy_group(
                taxonomies=group,
                round_index=round_index,
                group_index=group_index,
                model_id=model_id,
                output_dir=output_dir,
                debug=debug,
            )
            next_round.append(merged)

            for row in flatten_taxonomy_to_rows(merged):
                append_jsonl(taxonomy_jsonl, row)
                append_csv(taxonomy_csv, row, TAXONOMY_FIELDNAMES)

            print(
                f"[taxonomy-merge] round={round_index} group={group_index} "
                f"input_taxonomies={len(group)} output_categories={len(merged.get('categories', []))}",
                flush=True,
            )

        current_taxonomies = next_round
        round_index += 1

    final_taxonomy = current_taxonomies[0]
    write_json(output_dir / "final_taxonomy.json", final_taxonomy)

    final_rows_csv = output_dir / "final_taxonomy_rows.csv"
    if final_rows_csv.exists():
        final_rows_csv.unlink()
    for row in flatten_taxonomy_to_rows(final_taxonomy):
        append_csv(final_rows_csv, row, TAXONOMY_FIELDNAMES)

    print("[done] final taxonomy written to:", output_dir / "final_taxonomy.json", flush=True)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing trajectory JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--model_id", type=int, default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--merge_group_size", type=int, default=DEFAULT_MERGE_GROUP_SIZE)
    parser.add_argument("--max_cases", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--debug", type=parse_bool, default=False)
    parser.add_argument("--debug_sample_limit", type=int, default=50)
    parser.add_argument("--skip_llm", type=parse_bool, default=False)
    args = parser.parse_args()

    run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_id=args.model_id,
        batch_size=args.batch_size,
        merge_group_size=args.merge_group_size,
        max_cases=args.max_cases,
        debug=args.debug,
        debug_sample_limit=args.debug_sample_limit,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()