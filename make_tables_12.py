#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_tables_12.py

Creates:
  - Table1: Overall success + effort (per-task & per-run; excluding zero-task runs)
  - Table2: Task-length comparison vs BASE (win/lose/tie; excluding zero-task runs)

Directory layout assumed:
  trajectory_root/
    [BASE]Model_16/
      Q_101_trajectory.json
      ...
    [SPIRAL]Model_16/
    [SPIRAL_wo_sim]Model_16/
    [SPIRAL_wo_cri2]Model_16/

  exp_root/
    [BASE]Model_16/
      Model_16_Q_101_time_token.txt   (JSON)
      ...
    [SPIRAL]Model_16/
    ...

Usage example (absolute paths):
  python3 /Users/yusuke/Desktop/Program/codabench/AssetOpsBench/make_tables_12.py \
    --trajectory_root "/Users/yusuke/Desktop/Program/codabench/AssetOpsBench/benchmark/cods_track1/track1_result/trajectory" \
    --exp_root        "/Users/yusuke/Desktop/Program/codabench/AssetOpsBench/benchmark/cods_track1/track1_result/exp" \
    --model "Model_16" \
    --tags "BASE,SPIRAL,SPIRAL_wo_sim,SPIRAL_wo_cri2" \
    --out_dir "/Users/yusuke/Desktop/Program/codabench/AssetOpsBench/benchmark/cods_track1/track1_result/tables" \
    --debug
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Data structures
# -------------------------

@dataclass
class RunStats:
    run_id: int

    # Task-level outcomes
    num_tasks: int
    num_accomplished: int
    num_partial: int
    num_not: int
    num_error: int
    num_unknown: int

    # Effort metrics (from trajectory JSON)
    tool_calls: int           # count of non-Finish tool actions
    api_calls: int            # from logs.info.model_stats.api_calls
    tokens_sent: int          # from logs.info.model_stats.tokens_sent
    tokens_received: int      # from logs.info.model_stats.tokens_received

    # Run-level metrics (from exp time_token file, if provided)
    elapsed_seconds: Optional[float] = None
    exp_input_tokens: Optional[int] = None
    exp_generated_tokens: Optional[int] = None


# -------------------------
# Regex / helpers
# -------------------------

TRAJ_FNAME_RE = re.compile(r"Q_(\d+)_trajectory\.json$", re.IGNORECASE)
EXP_FILE_RE   = re.compile(r".*_Q_(\d+)_time_token\.txt$", re.IGNORECASE)

TASK_STATUS_RE = re.compile(r"task\s*status\s*:\s*(.+)", re.IGNORECASE)
TOOL_NAME_IN_ACTION_RE = re.compile(r"Tool Name:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)


def bracket_tag(tag: str) -> str:
    tag = tag.strip()
    if tag.startswith("[") and tag.endswith("]"):
        return tag
    return f"[{tag}]"


def get_dir_tag(path: str) -> str:
    """
    Extract leading bracket tag from directory basename.
    Example: '/.../[SPIRAL]Model_16/' -> '[SPIRAL]'
    """
    name = os.path.basename(os.path.normpath(path))
    m = re.match(r"^\[[^\]]+\]", name)
    return m.group(0) if m else name


def extract_run_id_from_traj_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = TRAJ_FNAME_RE.search(base)
    return int(m.group(1)) if m else None


def traj_len(run_json: Dict[str, Any]) -> int:
    traj = run_json.get("trajectory") or []
    return len(traj) if isinstance(traj, list) else 0


# -------------------------
# Robust directory finding (NO glob; handles literal [] safely)
# -------------------------

def find_method_dir(root: str, tag: str, model: str, debug: bool = False) -> str:
    """
    Find a directory under `root` whose name begins with '[TAG]' and contains model,
    typically exactly: '[TAG]Model_16' (or '[TAG]Model_16_something').
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory not found: {root}")

    btag = bracket_tag(tag)
    candidates: List[str] = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith(btag):
            continue
        if model not in name:
            continue
        candidates.append(full)

    if not candidates:
        raise FileNotFoundError(
            f"Not found under root.\n"
            f"  root: {root}\n"
            f"  expected prefix: {btag}\n"
            f"  expected model substring: {model}\n"
            f"Hint: check actual directory names with `ls {root}`."
        )

    # Prefer exact match if exists
    exact = os.path.join(root, f"{btag}{model}")
    if os.path.isdir(exact):
        if debug:
            print(f"[DEBUG] find_method_dir: exact match: {exact}")
        return exact

    # Otherwise pick lexicographically first (deterministic)
    candidates.sort()
    if debug:
        print(f"[DEBUG] find_method_dir: candidates={candidates}")
    return candidates[0]


# -------------------------
# Loaders
# -------------------------

def load_runs_from_dir(traj_dir: str, debug: bool = False) -> Dict[int, Dict[str, Any]]:
    runs: Dict[int, Dict[str, Any]] = {}
    if not os.path.isdir(traj_dir):
        return runs

    files = [f for f in os.listdir(traj_dir) if f.endswith("_trajectory.json")]
    files.sort()

    if debug:
        print(f"[DEBUG]   traj_dir={traj_dir}")
        print(f"[DEBUG]   found {len(files)} trajectory json files")
        for f in files[:5]:
            print(f"[DEBUG]     traj_file: {os.path.join(traj_dir, f)}")

    for fname in files:
        path = os.path.join(traj_dir, fname)
        run_id = extract_run_id_from_traj_filename(path)
        if run_id is None:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                runs[run_id] = json.load(fp)
        except Exception as e:
            if debug:
                print(f"[DEBUG]   failed to load {path}: {e}")
            continue

    return runs


def load_exp_metrics(exp_dir: Optional[str], debug: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Loads *time_token.txt JSON blobs.
    Returns metrics[qid] = {"elapsed_seconds":..., "total_input_tokens":..., "total_generated_tokens":..., ...}
    """
    metrics: Dict[int, Dict[str, Any]] = {}
    if not exp_dir or not os.path.isdir(exp_dir):
        return metrics

    files = [f for f in os.listdir(exp_dir) if f.endswith("_time_token.txt")]
    files.sort()

    if debug:
        print(f"[DEBUG]   exp_dir={exp_dir}")
        print(f"[DEBUG]   found {len(files)} exp json files")

    for fname in files:
        m = EXP_FILE_RE.match(fname)
        if not m:
            continue
        qid = int(m.group(1))
        path = os.path.join(exp_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                metrics[qid] = json.load(fp)
        except Exception as e:
            if debug:
                print(f"[DEBUG]   failed to load exp metric {path}: {e}")

    return metrics


# -------------------------
# Status parsing (same as your working script)
# -------------------------

def extract_status_from_task(task: Dict[str, Any]) -> str:
    """
    Returns one of:
      - "Accomplished"
      - "Partially accomplished"
      - "Not accomplished"
      - "Error"
      - "Unknown"
    """
    logs = task.get("logs") or {}

    reviews = task.get("reviews")
    if not reviews:
        reviews = logs.get("reviews") or []

    if not isinstance(reviews, list) or not reviews:
        return "Unknown"

    for rev in reviews:
        if not isinstance(rev, str):
            continue

        m = TASK_STATUS_RE.search(rev)
        if not m:
            continue

        first_line = m.group(1).strip().splitlines()[0].strip()
        s = first_line.lower()

        if "error" in s:
            return "Error"
        if "partially accomplished" in s or "partial" in s:
            return "Partially accomplished"
        if "not accomplished" in s or "failed" in s or "failure" in s:
            return "Not accomplished"
        if "accomplished" in s:
            return "Accomplished"

        return first_line

    return "Unknown"


# -------------------------
# Effort extraction (same as your working script)
# -------------------------

def extract_model_stats(task: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Extract (tokens_sent, tokens_received, api_calls) from:
      task["logs"]["info"]["model_stats"]
    Missing fields are treated as 0.
    """
    logs = task.get("logs") or {}
    info = logs.get("info") or {}
    ms = info.get("model_stats") or {}

    def to_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    return (
        to_int(ms.get("tokens_sent", 0)),
        to_int(ms.get("tokens_received", 0)),
        to_int(ms.get("api_calls", 0)),
    )


def count_tool_calls(task: Dict[str, Any]) -> int:
    """
    Count non-Finish tool actions.

    Priority:
      1) logs["trajectroy_log"] or logs["trajectory_log"] (step dicts with "action")
      2) logs["trajectory"] (dicts with "action" string containing "Tool Name: ...")

    Excludes:
      - Finish
      - Self-Ask / Self Ask
    """
    logs = task.get("logs") or {}

    steps = logs.get("trajectroy_log") or logs.get("trajectory_log") or []
    actions: List[str] = []

    if isinstance(steps, list) and steps:
        for s in steps:
            if isinstance(s, dict):
                a = s.get("action")
                if isinstance(a, str) and a.strip():
                    actions.append(a.strip())
    else:
        traj_steps = logs.get("trajectory") or []
        if isinstance(traj_steps, list):
            for s in traj_steps:
                if not isinstance(s, dict):
                    continue
                a = s.get("action")
                if not isinstance(a, str) or not a.strip():
                    continue
                m = TOOL_NAME_IN_ACTION_RE.search(a)
                actions.append(m.group(1).strip() if m else a.strip())

    def is_excluded(a: str) -> bool:
        al = a.strip().lower()
        return al in {"finish", "self-ask", "self ask"}

    return sum(1 for a in actions if a and not is_excluded(a))


def summarize_run(
    run_json: Dict[str, Any],
    run_id: int,
    exp_metrics: Optional[Dict[int, Dict[str, Any]]] = None
) -> RunStats:
    traj = run_json.get("trajectory") or []
    if not isinstance(traj, list):
        traj = []

    cnt = Counter()
    tool_calls = 0
    api_calls = 0
    tokens_sent = 0
    tokens_received = 0

    for t in traj:
        if not isinstance(t, dict):
            cnt["Unknown"] += 1
            continue

        status = extract_status_from_task(t)
        cnt[status] += 1

        tool_calls += count_tool_calls(t)
        ts, tr, ac = extract_model_stats(t)
        tokens_sent += ts
        tokens_received += tr
        api_calls += ac

    elapsed = None
    exp_in = None
    exp_gen = None
    if exp_metrics and run_id in exp_metrics:
        obj = exp_metrics[run_id]
        elapsed = obj.get("elapsed_seconds")
        exp_in = obj.get("total_input_tokens")
        exp_gen = obj.get("total_generated_tokens")

        try:
            elapsed = float(elapsed) if elapsed is not None else None
        except Exception:
            elapsed = None
        try:
            exp_in = int(exp_in) if exp_in is not None else None
        except Exception:
            exp_in = None
        try:
            exp_gen = int(exp_gen) if exp_gen is not None else None
        except Exception:
            exp_gen = None

    return RunStats(
        run_id=run_id,
        num_tasks=len(traj),
        num_accomplished=cnt.get("Accomplished", 0),
        num_partial=cnt.get("Partially accomplished", 0),
        num_not=cnt.get("Not accomplished", 0),
        num_error=cnt.get("Error", 0),
        num_unknown=cnt.get("Unknown", 0),
        tool_calls=tool_calls,
        api_calls=api_calls,
        tokens_sent=tokens_sent,
        tokens_received=tokens_received,
        elapsed_seconds=elapsed,
        exp_input_tokens=exp_in,
        exp_generated_tokens=exp_gen,
    )


def combined_tokens(st: RunStats) -> Tuple[int, int]:
    """
    Total token accounting = trajectory tokens + exp tokens.
    (If you want trajectory-only tokens, change this to return st.tokens_sent, st.tokens_received.)
    """
    exp_in = st.exp_input_tokens or 0
    exp_gen = st.exp_generated_tokens or 0
    return st.tokens_sent + exp_in, st.tokens_received + exp_gen


# -------------------------
# Aggregation for Table1
# -------------------------

def overall_stats(
    runs: Dict[int, Dict[str, Any]],
    exp_metrics: Optional[Dict[int, Dict[str, Any]]] = None
) -> Dict[str, float]:
    total_tasks = 0
    total_acc = 0
    total_part = 0
    total_not = 0
    total_err = 0
    total_unk = 0

    total_tool_calls = 0
    total_api_calls = 0
    total_tokens_sent = 0
    total_tokens_received = 0

    runs_included = 0

    total_elapsed = 0.0
    exp_count = 0

    for rid, rjson in runs.items():
        st = summarize_run(rjson, run_id=rid, exp_metrics=exp_metrics)

        # EXCLUDE runs with 0 tasks
        if st.num_tasks == 0:
            continue

        runs_included += 1

        total_tasks += st.num_tasks
        total_acc += st.num_accomplished
        total_part += st.num_partial
        total_not += st.num_not
        total_err += st.num_error
        total_unk += st.num_unknown

        total_tool_calls += st.tool_calls
        total_api_calls += st.api_calls

        tot_s, tot_r = combined_tokens(st)
        total_tokens_sent += tot_s
        total_tokens_received += tot_r

        if st.elapsed_seconds is not None:
            total_elapsed += st.elapsed_seconds
            exp_count += 1

    d: Dict[str, float] = {}
    d["Runs"] = float(runs_included)
    d["Tasks"] = float(total_tasks)

    # per-task outcome rates
    d["Acc"] = (total_acc / total_tasks) if total_tasks else 0.0
    d["PartialPlus"] = ((total_acc + total_part) / total_tasks) if total_tasks else 0.0
    d["Not"] = (total_not / total_tasks) if total_tasks else 0.0
    d["Error"] = (total_err / total_tasks) if total_tasks else 0.0
    d["Unknown"] = (total_unk / total_tasks) if total_tasks else 0.0

    # per-run
    d["AvgTasksPerRun"] = (total_tasks / runs_included) if runs_included else 0.0

    # per-task effort
    d["AvgToolCallsPerTask"] = (total_tool_calls / total_tasks) if total_tasks else 0.0
    d["AvgApiCallsPerTask"]  = (total_api_calls / total_tasks) if total_tasks else 0.0
    d["AvgTokSentPerTask"]   = (total_tokens_sent / total_tasks) if total_tasks else 0.0
    d["AvgTokRecvPerTask"]   = (total_tokens_received / total_tasks) if total_tasks else 0.0

    # per-run effort
    d["AvgToolCallsPerRun"]  = (total_tool_calls / runs_included) if runs_included else 0.0
    d["AvgApiCallsPerRun"]   = (total_api_calls / runs_included) if runs_included else 0.0
    d["AvgTokSentPerRun"]    = (total_tokens_sent / runs_included) if runs_included else 0.0
    d["AvgTokRecvPerRun"]    = (total_tokens_received / runs_included) if runs_included else 0.0

    # elapsed time (exp_count is only runs with metric)
    d["AvgElapsedSPerRun"]   = (total_elapsed / exp_count) if exp_count else 0.0

    return d


# -------------------------
# Table2: Task-length comparison vs BASE
# -------------------------

def task_length_vs_base(
    base_runs: Dict[int, Dict[str, Any]],
    other_runs: Dict[int, Dict[str, Any]]
) -> Dict[str, int]:
    base_longer = 0
    other_longer = 0
    ties = 0
    excluded = 0

    common_ids_all = sorted(set(base_runs.keys()) & set(other_runs.keys()))
    for rid in common_ids_all:
        b = traj_len(base_runs[rid])
        o = traj_len(other_runs[rid])
        if b == 0 or o == 0:
            excluded += 1
            continue
        if b > o:
            base_longer += 1
        elif o > b:
            other_longer += 1
        else:
            ties += 1

    return {
        "CommonRuns": len(common_ids_all),
        "ExcludedZeroTaskRuns": excluded,
        "BASE_longer": base_longer,
        "OTHER_longer": other_longer,
        "Ties": ties,
    }


# -------------------------
# Writers
# -------------------------

TABLE1_FIELDS = [
    "Method",
    "Runs", "Tasks",
    "Acc", "PartialPlus", "Not", "Error", "Unknown",
    "AvgTasksPerRun",
    "AvgToolCallsPerTask", "AvgApiCallsPerTask", "AvgTokSentPerTask", "AvgTokRecvPerTask",
    "AvgToolCallsPerRun", "AvgApiCallsPerRun", "AvgTokSentPerRun", "AvgTokRecvPerRun",
    "AvgElapsedSPerRun",
]

TABLE2_FIELDS = [
    "Method",
    "CommonRuns", "ExcludedZeroTaskRuns",
    "BASE_longer", "Method_longer", "Ties",
]


def write_csv(path: str, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def f3(x: float) -> str:
    return f"{x:.3f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def f1(x: float) -> str:
    return f"{x:.1f}"


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_root", required=True)
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated, e.g., BASE,SPIRAL,SPIRAL_wo_sim,SPIRAL_wo_cri2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if "BASE" not in tags:
        raise ValueError("tags must include BASE (Table2 compares others vs BASE).")

    if args.debug:
        print(f"[DEBUG] trajectory_root={args.trajectory_root}")
        print(f"[DEBUG] exp_root={args.exp_root}")
        print(f"[DEBUG] out_dir={args.out_dir}")
        print(f"[DEBUG] model={args.model}")
        print(f"[DEBUG] tags={tags}")

    # Load all methods
    method_runs: Dict[str, Dict[int, Dict[str, Any]]] = {}
    method_exp: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for tag in tags:
        traj_dir = find_method_dir(args.trajectory_root, tag, args.model, debug=args.debug)
        exp_dir  = find_method_dir(args.exp_root, tag, args.model, debug=args.debug)

        if args.debug:
            print(f"[DEBUG] {tag}: traj_dir={traj_dir}")
            print(f"[DEBUG] {tag}: exp_dir={exp_dir}")

        runs = load_runs_from_dir(traj_dir, debug=args.debug)
        expm = load_exp_metrics(exp_dir, debug=args.debug)

        method_runs[tag] = runs
        method_exp[tag] = expm

        if args.debug and runs:
            # Print one sample to help schema debugging
            sample_id = sorted(runs.keys())[0]
            top_keys = list((runs[sample_id] or {}).keys())
            print(f"[DEBUG] {tag}: sample trajectory run_id={sample_id}, top_keys={top_keys}")

    # -------------------------
    # Table1
    # -------------------------
    table1_rows: List[Dict[str, Any]] = []
    for tag in tags:
        runs = method_runs[tag]
        expm = method_exp.get(tag)

        stats = overall_stats(runs, exp_metrics=expm)
        row: Dict[str, Any] = {"Method": bracket_tag(tag)}
        row.update(stats)

        # Format numeric fields for CSV readability (keep as strings)
        row["Runs"]  = int(row["Runs"])
        row["Tasks"] = int(row["Tasks"])

        for k in ["Acc", "PartialPlus", "Not", "Error", "Unknown"]:
            row[k] = f3(float(row[k]))

        row["AvgTasksPerRun"] = f2(float(row["AvgTasksPerRun"]))

        row["AvgToolCallsPerTask"] = f2(float(row["AvgToolCallsPerTask"]))
        row["AvgApiCallsPerTask"]  = f2(float(row["AvgApiCallsPerTask"]))
        row["AvgTokSentPerTask"]   = f1(float(row["AvgTokSentPerTask"]))
        row["AvgTokRecvPerTask"]   = f1(float(row["AvgTokRecvPerTask"]))

        row["AvgToolCallsPerRun"]  = f2(float(row["AvgToolCallsPerRun"]))
        row["AvgApiCallsPerRun"]   = f2(float(row["AvgApiCallsPerRun"]))
        row["AvgTokSentPerRun"]    = f1(float(row["AvgTokSentPerRun"]))
        row["AvgTokRecvPerRun"]    = f1(float(row["AvgTokRecvPerRun"]))

        row["AvgElapsedSPerRun"]   = f2(float(row["AvgElapsedSPerRun"]))

        table1_rows.append(row)

    out1 = os.path.join(args.out_dir, "table1_overall.csv")
    write_csv(out1, TABLE1_FIELDS, table1_rows)
    print(f"Wrote: {out1}")

    # -------------------------
    # Table2 (vs BASE)
    # -------------------------
    base_runs = method_runs["BASE"]

    table2_rows: List[Dict[str, Any]] = []
    for tag in tags:
        if tag == "BASE":
            continue
        other_runs = method_runs[tag]
        comp = task_length_vs_base(base_runs, other_runs)
        row = {
            "Method": bracket_tag(tag),
            "CommonRuns": comp["CommonRuns"],
            "ExcludedZeroTaskRuns": comp["ExcludedZeroTaskRuns"],
            "BASE_longer": comp["BASE_longer"],
            "Method_longer": comp["OTHER_longer"],
            "Ties": comp["Ties"],
        }
        table2_rows.append(row)

    out2 = os.path.join(args.out_dir, "table2_task_length_vs_base.csv")
    write_csv(out2, TABLE2_FIELDS, table2_rows)
    print(f"Wrote: {out2}")

    # Safety message if everything is zero (schema mismatch)
    all_tasks = sum(int(r["Tasks"]) for r in table1_rows)
    if all_tasks == 0:
        print("\n[WARNING] Total tasks across all methods is 0.")
        print("This usually means trajectory JSON schema differs (e.g., no 'trajectory' list),")
        print("or status reviews are stored under different keys.")
        print("Run with --debug and paste:")
        print("  - one trajectory json file path + its top-level keys")
        print("  - one task object's top-level keys + logs keys + where reviews live")


if __name__ == "__main__":
    main()
