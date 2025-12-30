#!/usr/bin/env python3
import json
import os
import re
import sys
import glob
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


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
# Helpers: load & filter
# -------------------------

TRAJ_FNAME_RE = re.compile(r"Q_(\d+)_trajectory\.json$", re.IGNORECASE)
EXP_FNAME_RE = re.compile(r"_Q_(\d+)_time_token\.txt$", re.IGNORECASE)


def extract_run_id_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = TRAJ_FNAME_RE.search(base)
    return int(m.group(1)) if m else None


def load_runs_from_dir(root: str) -> Dict[int, Dict[str, Any]]:
    print(f"Loading runs from: {root}")
    runs: Dict[int, Dict[str, Any]] = {}

    if not os.path.isdir(root):
        print(f"[WARN] Directory not found: {root}")
        return runs

    for fname in os.listdir(root):
        if not fname.endswith("_trajectory.json"):
            continue
        path = os.path.join(root, fname)
        run_id = extract_run_id_from_filename(path)
        if run_id is None:
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        runs[run_id] = data

    print(f"Loaded {len(runs)} runs")
    return runs


EXP_FILE_RE = re.compile(r".*_Q_(\d+)_time_token\.txt$")

def load_exp_metrics(exp_root: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """
    Loads Model_*_Q_<qid>_time_token.txt JSON blobs.
    Returns: metrics[qid] = {"elapsed_seconds":..., "total_generated_tokens":..., ...}
    """
    metrics: Dict[int, Dict[str, Any]] = {}

    if not exp_root:
        return metrics
    if not os.path.isdir(exp_root):
        print(f"[WARN] exp dir not found: {exp_root}")
        return metrics

    for fname in os.listdir(exp_root):
        if not fname.endswith("_time_token.txt"):
            continue
        m = EXP_FILE_RE.match(fname)
        if not m:
            continue
        qid = int(m.group(1))
        path = os.path.join(exp_root, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            metrics[qid] = obj
        except Exception as e:
            print(f"[WARN] failed to load exp metric {path}: {e}")

    print(f"Loaded {len(metrics)} exp metric files from: {exp_root}")
    return metrics



# -------------------------
# Status parsing
# -------------------------

TASK_STATUS_RE = re.compile(r"task\s*status\s*:\s*(.+)", re.IGNORECASE)


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

        # Important: check "error" explicitly (your case: "Task Status: Error")
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
# Effort extraction
# -------------------------

TOOL_NAME_IN_ACTION_RE = re.compile(r"Tool Name:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)


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
      1) logs["trajectroy_log"] (step dicts with "action")
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

def combined_tokens(st: RunStats) -> Tuple[int, int]:
    exp_in = st.exp_input_tokens or 0
    exp_gen = st.exp_generated_tokens or 0
    return st.tokens_sent + exp_in, st.tokens_received + exp_gen

def summarize_run(run_json: Dict[str, Any],
                  run_id: int,
                  exp_metrics: Optional[Dict[int, Dict[str, Any]]] = None) -> RunStats:
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

    # Attach exp metrics if available
    elapsed = None
    exp_in = None
    exp_gen = None
    if exp_metrics and run_id in exp_metrics:
        obj = exp_metrics[run_id]
        elapsed = obj.get("elapsed_seconds")
        exp_in = obj.get("total_input_tokens")
        exp_gen = obj.get("total_generated_tokens")

        # normalize types
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


# -------------------------
# Aggregate stats & printing
# -------------------------

def compare_task_lengths(baseline_runs: Dict[int, Dict[str, Any]],
                         spiral_runs: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int]:
    baseline_longer = 0
    spiral_longer = 0
    tie = 0

    common_ids = sorted(set(baseline_runs.keys()) & set(spiral_runs.keys()))
    print(f"\nCommon run IDs: {len(common_ids)}\n")

    print("=== Task length comparison (baseline vs SPIRAL) ===")
    for rid in common_ids:
        b_len = len(baseline_runs[rid].get("trajectory") or [])
        s_len = len(spiral_runs[rid].get("trajectory") or [])

        if b_len > s_len:
            baseline_longer += 1
        elif s_len > b_len:
            spiral_longer += 1
        else:
            tie += 1

    print(f"SPIRAL longer: {baseline_longer}")
    print(f"SPIRAL without critic longer: {spiral_longer}")
    print(f"ties           : {tie}\n")
    return baseline_longer, spiral_longer, tie


def overall_stats(runs: Dict[int, Dict[str, Any]],
                  exp_metrics: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, float]:
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
    

    total_elapsed = 0.0
    total_exp_in = 0
    total_exp_gen = 0
    exp_count = 0

    for rid, rjson in runs.items():
        st = summarize_run(rjson, run_id=rid, exp_metrics=exp_metrics)
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
        if st.exp_input_tokens is not None:
            total_exp_in += st.exp_input_tokens
        if st.exp_generated_tokens is not None:
            total_exp_gen += st.exp_generated_tokens

    d: Dict[str, float] = {}
    d["total_tasks"] = float(total_tasks)
    d["acc_rate"] = (total_acc / total_tasks) if total_tasks else 0.0
    d["part_or_better_rate"] = ((total_acc + total_part) / total_tasks) if total_tasks else 0.0
    d["not_rate"] = (total_not / total_tasks) if total_tasks else 0.0
    d["err_rate"] = (total_err / total_tasks) if total_tasks else 0.0
    d["unk_rate"] = (total_unk / total_tasks) if total_tasks else 0.0

    d["tool_calls_per_task"] = (total_tool_calls / total_tasks) if total_tasks else 0.0
    d["api_calls_per_task"] = (total_api_calls / total_tasks) if total_tasks else 0.0
    d["tokens_sent_per_task"] = (total_tokens_sent / total_tasks) if total_tasks else 0.0
    d["tokens_received_per_task"] = (total_tokens_received / total_tasks) if total_tasks else 0.0

    d["total_tool_calls"] = float(total_tool_calls)
    d["total_api_calls"] = float(total_api_calls)
    d["total_tokens_sent"] = float(total_tokens_sent)
    d["total_tokens_received"] = float(total_tokens_received)

    d["exp_count"] = float(exp_count)
    d["total_elapsed_seconds"] = float(total_elapsed)
    d["avg_elapsed_seconds_per_run"] = (total_elapsed / exp_count) if exp_count else 0.0
    d["total_exp_input_tokens"] = float(total_exp_in)
    d["total_exp_generated_tokens"] = float(total_exp_gen)
    return d


def print_overall_stats(label: str,
                        runs: Dict[int, Dict[str, Any]],
                        exp_metrics: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
    d = overall_stats(runs, exp_metrics=exp_metrics)
    total_tasks = int(d["total_tasks"])

    print(
        f"[{label}] tasks={total_tasks}, "
        f"Acc={d['acc_rate']:.3f}, "
        f"Partial-or-better={d['part_or_better_rate']:.3f}, "
        f"Not={d['not_rate']:.3f}, "
        f"Error={d['err_rate']:.3f}, "
        f"Unknown={d['unk_rate']:.3f}"
    )

    print(
        f"[{label}] avg/tool_calls per task={d['tool_calls_per_task']:.2f}, "
        f"avg/api_calls per task={d['api_calls_per_task']:.2f}, "
        f"avg/tokens_sent per task={d['tokens_sent_per_task']:.1f}, "
        f"avg/tokens_received per task={d['tokens_received_per_task']:.1f}"
    )

    if exp_metrics:
        print(
            f"[{label}] exp_runs_with_metrics={int(d['exp_count'])}, "
            f"avg_elapsed_s/run={d['avg_elapsed_seconds_per_run']:.2f}, "
            f"total_elapsed_s={d['total_elapsed_seconds']:.2f}, "
            f"total_exp_input_tokens={int(d['total_exp_input_tokens'])}, "
            f"total_exp_generated_tokens={int(d['total_exp_generated_tokens'])}"
        )


def per_id_success_table(baseline_runs: Dict[int, Dict[str, Any]],
                         spiral_runs: Dict[int, Dict[str, Any]],
                         baseline_exp: Optional[Dict[int, Dict[str, Any]]] = None,
                         spiral_exp: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
    common_ids = sorted(set(baseline_runs.keys()) & set(spiral_runs.keys()))
    print("\n=== Per-ID success & effort (common IDs) ===")
    print("id | base: tasks acc part not err unk | tools api toks_s toks_r | spiral: tasks acc part not err unk | tools api toks_s toks_r")
    print("-" * 140)

    for rid in common_ids:
        b = summarize_run(baseline_runs[rid], run_id=rid, exp_metrics=baseline_exp)
        s = summarize_run(spiral_runs[rid], run_id=rid, exp_metrics=spiral_exp)

        b_tot_s, b_tot_r = combined_tokens(b)
        s_tot_s, s_tot_r = combined_tokens(s)

        print(
            f"{rid:3d} | "
            f"{b.num_tasks:3d} {b.num_accomplished:3d} {b.num_partial:3d} {b.num_not:3d} {b.num_error:3d} {b.num_unknown:3d} | "
            f"{b.tool_calls:4d} {b.api_calls:3d} {b_tot_s:6d} {b_tot_r:6d} | "
            f"{s.num_tasks:3d} {s.num_accomplished:3d} {s.num_partial:3d} {s.num_not:3d} {s.num_error:3d} {s.num_unknown:3d} | "
            f"{s.tool_calls:4d} {s.api_calls:3d} {s_tot_s:6d} {s_tot_r:6d}"
        )



# -------------------------
# Main
# -------------------------

def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_traj_dir", help="Path to baseline trajectory directory")
    parser.add_argument("spiral_traj_dir", help="Path to SPIRAL trajectory directory")
    parser.add_argument("--baseline_exp_dir", default=None, help="Optional path to baseline exp/*_time_token.txt directory")
    parser.add_argument("--spiral_exp_dir", default=None, help="Optional path to SPIRAL exp/*_time_token.txt directory")
    args = parser.parse_args(argv[1:])

    baseline_runs = load_runs_from_dir(args.baseline_traj_dir)
    spiral_runs = load_runs_from_dir(args.spiral_traj_dir)

    baseline_exp = load_exp_metrics(args.baseline_exp_dir) if args.baseline_exp_dir else None
    spiral_exp = load_exp_metrics(args.spiral_exp_dir) if args.spiral_exp_dir else None

    compare_task_lengths(baseline_runs, spiral_runs)

    print("\n=== Overall success + effort statistics ===")
    print_overall_stats("SPIRAL", baseline_runs, exp_metrics=baseline_exp)
    print_overall_stats("SPIRAL without critic", spiral_runs, exp_metrics=spiral_exp)

    per_id_success_table(baseline_runs, spiral_runs, baseline_exp=baseline_exp, spiral_exp=spiral_exp)


if __name__ == "__main__":
    main(sys.argv)
