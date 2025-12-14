#!/usr/bin/env python3
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

# -------------------------
# Data structures
# -------------------------

@dataclass
class RunStats:
    run_id: int
    num_tasks: int
    num_accomplished: int
    num_partial: int
    num_not: int


# -------------------------
# Helpers: load & filter
# -------------------------

def extract_run_id_from_filename(path: str) -> Optional[int]:
    """
    Expect filenames like Q_4_trajectory.json or Q_41_trajectory.json.
    Return the integer part (4, 41, ...) or None on failure.
    """
    base = os.path.basename(path)
    m = re.search(r"Q_(\d+)_trajectory\.json", base)
    if not m:
        return None
    return int(m.group(1))


def load_runs_from_dir(root: str) -> Dict[int, Dict[str, Any]]:
    """
    Load *only* Q_1..Q_12 and Q_41..Q_48 trajectory JSONs from a directory.
    Return dict[id] = run_json.
    """
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

        # Filter to 1–12 and 41–48
        if not ((1 <= run_id <= 12) or (41 <= run_id <= 48)):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        runs[run_id] = data

    print(f"Loaded {len(runs)} runs (IDs in 1–12, 41–48).")
    return runs


# -------------------------
# Status parsing
# -------------------------

# Regex: capture text after "Task Status:" up to newline
TASK_STATUS_RE = re.compile(r"task\s*status\s*:\s*(.+)", re.IGNORECASE)


def extract_status_from_task(task: Dict[str, Any],
                             run_id: Optional[int] = None,
                             task_number: Optional[int] = None) -> str:
    """
    Parse status from `task["reviews"]` or `task["logs"]["reviews"]`.

    Returns one of:
      - "Accomplished"
      - "Partially accomplished"
      - "Not accomplished"
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

        raw = m.group(1).strip()
        first_line = raw.splitlines()[0].strip()
        s = first_line.lower()

        if "partially accomplished" in s or "partial" in s:
            return "Partially accomplished"

        if "not accomplished" in s or "failed" in s or "failure" in s:
            return "Not accomplished"

        if "accomplished" in s:
            return "Accomplished"

        # Fallback: return the first line if it’s something unexpected
        return first_line

    return "Unknown"


def summarize_run(run_json: Dict[str, Any], run_id: Optional[int] = None) -> RunStats:
    """
    Given one trajectory JSON, count tasks by status.
    """
    traj = run_json.get("trajectory") or []
    if not isinstance(traj, list):
        traj = []

    num_tasks = len(traj)
    cnt = Counter()

    for t in traj:
        status = extract_status_from_task(
            t,
            run_id=run_id,
            task_number=t.get("task_number"),
        )
        cnt[status] += 1

    num_acc = cnt.get("Accomplished", 0)
    num_part = cnt.get("Partially accomplished", 0)
    num_not = cnt.get("Not accomplished", 0)

    return RunStats(
        run_id=run_id if run_id is not None else -1,
        num_tasks=num_tasks,
        num_accomplished=num_acc,
        num_partial=num_part,
        num_not=num_not,
    )


# -------------------------
# Comparison & printing
# -------------------------

def compare_task_lengths(baseline_runs: Dict[int, Dict[str, Any]],
                         spiral_runs: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    For common run IDs, compare number of tasks (trajectory length).
    Returns (baseline_longer, spiral_longer, tie).
    """
    baseline_longer = 0
    spiral_longer = 0
    tie = 0

    common_ids = sorted(set(baseline_runs.keys()) & set(spiral_runs.keys()))
    print(f"\nCommon run IDs: {len(common_ids)}\n")

    print("=== Task length comparison (baseline vs SPIRAL) ===")
    for rid in common_ids:
        b_traj = baseline_runs[rid].get("trajectory") or []
        s_traj = spiral_runs[rid].get("trajectory") or []
        b_len, s_len = len(b_traj), len(s_traj)

        if b_len > s_len:
            baseline_longer += 1
        elif s_len > b_len:
            spiral_longer += 1
        else:
            tie += 1

    print(f"baseline longer: {baseline_longer}")
    print(f"SPIRAL  longer: {spiral_longer}")
    print(f"ties           : {tie}\n")

    return baseline_longer, spiral_longer, tie


def overall_success_stats(runs: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Aggregate success stats across all runs.
    Returns (total_tasks, total_acc, total_partial, total_not).
    """
    total_tasks = 0
    total_acc = 0
    total_part = 0
    total_not = 0

    for rid, rjson in runs.items():
        st = summarize_run(rjson, run_id=rid)
        total_tasks += st.num_tasks
        total_acc += st.num_accomplished
        total_part += st.num_partial
        total_not += st.num_not

    return total_tasks, total_acc, total_part, total_not


def print_overall_stats(label: str, runs: Dict[int, Dict[str, Any]]) -> None:
    total_tasks, total_acc, total_part, total_not = overall_success_stats(runs)
    if total_tasks == 0:
        print(f"[{label}] no tasks.")
        return

    acc_rate = total_acc / total_tasks
    part_or_better = (total_acc + total_part) / total_tasks
    not_rate = total_not / total_tasks

    print(
        f"[{label}] tasks={total_tasks}, "
        f"Accomplished={total_acc} ({acc_rate:.3f}), "
        f"Partially={total_part} ({part_or_better:.3f} partial-or-better), "
        f"Not accomplished={total_not} ({not_rate:.3f})"
    )


def per_id_success_table(baseline_runs: Dict[int, Dict[str, Any]],
                         spiral_runs: Dict[int, Dict[str, Any]]) -> None:
    """
    Print per-ID success summary for common IDs.
    """
    common_ids = sorted(set(baseline_runs.keys()) & set(spiral_runs.keys()))
    print("\n=== Per-ID success counts (common IDs) ===")
    print("id | base_tasks acc part not | spiral_tasks acc part not")
    print("-" * 72)

    for rid in common_ids:
        b_stats = summarize_run(baseline_runs[rid], run_id=rid)
        s_stats = summarize_run(spiral_runs[rid], run_id=rid)

        print(
            f"{rid:3d} | "
            f"{b_stats.num_tasks:3d} {b_stats.num_accomplished:3d} {b_stats.num_partial:3d} {b_stats.num_not:3d} | "
            f"{s_stats.num_tasks:3d} {s_stats.num_accomplished:3d} {s_stats.num_partial:3d} {s_stats.num_not:3d}"
        )


# -------------------------
# Main
# -------------------------

def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print(
            "Usage: python analyze_experiments.py "
            "/path/to/baseline/trajectory "
            "/path/to/spiral/trajectory"
        )
        sys.exit(1)

    baseline_dir = argv[1]
    spiral_dir = argv[2]

    print(f"Loading baseline runs from: {baseline_dir}")
    baseline_runs = load_runs_from_dir(baseline_dir)

    print(f"\nLoading SPIRAL runs from: {spiral_dir}")
    spiral_runs = load_runs_from_dir(spiral_dir)

    # Compare lengths
    compare_task_lengths(baseline_runs, spiral_runs)

    # # Overall stats
    # print("\n=== Overall success statistics ===")
    # print_overall_stats("Baseline", baseline_runs)
    # print_overall_stats("SPIRAL", spiral_runs)

    # Per-ID table
    per_id_success_table(baseline_runs, spiral_runs)


if __name__ == "__main__":
    main(sys.argv)
