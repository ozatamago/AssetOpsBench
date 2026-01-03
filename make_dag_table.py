#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from pathlib import Path
from collections import Counter
from typing import Iterable, Dict, Any, List, Tuple


# ============================================================
# 1) Core validator (あなたがOKした版: if/checkごとにRULE_IDを付与)
# ============================================================

def _validate_plan_text(plan_text: str, agents_allowed: Iterable[str]) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Validate a single plan text.

    Returns:
      ok: bool
      errors: list of dicts, each like:
        {"rule": "<RULE_ID>", "message": "<human-readable message>"}

    IMPORTANT:
      - Each "if"/check is treated as one error *type* via RULE_ID.
      - Directory-level counting is done by RULE_ID presence per file (max once per file).
    """
    TASK_RE   = re.compile(r"^#Task(\d+): (.+)$", re.M)
    AGENT_RE  = re.compile(r"^#Agent(\d+): (.+)$", re.M)
    DEP_RE    = re.compile(r"^#Dependency(\d+): (.+)$", re.M)
    OUT_RE    = re.compile(r"^#ExpectedOutput(\d+): (.+)$", re.M)
    DEP_TOKEN = re.compile(r"#S(\d+)")

    errors: List[Dict[str, str]] = []

    tks  = TASK_RE.findall(plan_text)
    ags  = AGENT_RE.findall(plan_text)
    deps = DEP_RE.findall(plan_text)
    outs = OUT_RE.findall(plan_text)

    def add(rule: str, message: str) -> None:
        errors.append({"rule": rule, "message": message})

    def _check_seq(pairs, label: str) -> None:
        if not pairs:
            add(f"{label.upper()}_LINES_MISSING", f"{label} lines missing")
            return

        nums = [int(n) for n, _ in pairs]
        if nums != list(range(1, len(nums) + 1)):
            add(
                f"{label.upper()}_NUMBERS_NOT_SEQ",
                f"{label} numbers must be 1..N in order; got {nums}",
            )

    _check_seq(tks,  "Task")
    _check_seq(ags,  "Agent")
    _check_seq(deps, "Dependency")
    _check_seq(outs, "ExpectedOutput")

    if len({len(tks), len(ags), len(deps), len(outs)}) != 1:
        add(
            "COUNTS_MISMATCH",
            "Counts of Task/Agent/Dependency/ExpectedOutput must match",
        )

    # Dependency checks
    if tks and deps:
        total = len(tks)
        for n_str, dep in deps:
            n = int(n_str)
            dep = dep.strip()

            if dep == "None":
                continue

            nums = [int(x) for x in DEP_TOKEN.findall(dep)]

            if not nums:
                add(
                    "DEPENDENCY_FORMAT_INVALID",
                    f"Dependency{n} must be 'None' or '#S1 #S2 ...'; got '{dep}'",
                )
                continue

            bad = [k for k in nums if k < 1 or k > total]
            if bad:
                add(
                    "DEPENDENCY_OUT_OF_RANGE",
                    f"Dependency{n} out of range {bad}; valid 1..{total}",
                )

            fwd = [k for k in nums if k >= n]
            if fwd:
                add(
                    "DEPENDENCY_FORWARD_REFERENCE",
                    f"Dependency{n} forward reference {fwd}; only past steps allowed",
                )

    # Agent name validation
    valid = set(agents_allowed)
    for n_str, name in AGENT_RE.findall(plan_text):
        if name not in valid:
            add(
                "AGENT_UNKNOWN",
                f"Agent{n_str} unknown '{name}'. Allowed: {sorted(valid)}",
            )

    return (len(errors) == 0, errors)


# ============================================================
# 2) Directory summarizer (分母はそのdirの txt_total)
# ============================================================

def summarize_one_directory(
    dir_path: str,
    agents_allowed: Iterable[str],
    pattern: str = "*.txt",
    recursive: bool = False,
) -> Dict[str, Any]:
    """
    Summarize plan validation results for one directory.

    Counting policy (important):
      - Denominator is the directory's txt_total (number of matched files).
      - For each file, each RULE_ID is counted at most once
        (i.e., file-level presence/absence of that rule).

    Returns dict with:
      - dir, txt_total, ok_files, success_rate
      - error_counts: {RULE_ID: count}
      - error_formatted: {RULE_ID: "count (pct%)"}
    """
    root = Path(dir_path)
    if not root.exists():
        return {
            "dir": str(root),
            "txt_total": 0,
            "ok_files": 0,
            "success_rate": 0.0,
            "error_counts": {},
            "error_formatted": {},
            "error": f"Directory not found: {dir_path}",
        }
    if not root.is_dir():
        return {
            "dir": str(root),
            "txt_total": 0,
            "ok_files": 0,
            "success_rate": 0.0,
            "error_counts": {},
            "error_formatted": {},
            "error": f"Not a directory: {dir_path}",
        }

    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    txt_total = len(files)

    ok_files = 0
    rule_counter: Counter = Counter()

    for fp in files:
        if not fp.is_file():
            continue

        try:
            text = fp.read_text(encoding="utf-8")
        except Exception:
            # I/O error is counted as one rule type
            rule_counter.update(["IO_ERROR"])
            continue

        ok, errs = _validate_plan_text(text, agents_allowed)
        if ok:
            ok_files += 1
            continue

        # Count "rule types" per file (each RULE_ID once per file)
        rules_in_this_file = {e["rule"] for e in errs if isinstance(e, dict) and "rule" in e}
        rule_counter.update(rules_in_this_file)

    success_rate = (ok_files / txt_total) if txt_total else 0.0

    error_counts = dict(rule_counter)
    error_formatted: Dict[str, str] = {}
    for rule, cnt in sorted(error_counts.items(), key=lambda x: (-x[1], x[0])):
        pct = (cnt / txt_total * 100.0) if txt_total else 0.0
        error_formatted[rule] = f"{cnt} ({pct:.1f}%)"

    return {
        "dir": str(root),
        "txt_total": txt_total,
        "ok_files": ok_files,
        "success_rate": success_rate,
        "error_counts": error_counts,
        "error_formatted": error_formatted,
        "error": None,
    }


# ============================================================
# 3) CSV column naming: [BASE]Model_16 -> B#16, [VALID]Model_20 -> V#20
# ============================================================

DIRNAME_RE = re.compile(r"^\[(?P<tag>[^\]]+)\]Model_(?P<mid>\d+)$")

def dir_to_column_name(dir_path: str) -> str:
    name = Path(dir_path).name
    m = DIRNAME_RE.match(name)
    if not m:
        return name

    tag = m.group("tag").strip().upper()
    mid = m.group("mid").strip()

    if tag.startswith("BASE"):
        prefix = "B"
    elif tag.startswith("VALID"):
        prefix = "V"
    else:
        prefix = tag[:1] if tag else "X"

    return f"{prefix}#{mid}"


# ============================================================
# 4) Printing + table building
# ============================================================

def print_dir_summary(summary: Dict[str, Any], col: str, topk: int = 12, show_pct: bool = True) -> None:
    if summary.get("error"):
        print(f"\n[{col}] ERROR: {summary['error']}")
        return

    txt_total = summary["txt_total"]
    ok_files = summary["ok_files"]
    success_rate = summary["success_rate"]
    err_counts = Counter(summary.get("error_counts", {}))

    print(f"\n[{col}] {summary['dir']}")
    print("1) Error type counts (top):")
    if not err_counts:
        print("   (no errors)")
    else:
        for rule, cnt in err_counts.most_common(topk):
            if show_pct and txt_total:
                print(f"   - {rule}: {cnt} ({cnt/txt_total*100:.1f}%)")
            else:
                print(f"   - {rule}: {cnt}")

    print("2) Summary:")
    print(f"   TXT files: {txt_total}")
    print(f"   OK files:  {ok_files}")
    print(f"   OK ratio:  {success_rate:.3f}")


def build_wide_csv_rows(
    per_dir: List[Dict[str, Any]],
    cols_in_order: List[str],
    show_pct: bool = True,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Output format:
      Metric,B#16,B#20,...,V#38

    Rows:
      success_rate
      txt_total
      ok_files
      error::<RULE_ID>   (count or "count (pct%)", pct denom = txt_total)
    """
    fieldnames = ["Metric"] + cols_in_order

    # Union of all rules across all dirs, ordered by global frequency
    global_counter = Counter()
    for d in per_dir:
        if d.get("error"):
            continue
        global_counter.update(d.get("error_counts", {}))

    ordered_rules = [r for r, _ in global_counter.most_common()]

    rows: List[Dict[str, str]] = []

    # success_rate
    r = {"Metric": "success_rate"}
    for d in per_dir:
        col = d["col"]
        r[col] = f"{d['success_rate']:.3f}" if not d.get("error") else ""
    rows.append(r)

    # txt_total
    r = {"Metric": "txt_total"}
    for d in per_dir:
        col = d["col"]
        r[col] = str(d["txt_total"]) if not d.get("error") else ""
    rows.append(r)

    # ok_files
    r = {"Metric": "ok_files"}
    for d in per_dir:
        col = d["col"]
        r[col] = str(d["ok_files"]) if not d.get("error") else ""
    rows.append(r)

    # error rows
    for rule in ordered_rules:
        r = {"Metric": f"error::{rule}"}
        for d in per_dir:
            col = d["col"]
            if d.get("error"):
                r[col] = ""
                continue
            cnt = int(d.get("error_counts", {}).get(rule, 0))
            denom = int(d["txt_total"])
            if show_pct and denom:
                r[col] = f"{cnt} ({cnt/denom*100:.1f}%)"
            else:
                r[col] = str(cnt)
        rows.append(r)

    return fieldnames, rows


def write_csv(csv_out: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    outp = Path(csv_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# 5) Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch validate DAG plan *.txt under multiple directories and emit a wide CSV for the LaTeX table."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="List of plan directories (ABSOLUTE paths recommended). Example: .../plan/[BASE]Model_16",
    )
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories (default: false).")
    parser.add_argument("--agents", nargs="*", default=None, help="Override allowed agents (space-separated).")
    parser.add_argument("--topk", type=int, default=12, help="Show top-K error rule types per directory.")
    parser.add_argument("--csv_out", required=True, help="Output CSV path, e.g., .../dag_success_validate_plan_text.csv")
    parser.add_argument("--no_pct", action="store_true", help="Counts only (no percentages) in error rows.")

    args = parser.parse_args()

    default_agents = [
        "IoT Data Download",
        "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
        "Time Series Analytics and Forecasting",
        "WorkOrder Agent",
    ]
    agents_allowed = args.agents if args.agents else default_agents

    # Summarize each directory in the order provided
    per_dir: List[Dict[str, Any]] = []
    cols_in_order: List[str] = []

    for dp in args.dirs:
        col = dir_to_column_name(dp)
        cols_in_order.append(col)

        s = summarize_one_directory(dp, agents_allowed, pattern="*.txt", recursive=args.recursive)
        s["col"] = col
        per_dir.append(s)

        print_dir_summary(s, col=col, topk=args.topk, show_pct=(not args.no_pct))

    # Build wide table and write CSV
    fieldnames, rows = build_wide_csv_rows(per_dir, cols_in_order=cols_in_order, show_pct=(not args.no_pct))
    write_csv(args.csv_out, fieldnames, rows)

    print(f"\nWrote CSV: {args.csv_out}")
    print("Columns:", ", ".join(fieldnames))


if __name__ == "__main__":
    main()
