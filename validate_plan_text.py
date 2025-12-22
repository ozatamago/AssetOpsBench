import re
from pathlib import Path
from typing import Iterable, List, Dict, Any
import argparse

def _validate_plan_text(plan_text: str, agents_allowed):
    # NOTE: standalone script => no "self" argument
    TASK_RE   = re.compile(r"^#Task(\d+): (.+)$", re.M)
    AGENT_RE  = re.compile(r"^#Agent(\d+): (.+)$", re.M)
    DEP_RE    = re.compile(r"^#Dependency(\d+): (.+)$", re.M)
    OUT_RE    = re.compile(r"^#ExpectedOutput(\d+): (.+)$", re.M)
    DEP_TOKEN = re.compile(r"#S(\d+)")

    errors = []
    tks  = TASK_RE.findall(plan_text)
    ags  = AGENT_RE.findall(plan_text)
    deps = DEP_RE.findall(plan_text)
    outs = OUT_RE.findall(plan_text)

    def _check_seq(pairs, label):
        if not pairs:
            errors.append(f"{label} lines missing")
            return
        nums = [int(n) for n, _ in pairs]
        if nums != list(range(1, len(nums) + 1)):
            errors.append(f"{label} numbers must be 1..N in order; got {nums}")

    _check_seq(tks,  "Task")
    _check_seq(ags,  "Agent")
    _check_seq(deps, "Dependency")
    _check_seq(outs, "ExpectedOutput")

    if len({len(tks), len(ags), len(deps), len(outs)}) != 1:
        errors.append("Counts of Task/Agent/Dependency/ExpectedOutput must match")

    if tks and deps:
        total = len(tks)
        for n, dep in deps:
            n = int(n)
            dep = dep.strip()
            if dep == "None":
                continue
            nums = [int(x) for x in DEP_TOKEN.findall(dep)]
            if not nums:
                errors.append(f"Dependency{n} must be 'None' or '#S1 #S2 ...'; got '{dep}'")
                continue
            bad = [k for k in nums if k < 1 or k > total]
            if bad:
                errors.append(f"Dependency{n} out of range {bad}; valid 1..{total}")
            fwd = [k for k in nums if k >= n]
            if fwd:
                errors.append(f"Dependency{n} forward reference {fwd}; only past steps allowed")

    valid = set(agents_allowed)
    for n, name in AGENT_RE.findall(plan_text):
        if name not in valid:
            errors.append(f"Agent{n} unknown '{name}'. Allowed: {sorted(valid)}")

    return (len(errors) == 0, errors)


def validate_plans_in_dir(dir_path: str, agents_allowed: Iterable[str], recursive: bool = False) -> Dict[str, Any]:
    root = Path(dir_path)
    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(root.glob(pattern))

    results: List[Dict[str, Any]] = []
    ok_count = 0

    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            results.append({"path": str(p), "ok": False, "errors": [f"I/O error: {e}"]})
            continue

        ok, errs = _validate_plan_text(text, agents_allowed)
        results.append({"path": str(p), "ok": ok, "errors": errs})
        if ok:
            ok_count += 1

    total = len(files)
    ok_ratio = (ok_count / total) if total else 0.0
    return {"total": total, "ok_count": ok_count, "ok_ratio": ok_ratio, "results": results}


def check_one_level_subdirs_filecount(dir_path: str, max_files_allowed: int = 1) -> Dict[str, Any]:
    """
    Check immediate subdirectories (1-level depth) under dir_path.
    For each subdir, count *files directly inside it* (no recursion).
    If file_count >= 2 (i.e., > max_files_allowed when max_files_allowed=1), mark it as NG.

    Uses Path.iterdir() to list directory contents without recursion. :contentReference[oaicite:1]{index=1}
    """
    root = Path(dir_path)
    if not root.exists():
        return {"root": str(root), "error": "root path does not exist", "ng": [], "ok": []}
    if not root.is_dir():
        return {"root": str(root), "error": "root path is not a directory", "ng": [], "ok": []}

    ng: List[Dict[str, Any]] = []
    ok: List[Dict[str, Any]] = []

    # iterdir() yields immediate children only (1-level), in arbitrary order. :contentReference[oaicite:2]{index=2}
    for child in root.iterdir():
        if not child.is_dir():
            continue

        # Count only direct files inside this subdir (no recursion).
        file_count = 0
        for p in child.iterdir():
            if p.is_file():
                file_count += 1

        item = {"dir": str(child), "file_count": file_count}
        if file_count > max_files_allowed:   # "2つ以上ならNG" => max_files_allowed=1
            ng.append(item)
        else:
            ok.append(item)

    return {"root": str(root), "max_files_allowed": max_files_allowed, "ng": ng, "ok": ok}

from pathlib import Path
from typing import Iterable, List, Dict, Any

def validate_suffix0_in_immediate_subdirs(
    dir_path: str,
    agents_allowed: Iterable[str],
    filename_suffix: str = "_0.txt",
) -> Dict[str, Any]:
    """
    Scan *only one level* of subdirectories under dir_path, and validate files
    whose names end with filename_suffix (default: '_0.txt') using _validate_plan_text().

    Returns a summary dict with per-directory and per-file validation results.
    """
    root = Path(dir_path)
    if not root.exists():
        return {"error": f"Directory not found: {dir_path}", "total_dirs": 0, "total_files": 0, "ok_count": 0, "results": []}
    if not root.is_dir():
        return {"error": f"Not a directory: {dir_path}", "total_dirs": 0, "total_files": 0, "ok_count": 0, "results": []}

    # Only immediate children directories (one-level scan).
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    all_items: List[Dict[str, Any]] = []
    total_files = 0
    ok_count = 0

    for sd in subdirs:
        # Match only files ending with *_0.txt inside this subdir (non-recursive).
        files = sorted([p for p in sd.glob(f"*{filename_suffix}") if p.is_file()])

        dir_items: List[Dict[str, Any]] = []
        dir_ok = 0

        for fp in files:
            total_files += 1
            try:
                text = fp.read_text(encoding="utf-8")
            except Exception as e:
                dir_items.append({"path": str(fp), "ok": False, "errors": [f"I/O error: {e}"]})
                continue

            ok, errs = _validate_plan_text(text, agents_allowed)
            dir_items.append({"path": str(fp), "ok": ok, "errors": errs})
            if ok:
                ok_count += 1
                dir_ok += 1

        all_items.append({
            "subdir": str(sd),
            "matched_suffix": filename_suffix,
            "files": dir_items,
            "dir_total": len(files),
            "dir_ok": dir_ok,
            "dir_ok_ratio": (dir_ok / len(files)) if files else 0.0,
        })

    return {
        "total_dirs": len(subdirs),
        "total_files": total_files,
        "ok_count": ok_count,
        "ok_ratio": (ok_count / total_files) if total_files else 0.0,
        "results": all_items,
    }

def main():
    parser = argparse.ArgumentParser(description="Validate plan *.txt files and check subdir file counts.")
    parser.add_argument("dir", help="Directory containing plan .txt files (and optionally subdirs to check)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories for .txt validation")
    parser.add_argument("--agents", nargs="*", default=None, help="Override allowed agents (space-separated).")
    parser.add_argument("--check-subdirs", action="store_true", help="Check 1-level subdirectories file counts (>=2 => NG).")
    parser.add_argument("--max-files", type=int, default=1, help="Max files allowed in each immediate subdirectory (default: 1).")
    args = parser.parse_args()

    default_agents = [
        "IoT Data Download",
        "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
        "Time Series Analytics and Forecasting",
        "WorkOrder Agent",
    ]
    agents_allowed = args.agents if args.agents else default_agents

    summary = validate_plans_in_dir(args.dir, agents_allowed, recursive=args.recursive)
    print(f"TXT files: {summary['total']}")
    print(f"OK files:  {summary['ok_count']}")
    print(f"OK ratio:  {summary['ok_ratio']:.3f}")

    for item in summary["results"]:
        flag = "OK " if item["ok"] else "NG "
        print(f"{flag} {item['path']}")
        if not item["ok"]:
            for e in item["errors"]:
                print(f"  - {e}")

    if args.check_subdirs:
        sub = check_one_level_subdirs_filecount(args.dir, max_files_allowed=args.max_files)
        print("\n[Subdirectory file count check (1-level)]")
        if sub.get("error"):
            print(f"NG  root={sub['root']}  reason={sub['error']}")
        else:
            if not sub["ng"]:
                print("OK  No subdirectories exceed the file count limit.")
            else:
                for d in sub["ng"]:
                    print(f"NG  {d['dir']}  file_count={d['file_count']} (limit={sub['max_files_allowed']})")

    #     # --- NEW: one-level subdir scan for *_0.txt files ---
    # subdir_summary = validate_suffix0_in_immediate_subdirs(args.dir, agents_allowed, filename_suffix="_0.txt")

    # print("\n=== One-level subdir check: files ending with _0.txt ===")
    # print(f"Subdirs scanned: {subdir_summary.get('total_dirs', 0)}")
    # print(f"Matched files:   {subdir_summary.get('total_files', 0)}")
    # print(f"OK files:        {subdir_summary.get('ok_count', 0)}")
    # print(f"OK ratio:        {subdir_summary.get('ok_ratio', 0.0):.3f}")

    # # Optional: per-subdir details
    # for d in subdir_summary.get("results", []):
    #     print(f"\n[DIR] {d['subdir']}  (matched: {d['dir_total']}, ok: {d['dir_ok']}, ratio: {d['dir_ok_ratio']:.3f})")
    #     for item in d["files"]:
    #         flag = "OK " if item["ok"] else "NG "
    #         print(f"  {flag} {item['path']}")
    #         if not item["ok"]:
    #             for e in item["errors"]:
    #                 print(f"    - {e}")


if __name__ == "__main__":
    main()
