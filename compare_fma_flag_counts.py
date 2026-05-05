import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare overlapping-qid FMA reports in two directories and "
            "print total flag counts and per-flag counts for each directory."
        )
    )
    parser.add_argument(
        "dir_a",
        type=str,
        help="First FMA report directory (e.g. oracle_verify_recovery).",
    )
    parser.add_argument(
        "dir_b",
        type=str,
        help="Second FMA report directory (e.g. no_verify).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search JSON files recursively.",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Also print the final summary as JSON.",
    )
    return parser.parse_args()


def extract_qid_from_filename(path: Path) -> Optional[str]:
    patterns = [
        r"Q[_-]?(\d+)",
        r"qid[_-]?(\d+)",
        r"(\d+)",
    ]
    name = path.stem
    for pattern in patterns:
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1)))
    return None


def extract_qid_from_data(data: Dict[str, Any], path: Path) -> Optional[str]:
    # Common direct keys
    for key in ("qid", "QID", "question_id", "id"):
        if key in data:
            value = data[key]
            if isinstance(value, (int, str)):
                text = str(value).strip()
                m = re.search(r"(\d+)", text)
                if m:
                    return str(int(m.group(1)))

    # Fallback to filename
    return extract_qid_from_filename(path)


def normalize_flag_record(raw_flag: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_flag, dict):
        return None
    flag_name = raw_flag.get("flag")
    if not isinstance(flag_name, str) or not flag_name.strip():
        return None

    return {
        "flag": flag_name.strip(),
        "subject": raw_flag.get("subject"),
        "reason": raw_flag.get("reason"),
        "evidence_refs": raw_flag.get("evidence_refs"),
    }


def extract_all_flags(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Preferred source
    if isinstance(data.get("all_flags"), list):
        out = []
        for raw in data["all_flags"]:
            item = normalize_flag_record(raw)
            if item is not None:
                out.append(item)
        return out

    # Fallback: flags_by_layer
    if isinstance(data.get("flags_by_layer"), dict):
        out = []
        for _, flags in data["flags_by_layer"].items():
            if isinstance(flags, list):
                for raw in flags:
                    item = normalize_flag_record(raw)
                    if item is not None:
                        out.append(item)
        return out

    # Fallback: flags_by_node
    if isinstance(data.get("flags_by_node"), dict):
        out = []
        for _, flags in data["flags_by_node"].items():
            if isinstance(flags, list):
                for raw in flags:
                    item = normalize_flag_record(raw)
                    if item is not None:
                        out.append(item)
        return out

    return []


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scan_report_dir(report_dir: Path, recursive: bool) -> Dict[str, Path]:
    if not report_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {report_dir}")
    if not report_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {report_dir}")

    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(report_dir.glob(pattern))

    qid_to_path: Dict[str, Path] = {}
    for path in paths:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[warning] failed to read JSON: {path} ({e})", file=sys.stderr)
            continue

        qid = extract_qid_from_data(data, path)
        if qid is None:
            print(f"[warning] could not extract qid: {path}", file=sys.stderr)
            continue

        if qid in qid_to_path:
            print(
                f"[warning] duplicate qid={qid} in {report_dir}. "
                f"Keeping first: {qid_to_path[qid]}, skipping: {path}",
                file=sys.stderr,
            )
            continue

        qid_to_path[qid] = path

    return qid_to_path


def sort_flag_key(flag_name: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", flag_name)
    if nums:
        return tuple(int(x) for x in nums)
    return (10**9,)


def summarize_for_qids(qid_to_path: Dict[str, Path], target_qids: List[str]) -> Dict[str, Any]:
    total_flags = 0
    per_flag = Counter()
    per_qid: Dict[str, int] = {}

    for qid in target_qids:
        path = qid_to_path[qid]
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[warning] failed to re-read JSON: {path} ({e})", file=sys.stderr)
            continue

        flags = extract_all_flags(data)
        count = len(flags)
        per_qid[qid] = count
        total_flags += count

        for item in flags:
            per_flag[item["flag"]] += 1

    return {
        "num_overlap_qids": len(target_qids),
        "total_flags": total_flags,
        "per_flag": dict(sorted(per_flag.items(), key=lambda kv: sort_flag_key(kv[0]))),
        "per_qid_total_flags": dict(sorted(per_qid.items(), key=lambda kv: int(kv[0]))),
    }


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print(label)
    print("-" * 80)
    print(f"overlapping qids used: {summary['num_overlap_qids']}")
    print(f"total flag count: {summary['total_flags']}")
    print("per-flag totals:")

    per_flag: Dict[str, int] = summary["per_flag"]
    if not per_flag:
        print("  (none)")
    else:
        for flag_name, count in per_flag.items():
            print(f"  {flag_name}: {count}")


def main() -> int:
    args = parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    qid_to_path_a = scan_report_dir(dir_a, recursive=args.recursive)
    qid_to_path_b = scan_report_dir(dir_b, recursive=args.recursive)

    qids_a = set(qid_to_path_a.keys())
    qids_b = set(qid_to_path_b.keys())
    overlap_qids = sorted(qids_a & qids_b, key=int)

    print(f"[info] dir_a: {dir_a}")
    print(f"[info] dir_b: {dir_b}")
    print(f"[info] qids in dir_a: {len(qids_a)}")
    print(f"[info] qids in dir_b: {len(qids_b)}")
    print(f"[info] overlapping qids: {len(overlap_qids)}")

    if not overlap_qids:
        print("[error] no overlapping qids found.", file=sys.stderr)
        return 1

    summary_a = summarize_for_qids(qid_to_path_a, overlap_qids)
    summary_b = summarize_for_qids(qid_to_path_b, overlap_qids)

    print_summary("Directory A summary", summary_a)
    print_summary("Directory B summary", summary_b)

    if args.pretty_json:
        final_summary = {
            "dir_a": str(dir_a),
            "dir_b": str(dir_b),
            "num_qids_dir_a": len(qids_a),
            "num_qids_dir_b": len(qids_b),
            "num_overlap_qids": len(overlap_qids),
            "overlap_qids": overlap_qids,
            "summary_dir_a": summary_a,
            "summary_dir_b": summary_b,
        }
        print("=" * 80)
        print("JSON summary")
        print("-" * 80)
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())