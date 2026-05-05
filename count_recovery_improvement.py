#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


STATUS_MAP = {
    "accomplished": "A",
    "partially accomplished": "P",
    "not accomplished": "N",
    "error": "E",
    "exception": "E",
}

RECOVERY_NODE_RE = re.compile(r"^R_(S\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze status transitions from source nodes S_i to their "
            "corresponding recovery nodes R_Si."
        )
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--trajectory_path",
        type=str,
        default="",
        help="Path to one trajectory JSON file.",
    )
    target.add_argument(
        "--trajectory_dir",
        type=str,
        default="",
        help="Path to a directory containing Q_*_trajectory.json files.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional path to save the result JSON. If omitted, print to stdout.",
    )
    return parser.parse_args()


def normalize_status(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return STATUS_MAP.get(str(raw).strip().lower())


def extract_status_from_reviews(reviews: Any) -> Optional[str]:
    if not isinstance(reviews, list):
        return None

    for item in reviews:
        if not isinstance(item, str):
            continue
        m = re.search(r"Task Status:\s*(.+)", item)
        if m:
            return normalize_status(m.group(1))
    return None


def extract_status(node: Dict[str, Any]) -> Optional[str]:
    """
    Expected schema:
    - response = [<free text>, {"status": "...", ...}]
    Fallbacks:
    - response may be a dict with "status"
    - reviews may contain 'Task Status: ...'
    """
    response = node.get("response")

    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and "status" in item:
                status = normalize_status(item.get("status"))
                if status is not None:
                    return status

    if isinstance(response, dict) and "status" in response:
        status = normalize_status(response.get("status"))
        if status is not None:
            return status

    status = extract_status_from_reviews(node.get("reviews"))
    if status is not None:
        return status

    return None


def extract_qid(path: Path) -> Optional[int]:
    m = re.search(r"Q_(\d+)_trajectory\.json$", path.name)
    if m:
        return int(m.group(1))
    return None


def transition_key(src: Optional[str], dst: Optional[str]) -> str:
    s = src if src is not None else "UNK"
    d = dst if dst is not None else "UNK"
    return f"{s}->{d}"


def analyze_one_trajectory(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    traj = data.get("trajectory", [])
    if not isinstance(traj, list):
        raise ValueError(f"'trajectory' is not a list: {path}")

    node_status: Dict[str, Optional[str]] = {}
    for node in traj:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        node_status[node_id] = extract_status(node)

    pairs: List[Dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()

    for node_id, recovery_status in node_status.items():
        m = RECOVERY_NODE_RE.match(node_id)
        if not m:
            continue

        source_id = m.group(1)
        source_status = node_status.get(source_id)
        tkey = transition_key(source_status, recovery_status)
        transition_counts[tkey] += 1

        pairs.append(
            {
                "source_node_id": source_id,
                "recovery_node_id": node_id,
                "source_status": source_status,
                "recovery_status": recovery_status,
                "transition": tkey,
            }
        )

    summary = {
        "trajectory_path": str(path),
        "qid": extract_qid(path),
        "num_recovery_pairs": len(pairs),
        "transition_counts": dict(transition_counts),

        # 以前の主要指標
        "num_source_PNE": sum(
            1 for p in pairs if p["source_status"] in {"P", "N", "E"}
        ),
        "num_improved_PNE_to_A": sum(
            1 for p in pairs
            if p["source_status"] in {"P", "N", "E"} and p["recovery_status"] == "A"
        ),

        # 追加したい悪化遷移
        "num_A_to_P": transition_counts["A->P"],
        "num_A_to_N": transition_counts["A->N"],
        "num_A_to_E": transition_counts["A->E"],
        "num_A_to_PNE": (
            transition_counts["A->P"]
            + transition_counts["A->N"]
            + transition_counts["A->E"]
        ),
        "num_P_to_N": transition_counts["P->N"],
        "num_P_to_E": transition_counts["P->E"],
        "num_N_to_E": transition_counts["N->E"],
        "num_P_or_N_to_E": transition_counts["P->E"] + transition_counts["N->E"],

        "pairs": pairs,
    }
    return summary


def aggregate_file_results(file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_transition_counts: Counter[str] = Counter()

    for fr in file_results:
        total_transition_counts.update(fr.get("transition_counts", {}))

    total = {
        "num_files": len(file_results),
        "total_recovery_pairs": sum(fr["num_recovery_pairs"] for fr in file_results),
        "total_source_PNE": sum(fr["num_source_PNE"] for fr in file_results),
        "total_improved_PNE_to_A": sum(fr["num_improved_PNE_to_A"] for fr in file_results),

        "total_A_to_P": sum(fr["num_A_to_P"] for fr in file_results),
        "total_A_to_N": sum(fr["num_A_to_N"] for fr in file_results),
        "total_A_to_E": sum(fr["num_A_to_E"] for fr in file_results),
        "total_A_to_PNE": sum(fr["num_A_to_PNE"] for fr in file_results),

        "total_P_to_N": sum(fr["num_P_to_N"] for fr in file_results),
        "total_P_to_E": sum(fr["num_P_to_E"] for fr in file_results),
        "total_N_to_E": sum(fr["num_N_to_E"] for fr in file_results),
        "total_P_or_N_to_E": sum(fr["num_P_or_N_to_E"] for fr in file_results),

        "transition_counts": dict(total_transition_counts),
        "files": file_results,
    }

    return total


def main() -> None:
    args = parse_args()

    if args.trajectory_path:
        paths = [Path(args.trajectory_path)]
    else:
        root = Path(args.trajectory_dir)
        paths = sorted(
            root.glob("Q_*_trajectory.json"),
            key=lambda p: (extract_qid(p) is None, extract_qid(p), p.name),
        )

    file_results = [analyze_one_trajectory(p) for p in paths]
    result = aggregate_file_results(file_results)

    text = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()