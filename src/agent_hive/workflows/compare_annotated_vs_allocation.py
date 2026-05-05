#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not find JSON object/array in text.")


def normalize_plan_doc(obj: Any) -> Dict[str, Any]:
    """
    Accept:
      1) {"answer_contract": ..., "nodes": [...]}
      2) [ {...node...}, {...node...} ]
      3) {...single node...}
    and normalize to {"answer_contract": ..., "nodes": [...]}
    """
    if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
        return obj

    if isinstance(obj, list):
        return {
            "answer_contract": None,
            "nodes": [x for x in obj if isinstance(x, dict)],
        }

    if isinstance(obj, dict):
        return {
            "answer_contract": None,
            "nodes": [obj],
        }

    raise ValueError(f"Unsupported parsed object type: {type(obj).__name__}")


def parse_plan_txt(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    obj = extract_first_json_value(text)
    return normalize_plan_doc(obj)


def extract_qid_from_filename(path: Path) -> str:
    m = re.search(r"Q_(\d+)_plan\.txt$", path.name)
    if m:
        return f"Q{m.group(1)}"
    m2 = re.search(r"Q[_-]?(\d+)", path.name)
    if m2:
        return f"Q{m2.group(1)}"
    return path.stem


def extract_expected_exception_labels(node: Dict[str, Any]) -> List[str]:
    raw = node.get("expected_exception", [])
    labels: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                label = item.get("label")
                if isinstance(label, str) and label.strip():
                    labels.append(label.strip())
            elif isinstance(item, str) and item.strip():
                labels.append(item.strip())
    return sorted(set(labels))


def extract_branch_labels(node: Dict[str, Any]) -> List[str]:
    raw = node.get("branches", [])
    labels: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                label = item.get("label")
                if isinstance(label, str) and label.strip():
                    labels.append(label.strip())
            elif isinstance(item, str) and item.strip():
                labels.append(item.strip())
    return sorted(set(labels))


def scan_plan_folder(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*_plan.txt"))
    return {extract_qid_from_filename(p): p for p in files}


def node_key(qid: str, node: Dict[str, Any]) -> Tuple[str, str]:
    return qid, str(node.get("id", "")).strip()


def safe_bool(x: Any) -> bool:
    return bool(x)


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa and sb:
        return 0.0
    if sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def precision_recall_f1(pred: List[str], gold: List[str]) -> Tuple[float, float, float]:
    sp = set(pred)
    sg = set(gold)
    tp = len(sp & sg)
    fp = len(sp - sg)
    fn = len(sg - sp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compare_folders(
    annotated_dir: Path,
    allocation_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotated_map = scan_plan_folder(annotated_dir)
    allocation_map = scan_plan_folder(allocation_dir)

    common_qids = sorted(set(annotated_map.keys()) & set(allocation_map.keys()))
    annotated_only_qids = sorted(set(annotated_map.keys()) - set(allocation_map.keys()))
    allocation_only_qids = sorted(set(allocation_map.keys()) - set(annotated_map.keys()))

    merged_rows: List[Dict[str, Any]] = []

    total_nodes = 0
    matched_nodes = 0

    verify_true_count = 0
    verify_false_count = 0

    exact_label_match_count = 0
    subset_label_match_count = 0
    empty_branch_count = 0
    copied_all_expected_count = 0

    expected_label_counts: List[float] = []
    allocated_label_counts: List[float] = []
    jaccards: List[float] = []

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for qid in common_qids:
        ann_doc = parse_plan_txt(annotated_map[qid])
        alloc_doc = parse_plan_txt(allocation_map[qid])

        ann_nodes = {
            node_key(qid, n): n
            for n in ann_doc.get("nodes", [])
            if isinstance(n, dict) and str(n.get("id", "")).strip()
        }
        alloc_nodes = {
            node_key(qid, n): n
            for n in alloc_doc.get("nodes", [])
            if isinstance(n, dict) and str(n.get("id", "")).strip()
        }

        common_node_keys = sorted(set(ann_nodes.keys()) & set(alloc_nodes.keys()))
        ann_only_node_keys = sorted(set(ann_nodes.keys()) - set(alloc_nodes.keys()))
        alloc_only_node_keys = sorted(set(alloc_nodes.keys()) - set(ann_nodes.keys()))

        for nk in common_node_keys:
            total_nodes += 1
            matched_nodes += 1

            ann_node = ann_nodes[nk]
            alloc_node = alloc_nodes[nk]

            expected_labels = extract_expected_exception_labels(ann_node)
            allocated_labels = extract_branch_labels(alloc_node)

            pred_verification = safe_bool(alloc_node.get("verification", False))

            if pred_verification:
                verify_true_count += 1
            else:
                verify_false_count += 1

            if allocated_labels == expected_labels:
                exact_label_match_count += 1

            if set(allocated_labels).issubset(set(expected_labels)):
                subset_label_match_count += 1

            if len(allocated_labels) == 0:
                empty_branch_count += 1

            if pred_verification and set(allocated_labels) == set(expected_labels) and len(expected_labels) > 0:
                copied_all_expected_count += 1

            expected_label_counts.append(float(len(expected_labels)))
            allocated_label_counts.append(float(len(allocated_labels)))
            jaccards.append(jaccard(allocated_labels, expected_labels))

            sp = set(allocated_labels)
            sg = set(expected_labels)
            micro_tp += len(sp & sg)
            micro_fp += len(sp - sg)
            micro_fn += len(sg - sp)

            p, r, f1 = precision_recall_f1(allocated_labels, expected_labels)

            merged_rows.append({
                "qid": qid,
                "node_id": ann_node.get("id"),
                "task": ann_node.get("task"),
                "node_contract": ann_node.get("node_contract"),
                "agent": ann_node.get("agent"),
                "deps": ann_node.get("deps", []),
                "expected_exception_labels": expected_labels,
                "allocation_verification": pred_verification,
                "allocation_branch_labels": allocated_labels,
                "num_expected_labels": len(expected_labels),
                "num_allocated_labels": len(allocated_labels),
                "label_count_difference": len(allocated_labels) - len(expected_labels),
                "exact_label_match": allocated_labels == expected_labels,
                "allocated_is_subset_of_expected": set(allocated_labels).issubset(set(expected_labels)),
                "copied_all_expected_labels": pred_verification and set(allocated_labels) == set(expected_labels) and len(expected_labels) > 0,
                "label_jaccard": jaccard(allocated_labels, expected_labels),
                "label_precision": p,
                "label_recall": r,
                "label_f1": f1,
                "annotated_plan_file": str(annotated_map[qid]),
                "allocation_plan_file": str(allocation_map[qid]),
            })

        for nk in ann_only_node_keys:
            total_nodes += 1
            ann_node = ann_nodes[nk]
            merged_rows.append({
                "qid": qid,
                "node_id": ann_node.get("id"),
                "task": ann_node.get("task"),
                "node_contract": ann_node.get("node_contract"),
                "agent": ann_node.get("agent"),
                "deps": ann_node.get("deps", []),
                "expected_exception_labels": extract_expected_exception_labels(ann_node),
                "allocation_verification": None,
                "allocation_branch_labels": None,
                "status": "missing_in_allocation_only",
                "annotated_plan_file": str(annotated_map[qid]),
                "allocation_plan_file": str(allocation_map[qid]),
            })

        for nk in alloc_only_node_keys:
            total_nodes += 1
            alloc_node = alloc_nodes[nk]
            merged_rows.append({
                "qid": qid,
                "node_id": alloc_node.get("id"),
                "task": alloc_node.get("task"),
                "node_contract": alloc_node.get("node_contract"),
                "agent": alloc_node.get("agent"),
                "deps": alloc_node.get("deps", []),
                "expected_exception_labels": None,
                "allocation_verification": safe_bool(alloc_node.get("verification", False)),
                "allocation_branch_labels": extract_branch_labels(alloc_node),
                "status": "missing_in_annotated_only",
                "annotated_plan_file": str(annotated_map[qid]),
                "allocation_plan_file": str(allocation_map[qid]),
            })

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    verification_rate = verify_true_count / matched_nodes if matched_nodes else 0.0

    metrics = {
        "num_common_qids": len(common_qids),
        "annotated_only_qids_missing_in_allocation_only": annotated_only_qids,
        "allocation_only_qids_missing_in_annotated_only": allocation_only_qids,

        "num_total_rows_emitted": len(merged_rows),
        "num_matched_nodes": matched_nodes,
        "num_total_nodes_seen": total_nodes,

        "verification_true_count": verify_true_count,
        "verification_false_count": verify_false_count,
        "verification_rate": verification_rate,

        "average_expected_label_count": mean(expected_label_counts),
        "average_allocated_label_count": mean(allocated_label_counts),
        "average_label_jaccard": mean(jaccards),

        "exact_label_match_count": exact_label_match_count,
        "exact_label_match_rate": exact_label_match_count / matched_nodes if matched_nodes else 0.0,

        "allocated_subset_of_expected_count": subset_label_match_count,
        "allocated_subset_of_expected_rate": subset_label_match_count / matched_nodes if matched_nodes else 0.0,

        "empty_branch_count": empty_branch_count,
        "empty_branch_rate": empty_branch_count / matched_nodes if matched_nodes else 0.0,

        "copied_all_expected_count": copied_all_expected_count,
        "copied_all_expected_rate": copied_all_expected_count / matched_nodes if matched_nodes else 0.0,

        "micro_label_tp": micro_tp,
        "micro_label_fp": micro_fp,
        "micro_label_fn": micro_fn,
        "micro_label_precision": micro_precision,
        "micro_label_recall": micro_recall,
        "micro_label_f1": micro_f1,

        # heuristic summary
        "always_verify_like_score": verification_rate,
        "copy_all_expected_like_score": copied_all_expected_count / matched_nodes if matched_nodes else 0.0,
    }

    return merged_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare annotated_only and allocation_only plans and measure whether allocation is selective or trivial."
    )
    parser.add_argument("--annotated_dir", required=True, help="Folder containing annotated_only *_plan.txt files.")
    parser.add_argument("--allocation_dir", required=True, help="Folder containing allocation_only *_plan.txt files.")
    parser.add_argument("--output_examples", required=True, help="Output JSON for merged node-level comparisons.")
    parser.add_argument("--output_metrics", required=True, help="Output JSON for aggregate metrics.")
    args = parser.parse_args()

    annotated_dir = Path(args.annotated_dir)
    allocation_dir = Path(args.allocation_dir)
    output_examples = Path(args.output_examples)
    output_metrics = Path(args.output_metrics)

    merged_rows, metrics = compare_folders(annotated_dir, allocation_dir)

    output_examples.parent.mkdir(parents=True, exist_ok=True)
    with output_examples.open("w", encoding="utf-8") as f:
        json.dump(merged_rows, f, ensure_ascii=False, indent=2)

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved examples: {output_examples}")
    print(f"Saved metrics: {output_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()