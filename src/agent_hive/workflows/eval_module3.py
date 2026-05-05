#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
        return obj
    if isinstance(obj, list):
        return {"answer_contract": None, "nodes": [x for x in obj if isinstance(x, dict)]}
    if isinstance(obj, dict):
        return {"answer_contract": None, "nodes": [obj]}
    raise ValueError(f"Unsupported parsed object type: {type(obj).__name__}")


def parse_plan_txt(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
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


def extract_branch_labels(node: Dict[str, Any]) -> List[str]:
    raw = node.get("branches", [])
    labels: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                lab = item.get("label")
                if isinstance(lab, str) and lab.strip():
                    labels.append(lab.strip())
            elif isinstance(item, str) and item.strip():
                labels.append(item.strip())
    return sorted(set(labels))


def scan_plan_folder(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*_plan.txt"))
    return {extract_qid_from_filename(p): p for p in files}


def build_node_signature_index(node_signature_table: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in node_signature_table:
        qid = str(row["qid"])
        node_id = str(row["node_id"])
        out[(qid, node_id)] = row
    return out


def set_metrics(pred: List[str], gold: List[str]) -> Dict[str, Any]:
    sp = set(pred)
    sg = set(gold)
    tp = len(sp & sg)
    fp = len(sp - sg)
    fn = len(sg - sp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": sp == sg,
    }


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_module3(
    node_signature_table_path: Path,
    allocation_plan_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    node_signature_table = load_json(node_signature_table_path)
    if not isinstance(node_signature_table, list):
        raise ValueError("node_signature_table must be a JSON array.")

    gold_index = build_node_signature_index(node_signature_table)
    allocation_map = scan_plan_folder(allocation_plan_dir)

    rows: List[Dict[str, Any]] = []

    total_nodes = 0
    matched_nodes = 0
    unmatched_nodes = 0

    # node-level gate metrics
    gate_tp = gate_fp = gate_fn = gate_tn = 0

    # label-level metrics
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    exact_label_match_count = 0
    per_node_precision: List[float] = []
    per_node_recall: List[float] = []
    per_node_f1: List[float] = []

    for qid, plan_path in sorted(allocation_map.items()):
        plan_doc = parse_plan_txt(plan_path)
        for node in plan_doc.get("nodes", []):
            if not isinstance(node, dict):
                continue

            total_nodes += 1
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue

            key = (qid, node_id)
            gold_row = gold_index.get(key)

            pred_verification = bool(node.get("verification", False))
            pred_branch_labels = extract_branch_labels(node)

            if gold_row is None:
                unmatched_nodes += 1
                rows.append({
                    "qid": qid,
                    "node_id": node_id,
                    "status": "missing_gold_row",
                    "task": node.get("task"),
                    "node_contract": node.get("node_contract"),
                    "pred_verification": pred_verification,
                    "pred_branch_labels": pred_branch_labels,
                    "gold_failure_reason_labels": None,
                    "gold_should_verify": None,
                    "allocation_plan_file": str(plan_path),
                })
                continue

            matched_nodes += 1
            gold_labels = sorted(set(gold_row.get("failure_reason_labels", [])))
            gold_should_verify = len(gold_labels) > 0

            # node-level gate
            if pred_verification and gold_should_verify:
                gate_tp += 1
            elif pred_verification and not gold_should_verify:
                gate_fp += 1
            elif (not pred_verification) and gold_should_verify:
                gate_fn += 1
            else:
                gate_tn += 1

            # label-level
            m = set_metrics(pred_branch_labels, gold_labels)

            micro_tp += m["tp"]
            micro_fp += m["fp"]
            micro_fn += m["fn"]

            if m["exact_match"]:
                exact_label_match_count += 1

            per_node_precision.append(m["precision"])
            per_node_recall.append(m["recall"])
            per_node_f1.append(m["f1"])

            rows.append({
                "qid": qid,
                "node_id": node_id,
                "task": node.get("task"),
                "node_contract": node.get("node_contract"),
                "pred_verification": pred_verification,
                "pred_branch_labels": pred_branch_labels,
                "gold_failure_reason_labels": gold_labels,
                "gold_should_verify": gold_should_verify,
                "label_precision": m["precision"],
                "label_recall": m["recall"],
                "label_f1": m["f1"],
                "exact_label_match": m["exact_match"],
                "allocation_plan_file": str(plan_path),
            })

    gate_precision = gate_tp / (gate_tp + gate_fp) if (gate_tp + gate_fp) else 0.0
    gate_recall = gate_tp / (gate_tp + gate_fn) if (gate_tp + gate_fn) else 0.0
    gate_f1 = (
        2 * gate_precision * gate_recall / (gate_precision + gate_recall)
        if (gate_precision + gate_recall)
        else 0.0
    )
    gate_accuracy = (gate_tp + gate_tn) / matched_nodes if matched_nodes else 0.0

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    metrics = {
        "num_total_allocation_nodes": total_nodes,
        "num_matched_nodes": matched_nodes,
        "num_unmatched_nodes": unmatched_nodes,

        "node_gate_precision": gate_precision,
        "node_gate_recall": gate_recall,
        "node_gate_f1": gate_f1,
        "node_gate_accuracy": gate_accuracy,
        "node_gate_tp": gate_tp,
        "node_gate_fp": gate_fp,
        "node_gate_fn": gate_fn,
        "node_gate_tn": gate_tn,

        "label_micro_precision": micro_precision,
        "label_micro_recall": micro_recall,
        "label_micro_f1": micro_f1,

        "label_macro_precision": mean(per_node_precision),
        "label_macro_recall": mean(per_node_recall),
        "label_macro_f1": mean(per_node_f1),

        "exact_label_match_count": exact_label_match_count,
        "exact_label_match_rate": exact_label_match_count / matched_nodes if matched_nodes else 0.0,
    }

    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Module 3 using node_signature_table and allocation_only plans.")
    parser.add_argument("--node_signature_table", required=True)
    parser.add_argument("--allocation_plan_dir", required=True)
    parser.add_argument("--output_examples", required=True)
    parser.add_argument("--output_metrics", required=True)
    args = parser.parse_args()

    rows, metrics = evaluate_module3(
        node_signature_table_path=Path(args.node_signature_table),
        allocation_plan_dir=Path(args.allocation_plan_dir),
    )

    dump_json(rows, Path(args.output_examples))
    dump_json(metrics, Path(args.output_metrics))

    print(f"Saved examples: {args.output_examples}")
    print(f"Saved metrics: {args.output_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()