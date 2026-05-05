#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def extract_expected_exception_labels(node: Dict[str, Any]) -> List[str]:
    raw = node.get("expected_exception", [])
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
        "gold_contained_in_pred": sg.issubset(sp),
        "pred_subset_of_gold": sp.issubset(sg),
        "num_pred_only": len(sp - sg),
        "num_gold_missed": len(sg - sp),
    }


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def nested_counters_to_sorted_dict(d: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for key in sorted(d.keys()):
        out[key] = counter_to_sorted_dict(d[key])
    return out


def outcome_task_exception_to_sorted_dict(
    d: Dict[str, Dict[str, Counter]]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for outcome in sorted(d.keys()):
        out[outcome] = {}
        for task_label in sorted(d[outcome].keys()):
            out[outcome][task_label] = counter_to_sorted_dict(d[outcome][task_label])
    return out


def evaluate_module2(
    node_signature_table_path: Path,
    annotated_plan_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    node_signature_table = load_json(node_signature_table_path)
    if not isinstance(node_signature_table, list):
        raise ValueError("node_signature_table must be a JSON array.")

    gold_index = build_node_signature_index(node_signature_table)
    annotated_map = scan_plan_folder(annotated_plan_dir)

    rows: List[Dict[str, Any]] = []

    total_nodes = 0
    matched_nodes = 0
    unmatched_nodes = 0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    exact_match_count = 0
    containment_count = 0
    pred_subset_count = 0

    per_node_precision: List[float] = []
    per_node_recall: List[float] = []
    per_node_f1: List[float] = []
    pred_only_counts: List[float] = []
    missed_counts: List[float] = []

    # task-level 基本統計
    nodes_total_by_task_label: Counter = Counter()
    nodes_with_any_missed_by_task_label: Counter = Counter()
    nodes_with_any_extra_by_task_label: Counter = Counter()

    # outcome 統計
    outcome_counts_global: Counter = Counter()
    outcome_counts_by_task_label: Dict[str, Counter] = defaultdict(Counter)

    # P/N/E それぞれで、どの task label にどの exception が何回出たか
    gold_failure_label_counts_by_outcome_and_task_label: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    # 予測 / gold の総付与回数
    pred_label_counts_global: Counter = Counter()
    gold_label_counts_global: Counter = Counter()

    pred_label_counts_by_task_label: Dict[str, Counter] = defaultdict(Counter)
    gold_label_counts_by_task_label: Dict[str, Counter] = defaultdict(Counter)

    # 取りこぼし / 余剰
    missed_label_counts_global: Counter = Counter()
    extra_label_counts_global: Counter = Counter()

    missed_label_counts_by_task_label: Dict[str, Counter] = defaultdict(Counter)
    extra_label_counts_by_task_label: Dict[str, Counter] = defaultdict(Counter)

    for qid, plan_path in sorted(annotated_map.items()):
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

            pred_labels = extract_expected_exception_labels(node)

            if gold_row is None:
                unmatched_nodes += 1
                rows.append({
                    "qid": qid,
                    "node_id": node_id,
                    "status": "missing_gold_row",
                    "task": node.get("task"),
                    "node_contract": node.get("node_contract"),
                    "pred_expected_exception_labels": pred_labels,
                    "gold_failure_reason_labels": None,
                    "gold_outcome_label": None,
                    "annotated_plan_file": str(plan_path),
                })
                continue

            matched_nodes += 1
            gold_labels = sorted(set(gold_row.get("failure_reason_labels", [])))
            task_label = str(gold_row.get("task_label", "UNKNOWN"))
            outcome_label = str(gold_row.get("outcome_label", "UNKNOWN"))

            nodes_total_by_task_label[task_label] += 1
            outcome_counts_global[outcome_label] += 1
            outcome_counts_by_task_label[task_label][outcome_label] += 1

            # P/N/E について task x exception 集計
            if outcome_label in {"P", "N", "E"}:
                for lab in gold_labels:
                    gold_failure_label_counts_by_outcome_and_task_label[outcome_label][task_label][lab] += 1

            # predicted / gold 総付与回数
            for lab in pred_labels:
                pred_label_counts_global[lab] += 1
                pred_label_counts_by_task_label[task_label][lab] += 1

            for lab in gold_labels:
                gold_label_counts_global[lab] += 1
                gold_label_counts_by_task_label[task_label][lab] += 1

            m = set_metrics(pred_labels, gold_labels)

            micro_tp += m["tp"]
            micro_fp += m["fp"]
            micro_fn += m["fn"]

            if m["exact_match"]:
                exact_match_count += 1
            if m["gold_contained_in_pred"]:
                containment_count += 1
            if m["pred_subset_of_gold"]:
                pred_subset_count += 1

            per_node_precision.append(m["precision"])
            per_node_recall.append(m["recall"])
            per_node_f1.append(m["f1"])
            pred_only_counts.append(float(m["num_pred_only"]))
            missed_counts.append(float(m["num_gold_missed"]))

            pred_set = set(pred_labels)
            gold_set = set(gold_labels)
            missed_labels = sorted(gold_set - pred_set)
            extra_labels = sorted(pred_set - gold_set)

            if missed_labels:
                nodes_with_any_missed_by_task_label[task_label] += 1
            if extra_labels:
                nodes_with_any_extra_by_task_label[task_label] += 1

            for lab in missed_labels:
                missed_label_counts_global[lab] += 1
                missed_label_counts_by_task_label[task_label][lab] += 1

            for lab in extra_labels:
                extra_label_counts_global[lab] += 1
                extra_label_counts_by_task_label[task_label][lab] += 1

            rows.append({
                "qid": qid,
                "node_id": node_id,
                "task": node.get("task"),
                "node_contract": node.get("node_contract"),
                "task_label": task_label,
                "gold_outcome_label": outcome_label,
                "pred_expected_exception_labels": pred_labels,
                "gold_failure_reason_labels": gold_labels,
                "missed_labels": missed_labels,
                "extra_labels": extra_labels,
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "exact_match": m["exact_match"],
                "gold_contained_in_pred": m["gold_contained_in_pred"],
                "pred_subset_of_gold": m["pred_subset_of_gold"],
                "num_pred_only": m["num_pred_only"],
                "num_gold_missed": m["num_gold_missed"],
                "annotated_plan_file": str(plan_path),
            })

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    metrics = {
        "num_total_annotated_nodes": total_nodes,
        "num_matched_nodes": matched_nodes,
        "num_unmatched_nodes": unmatched_nodes,

        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,

        "macro_precision": mean(per_node_precision),
        "macro_recall": mean(per_node_recall),
        "macro_f1": mean(per_node_f1),

        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / matched_nodes if matched_nodes else 0.0,

        "containment_count": containment_count,
        "containment_rate": containment_count / matched_nodes if matched_nodes else 0.0,

        "pred_subset_count": pred_subset_count,
        "pred_subset_rate": pred_subset_count / matched_nodes if matched_nodes else 0.0,

        "avg_extra_labels": mean(pred_only_counts),
        "avg_missed_labels": mean(missed_counts),

        # task 総数
        "nodes_total_by_task_label": counter_to_sorted_dict(nodes_total_by_task_label),

        # outcome 総数
        "outcome_counts_global": counter_to_sorted_dict(outcome_counts_global),
        "outcome_counts_by_task_label": nested_counters_to_sorted_dict(outcome_counts_by_task_label),

        # P/N/E ごとの task x exception 集計
        "gold_failure_label_counts_by_outcome_and_task_label": outcome_task_exception_to_sorted_dict(
            gold_failure_label_counts_by_outcome_and_task_label
        ),

        # predicted / gold 総付与回数
        "pred_label_counts_global": counter_to_sorted_dict(pred_label_counts_global),
        "gold_label_counts_global": counter_to_sorted_dict(gold_label_counts_global),

        "pred_label_counts_by_task_label": nested_counters_to_sorted_dict(pred_label_counts_by_task_label),
        "gold_label_counts_by_task_label": nested_counters_to_sorted_dict(gold_label_counts_by_task_label),

        # 取りこぼし / 余剰
        "missed_label_counts_global": counter_to_sorted_dict(missed_label_counts_global),
        "extra_label_counts_global": counter_to_sorted_dict(extra_label_counts_global),

        "missed_label_counts_by_task_label": nested_counters_to_sorted_dict(missed_label_counts_by_task_label),
        "extra_label_counts_by_task_label": nested_counters_to_sorted_dict(extra_label_counts_by_task_label),

        "nodes_with_any_missed_by_task_label": counter_to_sorted_dict(nodes_with_any_missed_by_task_label),
        "nodes_with_any_extra_by_task_label": counter_to_sorted_dict(nodes_with_any_extra_by_task_label),
    }

    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Module 2 using node_signature_table and annotated_only plans.")
    parser.add_argument("--node_signature_table", required=True)
    parser.add_argument("--annotated_plan_dir", required=True)
    parser.add_argument("--output_examples", required=True)
    parser.add_argument("--output_metrics", required=True)
    args = parser.parse_args()

    rows, metrics = evaluate_module2(
        node_signature_table_path=Path(args.node_signature_table),
        annotated_plan_dir=Path(args.annotated_plan_dir),
    )

    dump_json(rows, Path(args.output_examples))
    dump_json(metrics, Path(args.output_metrics))

    print(f"Saved examples: {args.output_examples}")
    print(f"Saved metrics: {args.output_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()