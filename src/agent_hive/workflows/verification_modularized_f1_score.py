#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def canonicalize_labels(labels: Sequence[str]) -> List[str]:
    return sorted({normalize_text(x) for x in labels if normalize_text(x)})


def normalize_dep_bucket(dep_count: int) -> str:
    if dep_count <= 0:
        return "0"
    if dep_count == 1:
        return "1"
    return "2_plus"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_qid_from_filename(path: Path) -> str:
    name = path.name
    m = re.search(r"Q[_-]?(\d+)", name)
    if m:
        return f"Q{m.group(1)}"
    raise ValueError(f"Could not extract qid from filename: {path.name}")


# -----------------------------------------------------------------------------
# Plan parsing
# -----------------------------------------------------------------------------

def find_balanced_json_block(text: str, start_char: str, end_char: str) -> Optional[str]:
    start = text.find(start_char)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def extract_first_plan_json_object(plan_text: str) -> Any:
    marker = "Conditional Plan JSON:"
    if marker in plan_text:
        suffix = plan_text.split(marker, 1)[1].strip()
    else:
        suffix = plan_text.strip()

    arr_block = find_balanced_json_block(suffix, "[", "]")
    if arr_block:
        try:
            return json.loads(arr_block)
        except Exception:
            try:
                return ast.literal_eval(arr_block)
            except Exception:
                pass

    obj_block = find_balanced_json_block(suffix, "{", "}")
    if obj_block:
        try:
            return json.loads(obj_block)
        except Exception:
            try:
                return ast.literal_eval(obj_block)
            except Exception:
                pass

    raise ValueError("Failed to parse plan JSON from plan txt.")


def extract_expected_exception_labels(raw_node: Dict[str, Any]) -> List[str]:
    raw = raw_node.get("expected_exception", [])
    labels: List[str] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                s = normalize_text(item)
                if s:
                    labels.append(s)
            elif isinstance(item, dict):
                s = normalize_text(item.get("label"))
                if s:
                    labels.append(s)
    elif isinstance(raw, dict):
        s = normalize_text(raw.get("label"))
        if s:
            labels.append(s)

    return canonicalize_labels(labels)


def extract_plan_nodes(plan_doc: Any, qid: str, source_path: Path) -> List[Dict[str, Any]]:
    container: List[Dict[str, Any]] = []

    if isinstance(plan_doc, dict):
        if isinstance(plan_doc.get("nodes"), list):
            container = [x for x in plan_doc["nodes"] if isinstance(x, dict)]
        elif isinstance(plan_doc.get("tasks"), list):
            container = [x for x in plan_doc["tasks"] if isinstance(x, dict)]
    elif isinstance(plan_doc, list):
        container = [x for x in plan_doc if isinstance(x, dict)]

    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw_node in container:
        node_id = normalize_text(raw_node.get("id") or raw_node.get("node_id"))
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)

        task_text = normalize_text(raw_node.get("task"))
        node_contract_text = normalize_text(raw_node.get("node_contract"))
        agent_name = normalize_text(raw_node.get("agent") or raw_node.get("agent_name"))
        deps = raw_node.get("deps", [])
        if not isinstance(deps, list):
            deps = []
        deps = [normalize_text(x) for x in deps if normalize_text(x)]

        nodes.append({
            "qid": qid,
            "node_id": node_id,
            "task_text": task_text,
            "node_contract_text": node_contract_text,
            "agent_name": agent_name,
            "deps": deps,
            "dep_bucket": normalize_dep_bucket(len(deps)),
            "expected_exception_labels": extract_expected_exception_labels(raw_node),
            "pred_verify": bool(raw_node.get("verification", False)),
            "plan_path": str(source_path),
            "raw_node": raw_node,
        })

    return nodes


def scan_plan_files(plan_dir: Path) -> Dict[str, Path]:
    files = sorted(plan_dir.glob("*.txt"))
    mapping: Dict[str, Path] = {}

    for path in files:
        qid = extract_qid_from_filename(path)
        if qid in mapping:
            raise ValueError(f"Duplicate qid in plan_dir for {qid}: {mapping[qid].name}, {path.name}")
        mapping[qid] = path

    return mapping


def parse_plan_dir(plan_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    plan_map = scan_plan_files(plan_dir)
    raw_nodes: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []

    for qid, plan_path in sorted(plan_map.items()):
        try:
            plan_text = plan_path.read_text(encoding="utf-8")
            plan_doc = extract_first_plan_json_object(plan_text)
            batch = extract_plan_nodes(plan_doc, qid=qid, source_path=plan_path)
            raw_nodes.extend(batch)
        except Exception as exc:
            parse_errors.append({
                "qid": qid,
                "plan_path": str(plan_path),
                "error": repr(exc),
            })

    return raw_nodes, parse_errors


# -----------------------------------------------------------------------------
# Oracle loading
# -----------------------------------------------------------------------------

def build_oracle_lookup(oracle_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in oracle_rows:
        if not isinstance(row, dict):
            continue

        qid = normalize_text(row.get("qid"))
        node_id = normalize_text(row.get("node_id"))
        if not qid or not node_id:
            continue

        key = (qid, node_id)
        if key in lookup:
            raise ValueError(f"Duplicate oracle row for {(qid, node_id)}")

        lookup[key] = {
            "oracle_verify": bool(row.get("oracle_verify", False)),
            "task_label": normalize_text(row.get("task_label")),
            "contract_label": normalize_text(row.get("contract_label")),
            "outcome_label": normalize_text(row.get("outcome_label")),
            "gold_labels": canonicalize_labels(row.get("failure_reason_labels", [])),
            "exact_signature": normalize_text(row.get("exact_signature")),
            "assigned_risk_simple": row.get("assigned_risk_simple"),
            "assigned_risk_weighted": row.get("assigned_risk_weighted"),
            "label_risk": row.get("label_risk", {}),
            "label_support": row.get("label_support", {}),
            "gold_verify_from_oracle_rows": row.get("gold_verify"),
        }

    return lookup


# -----------------------------------------------------------------------------
# Joining three sources
# -----------------------------------------------------------------------------

def build_node_lookup(rows: List[Dict[str, Any]], source_name: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        key = (row["qid"], row["node_id"])
        if key in lookup:
            raise ValueError(f"Duplicate node in {source_name} for {key}")
        lookup[key] = row

    return lookup


def join_sources(
    annotated_rows: List[Dict[str, Any]],
    allocation_rows: List[Dict[str, Any]],
    oracle_lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    annotated_lookup = build_node_lookup(annotated_rows, "annotated_only")
    allocation_lookup = build_node_lookup(allocation_rows, "allocation_only")

    all_keys = sorted(set(annotated_lookup.keys()) | set(allocation_lookup.keys()) | set(oracle_lookup.keys()))

    joined: List[Dict[str, Any]] = []
    unmatched = {
        "annotated_missing": [],
        "allocation_missing": [],
        "oracle_missing": [],
    }

    for key in all_keys:
        ann = annotated_lookup.get(key)
        alloc = allocation_lookup.get(key)
        oracle = oracle_lookup.get(key)

        qid, node_id = key

        if ann is None:
            unmatched["annotated_missing"].append({"qid": qid, "node_id": node_id})
            continue
        if alloc is None:
            unmatched["allocation_missing"].append({"qid": qid, "node_id": node_id})
            continue
        if oracle is None:
            unmatched["oracle_missing"].append({"qid": qid, "node_id": node_id})
            continue

        module2_pred_labels = canonicalize_labels(ann.get("expected_exception_labels", []))
        allocation_labels = canonicalize_labels(alloc.get("expected_exception_labels", []))
        gold_labels = canonicalize_labels(oracle.get("gold_labels", []))

        module2_tp_labels = sorted(set(module2_pred_labels) & set(gold_labels))
        module2_fp_labels = sorted(set(module2_pred_labels) - set(gold_labels))
        module2_fn_labels = sorted(set(gold_labels) - set(module2_pred_labels))

        module2_hit = len(module2_tp_labels) > 0
        oracle_verify = bool(oracle["oracle_verify"])
        reachable_oracle_verify = bool(oracle_verify and module2_hit)

        joined.append({
            "qid": qid,
            "node_id": node_id,

            "task_text": alloc["task_text"] or ann["task_text"],
            "node_contract_text": alloc["node_contract_text"] or ann["node_contract_text"],
            "agent_name": alloc["agent_name"] or ann["agent_name"],
            "deps": alloc["deps"] if alloc["deps"] else ann["deps"],
            "dep_bucket": alloc["dep_bucket"] or ann["dep_bucket"],

            "task_label": oracle["task_label"],
            "contract_label": oracle["contract_label"],
            "outcome_label": oracle["outcome_label"],

            "module2_pred_labels": module2_pred_labels,
            "allocation_expected_labels": allocation_labels,
            "gold_labels": gold_labels,

            "module2_tp_labels": module2_tp_labels,
            "module2_fp_labels": module2_fp_labels,
            "module2_fn_labels": module2_fn_labels,
            "module2_hit": module2_hit,

            "module3_pred_verify": bool(alloc["pred_verify"]),
            "pipeline_pred_verify": bool(alloc["pred_verify"]),
            "oracle_verify": oracle_verify,
            "reachable_oracle_verify": reachable_oracle_verify,

            "annotated_plan_path": ann["plan_path"],
            "allocation_plan_path": alloc["plan_path"],

            "exact_signature": oracle["exact_signature"],
            "assigned_risk_simple": oracle["assigned_risk_simple"],
            "assigned_risk_weighted": oracle["assigned_risk_weighted"],
            "label_risk": oracle["label_risk"],
            "label_support": oracle["label_support"],

            "allocation_labels_match_annotated": allocation_labels == module2_pred_labels,
        })

    return joined, unmatched


# -----------------------------------------------------------------------------
# Module2 metrics
# -----------------------------------------------------------------------------

def compute_module2_label_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = 0

    for row in rows:
        tp += len(row["module2_tp_labels"])
        fp += len(row["module2_fp_labels"])
        fn += len(row["module2_fn_labels"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_module2_node_hit_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive_nodes = 0
    hit_nodes = 0
    missed_nodes = 0
    extra_only_nodes = 0

    for row in rows:
        gold = row["gold_labels"]
        pred = row["module2_pred_labels"]

        if gold:
            positive_nodes += 1
            if row["module2_hit"]:
                hit_nodes += 1
            else:
                missed_nodes += 1
        else:
            if pred:
                extra_only_nodes += 1

    hit_rate = hit_nodes / positive_nodes if positive_nodes else 0.0
    miss_rate = missed_nodes / positive_nodes if positive_nodes else 0.0

    return {
        "num_positive_nodes": positive_nodes,
        "num_hit_nodes": hit_nodes,
        "num_missed_nodes": missed_nodes,
        "num_extra_only_nodes": extra_only_nodes,
        "hit_rate": hit_rate,
        "miss_rate": miss_rate,
    }


def build_module2_by_label(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "support_pred": 0,
        "support_gold": 0,
        "miss_examples": [],
        "extra_examples": [],
    })

    for row in rows:
        pred = set(row["module2_pred_labels"])
        gold = set(row["gold_labels"])

        for z in pred:
            stats[z]["support_pred"] += 1
        for z in gold:
            stats[z]["support_gold"] += 1

        for z in pred & gold:
            stats[z]["tp"] += 1

        for z in pred - gold:
            stats[z]["fp"] += 1
            stats[z]["extra_examples"].append({
                "qid": row["qid"],
                "node_id": row["node_id"],
                "task_label": row["task_label"],
                "contract_label": row["contract_label"],
                "agent_name": row["agent_name"],
                "module2_pred_labels": row["module2_pred_labels"],
                "gold_labels": row["gold_labels"],
            })

        for z in gold - pred:
            stats[z]["fn"] += 1
            stats[z]["miss_examples"].append({
                "qid": row["qid"],
                "node_id": row["node_id"],
                "task_label": row["task_label"],
                "contract_label": row["contract_label"],
                "agent_name": row["agent_name"],
                "module2_pred_labels": row["module2_pred_labels"],
                "gold_labels": row["gold_labels"],
            })

    out: Dict[str, Dict[str, Any]] = {}
    for z, st in stats.items():
        tp = st["tp"]
        fp = st["fp"]
        fn = st["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        out[z] = {
            **st,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_miss_examples": len(st["miss_examples"]),
            "num_extra_examples": len(st["extra_examples"]),
        }

    return out


# -----------------------------------------------------------------------------
# Module3 metrics
# -----------------------------------------------------------------------------

def compute_confusion(rows: List[Dict[str, Any]], pred_field: str, gold_field: str) -> Dict[str, int]:
    tp = fp = fn = tn = 0

    for row in rows:
        pred = bool(row[pred_field])
        gold = bool(row[gold_field])

        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif (not pred) and gold:
            fn += 1
        else:
            tn += 1

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def confusion_to_metrics(conf: Dict[str, int]) -> Dict[str, Any]:
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    tn = conf["tn"]
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    over_verification_rate = fp / (tp + fp) if (tp + fp) else 0.0

    return {
        "num_nodes": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "miss_rate": miss_rate,
        "over_verification_rate": over_verification_rate,
    }


def build_module3_by_label(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Module3 is evaluated only on labels that are reachable from module2.
    For each predicted module2 label z:
      reachable positive for z: oracle_verify and z in gold and z in module2_pred
      pred positive for z: module3_pred_verify and z in module2_pred
    """
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "support_predicted_by_module2": 0,
        "reachable_positive_nodes": 0,
        "predicted_verify_nodes": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "miss_examples": [],
        "over_verification_examples": [],
    })

    for row in rows:
        pred_labels = set(row["module2_pred_labels"])
        gold_labels = set(row["gold_labels"])
        oracle_verify = bool(row["oracle_verify"])
        pred_verify = bool(row["module3_pred_verify"])

        for z in pred_labels:
            st = stats[z]
            st["support_predicted_by_module2"] += 1

            reachable_positive = bool(oracle_verify and z in gold_labels)
            predicted_positive = bool(pred_verify)

            if reachable_positive:
                st["reachable_positive_nodes"] += 1
            if predicted_positive:
                st["predicted_verify_nodes"] += 1

            if predicted_positive and reachable_positive:
                st["tp"] += 1
            elif predicted_positive and not reachable_positive:
                st["fp"] += 1
                st["over_verification_examples"].append({
                    "qid": row["qid"],
                    "node_id": row["node_id"],
                    "label": z,
                    "task_label": row["task_label"],
                    "contract_label": row["contract_label"],
                    "agent_name": row["agent_name"],
                    "module2_pred_labels": row["module2_pred_labels"],
                    "gold_labels": row["gold_labels"],
                    "oracle_verify": row["oracle_verify"],
                    "module3_pred_verify": row["module3_pred_verify"],
                })
            elif (not predicted_positive) and reachable_positive:
                st["fn"] += 1
                st["miss_examples"].append({
                    "qid": row["qid"],
                    "node_id": row["node_id"],
                    "label": z,
                    "task_label": row["task_label"],
                    "contract_label": row["contract_label"],
                    "agent_name": row["agent_name"],
                    "module2_pred_labels": row["module2_pred_labels"],
                    "gold_labels": row["gold_labels"],
                    "oracle_verify": row["oracle_verify"],
                    "module3_pred_verify": row["module3_pred_verify"],
                })
            else:
                st["tn"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for z, st in stats.items():
        tp = st["tp"]
        fp = st["fp"]
        fn = st["fn"]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
        over_verification_rate = fp / (tp + fp) if (tp + fp) else 0.0

        out[z] = {
            **st,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "miss_rate": miss_rate,
            "over_verification_rate": over_verification_rate,
            "num_miss_examples": len(st["miss_examples"]),
            "num_over_verification_examples": len(st["over_verification_examples"]),
        }

    return out


# -----------------------------------------------------------------------------
# Pipeline attribution summary
# -----------------------------------------------------------------------------

def build_attribution_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    oracle_positive_nodes = sum(1 for r in rows if r["oracle_verify"])
    reachable_positive_nodes = sum(1 for r in rows if r["reachable_oracle_verify"])
    blocked_by_module2_nodes = sum(1 for r in rows if r["oracle_verify"] and not r["reachable_oracle_verify"])
    module3_tp_nodes = sum(1 for r in rows if r["module3_pred_verify"] and r["reachable_oracle_verify"])
    module3_fn_nodes = sum(1 for r in rows if (not r["module3_pred_verify"]) and r["reachable_oracle_verify"])

    return {
        "oracle_positive_nodes": oracle_positive_nodes,
        "reachable_positive_nodes_for_module3": reachable_positive_nodes,
        "blocked_by_module2_nodes": blocked_by_module2_nodes,
        "blocked_fraction_among_oracle_positive": (
            blocked_by_module2_nodes / oracle_positive_nodes if oracle_positive_nodes else 0.0
        ),
        "module3_tp_nodes": module3_tp_nodes,
        "module3_fn_nodes": module3_fn_nodes,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate module2 and module3 separately. "
            "Module2 prediction comes from annotated_only plan expected_exception labels. "
            "Module3 prediction comes from allocation_only plan verification boolean. "
            "Gold oracle comes from oracle_rows_with_threshold.json."
        )
    )
    parser.add_argument("--annotated_plan_dir", required=True)
    parser.add_argument("--allocation_plan_dir", required=True)
    parser.add_argument("--oracle_rows_json", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotated_plan_dir = Path(args.annotated_plan_dir)
    allocation_plan_dir = Path(args.allocation_plan_dir)
    oracle_rows_json = Path(args.oracle_rows_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = read_json(oracle_rows_json)
    if not isinstance(oracle_rows, list):
        raise ValueError("oracle_rows_json must be a JSON list.")
    oracle_lookup = build_oracle_lookup(oracle_rows)

    annotated_rows, annotated_parse_errors = parse_plan_dir(annotated_plan_dir)
    allocation_rows, allocation_parse_errors = parse_plan_dir(allocation_plan_dir)

    joined_rows, unmatched = join_sources(
        annotated_rows=annotated_rows,
        allocation_rows=allocation_rows,
        oracle_lookup=oracle_lookup,
    )

    # module2
    module2_label_metrics = compute_module2_label_metrics(joined_rows)
    module2_node_hit_metrics = compute_module2_node_hit_metrics(joined_rows)
    module2_by_label = build_module2_by_label(joined_rows)

    # module3 conditional on module2 reachability
    module3_conf = compute_confusion(
        joined_rows,
        pred_field="module3_pred_verify",
        gold_field="reachable_oracle_verify",
    )
    module3_metrics = confusion_to_metrics(module3_conf)
    module3_by_label = build_module3_by_label(joined_rows)

    # total pipeline versus raw oracle
    pipeline_conf = compute_confusion(
        joined_rows,
        pred_field="pipeline_pred_verify",
        gold_field="oracle_verify",
    )
    pipeline_metrics = confusion_to_metrics(pipeline_conf)

    attribution_summary = build_attribution_summary(joined_rows)

    summary = {
        "status": "ok",
        "annotated_plan_dir": str(annotated_plan_dir),
        "allocation_plan_dir": str(allocation_plan_dir),
        "oracle_source": str(oracle_rows_json),

        "num_annotated_nodes": len(annotated_rows),
        "num_allocation_nodes": len(allocation_rows),
        "num_joined_nodes": len(joined_rows),

        "num_annotated_parse_errors": len(annotated_parse_errors),
        "num_allocation_parse_errors": len(allocation_parse_errors),

        "num_unmatched_annotated_missing": len(unmatched["annotated_missing"]),
        "num_unmatched_allocation_missing": len(unmatched["allocation_missing"]),
        "num_unmatched_oracle_missing": len(unmatched["oracle_missing"]),

        "module2_label_metrics": module2_label_metrics,
        "module2_node_hit_metrics": module2_node_hit_metrics,

        "module3_conditional_metrics": module3_metrics,
        "pipeline_total_metrics": pipeline_metrics,
        "attribution_summary": attribution_summary,

        "top_module2_missed_labels": sorted(
            [
                {
                    "label": z,
                    "miss_count": st["fn"],
                    "support_gold": st["support_gold"],
                    "recall": st["recall"],
                }
                for z, st in module2_by_label.items()
            ],
            key=lambda x: (x["miss_count"], x["support_gold"], -x["recall"]),
            reverse=True,
        )[:20],

        "top_module2_extra_labels": sorted(
            [
                {
                    "label": z,
                    "extra_count": st["fp"],
                    "support_pred": st["support_pred"],
                    "precision": st["precision"],
                }
                for z, st in module2_by_label.items()
            ],
            key=lambda x: (x["extra_count"], x["support_pred"], -x["precision"]),
            reverse=True,
        )[:20],

        "top_module3_missed_labels": sorted(
            [
                {
                    "label": z,
                    "miss_count": st["fn"],
                    "reachable_positive_nodes": st["reachable_positive_nodes"],
                    "miss_rate": st["miss_rate"],
                }
                for z, st in module3_by_label.items()
            ],
            key=lambda x: (x["miss_count"], x["reachable_positive_nodes"], x["miss_rate"]),
            reverse=True,
        )[:20],

        "top_module3_over_verified_labels": sorted(
            [
                {
                    "label": z,
                    "over_verification_count": st["fp"],
                    "support_predicted_by_module2": st["support_predicted_by_module2"],
                    "over_verification_rate": st["over_verification_rate"],
                }
                for z, st in module3_by_label.items()
            ],
            key=lambda x: (x["over_verification_count"], x["support_predicted_by_module2"], x["over_verification_rate"]),
            reverse=True,
        )[:20],

        "notes": {
            "module2_prediction": "expected_exception labels from annotated_only plan",
            "module2_gold": "failure_reason_labels from oracle_rows_with_threshold.json",
            "module2_metric_unit": "node-label pairs",
            "module3_prediction": "verification boolean from allocation_only plan",
            "module3_gold": "reachable_oracle_verify = oracle_verify AND module2_hit",
            "pipeline_prediction": "verification boolean from allocation_only plan",
            "pipeline_gold": "raw oracle_verify from oracle_rows_with_threshold.json",
        },
    }

    dump_json(joined_rows, output_dir / "verification_f1_rows.json")
    dump_json(summary, output_dir / "verification_f1_summary.json")
    dump_json(module2_by_label, output_dir / "verification_f1_module2_by_label.json")
    dump_json(module3_by_label, output_dir / "verification_f1_module3_by_label.json")
    dump_json(annotated_parse_errors, output_dir / "verification_f1_annotated_parse_errors.json")
    dump_json(allocation_parse_errors, output_dir / "verification_f1_allocation_parse_errors.json")
    dump_json(unmatched, output_dir / "verification_f1_unmatched.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()