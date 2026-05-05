#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


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
            "pred_verify": bool(raw_node.get("verification", False)),
            "plan_path": str(source_path),
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
        }

    return lookup


# -----------------------------------------------------------------------------
# Join plan predictions with oracle
# -----------------------------------------------------------------------------

def build_node_lookup(rows: List[Dict[str, Any]], source_name: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        key = (row["qid"], row["node_id"])
        if key in lookup:
            raise ValueError(f"Duplicate node in {source_name} for {key}")
        lookup[key] = row

    return lookup


def join_plan_with_oracle(
    plan_rows: List[Dict[str, Any]],
    oracle_lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    plan_lookup = build_node_lookup(plan_rows, "plan_rows")
    all_keys = sorted(set(plan_lookup.keys()) | set(oracle_lookup.keys()))

    joined: List[Dict[str, Any]] = []
    unmatched = {
        "plan_missing": [],
        "oracle_missing": [],
    }

    for key in all_keys:
        plan_row = plan_lookup.get(key)
        oracle = oracle_lookup.get(key)

        qid, node_id = key

        if plan_row is None:
            unmatched["plan_missing"].append({"qid": qid, "node_id": node_id})
            continue
        if oracle is None:
            unmatched["oracle_missing"].append({"qid": qid, "node_id": node_id})
            continue

        joined.append({
            "qid": qid,
            "node_id": node_id,
            "task_text": plan_row["task_text"],
            "node_contract_text": plan_row["node_contract_text"],
            "agent_name": plan_row["agent_name"],
            "deps": plan_row["deps"],
            "dep_bucket": plan_row["dep_bucket"],
            "pred_verify": bool(plan_row["pred_verify"]),
            "oracle_verify": bool(oracle["oracle_verify"]),
            "plan_path": plan_row["plan_path"],
        })

    return joined, unmatched


# -----------------------------------------------------------------------------
# Verification metrics
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
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate verification F1 using only the `verification` boolean in plan text."
    )
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--oracle_rows_json", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plan_dir = Path(args.plan_dir)
    oracle_rows_json = Path(args.oracle_rows_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = read_json(oracle_rows_json)
    if not isinstance(oracle_rows, list):
        raise ValueError("oracle_rows_json must be a JSON list.")

    oracle_lookup = build_oracle_lookup(oracle_rows)

    plan_rows, parse_errors = parse_plan_dir(plan_dir)

    joined_rows, unmatched = join_plan_with_oracle(
        plan_rows=plan_rows,
        oracle_lookup=oracle_lookup,
    )

    conf = compute_confusion(
        joined_rows,
        pred_field="pred_verify",
        gold_field="oracle_verify",
    )
    metrics = confusion_to_metrics(conf)

    summary = {
        "status": "ok",
        "plan_dir": str(plan_dir),
        "oracle_source": str(oracle_rows_json),
        "num_plan_nodes": len(plan_rows),
        "num_joined_nodes": len(joined_rows),
        "num_parse_errors": len(parse_errors),
        "num_unmatched_plan_missing": len(unmatched["plan_missing"]),
        "num_unmatched_oracle_missing": len(unmatched["oracle_missing"]),
        "verification_metrics": metrics,
        "notes": {
            "prediction": "verification boolean from plan text only",
            "gold": "oracle_verify from oracle_rows_json",
            "labels_ignored": True,
            "missing_verification_defaults_to_false": True,
        },
    }

    dump_json(joined_rows, output_dir / "verification_only_rows.json")
    dump_json(summary, output_dir / "verification_only_summary.json")
    dump_json(parse_errors, output_dir / "verification_only_parse_errors.json")
    dump_json(unmatched, output_dir / "verification_only_unmatched.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()