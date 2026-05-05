#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


OUTCOME_LABELS = ["A", "P", "N", "E"]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def canonicalize_labels(labels: Sequence[str]) -> List[str]:
    return sorted({normalize_text(x) for x in labels if normalize_text(x)})


def deps_to_key(deps: Any) -> str:
    if deps is None:
        return "[]"
    if not isinstance(deps, list):
        s = normalize_text(deps)
        return f"[{s}]" if s else "[]"
    cleaned = [normalize_text(x) for x in deps if normalize_text(x)]
    return "[" + ",".join(cleaned) + "]"


def get_dep_bucket(row: Dict[str, Any]) -> str:
    dep_bucket = normalize_text(row.get("dep_bucket"))
    if dep_bucket:
        return dep_bucket

    deps = row.get("deps", [])
    if not isinstance(deps, list):
        return "0"

    n = len(deps)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    return "2_plus"


def extract_label_list(row: Dict[str, Any], field_name: str) -> List[str]:
    raw = row.get(field_name)

    if raw is None:
        return []

    if isinstance(raw, str):
        s = normalize_text(raw)
        if not s:
            return []
        if "+" in s:
            return canonicalize_labels(s.split("+"))
        if "," in s:
            return canonicalize_labels(s.split(","))
        return [s]

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


def get_expected_labels(
    row: Dict[str, Any],
    expected_label_source: str,
    fallback_expected_label_source: Optional[str],
) -> List[str]:
    labels = extract_label_list(row, expected_label_source)
    if labels:
        return labels
    if fallback_expected_label_source:
        return extract_label_list(row, fallback_expected_label_source)
    return []


def is_harmful_outcome(label: str, harmful_outcomes: Sequence[str]) -> bool:
    return normalize_text(label) in {normalize_text(x) for x in harmful_outcomes}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Signature building for the new frequency model format
# -----------------------------------------------------------------------------

def make_signature_key_from_row(
    row: Dict[str, Any],
    expected_label_source: str,
    fallback_expected_label_source: Optional[str],
) -> str:
    task_label = normalize_text(row.get("task_label"))
    contract_label = normalize_text(row.get("contract_label"))
    agent_name = normalize_text(row.get("agent_name")) or "unknown"
    deps_key = deps_to_key(row.get("deps", []))
    dep_bucket = get_dep_bucket(row)
    expected_labels = get_expected_labels(
        row=row,
        expected_label_source=expected_label_source,
        fallback_expected_label_source=fallback_expected_label_source,
    )
    x_key = "+".join(expected_labels) if expected_labels else "none"

    return (
        f"task={task_label}"
        f"|contract={contract_label}"
        f"|agent={agent_name}"
        f"|deps={deps_key}"
        f"|dep_bucket={dep_bucket}"
        f"|x={x_key}"
    )


# -----------------------------------------------------------------------------
# Frequency model parsing
# -----------------------------------------------------------------------------

def get_risk_table(obj: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("frequency model JSON must be a dict.")
    risk_table = obj.get("risk_table")
    if not isinstance(risk_table, dict):
        raise ValueError("frequency model JSON must contain dict field 'risk_table'.")
    return risk_table


def available_oracle_fields(risk_table: Dict[str, Dict[str, Any]]) -> List[str]:
    candidates = [
        "oracle_verify_weighted_q80",
        "oracle_verify_simple_q80",
        "oracle_verify_weighted_q",
        "oracle_verify_simple_q",
    ]
    found: List[str] = []
    for name in candidates:
        if any(isinstance(v, dict) and name in v for v in risk_table.values()):
            found.append(name)
    return found


# -----------------------------------------------------------------------------
# Row attachment
# -----------------------------------------------------------------------------

def attach_frequency_model_to_rows(
    node_rows: List[Dict[str, Any]],
    risk_table: Dict[str, Dict[str, Any]],
    *,
    expected_label_source: str,
    fallback_expected_label_source: Optional[str],
    oracle_field: str,
    harmful_outcomes: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for row in node_rows:
        if not isinstance(row, dict):
            continue

        sig = make_signature_key_from_row(
            row=row,
            expected_label_source=expected_label_source,
            fallback_expected_label_source=fallback_expected_label_source,
        )
        entry = risk_table.get(sig)

        matched = entry is not None
        assigned_risk_simple = None
        assigned_risk_weighted = None
        existing_oracle = None
        label_risk = None
        label_support = None

        if matched:
            assigned_risk_simple = entry.get("node_risk_simple_mean")
            assigned_risk_weighted = entry.get("node_risk_support_weighted")
            existing_oracle = entry.get(oracle_field)
            label_risk = entry.get("label_risk")
            label_support = entry.get("label_support")

        outcome_label = normalize_text(row.get("outcome_label"))
        gold_verify = is_harmful_outcome(outcome_label, harmful_outcomes)

        out.append({
            **row,
            "exact_signature": sig,
            "matched": matched,
            "gold_verify": gold_verify,
            "assigned_risk_simple": assigned_risk_simple,
            "assigned_risk_weighted": assigned_risk_weighted,
            "existing_oracle_verify": existing_oracle,
            "label_risk": label_risk,
            "label_support": label_support,
        })

    return out


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def compute_metrics_from_predictions(rows: List[Dict[str, Any]], pred_field: str) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    evaluated = 0

    for row in rows:
        pred = row.get(pred_field)
        gold = row.get("gold_verify")

        if pred is None or gold is None:
            continue

        pred_b = bool(pred)
        gold_b = bool(gold)
        evaluated += 1

        if pred_b and gold_b:
            tp += 1
        elif pred_b and not gold_b:
            fp += 1
        elif (not pred_b) and gold_b:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    verify_rate = (tp + fp) / evaluated if evaluated else 0.0

    return {
        "num_evaluated_nodes": evaluated,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "miss_rate": miss_rate,
        "verify_rate": verify_rate,
    }


def evaluate_threshold(rows: List[Dict[str, Any]], threshold: float, risk_field: str) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    evaluated = 0

    for row in rows:
        risk = row.get(risk_field)
        gold = row.get("gold_verify")
        if risk is None or gold is None:
            continue

        pred = float(risk) >= threshold
        gold_b = bool(gold)
        evaluated += 1

        if pred and gold_b:
            tp += 1
        elif pred and not gold_b:
            fp += 1
        elif (not pred) and gold_b:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    verify_rate = (tp + fp) / evaluated if evaluated else 0.0

    return {
        "threshold": threshold,
        "num_evaluated_nodes": evaluated,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "miss_rate": miss_rate,
        "verify_rate": verify_rate,
    }


def sweep_thresholds(
    rows: List[Dict[str, Any]],
    *,
    risk_field: str,
    target_miss_rate: float,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    risks = sorted({
        float(row[risk_field])
        for row in rows
        if row.get(risk_field) is not None
    }, reverse=True)

    if not risks:
        return None, []

    thresholds = [max(risks) + 1e-12] + risks
    curve: List[Dict[str, Any]] = []
    chosen: Optional[Dict[str, Any]] = None

    for tau in thresholds:
        metrics = evaluate_threshold(rows, tau, risk_field)
        curve.append(metrics)
        if chosen is None and metrics["miss_rate"] <= target_miss_rate:
            chosen = metrics

    return chosen, curve


def apply_threshold(rows: List[Dict[str, Any]], threshold: float, risk_field: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rr = dict(row)
        risk = rr.get(risk_field)
        if risk is None:
            rr["oracle_verify"] = None
        else:
            rr["oracle_verify"] = float(risk) >= threshold
        out.append(rr)
    return out


def apply_existing_oracle(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rr = dict(row)
        rr["oracle_verify"] = rr.get("existing_oracle_verify")
        out.append(rr)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach oracle verify decisions from the new frequency_model_with_risk.json "
            "to node_signature_table_v2.json. Supports both exact oracle join and threshold resweep."
        )
    )
    parser.add_argument("--node-signature-table", required=True)
    parser.add_argument("--frequency-model", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--mode",
        choices=["use_existing_oracle", "resweep"],
        default="use_existing_oracle",
        help=(
            "use_existing_oracle reads oracle flags already stored in risk_table. "
            "resweep ignores stored oracle flags and chooses a new threshold from assigned risk."
        ),
    )
    parser.add_argument(
        "--oracle-field",
        default="oracle_verify_weighted_q80",
        help="Risk-table boolean field to use when mode=use_existing_oracle.",
    )
    parser.add_argument(
        "--risk-field",
        choices=["assigned_risk_weighted", "assigned_risk_simple"],
        default="assigned_risk_weighted",
        help="Risk field used for threshold sweep when mode=resweep.",
    )
    parser.add_argument(
        "--target-miss-rate",
        type=float,
        default=0.10,
        help="Target upper bound on miss rate when mode=resweep.",
    )
    parser.add_argument(
        "--harmful-outcomes",
        default="P,N,E",
        help="Comma-separated outcome labels treated as gold positives.",
    )
    parser.add_argument(
        "--expected-label-source",
        default="expected_exception_labels",
        help="Field used to build x in the exact signature key.",
    )
    parser.add_argument(
        "--fallback-expected-label-source",
        default="failure_reason_labels",
        help="Fallback field used when expected-label-source is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    node_rows = read_json(Path(args.node_signature_table))
    if not isinstance(node_rows, list):
        raise ValueError("node_signature_table JSON must be a list.")

    frequency_model = read_json(Path(args.frequency_model))
    risk_table = get_risk_table(frequency_model)

    harmful_outcomes = [normalize_text(x) for x in args.harmful_outcomes.split(",") if normalize_text(x)]

    rows = attach_frequency_model_to_rows(
        node_rows=node_rows,
        risk_table=risk_table,
        expected_label_source=args.expected_label_source,
        fallback_expected_label_source=args.fallback_expected_label_source,
        oracle_field=args.oracle_field,
        harmful_outcomes=harmful_outcomes,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_count = sum(1 for r in rows if r.get("matched"))
    unmatched_count = len(rows) - matched_count
    available_oracles = available_oracle_fields(risk_table)

    dump_json(rows, output_dir / "rows_with_joined_frequency_model.json")

    if args.mode == "use_existing_oracle":
        oracle_rows = apply_existing_oracle(rows)
        metrics = compute_metrics_from_predictions(oracle_rows, pred_field="oracle_verify")

        summary = {
            "status": "ok",
            "mode": args.mode,
            "oracle_field": args.oracle_field,
            "available_oracle_fields_in_risk_table": available_oracles,
            "harmful_outcomes": harmful_outcomes,
            "expected_label_source": args.expected_label_source,
            "fallback_expected_label_source": args.fallback_expected_label_source,
            "num_total_rows": len(rows),
            "num_matched_rows": matched_count,
            "num_unmatched_rows": unmatched_count,
            "match_coverage": matched_count / len(rows) if rows else 0.0,
            "metrics": metrics,
        }

        dump_json(oracle_rows, output_dir / "oracle_rows_with_threshold.json")
        dump_json(summary, output_dir / "oracle_threshold_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    chosen, curve = sweep_thresholds(
        rows=rows,
        risk_field=args.risk_field,
        target_miss_rate=float(args.target_miss_rate),
    )

    dump_json(curve, output_dir / "threshold_sweep_curve.json")
    with (output_dir / "threshold_sweep_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "num_evaluated_nodes",
                "tp", "fp", "fn", "tn",
                "precision", "recall", "f1", "accuracy", "miss_rate", "verify_rate",
            ],
        )
        writer.writeheader()
        for row in curve:
            writer.writerow(row)

    if chosen is None:
        summary = {
            "status": "no_threshold_found",
            "mode": args.mode,
            "risk_field": args.risk_field,
            "target_miss_rate": args.target_miss_rate,
            "harmful_outcomes": harmful_outcomes,
            "expected_label_source": args.expected_label_source,
            "fallback_expected_label_source": args.fallback_expected_label_source,
            "num_total_rows": len(rows),
            "num_matched_rows": matched_count,
            "num_unmatched_rows": unmatched_count,
            "match_coverage": matched_count / len(rows) if rows else 0.0,
            "available_oracle_fields_in_risk_table": available_oracles,
        }
        dump_json(summary, output_dir / "oracle_threshold_summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    oracle_rows = apply_threshold(rows, chosen["threshold"], args.risk_field)

    summary = {
        "status": "ok",
        "mode": args.mode,
        "risk_field": args.risk_field,
        "target_miss_rate": args.target_miss_rate,
        "harmful_outcomes": harmful_outcomes,
        "expected_label_source": args.expected_label_source,
        "fallback_expected_label_source": args.fallback_expected_label_source,
        "num_total_rows": len(rows),
        "num_matched_rows": matched_count,
        "num_unmatched_rows": unmatched_count,
        "match_coverage": matched_count / len(rows) if rows else 0.0,
        "available_oracle_fields_in_risk_table": available_oracles,
        "chosen_threshold": chosen["threshold"],
        "chosen_metrics": chosen,
    }

    dump_json(oracle_rows, output_dir / "oracle_rows_with_threshold.json")
    dump_json(summary, output_dir / "oracle_threshold_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()