#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

OUTCOME_LABELS = ["A", "P", "N", "E"]


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def canonicalize_string_list(values: Sequence[str]) -> List[str]:
    cleaned = sorted({normalize_text(v) for v in values if normalize_text(v)})
    return cleaned


def deps_to_key(deps: Any) -> str:
    if deps is None:
        return "[]"
    if not isinstance(deps, list):
        return f"[{normalize_text(deps)}]"
    cleaned = [normalize_text(x) for x in deps if normalize_text(x)]
    return "[" + ",".join(cleaned) + "]"


def extract_label_list(row: Dict[str, Any], field_name: str) -> List[str]:
    raw = row.get(field_name)

    if raw is None:
        return []

    if isinstance(raw, str):
        s = normalize_text(raw)
        if not s:
            return []
        if "+" in s:
            return canonicalize_string_list(s.split("+"))
        if "," in s:
            return canonicalize_string_list(s.split(","))
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

    return canonicalize_string_list(labels)


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


def make_state_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_label": normalize_text(row.get("task_label")),
        "contract_label": normalize_text(row.get("contract_label")),
        "agent_name": normalize_text(row.get("agent_name")) or "unknown",
        "deps": row.get("deps", []),
        "deps_key": deps_to_key(row.get("deps", [])),
        "dependency_bucket": get_dep_bucket(row),
    }


def make_state_key(meta: Dict[str, Any], dep_key_mode: str) -> str:
    parts = [
        f"task={meta['task_label']}",
        f"contract={meta['contract_label']}",
        f"agent={meta['agent_name']}",
    ]

    if dep_key_mode in {"both", "raw_only"}:
        parts.append(f"deps={meta['deps_key']}")
    if dep_key_mode in {"both", "bucket_only"}:
        parts.append(f"dep_bucket={meta['dependency_bucket']}")

    return "|".join(parts)


def make_signature_key(state_key: str, expected_labels: Sequence[str]) -> str:
    x_key = "+".join(canonicalize_string_list(expected_labels)) if expected_labels else "none"
    return f"{state_key}|x={x_key}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_weights(path: Path) -> Dict[str, Dict[str, float]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("weights JSON must be a dict: {label: {A:..., P:..., N:..., E:...}}")

    out: Dict[str, Dict[str, float]] = {}
    for label, row in raw.items():
        if not isinstance(row, dict):
            raise ValueError(f"weights row for '{label}' must be a dict")
        out[normalize_text(label)] = {
            y: float(row.get(y, 0.0)) for y in OUTCOME_LABELS
        }
    return out


def empirical_quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(float(x) for x in xs)
    if q <= 0.0:
        return sorted_xs[0]
    if q >= 1.0:
        return sorted_xs[-1]
    idx = int(math.floor(q * (len(sorted_xs) - 1)))
    idx = max(0, min(idx, len(sorted_xs) - 1))
    return sorted_xs[idx]


def build_observed_state_label_counts(
    rows: List[Dict[str, Any]],
    observed_label_source: str,
    dep_key_mode: str,
    include_none_label: bool = True,
) -> tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Any]]]:
    counts: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {y: 0.0 for y in OUTCOME_LABELS})
    )
    state_metadata: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        meta = make_state_metadata(row)
        state_key = make_state_key(meta, dep_key_mode)
        state_metadata[state_key] = meta

        outcome = normalize_text(row.get("outcome_label"))
        if outcome not in OUTCOME_LABELS:
            continue

        observed_labels = extract_label_list(row, observed_label_source)

        if observed_labels:
            for z in observed_labels:
                counts[state_key][z][outcome] += 1.0
        elif include_none_label:
            counts[state_key]["none"][outcome] += 1.0

    return counts, state_metadata


def build_state_label_conditional_probs(
    counts: Dict[str, Dict[str, Dict[str, float]]],
    alpha: float,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    conditional_probs: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    for state_key, per_label in counts.items():
        for z, outcome_counts in per_label.items():
            total = sum(float(outcome_counts.get(y, 0.0)) for y in OUTCOME_LABELS)
            denom = total + len(OUTCOME_LABELS) * alpha
            conditional_probs[state_key][z] = {
                y: (float(outcome_counts.get(y, 0.0)) + alpha) / denom
                for y in OUTCOME_LABELS
            }

    return conditional_probs


def build_signature_metadata_and_index(
    rows: List[Dict[str, Any]],
    dep_key_mode: str,
    expected_label_source: str,
    fallback_expected_label_source: Optional[str],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    signature_metadata: Dict[str, Dict[str, Any]] = {}
    signature_expected_labels: Dict[str, List[str]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        meta = make_state_metadata(row)
        state_key = make_state_key(meta, dep_key_mode)
        expected_labels = get_expected_labels(
            row=row,
            expected_label_source=expected_label_source,
            fallback_expected_label_source=fallback_expected_label_source,
        )
        sig = make_signature_key(state_key, expected_labels)

        signature_metadata[sig] = {
            "task_label": meta["task_label"],
            "contract_label": meta["contract_label"],
            "agent_name": meta["agent_name"],
            "deps": meta["deps"],
            "deps_key": meta["deps_key"],
            "dependency_bucket": meta["dependency_bucket"],
            "planning_exception_labels": canonicalize_string_list(expected_labels),
        }
        signature_expected_labels[sig] = canonicalize_string_list(expected_labels)

    return signature_metadata, signature_expected_labels


def compute_label_risk(
    probs: Dict[str, float],
    weight_row: Dict[str, float],
) -> float:
    return sum(float(weight_row.get(y, 0.0)) * float(probs.get(y, 0.0)) for y in OUTCOME_LABELS)


def build_risk_table(
    signature_metadata: Dict[str, Dict[str, Any]],
    signature_expected_labels: Dict[str, List[str]],
    state_label_counts: Dict[str, Dict[str, Dict[str, float]]],
    state_label_conditional_probs: Dict[str, Dict[str, Dict[str, float]]],
    weights: Dict[str, Dict[str, float]],
    dep_key_mode: str,
) -> Dict[str, Dict[str, Any]]:
    risk_table: Dict[str, Dict[str, Any]] = {}

    zero_probs = {y: 0.0 for y in OUTCOME_LABELS}

    for sig, meta in signature_metadata.items():
        state_key = make_state_key(meta, dep_key_mode)
        expected_labels = signature_expected_labels.get(sig, [])

        label_risk: Dict[str, float] = {}
        label_support: Dict[str, float] = {}
        label_conditional_probs: Dict[str, Dict[str, float]] = {}

        for z in expected_labels:
            observed_counts = state_label_counts.get(state_key, {}).get(z)
            observed_probs = state_label_conditional_probs.get(state_key, {}).get(z)

            if observed_counts is None or observed_probs is None:
                label_support[z] = 0.0
                label_conditional_probs[z] = dict(zero_probs)
                label_risk[z] = 0.0
                continue

            support = sum(float(observed_counts.get(y, 0.0)) for y in OUTCOME_LABELS)
            label_support[z] = support
            label_conditional_probs[z] = dict(observed_probs)

            weight_row = weights.get(z, {})
            label_risk[z] = compute_label_risk(observed_probs, weight_row)

        num_labels = len(expected_labels)
        support_count_total = sum(label_support.values())

        if num_labels > 0:
            node_risk_simple_mean = sum(label_risk.values()) / num_labels
        else:
            node_risk_simple_mean = 0.0

        if support_count_total > 0.0:
            node_risk_support_weighted = (
                sum(label_risk[z] * label_support[z] for z in expected_labels) / support_count_total
            )
        else:
            node_risk_support_weighted = 0.0

        risk_table[sig] = {
            "metadata": meta,
            "label_conditional_probs": label_conditional_probs,
            "label_risk": label_risk,
            "label_support": label_support,
            "num_labels": num_labels,
            "support_count_total": support_count_total,
            "node_risk_simple_mean": node_risk_simple_mean,
            "node_risk_support_weighted": node_risk_support_weighted,
        }

    return risk_table


def to_plain_dict(d: Any) -> Any:
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        return {k: to_plain_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [to_plain_dict(x) for x in d]
    return d


def build_frequency_model_with_risk(
    rows: List[Dict[str, Any]],
    *,
    alpha: float,
    weights: Dict[str, Dict[str, float]],
    observed_label_source: str,
    expected_label_source: str,
    fallback_expected_label_source: Optional[str],
    dep_key_mode: str,
    oracle_quantile: float,
) -> Dict[str, Any]:
    state_label_counts, state_metadata = build_observed_state_label_counts(
        rows=rows,
        observed_label_source=observed_label_source,
        dep_key_mode=dep_key_mode,
        include_none_label=True,
    )

    state_label_conditional_probs = build_state_label_conditional_probs(
        counts=state_label_counts,
        alpha=alpha,
    )

    signature_metadata, signature_expected_labels = build_signature_metadata_and_index(
        rows=rows,
        dep_key_mode=dep_key_mode,
        expected_label_source=expected_label_source,
        fallback_expected_label_source=fallback_expected_label_source,
    )

    risk_table = build_risk_table(
        signature_metadata=signature_metadata,
        signature_expected_labels=signature_expected_labels,
        state_label_counts=state_label_counts,
        state_label_conditional_probs=state_label_conditional_probs,
        weights=weights,
        dep_key_mode=dep_key_mode,
    )

    simple_threshold = empirical_quantile(
        [entry["node_risk_simple_mean"] for entry in risk_table.values()], oracle_quantile
    )
    weighted_threshold = empirical_quantile(
        [entry["node_risk_support_weighted"] for entry in risk_table.values()], oracle_quantile
    )

    simple_q80 = empirical_quantile(
        [entry["node_risk_simple_mean"] for entry in risk_table.values()], 0.80
    )
    weighted_q80 = empirical_quantile(
        [entry["node_risk_support_weighted"] for entry in risk_table.values()], 0.80
    )

    for entry in risk_table.values():
        entry["oracle_verify_simple_q"] = entry["node_risk_simple_mean"] >= simple_threshold
        entry["oracle_verify_weighted_q"] = entry["node_risk_support_weighted"] >= weighted_threshold
        entry["oracle_quantile"] = oracle_quantile
        entry["oracle_verify_simple_q80"] = entry["node_risk_simple_mean"] >= simple_q80
        entry["oracle_verify_weighted_q80"] = entry["node_risk_support_weighted"] >= weighted_q80

    return {
        "alpha": alpha,
        "outcome_labels": list(OUTCOME_LABELS),
        "observed_label_source": observed_label_source,
        "expected_label_source": expected_label_source,
        "fallback_expected_label_source": fallback_expected_label_source,
        "dep_key_mode": dep_key_mode,
        "state_metadata": state_metadata,
        "state_label_counts": to_plain_dict(state_label_counts),
        "state_label_conditional_probs": to_plain_dict(state_label_conditional_probs),
        "signature_metadata": signature_metadata,
        "risk_table": risk_table,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frequency_model_with_risk.json directly from node_signature_table_v2.json. "
            "Observed labels are used to estimate p(Y | state, label). "
            "Expected labels are used to assemble each signature's risk entry. "
            "If an expected label is unseen for that state, its risk is set to 0."
        )
    )
    parser.add_argument("--node-signature-table", required=True)
    parser.add_argument("--weights-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--oracle-quantile", type=float, default=0.80)
    parser.add_argument("--observed-label-source", default="failure_reason_labels")
    parser.add_argument("--expected-label-source", default="expected_exception_labels")
    parser.add_argument("--fallback-expected-label-source", default="failure_reason_labels")
    parser.add_argument(
        "--dep-key-mode",
        choices=["both", "bucket_only", "raw_only"],
        default="both",
        help="both uses raw deps and dep_bucket in the state key.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_json(Path(args.node_signature_table))
    if not isinstance(rows, list):
        raise ValueError("node_signature_table JSON must be a list of row dicts.")

    weights = load_weights(Path(args.weights_json))

    model = build_frequency_model_with_risk(
        rows=rows,
        alpha=float(args.alpha),
        weights=weights,
        observed_label_source=args.observed_label_source,
        expected_label_source=args.expected_label_source,
        fallback_expected_label_source=args.fallback_expected_label_source,
        dep_key_mode=args.dep_key_mode,
        oracle_quantile=float(args.oracle_quantile),
    )

    dump_json(model, Path(args.output_json))
    print(f"Wrote frequency model with risk to: {args.output_json}")


if __name__ == "__main__":
    main()