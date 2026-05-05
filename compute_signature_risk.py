#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_WEIGHTS = {
    "A": 0.0,
    "P": 0.5,
    "N": 1.0,
    "E": 1.5,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def validate_top_level_schema(data: Dict[str, Any]) -> None:
    required_keys = [
        "outcome_labels",
        "signature_metadata",
        "counts",
        "conditional_probs",
        "risk_table",
    ]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing top-level key: {key}")

    outcome_labels = data["outcome_labels"]
    expected_labels = {"A", "P", "N", "E"}
    if set(outcome_labels) != expected_labels:
        raise ValueError(
            f"outcome_labels must be exactly {sorted(expected_labels)}, got {outcome_labels}"
        )


def check_probability_vector(prob_dict: Dict[str, float], tol: float = 1e-8) -> None:
    required = {"A", "P", "N", "E"}
    missing = required - set(prob_dict.keys())
    if missing:
        raise ValueError(f"Probability dict missing outcome keys: {sorted(missing)}")

    total = float(prob_dict["A"] + prob_dict["P"] + prob_dict["N"] + prob_dict["E"])
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Probability dict does not sum to 1.0. sum={total}, dict={prob_dict}"
        )


def compute_label_risk(prob_dict: Dict[str, float], weights: Dict[str, float]) -> float:
    check_probability_vector(prob_dict)
    return (
        weights["A"] * float(prob_dict["A"])
        + weights["P"] * float(prob_dict["P"])
        + weights["N"] * float(prob_dict["N"])
        + weights["E"] * float(prob_dict["E"])
    )


def compute_label_support(count_dict: Dict[str, float]) -> float:
    return float(
        count_dict.get("A", 0.0)
        + count_dict.get("P", 0.0)
        + count_dict.get("N", 0.0)
        + count_dict.get("E", 0.0)
    )


def quantile(sorted_vals, q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])

    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def build_risk_table(
    data: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    signature_metadata = data["signature_metadata"]
    counts = data["counts"]
    conditional_probs = data["conditional_probs"]

    risk_table: Dict[str, Any] = {}

    simple_risks = []
    weighted_risks = []

    # signature_metadata を基準に回す
    for signature, metadata in signature_metadata.items():
        label_prob_map = conditional_probs.get(signature, {})
        label_count_map = counts.get(signature, {})

        label_risk: Dict[str, float] = {}
        label_support: Dict[str, float] = {}

        weighted_sum = 0.0
        total_support = 0.0

        for label, prob_dict in label_prob_map.items():
            r = compute_label_risk(prob_dict, weights)
            label_risk[label] = r

            support = 0.0
            if label in label_count_map:
                support = compute_label_support(label_count_map[label])
            label_support[label] = support

            weighted_sum += support * r
            total_support += support

        if label_risk:
            node_risk_simple_mean = sum(label_risk.values()) / len(label_risk)
            simple_risks.append(node_risk_simple_mean)
        else:
            node_risk_simple_mean = None

        if total_support > 0.0:
            node_risk_support_weighted = weighted_sum / total_support
            weighted_risks.append(node_risk_support_weighted)
        else:
            node_risk_support_weighted = None

        risk_table[signature] = {
            "metadata": metadata,
            "label_risk": label_risk,
            "label_support": label_support,
            "num_labels": len(label_risk),
            "support_count_total": total_support,
            "node_risk_simple_mean": node_risk_simple_mean,
            "node_risk_support_weighted": node_risk_support_weighted,
        }

    simple_sorted = sorted(simple_risks)
    weighted_sorted = sorted(weighted_risks)

    tau_summary = {
        "weights": weights,
        "tau_simple_q50": quantile(simple_sorted, 0.50),
        "tau_simple_q75": quantile(simple_sorted, 0.75),
        "tau_simple_q80": quantile(simple_sorted, 0.80),
        "tau_simple_q90": quantile(simple_sorted, 0.90),
        "tau_simple_q95": quantile(simple_sorted, 0.95),
        "tau_weighted_q50": quantile(weighted_sorted, 0.50),
        "tau_weighted_q75": quantile(weighted_sorted, 0.75),
        "tau_weighted_q80": quantile(weighted_sorted, 0.80),
        "tau_weighted_q90": quantile(weighted_sorted, 0.90),
        "tau_weighted_q95": quantile(weighted_sorted, 0.95),
        "num_signatures": len(signature_metadata),
    }

    # 例として q80 を oracle_verify の境界に使う
    tau_simple = tau_summary["tau_simple_q80"]
    tau_weighted = tau_summary["tau_weighted_q80"]

    for signature, entry in risk_table.items():
        rs = entry["node_risk_simple_mean"]
        rw = entry["node_risk_support_weighted"]

        entry["oracle_verify_simple_q80"] = (
            False if rs is None or tau_simple is None else rs > tau_simple
        )
        entry["oracle_verify_weighted_q80"] = (
            False if rw is None or tau_weighted is None else rw > tau_weighted
        )

    return risk_table, tau_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute risk for each node signature from frequency_model JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input frequency_model JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON. If omitted, a new file with suffix _with_risk.json is created.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file directly.",
    )
    parser.add_argument("--wP", type=float, default=0.5, help="Weight for P.")
    parser.add_argument("--wN", type=float, default=1.0, help="Weight for N.")
    parser.add_argument("--wE", type=float, default=1.5, help="Weight for E.")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.inplace and args.output is not None:
        raise ValueError("Use either --inplace or --output, not both.")

    if args.inplace:
        output_path = input_path
    elif args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_with_risk.json")

    weights = {
        "A": 0.0,
        "P": float(args.wP),
        "N": float(args.wN),
        "E": float(args.wE),
    }

    data = load_json(input_path)
    validate_top_level_schema(data)

    risk_table, tau_summary = build_risk_table(data, weights)

    data["risk_weights"] = weights
    data["risk_table"] = risk_table
    data["tau_summary"] = tau_summary

    save_json(output_path, data)

    print(f"Saved: {output_path}")
    print("tau_summary:")
    for k, v in tau_summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()