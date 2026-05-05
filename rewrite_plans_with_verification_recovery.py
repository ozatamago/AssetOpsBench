#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


QUESTION_PREFIX = "Question:"
PLAN_MARKER = "Conditional Plan JSON:"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite plan files by inserting verification and recovery nodes "
            "for oracle_verify=true nodes, while keeping each output node limited to "
            "id/task/agent/deps. Input nodes may optionally contain node_contract, "
            "which will be ignored."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--plan_dir",
        type=str,
        default="",
        help="Directory containing plan txt files such as Model_16_Q_1_plan.txt",
    )
    group.add_argument(
        "--plan_path",
        type=str,
        default="",
        help="Single plan txt file to rewrite",
    )

    parser.add_argument(
        "--oracle_path",
        type=str,
        required=True,
        help="Path to oracle_rows_with_threshold.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save rewritten plan txt files",
    )
    parser.add_argument(
        "--only_qids",
        type=str,
        default="",
        help='Optional comma-separated QIDs, e.g. "Q1,Q10,Q25"',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return parser.parse_args()


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def remove_trailing_commas(text: str) -> str:
    """
    Make the parser a bit more robust to trailing commas before } or ].
    """
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r",(\s*[}\]])", r"\1", current)
    return current


def extract_balanced_json_object(text: str) -> str:
    """
    Return the first balanced top-level JSON object found in text.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found.")

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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Could not find a balanced JSON object.")


def extract_question_and_plan_json_text(raw_text: str) -> Tuple[str, str]:
    question = ""
    m = re.search(r"^Question:\s*(.*)$", raw_text, flags=re.MULTILINE)
    if m:
        question = m.group(1).strip()

    if PLAN_MARKER in raw_text:
        tail = raw_text.split(PLAN_MARKER, 1)[1].strip()
        json_text = extract_balanced_json_object(tail)
        return question, json_text

    json_text = extract_balanced_json_object(raw_text)
    return question, json_text


def normalize_plan_structure(plan: Dict[str, Any], plan_path: Path) -> Dict[str, Any]:
    """
    Accept input nodes that contain:
      - required: id, task, agent, deps
      - optional: node_contract
    and normalize them so that output nodes contain only:
      - id, task, agent, deps
    """
    if not isinstance(plan, dict):
        raise ValueError(f"{plan_path}: plan root must be a JSON object.")

    if "answer_contract" not in plan:
        raise ValueError(f"{plan_path}: missing top-level key 'answer_contract'.")
    if "nodes" not in plan:
        raise ValueError(f"{plan_path}: missing top-level key 'nodes'.")
    if not isinstance(plan["nodes"], list):
        raise ValueError(f"{plan_path}: 'nodes' must be a list.")

    normalized_nodes: List[Dict[str, Any]] = []

    for i, node in enumerate(plan["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"{plan_path}: node index {i} is not an object.")

        required_keys = {"id", "task", "agent", "deps"}
        allowed_extra_keys = {"node_contract"}

        actual_keys = set(node.keys())
        missing_keys = required_keys - actual_keys
        unexpected_keys = actual_keys - required_keys - allowed_extra_keys

        if missing_keys:
            raise ValueError(
                f"{plan_path}: node index {i} is missing required keys "
                f"{sorted(missing_keys)}."
            )

        if unexpected_keys:
            raise ValueError(
                f"{plan_path}: node index {i} has unexpected keys "
                f"{sorted(unexpected_keys)}. Allowed keys are "
                f"{sorted(required_keys | allowed_extra_keys)}."
            )

        normalized_node = {
            "id": node["id"],
            "task": node["task"],
            "agent": node["agent"],
            "deps": node["deps"],
        }
        normalized_nodes.append(normalized_node)

    return {
        "answer_contract": plan["answer_contract"],
        "nodes": normalized_nodes,
    }


def load_plan_file(plan_path: Path) -> Tuple[str, Dict[str, Any]]:
    raw = plan_path.read_text(encoding="utf-8")
    question, json_text = extract_question_and_plan_json_text(raw)
    json_text = remove_trailing_commas(json_text)

    try:
        plan = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from {plan_path}. Error: {e}"
        ) from e

    plan = normalize_plan_structure(plan, plan_path)
    validate_plan_structure(plan, plan_path)
    return question, plan


def validate_plan_structure(plan: Dict[str, Any], plan_path: Path) -> None:
    if not isinstance(plan, dict):
        raise ValueError(f"{plan_path}: plan root must be a JSON object.")

    if "answer_contract" not in plan:
        raise ValueError(f"{plan_path}: missing top-level key 'answer_contract'.")
    if "nodes" not in plan:
        raise ValueError(f"{plan_path}: missing top-level key 'nodes'.")
    if not isinstance(plan["nodes"], list):
        raise ValueError(f"{plan_path}: 'nodes' must be a list.")

    seen_ids: Set[str] = set()

    for i, node in enumerate(plan["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"{plan_path}: node index {i} is not an object.")

        required_keys = {"id", "task", "agent", "deps"}
        actual_keys = set(node.keys())

        if actual_keys != required_keys:
            raise ValueError(
                f"{plan_path}: node index {i} must contain exactly "
                f"{sorted(required_keys)}, but got {sorted(actual_keys)}."
            )

        node_id = node["id"]
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"{plan_path}: node index {i} has invalid 'id'.")
        if node_id in seen_ids:
            raise ValueError(f"{plan_path}: duplicate node id '{node_id}'.")
        seen_ids.add(node_id)

        if not isinstance(node["task"], str):
            raise ValueError(f"{plan_path}: node '{node_id}' has non-string 'task'.")
        if not isinstance(node["agent"], str):
            raise ValueError(f"{plan_path}: node '{node_id}' has non-string 'agent'.")
        if not isinstance(node["deps"], list) or not all(
            isinstance(dep, str) for dep in node["deps"]
        ):
            raise ValueError(
                f"{plan_path}: node '{node_id}' has invalid 'deps'; "
                "it must be a list of strings."
            )

    for node in plan["nodes"]:
        for dep in node["deps"]:
            if dep not in seen_ids:
                raise ValueError(
                    f"{plan_path}: node '{node['id']}' depends on unknown node '{dep}'."
                )


def load_oracle_rows(oracle_path: Path) -> List[Dict[str, Any]]:
    raw = oracle_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        rows = data["rows"]
    else:
        raise ValueError(
            f"{oracle_path}: oracle JSON must be either a list or an object with a 'rows' list."
        )

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{oracle_path}: row index {i} is not an object.")

    return rows


def build_oracle_verify_map(rows: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Returns:
        {
            "Q1": {"S1", ...},
            "Q10": {"S1", "S3", ...},
            ...
        }
    """
    qid_to_nodes: Dict[str, Set[str]] = {}

    for row in rows:
        qid = row.get("qid")
        node_id = row.get("node_id")
        oracle_verify = normalize_bool(row.get("oracle_verify", False))

        if not isinstance(qid, str) or not qid:
            continue
        if not isinstance(node_id, str) or not node_id:
            continue
        if not oracle_verify:
            continue

        qid_to_nodes.setdefault(qid, set()).add(node_id)

    return qid_to_nodes


def extract_qid_from_filename(path: Path) -> str:
    """
    Example:
        Model_16_Q_1_plan.txt -> Q1
    """
    m = re.search(r"_Q_(\d+)_plan\.txt$", path.name)
    if not m:
        raise ValueError(
            f"Could not extract QID from filename '{path.name}'. "
            "Expected a name like 'Model_16_Q_1_plan.txt'."
        )
    return f"Q{m.group(1)}"


def make_verification_node(original_node_id: str, original_task: str) -> Dict[str, Any]:
    return {
        "id": f"V_{original_node_id}",
        "task": (
            f"Verify whether node {original_node_id} correctly completed its task: "
            f"{original_task}"
        ),
        "agent": "Verification Agent",
        "deps": [original_node_id],
    }


def make_recovery_node(
    original_node_id: str,
    original_task: str,
    rewritten_upstream_deps: List[str],
) -> Dict[str, Any]:
    verification_node_id = f"V_{original_node_id}"
    return {
        "id": f"R_{original_node_id}",
        "task": original_task,
        "agent": "Recovery Agent",
        # 重要:
        # R_Sx は V_Sx だけではなく、
        # Sx が本来必要としていた upstream final outputs も deps に持つ
        "deps": list(rewritten_upstream_deps) + [verification_node_id],
    }


def rewrite_plan(
    plan: Dict[str, Any],
    verify_node_ids: Set[str],
    qid: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    original_nodes: List[Dict[str, Any]] = plan["nodes"]
    original_ids = {node["id"] for node in original_nodes}

    for node_id in original_ids:
        if node_id.startswith("V_") or node_id.startswith("R_"):
            raise ValueError(
                f"Input plan already contains rewritten-looking node id '{node_id}'. "
                "This script expects an original plan."
            )

    unknown_verify_node_ids = sorted(verify_node_ids - original_ids)
    valid_verify_node_ids = verify_node_ids & original_ids

    if unknown_verify_node_ids:
        prefix = f"[{qid}] " if qid else ""
        logger.warning(
            "%sOracle requested verification for unknown node ids: %s. "
            "They will be skipped.",
            prefix,
            unknown_verify_node_ids,
        )

    collision_ids = {f"V_{nid}" for nid in valid_verify_node_ids} | {
        f"R_{nid}" for nid in valid_verify_node_ids
    }
    if original_ids & collision_ids:
        raise ValueError(
            "ID collision detected between original nodes and generated V_/R_ nodes."
        )

    # downstream で参照すべき "final output node"
    # verify/recovery が入る node は R_<node_id> に差し替える
    dep_rewrite_map = {nid: f"R_{nid}" for nid in valid_verify_node_ids}

    new_nodes: List[Dict[str, Any]] = []

    for node in original_nodes:
        node_id = node["id"]

        # 元 node Sx の deps を、upstream final deps に書き換える
        rewritten_upstream_deps = [
            dep_rewrite_map.get(dep, dep) for dep in node["deps"]
        ]

        # 通常実行 node Sx
        rewritten_node = copy.deepcopy(node)
        rewritten_node["deps"] = rewritten_upstream_deps
        new_nodes.append(rewritten_node)

        # verify/recovery を入れる node なら、
        # V_Sx は Sx に依存し、
        # R_Sx は rewritten upstream deps + V_Sx に依存する
        if node_id in valid_verify_node_ids:
            new_nodes.append(make_verification_node(node_id, node["task"]))
            new_nodes.append(
                make_recovery_node(
                    original_node_id=node_id,
                    original_task=node["task"],
                    rewritten_upstream_deps=rewritten_upstream_deps,
                )
            )

    rewritten_plan = {
        "answer_contract": plan["answer_contract"],
        "nodes": new_nodes,
    }
    return rewritten_plan, unknown_verify_node_ids


def serialize_plan(question: str, plan: Dict[str, Any]) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    if question:
        return f"{QUESTION_PREFIX} {question}\n{PLAN_MARKER}\n{plan_json}\n"

    return f"{PLAN_MARKER}\n{plan_json}\n"


def discover_plan_files(plan_dir: Path) -> List[Path]:
    files = sorted(plan_dir.glob("*_plan.txt"))
    if not files:
        raise ValueError(f"No '*_plan.txt' files found in {plan_dir}")
    return files


def parse_only_qids(raw: str) -> Set[str]:
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> None:
    args = parse_args()

    oracle_path = Path(args.oracle_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rewritten_dir = output_dir / "Rewritten files"
    unchanged_dir = output_dir / "Unchanged files"
    rewritten_dir.mkdir(parents=True, exist_ok=True)
    unchanged_dir.mkdir(parents=True, exist_ok=True)

    rows = load_oracle_rows(oracle_path)
    verify_map = build_oracle_verify_map(rows)
    only_qids = parse_only_qids(args.only_qids)

    if args.plan_path:
        plan_files = [Path(args.plan_path)]
    else:
        plan_files = discover_plan_files(Path(args.plan_dir))

    rewritten_count = 0
    unchanged_count = 0
    summary = []

    for plan_path in plan_files:
        qid = extract_qid_from_filename(plan_path)

        if only_qids and qid not in only_qids:
            continue

        question, plan = load_plan_file(plan_path)
        verify_node_ids = verify_map.get(qid, set())

        rewritten_plan, skipped_oracle_nodes = rewrite_plan(
            plan,
            verify_node_ids,
            qid=qid,
        )

        valid_verified_nodes = sorted(set(verify_node_ids) - set(skipped_oracle_nodes))

        if valid_verified_nodes:
            target_dir = rewritten_dir
            rewritten_count += 1
        else:
            target_dir = unchanged_dir
            unchanged_count += 1

        output_path = target_dir / plan_path.name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Use --overwrite to replace it."
            )

        output_text = serialize_plan(question, rewritten_plan)
        output_path.write_text(output_text, encoding="utf-8")

        summary.append(
            {
                "qid": qid,
                "input_plan_path": str(plan_path),
                "output_plan_path": str(output_path),
                "bucket": "rewritten" if valid_verified_nodes else "unchanged",
                "verified_original_nodes": valid_verified_nodes,
                "skipped_oracle_nodes": skipped_oracle_nodes,
                "original_node_count": len(plan["nodes"]),
                "rewritten_node_count": len(rewritten_plan["nodes"]),
            }
        )

    summary_path = output_dir / "rewrite_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Processed files: {len(summary)}")
    print(f"Rewritten files: {rewritten_count} -> {rewritten_dir}")
    print(f"Unchanged files: {unchanged_count} -> {unchanged_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()