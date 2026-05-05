#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# -----------------------------------------------------------------------------
# Dynamic import of the original builder
# -----------------------------------------------------------------------------

def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("freq_builder_module", str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {py_path}")

    module = importlib.util.module_from_spec(spec)

    # Python 3.12 dataclasses workaround
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


# -----------------------------------------------------------------------------
# Output dataclasses
# -----------------------------------------------------------------------------

@dataclass
class NodeSignatureRecord:
    qid: str
    node_id: str
    task_text: str
    node_contract_text: str
    agent_name: str
    deps: List[str]
    dep_bucket: str
    task_label: str
    contract_label: str
    outcome_label: str
    failure_reason_labels: List[str]
    plan_path: str
    trajectory_path: str


@dataclass
class NodeOutcomeUpdateRecord:
    qid: str
    node_id: str
    outcome_label: str
    failure_reason_labels: List[str]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", _as_str(x).strip())


def _norm_list_str(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    for x in xs:
        s = _norm_text(x)
        if s:
            out.append(s)
    return out


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (_as_str(row.get("qid")), _as_str(row.get("node_id")))


# -----------------------------------------------------------------------------
# Exception taxonomy adaptation
# -----------------------------------------------------------------------------

def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        return default
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _iter_exception_taxonomy_leaves(obj: Any) -> Iterable[Any]:
    """
    Supports:
      - old flat list of builder objects
      - flat list of dict leaves
      - hierarchical dict/list with children
    """
    if obj is None:
        return

    if isinstance(obj, list):
        for item in obj:
            yield from _iter_exception_taxonomy_leaves(item)
        return

    if isinstance(obj, dict):
        children = obj.get("children")
        if isinstance(children, list):
            for child in children:
                yield from _iter_exception_taxonomy_leaves(child)
            return
        yield obj
        return

    # builder object or unknown leaf-like object
    yield obj


def exception_leaf_label(c: Any) -> str:
    # New hierarchical / flat JSON first
    s = _norm_text(_get_field(c, "full_label", "category_name", "label", "name"))
    if s:
        return s
    raise ValueError(f"Could not determine exception label from taxonomy leaf: {c!r}")


def exception_leaf_definition(c: Any) -> str:
    return _norm_text(_get_field(c, "description", "definition", default=""))


def exception_leaf_representative_signals(c: Any) -> List[str]:
    vals = _get_field(c, "representative_signals", "inclusion_criteria", default=[])
    return _norm_list_str(vals)


def exception_leaf_applies_when(c: Any) -> List[str]:
    vals = _get_field(c, "applies_when", default=[])
    return _norm_list_str(vals)


def exception_leaf_not_applies_when(c: Any) -> List[str]:
    vals = _get_field(c, "not_applies_when", default=[])
    return _norm_list_str(vals)


def build_exception_taxonomy_signature(exception_taxonomy: Sequence[Any]) -> str:
    rows: List[Dict[str, Any]] = []
    for c in _iter_exception_taxonomy_leaves(exception_taxonomy):
        rows.append({
            "label": exception_leaf_label(c),
            "definition": exception_leaf_definition(c),
            "representative_signals": exception_leaf_representative_signals(c),
            "applies_when": exception_leaf_applies_when(c),
            "not_applies_when": exception_leaf_not_applies_when(c),
        })
    rows.sort(key=lambda x: x["label"])
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# -----------------------------------------------------------------------------
# Cache / retry helpers
# -----------------------------------------------------------------------------

def evict_cached_result(builder, cache, model_signature: str, purpose: str, payload: Dict[str, Any]) -> None:
    key = builder.stable_hash({
        "purpose": purpose,
        "payload": payload,
        "model": model_signature,
    })
    if cache.get(key) is not None:
        cache.data.pop(key, None)
        if cache.path is not None:
            builder.ensure_dir(cache.path.parent)
            builder.dump_json(cache.data, cache.path)


def call_with_retry(
    *,
    builder,
    cache,
    model_signature: str,
    purpose: str,
    payload: Dict[str, Any],
    fn,
    max_retries: int = 3,
) -> Tuple[Optional[Any], int, Optional[str]]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            value = fn()
            return value, attempt, None
        except Exception as e:
            last_error = e
            evict_cached_result(
                builder=builder,
                cache=cache,
                model_signature=model_signature,
                purpose=purpose,
                payload=payload,
            )
            time.sleep(0.2)
    return None, max_retries, f"{type(last_error).__name__}: {last_error}"


# -----------------------------------------------------------------------------
# Existing table reuse
# -----------------------------------------------------------------------------

def load_existing_node_signature_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("existing_node_signature_table must be a JSON array.")
    return [x for x in data if isinstance(x, dict)]


def load_existing_node_signature_index(path: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows = load_existing_node_signature_rows(path)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        out[row_key(row)] = row
    return out


# -----------------------------------------------------------------------------
# Reviews / status parsing
# -----------------------------------------------------------------------------

def node_id_to_task_number(node_id: str) -> Optional[str]:
    m = re.search(r"(\d+)", str(node_id))
    if not m:
        return None
    return m.group(1)


def normalize_status_to_outcome_label(raw_status: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", str(raw_status).strip().lower())

    if not s:
        return None

    if s in {"accomplished", "success", "successful", "done", "completed"}:
        return "A"

    if "partially accomplished" in s or s == "partial" or s == "partially completed":
        return "P"

    if "not accomplished" in s or s == "not completed" or s == "incomplete" or s == "failed":
        return "N"

    if "error" in s or "exception" in s or "crash" in s:
        return "E"

    return None


def parse_task_status_from_review_text(review_text: str) -> Optional[str]:
    if not review_text:
        return None

    m = re.search(r"Task Status:\s*([^\n\r]+)", review_text, flags=re.IGNORECASE)
    if not m:
        return None

    raw_status = m.group(1).strip()
    return normalize_status_to_outcome_label(raw_status)


def join_reviews(reviews: Any) -> str:
    if not isinstance(reviews, list):
        return ""
    parts: List[str] = []
    for x in reviews:
        if isinstance(x, str) and x.strip():
            parts.append(x.strip())
    return "\n\n".join(parts)


def extract_response_status(response: Any) -> Optional[str]:
    if not isinstance(response, list):
        return None
    for x in response:
        if isinstance(x, dict):
            status = x.get("status")
            if isinstance(status, str) and status.strip():
                return status.strip()
    return None


def build_raw_trajectory_aux_index(traj_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build lookup by task_number string.
    """
    out: Dict[str, Dict[str, Any]] = {}
    raw_nodes = traj_doc.get("trajectory", [])
    if not isinstance(raw_nodes, list):
        return out

    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        task_number = item.get("task_number")
        if task_number is None:
            continue

        key = str(task_number)
        review_text = join_reviews(item.get("reviews"))
        response_status = extract_response_status(item.get("response"))

        out[key] = {
            "review_text": review_text,
            "response_status": response_status,
            "final_answer": item.get("final_answer", ""),
            "raw_item": item,
        }

    return out


def extract_execution_evidence(
    *,
    builder,
    qid: str,
    node_id: str,
    trajectory_path: Path,
    logs_max_chars: int,
) -> Dict[str, str]:
    """
    Extract review_text / response_status / node_output_text / logs_text / raw_status
    from a trajectory JSON using the row's node_id and qid.
    """
    traj_doc = builder.read_json(trajectory_path)
    raw_aux_index = build_raw_trajectory_aux_index(traj_doc)

    task_number = node_id_to_task_number(node_id)
    raw_aux = raw_aux_index.get(task_number or "", {})

    review_text = _as_str(raw_aux.get("review_text", ""))
    response_status = _as_str(raw_aux.get("response_status", ""))
    node_output_text = _as_str(raw_aux.get("final_answer", ""))
    raw_status = ""
    logs_text = ""

    try:
        traj_nodes = builder.extract_trajectory_nodes(traj_doc, qid=qid, source_path=trajectory_path)
    except Exception:
        traj_nodes = []

    matched = None
    for t in traj_nodes:
        t_node_id = _as_str(getattr(t, "node_id", ""))
        if t_node_id and t_node_id == node_id:
            matched = t
            break
        t_task_number = _as_str(getattr(t, "task_number", ""))
        if task_number and t_task_number == task_number:
            matched = t
            break

    if matched is not None:
        raw_status = _as_str(getattr(matched, "raw_status", ""))
        review_text = review_text or _as_str(getattr(matched, "review_text", ""))
        node_output_text = _as_str(getattr(matched, "node_output_text", "")) or node_output_text
        logs_text = _as_str(getattr(matched, "logs_text", ""))

    if not logs_text:
        raw_item = raw_aux.get("raw_item", {})
        try:
            logs_text = builder.truncate_text(json.dumps(raw_item, ensure_ascii=False), logs_max_chars)
        except Exception:
            logs_text = ""

    return {
        "review_text": review_text,
        "response_status": response_status,
        "node_output_text": node_output_text,
        "logs_text": logs_text,
        "trajectory_status_raw": raw_status,
    }


# -----------------------------------------------------------------------------
# Outcome resolution
# -----------------------------------------------------------------------------

def resolve_outcome_label(
    *,
    builder,
    labeler,
    llm,
    cache,
    qid: str,
    node_id: str,
    plan_path: str,
    trajectory_path: str,
    task_text: str,
    node_contract_text: str,
    evidence: Dict[str, str],
    existing_outcome_label: Optional[str] = None,
) -> Tuple[Optional[str], str, Optional[Dict[str, Any]]]:
    """
    Returns:
      (outcome_label, source, error_record_if_any)
    source in {"existing", "review", "response_status", "raw_status", "llm"}
    """
    existing = _norm_text(existing_outcome_label)
    if existing in {"A", "P", "N", "E"}:
        return existing, "existing", None

    review_text = evidence.get("review_text", "")
    response_status = evidence.get("response_status", "")
    trajectory_status_raw = evidence.get("trajectory_status_raw", "")
    node_output_text = evidence.get("node_output_text", "")
    logs_text = evidence.get("logs_text", "")

    outcome_label = parse_task_status_from_review_text(review_text)
    if outcome_label is not None:
        return outcome_label, "review", None

    outcome_label = normalize_status_to_outcome_label(response_status or "")
    if outcome_label is not None:
        return outcome_label, "response_status", None

    outcome_label = normalize_status_to_outcome_label(trajectory_status_raw or "")
    if outcome_label is not None:
        return outcome_label, "raw_status", None

    merged_context = {
        "task_text": task_text,
        "node_contract_text": node_contract_text,
        "trajectory_status_raw": trajectory_status_raw,
        "node_output_text": node_output_text,
        "review_text": review_text,
        "logs_text": logs_text,
    }
    outcome_payload = {
        "task_text": task_text,
        "node_contract_text": node_contract_text,
        "trajectory_status_raw": trajectory_status_raw,
        "node_output_text": builder.truncate_text(node_output_text, 2500),
        "review_text": builder.truncate_text(review_text, 2500),
        "logs_text": builder.truncate_text(logs_text, 2500),
    }
    value, attempts, err = call_with_retry(
        builder=builder,
        cache=cache,
        model_signature=llm.model_signature,
        purpose="outcome_label",
        payload=outcome_payload,
        fn=lambda: labeler.label_outcome(merged_context),
        max_retries=3,
    )
    if value is None:
        return None, "llm", {
            "qid": qid,
            "node_id": node_id,
            "stage": "outcome_label",
            "attempts": attempts,
            "plan_path": plan_path,
            "trajectory_path": trajectory_path,
            "error": err,
        }
    return value, "llm", None


# -----------------------------------------------------------------------------
# Multi-label failure reason labeling
# -----------------------------------------------------------------------------

def label_failure_reasons_multi(
    *,
    builder,
    llm,
    cache,
    exception_taxonomy: Sequence[Any],
    exception_taxonomy_signature: str,
    task_text: str,
    node_contract_text: str,
    outcome_label: str,
    node_output_text: str,
    review_text: str,
    logs_text: str,
    max_labels: int = 5,
) -> List[str]:
    """
    Multi-label version of failure reason classification.
    Returns [] when outcome is A.
    For P/N/E, returns one or more exception taxonomy labels.
    """
    if outcome_label not in {"P", "N", "E"}:
        return []

    allowed_rows = []
    allowed_labels = []

    for c in _iter_exception_taxonomy_leaves(exception_taxonomy):
        label = exception_leaf_label(c)
        allowed_labels.append(label)
        allowed_rows.append({
            "label": label,
            "definition": exception_leaf_definition(c),
            "representative_signals": exception_leaf_representative_signals(c),
            "applies_when": exception_leaf_applies_when(c),
            "not_applies_when": exception_leaf_not_applies_when(c),
        })

    payload = {
        "task_text": task_text,
        "node_contract_text": node_contract_text,
        "outcome_label": outcome_label,
        "node_output_text": builder.truncate_text(node_output_text, 2500),
        "review_text": builder.truncate_text(review_text, 2500),
        "logs_text": builder.truncate_text(logs_text, 2500),
        "allowed_labels": allowed_labels,
        "max_labels": max_labels,
        "exception_taxonomy_signature": exception_taxonomy_signature,
    }

    # purpose name changed so old cache entries from the previous taxonomy format
    # are not accidentally reused.
    purpose = "failure_reason_labels_multi_v2_taxonomy_aware"

    key = builder.stable_hash({
        "purpose": purpose,
        "payload": payload,
        "model": llm.model_signature,
    })

    cached = cache.get(key)
    if cached is not None:
        obj = cached
    else:
        system_prompt = (
            "You are a precise multi-label failure-reason classifier. "
            "Given an executed node with a negative outcome, assign one or more applicable failure-reason labels "
            "from the provided exception taxonomy. "
            "Use all available evidence, including the reviewer text, execution logs, and produced output. "
            "Prefer the most upstream, most causally central failure labels. "
            "Do not duplicate both an upstream cause and its downstream symptom unless the evidence strongly supports both. "
            "Return JSON only."
        )

        user_prompt = f"""
Assign one or more applicable failure-reason labels for this node.

Rules:
- Only choose labels from the allowed labels below.
- Use the reviewer reasoning, logs, and produced output jointly.
- Return all labels that are strongly supported by the evidence.
- Do not include weakly speculative labels.
- If exactly one label is clearly dominant, returning one label is allowed.
- If multiple labels are clearly supported, return multiple labels.
- Prefer the most causally central labels when labels are near-duplicates.
- Do not invent new labels.
- Return at most {max_labels} labels.
- If outcome_label is P, N, or E, you must return at least one label.
- Do not return an empty labels list for P, N, or E.

Allowed exception labels:
{json.dumps(allowed_rows, ensure_ascii=False, indent=2)}

Node context:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "labels": ["<label1>", "<label2>"],
  "rationale": "<brief explanation>"
}}
""".strip()

        obj = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        cache.set(key, obj)

    raw_labels = obj.get("labels", [])
    if not isinstance(raw_labels, list):
        raise ValueError(f"Invalid multi-label failure reason response: {raw_labels}")

    out: List[str] = []
    allowed = set(allowed_labels)
    for x in raw_labels:
        lab = _norm_text(x)
        if not lab:
            continue
        if lab not in allowed:
            raise ValueError(f"Invalid failure reason label from LLM: {lab}")
        if lab not in out:
            out.append(lab)

    if outcome_label in {"P", "N", "E"} and len(out) == 0:
        raise ValueError(
            f"Empty failure_reason_labels for harmful outcome '{outcome_label}'. "
            "At least one exception label is required."
        )

    return out


# -----------------------------------------------------------------------------
# Existing-row-first relabel path
# -----------------------------------------------------------------------------

def relabel_from_existing_rows(
    *,
    builder,
    labeler,
    llm,
    cache,
    exception_taxonomy,
    exception_taxonomy_signature: str,
    existing_rows: List[Dict[str, Any]],
    logs_max_chars: int,
) -> Tuple[List[NodeSignatureRecord], List[NodeOutcomeUpdateRecord], List[Dict[str, Any]], Dict[str, Any], Set[Tuple[str, str]]]:
    records: List[NodeSignatureRecord] = []
    updates: List[NodeOutcomeUpdateRecord] = []
    errors: List[Dict[str, Any]] = []
    processed_keys: Set[Tuple[str, str]] = set()

    existing_rows_total = 0
    relabeled_from_existing = 0
    existing_duplicate_skipped = 0
    reused_outcome_labels = 0
    reparsed_outcome_labels = 0
    evidence_extract_failures = 0

    for row in existing_rows:
        existing_rows_total += 1
        qid, node_id = row_key(row)
        if not qid or not node_id:
            errors.append({
                "qid": qid,
                "node_id": node_id,
                "stage": "existing_row_validation",
                "error": "Missing qid or node_id in existing row.",
                "row": row,
            })
            continue

        key = (qid, node_id)
        if key in processed_keys:
            existing_duplicate_skipped += 1
            continue

        trajectory_path_str = _as_str(row.get("trajectory_path"))
        if not trajectory_path_str:
            evidence_extract_failures += 1
            errors.append({
                "qid": qid,
                "node_id": node_id,
                "stage": "existing_row_evidence",
                "error": "Missing trajectory_path in existing row.",
                "row": row,
            })
            continue

        trajectory_path = Path(trajectory_path_str)
        if not trajectory_path.exists():
            evidence_extract_failures += 1
            errors.append({
                "qid": qid,
                "node_id": node_id,
                "stage": "existing_row_evidence",
                "trajectory_path": trajectory_path_str,
                "error": "trajectory_path does not exist.",
            })
            continue

        try:
            evidence = extract_execution_evidence(
                builder=builder,
                qid=qid,
                node_id=node_id,
                trajectory_path=trajectory_path,
                logs_max_chars=logs_max_chars,
            )
        except Exception as e:
            evidence_extract_failures += 1
            errors.append({
                "qid": qid,
                "node_id": node_id,
                "stage": "existing_row_evidence",
                "trajectory_path": trajectory_path_str,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        outcome_label, outcome_source, outcome_err = resolve_outcome_label(
            builder=builder,
            labeler=labeler,
            llm=llm,
            cache=cache,
            qid=qid,
            node_id=node_id,
            plan_path=_as_str(row.get("plan_path")),
            trajectory_path=trajectory_path_str,
            task_text=_as_str(row.get("task_text")),
            node_contract_text=_as_str(row.get("node_contract_text")),
            evidence=evidence,
            existing_outcome_label=_as_str(row.get("outcome_label")),
        )
        if outcome_err is not None:
            errors.append(outcome_err)
            continue
        if outcome_source == "existing":
            reused_outcome_labels += 1
        else:
            reparsed_outcome_labels += 1

        failure_payload = {
            "task_text": _as_str(row.get("task_text")),
            "node_contract_text": _as_str(row.get("node_contract_text")),
            "outcome_label": outcome_label,
            "node_output_text": builder.truncate_text(evidence.get("node_output_text", ""), 2500),
            "review_text": builder.truncate_text(evidence.get("review_text", ""), 2500),
            "logs_text": builder.truncate_text(evidence.get("logs_text", ""), 2500),
            "exception_taxonomy_signature": exception_taxonomy_signature,
        }

        failure_reason_labels, attempts, failure_err = call_with_retry(
            builder=builder,
            cache=cache,
            model_signature=llm.model_signature,
            purpose="failure_reason_labels_multi_v2_taxonomy_aware",
            payload=failure_payload,
            fn=lambda: label_failure_reasons_multi(
                builder=builder,
                llm=llm,
                cache=cache,
                exception_taxonomy=exception_taxonomy,
                exception_taxonomy_signature=exception_taxonomy_signature,
                task_text=_as_str(row.get("task_text")),
                node_contract_text=_as_str(row.get("node_contract_text")),
                outcome_label=outcome_label,
                node_output_text=evidence.get("node_output_text", ""),
                review_text=evidence.get("review_text", ""),
                logs_text=evidence.get("logs_text", ""),
            ),
            max_retries=3,
        )
        if failure_reason_labels is None:
            errors.append({
                "qid": qid,
                "node_id": node_id,
                "stage": "failure_reason_labels",
                "attempts": attempts,
                "plan_path": _as_str(row.get("plan_path")),
                "trajectory_path": trajectory_path_str,
                "error": failure_err,
            })
            continue

        rec = NodeSignatureRecord(
            qid=qid,
            node_id=node_id,
            task_text=_as_str(row.get("task_text")),
            node_contract_text=_as_str(row.get("node_contract_text")),
            agent_name=_as_str(row.get("agent_name")),
            deps=_norm_list_str(row.get("deps")),
            dep_bucket=_as_str(row.get("dep_bucket")),
            task_label=_as_str(row.get("task_label")),
            contract_label=_as_str(row.get("contract_label")),
            outcome_label=outcome_label,
            failure_reason_labels=failure_reason_labels,
            plan_path=_as_str(row.get("plan_path")),
            trajectory_path=trajectory_path_str,
        )
        records.append(rec)
        updates.append(
            NodeOutcomeUpdateRecord(
                qid=qid,
                node_id=node_id,
                outcome_label=outcome_label,
                failure_reason_labels=failure_reason_labels,
            )
        )
        processed_keys.add(key)
        relabeled_from_existing += 1

    summary = {
        "existing_rows_total": existing_rows_total,
        "num_relabeled_from_existing": relabeled_from_existing,
        "num_existing_duplicates_skipped": existing_duplicate_skipped,
        "num_reused_outcome_labels_from_existing": reused_outcome_labels,
        "num_reparsed_outcome_labels_from_existing": reparsed_outcome_labels,
        "num_existing_evidence_extract_failures": evidence_extract_failures,
        "num_existing_errors": len(errors),
    }

    return records, updates, errors, summary, processed_keys


# -----------------------------------------------------------------------------
# Fallback full rebuild path
# -----------------------------------------------------------------------------

def build_node_signature_records(
    *,
    builder,
    labeler,
    llm,
    cache,
    exception_taxonomy,
    exception_taxonomy_signature: str,
    plan_dir: Path,
    trajectory_dir: Path,
    existing_index: Dict[Tuple[str, str], Dict[str, Any]],
    skip_keys: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[List[NodeSignatureRecord], List[NodeOutcomeUpdateRecord], List[Dict[str, Any]], Dict[str, Any]]:
    skip_keys = skip_keys or set()

    plan_map = builder.scan_plan_files(plan_dir)
    traj_map = builder.scan_trajectory_files(trajectory_dir)

    common_qids = sorted(set(plan_map) & set(traj_map))
    missing_plan = sorted(set(traj_map) - set(plan_map))
    missing_traj = sorted(set(plan_map) - set(traj_map))

    if not common_qids:
        raise RuntimeError("No common qids found between plan_dir and trajectory_dir.")

    records: List[NodeSignatureRecord] = []
    updates: List[NodeOutcomeUpdateRecord] = []
    errors: List[Dict[str, Any]] = []

    reused_task_contract_count = 0
    recomputed_task_count = 0
    recomputed_contract_count = 0
    reused_outcome_count = 0

    parsed_outcome_from_review_count = 0
    parsed_outcome_from_response_count = 0
    parsed_outcome_from_raw_status_count = 0
    llm_outcome_fallback_count = 0

    skipped_due_to_existing_cache = 0

    for qid in common_qids:
        plan_path = plan_map[qid]
        traj_path = traj_map[qid]

        try:
            plan_text = plan_path.read_text(encoding="utf-8")
            plan_doc = builder.extract_first_plan_json_object(plan_text)
            traj_doc = builder.read_json(traj_path)

            raw_aux_index = build_raw_trajectory_aux_index(traj_doc)

            plan_nodes = builder.extract_plan_nodes(plan_doc, qid=qid, source_path=plan_path)
            traj_nodes = builder.extract_trajectory_nodes(traj_doc, qid=qid, source_path=traj_path)
            aligned = builder.align_plan_and_trajectory(plan_nodes, traj_nodes)
        except Exception as e:
            errors.append({
                "qid": qid,
                "stage": "parse_or_align",
                "plan_path": str(plan_path),
                "trajectory_path": str(traj_path),
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        for pair in aligned:
            p = pair["plan"]
            t = pair["trajectory"]

            key = (qid, p.node_id)
            if key in skip_keys:
                skipped_due_to_existing_cache += 1
                continue

            existing_row = existing_index.get(key)

            trajectory_status_raw = t.raw_status if t is not None else ""
            node_output_text = t.node_output_text if t is not None else ""
            review_text_from_builder = t.review_text if t is not None else ""
            logs_text = t.logs_text if t is not None else ""

            task_number = node_id_to_task_number(p.node_id)
            raw_aux = raw_aux_index.get(task_number or "", {})
            review_text = _as_str(raw_aux.get("review_text")) or review_text_from_builder
            response_status = _as_str(raw_aux.get("response_status"))

            # Reuse task/contract labels if existing table has them
            if existing_row is not None and _norm_text(existing_row.get("task_label")) and _norm_text(existing_row.get("contract_label")):
                task_label = str(existing_row["task_label"])
                contract_label = str(existing_row["contract_label"])
                reused_task_contract_count += 1
            else:
                task_payload = {"task_text": p.task_text}
                task_label, task_attempts, task_err = call_with_retry(
                    builder=builder,
                    cache=cache,
                    model_signature=llm.model_signature,
                    purpose="task_label",
                    payload=task_payload,
                    fn=lambda: labeler.label_task(p.task_text),
                    max_retries=3,
                )
                if task_label is None:
                    errors.append({
                        "qid": qid,
                        "node_id": p.node_id,
                        "stage": "task_label",
                        "attempts": task_attempts,
                        "task_text": p.task_text,
                        "plan_path": str(plan_path),
                        "trajectory_path": str(traj_path),
                        "error": task_err,
                    })
                    continue
                recomputed_task_count += 1

                contract_payload = {"contract_text": p.node_contract_text}
                contract_label, contract_attempts, contract_err = call_with_retry(
                    builder=builder,
                    cache=cache,
                    model_signature=llm.model_signature,
                    purpose="contract_label",
                    payload=contract_payload,
                    fn=lambda: labeler.label_contract(p.node_contract_text),
                    max_retries=3,
                )
                if contract_label is None:
                    errors.append({
                        "qid": qid,
                        "node_id": p.node_id,
                        "stage": "contract_label",
                        "attempts": contract_attempts,
                        "node_contract_text": p.node_contract_text,
                        "plan_path": str(plan_path),
                        "trajectory_path": str(traj_path),
                        "error": contract_err,
                    })
                    continue
                recomputed_contract_count += 1

            evidence = {
                "review_text": review_text,
                "response_status": response_status,
                "node_output_text": node_output_text,
                "logs_text": logs_text,
                "trajectory_status_raw": trajectory_status_raw,
            }

            outcome_label, outcome_source, outcome_err = resolve_outcome_label(
                builder=builder,
                labeler=labeler,
                llm=llm,
                cache=cache,
                qid=qid,
                node_id=p.node_id,
                plan_path=str(plan_path),
                trajectory_path=str(traj_path),
                task_text=p.task_text,
                node_contract_text=p.node_contract_text,
                evidence=evidence,
                existing_outcome_label=_as_str(existing_row.get("outcome_label")) if existing_row else None,
            )
            if outcome_err is not None:
                errors.append(outcome_err)
                continue

            if outcome_source == "existing":
                reused_outcome_count += 1
            elif outcome_source == "review":
                parsed_outcome_from_review_count += 1
            elif outcome_source == "response_status":
                parsed_outcome_from_response_count += 1
            elif outcome_source == "raw_status":
                parsed_outcome_from_raw_status_count += 1
            elif outcome_source == "llm":
                llm_outcome_fallback_count += 1

            failure_payload = {
                "task_text": p.task_text,
                "node_contract_text": p.node_contract_text,
                "outcome_label": outcome_label,
                "node_output_text": builder.truncate_text(node_output_text, 2500),
                "review_text": builder.truncate_text(review_text, 2500),
                "logs_text": builder.truncate_text(logs_text, 2500),
                "exception_taxonomy_signature": exception_taxonomy_signature,
            }
            failure_reason_labels, failure_attempts, failure_err = call_with_retry(
                builder=builder,
                cache=cache,
                model_signature=llm.model_signature,
                purpose="failure_reason_labels_multi_v2_taxonomy_aware",
                payload=failure_payload,
                fn=lambda: label_failure_reasons_multi(
                    builder=builder,
                    llm=llm,
                    cache=cache,
                    exception_taxonomy=exception_taxonomy,
                    exception_taxonomy_signature=exception_taxonomy_signature,
                    task_text=p.task_text,
                    node_contract_text=p.node_contract_text,
                    outcome_label=outcome_label,
                    node_output_text=node_output_text,
                    review_text=review_text,
                    logs_text=logs_text,
                ),
                max_retries=3,
            )
            if failure_reason_labels is None:
                errors.append({
                    "qid": qid,
                    "node_id": p.node_id,
                    "stage": "failure_reason_labels",
                    "attempts": failure_attempts,
                    "plan_path": str(plan_path),
                    "trajectory_path": str(traj_path),
                    "error": failure_err,
                })
                continue

            rec = NodeSignatureRecord(
                qid=qid,
                node_id=p.node_id,
                task_text=p.task_text,
                node_contract_text=p.node_contract_text,
                agent_name=p.agent_name,
                deps=p.deps,
                dep_bucket=builder.normalize_dep_bucket(len(p.deps)),
                task_label=task_label,
                contract_label=contract_label,
                outcome_label=outcome_label,
                failure_reason_labels=failure_reason_labels,
                plan_path=str(plan_path),
                trajectory_path=str(traj_path),
            )
            records.append(rec)

            updates.append(
                NodeOutcomeUpdateRecord(
                    qid=qid,
                    node_id=p.node_id,
                    outcome_label=outcome_label,
                    failure_reason_labels=failure_reason_labels,
                )
            )

    summary = {
        "num_common_qids": len(common_qids),
        "missing_plan_qids": missing_plan,
        "missing_trajectory_qids": missing_traj,
        "num_records": len(records),
        "num_updates": len(updates),
        "num_errors": len(errors),
        "num_reused_task_contract_labels": reused_task_contract_count,
        "num_recomputed_task_labels": recomputed_task_count,
        "num_recomputed_contract_labels": recomputed_contract_count,
        "num_reused_outcome_labels": reused_outcome_count,
        "num_outcomes_parsed_from_reviews": parsed_outcome_from_review_count,
        "num_outcomes_parsed_from_response_status": parsed_outcome_from_response_count,
        "num_outcomes_parsed_from_raw_status": parsed_outcome_from_raw_status_count,
        "num_outcomes_llm_fallback": llm_outcome_fallback_count,
        "num_skipped_due_to_existing_cache_hit": skipped_due_to_existing_cache,
    }

    return records, updates, errors, summary


# -----------------------------------------------------------------------------
# Merge / sort helpers
# -----------------------------------------------------------------------------

def sort_records(records: List[NodeSignatureRecord]) -> List[NodeSignatureRecord]:
    def _key(r: NodeSignatureRecord):
        q = _as_str(r.qid)
        nid = _as_str(r.node_id)
        m = re.search(r"(\d+)", nid)
        node_num = int(m.group(1)) if m else 10**9
        return (q, node_num, nid)
    return sorted(records, key=_key)


def merge_summaries(existing_summary: Dict[str, Any], fallback_summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(existing_summary)
    for k, v in fallback_summary.items():
        if k not in out:
            out[k] = v
            continue
        if isinstance(out[k], int) and isinstance(v, int):
            out[k] += v
        elif isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or refresh a qid,node_id keyed node signature table. "
            "If --existing_node_signature_table is provided, the script first tries to reuse rows from that table "
            "and only recomputes failure_reason_labels. Rows that cannot be reused fall back to plan/trajectory rebuilding."
        )
    )
    parser.add_argument("--builder_script", required=True, help="Path to build_frequency_model.py")
    parser.add_argument("--plan_dir", required=True, help="Directory containing no_verify plan txt files.")
    parser.add_argument("--trajectory_dir", required=True, help="Directory containing no_verify trajectory json files.")
    parser.add_argument("--task_taxonomy_json", required=True, help="Path to merged_task_taxonomy JSON.")
    parser.add_argument("--contract_taxonomy_json", required=True, help="Path to merged_node_contract_taxonomy JSON.")
    parser.add_argument("--exception_taxonomy", required=True, help="Path to exception taxonomy (.json or .py).")

    parser.add_argument(
        "--existing_node_signature_table",
        default=None,
        help=(
            "Optional existing node_signature_table JSON. "
            "If provided, rows are reused as cache for qid/node/task/contract/outcome/paths, "
            "and only failure_reason_labels are refreshed when possible."
        ),
    )

    parser.add_argument("--output_json", required=True, help="Output full JSON array file.")
    parser.add_argument("--output_updates_json", default=None, help="Optional output JSON file containing only qid,node_id,outcome_label,failure_reason_labels.")
    parser.add_argument("--output_jsonl", default=None, help="Optional output JSONL file.")
    parser.add_argument("--output_errors", required=True, help="Output JSON file for skipped-node errors.")
    parser.add_argument("--output_summary", required=True, help="Output JSON file for summary.")
    parser.add_argument("--cache_json", default=None, help="Optional JSON cache path for LLM calls.")
    parser.add_argument("--model_id", type=int, default=20, help="watsonx model id.")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--logs_max_chars", type=int, default=2500)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    builder = load_module_from_path(Path(args.builder_script))

    task_taxonomy = builder.load_category_taxonomy_json(Path(args.task_taxonomy_json))
    contract_taxonomy = builder.load_category_taxonomy_json(Path(args.contract_taxonomy_json))

    # Keep the builder loader for compatibility with existing taxonomy files.
    # The downstream failure-label prompt adapts both old-style objects and new-style dict leaves.
    exception_taxonomy = builder.load_exception_taxonomy(Path(args.exception_taxonomy))
    exception_taxonomy_signature = build_exception_taxonomy_signature(exception_taxonomy)

    existing_table_path = Path(args.existing_node_signature_table) if args.existing_node_signature_table else None
    existing_rows = load_existing_node_signature_rows(existing_table_path)
    existing_index = load_existing_node_signature_index(existing_table_path)

    output_json = Path(args.output_json)
    output_errors = Path(args.output_errors)
    output_summary = Path(args.output_summary)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None
    output_updates_json = Path(args.output_updates_json) if args.output_updates_json else None

    cache_path = Path(args.cache_json) if args.cache_json else output_json.parent / "node_signature_cache.json"
    builder.ensure_dir(cache_path.parent)
    cache = builder.JsonCache(cache_path)

    llm = builder.WatsonxJsonLLM(
        model_id=args.model_id,
        temperature=args.llm_temperature,
        debug=args.debug,
    )
    labeler = builder.NodeLabeler(
        llm=llm,
        cache=cache,
        task_taxonomy=task_taxonomy,
        contract_taxonomy=contract_taxonomy,
        exception_taxonomy=exception_taxonomy,
        logs_max_chars=args.logs_max_chars,
    )

    # Pass 1: try to relabel directly from existing rows
    existing_records, existing_updates, existing_errors, existing_summary, processed_keys = relabel_from_existing_rows(
        builder=builder,
        labeler=labeler,
        llm=llm,
        cache=cache,
        exception_taxonomy=exception_taxonomy,
        exception_taxonomy_signature=exception_taxonomy_signature,
        existing_rows=existing_rows,
        logs_max_chars=args.logs_max_chars,
    )

    # Pass 2: fall back to full rebuild for rows that could not be reused
    fallback_records, fallback_updates, fallback_errors, fallback_summary = build_node_signature_records(
        builder=builder,
        labeler=labeler,
        llm=llm,
        cache=cache,
        exception_taxonomy=exception_taxonomy,
        exception_taxonomy_signature=exception_taxonomy_signature,
        plan_dir=Path(args.plan_dir),
        trajectory_dir=Path(args.trajectory_dir),
        existing_index=existing_index,
        skip_keys=processed_keys,
    )

    records = sort_records(existing_records + fallback_records)
    updates = sort_records([NodeSignatureRecord(
        qid=x.qid,
        node_id=x.node_id,
        task_text="",
        node_contract_text="",
        agent_name="",
        deps=[],
        dep_bucket="",
        task_label="",
        contract_label="",
        outcome_label=x.outcome_label,
        failure_reason_labels=x.failure_reason_labels,
        plan_path="",
        trajectory_path="",
    ) for x in (existing_updates + fallback_updates)])
    updates_out = [
        NodeOutcomeUpdateRecord(
            qid=x.qid,
            node_id=x.node_id,
            outcome_label=x.outcome_label,
            failure_reason_labels=x.failure_reason_labels,
        )
        for x in updates
    ]

    errors = existing_errors + fallback_errors
    summary = merge_summaries(existing_summary, fallback_summary)
    summary["exception_taxonomy_signature"] = exception_taxonomy_signature
    summary["num_records_final"] = len(records)
    summary["num_updates_final"] = len(updates_out)
    summary["num_errors_final"] = len(errors)

    builder.ensure_dir(output_json.parent)
    builder.dump_json([asdict(x) for x in records], output_json)

    if output_jsonl is not None:
        builder.ensure_dir(output_jsonl.parent)
        builder.dump_jsonl((asdict(x) for x in records), output_jsonl)

    if output_updates_json is not None:
        builder.ensure_dir(output_updates_json.parent)
        builder.dump_json([asdict(x) for x in updates_out], output_updates_json)

    builder.ensure_dir(output_errors.parent)
    builder.dump_json(errors, output_errors)

    builder.ensure_dir(output_summary.parent)
    builder.dump_json(summary, output_summary)

    print(f"Saved node signature table: {output_json}")
    if output_jsonl is not None:
        print(f"Saved node signature table JSONL: {output_jsonl}")
    if output_updates_json is not None:
        print(f"Saved outcome/failure updates: {output_updates_json}")
    print(f"Saved errors: {output_errors}")
    print(f"Saved summary: {output_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()