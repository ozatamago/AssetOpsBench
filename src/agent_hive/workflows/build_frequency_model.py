#!/usr/bin/env python3
"""
Build a frequency model from plan and trajectory folders using LLM-based discretization.

What this script does
---------------------
1. Scan plan/trajectory folders and match files by qid.
2. Parse plan nodes and trajectory node records.
3. Align plan nodes with trajectory nodes.
4. Use an LLM to assign:
   - unified task label
   - unified node-contract label
   - planning-time exception label X
   - outcome label Y in {A, P, N, E}
   - failure-reason label Z from the exception taxonomy (for P/N/E)
5. Build frequency counts and conditional probabilities:
   - counts over (S, Y, Z)
   - p_data(Y | S, Z)
6. Optionally compute planner-side risk from a uniform prior over Z and a user-provided weight matrix.

Notes
-----
- The script is intentionally adapter-heavy because plan/trajectory JSON schemas often vary.
- LLM calls are cached on disk.
- This version uses reactxen.utils.model_inference.watsonx_llm(prompt, model_id=...).
- JSON responses are extracted with retry and JSON-repair logic similar to the taxonomy builder script.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

OUTCOME_LABELS = ["A", "P", "N", "E"]
NEGATIVE_OUTCOME_LABELS = {"P", "N", "E"}
DEFAULT_MODEL_ID = 20
LLM_MAX_RETRIES = 3
LLM_RETRY_SLEEP = 2.0


# Shared upper taxonomy used by the frequency model.
UNIFIED_UPPER_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "failure_mode": {
        "definition": "Failure-mode enumeration, validation, failure inference from symptoms, or failure-behavior characterization.",
        "examples": [
            "List all failure modes of a power transformer.",
            "Identify the likely failure when a measurement drops.",
            "Describe temporal behavior under a failure condition.",
        ],
    },
    "sensor_identification": {
        "definition": "Identify, rank, or assess sensors relevant to a monitoring or diagnosis objective.",
        "examples": [
            "Identify relevant sensors for an air leak failure mode.",
            "Prioritize sensors by detection effectiveness.",
            "Estimate detection reliability of selected sensors.",
        ],
    },
    "sensor_data_retrieval": {
        "definition": "Retrieve raw sensor or historical time-series data for a specified asset, variable, or time range.",
        "examples": [
            "Download compressor sensor data for the previous week.",
            "Retrieve historical data for equipment CWC04009 for June 2020.",
        ],
    },
    "data_analysis": {
        "definition": "Analyze previously obtained data to detect anomalies, infer failures, forecast risks, diagnose root causes, or support maintenance decisions.",
        "examples": [
            "Analyze the downloaded data to detect anomalies.",
            "Forecast the risk of failure within the next 7 days.",
            "Diagnose the root cause based on anomaly analysis.",
        ],
    },
    "work_order": {
        "definition": "Create, retrieve, prioritize, bundle, schedule, summarize, or assess maintenance work orders.",
        "examples": [
            "Generate a work order if the risk is high.",
            "Retrieve work orders for the asset in 2017.",
        ],
    },
    "model": {
        "definition": "Find, verify, train, fine-tune, configure, or run predictive models or model recipes.",
        "examples": [
            "Check whether a model with context length 1024 exists.",
            "Train the selected model using historical data.",
            "Generate a machine-learning recipe.",
        ],
    },
    "entity_or_metadata": {
        "definition": "List or retrieve descriptive entities, metadata, inventories, identifiers, sites, assets, or monitored metrics.",
        "examples": [
            "List available IoT sites.",
            "Identify the site where Chiller 6 is located.",
            "Download the metadata for Chiller 3.",
        ],
    },
    "event": {
        "definition": "Identify, group, summarize, classify, or generate rules/messages for alerts, anomalies, and other operational events.",
        "examples": [
            "Identify events for work order, alert, and anomaly.",
            "Generate rules to distinguish meaningful alerts.",
            "Provide a summary based on event groups.",
        ],
    },
    "data_handling": {
        "definition": "Store, merge, filter, sort, export, reshape, or otherwise prepare data; also includes explicit temporal-range calculation/specification tasks.",
        "examples": [
            "Merge the retrieved data into a single file.",
            "Calculate the start and end dates for last week.",
            "Return the downloaded data in a file.",
        ],
    },
}

UPPER_LABEL_ALIASES = {
    "maintenance_decision": "work_order",
    "maintenance_action": "work_order",
    "workorder": "work_order",
    "work order": "work_order",
    "sensor_retrieval": "sensor_data_retrieval",
    "metadata": "entity_or_metadata",
    "entity": "entity_or_metadata",
}

def normalize_upper_label(label: Any) -> str:
    s = normalize_whitespace(label)
    return UPPER_LABEL_ALIASES.get(s, s)

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class TaxonomyCategory:
    category_name: str
    definition: str
    inclusion_criteria: List[str] = field(default_factory=list)
    example_texts: List[str] = field(default_factory=list)


@dataclass
class PlanNodeRecord:
    qid: str
    node_id: str
    task_text: str
    node_contract_text: str
    agent_name: str
    deps: List[str]
    expected_exception_candidates: List[str]
    raw_node: Dict[str, Any]
    source_path: str


@dataclass
class TrajectoryNodeRecord:
    qid: str
    node_id: str
    task_text: str
    raw_status: str
    normalized_status: Optional[str]
    node_output_text: str
    review_text: str
    logs_text: str
    raw_record: Dict[str, Any]
    source_path: str


@dataclass
class MergedNodeExample:
    qid: str
    node_id: str
    task_text: str
    node_contract_text: str
    agent_name: str
    deps: List[str]
    dep_bucket: str
    expected_exception_candidates: List[str]
    planning_exception_labels: List[str]
    task_label: str
    contract_label: str
    outcome_label: str
    failure_reason_label: Optional[str]
    trajectory_status_raw: str
    node_output_text: str
    review_text: str
    logs_text: str
    plan_path: str
    trajectory_path: str


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()



def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\u0000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s



def truncate_text(text: str, max_chars: int = 3000) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + " ...[TRUNCATED]"



def extract_qid_from_filename(path: Path) -> str:
    name = path.name
    m = re.search(r"Q[_-]?(\d+)", name)
    if m:
        return f"Q{m.group(1)}"
    return path.stem



def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def dump_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def dump_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def first_present(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default



def walk_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_dicts(item)



def normalize_dep_bucket(dep_count: int) -> str:
    if dep_count <= 0:
        return "0"
    if dep_count == 1:
        return "1"
    return "2_plus"



def normalize_outcome_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = normalize_whitespace(value).lower()
    if not s:
        return None

    exact_map = {
        "a": "A",
        "p": "P",
        "n": "N",
        "e": "E",
        "accomplished": "A",
        "pass": "A",
        "passed": "A",
        "success": "A",
        "successful": "A",
        "completed": "A",
        "done": "A",
        "partial": "P",
        "partially_accomplished": "P",
        "incomplete": "P",
        "partially complete": "P",
        "not accomplished": "N",
        "not_accomplished": "N",
        "failed": "N",
        "failure": "N",
        "fail": "N",
        "error": "E",
        "exception": "E",
        "runtime error": "E",
    }
    if s in exact_map:
        return exact_map[s]

    if any(tok in s for tok in ["runtime error", "exception", "traceback", "tool invocation failed", "error"]):
        return "E"
    if any(tok in s for tok in ["partial", "incomplete", "partially"]):
        return "P"
    if any(tok in s for tok in ["accomplished", "pass", "success", "completed", "done"]):
        return "A"
    if any(tok in s for tok in ["not accomplished", "failed", "failure"]):
        return "N"
    return None



def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try fenced JSON.
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # Find the first balanced JSON object.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

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
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    return json.loads(candidate)
    raise ValueError("Unable to extract balanced JSON object from LLM response.")



def string_or_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    try:
        return normalize_whitespace(json.dumps(value, ensure_ascii=False))
    except Exception:
        return normalize_whitespace(str(value))


# -----------------------------------------------------------------------------
# Taxonomy loading
# -----------------------------------------------------------------------------


def load_category_taxonomy_json(path: Path) -> List[TaxonomyCategory]:
    raw = read_json(path)
    categories = raw.get("categories", []) if isinstance(raw, dict) else raw
    result: List[TaxonomyCategory] = []
    for item in categories:
        if not isinstance(item, dict):
            continue
        result.append(
            TaxonomyCategory(
                category_name=str(item.get("category_name", "")).strip(),
                definition=normalize_whitespace(item.get("definition", "")),
                inclusion_criteria=[normalize_whitespace(x) for x in item.get("inclusion_criteria", []) if normalize_whitespace(x)],
                example_texts=[normalize_whitespace(x) for x in item.get("example_texts", []) if normalize_whitespace(x)],
            )
        )
    return result



def load_exception_taxonomy(path: Path) -> List[TaxonomyCategory]:
    if path.suffix.lower() == ".json":
        raw = read_json(path)
        if isinstance(raw, dict) and "categories" in raw:
            categories = raw["categories"]
        elif isinstance(raw, list):
            categories = raw
        else:
            raise ValueError(f"Unexpected JSON exception taxonomy format: {path}")
        result: List[TaxonomyCategory] = []
        for item in categories:
            result.append(
                TaxonomyCategory(
                    category_name=str(item["label"]).strip() if "label" in item else str(item["category_name"]).strip(),
                    definition=normalize_whitespace(item.get("description") or item.get("definition", "")),
                    inclusion_criteria=[normalize_whitespace(x) for x in item.get("representative_signals", []) if normalize_whitespace(x)],
                    example_texts=[],
                )
            )
        return result

    # Support a Python file/snippet containing: exception_taxonomy = [...]
    text = path.read_text(encoding="utf-8")
    m = re.search(r"exception_taxonomy\s*=\s*(\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find 'exception_taxonomy = [...]' in: {path}")
    payload = ast.literal_eval(m.group(1))
    result = []
    for item in payload:
        result.append(
            TaxonomyCategory(
                category_name=str(item["label"]).strip(),
                definition=normalize_whitespace(item.get("description", "")),
                inclusion_criteria=[normalize_whitespace(x) for x in item.get("representative_signals", []) if normalize_whitespace(x)],
                example_texts=[],
            )
        )
    return result


# -----------------------------------------------------------------------------
# Folder scanning
# -----------------------------------------------------------------------------


def scan_plan_files(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*_plan.txt"))
    mapping: Dict[str, Path] = {}
    for path in files:
        qid = extract_qid_from_filename(path)
        if qid in mapping:
            raise ValueError(f"Duplicate qid '{qid}' found in {folder}: {mapping[qid].name}, {path.name}")
        mapping[qid] = path
    return mapping


def scan_trajectory_files(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*_trajectory.json"))
    if not files:
        files = sorted(folder.glob("*.json"))

    mapping: Dict[str, Path] = {}
    for path in files:
        qid = extract_qid_from_filename(path)
        if qid in mapping:
            raise ValueError(f"Duplicate qid '{qid}' found in {folder}: {mapping[qid].name}, {path.name}")
        mapping[qid] = path
    return mapping


# -----------------------------------------------------------------------------
# Plan parsing
# -----------------------------------------------------------------------------


def extract_expected_exception_labels(raw_node: Dict[str, Any]) -> List[str]:
    raw = first_present(raw_node, ["expected_exception", "expected_exceptions", "possible_exceptions"], default=[])
    labels: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                labels.append(normalize_whitespace(item))
            elif isinstance(item, dict):
                label = normalize_whitespace(item.get("label") or item.get("name"))
                if label:
                    labels.append(label)
    elif isinstance(raw, dict):
        label = normalize_whitespace(raw.get("label") or raw.get("name"))
        if label:
            labels.append(label)
    return sorted({x for x in labels if x})



def extract_plan_nodes(plan_doc: Any, qid: str, source_path: Path) -> List[PlanNodeRecord]:
    container: List[Dict[str, Any]] = []

    if isinstance(plan_doc, dict):
        if isinstance(plan_doc.get("nodes"), list):
            container = [x for x in plan_doc["nodes"] if isinstance(x, dict)]
        elif isinstance(plan_doc.get("tasks"), list):
            container = [x for x in plan_doc["tasks"] if isinstance(x, dict)]
        else:
            # Fallback: recursively collect dicts that look like plan nodes.
            for d in walk_dicts(plan_doc):
                if any(k in d for k in ["node_id", "id"]) and any(k in d for k in ["task", "node_contract", "deps", "agent"]):
                    container.append(d)
    elif isinstance(plan_doc, list):
        container = [x for x in plan_doc if isinstance(x, dict)]

    nodes: List[PlanNodeRecord] = []
    seen: set[str] = set()
    for raw_node in container:
        node_id = normalize_whitespace(first_present(raw_node, ["node_id", "id"]))
        if not node_id:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)

        task_text = normalize_whitespace(first_present(raw_node, ["task", "description", "instruction"], default=""))
        node_contract_text = normalize_whitespace(first_present(raw_node, ["node_contract", "contract", "done_when", "success_condition"], default=""))
        agent_name = normalize_whitespace(first_present(raw_node, ["agent", "agent_name", "tool_name"], default=""))
        deps_raw = first_present(raw_node, ["deps", "dependencies", "parents"], default=[])
        if not isinstance(deps_raw, list):
            deps_raw = []
        deps = [normalize_whitespace(x) for x in deps_raw if normalize_whitespace(x)]
        expected_exception_candidates = extract_expected_exception_labels(raw_node)

        nodes.append(
            PlanNodeRecord(
                qid=qid,
                node_id=node_id,
                task_text=task_text,
                node_contract_text=node_contract_text,
                agent_name=agent_name,
                deps=deps,
                expected_exception_candidates=expected_exception_candidates,
                raw_node=copy.deepcopy(raw_node),
                source_path=str(source_path),
            )
        )

    return nodes


# -----------------------------------------------------------------------------
# Trajectory parsing
# -----------------------------------------------------------------------------


def _candidate_trajectory_containers(doc: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if isinstance(doc, dict):
        for key in ["history", "trajectory", "events", "steps", "records", "node_records", "messages"]:
            value = doc.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        candidates.append(item)
        # Fallback recursive scan.
        if not candidates:
            for d in walk_dicts(doc):
                has_id = any(k in d for k in ["node_id", "id", "step_id"])
                has_signal = any(k in d for k in ["normalized_status", "status", "node_output", "output", "review_result", "logs", "observation"])
                if has_id and has_signal:
                    candidates.append(d)
    elif isinstance(doc, list):
        for item in doc:
            if isinstance(item, dict):
                candidates.append(item)
    return candidates



def extract_trajectory_nodes(traj_doc: Any, qid: str, source_path: Path) -> List[TrajectoryNodeRecord]:
    container = _candidate_trajectory_containers(traj_doc)
    records: List[TrajectoryNodeRecord] = []
    seen: set[Tuple[str, str]] = set()

    for raw in container:
        node_id = normalize_whitespace(first_present(raw, ["node_id", "id", "step_id"]))
        if not node_id:
            continue

        raw_status = normalize_whitespace(first_present(raw, ["normalized_status", "status", "decision"], default=""))
        review_obj = first_present(raw, ["review_result", "review", "judge_result"], default={})
        review_text = string_or_json(review_obj)
        node_output_text = string_or_json(first_present(raw, ["node_output", "output", "final_answer", "answer", "observation"], default=""))
        logs_text = string_or_json(first_present(raw, ["logs", "log", "trace", "trajectory"], default=""))
        task_text = normalize_whitespace(first_present(raw, ["task", "task_description", "user_input"], default=""))

        normalized_status = normalize_outcome_label(raw_status)
        if normalized_status is None and isinstance(review_obj, dict):
            for key in ["label", "status", "normalized_status", "outcome"]:
                normalized_status = normalize_outcome_label(review_obj.get(key))
                if normalized_status is not None:
                    break

        dedupe_key = (qid, node_id)
        if dedupe_key in seen:
            # Prefer richer record if duplicate exists.
            continue
        seen.add(dedupe_key)

        records.append(
            TrajectoryNodeRecord(
                qid=qid,
                node_id=node_id,
                task_text=task_text,
                raw_status=raw_status,
                normalized_status=normalized_status,
                node_output_text=node_output_text,
                review_text=review_text,
                logs_text=logs_text,
                raw_record=copy.deepcopy(raw),
                source_path=str(source_path),
            )
        )

    return records


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------


def align_plan_and_trajectory(plan_nodes: List[PlanNodeRecord], traj_nodes: List[TrajectoryNodeRecord]) -> List[Dict[str, Any]]:
    traj_by_id = {x.node_id: x for x in traj_nodes}
    aligned: List[Dict[str, Any]] = []

    for p in plan_nodes:
        t = traj_by_id.get(p.node_id)
        aligned.append(
            {
                "plan": p,
                "trajectory": t,
            }
        )

    return aligned


# -----------------------------------------------------------------------------
# LLM cache
# -----------------------------------------------------------------------------


class JsonCache:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        if self.path is not None and self.path.exists():
            try:
                self.data = read_json(self.path)
            except Exception:
                self.data = {}

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        if self.path is not None:
            ensure_dir(self.path.parent)
            dump_json(self.data, self.path)


# -----------------------------------------------------------------------------
# watsonx LLM client
# -----------------------------------------------------------------------------


def raw_watsonx_generate(prompt: str, model_id: int = DEFAULT_MODEL_ID) -> str:
    try:
        from reactxen.utils.model_inference import watsonx_llm
    except Exception as e:
        raise RuntimeError(
            "Could not import watsonx_llm from reactxen.utils.model_inference. "
            f"Import error: {type(e).__name__}: {e}"
        )

    ret = watsonx_llm(prompt, model_id=model_id)

    if isinstance(ret, str):
        return normalize_whitespace(ret)

    if isinstance(ret, dict):
        if ret.get("generated_text") is not None:
            return normalize_whitespace(ret["generated_text"])
        results = ret.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and first.get("generated_text") is not None:
                return normalize_whitespace(first["generated_text"])

    raise RuntimeError(
        "watsonx_llm returned an unsupported response shape: "
        f"{type(ret).__name__} -> {repr(ret)[:500]}"
    )


class WatsonxJsonLLM:
    def __init__(
        self,
        model_id: int = DEFAULT_MODEL_ID,
        temperature: float = 0.0,
        max_retries: int = LLM_MAX_RETRIES,
        retry_sleep: float = LLM_RETRY_SLEEP,
        debug: bool = False,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.debug = debug
        self.model_signature = f"watsonx_model_id:{model_id}"

    def _repair_json(self, bad_text: str) -> Dict[str, Any]:
        repair_prompt = f"""
Convert the following text into ONE valid JSON object.

Rules:
- Output JSON only.
- No markdown.
- No code fences.
- Use double quotes for all property names and all string values.
- Do not add any explanation before or after the JSON.
- Preserve all recoverable fields.
- If a value is unclear, preserve it as a string.
- Return exactly one JSON object.

Text:
{bad_text}
""".strip()
        repaired = raw_watsonx_generate(repair_prompt, model_id=self.model_id)
        return extract_json_object(repaired)

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        prompt = f"""
{system_prompt}

{user_prompt}
""".strip()

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = raw_watsonx_generate(prompt, model_id=self.model_id)
                if self.debug:
                    print(f"[debug][watsonx] attempt={attempt} raw head={repr(raw[:300])}", flush=True)
                if not raw:
                    raise ValueError("raw_watsonx_generate returned empty text.")
                try:
                    return extract_json_object(raw)
                except Exception:
                    return self._repair_json(raw)
            except Exception as exc:
                last_error = exc
                if self.debug:
                    print(f"[debug][watsonx] attempt={attempt} error={type(exc).__name__}: {exc}", flush=True)
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
        if last_error is None:
            raise RuntimeError("watsonx json generation failed without an explicit exception.")
        raise RuntimeError(f"watsonx json generation failed after retries: {last_error}")


# -----------------------------------------------------------------------------
# Prompt formatting
# -----------------------------------------------------------------------------


def format_upper_taxonomy_for_prompt() -> str:
    rows = []
    for name, meta in UNIFIED_UPPER_TAXONOMY.items():
        rows.append(
            {
                "label": name,
                "definition": meta["definition"],
                "examples": meta["examples"],
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)



def format_categories_for_prompt(categories: Sequence[TaxonomyCategory]) -> str:
    rows = []
    for c in categories:
        rows.append(
            {
                "label": c.category_name,
                "definition": c.definition,
                "criteria": c.inclusion_criteria,
                "examples": c.example_texts,
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# LLM labelers
# -----------------------------------------------------------------------------


class NodeLabeler:
    def __init__(
        self,
        llm: WatsonxJsonLLM,
        cache: JsonCache,
        task_taxonomy: Sequence[TaxonomyCategory],
        contract_taxonomy: Sequence[TaxonomyCategory],
        exception_taxonomy: Sequence[TaxonomyCategory],
        logs_max_chars: int = 2500,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.task_taxonomy = list(task_taxonomy)
        self.contract_taxonomy = list(contract_taxonomy)
        self.exception_taxonomy = list(exception_taxonomy)
        self.logs_max_chars = logs_max_chars
        self.exception_labels = [x.category_name for x in self.exception_taxonomy]

    def _cached_json_call(self, purpose: str, payload: Dict[str, Any], system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        key = stable_hash({"purpose": purpose, "payload": payload, "model": self.llm.model_signature})
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        self.cache.set(key, result)
        return result
    
    

    def label_task(self, task_text: str) -> str:
        payload = {"task_text": task_text}
        system_prompt = (
            "You are a precise taxonomy classifier. "
            "Return JSON only. Never return explanations outside JSON."
        )
        user_prompt = f"""
    Classify the following task text into exactly one label from the unified upper taxonomy.

    Unified upper taxonomy:
    {format_upper_taxonomy_for_prompt()}

    Optional source task taxonomy context:
    {format_categories_for_prompt(self.task_taxonomy)}

    Task text:
    {task_text}

    Return exactly one JSON object:
    {{
    "label": "<one unified upper taxonomy label>",
    "rationale": "<brief explanation>"
    }}
    """.strip()

        obj = self._cached_json_call("task_label", payload, system_prompt, user_prompt)
        label = normalize_upper_label(obj.get("label"))
        if label not in UNIFIED_UPPER_TAXONOMY:
            raise ValueError(f"Invalid task label from LLM: {label}")
        return label

    def label_contract(self, contract_text: str) -> str:
        payload = {"contract_text": contract_text}
        system_prompt = (
            "You are a precise taxonomy classifier. "
            "Return JSON only. Never return explanations outside JSON."
        )
        user_prompt = f"""
    Classify the following node-contract text into exactly one label from the unified upper taxonomy.

    Unified upper taxonomy:
    {format_upper_taxonomy_for_prompt()}

    Optional source node-contract taxonomy context:
    {format_categories_for_prompt(self.contract_taxonomy)}

    Node-contract text:
    {contract_text}

    Return exactly one JSON object:
    {{
    "label": "<one unified upper taxonomy label>",
    "rationale": "<brief explanation>"
    }}
    """.strip()

        obj = self._cached_json_call("contract_label", payload, system_prompt, user_prompt)
        label = normalize_upper_label(obj.get("label"))
        if label not in UNIFIED_UPPER_TAXONOMY:
            raise ValueError(f"Invalid contract label from LLM: {label}")
        return label
    
    def label_planning_exceptions(
        self,
        task_text: str,
        node_contract_text: str,
        outcome_label: str,
        node_output_text: str,
        review_text: str,
        logs_text: str,
    ) -> List[str]:
        allowed_rows = [
            {
                "label": c.category_name,
                "definition": c.definition,
                "representative_signals": c.inclusion_criteria,
            }
            for c in self.exception_taxonomy
        ]

        payload = {
            "task_text": task_text,
            "node_contract_text": node_contract_text,
            "outcome_label": outcome_label,
            "node_output_text": truncate_text(node_output_text, 2500),
            "review_text": truncate_text(review_text, 2500),
            "logs_text": truncate_text(logs_text, self.logs_max_chars),
        }

        system_prompt = (
            "You are a precise exception-label classifier. "
            "Given the node context and execution evidence, assign all exception labels "
            "from the provided taxonomy that clearly apply. "
            "Return JSON only."
        )

        user_prompt = f"""
    Assign zero or more applicable exception labels for this node.

    Exception taxonomy:
    {json.dumps(allowed_rows, ensure_ascii=False, indent=2)}

    Rules:
    - You may return multiple labels.
    - Return only labels that are clearly supported by the evidence.
    - If no label clearly applies, return an empty list.
    - Do not invent new labels.

    Node context:
    {json.dumps(payload, ensure_ascii=False, indent=2)}

    Return exactly one JSON object:
    {{
    "labels": ["<label1>", "<label2>"],
    "rationale": "<brief explanation>"
    }}
    """.strip()

        obj = self._cached_json_call(
            "planning_exception_labels",
            payload,
            system_prompt,
            user_prompt,
        )

        raw_labels = obj.get("labels", [])
        if not isinstance(raw_labels, list):
            raise ValueError(f"Invalid planning exception labels from LLM: {raw_labels}")

        allowed = set(self.exception_labels)
        labels: List[str] = []
        for x in raw_labels:
            label = normalize_whitespace(x)
            if not label:
                continue
            if label not in allowed:
                raise ValueError(f"Invalid planning exception label from LLM: {label}")
            if label not in labels:
                labels.append(label)

        return labels

    def label_outcome(self, merged_context: Dict[str, Any]) -> str:
        existing = normalize_outcome_label(merged_context.get("trajectory_status_raw"))
        if existing is not None:
            return existing
        payload = {
            "task_text": merged_context.get("task_text", ""),
            "node_contract_text": merged_context.get("node_contract_text", ""),
            "node_output_text": truncate_text(merged_context.get("node_output_text", ""), 2500),
            "review_text": truncate_text(merged_context.get("review_text", ""), 2500),
            "logs_text": truncate_text(merged_context.get("logs_text", ""), self.logs_max_chars),
        }
        system_prompt = (
            "You are an outcome judge. "
            "Classify execution result into exactly one of A, P, N, E where "
            "A=accomplished, P=partial, N=not accomplished, E=error. "
            "Return JSON only."
        )
        user_prompt = f"""
Judge the outcome of this executed node.

Definitions:
- A (accomplished): the local objective is satisfied.
- P (partial): partially useful result exists, but some required elements are missing or insufficient.
- N (not accomplished): the intended local objective was not achieved.
- E (error): runtime/tool/format/operational failure prevented valid completion.

Node context:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "label": "A|P|N|E",
  "rationale": "<brief explanation>"
}}
""".strip()
        obj = self._cached_json_call("outcome_label", payload, system_prompt, user_prompt)
        label = normalize_whitespace(obj.get("label"))
        if label not in OUTCOME_LABELS:
            raise ValueError(f"Invalid outcome label from LLM: {label}")
        return label

    def label_failure_reason(
        self,
        task_text: str,
        node_contract_text: str,
        expected_exception_candidates: Sequence[str],
        outcome_label: str,
        node_output_text: str,
        review_text: str,
        logs_text: str,
    ) -> Optional[str]:
        if outcome_label not in NEGATIVE_OUTCOME_LABELS:
            return None

        allowed_labels = [x for x in expected_exception_candidates if x] or self.exception_labels
        if len(allowed_labels) == 1:
            return allowed_labels[0]

        allowed_rows = [
            {
                "label": c.category_name,
                "definition": c.definition,
                "representative_signals": c.inclusion_criteria,
            }
            for c in self.exception_taxonomy
            if c.category_name in set(allowed_labels)
        ]

        payload = {
            "task_text": task_text,
            "node_contract_text": node_contract_text,
            "expected_exception_candidates": list(expected_exception_candidates),
            "allowed_labels": list(allowed_labels),
            "outcome_label": outcome_label,
            "node_output_text": truncate_text(node_output_text, 2500),
            "review_text": truncate_text(review_text, 2500),
            "logs_text": truncate_text(logs_text, self.logs_max_chars),
        }

        system_prompt = (
            "You are a precise failure-reason classifier. "
            "Given an executed node with a negative outcome, you must assign exactly one dominant exception label. "
            "Choose only from the allowed labels. Return JSON only."
        )
        user_prompt = f"""
Assign the dominant failure-reason label for this node.

Allowed exception labels:
{json.dumps(allowed_rows, ensure_ascii=False, indent=2)}

Important rules:
- You must choose exactly one label.
- Prefer a label from expected_exception_candidates when they are provided.
- Focus on the dominant reason for the negative outcome.
- Do not invent new labels.

Executed node context:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "label": "<one allowed exception label>",
  "rationale": "<brief explanation>"
}}
""".strip()
        obj = self._cached_json_call("failure_reason_label", payload, system_prompt, user_prompt)
        label = normalize_whitespace(obj.get("label"))
        if label not in set(allowed_labels):
            raise ValueError(f"Invalid failure reason label from LLM: {label} not in {allowed_labels}")
        return label


# -----------------------------------------------------------------------------
# Dataset building
# -----------------------------------------------------------------------------

def iter_fenced_blocks(text: str) -> List[str]:
    text = normalize_whitespace(text)
    blocks = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I):
        block = normalize_whitespace(m.group(1))
        if block:
            blocks.append(block)
    return blocks


def iter_balanced_object_candidates(text: str) -> List[str]:
    text = normalize_whitespace(text)
    candidates = []

    for start in range(len(text)):
        if text[start] != "{":
            continue

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
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = normalize_whitespace(text[start:i + 1])
                    if candidate:
                        candidates.append(candidate)
                    break

    return sorted(set(candidates), key=len, reverse=True)


def extract_first_plan_json_object(text: str) -> Dict[str, Any]:
    text = normalize_whitespace(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
            return obj
    except Exception:
        pass

    for block in iter_fenced_blocks(text):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
                return obj
        except Exception:
            pass

    for candidate in iter_balanced_object_candidates(text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
                return obj
        except Exception:
            pass

    raise ValueError("Failed to parse plan JSON from plan txt.")

def build_merged_examples_for_qid(
    qid: str,
    plan_path: Path,
    traj_path: Path,
    labeler: NodeLabeler,
) -> List[MergedNodeExample]:
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_doc = extract_first_plan_json_object(plan_text)
    traj_doc = read_json(traj_path)

    plan_nodes = extract_plan_nodes(plan_doc, qid=qid, source_path=plan_path)
    traj_nodes = extract_trajectory_nodes(traj_doc, qid=qid, source_path=traj_path)
    aligned = align_plan_and_trajectory(plan_nodes, traj_nodes)

    examples: List[MergedNodeExample] = []
    for pair in aligned:
        p: PlanNodeRecord = pair["plan"]
        t: Optional[TrajectoryNodeRecord] = pair["trajectory"]

        trajectory_status_raw = t.raw_status if t is not None else ""
        node_output_text = t.node_output_text if t is not None else ""
        review_text = t.review_text if t is not None else ""
        logs_text = t.logs_text if t is not None else ""

        task_label = labeler.label_task(p.task_text)
        contract_label = labeler.label_contract(p.node_contract_text)

        merged_context = {
            "task_text": p.task_text,
            "node_contract_text": p.node_contract_text,
            "trajectory_status_raw": trajectory_status_raw,
            "node_output_text": node_output_text,
            "review_text": review_text,
            "logs_text": logs_text,
        }
        outcome_label = labeler.label_outcome(merged_context)

        planning_exception_labels = labeler.label_planning_exceptions(
            task_text=p.task_text,
            node_contract_text=p.node_contract_text,
            outcome_label=outcome_label,
            node_output_text=node_output_text,
            review_text=review_text,
            logs_text=logs_text,
        )

        failure_reason_label = labeler.label_failure_reason(
            task_text=p.task_text,
            node_contract_text=p.node_contract_text,
            expected_exception_candidates=planning_exception_labels,
            outcome_label=outcome_label,
            node_output_text=node_output_text,
            review_text=review_text,
            logs_text=logs_text,
        )

        examples.append(
            MergedNodeExample(
                qid=qid,
                node_id=p.node_id,
                task_text=p.task_text,
                node_contract_text=p.node_contract_text,
                agent_name=p.agent_name,
                deps=p.deps,
                dep_bucket=normalize_dep_bucket(len(p.deps)),
                expected_exception_candidates=p.expected_exception_candidates,
                planning_exception_labels=planning_exception_labels,
                task_label=task_label,
                contract_label=contract_label,
                outcome_label=outcome_label,
                failure_reason_label=failure_reason_label,
                trajectory_status_raw=trajectory_status_raw,
                node_output_text=node_output_text,
                review_text=review_text,
                logs_text=logs_text,
                plan_path=str(plan_path),
                trajectory_path=str(traj_path),
            )
        )
    return examples


# -----------------------------------------------------------------------------
# Frequency estimation
# -----------------------------------------------------------------------------


def canonicalize_exception_labels(labels: Sequence[str]) -> str:
    uniq = sorted({normalize_whitespace(x) for x in labels if normalize_whitespace(x)})
    if not uniq:
        return "none"
    return "+".join(uniq)

def make_signature_key(example: MergedNodeExample) -> str:
    x_key = canonicalize_exception_labels(example.planning_exception_labels)
    parts = [
        f"task={example.task_label}",
        f"contract={example.contract_label}",
        f"x={x_key}",
        f"agent={example.agent_name or 'unknown'}",
        f"deps={example.dep_bucket}",
    ]
    return "|".join(parts)



def build_frequency_model(
    examples: Sequence[MergedNodeExample],
    alpha: float,
    exception_labels_for_planner: Sequence[str],
    weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    # counts[(sig, z)][y] += 1 ; note: Z is only populated for negative examples.
    counts: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {y: 0.0 for y in OUTCOME_LABELS}))
    sig_meta: Dict[str, Dict[str, Any]] = {}

    for ex in examples:
        sig = make_signature_key(ex)
        sig_meta[sig] = {
            "task_label": ex.task_label,
            "contract_label": ex.contract_label,
            "planning_exception_labels": list(sorted(set(ex.planning_exception_labels))),
            "agent_name": ex.agent_name or "unknown",
            "dependency_bucket": ex.dep_bucket,
        }
        z = ex.failure_reason_label if ex.failure_reason_label else "none"
        counts[sig][z][ex.outcome_label] += 1.0

    conditional_probs: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    risk_table: Dict[str, float] = {}

    for sig, per_reason in counts.items():
        for z, outcome_counts in per_reason.items():
            total = sum(outcome_counts.values())
            denom = total + len(OUTCOME_LABELS) * alpha
            probs = {y: (outcome_counts[y] + alpha) / denom for y in OUTCOME_LABELS}
            conditional_probs[sig][z] = probs

        if weights is not None:
            k = len(exception_labels_for_planner)
            if k <= 0:
                raise ValueError("exception_labels_for_planner must be non-empty when computing risk.")
            risk = 0.0
            for z in exception_labels_for_planner:
                probs = conditional_probs[sig].get(z)
                if probs is None:
                    # Unseen (sig, z): use smoothing-only pseudo-row.
                    denom = len(OUTCOME_LABELS) * alpha
                    probs = {y: alpha / denom for y in OUTCOME_LABELS}
                row = weights.get(z, {})
                risk += (1.0 / k) * sum(float(row.get(y, 0.0)) * probs[y] for y in OUTCOME_LABELS)
            risk_table[sig] = risk

    return {
        "alpha": alpha,
        "outcome_labels": list(OUTCOME_LABELS),
        "exception_labels_for_planner": list(exception_labels_for_planner),
        "counts": counts,
        "conditional_probs": conditional_probs,
        "signature_metadata": sig_meta,
        "risk_table": risk_table,
    }


# -----------------------------------------------------------------------------
# Optional weights loader
# -----------------------------------------------------------------------------


def load_weights_json(path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    if path is None:
        return None
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("weights JSON must be a dict: {reason_label: {A:0, P:..., N:..., E:...}}")
    cleaned: Dict[str, Dict[str, float]] = {}
    for reason, row in raw.items():
        if not isinstance(row, dict):
            raise ValueError(f"weights row for '{reason}' must be a dict")
        cleaned[reason] = {y: float(row.get(y, 0.0)) for y in OUTCOME_LABELS}
    return cleaned


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frequency model from plan/trajectory folders using LLM labeling.")
    parser.add_argument("--plan_dir", required=True, help="Directory containing plan JSON files.")
    parser.add_argument("--trajectory_dir", required=True, help="Directory containing trajectory JSON files.")
    parser.add_argument("--task_taxonomy_json", required=True, help="Path to merged_task_taxonomy JSON.")
    parser.add_argument("--contract_taxonomy_json", required=True, help="Path to merged_node_contract_taxonomy JSON.")
    parser.add_argument("--exception_taxonomy", required=True, help="Path to exception taxonomy (.json or .py containing exception_taxonomy = [...]).")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Additive smoothing constant.")
    parser.add_argument("--weights_json", default=None, help="Optional planner weight matrix JSON.")
    parser.add_argument("--cache_json", default=None, help="Optional JSON cache path for LLM calls.")
    parser.add_argument("--model_id", type=int, default=DEFAULT_MODEL_ID, help="watsonx model_id passed to reactxen.utils.model_inference.watsonx_llm.")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--logs_max_chars", type=int, default=2500)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    plan_dir = Path(args.plan_dir)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    task_taxonomy = load_category_taxonomy_json(Path(args.task_taxonomy_json))
    contract_taxonomy = load_category_taxonomy_json(Path(args.contract_taxonomy_json))
    exception_taxonomy = load_exception_taxonomy(Path(args.exception_taxonomy))
    weights = load_weights_json(Path(args.weights_json)) if args.weights_json else None

    cache_path = Path(args.cache_json) if args.cache_json else output_dir / "llm_cache.json"
    cache = JsonCache(cache_path)
    llm = WatsonxJsonLLM(
        model_id=args.model_id,
        temperature=args.llm_temperature,
        debug=args.debug,
    )
    labeler = NodeLabeler(
        llm=llm,
        cache=cache,
        task_taxonomy=task_taxonomy,
        contract_taxonomy=contract_taxonomy,
        exception_taxonomy=exception_taxonomy,
        logs_max_chars=args.logs_max_chars,
    )

    plan_map = scan_plan_files(plan_dir)
    traj_map = scan_trajectory_files(trajectory_dir)
    common_qids = sorted(set(plan_map) & set(traj_map))
    missing_plan = sorted(set(traj_map) - set(plan_map))
    missing_traj = sorted(set(plan_map) - set(traj_map))

    if not common_qids:
        raise SystemExit("No common qids found between plan_dir and trajectory_dir.")

    examples: List[MergedNodeExample] = []
    errors: List[Dict[str, Any]] = []

    for idx, qid in enumerate(common_qids, start=1):
        try:
            batch = build_merged_examples_for_qid(
                qid=qid,
                plan_path=plan_map[qid],
                traj_path=traj_map[qid],
                labeler=labeler,
            )
            examples.extend(batch)
            print(f"[{idx}/{len(common_qids)}] processed {qid}: {len(batch)} nodes", file=sys.stderr)
        except Exception as exc:
            errors.append({"qid": qid, "error": repr(exc)})
            print(f"[{idx}/{len(common_qids)}] ERROR {qid}: {exc}", file=sys.stderr)

    example_rows = [asdict(x) for x in examples]
    dump_jsonl(example_rows, output_dir / "merged_node_examples.jsonl")
    dump_json(
        {
            "common_qids": common_qids,
            "missing_plan": missing_plan,
            "missing_trajectory": missing_traj,
            "n_examples": len(examples),
            "errors": errors,
        },
        output_dir / "run_summary.json",
    )

    planner_exception_labels = [x.category_name for x in exception_taxonomy]
    model = build_frequency_model(
        examples=examples,
        alpha=args.alpha,
        exception_labels_for_planner=planner_exception_labels,
        weights=weights,
    )

    # Convert defaultdicts to plain dicts for serialization.
    serializable_model = {
        "alpha": model["alpha"],
        "outcome_labels": model["outcome_labels"],
        "exception_labels_for_planner": model["exception_labels_for_planner"],
        "signature_metadata": model["signature_metadata"],
        "counts": {sig: dict(reason_map) for sig, reason_map in model["counts"].items()},
        "conditional_probs": {sig: dict(reason_map) for sig, reason_map in model["conditional_probs"].items()},
        "risk_table": model["risk_table"],
    }
    dump_json(serializable_model, output_dir / "frequency_model.json")

    print(f"Wrote {len(examples)} merged node examples to: {output_dir / 'merged_node_examples.jsonl'}")
    print(f"Wrote frequency model to: {output_dir / 'frequency_model.json'}")
    print(f"Wrote run summary to: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
