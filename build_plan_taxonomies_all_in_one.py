#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# Config
# ============================================================

DEFAULT_MODEL_ID = 20
DEFAULT_BATCH_SIZE = 10
DEFAULT_MERGE_GROUP_SIZE = 5
LLM_MAX_RETRIES = 3
LLM_RETRY_SLEEP = 2.0

PLAN_NODE_FIELDNAMES = [
    "source_file",
    "qid",
    "node_id",
    "agent",
    "task",
    "node_contract",
]

TEXT_ITEM_FIELDNAMES = [
    "item_id",
    "source_file",
    "qid",
    "node_id",
    "agent",
    "field_name",
    "text",
]

TEXT_TAXONOMY_FIELDNAMES = [
    "field_name",
    "taxonomy_id",
    "round_index",
    "group_index",
    "category_name",
    "definition",
    "inclusion_criteria",
    "example_item_ids",
    "example_texts",
]


# ============================================================
# Basic utils
# ============================================================

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u0000", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        safe_row: Dict[str, Any] = {}
        for k in fieldnames:
            v = row.get(k, "")
            if isinstance(v, (list, dict)):
                safe_row[k] = json.dumps(v, ensure_ascii=False)
            else:
                safe_row[k] = v
        writer.writerow(safe_row)


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def group_list(xs: List[Any], group_size: int) -> List[List[Any]]:
    return [xs[i:i + group_size] for i in range(0, len(xs), group_size)]


# ============================================================
# plan.txt scanning / JSON extraction
# ============================================================

def iter_plan_txt_files(plan_dir: Path) -> Iterable[Path]:
    for p in sorted(plan_dir.rglob("*_plan.txt")):
        if p.is_file():
            yield p


def parse_qid_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"_Q_(\d+)_plan\.txt$", path.name)
    if not m:
        return None
    return int(m.group(1))


def iter_fenced_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    blocks = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I):
        block = normalize_text(m.group(1))
        if block:
            blocks.append(block)
    return blocks


def iter_balanced_object_candidates(text: str) -> List[str]:
    text = normalize_text(text)
    candidates = []

    for start in range(len(text)):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        quote_char = None

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                    quote_char = None
                continue

            if ch in ['"', "'"]:
                in_string = True
                quote_char = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = normalize_text(text[start:i + 1])
                    if candidate:
                        candidates.append(candidate)
                    break

    return sorted(set(candidates), key=len, reverse=True)


def try_parse_json_strict(candidate: str) -> Dict[str, Any]:
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def try_parse_python_literal(candidate: str) -> Dict[str, Any]:
    obj = ast.literal_eval(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed Python literal is not a dict.")
    return obj


def looks_like_plan_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    return "nodes" in obj and isinstance(obj.get("nodes"), list)


def extract_first_plan_json_object(text: str, debug: bool = False) -> Dict[str, Any]:
    text = normalize_text(text)
    parse_errors = []

    try:
        obj = try_parse_json_strict(text)
        if looks_like_plan_schema(obj):
            return obj
    except Exception as e:
        parse_errors.append(f"strict_full: {type(e).__name__}: {e}")

    for block in iter_fenced_blocks(text):
        try:
            obj = try_parse_json_strict(block)
            if looks_like_plan_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"strict_fenced: {type(e).__name__}: {e}")

        try:
            obj = try_parse_python_literal(block)
            if looks_like_plan_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"py_fenced: {type(e).__name__}: {e}")

    for candidate in iter_balanced_object_candidates(text):
        try:
            obj = try_parse_json_strict(candidate)
            if looks_like_plan_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"strict_candidate: {type(e).__name__}: {e}")

        try:
            obj = try_parse_python_literal(candidate)
            if looks_like_plan_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"py_candidate: {type(e).__name__}: {e}")

    if debug:
        print("[debug][plan-json-extract] parse errors sample:", parse_errors[:10], flush=True)
        print("[debug][plan-json-extract] raw head:", repr(text[:500]), flush=True)

    raise ValueError("Failed to parse plan JSON from text.")


def iter_plan_nodes(plan_dir: Path, debug: bool = False) -> Iterable[Dict[str, Any]]:
    for file_path in iter_plan_txt_files(plan_dir):
        try:
            text = file_path.read_text(encoding="utf-8")
            plan_obj = extract_first_plan_json_object(text, debug=debug)
        except Exception as e:
            if debug:
                print(f"[debug][parse-fail] {file_path}: {type(e).__name__}: {e}", flush=True)
            continue

        qid = parse_qid_from_filename(file_path)
        nodes = plan_obj.get("nodes", [])
        if not isinstance(nodes, list):
            continue

        for node in nodes:
            if not isinstance(node, dict):
                continue

            yield {
                "source_file": str(file_path),
                "qid": qid,
                "node_id": normalize_text(node.get("id", "")),
                "agent": normalize_text(node.get("agent", "")),
                "task": normalize_text(node.get("task", "")),
                "node_contract": normalize_text(node.get("node_contract", "")),
            }


def build_task_item(node_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = normalize_text(node_row.get("task", ""))
    if not text:
        return None

    qid = node_row.get("qid")
    node_id = node_row.get("node_id", "")
    return {
        "item_id": f"Q{qid}_{node_id}_task" if qid is not None else f"{node_id}_task",
        "source_file": node_row.get("source_file", ""),
        "qid": qid,
        "node_id": node_id,
        "agent": node_row.get("agent", ""),
        "field_name": "task",
        "text": text,
    }


def build_node_contract_item(node_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = normalize_text(node_row.get("node_contract", ""))
    if not text:
        return None

    qid = node_row.get("qid")
    node_id = node_row.get("node_id", "")
    return {
        "item_id": f"Q{qid}_{node_id}_node_contract" if qid is not None else f"{node_id}_node_contract",
        "source_file": node_row.get("source_file", ""),
        "qid": qid,
        "node_id": node_id,
        "agent": node_row.get("agent", ""),
        "field_name": "node_contract",
        "text": text,
    }


# ============================================================
# watsonx adapter
# ============================================================

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
        return normalize_text(ret)

    if isinstance(ret, dict):
        if "generated_text" in ret and ret["generated_text"] is not None:
            return normalize_text(ret["generated_text"])

        results = ret.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and first.get("generated_text") is not None:
                return normalize_text(first["generated_text"])

    raise RuntimeError(
        "watsonx_llm returned an unsupported response shape: "
        f"{type(ret).__name__} -> {repr(ret)[:500]}"
    )


# ============================================================
# taxonomy JSON extraction
# ============================================================

def looks_like_taxonomy_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    return "taxonomy_name" in obj and "categories" in obj


def parse_best_taxonomy_object_without_repair(text: str, debug: bool = False) -> Dict[str, Any]:
    text = normalize_text(text)
    parse_errors = []

    try:
        obj = try_parse_json_strict(text)
        if looks_like_taxonomy_schema(obj):
            return obj
    except Exception as e:
        parse_errors.append(f"strict_full: {type(e).__name__}: {e}")

    for block in iter_fenced_blocks(text):
        try:
            obj = try_parse_json_strict(block)
            if looks_like_taxonomy_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"strict_fenced: {type(e).__name__}: {e}")

        try:
            obj = try_parse_python_literal(block)
            if looks_like_taxonomy_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"py_fenced: {type(e).__name__}: {e}")

    for candidate in iter_balanced_object_candidates(text):
        try:
            obj = try_parse_json_strict(candidate)
            if looks_like_taxonomy_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"strict_candidate: {type(e).__name__}: {e}")

        try:
            obj = try_parse_python_literal(candidate)
            if looks_like_taxonomy_schema(obj):
                return obj
        except Exception as e:
            parse_errors.append(f"py_candidate: {type(e).__name__}: {e}")

    if debug:
        print("[debug][taxonomy-json-extract] parse errors sample:", parse_errors[:10], flush=True)
        print("[debug][taxonomy-json-extract] raw head:", repr(text[:500]), flush=True)

    raise ValueError("Failed to parse taxonomy JSON from text.")


def coerce_to_taxonomy_json_via_repair_prompt(bad_text: str, model_id: int, debug: bool = False) -> Dict[str, Any]:
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

    repaired = raw_watsonx_generate(repair_prompt, model_id=model_id)
    repaired = normalize_text(repaired)

    if debug:
        print("[debug][json-repair] repaired raw head:", repr(repaired[:300]), flush=True)

    if not repaired:
        raise ValueError("Repair prompt returned empty text.")

    return parse_best_taxonomy_object_without_repair(repaired, debug=debug)


def extract_first_taxonomy_object(text: str, model_id: int, debug: bool = False) -> Dict[str, Any]:
    text = normalize_text(text)
    if not text:
        raise ValueError("Model output is empty.")

    try:
        return parse_best_taxonomy_object_without_repair(text, debug=debug)
    except Exception as e:
        if debug:
            print(f"[debug][taxonomy-json-extract] non-repair parse failed: {type(e).__name__}: {e}", flush=True)

    return coerce_to_taxonomy_json_via_repair_prompt(text, model_id=model_id, debug=debug)


def call_watsonx_json(prompt: str, model_id: int = DEFAULT_MODEL_ID, debug: bool = False) -> Dict[str, Any]:
    last_err = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            raw = raw_watsonx_generate(prompt, model_id=model_id)
            raw = normalize_text(raw)

            if debug:
                print(f"[debug][watsonx] attempt={attempt} raw head={repr(raw[:300])}", flush=True)

            if not raw:
                raise ValueError("raw_watsonx_generate returned empty text.")

            return extract_first_taxonomy_object(raw, model_id=model_id, debug=debug)

        except Exception as e:
            last_err = e
            if debug:
                print(f"[debug][watsonx] attempt={attempt} error={type(e).__name__}: {e}", flush=True)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_SLEEP)

    raise RuntimeError(f"watsonx json generation failed after retries: {last_err}")


# ============================================================
# taxonomy item preparation
# ============================================================

def deduplicate_text_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for item in items:
        text = normalize_text(item.get("text", ""))
        if not text:
            continue

        key = text.lower()

        if key not in grouped:
            grouped[key] = {
                "item_id": item["item_id"],
                "field_name": item.get("field_name", ""),
                "canonical_text": text,
                "count": 1,
                "member_item_ids": [item["item_id"]],
                "agents": [item.get("agent", "")] if item.get("agent", "") else [],
                "source_files": [item.get("source_file", "")] if item.get("source_file", "") else [],
                "qids": [item.get("qid")] if item.get("qid") is not None else [],
                "node_ids": [item.get("node_id", "")] if item.get("node_id", "") else [],
            }
        else:
            grouped[key]["count"] += 1
            grouped[key]["member_item_ids"].append(item["item_id"])

            agent = item.get("agent", "")
            if agent and agent not in grouped[key]["agents"]:
                grouped[key]["agents"].append(agent)

            src = item.get("source_file", "")
            if src and src not in grouped[key]["source_files"]:
                grouped[key]["source_files"].append(src)

            qid = item.get("qid")
            if qid is not None and qid not in grouped[key]["qids"]:
                grouped[key]["qids"].append(qid)

            node_id = item.get("node_id", "")
            if node_id and node_id not in grouped[key]["node_ids"]:
                grouped[key]["node_ids"].append(node_id)

    return list(grouped.values())


# ============================================================
# taxonomy prompts
# ============================================================

def build_text_taxonomy_batch_prompt(items: List[Dict[str, Any]], field_name: str) -> str:
    compact_items = []
    for x in items:
        compact_items.append({
            "item_id": x["item_id"],
            "text": x["canonical_text"],
            "count": x.get("count", 1),
            "member_item_ids": x.get("member_item_ids", [x["item_id"]]),
        })

    return f"""
You are building a reusable taxonomy for {field_name} strings.

Your job:
1. Group texts that express the same underlying type.
2. Merge lexical variants into the same category.
3. Create reusable, stable category names.
4. Do NOT cluster by superficial wording only. Focus on semantic role.

Return JSON only.
Return exactly one JSON object starting with "{{" and ending with "}}".
Do not use markdown.
Do not use code fences.
Do not add any text before or after the JSON.

Output schema:
{{
  "taxonomy_name": "{field_name}_taxonomy",
  "categories": [
    {{
      "category_name": "<short reusable category name>",
      "definition": "<1-2 sentence definition>",
      "inclusion_criteria": ["<criterion>", "..."],
      "example_item_ids": ["<item_id>", "..."],
      "example_texts": ["<text>", "..."]
    }}
  ]
}}

Items:
{json.dumps(compact_items, ensure_ascii=False, indent=2)}

Guidelines:
- Prefer fewer, broader, reusable categories.
- Merge near-duplicates and wording variants.
- Keep category names stable and interpretable.
- The output should be useful as a planner-side taxonomy.
""".strip()


def build_text_taxonomy_merge_prompt(taxonomies: List[Dict[str, Any]], field_name: str) -> str:
    return f"""
You are merging several partial taxonomies for {field_name} into one unified taxonomy.

Goals:
- Merge overlapping categories.
- Remove near-duplicates.
- Keep category names stable and reusable.
- Prefer broad but still meaningful categories.

Return JSON only.
Return exactly one JSON object starting with "{{" and ending with "}}".
Do not use markdown.
Do not use code fences.
Do not add any text before or after the JSON.

Output schema:
{{
  "taxonomy_name": "merged_{field_name}_taxonomy",
  "categories": [
    {{
      "category_name": "<short reusable category name>",
      "definition": "<1-2 sentence definition>",
      "inclusion_criteria": ["<criterion>", "..."],
      "example_item_ids": ["<item_id>", "..."],
      "example_texts": ["<text>", "..."]
    }}
  ]
}}

Input taxonomies:
{json.dumps(taxonomies, ensure_ascii=False, indent=2)}
""".strip()


# ============================================================
# local fallback merge
# ============================================================

def build_local_merged_text_taxonomy(
    taxonomies: List[Dict[str, Any]],
    field_name: str,
) -> Dict[str, Any]:
    category_map: Dict[str, Dict[str, Any]] = {}

    for tx in taxonomies:
        for cat in tx.get("categories", []):
            name = normalize_text(cat.get("category_name", "")) or "Unclassified"
            if name not in category_map:
                category_map[name] = {
                    "category_name": name,
                    "definition": normalize_text(cat.get("definition", "")),
                    "inclusion_criteria": [],
                    "example_item_ids": [],
                    "example_texts": [],
                }

            dst = category_map[name]

            for v in cat.get("inclusion_criteria", []) or []:
                v = normalize_text(v)
                if v and v not in dst["inclusion_criteria"]:
                    dst["inclusion_criteria"].append(v)

            for v in cat.get("example_item_ids", []) or []:
                v = normalize_text(v)
                if v and v not in dst["example_item_ids"]:
                    dst["example_item_ids"].append(v)

            for v in cat.get("example_texts", []) or []:
                v = normalize_text(v)
                if v and v not in dst["example_texts"]:
                    dst["example_texts"].append(v)

            if not dst["definition"] and normalize_text(cat.get("definition", "")):
                dst["definition"] = normalize_text(cat.get("definition", ""))

    return {
        "taxonomy_name": f"merged_{field_name}_taxonomy",
        "categories": list(category_map.values()),
    }


# ============================================================
# taxonomy batch / merge
# ============================================================

def build_taxonomy_for_batch(
    items: List[Dict[str, Any]],
    batch_index: int,
    model_id: int,
    output_dir: Path,
    field_name: str,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt = build_text_taxonomy_batch_prompt(items, field_name=field_name)
    taxonomy = call_watsonx_json(prompt, model_id=model_id, debug=debug)

    taxonomy["_meta"] = {
        "taxonomy_id": f"batch_{batch_index:04d}",
        "round_index": 0,
        "group_index": batch_index,
        "source_item_ids": [x["item_id"] for x in items],
    }

    out_path = output_dir / "taxonomy_batches" / f"{field_name}_taxonomy_batch_{batch_index:04d}.json"
    write_json(out_path, taxonomy)
    return taxonomy


def merge_taxonomy_group(
    taxonomies: List[Dict[str, Any]],
    round_index: int,
    group_index: int,
    model_id: int,
    output_dir: Path,
    field_name: str,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt = build_text_taxonomy_merge_prompt(taxonomies, field_name=field_name)

    try:
        merged = call_watsonx_json(prompt, model_id=model_id, debug=debug)
    except Exception as e:
        if debug:
            print(
                f"[debug][merge-fallback] field={field_name} round={round_index} "
                f"group={group_index} error={type(e).__name__}: {e}",
                flush=True,
            )
        merged = build_local_merged_text_taxonomy(taxonomies, field_name=field_name)

    src_ids = []
    for t in taxonomies:
        meta = t.get("_meta", {})
        tid = meta.get("taxonomy_id")
        if tid:
            src_ids.append(tid)

    merged["_meta"] = {
        "taxonomy_id": f"round_{round_index:02d}_group_{group_index:04d}",
        "round_index": round_index,
        "group_index": group_index,
        "source_taxonomy_ids": src_ids,
    }

    out_path = (
        output_dir
        / f"taxonomy_merge_round_{round_index:02d}"
        / f"{field_name}_taxonomy_round_{round_index:02d}_group_{group_index:04d}.json"
    )
    write_json(out_path, merged)
    return merged


def flatten_text_taxonomy_to_rows(taxonomy: Dict[str, Any], field_name: str) -> List[Dict[str, Any]]:
    rows = []
    meta = taxonomy.get("_meta", {})
    taxonomy_id = meta.get("taxonomy_id", "")
    round_index = meta.get("round_index", "")
    group_index = meta.get("group_index", "")

    for cat in taxonomy.get("categories", []):
        rows.append({
            "field_name": field_name,
            "taxonomy_id": taxonomy_id,
            "round_index": round_index,
            "group_index": group_index,
            "category_name": cat.get("category_name", ""),
            "definition": cat.get("definition", ""),
            "inclusion_criteria": cat.get("inclusion_criteria", []),
            "example_item_ids": cat.get("example_item_ids", []),
            "example_texts": cat.get("example_texts", []),
        })
    return rows


# ============================================================
# field pipeline
# ============================================================

def run_text_taxonomy_pipeline(
    raw_items: List[Dict[str, Any]],
    output_dir: Path,
    model_id: int,
    batch_size: int,
    merge_group_size: int,
    field_name: str,
    max_items: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    ensure_dir(output_dir)

    if max_items is not None:
        raw_items = raw_items[:max_items]

    dedup_items = deduplicate_text_items(raw_items)
    if not dedup_items:
        raise ValueError(f"No non-empty {field_name} items remained after deduplication.")

    dedup_jsonl = output_dir / f"{field_name}_dedup_items.jsonl"
    if dedup_jsonl.exists():
        dedup_jsonl.unlink()
    for item in dedup_items:
        append_jsonl(dedup_jsonl, item)

    taxonomy_jsonl = output_dir / f"{field_name}_taxonomy_rows.jsonl"
    taxonomy_csv = output_dir / f"{field_name}_taxonomy_rows.csv"
    if taxonomy_jsonl.exists():
        taxonomy_jsonl.unlink()
    if taxonomy_csv.exists():
        taxonomy_csv.unlink()

    item_batches = chunk_list(dedup_items, batch_size)
    batch_taxonomies: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(item_batches, start=1):
        taxonomy = build_taxonomy_for_batch(
            items=batch,
            batch_index=batch_index,
            model_id=model_id,
            output_dir=output_dir,
            field_name=field_name,
            debug=debug,
        )
        batch_taxonomies.append(taxonomy)

        for row in flatten_text_taxonomy_to_rows(taxonomy, field_name=field_name):
            append_jsonl(taxonomy_jsonl, row)
            append_csv(taxonomy_csv, row, TEXT_TAXONOMY_FIELDNAMES)

    current_taxonomies = batch_taxonomies
    round_index = 1

    while len(current_taxonomies) > 1:
        next_round: List[Dict[str, Any]] = []
        groups = group_list(current_taxonomies, merge_group_size)

        for group_index, group in enumerate(groups, start=1):
            if len(group) == 1:
                next_round.append(group[0])
                continue

            merged = merge_taxonomy_group(
                taxonomies=group,
                round_index=round_index,
                group_index=group_index,
                model_id=model_id,
                output_dir=output_dir,
                field_name=field_name,
                debug=debug,
            )
            next_round.append(merged)

            for row in flatten_text_taxonomy_to_rows(merged, field_name=field_name):
                append_jsonl(taxonomy_jsonl, row)
                append_csv(taxonomy_csv, row, TEXT_TAXONOMY_FIELDNAMES)

        current_taxonomies = next_round
        round_index += 1

    final_taxonomy = current_taxonomies[0]
    final_path = output_dir / f"final_{field_name}_taxonomy.json"
    write_json(final_path, final_taxonomy)

    report = {
        "field_name": field_name,
        "raw_item_count": len(raw_items),
        "dedup_item_count": len(dedup_items),
        "batch_count": len(item_batches),
        "final_taxonomy_path": str(final_path),
    }
    write_json(output_dir / f"{field_name}_run_report.json", report)
    return report


# ============================================================
# unified end-to-end pipeline
# ============================================================

def run_all(
    plan_dir: Path,
    output_dir: Path,
    model_id: int,
    batch_size: int,
    merge_group_size: int,
    max_items_per_field: Optional[int] = None,
    debug: bool = False,
) -> None:
    ensure_dir(output_dir)

    plan_nodes_jsonl = output_dir / "plan_nodes.jsonl"
    plan_nodes_csv = output_dir / "plan_nodes.csv"
    task_only_jsonl = output_dir / "task_only.jsonl"
    task_only_csv = output_dir / "task_only.csv"
    node_contract_only_jsonl = output_dir / "node_contract_only.jsonl"
    node_contract_only_csv = output_dir / "node_contract_only.csv"

    for p in [
        plan_nodes_jsonl,
        plan_nodes_csv,
        task_only_jsonl,
        task_only_csv,
        node_contract_only_jsonl,
        node_contract_only_csv,
    ]:
        if p.exists():
            p.unlink()

    node_rows: List[Dict[str, Any]] = []
    task_items: List[Dict[str, Any]] = []
    contract_items: List[Dict[str, Any]] = []
    seen_plan_files = set()

    for node_row in iter_plan_nodes(plan_dir, debug=debug):
        node_rows.append(node_row)
        seen_plan_files.add(node_row["source_file"])

        append_jsonl(plan_nodes_jsonl, node_row)
        append_csv(plan_nodes_csv, node_row, PLAN_NODE_FIELDNAMES)

        task_item = build_task_item(node_row)
        if task_item is not None:
            task_items.append(task_item)
            append_jsonl(task_only_jsonl, task_item)
            append_csv(task_only_csv, task_item, TEXT_ITEM_FIELDNAMES)

        contract_item = build_node_contract_item(node_row)
        if contract_item is not None:
            contract_items.append(contract_item)
            append_jsonl(node_contract_only_jsonl, contract_item)
            append_csv(node_contract_only_csv, contract_item, TEXT_ITEM_FIELDNAMES)

    extraction_report = {
        "plan_dir": str(plan_dir),
        "total_plan_files": len(seen_plan_files),
        "total_nodes": len(node_rows),
        "total_task_items": len(task_items),
        "total_node_contract_items": len(contract_items),
    }
    write_json(output_dir / "extract_report.json", extraction_report)

    if not task_items:
        raise ValueError("No task items were extracted from plan.txt files.")
    if not contract_items:
        raise ValueError("No node_contract items were extracted from plan.txt files.")

    task_out_dir = output_dir / "task_taxonomy"
    contract_out_dir = output_dir / "node_contract_taxonomy"

    task_report = run_text_taxonomy_pipeline(
        raw_items=task_items,
        output_dir=task_out_dir,
        model_id=model_id,
        batch_size=batch_size,
        merge_group_size=merge_group_size,
        field_name="task",
        max_items=max_items_per_field,
        debug=debug,
    )

    contract_report = run_text_taxonomy_pipeline(
        raw_items=contract_items,
        output_dir=contract_out_dir,
        model_id=model_id,
        batch_size=batch_size,
        merge_group_size=merge_group_size,
        field_name="node_contract",
        max_items=max_items_per_field,
        debug=debug,
    )

    final_report = {
        "plan_dir": str(plan_dir),
        "output_dir": str(output_dir),
        "model_id": model_id,
        "batch_size": batch_size,
        "merge_group_size": merge_group_size,
        "max_items_per_field": max_items_per_field,
        "extraction_report": extraction_report,
        "task_report": task_report,
        "node_contract_report": contract_report,
    }
    write_json(output_dir / "final_run_report.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=int, default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--merge_group_size", type=int, default=DEFAULT_MERGE_GROUP_SIZE)
    parser.add_argument("--max_items_per_field", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_all(
        plan_dir=Path(args.plan_dir),
        output_dir=Path(args.output_dir),
        model_id=args.model_id,
        batch_size=args.batch_size,
        merge_group_size=args.merge_group_size,
        max_items_per_field=args.max_items_per_field,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()