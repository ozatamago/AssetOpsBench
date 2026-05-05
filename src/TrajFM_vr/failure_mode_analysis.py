from __future__ import annotations

from agent_hive.workflows.verification_agent import VerificationAgent

import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterator, Optional, List
import time
import copy

from reactxen.utils.model_inference import watsonx_llm

logger = logging.getLogger(__name__)

class WatsonxJSONCaller:
    """
    Minimal robust JSON caller for watsonx.

    Responsibility:
    1. call watsonx_llm(prompt, model_id=...)
    2. normalize backend response into text
    3. recover one schema-matching JSON object
    4. currently supports only kind="fma_flags"

    Expected output for kind="fma_flags":
    {
      "flags": [
        {
          "flag": "2.4",
          "reason": "short reason",
          "evidence_refs": ["task_intent", "prev_response"]
        }
      ]
    }
    """

    def __init__(
        self,
        model_id: int | str,
        *,
        debug: bool = False,
    ) -> None:
        self.model_id = model_id
        self.debug = debug

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def call_json(self, prompt: str, kind: str) -> Dict[str, Any]:
        raw = watsonx_llm(prompt, model_id=self.model_id)
        print(f"raw: {raw}", flush=True)
        import time
        # time.sleep(10)
        raw_text = self._extract_text_from_backend_response(raw)

        if self.debug:
            logger.debug(
                "WatsonxJSONCaller.call_json raw_text preview=%s",
                self._debug_preview(raw_text, limit=1000),
            )

        if kind == "fma_flags":
            return self._extract_fma_flags_dict(raw_text)

        raise ValueError(f"Unsupported kind: {kind}")

    # ------------------------------------------------------------------
    # backend response normalization
    # ------------------------------------------------------------------
    def _extract_text_from_backend_response(self, raw: Any) -> str:
        """
        Normalize watsonx_llm response into a text string.

        Accepts:
        - str
        - bytes
        - dict with "generated_text"
        - dict with results[0]["generated_text"]
        """
        if isinstance(raw, str):
            return raw

        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")

        if isinstance(raw, dict):
            generated_text = raw.get("generated_text")
            if isinstance(generated_text, str):
                return generated_text

            results = raw.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    nested_generated_text = first.get("generated_text")
                    if isinstance(nested_generated_text, str):
                        return nested_generated_text

            raise ValueError(
                "watsonx response dict does not contain a string generated_text"
            )

        raise ValueError(f"Unexpected watsonx response type: {type(raw).__name__}")

    # ------------------------------------------------------------------
    # schema matcher: fma_flags
    # ------------------------------------------------------------------
    def _build_recovery_text_variants(self, text: str) -> list[str]:
        print("\n[_build_recovery_text_variants] ENTER", flush=True)
        print(f"[_build_recovery_text_variants] input_len={len(text)}", flush=True)
        print(
            f"[_build_recovery_text_variants] input_preview={self._debug_preview(text, limit=1200)}",
            flush=True,
        )
        # time.sleep(0.2)

        variants: list[str] = []

        if text.startswith('"'):
            print("[_build_recovery_text_variants] detected leading double quote", flush=True)
            variants.append(text[1:])
            if len(text) >= 2 and text.endswith('"'):
                print("[_build_recovery_text_variants] detected surrounding double quotes", flush=True)
                variants.append(text[1:-1])
            # time.sleep(0.2)

        try:
            loaded = json.loads(text)
            print(
                f"[_build_recovery_text_variants] json.loads(text) type={type(loaded).__name__}",
                flush=True,
            )
            if isinstance(loaded, str):
                print(
                    "[_build_recovery_text_variants] loaded value is str, appending nested text variant",
                    flush=True,
                )
                variants.append(loaded)
            # time.sleep(0.2)
        except Exception as exc:
            print(
                f"[_build_recovery_text_variants] json.loads(text) failed: {exc!r}",
                flush=True,
            )
            # time.sleep(0.2)

        if "\\n" in text or '\\"' in text:
            print("[_build_recovery_text_variants] detected escaped newline or quote", flush=True)
            variants.append(
                text.replace("\\n", "\n").replace('\\"', '"')
            )
            # time.sleep(0.2)

        out: list[str] = []
        seen: set[str] = set()
        for i, v in enumerate(variants):
            if not v:
                print(f"[_build_recovery_text_variants] skip empty variant idx={i}", flush=True)
                continue
            if v in seen:
                print(f"[_build_recovery_text_variants] skip duplicate variant idx={i}", flush=True)
                continue
            seen.add(v)
            out.append(v)
            print(
                f"[_build_recovery_text_variants] keep variant idx={i} preview={self._debug_preview(v, limit=800)}",
                flush=True,
            )
            # time.sleep(0.2)

        print(f"[_build_recovery_text_variants] total_variants={len(out)}", flush=True)
        # time.sleep(0.2)
        return out


    def _iter_prefixed_json_candidates(self, text: str) -> Iterator[str]:
        print("\n[_iter_prefixed_json_candidates] ENTER", flush=True)
        print(f"[_iter_prefixed_json_candidates] text_preview={self._debug_preview(text, limit=1200)}", flush=True)
        # time.sleep(0.2)

        markers = [
            "assistantfinal",
            "assistant",
            '{"flags"',
        ]

        for marker in markers:
            start = text.rfind(marker)
            print(
                f"[_iter_prefixed_json_candidates] marker={marker!r} rfind={start}",
                flush=True,
            )
            # time.sleep(0.2)

            if start == -1:
                continue

            brace_pos = text.find("{", start)
            print(
                f"[_iter_prefixed_json_candidates] marker={marker!r} brace_pos={brace_pos}",
                flush=True,
            )
            # time.sleep(0.2)

            if brace_pos == -1:
                continue

            suffix = text[brace_pos:]
            print(
                f"[_iter_prefixed_json_candidates] suffix_preview={self._debug_preview(suffix, limit=1000)}",
                flush=True,
            )
            # time.sleep(0.2)

            for candidate in self._iter_balanced_json_object_candidates(suffix):
                print(
                    f"[_iter_prefixed_json_candidates] yield candidate preview={self._debug_preview(candidate, limit=1000)}",
                    flush=True,
                )
                # time.sleep(0.2)
                yield candidate


    def _try_load_json_dict(self, text: str) -> Optional[Dict[str, Any]]:
        print("\n[_try_load_json_dict] ENTER", flush=True)
        print(f"[_try_load_json_dict] text_preview={self._debug_preview(text, limit=1200)}", flush=True)
        # time.sleep(0.2)

        try:
            obj = json.loads(text)
            print(f"[_try_load_json_dict] json.loads(text) type={type(obj).__name__}", flush=True)
            print(f"[_try_load_json_dict] loaded_preview={self._debug_preview(obj, limit=800)}", flush=True)
            # time.sleep(0.2)
        except json.JSONDecodeError as exc:
            around_start = max(0, exc.pos - 80)
            around_end = min(len(text), exc.pos + 80)
            around = text[around_start:around_end].replace("\n", "\\n")
            print(
                f"[_try_load_json_dict] JSONDecodeError msg={exc.msg} pos={exc.pos} "
                f"line={exc.lineno} col={exc.colno} around={around!r}",
                flush=True,
            )
            # time.sleep(0.2)
            return None
        except Exception as exc:
            print(f"[_try_load_json_dict] unexpected error: {exc!r}", flush=True)
            # time.sleep(0.2)
            return None

        if isinstance(obj, dict):
            print("[_try_load_json_dict] RETURN direct dict", flush=True)
            # time.sleep(0.2)
            return obj

        if isinstance(obj, str):
            print("[_try_load_json_dict] loaded object is str, trying nested json.loads", flush=True)
            # time.sleep(0.2)
            try:
                nested = json.loads(obj)
                print(
                    f"[_try_load_json_dict] nested json.loads type={type(nested).__name__}",
                    flush=True,
                )
                print(
                    f"[_try_load_json_dict] nested_preview={self._debug_preview(nested, limit=800)}",
                    flush=True,
                )
                # time.sleep(0.2)
                if isinstance(nested, dict):
                    print("[_try_load_json_dict] RETURN nested dict", flush=True)
                    # time.sleep(0.2)
                    return nested
            except Exception as exc:
                print(f"[_try_load_json_dict] nested json.loads failed: {exc!r}", flush=True)
                # time.sleep(0.2)

        print("[_try_load_json_dict] RETURN None", flush=True)
        # time.sleep(0.2)
        return None


    def _is_fma_flags_schema_match(self, obj: Any) -> bool:
        print("\n[_is_fma_flags_schema_match] ENTER", flush=True)
        print(f"[_is_fma_flags_schema_match] obj_type={type(obj).__name__}", flush=True)
        print(f"[_is_fma_flags_schema_match] obj_preview={self._debug_preview(obj, limit=500)}", flush=True)
        # time.sleep(0.2)

        if not isinstance(obj, dict):
            print("[_is_fma_flags_schema_match] FAIL: obj is not dict", flush=True)
            # time.sleep(0.2)
            return False

        flags = obj.get("flags")
        print(f"[_is_fma_flags_schema_match] flags_type={type(flags).__name__}", flush=True)
        print(f"[_is_fma_flags_schema_match] flags_preview={self._debug_preview(flags, limit=500)}", flush=True)
        # time.sleep(0.2)

        if not isinstance(flags, list):
            print("[_is_fma_flags_schema_match] FAIL: flags is not list", flush=True)
            # time.sleep(0.2)
            return False

        for i, item in enumerate(flags):
            print(f"[_is_fma_flags_schema_match] checking flags[{i}] type={type(item).__name__}", flush=True)
            print(f"[_is_fma_flags_schema_match] flags[{i}] preview={self._debug_preview(item, limit=500)}", flush=True)
            # time.sleep(0.2)

            if not isinstance(item, dict):
                print(f"[_is_fma_flags_schema_match] FAIL: flags[{i}] is not dict", flush=True)
                # time.sleep(0.2)
                return False

            flag = item.get("flag")
            reason = item.get("reason")
            evidence_refs = item.get("evidence_refs", [])

            print(f"[_is_fma_flags_schema_match] flags[{i}].flag type={type(flag).__name__} value={flag!r}", flush=True)
            print(f"[_is_fma_flags_schema_match] flags[{i}].reason type={type(reason).__name__} value={reason!r}", flush=True)
            print(
                f"[_is_fma_flags_schema_match] flags[{i}].evidence_refs type={type(evidence_refs).__name__} "
                f"value={self._debug_preview(evidence_refs, limit=500)}",
                flush=True,
            )
            # time.sleep(0.2)

            if not isinstance(flag, str):
                print(f"[_is_fma_flags_schema_match] FAIL: flags[{i}].flag is not str", flush=True)
                # time.sleep(0.2)
                return False
            if not isinstance(reason, str):
                print(f"[_is_fma_flags_schema_match] FAIL: flags[{i}].reason is not str", flush=True)
                # time.sleep(0.2)
                return False
            if not isinstance(evidence_refs, list):
                print(f"[_is_fma_flags_schema_match] FAIL: flags[{i}].evidence_refs is not list", flush=True)
                # time.sleep(0.2)
                return False

            for j, ref in enumerate(evidence_refs):
                print(
                    f"[_is_fma_flags_schema_match] flags[{i}].evidence_refs[{j}] "
                    f"type={type(ref).__name__} value={ref!r}",
                    flush=True,
                )
                time.sleep(0.1)

                if not isinstance(ref, str):
                    print(
                        f"[_is_fma_flags_schema_match] FAIL: flags[{i}].evidence_refs[{j}] is not str",
                        flush=True,
                    )
                    # time.sleep(0.2)
                    return False

        print("[_is_fma_flags_schema_match] PASS", flush=True)
        # time.sleep(0.2)
        return True


    def _extract_fma_flags_dict(self, raw: Any) -> Dict[str, Any]:
        """
        Accepts:
        - dict
        - str containing only JSON
        - str containing extra text before/after JSON
        - str containing fenced JSON
        - str containing assistantfinal{...}
        - str that is itself a quoted JSON-like wrapper

        Strategy:
        - first normalize to text
        - then try original text plus recovery variants
        - for each text:
        1. full text parse
        2. fenced JSON blocks
        3. assistant/prefixed JSON candidates
        4. balanced {...} candidates
        - return the first schema-matching dict found
        """
        print("\n[_extract_fma_flags_dict] ENTER", flush=True)
        print(f"[_extract_fma_flags_dict] raw_type={type(raw).__name__}", flush=True)
        print(f"[_extract_fma_flags_dict] raw_preview={self._debug_preview(raw, limit=1000)}", flush=True)
        # time.sleep(0.2)

        if isinstance(raw, dict):
            print("[_extract_fma_flags_dict] raw is dict, checking schema directly", flush=True)
            # time.sleep(0.2)

            schema_ok = self._is_fma_flags_schema_match(raw)
            print(f"[_extract_fma_flags_dict] direct_dict_schema_ok={schema_ok}", flush=True)
            # time.sleep(0.2)

            if schema_ok:
                print("[_extract_fma_flags_dict] RETURN direct raw dict", flush=True)
                # time.sleep(0.2)
                return raw

            print(
                f"[_extract_fma_flags_dict] FAIL direct dict keys={sorted(raw.keys())}",
                flush=True,
            )
            # time.sleep(0.2)
            raise ValueError(
                f"fma_flags dict does not match required schema; keys={sorted(raw.keys())}"
            )

        if isinstance(raw, bytes):
            print("[_extract_fma_flags_dict] raw is bytes, decoding utf-8", flush=True)
            # time.sleep(0.2)
            raw = raw.decode("utf-8", errors="replace")
            print(f"[_extract_fma_flags_dict] decoded_raw_preview={self._debug_preview(raw, limit=1000)}", flush=True)
            # time.sleep(0.2)

        if not isinstance(raw, str):
            print(f"[_extract_fma_flags_dict] FAIL: unsupported raw type={type(raw).__name__}", flush=True)
            # time.sleep(0.2)
            raise ValueError(
                f"fma_flags output must be a dict or str, got {type(raw).__name__}"
            )

        text = raw.strip()
        print(f"[_extract_fma_flags_dict] stripped_text_len={len(text)}", flush=True)
        print(f"[_extract_fma_flags_dict] stripped_text_preview={self._debug_preview(text, limit=1200)}", flush=True)
        # time.sleep(0.2)

        if not text:
            print("[_extract_fma_flags_dict] FAIL: empty text", flush=True)
            # time.sleep(0.2)
            raise ValueError("fma_flags text is empty")

        texts_to_try = [text] + self._build_recovery_text_variants(text)

        deduped_texts: list[str] = []
        seen_texts: set[str] = set()
        for i, candidate_text in enumerate(texts_to_try):
            if candidate_text in seen_texts:
                print(f"[_extract_fma_flags_dict] skip duplicate text variant idx={i}", flush=True)
                continue
            seen_texts.add(candidate_text)
            deduped_texts.append(candidate_text)
            print(
                f"[_extract_fma_flags_dict] keep text variant idx={i} preview={self._debug_preview(candidate_text, limit=1000)}",
                flush=True,
            )
            # time.sleep(0.2)

        def _try_candidates_from_text(current_text: str, label: str) -> Optional[Dict[str, Any]]:
            print(f"\n[_extract_fma_flags_dict] TRY TEXT label={label}", flush=True)
            print(
                f"[_extract_fma_flags_dict] current_text_preview={self._debug_preview(current_text, limit=1200)}",
                flush=True,
            )
            # time.sleep(0.2)

            # 1. full text parse
            print(f"[_extract_fma_flags_dict] {label} STEP1: try full text parse", flush=True)
            # time.sleep(0.2)
            obj = self._try_load_json_dict(current_text)
            print(
                f"[_extract_fma_flags_dict] {label} STEP1 loaded_obj_type={type(obj).__name__ if obj is not None else 'None'}",
                flush=True,
            )
            print(
                f"[_extract_fma_flags_dict] {label} STEP1 loaded_obj_preview={self._debug_preview(obj, limit=800)}",
                flush=True,
            )
            # time.sleep(0.2)

            if obj is not None:
                schema_ok = self._is_fma_flags_schema_match(obj)
                print(f"[_extract_fma_flags_dict] {label} STEP1 schema_ok={schema_ok}", flush=True)
                # time.sleep(0.2)
                if schema_ok:
                    print(f"[_extract_fma_flags_dict] RETURN from {label} STEP1", flush=True)
                    # time.sleep(0.2)
                    return obj

            # 2. fenced JSON blocks
            print(f"[_extract_fma_flags_dict] {label} STEP2: search fenced JSON blocks", flush=True)
            # time.sleep(0.2)
            fenced_blocks = re.findall(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                current_text,
                flags=re.DOTALL,
            )
            print(f"[_extract_fma_flags_dict] {label} STEP2 fenced_blocks_count={len(fenced_blocks)}", flush=True)
            # time.sleep(0.2)

            for idx, fenced in enumerate(reversed(fenced_blocks)):
                original_idx = len(fenced_blocks) - 1 - idx
                print(
                    f"[_extract_fma_flags_dict] {label} STEP2 checking fenced block original_idx={original_idx}",
                    flush=True,
                )
                print(
                    f"[_extract_fma_flags_dict] {label} STEP2 fenced_preview={self._debug_preview(fenced, limit=1000)}",
                    flush=True,
                )
                # time.sleep(0.2)

                obj = self._try_load_json_dict(fenced)
                print(
                    f"[_extract_fma_flags_dict] {label} STEP2 parsed_obj_type={type(obj).__name__ if obj is not None else 'None'}",
                    flush=True,
                )
                print(
                    f"[_extract_fma_flags_dict] {label} STEP2 parsed_obj_preview={self._debug_preview(obj, limit=800)}",
                    flush=True,
                )
                # time.sleep(0.2)

                if obj is not None:
                    schema_ok = self._is_fma_flags_schema_match(obj)
                    print(f"[_extract_fma_flags_dict] {label} STEP2 schema_ok={schema_ok}", flush=True)
                    # time.sleep(0.2)
                    if schema_ok:
                        print(
                            f"[_extract_fma_flags_dict] RETURN from {label} STEP2 fenced block original_idx={original_idx}",
                            flush=True,
                        )
                        # time.sleep(0.2)
                        return obj

            # 3. prefixed JSON candidates
            print(f"[_extract_fma_flags_dict] {label} STEP3: search prefixed JSON candidates", flush=True)
            # time.sleep(0.2)
            prefixed_candidates = list(self._iter_prefixed_json_candidates(current_text))
            print(
                f"[_extract_fma_flags_dict] {label} STEP3 prefixed_candidates_count={len(prefixed_candidates)}",
                flush=True,
            )
            # time.sleep(0.2)

            for idx, candidate in enumerate(reversed(prefixed_candidates)):
                original_idx = len(prefixed_candidates) - 1 - idx
                print(
                    f"[_extract_fma_flags_dict] {label} STEP3 checking prefixed candidate original_idx={original_idx}",
                    flush=True,
                )
                print(
                    f"[_extract_fma_flags_dict] {label} STEP3 candidate_preview={self._debug_preview(candidate, limit=1000)}",
                    flush=True,
                )
                # time.sleep(0.2)

                obj = self._try_load_json_dict(candidate)
                print(
                    f"[_extract_fma_flags_dict] {label} STEP3 parsed_obj_type={type(obj).__name__ if obj is not None else 'None'}",
                    flush=True,
                )
                print(
                    f"[_extract_fma_flags_dict] {label} STEP3 parsed_obj_preview={self._debug_preview(obj, limit=800)}",
                    flush=True,
                )
                # time.sleep(0.2)

                if obj is not None:
                    schema_ok = self._is_fma_flags_schema_match(obj)
                    print(f"[_extract_fma_flags_dict] {label} STEP3 schema_ok={schema_ok}", flush=True)
                    # time.sleep(0.2)
                    if schema_ok:
                        print(
                            f"[_extract_fma_flags_dict] RETURN from {label} STEP3 prefixed candidate original_idx={original_idx}",
                            flush=True,
                        )
                        # time.sleep(0.2)
                        return obj

            # 4. balanced JSON candidates
            print(f"[_extract_fma_flags_dict] {label} STEP4: search balanced JSON object candidates", flush=True)
            # time.sleep(0.2)
            balanced_candidates = list(self._iter_balanced_json_object_candidates(current_text))
            print(
                f"[_extract_fma_flags_dict] {label} STEP4 balanced_candidates_count={len(balanced_candidates)}",
                flush=True,
            )
            # time.sleep(0.2)

            for idx, candidate in enumerate(reversed(balanced_candidates)):
                original_idx = len(balanced_candidates) - 1 - idx
                print(
                    f"[_extract_fma_flags_dict] {label} STEP4 checking balanced candidate original_idx={original_idx}",
                    flush=True,
                )
                print(
                    f"[_extract_fma_flags_dict] {label} STEP4 candidate_preview={self._debug_preview(candidate, limit=1000)}",
                    flush=True,
                )
                # time.sleep(0.2)

                obj = self._try_load_json_dict(candidate)
                print(
                    f"[_extract_fma_flags_dict] {label} STEP4 parsed_obj_type={type(obj).__name__ if obj is not None else 'None'}",
                    flush=True,
                )
                print(
                    f"[_extract_fma_flags_dict] {label} STEP4 parsed_obj_preview={self._debug_preview(obj, limit=800)}",
                    flush=True,
                )
                # time.sleep(0.2)

                if obj is not None:
                    schema_ok = self._is_fma_flags_schema_match(obj)
                    print(f"[_extract_fma_flags_dict] {label} STEP4 schema_ok={schema_ok}", flush=True)
                    # time.sleep(0.2)
                    if schema_ok:
                        print(
                            f"[_extract_fma_flags_dict] RETURN from {label} STEP4 balanced candidate original_idx={original_idx}",
                            flush=True,
                        )
                        # time.sleep(0.2)
                        return obj

            print(f"[_extract_fma_flags_dict] {label} exhausted without match", flush=True)
            # time.sleep(0.2)
            return None

        for idx, candidate_text in enumerate(deduped_texts):
            label = f"text_variant_{idx}"
            result = _try_candidates_from_text(candidate_text, label)
            if result is not None:
                return result

        preview = text[:300].replace("\n", "\\n")
        print("[_extract_fma_flags_dict] FAIL: no schema-matching dict recovered", flush=True)
        print(f"[_extract_fma_flags_dict] final_preview={preview!r}", flush=True)
        # time.sleep(0.2)
        raise ValueError(
            "could not recover a schema-matching fma_flags dict from text; "
            f"prefix={preview!r}"
        )

    # ------------------------------------------------------------------
    # robust json extraction helpers
    # ------------------------------------------------------------------
    def _try_load_json_dict(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            if self.debug:
                around_start = max(0, exc.pos - 60)
                around_end = min(len(text), exc.pos + 60)
                around = text[around_start:around_end].replace("\n", "\\n")
                logger.debug(
                    "_try_load_json_dict failed: msg=%s pos=%d line=%d col=%d around=%s",
                    exc.msg,
                    exc.pos,
                    exc.lineno,
                    exc.colno,
                    around,
                )
            return None

        if isinstance(obj, dict):
            return obj
        return None

    def _iter_balanced_json_object_candidates(
        self,
        text: str,
    ) -> Iterator[str]:
        """
        Yield balanced {...} substrings while respecting JSON string literals.
        """
        start: Optional[int] = None
        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text):
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
                if depth == 0:
                    start = i
                depth += 1
                continue

            if ch == "}":
                if depth == 0:
                    continue

                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None

    def _debug_preview(self, value: Any, limit: int = 240) -> str:
        try:
            text = value if isinstance(value, str) else repr(value)
        except Exception:
            text = f"<unreprable {type(value).__name__}>"

        text = text.replace("\n", "\\n")
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text
    
@dataclass
class TrajectoryRecord:
    qid: str
    trajectory_path: Path
    raw_obj: Dict[str, Any]


@dataclass
class TrajectoryIndex:
    by_node_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ordered_node_ids: List[str] = field(default_factory=list)
    s_nodes: List[str] = field(default_factory=list)
    v_by_s: Dict[str, str] = field(default_factory=dict)
    r_by_s: Dict[str, str] = field(default_factory=dict)
    prev_node_of: Dict[str, Optional[str]] = field(default_factory=dict)
    next_node_of: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class FlagRecord:
    flag: str
    subject: str
    reason: str
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class FlagBundle:
    layer: str
    flags: List[FlagRecord] = field(default_factory=list)

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")

    return data

FAILURE_MODE_DEFINITIONS = """
1.1 Disobey Task Specification:
The agent or system fails to follow explicit task requirements, constraints, or instructions.

1.2 Disobey Role Specification:
The agent fails to behave according to its assigned role or responsibility.

1.3 Step Repetition:
The agent unnecessarily repeats a task, step, or phase that was already completed.

1.4 Loss of Conversation History:
The agent loses or ignores important recent context and reverts to an earlier state.

1.5 Unaware of Termination Conditions:
The agent fails to recognize when stopping conditions have been met or when progress is no longer possible.

1.6 Premature Termination:
Ending a task or conversation before the necessary information has been exchanged or objectives fully met.

2.1 Conversation Reset:
The interaction is unexpectedly restarted or reset, causing loss of progress or context.

2.2 Fail to Ask for Clarification:
The agent proceeds despite ambiguity or missing information that should have triggered clarification.

2.3 Task Derailment:
The agent deviates from the intended task and pursues irrelevant or unproductive actions.

2.4 Information Withholding:
An agent has important information but fails to pass it to other agents or system components.

2.5 Ignored Other Agent's Input:
An agent fails to consider or appropriately act on another agent's useful input.

2.6 Action-Reasoning Mismatch:
The agent's reasoning and actual action contradict each other.

3.1 Failure Signal Miss or Misdetection:
A verification stage fails to detect, or incorrectly detects, failure signals grounded in available observations such as logs, tool outputs, or responses.

3.2 Failure Root Not Isolated:
A verification stage detects that something is wrong but does not isolate the earliest structural break or root failure at a useful granularity.

3.3 Failure Representation Breakdown:
A verification stage fails to integrate detected failure signals into a coherent representation that preserves temporal order or causal relations.

3.4 Diagnosis Compression Mismatch:
The diagnosis passed to recovery is too compressed or too detailed for effective recovery use.

3.5 Unsupported Fault Hypothesis:
The diagnosis does not support a grounded fault hypothesis that recovery can use.

3.6 Missing Upstream Repair Signal:
The diagnosis fails to indicate that the fault originates upstream and requires upstream repair, escalation, or stopping rather than local retry.

4.1 Fault Misidentification:
The recovery stage identifies the wrong fault, or acts without correctly determining the fault.

4.2 Incorrect Probe Selection:
The recovery stage chooses the wrong probe, test, or tool interaction for the current fault hypothesis.

4.3 Unsafe or Improper Termination:
The recovery stage terminates too early, too late, or in a state that is not safe or not sufficiently consistent.
""".strip()


class FailureModeAnalysis:
    def __init__(
        self,
        verification_agent: VerificationAgent,
        llm_json_caller: WatsonxJSONCaller,
        cache_dir: Path,
        output_dir: Path,
        overwrite_verification: bool = False,
        overwrite_stage_cache: bool = False,
        overwrite_stage_prefixes: Optional[List[str]] = None,
    ) -> None:
        self.verification_agent = verification_agent
        self.llm_json_caller = llm_json_caller
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.overwrite_verification = overwrite_verification
        self.overwrite_stage_cache = overwrite_stage_cache
        self.overwrite_stage_prefixes = {
            x.strip() for x in (overwrite_stage_prefixes or []) if str(x).strip()
        }
        self.logger = logging.getLogger(__name__)

        print(
            f"self.overwrite_verification: {self.overwrite_verification}, "
            f"self.overwrite_stage_cache: {self.overwrite_stage_cache}, "
            f"self.overwrite_stage_prefixes: {sorted(self.overwrite_stage_prefixes)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 4. json/path helpers
    # ------------------------------------------------------------------
    def _should_overwrite_stage(self, stage_name: str) -> bool:
        if self.overwrite_stage_cache:
            return True

        return any(
            stage_name.startswith(prefix)
            for prefix in self.overwrite_stage_prefixes
        )

    def read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected top-level JSON object in {path}, got {type(data).__name__}"
            )
        return data

    def write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def build_verification_cache_path(self, qid: str) -> Path:
        return self.cache_dir / f"verification_only_Q{qid}.json"

    def build_flag_report_path(self, qid: str) -> Path:
        return self.output_dir / f"fma_flags_Q{qid}.json"

    def build_recovery_only_cache_path(self, qid: str) -> Path:
        return self.cache_dir / f"recovery_only_Q{qid}.json"

    def build_stage_cache_path(self, qid: str, stage_name: str) -> Path:
        return self.cache_dir / f"{stage_name}_Q{qid}.json"

    def _run_llm_flag_stage(
        self,
        qid: str,
        stage_name: str,
        layer: str,
        subject: str,
        prompt: str,
    ) -> FlagBundle:
        bundle = FlagBundle(layer=layer)
        cache_path = self.build_stage_cache_path(qid, stage_name)

        cache_obj: Dict[str, Any] = {}
        result_obj: Dict[str, Any] = {}

        should_overwrite = self._should_overwrite_stage(stage_name)

        if should_overwrite:
            print(
                f"[stage] overwrite matched -> ignore cache and rerun: "
                f"stage={stage_name}, qid={qid}, path={cache_path}",
                flush=True,
            )
        elif cache_path.exists():
            print(
                f"[stage] cache hit -> reuse cached result: "
                f"stage={stage_name}, qid={qid}, path={cache_path}",
                flush=True,
            )
            try:
                cache_obj = self.read_json(cache_path)

                if cache_obj.get("error"):
                    print(
                        f"[stage] cached error found -> rerun stage: "
                        f"stage={stage_name}, qid={qid}, path={cache_path}",
                        flush=True,
                    )
                    result_obj = {}
                else:
                    cached_result = cache_obj.get("result", {})
                    if isinstance(cached_result, dict):
                        result_obj = cached_result
                    else:
                        result_obj = {}
            except Exception as exc:
                self.logger.warning(
                    "Failed to read stage cache %s: %s",
                    cache_path,
                    repr(exc),
                )
                cache_obj = {}
                result_obj = {}

        if not result_obj:
            print(
                f"[stage] executing LLM: stage={stage_name}, qid={qid}, subject={subject}",
                flush=True,
            )
            try:
                result_obj = self.llm_json_caller.call_json(
                    prompt,
                    kind="fma_flags",
                )

                if not isinstance(result_obj, dict):
                    raise TypeError(
                        f"LLM result must be a dict, got {type(result_obj).__name__}"
                    )

                cache_obj = {
                    "qid": qid,
                    "stage_name": stage_name,
                    "layer": layer,
                    "subject": subject,
                    "prompt": prompt,
                    "result": result_obj,
                }
                self.write_json(cache_path, cache_obj)
                print(
                    f"[stage] wrote cache: stage={stage_name}, qid={qid}, path={cache_path}",
                    flush=True,
                )

            except Exception as exc:
                self.logger.warning(
                    "LLM flag stage failed: stage=%s qid=%s subject=%s error=%s",
                    stage_name,
                    qid,
                    subject,
                    repr(exc),
                )

                error_cache = {
                    "qid": qid,
                    "stage_name": stage_name,
                    "layer": layer,
                    "subject": subject,
                    "prompt": prompt,
                    "result": {"flags": []},
                    "error": repr(exc),
                }
                self.write_json(cache_path, error_cache)
                print(
                    f"[stage] wrote error cache: stage={stage_name}, qid={qid}, path={cache_path}",
                    flush=True,
                )
                return bundle

        raw_flags = result_obj.get("flags", [])
        if not isinstance(raw_flags, list):
            self.logger.warning(
                "Stage result has non-list 'flags': stage=%s qid=%s type=%s",
                stage_name,
                qid,
                type(raw_flags).__name__,
            )
            return bundle

        for item in raw_flags:
            if not isinstance(item, dict):
                continue

            flag = str(item.get("flag", "")).strip()
            reason = self._as_text(item.get("reason", "")).strip()

            evidence_refs_raw = item.get("evidence_refs", [])
            if isinstance(evidence_refs_raw, list):
                evidence_refs = [
                    self._as_text(x).strip()
                    for x in evidence_refs_raw
                    if self._as_text(x).strip()
                ]
            else:
                evidence_refs = []

            if not flag:
                continue

            self._add_flag(
                bundle=bundle,
                flag=flag,
                subject=subject,
                reason=reason if reason else f"flagged by LLM stage {stage_name}",
                evidence_refs=evidence_refs,
            )

        return bundle


    # ------------------------------------------------------------------
    # 5. trajectory/index helpers
    # ------------------------------------------------------------------
    def load_trajectory(self, trajectory_path: Path) -> TrajectoryRecord:
        raw_obj = self.read_json(trajectory_path)
        qid = self._extract_qid(raw_obj, trajectory_path)
        return TrajectoryRecord(
            qid=qid,
            trajectory_path=trajectory_path,
            raw_obj=raw_obj,
        )
    
    def _normalize_or_derive_node_id(self, node: Dict[str, Any]) -> str:
        node_id = str(node.get("node_id", "")).strip()
        if node_id:
            return node_id

        task_number = node.get("task_number")
        try:
            n = int(task_number)
        except (TypeError, ValueError):
            return ""

        if n <= 0:
            return ""

        return f"S{n}"

    def build_trajectory_index(self, record: TrajectoryRecord) -> TrajectoryIndex:
        raw_traj = record.raw_obj.get("trajectory", [])
        if not isinstance(raw_traj, list):
            raise TypeError(
                f"Expected 'trajectory' to be a list in {record.trajectory_path}, "
                f"got {type(raw_traj).__name__}"
            )

        index = TrajectoryIndex()
        previous_node_id: Optional[str] = None

        for i, node in enumerate(raw_traj):
            if not isinstance(node, dict):
                print(f"[index] skip non-dict node at i={i}: type={type(node).__name__}", flush=True)
                continue

            original_node_id = str(node.get("node_id", "")).strip()
            node_id = self._normalize_or_derive_node_id(node)

            print(
                f"[index] i={i} original_node_id={original_node_id!r} "
                f"task_number={node.get('task_number')!r} normalized_node_id={node_id!r}",
                flush=True,
            )

            if not node_id:
                print(f"[index] skip unresolved node_id at i={i}", flush=True)
                continue

            normalized_node = dict(node)
            normalized_node["node_id"] = node_id

            index.by_node_id[node_id] = normalized_node
            index.ordered_node_ids.append(node_id)

            if previous_node_id is not None:
                index.prev_node_of[node_id] = previous_node_id
                index.next_node_of[previous_node_id] = node_id
            else:
                index.prev_node_of[node_id] = None

            previous_node_id = node_id

            if self._is_s_node(node_id):
                index.s_nodes.append(node_id)

        if previous_node_id is not None and previous_node_id not in index.next_node_of:
            index.next_node_of[previous_node_id] = None

        for s_node_id in index.s_nodes:
            verifier_id = self._candidate_verifier_id(s_node_id)
            recovery_id = self._candidate_recovery_id(s_node_id)

            if verifier_id in index.by_node_id:
                index.v_by_s[s_node_id] = verifier_id
            if recovery_id in index.by_node_id:
                index.r_by_s[s_node_id] = recovery_id

        return index

    def get_s_node_ids(self, index: TrajectoryIndex) -> List[str]:
        return list(index.s_nodes)

    def get_verifier_for_s(
        self,
        index: TrajectoryIndex,
        s_node_id: str,
    ) -> Optional[str]:
        return index.v_by_s.get(s_node_id)

    def get_recovery_for_s(
        self,
        index: TrajectoryIndex,
        s_node_id: str,
    ) -> Optional[str]:
        return index.r_by_s.get(s_node_id)

    def get_prev_response(
        self,
        index: TrajectoryIndex,
        node_id: str,
    ) -> Optional[Any]:
        prev_node_id = index.prev_node_of.get(node_id)
        if prev_node_id is None:
            return None

        prev_node = index.by_node_id.get(prev_node_id)
        if not isinstance(prev_node, dict):
            return None

        return prev_node.get("response")
    
    def get_prev_node_id(
        self,
        index: TrajectoryIndex,
        node_id: str,
    ) -> Optional[str]:
        return index.prev_node_of.get(node_id)


    def get_prev_logs(
        self,
        index: TrajectoryIndex,
        node_id: str,
    ) -> Optional[Any]:
        prev_node_id = self.get_prev_node_id(index, node_id)
        if prev_node_id is None:
            return None

        prev_node = index.by_node_id.get(prev_node_id)
        if not isinstance(prev_node, dict):
            return None

        return prev_node.get("logs")

    # ------------------------------------------------------------------
    # 6. verification cache helpers
    # ------------------------------------------------------------------
    def load_verification_cache(self, qid: str) -> Dict[str, Any]:
        cache_path = self.build_verification_cache_path(qid)
        if not cache_path.exists():
            return {"qid": qid, "nodes": {}}

        cache_obj = self.read_json(cache_path)
        if "nodes" not in cache_obj or not isinstance(cache_obj["nodes"], dict):
            cache_obj["nodes"] = {}
        if "qid" not in cache_obj:
            cache_obj["qid"] = qid
        return cache_obj

    def save_verification_cache(self, qid: str, cache_obj: Dict[str, Any]) -> None:
        cache_path = self.build_verification_cache_path(qid)
        self.write_json(cache_path, cache_obj)

    def run_or_load_verification_for_s_nodes(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> Dict[str, Any]:
        if self.overwrite_verification:
            print(
                f"[verification] overwrite_verification=True -> ignore verification cache and rerun all S nodes: qid={record.qid}",
                flush=True,
            )
            cache_obj = {"qid": record.qid, "nodes": {}}
        else:
            cache_obj = self.load_verification_cache(record.qid)
            print(
                f"[verification] loaded verification cache: qid={record.qid}, "
                f"cached_nodes={list(cache_obj.get('nodes', {}).keys())}",
                flush=True,
            )

        nodes_cache = cache_obj.setdefault("nodes", {})

        for s_node_id in self.get_s_node_ids(index):
            if (not self.overwrite_verification) and s_node_id in nodes_cache:
                print(
                    f"[verification] cache hit -> skip verify: qid={record.qid}, node={s_node_id}",
                    flush=True,
                )
                continue

            print(
                f"[verification] executing verify: qid={record.qid}, node={s_node_id}",
                flush=True,
            )

            s_node = index.by_node_id.get(s_node_id)
            if not isinstance(s_node, dict):
                print(
                    f"[verification] skip invalid node object: qid={record.qid}, node={s_node_id}",
                    flush=True,
                )
                continue

            node_payload = self._build_verification_node_payload(s_node)
            logs_payload = self._extract_logs_for_verification(s_node)

            verification_result = self.verification_agent.verify(
                node=node_payload,
                logs=logs_payload,
            )

            nodes_cache[s_node_id] = {
                "node_id": s_node_id,
                "verification_result": verification_result,
            }

            print(
                f"[verification] wrote node result into cache object: qid={record.qid}, node={s_node_id}",
                flush=True,
            )

        self.save_verification_cache(record.qid, cache_obj)
        print(
            f"[verification] saved verification cache: qid={record.qid}, "
            f"path={self.build_verification_cache_path(record.qid)}",
            flush=True,
        )
        return cache_obj

    # ------------------------------------------------------------------
    # private helpers for sections 5-6
    # ------------------------------------------------------------------
    def _extract_qid(self, raw_obj: Dict[str, Any], trajectory_path: Path) -> str:
        direct_id = raw_obj.get("id")
        if direct_id is not None:
            return str(direct_id)

        stem = trajectory_path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return digits

        raise ValueError(f"Could not determine qid from {trajectory_path}")

    def _is_s_node(self, node_id: str) -> bool:
        return node_id.startswith("S") and not node_id.startswith("V_") and not node_id.startswith("R_")

    def _candidate_verifier_id(self, s_node_id: str) -> str:
        return f"V_{s_node_id}"

    def _candidate_recovery_id(self, s_node_id: str) -> str:
        return f"R_{s_node_id}"

    def _build_verification_node_payload(self, node_obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(node_obj.get("node_id", "")),
            "task": str(node_obj.get("task_description", "")),
            "agent": str(node_obj.get("agent_name", "")),
            "deps": self._infer_deps_from_node_id(str(node_obj.get("node_id", "")).strip()),
            "node_contract": str(node_obj.get("node_contract", "")),
        }

    def _extract_logs_for_verification(self, node_obj: Dict[str, Any]) -> Dict[str, Any]:
        logs = node_obj.get("logs", {})
        if isinstance(logs, dict):
            logs_copy = dict(logs)
        else:
            logs_copy = {"final_answer": "", "reviews": []}

        response = node_obj.get("response")
        reviews = logs_copy.get("reviews")

        if not isinstance(reviews, list):
            reviews = []

        if isinstance(response, list) and len(response) >= 2 and isinstance(response[1], dict):
            review_dict = response[1]
            review_text_parts: List[str] = []

            status = review_dict.get("status")
            reasoning = review_dict.get("reasoning")
            suggestions = review_dict.get("suggestions")

            if status:
                review_text_parts.append(f"Task Status: {status}")
            if reasoning:
                review_text_parts.append(f"Reasoning: {reasoning}")
            if suggestions:
                review_text_parts.append(f"Suggestions for Improvement: {suggestions}")

            if review_text_parts:
                reviews.append("\n".join(review_text_parts))

        logs_copy["reviews"] = reviews

        if "final_answer" not in logs_copy:
            if isinstance(response, list) and response:
                logs_copy["final_answer"] = response[0]
            elif isinstance(response, str):
                logs_copy["final_answer"] = response
            else:
                logs_copy["final_answer"] = ""

        return logs_copy

    def _infer_deps_from_node_id(self, node_id: str) -> List[str]:
        if node_id.startswith("S"):
            digits = node_id[1:]
            if digits.isdigit() and int(digits) > 1:
                return [f"S{int(digits) - 1}"]
            return []

        if node_id.startswith("V_S"):
            base = node_id[2:]
            return [base]

        if node_id.startswith("R_S"):
            base = node_id[2:]
            return [base, f"V_{base}"]

        return []

# 追加 import
# import re

    # ------------------------------------------------------------------
    # 7. flag-producing evaluators
    # ------------------------------------------------------------------

    def _build_stage2_prompt(
        self,
        current_node_id: str,
        task_intent: str,
        prev_node_id: str,
        prev_response: Any,
        prev_logs: Any,
    ) -> str:
        prev_response_text = self._as_text(prev_response)
        prev_logs_text = self._as_text(prev_logs)

        return f"""
    You are a failure mode analysis judge.

    You will judge whether the previous node failed to pass required information to the current node.

    Failure mode definitions:
    1.4 Loss of Conversation History:
    The agent loses or ignores important recent context and reverts to an earlier state.

    2.1 Conversation Reset:
    The interaction is unexpectedly restarted or reset, causing loss of progress or context.

    2.4 Information Withholding:
    An agent has important information but fails to pass it to other agents or system components.

    Stage objective:
    - This is Stage 2: handoff payload.
    - Compare the current node task intent against the previous node response.
    - Use the previous node logs only as supporting evidence for whether the previous node actually had or obtained the information that should have been passed forward.
    - Judge whether the previous node failed to pass required information to the current node.
    - In this stage, focus mainly on:
        - 1.4 Loss of Conversation History
        - 2.1 Conversation Reset
        - 2.4 Information Withholding

    Important scope restriction:
    - Judge only payload adequacy at the handoff boundary.
    - Do not judge whether the previous node's answer was correct in itself.
    - Do not judge whether the current node later used the input correctly; that belongs to Stage 3.
    - Do not judge verifier-quality failures; those belong to later verifier stages.
    - Do not judge recovery-execution failures; those belong to recovery stages.

    Role-sensitive handoff rule:
    - Judge payload adequacy relative to the current node's role.
    - For normal task nodes, ask whether the previous response passed the information needed for the current task.
    - For verifier nodes, ask whether the previous response and surrounding context provide enough material to verify the source node.
    - For recovery nodes, ask whether the previous response passed the diagnosis and recovery-relevant information needed for recovery.

    Judging rules:
    - Flag 2.4 only if the previous node had, obtained, but did not pass it adequately in the previous response.
    - Use previous node logs as supporting evidence for whether the previous node actually had or obtained that information.
    - Do not flag 2.4 if the previous node response omitted information merely because the previous node logs show that the information was never obtained upstream.
    - Flag 1.4 only if important recent context was lost or dropped across the handoff.
    - Flag 2.1 only if the handoff effectively resets progress or discards established context.
    - Do not flag simply because the current node still has work left to do.
    - Do not flag simply because the previous response did not already contain the current node's final answer.
    - If no listed failure mode is clearly supported, return an empty flags list.

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return Markdown.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "2.4",
        "reason": "short reason grounded in the given evidence",
        "evidence_refs": ["task_intent", "prev_response", "prev_logs"]
        }}
    ]
    }}

    Current node id:
    {current_node_id}

    Previous node id:
    {prev_node_id}

    Current node task intent:
    {task_intent}

    Previous node response:
    {prev_response_text}

    Previous node logs:
    {prev_logs_text}
    """.strip()


    def _llm_eval_stage2_handoff_payload_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
        verification_cache: Dict[str, Any],
    ) -> FlagBundle:
        merged = FlagBundle(layer="handoff_payload")

        for node_id in index.ordered_node_ids:
            if node_id.startswith("V_") or node_id.startswith("R_"):
                continue

            current_node = index.by_node_id.get(node_id, {})
            if not isinstance(current_node, dict):
                continue

            prev_node_id = self.get_prev_node_id(index, node_id)
            if prev_node_id is None:
                continue

            prev_response = self.get_prev_response(index, node_id)
            prev_logs = self.get_prev_logs(index, node_id)

            if prev_response is None and prev_logs is None:
                continue

            task_intent = self._get_task_intent_for_node(
                node_id=node_id,
                current_node=current_node,
                verification_cache=verification_cache,
            )
            if not task_intent:
                continue

            subject = f"{node_id}<-prev"

            prompt = self._build_stage2_prompt(
                current_node_id=node_id,
                task_intent=task_intent,
                prev_node_id=prev_node_id,
                prev_response=prev_response,
                prev_logs=prev_logs,
            )

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage2_handoff_payload_{node_id}",
                layer="handoff_payload",
                subject=subject,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged
        
    def _build_stage3_prompt(
        self,
        current_node_id: str,
        prev_response: Any,
        current_logs: Any,
    ) -> str:
        prev_response_text = self._as_text(prev_response)
        current_logs_text = self._as_text(current_logs)

        return f"""
    You are a failure mode analysis judge.

    You are judging only whether the current node used useful information passed from the previous node.

    Failure mode definitions:
    2.5 Ignored Other Agent's Input: An agent fails to consider or appropriately act on another agent's useful input.

    Stage objective:
    - Compare the previous node response against the current node logs.
    - Judge only this failure mode:
    - 2.5 Ignored Other Agent's Input
    - In this stage, do not judge other failure modes.

    Judging rules:
    - Flag 2.5 only when the previous response contains clearly useful information for the current node,
    but the current logs do not show that the node considered or used that information.
    - If the current logs use the upstream information in paraphrased, transformed, or equivalent form,
    do not flag 2.5.
    - If the previous response does not contain clearly useful downstream input,
    do not flag 2.5.
    - Do not judge whether the upstream input was correct.
    - Do not judge whether the current node eventually succeeded.
    - If no relevant failure mode is supported, return an empty flags list.

    Few-shot examples:

    Example 1
    Input:
    response[i-1]:
    "The target site is MAIN and the relevant chillers are Chiller 4, Chiller 6, Chiller 9, and Chiller 3."

    logs[i]:
    "Thought 1: I need to identify the relevant assets.
    Action 1: assets
    Action Input 1: site_name=MAIN
    Observation 1: ...
    Thought 2: I will now inspect only Chiller 4 and Chiller 6.
    Action 2: sensors
    Action Input 2: site_name=MAIN, assetnum=Chiller 4
    Observation 2: ...
    Action 3: sensors
    Action Input 3: site_name=MAIN, assetnum=Chiller 6
    Observation 3: ..."

    Output:
    {{
    "flags": [
        {{
        "flag": "2.5",
        "reason": "The upstream response explicitly listed four relevant chillers, but the current logs only acted on Chiller 4 and Chiller 6 and show no use of Chiller 9 or Chiller 3.",
        "evidence_refs": ["prev_response", "current_logs"]
        }}
    ]
    }}



    Example 3
    Input:
    response[i-1]:
    "I made progress and found some preliminary context."

    logs[i]:
    "Thought 1: I still need the concrete asset and time range.
    Action 1: sites
    Action Input 1: {{}}
    Observation 1: {{\\"sites\\": [\\"MAIN\\"]}}
    Action 2: assets
    Action Input 2: site_name=MAIN
    Observation 2: ..."

    Output:
    {{
    "flags": []
    }}


    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return Markdown.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "2.5",
        "reason": "short reason grounded in the given evidence",
        "evidence_refs": ["prev_response", "current_logs"]
        }}
    ]
    }}

    Current node id:
    {current_node_id}

    Previous node response:
    {prev_response_text}

    Current node logs:
    {current_logs_text}
    """.strip()

    def _llm_eval_stage3_handoff_usage_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        merged = FlagBundle(layer="handoff_usage")

        for node_id in index.ordered_node_ids:
            if node_id.startswith("V_") or node_id.startswith("R_"):
                    continue

            prev_response = self.get_prev_response(index, node_id)
            if prev_response is None:
                continue

            current_node = index.by_node_id.get(node_id, {})
            if not isinstance(current_node, dict):
                continue

            current_logs = current_node.get("logs", {})

            prompt = self._build_stage3_prompt(
                current_node_id=node_id,
                prev_response=prev_response,
                current_logs=current_logs,
            )

            subject = f"{node_id}<-prev"

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage3_handoff_usage_{node_id}",
                layer="handoff_usage",
                subject=subject,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged

    def _build_stage4_prompt(
        self,
        s_node_id: str,
        s_logs: Any,
        # s_reviews: List[str],
        verifier_id: str,
        step_diagnoses: List[Dict[str, Any]],
    ) -> str:
        s_logs_text = self._as_text(s_logs)
        print(f"s_logs_text: {s_logs_text}")
        # s_reviews_text = self._as_text(s_reviews)
        step_diagnoses_text = self._as_text(step_diagnoses)

        return f"""
    You are a failure mode analysis judge.

    You will judge whether the verifier missed or misdetected failure signals that were already visible in the source node logs and reviews.

    Failure mode definitions:
    3.1 Failure Signal Miss or Misdetection: A verification stage fails to detect, or incorrectly detects, failure signals grounded in available observations such as logs, tool outputs, responses, or reviews.

    Stage objective:
    - Compare the source node logs and reviews against the verifier's step-level diagnoses.
    - Decide whether the verifier failed to detect, or incorrectly detected, failure signals grounded in the available observations.
    - Focus mainly on:
    - 3.1 Failure Signal Miss or Misdetection

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "3.1",
        "reason": "short reason grounded in the given evidence",
        "evidence_refs": ["source_logs", "source_reviews", "step_diagnoses"]
        }}
    ]
    }}

    Source node id:
    {s_node_id}

    Verifier node id:
    {verifier_id}

    Source node logs:
    {s_logs_text}

    Verifier step-level diagnoses:
    {step_diagnoses_text}
    """.strip()

    def _llm_eval_stage4_verifier_miss_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        merged = FlagBundle(layer="verifier_miss")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            if verifier_id is None:
                continue

            s_node = index.by_node_id.get(s_node_id, {})
            v_node = index.by_node_id.get(verifier_id, {})

            if not isinstance(s_node, dict) or not isinstance(v_node, dict):
                continue

            s_logs = self._extract_logs_for_stage4_source_only(s_node)
            s_logs = self._scrub_status_labels(s_logs)
            step_diagnoses = self._extract_verifier_step_diagnoses(v_node)

            prompt = self._build_stage4_prompt(
                s_node_id=s_node_id,
                s_logs=s_logs,
                verifier_id=verifier_id,
                step_diagnoses=step_diagnoses,
            )

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage4_verifier_miss_{s_node_id}",
                layer="verifier_miss",
                subject=verifier_id,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged

    def _scrub_status_labels(self, obj):
        if isinstance(obj, str):
            out = obj
            status_strings = [
                "Partially Accomplished",
                "Not Accomplished",
                "Accomplished",
                "Error",
            ]
            for label in status_strings:
                out = out.replace(label, "")
            return out

        if isinstance(obj, list):
            return [self._scrub_status_labels(x) for x in obj]

        if isinstance(obj, dict):
            return {k: self._scrub_status_labels(v) for k, v in obj.items()}

        return obj
    

    def _extract_logs_for_stage4_source_only(
        self,
        node_obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        logs = node_obj.get("logs", {})
        if not isinstance(logs, dict):
            return {}

        # shallow copy ではなく deep copy
        logs_copy: Dict[str, Any] = copy.deepcopy(logs)

        # 1. まず review / post-hoc 系は丸ごと落とす
        for key in [
            "reviews",
            "final_answer",
            "reflections",
            "verification_logs",
            "recovery_suggestion_log",
            "fallback_error",
            "history",
            "scratchpad",
        ]:
            logs_copy.pop(key, None)

        # 2. task 内の injected context を切り落とす
        task = logs_copy.get("task")
        if isinstance(task, str):
            logs_copy["task"] = self._strip_injected_context_block(task)

        # 3. trajectory は thought を落として action/observation 側だけ残す
        trajectory = logs_copy.get("trajectory")
        if isinstance(trajectory, list):
            cleaned_trajectory: List[Dict[str, Any]] = []
            for step in trajectory:
                if not isinstance(step, dict):
                    continue

                cleaned_step: Dict[str, Any] = {}
                for key in ["action", "observation"]:
                    value = step.get(key)
                    if isinstance(value, str):
                        cleaned_step[key] = self._strip_injected_context_block(value)
                    elif value is not None:
                        cleaned_step[key] = value

                if cleaned_step:
                    cleaned_trajectory.append(cleaned_step)

            logs_copy["trajectory"] = cleaned_trajectory

        # 4. trajectroy_log は thought 系を落として execution evidence だけ残す
        trajectroy_log = logs_copy.get("trajectroy_log")
        if isinstance(trajectroy_log, list):
            cleaned_tlog: List[Dict[str, Any]] = []
            for step in trajectroy_log:
                if not isinstance(step, dict):
                    continue

                cleaned_step: Dict[str, Any] = {}
                for key in [
                    "step",
                    "action",
                    "action_input",
                    "observation",
                    "state",
                    "llm_error",
                    "llm_error_detail",
                ]:
                    value = step.get(key)
                    if isinstance(value, str):
                        cleaned_step[key] = self._strip_injected_context_block(value)
                    elif value is not None:
                        cleaned_step[key] = value

                if cleaned_step:
                    cleaned_tlog.append(cleaned_step)

            logs_copy["trajectroy_log"] = cleaned_tlog

        return logs_copy


    def _strip_injected_context_block(self, text: str) -> str:
        if not isinstance(text, str):
            return text

        out = text

        # まず明示的な Context ブロックを切る
        context_markers = [
            "\n\nContext:\n[",
            "\nContext:\n[",
            "Context:\n[",
        ]
        cut_positions = [out.find(marker) for marker in context_markers if out.find(marker) != -1]
        if cut_positions:
            out = out[:min(cut_positions)].rstrip()

        # 追加で status/review 語を軽く scrub
        for label in [
            "Partially Accomplished",
            "Not Accomplished",
            "Accomplished",
            "Error",
        ]:
            out = out.replace(label, "")

        # review 的な key が文字列化されて混ざった場合の軽い除去
        out = re.sub(r'"status"\s*:\s*".*?"', '"status": ""', out)
        out = re.sub(r'"reasoning"\s*:\s*".*?"', '"reasoning": ""', out)
        out = re.sub(r'"suggestions"\s*:\s*".*?"', '"suggestions": ""', out)

        return out.strip()


    def _build_stage5_prompt(
        self,
        verifier_id: str,
        step_diagnoses: List[Dict[str, Any]],
        node_diagnosis: Dict[str, Any],
    ) -> str:
        step_diagnoses_text = self._as_text(step_diagnoses)
        node_diagnosis_text = self._as_text(node_diagnosis)

        return f"""
    You are a failure mode analysis judge.

    You will judge whether node-level diagnosis preserves the important information already present in the verifier's step-level diagnoses.

    Failure mode definitions:
    3.2 Failure Root Not Isolated: A verification stage detects that something is wrong but does not isolate the earliest structural break or root failure at a useful granularity. 
    3.3 Failure Representation Breakdown: A verification stage fails to integrate detected failure signals into a coherent representation that preserves temporal order or causal relations.

    Stage objective:
    - Compare the verifier's step-level diagnoses against the verifier's node-level diagnosis.
    - Decide whether important failure information was lost, over-compressed, or improperly organized during aggregation.
    - Focus mainly on:
        - 3.2 Failure Root Not Isolated
        - 3.3 Failure Representation Breakdown
    - Do not use 3.1 here.
    - Do not use 3.4, 3.5, or 3.6 here. Those belong to the handoff from diagnosis to recovery.
    - If no relevant failure mode is supported, return an empty flags list.

    Judging rules:
    - Flag 3.2 if the step-level diagnoses contain a clear earliest structural break, root candidate, or useful failure isolation signal, but the node-level diagnosis fails to preserve that root failure at a useful granularity.
    - Flag 3.2 if the node-level diagnosis collapses a concrete step-level root candidate into something too vague to support later recovery.
    - Flag 3.3 if multiple step-level failure signals or evidence items are not integrated into a coherent node-level representation.
    - Flag 3.3 if temporal order, causal linkage, or supporting evidence visible in the step-level diagnoses disappears from the node-level diagnosis.
    - Do not flag anything unless the evidence supports it.

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return Markdown.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "3.2",
        "reason": "short reason grounded in the given evidence",
        "evidence_refs": ["step_diagnoses", "node_diagnosis"]
        }}
    ]
    }}

    Example 1

    Verifier step level diagnoses:
    - step 1: failed tool execution while reading required sensor file; marked as root candidate
    - step 3: failed because required sensor list was missing; marked as root candidate at step level
    - step 5: failed for the same missing prerequisite on another asset; marked as root candidate at step level
    - step 6: final answer relied on general knowledge rather than retrieved data

    Verifier node level diagnosis:
    - first structural break is step 1
    - steps 3 and 5 are propagation or downstream symptoms caused by missing prerequisite information after step 1
    - step 6 is final unsupported answer
    - supporting evidence for steps 1, 3, 5, and 6 is preserved
    - root, propagation, and symptom are explicitly separated

    Incorrect output:
    - flag 3.2 because steps 3 and 5 were root candidates at step level but became downstream symptoms at node level
    - flag 3.3 because the node level diagnosis supposedly lost temporal order and causal linkage

    Why this is wrong:
    - This should be no flag.
    - A step level root candidate is a local candidate for that step, not necessarily the earliest root for the whole node.
    - The node level diagnosis correctly isolates step 1 as the earliest structural break.
    - The node level diagnosis also preserves causal structure by representing steps 3 and 5 as propagation and step 6 as symptom.
    - Temporal order and supporting evidence are preserved, not lost.

    Correct output:
    {{"flags": []}}

    
    Example 2

    Verifier step level diagnoses:
    - step 2: authentication failure when calling the database tool; marked as root candidate
    - step 4: later retrieval failed because authentication was never restored
    - step 5: final answer unsupported because retrieval never happened

    Verifier node level diagnosis:
    - says only that the node had multiple issues during execution
    - does not name the authentication failure
    - does not identify which step was the earliest structural break

    Correct judgment:
    - flag 3.2

    Why:
    - The step level diagnosis contains a clear concrete root failure at a useful recovery granularity.
    - The node level diagnosis collapses that root into a vague summary and does not preserve the useful root isolation.

    Correct output:
    {{"flags": [{{"flag": "3.2", "reason": "The step level diagnoses isolate a concrete earliest root at step 2, but the node level diagnosis collapses it into a vague multi issue summary and does not preserve the root failure at a useful granularity.", "evidence_refs": ["step_diagnoses", "node_diagnosis"]}}]}}

    
    Example 3

    Verifier step level diagnoses:
    - step 1: wrong entity id selected
    - step 2: retrieval query used the wrong id
    - step 4: tool returned empty results because of the wrong id
    - step 5: final answer claimed success despite empty retrieval

    Verifier node level diagnosis:
    - mentions only that the final answer was unsupported
    - does not preserve that wrong entity selection caused the wrong query, which caused empty retrieval, which caused unsupported finalization

    Correct judgment:
    - flag 3.3

    Why:
    - The step level diagnoses contain a multi step failure chain with clear temporal order and causal linkage.
    - The node level diagnosis keeps only the end symptom and loses the structured representation of the chain.

    Correct output:
    {{"flags": [{{"flag": "3.3", "reason": "The step level diagnoses contain a coherent causal chain from wrong entity selection to wrong query to empty retrieval to unsupported finalization, but the node level diagnosis keeps only the end symptom and loses the temporal and causal structure.", "evidence_refs": ["step_diagnoses", "node_diagnosis"]}}]}}


    Verifier node id:
    {verifier_id}

    Verifier step-level diagnoses:
    {step_diagnoses_text}

    Verifier node-level diagnosis:
    {node_diagnosis_text}
    """.strip()


    def _llm_eval_stage5_aggregation_loss_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        merged = FlagBundle(layer="aggregation_loss")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            if verifier_id is None:
                continue

            v_node = index.by_node_id.get(verifier_id, {})
            if not isinstance(v_node, dict):
                continue

            step_diagnoses = self._extract_verifier_step_diagnoses(v_node)
            node_diagnosis = self._extract_verifier_node_diagnosis(v_node)

            prompt = self._build_stage5_prompt(
                verifier_id=verifier_id,
                step_diagnoses=step_diagnoses,
                node_diagnosis=node_diagnosis,
            )

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage5_aggregation_loss_{s_node_id}",
                layer="aggregation_loss",
                subject=verifier_id,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged


    def _build_stage6_prompt(
        self,
        verifier_id: str,
        node_diagnosis: Dict[str, Any],
        recovery_suggestion: Dict[str, Any],
    ) -> str:
        node_diagnosis_text = self._as_text(node_diagnosis)
        recovery_suggestion_text = self._as_text(recovery_suggestion)

        return f"""
    You are a failure mode analysis judge.

    You will judge whether the node-level diagnosis is handed off to recovery in a form that is usable for effective recovery.

    Failure mode definitions:
    3.4 Diagnosis Compression Mismatch: The diagnosis passed to recovery is too compressed or too detailed for effective recovery use. 
    3.5 Unsupported Fault Hypothesis: The diagnosis does not support a grounded fault hypothesis that recovery can use. 
    3.6 Missing Upstream Repair Signal: The diagnosis fails to indicate that the fault originates upstream and requires upstream repair, escalation, or stopping rather than local retry.

    Stage objective:
    - Compare the verifier's node-level diagnosis against the verifier's recovery suggestion.
    - Decide whether the recovery suggestion is properly grounded in the diagnosis and whether important recovery-relevant information is missing or distorted.
    - Focus mainly on:
    - 3.4 Diagnosis Compression Mismatch
    - 3.5 Unsupported Fault Hypothesis
    - 3.6 Missing Upstream Repair Signal
    - Do not use 3.1, 3.2, or 3.3 here. Those belong to earlier verification stages.
    - Do not use 4.x here. Those belong to recovery execution.
    - If no relevant failure mode is supported, return an empty flags list.

    Judging rules:
    - Flag 3.4 if the diagnosis is handed to recovery in a form that is too compressed or too detailed to be directly useful for recovery.
    - Flag 3.4 if the recovery suggestion collapses an important diagnosed distinction that recovery would need.
    - Flag 3.5 if the recovery suggestion's fault hypothesis is not grounded in the node-level diagnosis.
    - Flag 3.5 if the recovery suggestion merely restates symptoms without a usable causal hypothesis supported by the diagnosis.
    - Flag 3.6 if the diagnosis indicates that the problem originates upstream, requires escalation, or requires stopping, but the recovery suggestion fails to signal that.
    - Do not flag anything unless the evidence supports it.

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return Markdown.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "3.5",
        "reason": "short reason grounded in the given evidence",
        "evidence_refs": ["node_diagnosis", "recovery_suggestion"]
        }}
    ]
    }}

    Verifier node id:
    {verifier_id}

    Verifier node-level diagnosis:
    {node_diagnosis_text}

    Verifier recovery suggestion:
    {recovery_suggestion_text}
    """.strip()


    def _llm_eval_stage6_diagnosis_to_recovery_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        merged = FlagBundle(layer="diagnosis_to_recovery")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            if verifier_id is None:
                continue

            v_node = index.by_node_id.get(verifier_id, {})
            if not isinstance(v_node, dict):
                continue

            node_diagnosis = self._extract_verifier_node_diagnosis(v_node)
            recovery_suggestion = self._extract_verifier_recovery_suggestion(v_node)

            prompt = self._build_stage6_prompt(
                verifier_id=verifier_id,
                node_diagnosis=node_diagnosis,
                recovery_suggestion=recovery_suggestion,
            )

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage6_diagnosis_to_recovery_{s_node_id}",
                layer="diagnosis_to_recovery",
                subject=verifier_id,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged


    def _build_stage7_prompt(
        self,
        verifier_id: str,
        recovery_id: str,
        recovery_suggestion: Dict[str, Any],
        recovery_logs: Any,
        recovery_response: Any,
    ) -> str:
        recovery_suggestion_text = self._as_text(recovery_suggestion)
        recovery_logs_text = self._as_text(recovery_logs)
        recovery_response_text = self._as_text(recovery_response)

        return f"""
    You are a failure mode analysis judge.

    You will judge whether the recovery node executed the handed-off recovery suggestion appropriately.

    Failure mode definitions:
    4.1 Fault Misidentification: The recovery stage identifies the wrong fault, or acts without correctly determining the fault. 
    4.2 Incorrect Probe Selection: The recovery stage chooses the wrong probe, test, or tool interaction for the current fault hypothesis. 
    4.3 Unsafe or Improper Termination: The recovery stage terminates too early, too late, or in a state that is not safe or not sufficiently consistent.

    Stage objective:
    - Compare the verifier's recovery suggestion against the recovery node's logs and response.
    - Decide whether recovery execution identified the right fault, used the right probe or action, and terminated safely.
    - Focus mainly on:
        - 4.1 Fault Misidentification
        - 4.2 Incorrect Probe Selection
        - 4.3 Unsafe or Improper Termination
    - Do not use 3.x here. Those belong to diagnosis and handoff stages.
    - If no relevant failure mode is supported, return an empty flags list.

    Judging rules:
    - Flag 4.1 if the recovery node appears to act on a different fault than the one described in the handed-off recovery suggestion.
    - Flag 4.1 if the recovery node ignores the primary fault hypothesis and instead pursues an unrelated failure explanation.
    - Flag 4.2 if the recovery node chooses probes, tools, or test actions that do not fit the handed-off fault hypothesis or recommended next actions.
    - Flag 4.2 if the recovery node does not meaningfully follow the handed-off recovery actions when those actions are relevant and feasible.
    - Flag 4.3 if the recovery node terminates with unresolved failure signals, unsupported assumptions, malformed finalization, or an unsafe/inconsistent state.
    - Do not flag anything unless the evidence supports it.

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return Markdown.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "4.1",
        "reason": "short reason grounded in the given evidence",
        "evidence_refs": ["recovery_suggestion", "recovery_logs", "recovery_response"]
        }}
    ]
    }}

    Verifier node id:
    {verifier_id}

    Recovery node id:
    {recovery_id}

    Handed-off recovery suggestion:
    {recovery_suggestion_text}

    Recovery node logs:
    {recovery_logs_text}

    Recovery node response:
    {recovery_response_text}
    """.strip()


    def _llm_eval_stage7_recovery_execution_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        merged = FlagBundle(layer="recovery_execution")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            recovery_id = self.get_recovery_for_s(index, s_node_id)

            if verifier_id is None or recovery_id is None:
                continue

            v_node = index.by_node_id.get(verifier_id, {})
            r_node = index.by_node_id.get(recovery_id, {})

            if not isinstance(v_node, dict) or not isinstance(r_node, dict):
                continue

            recovery_suggestion = self._extract_verifier_recovery_suggestion(v_node)
            recovery_goal = self._as_text(recovery_suggestion.get("recovery_goal")).strip().lower()

            if recovery_goal == "no recovery required":
                continue

            recovery_logs = r_node.get("logs", {})
            recovery_response = r_node.get("response")

            prompt = self._build_stage7_prompt(
                verifier_id=verifier_id,
                recovery_id=recovery_id,
                recovery_suggestion=recovery_suggestion,
                recovery_logs=recovery_logs,
                recovery_response=recovery_response,
            )

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage7_recovery_execution_{s_node_id}",
                layer="recovery_execution",
                subject=recovery_id,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged


    def _get_cached_verification_result(
        self,
        verification_cache: Dict[str, Any],
        s_node_id: str,
    ) -> Dict[str, Any]:
        nodes_cache = verification_cache.get("nodes", {})
        if not isinstance(nodes_cache, dict):
            return {}

        cached = nodes_cache.get(s_node_id, {})
        if not isinstance(cached, dict):
            return {}

        verification_result = cached.get("verification_result", {})
        return verification_result if isinstance(verification_result, dict) else {}


    def _get_cached_step_diagnoses(
        self,
        verification_cache: Dict[str, Any],
        s_node_id: str,
    ) -> List[Dict[str, Any]]:
        verification_result = self._get_cached_verification_result(
            verification_cache=verification_cache,
            s_node_id=s_node_id,
        )

        verification_logs = verification_result.get("verification_logs", {})
        if not isinstance(verification_logs, dict):
            return []

        step_analysis_logs = verification_logs.get("step_analysis_logs", [])
        if not isinstance(step_analysis_logs, list):
            return []

        parsed: List[Dict[str, Any]] = []
        for item in step_analysis_logs:
            if not isinstance(item, dict):
                continue

            maybe_parsed = item.get("parsed_step_diagnosis", {})
            if isinstance(maybe_parsed, dict) and maybe_parsed:
                parsed.append(maybe_parsed)
                continue

            maybe_result = item.get("parsed_verification_result", {})
            if isinstance(maybe_result, dict) and maybe_result:
                parsed.append(maybe_result)
                continue

        return parsed


    def _build_stage1_prompt(
        self,
        s_node_id: str,
        node_diagnosis: Dict[str, Any],
        step_diagnoses: List[Dict[str, Any]],
        source_logs: Any,
        source_response: Any,
        has_downstream_verifier: bool,
    ) -> str:
        node_diagnosis_text = self._as_text(node_diagnosis)
        step_diagnoses_text = self._as_text(step_diagnoses)
        source_logs_text = self._as_text(source_logs)
        source_response_text = self._as_text(source_response)

        verifier_presence_text = "yes" if has_downstream_verifier else "no"

        return f"""
    You are a failure mode analysis judge.

    You will judge which failure modes occurred at the source node itself, based primarily on the verifier's diagnosis of that node.

    Failure mode definitions:
    1.1 Disobey Task Specification:
    The agent or system fails to follow explicit task requirements, constraints, or instructions.

    1.2 Disobey Role Specification:
    The agent fails to behave according to its assigned role or responsibility.

    1.3 Step Repetition:
    The agent unnecessarily repeats a task, step, or phase that was already completed.

    1.4 Loss of Conversation History:
    The agent loses or ignores important recent context and reverts to an earlier state.

    1.5 Unaware of Termination Conditions:
    The agent fails to recognize when stopping conditions have been met or when progress is no longer possible.

    1.6 Premature Termination:
    Ending a task or conversation before the necessary information has been exchanged or objectives fully met.

    2.1 Conversation Reset:
    The interaction is unexpectedly restarted or reset, causing loss of progress or context.

    2.2 Fail to Ask for Clarification:
    The agent proceeds despite ambiguity or missing information that should have triggered clarification.

    2.3 Task Derailment:
    The agent deviates from the intended task and pursues irrelevant or unproductive actions.

    2.6 Action-Reasoning Mismatch:
    The agent's reasoning and actual action contradict each other.

    3.1 Failure Signal Miss or Misdetection:
    A verification stage fails to detect, or incorrectly detects, failure signals grounded in available observations such as logs, tool outputs, or responses.

    3.2 Failure Root Not Isolated:
    A verification stage detects that something is wrong but does not isolate the earliest structural break or root failure at a useful granularity.

    3.3 Failure Representation Breakdown:
    A verification stage fails to integrate detected failure signals into a coherent representation that preserves temporal order or causal relations.
    
    In this stage only, use 3.1, 3.2 and 3.3 only for the special case where a major source-node failure exists but no downstream virification step or verifier node is present to detect it.

    Stage objective:
    - This is Stage 1: node FMA.
    - Judge only source-node failures.
    - Use the verifier's node-level diagnosis as the primary evidence.
    - Use step-level diagnoses as supporting evidence.
    - Use source logs and source response only as supporting context.
    - In this stage, consider only:
        - 1.1 Disobey Task Specification
        - 1.2 Disobey Role Specification
        - 1.3 Step Repetition
        - 1.4 Loss of Conversation History
        - 1.5 Unaware of Termination Conditions
        - 1.6 Premature Termination
        - 2.1 Conversation Reset
        - 2.2 Fail to Ask for Clarification
        - 2.3 Task Derailment
        - 2.6 Action-Reasoning Mismatch
    - Also consider 3.1 only in this special Stage 1 case:
        - if a major failure exists at this source node
        - and no verification step about it in steps
        - also no downstream virification step or verifier node V_Si exists.
        
    Definition for the additional 3.1 condition in this step:
    - A major failure means the diagnosis indicates a serious unresolved failure, such as:
    - root_failure.category is not "none"
    - impact_on_node_contract indicates the node contract was not satisfied, was blocked, or remained unsupported
    - the failure timeline or supporting evidence indicates tool failure, invalid action, unsupported assumption, malformed finalization, missing prerequisite, or another contract-blocking failure
    - If such a major failure exists and downstream verifier presence is "no", flag 3.1 in this step with a reason that the node had a major failure but no downstream verification step and verifier node existed.

    Judging rules:
    - Base your judgment mainly on the diagnosis and step-level diagnoses.
    - Do not invent failures that are not supported by the diagnosis evidence.
    - Only flag a failure mode if the diagnosis meaningfully supports it.
    - If no listed failure mode is supported, return an empty flags list.
    - Flag 1.6 if the node ends, finishes, or returns an answer before the necessary information exchange or task objectives were completed.

    Additional rules for avoiding over-flagging
    1. Do not flag a substantive failure mode only because the output contains extra scaffold text, explanatory prose, or formatting artifacts such as LaTeX.
    2. If the core task output is still semantically correct and remains usable for downstream execution, treat the issue as formatting-only noise unless the stage explicitly judges output formatting.
    3. In node-level FMA, extra scaffold text alone does not justify 1.1 or 2.6 unless it makes the required output unusable, changes the semantic content, or clearly contradicts the node's reasoning.

    
    Example 1
    Observed source response:
    "The date range for the last week of April '20 is from April 27, 2020, to April 30, 2020.  (END OF FEEDBACK)  Now, here's the input question: Question"

    Observed finalization step:
    Action Input 3:
    "The date range for the last week of April '20 is from April 27, 2020, to April 30, 2020.  (END OF FEEDBACK)  Now, here's the input question: Question"

    Incorrect judgment:
    - flag 2.6 because the final answer includes extra scaffold text and a prompt fragment

    Why this is wrong:
    - The core semantic answer is still correct: "April 27, 2020, to April 30, 2020."
    - The issue is leaked scaffold text in finalization.
    - This is not automatically a contradiction between reasoning and action.
    - If the core answer remains correct and usable, extra scaffold text alone should not be escalated to 2.6.

    Example 2
    Observed source response:
    ["The date range for the last week of April '20 is from April 27, 2020, to April 30, 2020.  (END OF FEEDBACK)  Now, here's the input question: Question", {{"status": "Accomplished", "reasoning": "The agent successfully determined the date range for the last week of April '20 by identifying the correct start and end dates. The agent's reasoning was logical and accurate, as it considered the week that ends on or before April 30, 2020. The final answer provided is correct and relevant to the task.", "suggestions": null}}]

    Observed finalization step:
    Action Input 3:
    "The date range for the last week of April '20 is from April 27, 2020, to April 30, 2020.  (END OF FEEDBACK)  Now, here's the input question: Question"

    Verifier node-level diagnosis:
    {{
    "completion_condition": "The node must output a clean final answer containing only the date range for the last week of April 2020, with no extra scaffold or feedback text.",
    "root_failure": {{
        "category": "Malformed or Unsupported Finalization",
        "where": ["3"]
    }},
    "supporting_evidence": [
        {{
        "step": "3",
        "evidence_type": "Malformed or Unsupported Finalization",
        "snippet": "(END OF FEEDBACK)  Now, here's the input question: Question"
        }}
    ],
    "usable_outputs": [
        {{
        "name": "date_range",
        "value": "April 27, 2020 to April 30, 2020",
        "usable": true
        }}
    ],
    "unusable_outputs": [
        {{
        "name": "final_answer",
        "value": "The date range for the last week of April '20 is from April 27, 2020, to April 30, 2020.  (END OF FEEDBACK)  Now, here's the input question: Question",
        "usable": false
        }}
    ],
    "impact_on_node_contract": "Node contract not satisfied because the final answer is malformed despite containing the correct date range."
    }}

    Incorrect judgment:
    - flag 1.1 because the final answer contains extra scaffold text and a prompt fragment

    Why this is wrong:
    - The core semantic task result is still correct: the node identified the correct date range.
    - The problem is output contamination in finalization, not substantive failure to determine the date range.
    - A verifier may describe the final answer as malformed, but that alone should not be mapped to 1.1 in node-level FMA when the underlying task result remains correct and usable.
    - Do not convert formatting-only pollution into 1.1 unless the added text changes the semantic answer, makes the result unusable for downstream execution, or shows that the agent failed to solve the actual task.

    Correct output:
    {{"flags": []}}


    Example 3

    Observed source response:
    "The sensor data for Chiller 6 ... is available in file /tmp/cbmdir/0533cf56-25e2-4da1-b2ed-372f11c92dbd.json ... The final answer is: $\\boxed{{/tmp/cbmdir/0533cf56-25e2-4da1-b2ed-372f11c92dbd.json}}$"

    Verifier diagnosis:
    - usable output: "/tmp/cbmdir/0533cf56-25e2-4da1-b2ed-372f11c92dbd.json"
    - core requirement satisfied
    - final answer formatting is malformed because of extra explanation and LaTeX

    Incorrect judgment:
    - flag 1.1 because the output is not a concise file path only answer

    Why this is wrong:
    - The node completed the substantive task and returned the correct usable file path.
    - Extra explanation or LaTeX formatting alone does not mean the task specification was disobeyed.
    - Do not flag 1.1 when the core required artifact is correct and usable.

    Correct output:
    {{"flags": []}}


    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return Markdown.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "flags": [
        {{
        "flag": "1.3",
        "reason": "short reason grounded in the diagnosis evidence",
        "evidence_refs": ["node_diagnosis", "step_diagnoses"]
        }}
    ]
    }}

    Source node id:
    {s_node_id}

    Downstream verifier node exists:
    {verifier_presence_text}

    Verifier node-level diagnosis:
    {node_diagnosis_text}

    Verifier step-level diagnoses:
    {step_diagnoses_text}

    Source node logs:
    {source_logs_text}

    Source node response:
    {source_response_text}
    """.strip()


    def _llm_eval_stage1_node_fma_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
        verification_cache: Dict[str, Any],
    ) -> FlagBundle:
        merged = FlagBundle(layer="node_fma")

        for s_node_id in self.get_s_node_ids(index):
            s_node = index.by_node_id.get(s_node_id, {})
            if not isinstance(s_node, dict):
                continue

            node_diagnosis = self._get_cached_diagnosis(
                verification_cache=verification_cache,
                s_node_id=s_node_id,
            )
            step_diagnoses = self._get_cached_step_diagnoses(
                verification_cache=verification_cache,
                s_node_id=s_node_id,
            )

            source_logs = s_node.get("logs", {})
            source_response = s_node.get("response")
            has_downstream_verifier = self.get_verifier_for_s(index, s_node_id) is not None

            prompt = self._build_stage1_prompt(
                s_node_id=s_node_id,
                node_diagnosis=node_diagnosis,
                step_diagnoses=step_diagnoses,
                source_logs=source_logs,
                source_response=source_response,
                has_downstream_verifier=has_downstream_verifier,
            )

            bundle = self._run_llm_flag_stage(
                qid=record.qid,
                stage_name=f"stage1_node_fma_{s_node_id}",
                layer="node_fma",
                subject=s_node_id,
                prompt=prompt,
            )

            for flag_record in bundle.flags:
                self._add_flag(
                    bundle=merged,
                    flag=flag_record.flag,
                    subject=flag_record.subject,
                    reason=flag_record.reason,
                    evidence_refs=flag_record.evidence_refs,
                )

        return merged

































    def eval_node_fma_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
        verification_cache: Dict[str, Any],
    ) -> FlagBundle:
        bundle = FlagBundle(layer="node_fma")

        for s_node_id in self.get_s_node_ids(index):
            s_node = index.by_node_id.get(s_node_id, {})
            diagnosis = self._get_cached_diagnosis(verification_cache, s_node_id)

            response_text = self._node_response_text(s_node)
            logs_text = self._node_logs_text(s_node)
            combined_text = f"{response_text}\n{logs_text}"

            root_failure = diagnosis.get("root_failure", {})
            root_category = str(root_failure.get("category", "")).strip().lower()
            impact = str(diagnosis.get("impact_on_node_contract", "")).strip().lower()

            if self._contains_any(
                combined_text,
                ["(END OF FEEDBACK)", "Now, here is the task:", "Question:"],
            ):
                self._add_flag(
                    bundle,
                    "1.1",
                    s_node_id,
                    "final output contains leaked scaffold or prompt fragment",
                    [f"{s_node_id}.response"],
                )

            if self._contains_any(
                combined_text,
                [
                    "I will assume",
                    "use my knowledge",
                    "general knowledge",
                    "logical answer based on the information I have",
                ],
            ) and self._contains_any(
                combined_text,
                ["Final Answer", "the final answer is", "Finish"],
            ):
                self._add_flag(
                    bundle,
                    "1.5",
                    s_node_id,
                    "node finalized after switching to unsupported assumption-based answering",
                    [f"{s_node_id}.logs", f"{s_node_id}.response"],
                )

            if self._contains_any(
                combined_text,
                [
                    "Execution failed while calling the Tool",
                    "Error encountered",
                    "Invalid Action",
                ],
            ) and self._contains_any(
                combined_text,
                ["Final Answer", "the final answer is", "Finish"],
            ):
                self._add_flag(
                    bundle,
                    "2.6",
                    s_node_id,
                    "node proceeded to finalization after tool failure without restoring prerequisites",
                    [f"{s_node_id}.logs"],
                )

            action_names = self._extract_action_names_from_node(s_node)
            repeated_actions = self._find_repeated_actions(action_names)
            if repeated_actions:
                self._add_flag(
                    bundle,
                    "1.3",
                    s_node_id,
                    f"repeated action pattern detected: {', '.join(repeated_actions)}",
                    [f"{s_node_id}.logs"],
                )

            if root_category and root_category != "none":
                if "partially" in impact or "could not" in impact or "not supported" in combined_text.lower():
                    self._add_flag(
                        bundle,
                        "1.5",
                        s_node_id,
                        "node appears to terminate despite unresolved root failure",
                        [f"{s_node_id}.diagnosis"],
                    )

                verifier_id = self.get_verifier_for_s(index, s_node_id)
                if verifier_id is None:
                    self._add_flag(
                        bundle,
                        "3.1",
                        s_node_id,
                        "major failure exists but no downstream virification step or verifier node was found",
                        [f"{s_node_id}.diagnosis"],
                    )

        return bundle

    def eval_handoff_payload_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
        verification_cache: Dict[str, Any],
    ) -> FlagBundle:
        bundle = FlagBundle(layer="handoff_payload")

        for node_id in index.ordered_node_ids:
            prev_response = self.get_prev_response(index, node_id)
            if prev_response is None:
                continue

            current_node = index.by_node_id.get(node_id, {})
            prev_response_text = self._as_text(prev_response)

            task_intent = self._get_task_intent_for_node(node_id, current_node, verification_cache)
            if not task_intent:
                continue

            if not prev_response_text.strip():
                self._add_flag(
                    bundle,
                    "2.4",
                    f"{node_id}<-prev",
                    "upstream response is empty while downstream node has a task intent",
                    [f"{node_id}.task_intent"],
                )
                continue

            if self._contains_any(
                prev_response_text,
                ["(END OF FEEDBACK)", "Now, here is the task:", "Question:"],
            ):
                self._add_flag(
                    bundle,
                    "1.4",
                    f"{node_id}<-prev",
                    "upstream payload contains leaked scaffold and weak context preservation",
                    [f"{node_id}.prev_response"],
                )
                self._add_flag(
                    bundle,
                    "2.1",
                    f"{node_id}<-prev",
                    "upstream payload suggests conversational reset or injected task restart text",
                    [f"{node_id}.prev_response"],
                )

            intent_keywords = self._extract_keywords(task_intent)
            response_keywords = self._extract_keywords(prev_response_text)
            overlap = self._keyword_overlap_ratio(intent_keywords, response_keywords)

            if intent_keywords and overlap < 0.15:
                self._add_flag(
                    bundle,
                    "2.4",
                    f"{node_id}<-prev",
                    "upstream response does not appear to hand off information required by downstream task intent",
                    [f"{node_id}.task_intent", f"{node_id}.prev_response"],
                )

        return bundle

    def eval_handoff_usage_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        bundle = FlagBundle(layer="handoff_usage")

        for node_id in index.ordered_node_ids:
            prev_response = self.get_prev_response(index, node_id)
            if prev_response is None:
                continue

            current_node = index.by_node_id.get(node_id, {})
            prev_response_text = self._as_text(prev_response)
            current_logs_text = self._node_logs_text(current_node)

            prev_keywords = self._extract_keywords(prev_response_text)
            current_keywords = self._extract_keywords(current_logs_text)
            overlap = self._keyword_overlap_ratio(prev_keywords, current_keywords)

            if prev_keywords and overlap < 0.10:
                self._add_flag(
                    bundle,
                    "2.5",
                    f"{node_id}<-prev",
                    "downstream logs do not show clear use of information contained in the upstream response",
                    [f"{node_id}.prev_response", f"{node_id}.logs"],
                )

        return bundle

    def eval_verifier_miss_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        bundle = FlagBundle(layer="verifier_miss")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            if verifier_id is None:
                continue

            s_node = index.by_node_id.get(s_node_id, {})
            v_node = index.by_node_id.get(verifier_id, {})

            s_logs_text = self._node_logs_text(s_node)
            source_has_obvious_failure = self._contains_any(
                s_logs_text,
                [
                    "Execution failed while calling the Tool",
                    "Error encountered",
                    "Invalid Action",
                    "Sensor list is empty",
                    "(END OF FEEDBACK)",
                    "general knowledge",
                    "I will assume",
                ],
            )

            v_diagnosis = self._extract_verifier_node_diagnosis(v_node)
            v_detected_failure = self._diagnosis_has_failure(v_diagnosis)

            if source_has_obvious_failure and not v_detected_failure:
                self._add_flag(
                    bundle,
                    "3.1",
                    verifier_id,
                    "verifier appears to miss failure signals that are visible in the source node logs",
                    [f"{s_node_id}.logs", f"{verifier_id}.diagnosis"],
                )

        return bundle

    def eval_aggregation_loss_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        bundle = FlagBundle(layer="aggregation_loss")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            if verifier_id is None:
                continue

            v_node = index.by_node_id.get(verifier_id, {})
            step_diagnoses = self._extract_verifier_step_diagnoses(v_node)
            node_diagnosis = self._extract_verifier_node_diagnosis(v_node)

            step_has_failure = any(
                self._step_diag_has_failure(step_diag) for step_diag in step_diagnoses
            )
            node_has_failure = self._diagnosis_has_failure(node_diagnosis)

            if step_has_failure and not node_has_failure:
                self._add_flag(
                    bundle,
                    "3.2",
                    verifier_id,
                    "step-level diagnosis contains failure signal but node-level diagnosis does not preserve it",
                    [f"{verifier_id}.step_diagnosis", f"{verifier_id}.node_diagnosis"],
                )

            step_evidence_count = sum(
                len(step_diag.get("evidence", []))
                for step_diag in step_diagnoses
                if isinstance(step_diag, dict)
            )
            node_evidence_count = len(node_diagnosis.get("supporting_evidence", []))

            if step_evidence_count > 0 and node_evidence_count == 0:
                self._add_flag(
                    bundle,
                    "3.3",
                    verifier_id,
                    "supporting evidence present at step level disappears in node-level diagnosis",
                    [f"{verifier_id}.step_diagnosis", f"{verifier_id}.node_diagnosis"],
                )

            step_root_candidates = [
                step_diag
                for step_diag in step_diagnoses
                if isinstance(step_diag, dict) and bool(step_diag.get("is_root_candidate"))
            ]
            node_root = str(node_diagnosis.get("root_failure", {}).get("category", "")).strip().lower()

            if step_root_candidates and (not node_root or node_root == "none"):
                self._add_flag(
                    bundle,
                    "3.2",
                    verifier_id,
                    "root-candidate information from step level is absent in node-level diagnosis",
                    [f"{verifier_id}.step_diagnosis", f"{verifier_id}.node_diagnosis"],
                )

        return bundle

    def eval_diagnosis_to_recovery_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        bundle = FlagBundle(layer="diagnosis_to_recovery")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            if verifier_id is None:
                continue

            v_node = index.by_node_id.get(verifier_id, {})
            diagnosis = self._extract_verifier_node_diagnosis(v_node)
            suggestion = self._extract_verifier_recovery_suggestion(v_node)

            root_category = str(diagnosis.get("root_failure", {}).get("category", "")).strip().lower()
            has_root_failure = bool(root_category and root_category != "none")

            recovery_goal = self._as_text(suggestion.get("recovery_goal"))
            primary_fault = self._as_text(suggestion.get("primary_fault_hypothesis"))
            next_actions = suggestion.get("recommended_next_actions", [])
            if not isinstance(next_actions, list):
                next_actions = []

            diagnosis_text = self._diagnosis_to_text(diagnosis)

            if has_root_failure and recovery_goal.strip().lower() == "no recovery required":
                self._add_flag(
                    bundle,
                    "3.4",
                    verifier_id,
                    "node diagnosis indicates failure but recovery suggestion collapses to 'no recovery required'",
                    [f"{verifier_id}.diagnosis", f"{verifier_id}.recovery_suggestion"],
                )

            if has_root_failure and not primary_fault.strip():
                self._add_flag(
                    bundle,
                    "3.5",
                    verifier_id,
                    "recovery suggestion omits primary fault hypothesis despite diagnosed root failure",
                    [f"{verifier_id}.diagnosis", f"{verifier_id}.recovery_suggestion"],
                )

            if has_root_failure and not next_actions:
                self._add_flag(
                    bundle,
                    "3.6",
                    verifier_id,
                    "recovery suggestion omits concrete next actions despite diagnosed root failure",
                    [f"{verifier_id}.diagnosis", f"{verifier_id}.recovery_suggestion"],
                )

            if primary_fault.strip():
                diagnosis_keywords = self._extract_keywords(diagnosis_text)
                hypothesis_keywords = self._extract_keywords(primary_fault)
                overlap = self._keyword_overlap_ratio(hypothesis_keywords, diagnosis_keywords)

                if hypothesis_keywords and overlap < 0.20:
                    self._add_flag(
                        bundle,
                        "3.5",
                        verifier_id,
                        "primary fault hypothesis appears weakly grounded in the diagnosis content",
                        [f"{verifier_id}.diagnosis", f"{verifier_id}.recovery_suggestion"],
                    )

            if next_actions:
                action_text = "\n".join(self._as_text(x) for x in next_actions)
                diagnosis_keywords = self._extract_keywords(diagnosis_text)
                action_keywords = self._extract_keywords(action_text)
                overlap = self._keyword_overlap_ratio(action_keywords, diagnosis_keywords)

                if diagnosis_keywords and action_keywords and overlap < 0.15:
                    self._add_flag(
                        bundle,
                        "3.6",
                        verifier_id,
                        "recommended actions appear to miss important diagnostic content from upstream verification",
                        [f"{verifier_id}.diagnosis", f"{verifier_id}.recovery_suggestion"],
                    )

        return bundle

    def eval_recovery_execution_flags(
        self,
        record: TrajectoryRecord,
        index: TrajectoryIndex,
    ) -> FlagBundle:
        bundle = FlagBundle(layer="recovery_execution")

        for s_node_id in self.get_s_node_ids(index):
            verifier_id = self.get_verifier_for_s(index, s_node_id)
            recovery_id = self.get_recovery_for_s(index, s_node_id)

            if verifier_id is None or recovery_id is None:
                continue

            v_node = index.by_node_id.get(verifier_id, {})
            r_node = index.by_node_id.get(recovery_id, {})

            suggestion = self._extract_verifier_recovery_suggestion(v_node)
            recovery_goal = self._as_text(suggestion.get("recovery_goal"))
            primary_fault = self._as_text(suggestion.get("primary_fault_hypothesis"))
            next_actions = suggestion.get("recommended_next_actions", [])
            if not isinstance(next_actions, list):
                next_actions = []

            if recovery_goal.strip().lower() == "no recovery required":
                continue

            r_logs_text = self._node_logs_text(r_node)
            r_response_text = self._node_response_text(r_node)
            combined_text = f"{r_logs_text}\n{r_response_text}"

            if primary_fault.strip():
                hypothesis_keywords = self._extract_keywords(primary_fault)
                recovery_keywords = self._extract_keywords(combined_text)
                overlap = self._keyword_overlap_ratio(hypothesis_keywords, recovery_keywords)

                if hypothesis_keywords and overlap < 0.15:
                    self._add_flag(
                        bundle,
                        "4.1",
                        recovery_id,
                        "recovery execution does not appear to address the fault hypothesis handed off by the verifier",
                        [f"{verifier_id}.recovery_suggestion", f"{recovery_id}.logs"],
                    )

            if next_actions:
                action_text = "\n".join(self._as_text(x) for x in next_actions)
                action_keywords = self._extract_keywords(action_text)
                recovery_keywords = self._extract_keywords(combined_text)
                overlap = self._keyword_overlap_ratio(action_keywords, recovery_keywords)

                if action_keywords and overlap < 0.15:
                    self._add_flag(
                        bundle,
                        "4.2",
                        recovery_id,
                        "recovery execution does not appear to follow the suggested recovery actions",
                        [f"{verifier_id}.recovery_suggestion", f"{recovery_id}.logs"],
                    )

            if self._contains_any(
                combined_text,
                [
                    "Execution failed while calling the Tool",
                    "Error encountered",
                    "general knowledge",
                    "I will assume",
                    "(END OF FEEDBACK)",
                    "Now, here is the task:",
                ],
            ) and self._contains_any(
                combined_text,
                ["Final Answer", "the final answer is", "Finish"],
            ):
                self._add_flag(
                    bundle,
                    "4.3",
                    recovery_id,
                    "recovery node terminates with unresolved failure signals or unsupported finalization",
                    [f"{recovery_id}.logs", f"{recovery_id}.response"],
                )

        return bundle

    # ------------------------------------------------------------------
    # private helpers for evaluators
    # ------------------------------------------------------------------
    def _add_flag(
        self,
        bundle: FlagBundle,
        flag: str,
        subject: str,
        reason: str,
        evidence_refs: Optional[List[str]] = None,
    ) -> None:
        evidence_refs = evidence_refs or []

        for existing in bundle.flags:
            if existing.flag == flag and existing.subject == subject and existing.reason == reason:
                for ref in evidence_refs:
                    if ref not in existing.evidence_refs:
                        existing.evidence_refs.append(ref)
                return

        bundle.flags.append(
            FlagRecord(
                flag=flag,
                subject=subject,
                reason=reason,
                evidence_refs=list(evidence_refs),
            )
        )

    def _as_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _contains_any(self, text: str, needles: List[str]) -> bool:
        lowered = text.lower()
        return any(needle.lower() in lowered for needle in needles)

    def _extract_keywords(self, text: str) -> List[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "then", "than",
            "into", "over", "under", "node", "task", "logs", "final", "answer",
            "using", "used", "use", "tool", "step", "steps", "will", "have",
            "has", "had", "was", "were", "are", "is", "be", "been", "being",
            "into", "about", "because", "while", "despite", "after", "before",
            "through", "which", "their", "they", "them", "there", "here", "your",
            "agent", "failure", "recovery", "diagnosis",
        }
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", text.lower())
        return sorted({tok for tok in tokens if tok not in stopwords})

    def _keyword_overlap_ratio(self, left: List[str], right: List[str]) -> float:
        if not left or not right:
            return 0.0
        left_set = set(left)
        right_set = set(right)
        return len(left_set & right_set) / max(len(left_set), 1)

    def _node_response_text(self, node_obj: Dict[str, Any]) -> str:
        return self._as_text(node_obj.get("response"))

    def _node_logs_text(self, node_obj: Dict[str, Any]) -> str:
        return self._as_text(node_obj.get("logs"))

    def _get_cached_diagnosis(
        self,
        verification_cache: Dict[str, Any],
        s_node_id: str,
    ) -> Dict[str, Any]:
        nodes_cache = verification_cache.get("nodes", {})
        if not isinstance(nodes_cache, dict):
            return {}

        cached = nodes_cache.get(s_node_id, {})
        if not isinstance(cached, dict):
            return {}

        verification_result = cached.get("verification_result", {})
        if not isinstance(verification_result, dict):
            return {}

        diagnosis = verification_result.get("diagnosis", {})
        return diagnosis if isinstance(diagnosis, dict) else {}

    def _get_task_intent_for_node(
        self,
        node_id: str,
        current_node: Dict[str, Any],
        verification_cache: Dict[str, Any],
    ) -> str:
        if self._is_s_node(node_id):
            diagnosis = self._get_cached_diagnosis(verification_cache, node_id)
            task_intent = diagnosis.get("task_intent")
            if task_intent:
                return self._as_text(task_intent)

        if node_id.startswith("R_"):
            task_desc = current_node.get("task_description")
            return self._as_text(task_desc)

        task_desc = current_node.get("task_description")
        return self._as_text(task_desc)

    def _extract_action_names_from_node(self, node_obj: Dict[str, Any]) -> List[str]:
        logs = node_obj.get("logs", {})
        traj_log = logs.get("trajectroy_log", []) if isinstance(logs, dict) else []
        actions: List[str] = []

        if isinstance(traj_log, list):
            for step in traj_log:
                if not isinstance(step, dict):
                    continue
                action = str(step.get("action", "")).strip()
                if action:
                    actions.append(action)

        return actions

    def _find_repeated_actions(self, action_names: List[str]) -> List[str]:
        counts: Dict[str, int] = {}
        for action in action_names:
            counts[action] = counts.get(action, 0) + 1
        return sorted([name for name, count in counts.items() if count >= 2])

    def _extract_verifier_node_diagnosis(self, v_node: Dict[str, Any]) -> Dict[str, Any]:
        response = v_node.get("response", {})
        if isinstance(response, dict):
            diagnosis = response.get("diagnosis", {})
            if isinstance(diagnosis, dict):
                return diagnosis

        logs = v_node.get("logs", {})
        if isinstance(logs, dict):
            final_answer = logs.get("final_answer", {})
            if isinstance(final_answer, dict):
                diagnosis = final_answer.get("diagnosis", {})
                if isinstance(diagnosis, dict):
                    return diagnosis

        return {}

    def _extract_verifier_recovery_suggestion(self, v_node: Dict[str, Any]) -> Dict[str, Any]:
        response = v_node.get("response", {})
        if isinstance(response, dict):
            suggestion = response.get("recovery_suggestion", {})
            if isinstance(suggestion, dict):
                return suggestion

        logs = v_node.get("logs", {})
        if isinstance(logs, dict):
            final_answer = logs.get("final_answer", {})
            if isinstance(final_answer, dict):
                suggestion = final_answer.get("recovery_suggestion", {})
                if isinstance(suggestion, dict):
                    return suggestion

        return {}

    def _extract_verifier_step_diagnoses(self, v_node: Dict[str, Any]) -> List[Dict[str, Any]]:
        logs = v_node.get("logs", {})
        if not isinstance(logs, dict):
            return []

        verification_logs = logs.get("verification_logs", {})
        if not isinstance(verification_logs, dict):
            return []

        step_analysis_logs = verification_logs.get("step_analysis_logs", [])
        if not isinstance(step_analysis_logs, list):
            return []

        parsed: List[Dict[str, Any]] = []
        for item in step_analysis_logs:
            if not isinstance(item, dict):
                continue

            maybe_parsed = item.get("parsed_step_diagnosis", {})
            if isinstance(maybe_parsed, dict) and maybe_parsed:
                parsed.append(maybe_parsed)
                continue

            maybe_result = item.get("parsed_verification_result", {})
            if isinstance(maybe_result, dict) and maybe_result:
                parsed.append(maybe_result)
                continue

        return parsed

    def _step_diag_has_failure(self, step_diag: Dict[str, Any]) -> bool:
        step_status = str(step_diag.get("step_status", "")).strip().lower()
        if step_status in {"failure", "warning"}:
            return True

        if bool(step_diag.get("is_root_candidate")):
            return True

        evidence = step_diag.get("evidence", [])
        if isinstance(evidence, list) and len(evidence) > 0:
            return True

        return False

    def _diagnosis_has_failure(self, diagnosis: Dict[str, Any]) -> bool:
        if not isinstance(diagnosis, dict):
            return False

        root_failure = diagnosis.get("root_failure", {})
        if isinstance(root_failure, dict):
            category = str(root_failure.get("category", "")).strip().lower()
            if category and category != "none":
                return True

        failure_timeline = diagnosis.get("failure_timeline", [])
        if isinstance(failure_timeline, list):
            for item in failure_timeline:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("step_status", "")).strip().lower()
                if status in {"failure", "warning"}:
                    return True

        supporting_evidence = diagnosis.get("supporting_evidence", [])
        if isinstance(supporting_evidence, list) and supporting_evidence:
            return True

        return False

    def _diagnosis_to_text(self, diagnosis: Dict[str, Any]) -> str:
        parts: List[str] = []

        for key in [
            "task_intent",
            "completion_condition",
            "impact_on_node_contract",
            "diagnosis_confidence",
        ]:
            value = diagnosis.get(key)
            if value:
                parts.append(self._as_text(value))

        root_failure = diagnosis.get("root_failure", {})
        if isinstance(root_failure, dict):
            parts.append(self._as_text(root_failure))

        for key in [
            "failure_timeline",
            "failure_chain",
            "supporting_evidence",
            "downstream_symptoms",
            "usable_outputs",
            "unusable_outputs",
        ]:
            value = diagnosis.get(key)
            if value:
                parts.append(self._as_text(value))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 8. flag aggregation helpers
    # ------------------------------------------------------------------
    def merge_flag_bundles(self, bundles: List[FlagBundle]) -> List[FlagRecord]:
        merged: List[FlagRecord] = []
        seen: Dict[tuple[str, str, str], FlagRecord] = {}

        for bundle in bundles:
            for flag_record in bundle.flags:
                key = (flag_record.flag, flag_record.subject, flag_record.reason)

                if key not in seen:
                    copied = FlagRecord(
                        flag=flag_record.flag,
                        subject=flag_record.subject,
                        reason=flag_record.reason,
                        evidence_refs=list(flag_record.evidence_refs),
                    )
                    seen[key] = copied
                    merged.append(copied)
                    continue

                existing = seen[key]
                for ref in flag_record.evidence_refs:
                    if ref not in existing.evidence_refs:
                        existing.evidence_refs.append(ref)

        merged.sort(key=lambda x: (x.flag, x.subject, x.reason))
        return merged

    def build_final_report(
        self,
        record: TrajectoryRecord,
        bundles: List[FlagBundle],
    ) -> Dict[str, Any]:
        merged_flags = self.merge_flag_bundles(bundles)

        flags_by_layer: Dict[str, List[Dict[str, Any]]] = {}
        for bundle in bundles:
            flags_by_layer[bundle.layer] = [
                {
                    "flag": flag.flag,
                    "subject": flag.subject,
                    "reason": flag.reason,
                    "evidence_refs": list(flag.evidence_refs),
                }
                for flag in bundle.flags
            ]

        all_flags = [
            {
                "flag": flag.flag,
                "subject": flag.subject,
                "reason": flag.reason,
                "evidence_refs": list(flag.evidence_refs),
            }
            for flag in merged_flags
        ]

        flags_by_node: Dict[str, List[Dict[str, Any]]] = {}
        for flag in merged_flags:
            flags_by_node.setdefault(flag.subject, []).append(
                {
                    "flag": flag.flag,
                    "reason": flag.reason,
                    "evidence_refs": list(flag.evidence_refs),
                }
            )

        return {
            "qid": record.qid,
            "trajectory_path": str(record.trajectory_path),
            "layers": [bundle.layer for bundle in bundles],
            "flags_by_layer": flags_by_layer,
            "flags_by_node": flags_by_node,
            "all_flags": all_flags,
            "flag_count": len(all_flags),
        }

    # ------------------------------------------------------------------
    # 9. top-level runner
    # ------------------------------------------------------------------
    def run_fma_for_trajectory(self, trajectory_path: Path) -> Dict[str, Any]:
        record = self.load_trajectory(trajectory_path)
        index = self.build_trajectory_index(record)

        verification_cache = self.run_or_load_verification_for_s_nodes(
            record=record,
            index=index,
        )

        bundles: List[FlagBundle] = [
            self._llm_eval_stage1_node_fma_flags(record, index, verification_cache),
            self._llm_eval_stage2_handoff_payload_flags(record, index, verification_cache),
            self._llm_eval_stage3_handoff_usage_flags(record, index),
            self._llm_eval_stage4_verifier_miss_flags(record, index),
            self._llm_eval_stage5_aggregation_loss_flags(record, index),
            self._llm_eval_stage6_diagnosis_to_recovery_flags(record, index),
            self._llm_eval_stage7_recovery_execution_flags(record, index),
        ]

        final_report = self.build_final_report(record, bundles)

        report_path = self.build_flag_report_path(record.qid)
        self.write_json(report_path, final_report)

        return final_report

    def run_fma_for_dir(self, trajectory_dir: Path) -> List[Dict[str, Any]]:
        if not trajectory_dir.exists():
            raise FileNotFoundError(f"trajectory_dir does not exist: {trajectory_dir}")

        if not trajectory_dir.is_dir():
            raise NotADirectoryError(f"trajectory_dir is not a directory: {trajectory_dir}")

        reports: List[Dict[str, Any]] = []

        for trajectory_path in sorted(trajectory_dir.glob("*.json")):
            try:
                report = self.run_fma_for_trajectory(trajectory_path)
                reports.append(report)
            except Exception as exc:
                self.logger.exception(
                    "Failed to run FMA for trajectory: %s",
                    trajectory_path,
                )
                reports.append(
                    {
                        "qid": "",
                        "trajectory_path": str(trajectory_path),
                        "error": repr(exc),
                        "all_flags": [],
                        "flag_count": 0,
                    }
                )

        return reports
    




import argparse



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FailureModeAnalysis on one trajectory JSON file "
            "or on all trajectory JSON files in a directory."
        )
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--trajectory_path",
        type=str,
        default="",
        help="Path to one trajectory JSON file.",
    )
    target_group.add_argument(
        "--trajectory_dir",
        type=str,
        default="",
        help="Path to a directory containing trajectory JSON files.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory for intermediate FMA cache files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for final FMA report files.",
    )
    parser.add_argument("--llm_model", type=int, default=16)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--debug_json_caller",
        action="store_true",
        help="Enable debug logging in WatsonxJSONCaller.",
    )
    parser.add_argument(
        "--overwrite_verification",
        action="store_true",
        help="Rerun verification and ignore verification cache.",
    )
    parser.add_argument(
        "--overwrite_stage_cache",
        action="store_true",
        help="Rerun FMA stage caches (stage1-stage7) and ignore stage cache files.",
    )
    parser.add_argument(
        "--overwrite_stage_prefixes",
        nargs="*",
        default=[],
        help=(
            "Rerun only stage caches whose stage_name starts with one of these prefixes. "
            "Examples: stage4_ stage6_"
        ),
    )

    return parser.parse_args()

import inspect
import traceback

def _make_llm_generate(model_id: str):
    def _llm_generate(prompt: str) -> str:
        resp = watsonx_llm(prompt, model_id=model_id)

        # token accounting
        if isinstance(resp, dict):
            generated_text = resp.get("generated_text", None)
            if isinstance(generated_text, str):
                return generated_text

            raise ValueError(
                "watsonx_llm response dict does not contain a string 'generated_text'"
            )

        if isinstance(resp, str):
            return resp

        raise ValueError(f"Unexpected LLM response type: {type(resp)}")

    return _llm_generate

def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_generate = _make_llm_generate(args.llm_model)
    print(f"llm_generate: {llm_generate}", flush=True)

    verification_agent = VerificationAgent(
        llm_generate=llm_generate,
    )

    llm_json_caller = WatsonxJSONCaller(
        model_id=args.llm_model,
        debug=args.debug_json_caller,
    )
    print(f"llm_generate: {llm_generate}", flush=True)
        
    fma = FailureModeAnalysis(
        verification_agent=verification_agent,
        llm_json_caller=llm_json_caller,
        cache_dir=cache_dir,
        output_dir=output_dir,
        overwrite_verification=args.overwrite_verification,
        overwrite_stage_cache=args.overwrite_stage_cache,
        overwrite_stage_prefixes=args.overwrite_stage_prefixes,
    )
    print(f"llm_generate: {llm_generate}", flush=True)


    if args.trajectory_path:
        trajectory_path = Path(args.trajectory_path)
        report = fma.run_fma_for_trajectory(trajectory_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    
    # print(f"llm_generate: {llm_generate}", flush=True)

    trajectory_dir = Path(args.trajectory_dir)
    reports = fma.run_fma_for_dir(trajectory_dir)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())