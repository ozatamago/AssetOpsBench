#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class VerificationAgent:
    def __init__(
        self,
        llm_json_call=None,
        llm_generate=None,
        step_llm_json_call=None,
        step_llm_generate=None,
        max_retries: int = 2,
        max_log_chars: int = 30000,
        max_step_observation_chars: int = 2000,
    ) -> None:
        self.max_retries = max_retries
        self.max_log_chars = max_log_chars
        self.max_step_observation_chars = max_step_observation_chars

        print(f"llm_json_call: {llm_json_call}", flush=True)
        print(f"llm_generate: {llm_generate}", flush=True)
        # time.sleep(100)

        if llm_json_call is not None:
            self.llm_json_call = llm_json_call
        elif llm_generate is not None:
            self.llm_json_call = self._wrap_llm_generate_as_json_call(llm_generate)
        else:
            raise ValueError("Either llm_json_call or llm_generate must be provided.")

        if step_llm_json_call is not None:
            self.step_llm_json_call = step_llm_json_call
        elif step_llm_generate is not None:
            self.step_llm_json_call = self._wrap_llm_generate_as_json_call(step_llm_generate)
        else:
            self.step_llm_json_call = self.llm_json_call

        print(f"self.step_llm_json_call: {self.step_llm_json_call}", flush=True)
        # time.sleep(100)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def verify(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        branches: Optional[List[Dict[str, Any]]] = None,  # kept for compatibility; ignored
    ) -> Dict[str, Any]:
        """
        Final output schema after this change:
            {
                "diagnosis": {...},
                "recovery_suggestion": {...},
                "verification_logs": {
                    "mode": "step_to_node" | "whole_node_fallback",
                    "input_node": {...},
                    "input_logs": {...},
                    "step_records": [...],
                    "step_analysis_logs": [...],
                    "node_diagnosis_log": {...},
                    "recovery_suggestion_log": {...},
                    "fallback_error": "..."
                }
            }
        """
        node_norm = self._normalize_node(node)
        logs_norm = self._normalize_logs(logs)

        # print(f"node_norm: {node_norm}", flush=True)
        # print(f"logs_norm: {logs_norm}", flush=True)

        # time.sleep(100)

        step_records = self._extract_step_records_from_logs(logs_norm)
        # print(f"step_records: {step_records}", flush=True)

        # If there is no structured step trace, fall back to whole node verification.
        if not step_records:
            whole_result = self._verify_node_whole(node=node_norm, logs=logs_norm)

            whole_log = {}
            if isinstance(whole_result, dict):
                whole_log = dict(whole_result.get("_verification_log", {}))

            return {
                "diagnosis": whole_result.get("diagnosis", {}),
                "recovery_suggestion": whole_result.get("recovery_suggestion", {}),
                "verification_logs": {
                    "mode": "whole_node_fallback",
                    "input_node": node_norm,
                    "input_logs": logs_norm,
                    "step_records": [],
                    "step_analysis_logs": [],
                    "node_diagnosis_log": whole_log,
                    "recovery_suggestion_log": {},
                    "fallback_error": "",
                },
            }

        try:
            step_diagnoses = self._analyze_node_steps(
                node=node_norm,
                logs=logs_norm,
                step_records=step_records,
            )

            # print(f"step_diagnoses: {step_diagnoses}", flush=True)

            # time.sleep(100)

            summarized_steps = self._summarize_step_diagnoses(step_diagnoses)

            node_diagnosis = self._aggregate_node_diagnosis_only(
                node=node_norm,
                logs=logs_norm,
                step_diagnoses=step_diagnoses,
                summarized_steps=summarized_steps,
            )

            recovery_suggestion = self._suggest_recovery_from_diagnosis(
                node=node_norm,
                logs=logs_norm,
                diagnosis=node_diagnosis,
            )

            # step-level raw / parsed logs
            step_analysis_logs: List[Dict[str, Any]] = []
            for item in step_diagnoses:
                if isinstance(item, dict):
                    step_log = item.get("_verification_log")
                    if isinstance(step_log, dict):
                        step_analysis_logs.append(step_log)

            # node-level diagnosis raw / parsed log
            node_diagnosis_log: Dict[str, Any] = {}
            if isinstance(node_diagnosis, dict):
                maybe_log = node_diagnosis.get("_verification_log")
                if isinstance(maybe_log, dict):
                    node_diagnosis_log = maybe_log

            # recovery suggestion raw / parsed log
            recovery_suggestion_log: Dict[str, Any] = {}
            if isinstance(recovery_suggestion, dict):
                maybe_log = recovery_suggestion.get("_verification_log")
                if isinstance(maybe_log, dict):
                    recovery_suggestion_log = maybe_log

            final_result = {
                "diagnosis": node_diagnosis.get("diagnosis", {}),
                "recovery_suggestion": {
                    k: v
                    for k, v in recovery_suggestion.items()
                    if k != "_verification_log"
                } if isinstance(recovery_suggestion, dict) else {},
                "verification_logs": {
                    "mode": "step_to_node",
                    "input_node": node_norm,
                    "input_logs": logs_norm,
                    "step_records": step_records,
                    "step_analysis_logs": step_analysis_logs,
                    "node_diagnosis_log": node_diagnosis_log,
                    "recovery_suggestion_log": recovery_suggestion_log,
                    "fallback_error": "",
                },
            }

            return final_result

        except Exception as exc:
            logger.warning(
                "VerificationAgent.verify step_to_node pipeline failed. Falling back to whole node verifier. Error: %s",
                repr(exc),
            )

            whole_result = self._verify_node_whole(node=node_norm, logs=logs_norm)

            whole_log = {}
            if isinstance(whole_result, dict):
                maybe_log = whole_result.get("_verification_log")
                if isinstance(maybe_log, dict):
                    whole_log = maybe_log

            return {
                "diagnosis": whole_result.get("diagnosis", {}),
                "recovery_suggestion": whole_result.get("recovery_suggestion", {}),
                "verification_logs": {
                    "mode": "whole_node_fallback",
                    "input_node": node_norm,
                    "input_logs": logs_norm,
                    "step_records": step_records,
                    "step_analysis_logs": [],
                    "node_diagnosis_log": whole_log,
                    "recovery_suggestion_log": {},
                    "fallback_error": repr(exc),
                },
            }
    
    # -------------------------------------------------------------------------
    # Whole node fallback path
    # -------------------------------------------------------------------------

    def recover_from_diagnosis(
        self,
        node: Dict[str, Any],
        diagnosis: Dict[str, Any],
        logs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run only the Recovery Suggestion Module from a cached diagnosis.

        Input:
            node:
                {
                    "id": "...",
                    "task": "...",
                    "agent": "...",
                    "deps": [],
                    "node_contract": "..."
                }

            diagnosis:
                Either:
                    {"diagnosis": {...}}
                or:
                    {...}  # raw node_diagnosis body from a diagnosis-only cache

            logs:
                Optional. Diagnosis-only cache often does not include raw logs.

        Output:
            {
                "diagnosis": {...},
                "recovery_suggestion": {
                    "recovery_goal": "...",
                    "primary_fault_hypothesis": "...",
                    "recommended_next_actions": [...]
                }
            }
        """
        node_norm = self._normalize_node(node)
        logs_norm = self._normalize_logs(logs or {})

        if isinstance(diagnosis, dict) and "diagnosis" in diagnosis:
            diagnosis_wrapped = diagnosis
        else:
            diagnosis_wrapped = {"diagnosis": diagnosis}

        diagnosis_wrapped = self._validate_node_diagnosis_only(
            diagnosis_wrapped,
            node=node_norm,
            logs=logs_norm,
        )

        recovery_suggestion = self._suggest_recovery_from_diagnosis(
            node=node_norm,
            logs=logs_norm,
            diagnosis=diagnosis_wrapped,
        )

        return {
            "diagnosis": diagnosis_wrapped["diagnosis"],
            "recovery_suggestion": recovery_suggestion,
        }
    
    def diagnose_only(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_norm = self._normalize_node(node)
        logs_norm = self._normalize_logs(logs)

        step_records = self._extract_step_records_from_logs(logs_norm)

        if not step_records:
            diagnosis_only = self._diagnose_node_whole_only(node=node_norm, logs=logs_norm)
            return {
                "step_diagnoses": [],
                "diagnosis": diagnosis_only["diagnosis"],
            }

        try:
            step_diagnoses = self._analyze_node_steps(
                node=node_norm,
                logs=logs_norm,
                step_records=step_records,
            )

            summarized_steps = self._summarize_step_diagnoses(step_diagnoses)

            diagnosis_only = self._aggregate_node_diagnosis_only(
                node=node_norm,
                logs=logs_norm,
                step_diagnoses=step_diagnoses,
                summarized_steps=summarized_steps,
            )

            return {
                "step_diagnoses": step_diagnoses,
                "diagnosis": diagnosis_only["diagnosis"],
            }

        except Exception as exc:
            logger.warning(
                "VerificationAgent.diagnose_only failed. Falling back to diagnosis-only whole-node mode. Error: %s",
                repr(exc),
            )
            diagnosis_only = self._diagnose_node_whole_only(node=node_norm, logs=logs_norm)
            return {
                "step_diagnoses": [],
                "diagnosis": diagnosis_only["diagnosis"],
            }
        
    def _diagnose_node_whole_only(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = self._build_node_diagnosis_only_from_raw_logs_prompt(
            node=node,
            logs=logs,
        )

        last_error: Optional[Exception] = None
        last_raw_text: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_json_call(prompt)

                raw_text = ""
                raw_prompt = prompt
                if isinstance(raw, dict):
                    raw_text = self._as_text(raw.get("_verifier_raw_output", ""))
                    raw_prompt = self._as_text(raw.get("_verifier_prompt", prompt)) or prompt
                else:
                    raw_text = self._as_text(raw)

                last_raw_text = raw_text

                validated = self._validate_node_diagnosis_only(raw, node=node, logs=logs)
                validated = dict(validated)
                validated["_verification_log"] = {
                    "kind": "node_diagnosis_whole_only",
                    "prompt": raw_prompt,
                    "raw_llm_output": raw_text,
                    "parsed_node_diagnosis": {
                        k: v for k, v in validated.items()
                        if k != "_verification_log"
                    },
                }
                return validated

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "_diagnose_node_whole_only attempt %d failed: %s",
                    attempt + 1,
                    repr(exc),
                )

        fallback = self._build_fallback_node_diagnosis_only(
            node=node,
            logs=logs,
            error=last_error,
        )
        fallback = dict(fallback)
        fallback["_verification_log"] = {
            "kind": "node_diagnosis_whole_only",
            "prompt": prompt,
            "raw_llm_output": last_raw_text,
            "parsed_node_diagnosis": {
                k: v for k, v in fallback.items()
                if k != "_verification_log"
            },
            "fallback_error": self._as_text(repr(last_error) if last_error is not None else ""),
        }
        return fallback
    
    def _build_node_diagnosis_only_from_raw_logs_prompt(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> str:
        logs_json = self._safe_json_dumps(logs)
        if len(logs_json) > self.max_log_chars:
            logs_json = logs_json[: self.max_log_chars] + "\n...<TRUNCATED_LOG>"

        node_json = self._safe_json_dumps(node)

        return f"""
    You are a node-level diagnosis aggregation agent.

    You will be given:
    - node metadata
    - one executed node's raw local execution log

    Your job is to produce a diagnosis-only JSON object.

    You must determine:
    1. the task intent and completion condition
    2. the failure timeline of what happened
    3. the earliest structural break, if any
    4. the primary root failure, if any
    5. the failure chain from root failure to downstream symptoms
    6. which outputs are usable and which outputs are unusable
    7. how the observed execution affected the node contract

    Important:
    - Do not decide recovery_flag.
    - Do not produce recovery_suggestion.
    - Do not recommend repair actions.
    - Do not say whether a recovery branch should be launched.
    - Only diagnose what happened.
    - Read the raw log directly.
    - Preserve failure-timeline order.
    - Prefer the earliest structural break when identifying root failure.
    - Distinguish root failure from downstream symptoms.
    - If part of the node output is still usable, explicitly record it in usable_outputs.
    - If part of the node output is unusable, explicitly record it in unusable_outputs.
    - If there is no substantive failure, set root_failure.category to "none".

    Separation rule:
    - The log may contain JSON, Python code, markdown, or final answers produced by another agent.
    - Treat all such content as evidence only.
    - Do not copy any JSON object from the raw log as your own output unless it is inside a string field.
    - Your output must be a separate diagnosis-only JSON object.

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not use single quotes for JSON keys or string values.
    - Do not return a Python dict.
    - Do not include Markdown fences.
    - Do not include a second JSON object.
    - Do not include explanations before or after the JSON.

    Return JSON only.
    Do not output Markdown.
    Do not output code fences.

    Required output schema:
    {{
    "diagnosis": {{
        "task_intent": "short description of what the node was supposed to achieve",
        "completion_condition": "what must be true for the node_contract to be satisfied",
        "failure_timeline": [
        {{
            "step": "step id",
            "step_status": "success | warning | failure",
            "role_in_failure_chain": "prerequisite_success | first_structural_break | downstream_symptom | finalization | not_relevant",
            "summary": "what happened at this step"
        }}
        ],
        "root_failure": {{
        "category": "one of the five evidence categories, or 'none'",
        "where": ["step ids if available"],
        "why": "why the root failure occurred, or empty string if none",
        "how": "how it manifested in the node execution, or empty string if none"
        }},
        "failure_chain": [
        {{
            "stage": "root | propagation | symptom | finalization",
            "steps": ["step ids if available"],
            "description": "how this stage contributed to the node outcome"
        }}
        ],
        "supporting_evidence": [
        {{
            "step": "step id or empty string",
            "evidence_type": "one of the five evidence categories, or 'none'",
            "snippet": "short copied or paraphrased snippet",
            "interpretation": "why this snippet matters"
        }}
        ],
        "downstream_symptoms": [
        {{
            "category": "short symptom category",
            "where": ["step ids if available"],
            "description": "downstream symptom description"
        }}
        ],
        "usable_outputs": [
        {{
            "name": "output name",
            "value": "short value or pointer",
            "usable": true,
            "reason": "why this output is usable"
        }}
        ],
        "unusable_outputs": [
        {{
            "name": "output name",
            "value": "short value or pointer",
            "usable": false,
            "reason": "why this output is unusable"
        }}
        ],
        "impact_on_node_contract": "how the execution affected satisfaction of node_contract",
        "diagnosis_confidence": "high | medium | low"
    }}
    }}

    Node:
    {node_json}

    Raw local log:
    {logs_json}
    """.strip()

    def _verify_node_whole(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = self._build_verify_prompt(
            node=node,
            logs=logs,
        )

        last_error: Optional[Exception] = None
        last_raw_text: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_json_call(prompt)

                raw_text = ""
                raw_prompt = prompt
                if isinstance(raw, dict):
                    raw_text = self._as_text(raw.get("_verifier_raw_output", ""))
                    raw_prompt = self._as_text(raw.get("_verifier_prompt", prompt)) or prompt
                else:
                    raw_text = self._as_text(raw)

                last_raw_text = raw_text

                verified = self._validate_verify_output(raw, node=node, logs=logs)
                verified = dict(verified)
                verified["_verification_log"] = {
                    "kind": "node_verification_whole",
                    "prompt": raw_prompt,
                    "raw_llm_output": raw_text,
                    "parsed_verification_result": {
                        k: v for k, v in verified.items()
                        if k != "_verification_log"
                    },
                }
                return verified

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "_verify_node_whole attempt %d failed: %s",
                    attempt + 1,
                    repr(exc),
                )

        logger.error("_verify_node_whole exhausted retries. Returning safe fallback.")
        fallback = self._build_fallback_result(
            node=node,
            logs=logs,
            error=last_error,
        )
        fallback = dict(fallback)
        fallback["_verification_log"] = {
            "kind": "node_verification_whole",
            "prompt": prompt,
            "raw_llm_output": last_raw_text,
            "parsed_verification_result": {
                k: v for k, v in fallback.items()
                if k != "_verification_log"
            },
            "fallback_error": self._as_text(repr(last_error) if last_error is not None else ""),
        }
        return fallback
    
    def _aggregate_node_diagnosis_only(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        step_diagnoses: List[Dict[str, Any]],
        summarized_steps: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce node-level diagnosis only.

        This module must not decide recovery_flag.
        This module must not produce recovery_suggestion.
        """
        aggregation_context = self._build_node_aggregation_context(
            node=node,
            logs=logs,
            step_diagnoses=step_diagnoses,
            summarized_steps=summarized_steps,
        )

        prompt = self._build_node_diagnosis_only_prompt(aggregation_context)

        last_error: Optional[Exception] = None
        last_raw_text: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_json_call(prompt)

                raw_text = ""
                raw_prompt = prompt
                if isinstance(raw, dict):
                    raw_text = self._as_text(raw.get("_verifier_raw_output", ""))
                    raw_prompt = self._as_text(raw.get("_verifier_prompt", prompt)) or prompt
                else:
                    raw_text = self._as_text(raw)

                last_raw_text = raw_text

                validated = self._validate_node_diagnosis_only(raw, node=node, logs=logs)
                validated = dict(validated)
                validated["_verification_log"] = {
                    "kind": "node_diagnosis_aggregation",
                    "prompt": raw_prompt,
                    "raw_llm_output": raw_text,
                    "parsed_node_diagnosis": {
                        k: v for k, v in validated.items()
                        if k != "_verification_log"
                    },
                }
                return validated

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "_aggregate_node_diagnosis_only attempt %d failed: %s",
                    attempt + 1,
                    repr(exc),
                )

        logger.error(
            "_aggregate_node_diagnosis_only exhausted retries. Returning fallback diagnosis."
        )
        fallback = self._build_fallback_node_diagnosis_only(
            node=node,
            logs=logs,
            error=last_error,
        )
        fallback = dict(fallback)
        fallback["_verification_log"] = {
            "kind": "node_diagnosis_aggregation",
            "prompt": prompt,
            "raw_llm_output": last_raw_text,
            "parsed_node_diagnosis": {
                k: v for k, v in fallback.items()
                if k != "_verification_log"
            },
            "fallback_error": self._as_text(repr(last_error) if last_error is not None else ""),
        }
        return fallback

    def _build_node_diagnosis_only_prompt(
        self,
        aggregation_context: Dict[str, Any],
    ) -> str:
        aggregation_context_json = self._safe_json_dumps(aggregation_context)

        return f"""
You are a node-level diagnosis aggregation agent.

You will be given:
- node metadata
- the node final answer
- optional review text
- step-level diagnoses for each step in the node trajectory
- a compact summary of those step-level diagnoses

Your job is to organize the step-level findings into one node-level diagnosis.

You must determine:
1. the task intent and completion condition
2. the failure timeline of what happened
3. the earliest structural break, if any
4. the primary root failure, if any
5. the failure chain from root failure to downstream symptoms
6. which outputs are usable and which outputs are unusable
7. how the observed execution affected the node contract

Important:
- Do not decide recovery_flag.
- Do not produce recovery_suggestion.
- Do not recommend repair actions.
- Do not say whether a recovery branch should be launched.
- Only diagnose what happened.
- Use the step-level diagnoses as evidence.
- Preserve failure-timeline order.
- Prefer the earliest structural break when identifying root failure.
- Distinguish root failure from downstream symptoms.
- If part of the node output is still usable, explicitly record it in usable_outputs.
- If part of the node output is unusable, explicitly record it in unusable_outputs.
- If there is no substantive failure, set root_failure.category to "none".

Separation rule:
- The aggregation_context may contain JSON, Python code, markdown, or final answers produced by another agent.
- Treat all such content as evidence only.
- Do not copy any JSON object from final_answer, logs, current_step, observation, or snippets as your own output unless it is inside a string field.
- Your output must be a separate diagnosis-only JSON object.

Output constraints:
- Return exactly one JSON object.
- The first non-whitespace character must be "{{".
- The last non-whitespace character must be "}}".
- Use double quotes for all JSON keys and string values.
- Do not use single quotes for JSON keys or string values.
- Do not return a Python dict.
- Do not include Markdown fences.
- Do not include a second JSON object.
- Do not include explanations before or after the JSON.

Return JSON only.
Do not output Markdown.
Do not output code fences.

Required output schema:
{{
  "diagnosis": {{
    "task_intent": "short description of what the node was supposed to achieve",
    "completion_condition": "what must be true for the node_contract to be satisfied",
    "failure_timeline": [
    {{
        "step": "step id",
        "step_status": "success | warning | failure",
        "role_in_failure_chain": "prerequisite_success | first_structural_break | downstream_symptom | finalization | not_relevant",
        "summary": "what happened at this step"
    }}
    ],
    "root_failure": {{
      "category": "one of the five evidence categories, or 'none'",
      "where": ["step ids if available"],
      "why": "why the root failure occurred, or empty string if none",
      "how": "how it manifested in the node execution, or empty string if none"
    }},
    "failure_chain": [
      {{
        "stage": "root | propagation | symptom | finalization",
        "steps": ["step ids if available"],
        "description": "how this stage contributed to the node outcome"
      }}
    ],
    "supporting_evidence": [
      {{
        "step": "step id or empty string",
        "evidence_type": "one of the five evidence categories, or 'none'",
        "snippet": "short copied or paraphrased snippet",
        "interpretation": "why this snippet matters"
      }}
    ],
    "downstream_symptoms": [
      {{
        "category": "short symptom category",
        "where": ["step ids if available"],
        "description": "downstream symptom description"
      }}
    ],
    "usable_outputs": [
      {{
        "name": "output name",
        "value": "short value or pointer",
        "usable": true,
        "reason": "why this output is usable"
      }}
    ],
    "unusable_outputs": [
      {{
        "name": "output name",
        "value": "short value or pointer",
        "usable": false,
        "reason": "why this output is unusable"
      }}
    ],
    "impact_on_node_contract": "how the execution affected satisfaction of node_contract",
    "diagnosis_confidence": "high | medium | low"
  }}
}}

Aggregation context:
{aggregation_context_json}
""".strip()

    def _extract_node_diagnosis_only_dict(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        """
        Accepts:
        - dict
        - str containing only JSON
        - str containing extra text before/after JSON
        - str containing fenced JSON

        Strategy:
        - Prefer the last schema-matching JSON object in the text.
        - Restrict matches to the node-diagnosis-only shape only.

        Returns:
        - parsed dict for the node diagnosis only output

        Raises:
        - ValueError if no node-diagnosis-only-shaped dict can be recovered
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_diagnosis_only_dict: start type=%s preview=%s",
                type(raw).__name__,
                self._debug_preview(raw),
            )

        if isinstance(raw, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_diagnosis_only_dict: raw is already dict keys=%s",
                    sorted(raw.keys()),
                )
            if self._looks_like_node_diagnosis_only(raw):
                return raw
            raise ValueError(
                "node diagnosis only dict does not match required schema; "
                f"keys={sorted(raw.keys())}"
            )

        if isinstance(raw, bytes):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_diagnosis_only_dict: decoding bytes len=%d",
                    len(raw),
                )
            raw = raw.decode("utf-8", errors="replace")

        if not isinstance(raw, str):
            raise ValueError(
                f"node diagnosis only output must be a dict or str, got {type(raw).__name__}"
            )

        text = raw.strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_diagnosis_only_dict: normalized text len=%d preview=%s",
                len(text),
                self._debug_preview(text),
            )

        if not text:
            raise ValueError("node diagnosis only text is empty")

        # 1. Fast path: the entire text is already a JSON object
        obj = self._try_load_json_dict(text)
        if obj is not None:
            if self._looks_like_node_diagnosis_only(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_node_diagnosis_only_dict: full-text parse succeeded keys=%s",
                        sorted(obj.keys()),
                    )
                return obj
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_diagnosis_only_dict: full-text parse produced dict but schema mismatch keys=%s",
                    sorted(obj.keys()),
                )

        # 2. Fenced JSON blocks, searched from the end
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_diagnosis_only_dict: found %d fenced block(s)",
                len(fenced_blocks),
            )

        for reverse_idx, fenced in enumerate(reversed(fenced_blocks), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_diagnosis_only_dict: trying fenced candidate from end #%d len=%d preview=%s",
                    reverse_idx,
                    len(fenced),
                    self._debug_preview(fenced),
                )
            obj = self._try_load_json_dict(fenced)
            if obj is not None and self._looks_like_node_diagnosis_only(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_node_diagnosis_only_dict: fenced parse succeeded keys=%s",
                        sorted(obj.keys()),
                    )
                return obj

        # 3. Balanced {...} candidates, searched from the end
        candidates = list(self._iter_balanced_json_object_candidates(text))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_diagnosis_only_dict: found %d balanced candidate(s)",
                len(candidates),
            )

        for reverse_idx, candidate in enumerate(reversed(candidates), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_diagnosis_only_dict: trying candidate from end #%d preview=%s",
                    reverse_idx,
                    self._debug_preview(candidate),
                )

            obj = self._try_load_json_dict(candidate)
            if obj is None:
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_diagnosis_only_dict: candidate from end #%d parsed as dict keys=%s",
                    reverse_idx,
                    sorted(obj.keys()),
                )

            if self._looks_like_node_diagnosis_only(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_node_diagnosis_only_dict: selected candidate from end #%d as node diagnosis only",
                        reverse_idx,
                    )
                return obj

        preview = text[:300].replace("\n", "\\n")
        raise ValueError(
            "could not recover a node-diagnosis-only-shaped dict from text; "
            f"prefix={preview!r}"
        )


    def _validate_node_diagnosis_only(
        self,
        raw: Any,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_dict = self._extract_node_diagnosis_only_dict(raw)

        if "recovery_flag" in raw_dict:
            raise ValueError(
                "node diagnosis module must not output recovery_flag"
            )

        if "recovery_suggestion" in raw_dict:
            raise ValueError(
                "node diagnosis module must not output recovery_suggestion"
            )

        diagnosis = raw_dict.get("diagnosis", {})
        if not isinstance(diagnosis, dict):
            diagnosis = {}

        diagnosis_norm: Dict[str, Any] = {
            "task_intent": self._as_text(diagnosis.get("task_intent")),
            "completion_condition": self._as_text(diagnosis.get("completion_condition")),
            "failure_timeline": self._normalize_failure_timeline(
                diagnosis.get("failure_timeline", [])
            ),
            "root_failure": self._normalize_root_failure(
                diagnosis.get("root_failure")
            ),
            "failure_chain": self._normalize_failure_chain(
                diagnosis.get("failure_chain", [])
            ),
            "supporting_evidence": self._normalize_supporting_evidence(
                diagnosis.get("supporting_evidence", [])
            ),
            "downstream_symptoms": self._normalize_downstream_symptoms(
                diagnosis.get("downstream_symptoms", [])
            ),
            "usable_outputs": self._normalize_output_items(
                diagnosis.get("usable_outputs", []),
                expected_usable=True,
            ),
            "unusable_outputs": self._normalize_output_items(
                diagnosis.get("unusable_outputs", []),
                expected_usable=False,
            ),
            "impact_on_node_contract": self._as_text(
                diagnosis.get("impact_on_node_contract")
            ),
            "diagnosis_confidence": self._normalize_confidence(
                diagnosis.get("diagnosis_confidence")
            ),
        }

        if not diagnosis_norm["task_intent"]:
            diagnosis_norm["task_intent"] = self._infer_task_intent_from_node(node)

        if not diagnosis_norm["completion_condition"]:
            diagnosis_norm["completion_condition"] = (
                self._infer_completion_condition_from_node(node)
            )

        if not diagnosis_norm["root_failure"]["category"]:
            diagnosis_norm["root_failure"]["category"] = "none"

        if not diagnosis_norm["impact_on_node_contract"]:
            diagnosis_norm["impact_on_node_contract"] = (
                "the verifier could not determine the contract impact reliably"
            )

        return {
            "diagnosis": diagnosis_norm
        }

    def _build_fallback_node_diagnosis_only(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        error: Optional[Exception],
    ) -> Dict[str, Any]:
        return {
            "diagnosis": {
                "task_intent": self._infer_task_intent_from_node(node),
                "completion_condition": self._infer_completion_condition_from_node(node),
                "failure_timeline": [],
                "root_failure": {
                    "category": "Omitted or Miscontrolled Process Step",
                    "where": [],
                    "why": "the node diagnosis aggregation step itself failed",
                    "how": self._as_text(
                        repr(error) if error is not None else "unknown diagnosis aggregation failure"
                    ),
                },
                "failure_chain": [
                    {
                        "stage": "root",
                        "steps": [],
                        "description": "diagnosis aggregation failed before a reliable node-level diagnosis could be produced",
                    }
                ],
                "supporting_evidence": [
                    {
                        "step": "",
                        "evidence_type": "Omitted or Miscontrolled Process Step",
                        "snippet": self._as_text(
                            repr(error) if error is not None else "unknown diagnosis aggregation failure"
                        ),
                        "interpretation": "the verifier could not complete node-level diagnosis aggregation",
                    }
                ],
                "downstream_symptoms": [
                    {
                        "category": "diagnosis_unavailable",
                        "where": [],
                        "description": "downstream recovery decision would proceed without a reliable node diagnosis",
                    }
                ],
                "usable_outputs": [],
                "unusable_outputs": [],
                "impact_on_node_contract": (
                    "the node may or may not satisfy node_contract, but the diagnosis module could not determine that reliably"
                ),
                "diagnosis_confidence": "low",
            }
        }
    
    def _debug_preview(self, value: Any, limit: int = 240) -> str:
        """
        Return a short single-line preview for debug logs.
        """
        try:
            text = value if isinstance(value, str) else repr(value)
        except Exception:
            text = f"<unreprable {type(value).__name__}>"

        text = text.replace("\n", "\\n")
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text


    def _is_recovery_suggestion_schema_match(self, obj: Any) -> bool:
        """
        Return True only if obj is a dict with the required top-level keys.
        """
        if not isinstance(obj, dict):
            return False

        required_keys = {
            "recovery_goal",
            "primary_fault_hypothesis",
            "recommended_next_actions",
        }
        return required_keys.issubset(obj.keys())


    def _extract_recovery_suggestion_dict(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        """
        Accepts:
        - dict
        - str containing only JSON
        - str containing extra text before/after JSON
        - str containing fenced JSON

        Strategy:
        - Prefer the last schema-matching JSON object in the text.
        - This avoids being trapped by earlier 'expected output format' examples.

        Returns:
        - parsed dict for the recovery suggestion

        Raises:
        - ValueError if no schema-matching dict-shaped JSON object can be recovered
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_recovery_suggestion_dict: start type=%s preview=%s",
                type(raw).__name__,
                self._debug_preview(raw),
            )

        if isinstance(raw, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: raw is already dict keys=%s",
                    sorted(raw.keys()),
                )
            if self._is_recovery_suggestion_schema_match(raw):
                return raw
            raise ValueError(
                "recovery suggestion dict does not match required schema; "
                f"keys={sorted(raw.keys())}"
            )

        if isinstance(raw, bytes):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: decoding bytes len=%d",
                    len(raw),
                )
            raw = raw.decode("utf-8", errors="replace")

        if not isinstance(raw, str):
            raise ValueError(
                f"recovery suggestion must be a dict or str, got {type(raw).__name__}"
            )

        text = raw.strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_recovery_suggestion_dict: normalized text len=%d preview=%s",
                len(text),
                self._debug_preview(text),
            )

        if not text:
            raise ValueError("recovery suggestion text is empty")

        # 1. Fast path: the entire text is already a JSON object
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("_extract_recovery_suggestion_dict: trying full-text json.loads")
        obj = self._try_load_json_dict(text)
        if obj is not None:
            if self._is_recovery_suggestion_schema_match(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_recovery_suggestion_dict: full-text parse succeeded with schema-matching keys=%s",
                        sorted(obj.keys()),
                    )
                return obj
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: full-text parse produced dict but schema mismatch keys=%s",
                    sorted(obj.keys()),
                )

        # 2. Fenced JSON blocks
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("_extract_recovery_suggestion_dict: searching fenced JSON blocks")
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_recovery_suggestion_dict: found %d fenced block(s)",
                len(fenced_blocks),
            )

        for reverse_idx, fenced in enumerate(reversed(fenced_blocks), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: trying fenced candidate from end #%d len=%d preview=%s",
                    reverse_idx,
                    len(fenced),
                    self._debug_preview(fenced),
                )
            obj = self._try_load_json_dict(fenced)
            if obj is not None and self._is_recovery_suggestion_schema_match(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_recovery_suggestion_dict: fenced parse succeeded with schema-matching keys=%s",
                        sorted(obj.keys()),
                    )
                return obj

        # 3. Search balanced {...} candidates in the full text
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_recovery_suggestion_dict: scanning balanced JSON object candidates"
            )

        candidates = list(self._iter_balanced_json_object_candidates(text))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_recovery_suggestion_dict: found %d balanced candidate(s)",
                len(candidates),
            )
            for idx, candidate in enumerate(
                candidates[-5:], start=max(0, len(candidates) - 5)
            ):
                logger.debug(
                    "_extract_recovery_suggestion_dict: candidate[%d] len=%d preview=%s",
                    idx,
                    len(candidate),
                    self._debug_preview(candidate),
                )

        # Important: search from the end and accept the first schema-matching dict.
        for reverse_idx, candidate in enumerate(reversed(candidates), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: trying candidate from end #%d preview=%s",
                    reverse_idx,
                    self._debug_preview(candidate),
                )
            obj = self._try_load_json_dict(candidate)
            if obj is None:
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: candidate from end #%d parsed as dict keys=%s",
                    reverse_idx,
                    sorted(obj.keys()),
                )

            if self._is_recovery_suggestion_schema_match(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_recovery_suggestion_dict: selected candidate from end #%d as recovery suggestion",
                        reverse_idx,
                    )
                return obj

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_recovery_suggestion_dict: candidate from end #%d rejected بسبب schema mismatch",
                    reverse_idx,
                )

        preview = text[:300].replace("\n", "\\n")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_recovery_suggestion_dict: failed to recover schema-matching dict preview=%s",
                preview,
            )
        raise ValueError(
            "could not recover a schema-matching dict-shaped JSON object from "
            f"recovery suggestion text; prefix={preview!r}"
        )


    def _try_load_json_dict(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_try_load_json_dict: attempting parse len=%d preview=%s",
                len(text),
                self._debug_preview(text),
            )

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            if logger.isEnabledFor(logging.DEBUG):
                around_start = max(0, exc.pos - 60)
                around_end = min(len(text), exc.pos + 60)
                around = text[around_start:around_end].replace("\n", "\\n")
                logger.debug(
                    "_try_load_json_dict: JSONDecodeError msg=%s pos=%d line=%d col=%d around=%s",
                    exc.msg,
                    exc.pos,
                    exc.lineno,
                    exc.colno,
                    around,
                )
            return None

        if isinstance(obj, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_try_load_json_dict: parse succeeded with dict keys=%s",
                    sorted(obj.keys()),
                )
            return obj

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_try_load_json_dict: parse succeeded but type=%s, not dict",
                type(obj).__name__,
            )
        return None


    def _iter_balanced_json_object_candidates(
        self,
        text: str,
    ) -> Iterator[str]:
        """
        Yield balanced {...} substrings while respecting JSON string literals.
        This avoids being confused by braces inside quoted strings.
        """
        start: Optional[int] = None
        depth = 0
        in_string = False
        escape = False
        yielded = 0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_iter_balanced_json_object_candidates: scanning text len=%d",
                len(text),
            )

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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "_iter_balanced_json_object_candidates: object start at index=%d",
                            i,
                        )
                depth += 1
                continue

            if ch == "}":
                if depth == 0:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "_iter_balanced_json_object_candidates: stray closing brace at index=%d",
                            i,
                        )
                    continue

                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i + 1]
                    yielded += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "_iter_balanced_json_object_candidates: yielded candidate #%d span=[%d:%d] len=%d preview=%s",
                            yielded,
                            start,
                            i + 1,
                            len(candidate),
                            self._debug_preview(candidate),
                        )
                    yield candidate
                    start = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_iter_balanced_json_object_candidates: finished yielded=%d final_depth=%d in_string=%s",
                yielded,
                depth,
                in_string,
            )


    def _validate_recovery_suggestion(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        errors: List[str] = []

        raw_dict = self._extract_recovery_suggestion_dict(raw)

        recovery_goal_present = "recovery_goal" in raw_dict
        primary_fault_present = "primary_fault_hypothesis" in raw_dict
        actions_present = "recommended_next_actions" in raw_dict

        recovery_goal = self._as_text(raw_dict.get("recovery_goal"))
        primary_fault_hypothesis = self._as_text(
            raw_dict.get("primary_fault_hypothesis")
        )
        recommended_next_actions = self._normalize_list_of_text(
            raw_dict.get("recommended_next_actions", [])
        )

        if not recovery_goal_present:
            errors.append('missing required field "recovery_goal"')
        elif not recovery_goal:
            errors.append('"recovery_goal" is present but empty')

        if not primary_fault_present:
            errors.append('missing required field "primary_fault_hypothesis"')
        elif recovery_goal != "no recovery required" and not primary_fault_hypothesis:
            errors.append(
                '"primary_fault_hypothesis" must be non-empty unless '
                '"recovery_goal" is "no recovery required"'
            )

        if not actions_present:
            errors.append('missing required field "recommended_next_actions"')
        elif not isinstance(raw_dict.get("recommended_next_actions"), list):
            errors.append('"recommended_next_actions" must be a list of strings')
        elif recovery_goal != "no recovery required" and not recommended_next_actions:
            errors.append(
                '"recommended_next_actions" must contain at least one action unless '
                '"recovery_goal" is "no recovery required"'
            )

        if recovery_goal == "no recovery required":
            primary_fault_hypothesis = ""
            recommended_next_actions = []

        if errors:
            raise ValueError(" ; ".join(errors))

        return {
            "recovery_goal": recovery_goal,
            "primary_fault_hypothesis": primary_fault_hypothesis,
            "recommended_next_actions": recommended_next_actions,
        }
    
    def _build_recovery_retry_feedback(
        self,
        error: Exception,
        attempt: int,
    ) -> str:
        return f"""
    Previous attempt {attempt} failed.

    Validation or parsing error:
    {repr(error)}

    You must fix the previous output and try again.

    Retry requirements:
    - Return exactly one JSON object.
    - Do not include any explanation, reasoning, notes, or markdown.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Include all required fields:
    1. "recovery_goal"
    2. "primary_fault_hypothesis"
    3. "recommended_next_actions"
    - "recovery_goal" must be a non-empty string.
    - "primary_fault_hypothesis" must be a non-empty string unless "recovery_goal" is "no recovery required".
    - "recommended_next_actions" must be a JSON array of strings and must contain at least one item unless "recovery_goal" is "no recovery required".
    - If no recovery is required, use exactly:
    {{
        "recovery_goal": "no recovery required",
        "primary_fault_hypothesis": "",
        "recommended_next_actions": []
    }}

    Do not repeat the error message.
    Do not explain what you changed.
    Return only the corrected JSON object.
    """.strip()

    def _build_fallback_recovery_suggestion(
        self,
        diagnosis: Dict[str, Any],
        error: Optional[Exception],
    ) -> Dict[str, Any]:
        diagnosis_obj = diagnosis.get("diagnosis", diagnosis)
        root_failure = diagnosis_obj.get("root_failure", {})
        root_category = ""
        if isinstance(root_failure, dict):
            root_category = self._as_text(root_failure.get("category"))

        if not root_category or root_category == "none":
            return {
                "recovery_goal": "no recovery required",
                "primary_fault_hypothesis": "",
                "recommended_next_actions": [],
            }

        return {
            "recovery_goal": "repair the diagnosed root failure",
            "primary_fault_hypothesis": (
                f"the diagnosed root failure category '{root_category}' is the leading fault hypothesis"
            ),
            "recommended_next_actions": [
                "inspect the cached node diagnosis",
                "test the primary fault hypothesis before reusing this node output downstream",
            ],
        }

    def _suggest_recovery_from_diagnosis(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        diagnosis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce recovery_suggestion only.

        This module does not decide recovery_flag.
        It assumes diagnosis is the source of truth and returns a lightweight
        recovery suggestion object.

        Unified version:
        - context construction comes from the newer node/logs/diagnosis path
        - retry-with-feedback logic comes from the older 3-attempt path
        """
        recovery_context = self._build_recovery_suggestion_context(
            node=node,
            logs=logs,
            diagnosis=diagnosis,
        )

        base_prompt = self._build_recovery_suggestion_prompt(recovery_context)
        prompt = base_prompt

        last_error: Optional[Exception] = None
        retry_feedback_blocks: List[str] = []
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                raw = self.llm_json_call(prompt)
                return self._validate_recovery_suggestion(raw)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "_suggest_recovery_from_diagnosis attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    repr(exc),
                )

                retry_feedback_blocks.append(
                    self._build_recovery_retry_feedback(
                        error=exc,
                        attempt=attempt + 1,
                    )
                )

                prompt = (
                    base_prompt
                    + "\n\n"
                    + "\n\n".join(retry_feedback_blocks)
                )

        logger.error(
            "_suggest_recovery_from_diagnosis exhausted retries. Returning fallback recovery suggestion."
        )
        return self._build_fallback_recovery_suggestion(
            diagnosis=diagnosis,
            error=last_error,
        )

    def _build_recovery_suggestion_context(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        diagnosis: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "node": {
                "id": self._as_text(node.get("id")),
                "task": self._as_text(node.get("task")),
                "agent": self._as_text(node.get("agent")),
                "deps": self._normalize_list_of_text(node.get("deps", [])),
                "node_contract": self._as_text(node.get("node_contract")),
            },
            "final_answer": self._as_text(logs.get("final_answer", "")),
            "reviews": logs.get("reviews", []),
            "diagnosis": diagnosis.get("diagnosis", diagnosis),
        }

    def _build_recovery_suggestion_prompt(
        self,
        recovery_context: Dict[str, Any],
    ) -> str:
        recovery_context_json = self._safe_json_dumps(recovery_context)

        print("22222222",flush=True)

        return f"""
    You are a recovery suggestion agent.

    You will be given:
    - node metadata
    - optional final answer and review text
    - a node-level diagnosis produced by a separate diagnosis module

    Your job is to produce a recovery suggestion only.

    Important:
    - Do not re-diagnose the node from scratch.
    - Use the provided diagnosis as the source of truth.
    - The diagnosis is the detailed record of what failed.
    - Do not repeat the full diagnosis in your output.
    - Your output must contain only:
    1. recovery_goal
    2. primary_fault_hypothesis
    3. recommended_next_actions

    Terminology:
    - Use the classical distinction fault -> error -> failure.
    - A failure is an externally visible deviation of delivered service from correct service.
    - An error is the part of the system state that may lead to a subsequent failure.
    - A fault is the adjudged or hypothesized cause of an error.
    - A fault may be dormant until it becomes active and causes an error.
    - Therefore, primary_fault_hypothesis must describe the most likely underlying cause that produced the diagnosed error or externally visible failure.
    - Do not use primary_fault_hypothesis to merely restate the observed failure symptom.
    - If the diagnosis contains both symptoms and causes, prefer the earliest actionable structural cause that recovery should test or address first.

    Field guidance:
    - recovery_goal should state what correct state, artifact, or service condition should be restored. If no substantive recovery is needed, use "no recovery required".
    - primary_fault_hypothesis should state the leading causal hypothesis to test or act on first. It should explain why the node entered a bad state or produced an externally incorrect outcome.
    - recommended_next_actions should be a short list of concrete actions that directly test, isolate, or correct the primary_fault_hypothesis.

    How to distinguish the fields:
    - recovery_goal = what should be restored
    - primary_fault_hypothesis = the most likely cause behind the diagnosed bad state or observed failure
    - recommended_next_actions = the concrete next steps to verify or remediate that cause

    If no substantive recovery is needed, use:
    - recovery_goal = "no recovery required"
    - primary_fault_hypothesis = ""
    - recommended_next_actions = []

    Output constraints:
    - Return exactly one JSON object.
    - The first non-whitespace character must be "{{".
    - The last non-whitespace character must be "}}".
    - Use double quotes for all JSON keys and string values.
    - Do not return a Python dict.
    - Do not include Markdown fences.
    - Do not include explanations before or after the JSON.

    Required output schema:
    {{
    "recovery_goal": "what should be restored, or 'no recovery required'",
    "primary_fault_hypothesis": "the leading underlying cause hypothesis to test or act on first, or ''",
    "recommended_next_actions": [
        "first concrete next action if recovery is needed"
    ]
    }}

    few_shot_examples: 
    {{
        "recovery_goal": "restore the missing sensor-list prerequisite and produce a supported, properly formatted answer identifying the relevant sensors for monitoring the Liquid Refrigerant Evaporator Temperature of Chiller 6",
        "primary_fault_hypothesis": "the node failed because the failure-mode-to-sensor mapping tool was invoked before obtaining the available sensor list for Chiller 6, and the agent then finalized an answer from unsupported assumptions",
        "recommended_next_actions": [
            "retrieve the available sensor list for Chiller 6 at MAIN site",
            "rerun the failure-mode-to-sensor mapping using the retrieved sensor list as input",
            "regenerate the final answer using the retrieved mapping only, without unsupported assumptions or reasoning tags"
        ]
    }}
    {{
        "recovery_goal": "produce a valid final answer for the historical data retrieval node by resolving the empty-data issue and removing malformed finalization behavior",
        "primary_fault_hypothesis": "the node did not yield usable historical data for the requested interval, and the agent then used Finish incorrectly by outputting a planning narrative instead of a contract-satisfying final answer",
        "recommended_next_actions": [
            "verify that the selected sensor and requested time range are correct for Chiller 6 at MAIN site",
            "rerun the historical data retrieval after adjusting the query only if the interval or sensor selection is invalid",
            "finalize with a concise answer that reports the retrieval outcome without planning narrative or unsupported scaffold text"
        ]
    }}

    Recovery suggestion context:
    {recovery_context_json}
    """.strip()

    def _validate_recovery_plan(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError(
                f"recovery plan must be a dict, got {type(raw).__name__}"
            )

        decision = raw.get("recovery_decision", {})
        if not isinstance(decision, dict):
            decision = {}

        suggestion = raw.get("recovery_suggestion", {})
        if not isinstance(suggestion, dict):
            suggestion = {}

        recovery_flag = decision.get("recovery_flag")
        if not isinstance(recovery_flag, bool):
            raise ValueError("recovery_decision.recovery_flag must be a bool")

        severity = self._as_text(decision.get("severity"))
        if severity not in {
            "none",
            "cosmetic",
            "weak_but_usable",
            "contract_blocking",
            "final_answer_blocking",
        }:
            severity = "contract_blocking" if recovery_flag else "weak_but_usable"

        downstream_risk = self._as_text(decision.get("downstream_risk"))
        if downstream_risk not in {"none", "low", "medium", "high"}:
            downstream_risk = "high" if recovery_flag else "low"

        recoverability = self._as_text(decision.get("recoverability"))
        if recoverability not in {
            "not_needed",
            "recoverable",
            "not_recoverable_from_available_context",
        }:
            recoverability = "recoverable" if recovery_flag else "not_needed"

        recovery_norm = {
            "recovery_goal": self._as_text(suggestion.get("recovery_goal")),
            "actions": self._normalize_list_of_text(suggestion.get("actions", [])),
            "do_not": self._normalize_list_of_text(suggestion.get("do_not", [])),
            "handoff_payload": self._normalize_handoff_payload(
                suggestion.get("handoff_payload", {})
            ),
        }

        if recovery_flag:
            if not recovery_norm["recovery_goal"]:
                recovery_norm["recovery_goal"] = (
                    "restore the node to a state where its node_contract can be satisfied"
                )
            if not recovery_norm["actions"]:
                recovery_norm["actions"] = [
                    "inspect the node diagnosis",
                    "repair the root failure before reusing this node output downstream",
                ]
        else:
            if not recovery_norm["recovery_goal"]:
                recovery_norm["recovery_goal"] = "no recovery required"
            if not recovery_norm["handoff_payload"]["notes_for_recovery_node"]:
                recovery_norm["handoff_payload"]["notes_for_recovery_node"] = (
                    "No recovery branch should be launched for this node."
                )

        return {
            "recovery_decision": {
                "recovery_flag": recovery_flag,
                "decision_rationale": self._as_text(
                    decision.get("decision_rationale")
                ),
                "severity": severity,
                "downstream_risk": downstream_risk,
                "recoverability": recoverability,
            },
            "recovery_suggestion": recovery_norm,
        }

    def _build_fallback_recovery_plan(
        self,
        diagnosis: Dict[str, Any],
        error: Optional[Exception],
    ) -> Dict[str, Any]:
        diagnosis_obj = diagnosis.get("diagnosis", diagnosis)
        root_failure = diagnosis_obj.get("root_failure", {})
        root_category = ""
        if isinstance(root_failure, dict):
            root_category = self._as_text(root_failure.get("category"))

        # Conservative fallback:
        # If recovery decision itself fails, launch recovery unless diagnosis says "none".
        recovery_flag = root_category != "none"

        return {
            "recovery_decision": {
                "recovery_flag": recovery_flag,
                "decision_rationale": (
                    "Recovery decision generation failed. "
                    "Using conservative fallback based on the node diagnosis. "
                    f"Error: {self._as_text(repr(error) if error is not None else 'unknown recovery decision failure')}"
                ),
                "severity": "contract_blocking" if recovery_flag else "none",
                "downstream_risk": "high" if recovery_flag else "none",
                "recoverability": "recoverable" if recovery_flag else "not_needed",
            },
            "recovery_suggestion": {
                "recovery_goal": (
                    "repair the diagnosed root failure"
                    if recovery_flag
                    else "no recovery required"
                ),
                "actions": (
                    [
                        "inspect the node diagnosis",
                        "repair the diagnosed root failure before downstream use",
                    ]
                    if recovery_flag
                    else []
                ),
                "do_not": (
                    [
                        "do not treat the failed recovery-decision call as a successful verification pass"
                    ]
                    if recovery_flag
                    else []
                ),
                "handoff_payload": {
                    "retry_from_step": "",
                    "required_inputs": [],
                    "notes_for_recovery_node": (
                        "recovery decision fallback triggered because the recovery module failed"
                    ),
                },
            },
        }
    
    def _assemble_verification_result(
        self,
        diagnosis: Dict[str, Any],
        recovery_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert separated outputs back into the public VerificationAgent schema.

        Public top-level schema remains:
        {
          "recovery_flag": bool,
          "diagnosis": dict,
          "recovery_suggestion": dict
        }
        """
        diagnosis_obj = diagnosis.get("diagnosis", diagnosis)
        if not isinstance(diagnosis_obj, dict):
            diagnosis_obj = {}

        decision = recovery_plan.get("recovery_decision", {})
        if not isinstance(decision, dict):
            decision = {}

        recovery_flag = decision.get("recovery_flag", True)
        if not isinstance(recovery_flag, bool):
            recovery_flag = True

        recovery_suggestion = recovery_plan.get("recovery_suggestion", {})
        if not isinstance(recovery_suggestion, dict):
            recovery_suggestion = {}

        # Normalize required public diagnosis fields while preserving extra fields.
        diagnosis_norm = dict(diagnosis_obj)

        diagnosis_norm["task_intent"] = self._as_text(
            diagnosis_norm.get("task_intent")
        )
        diagnosis_norm["completion_condition"] = self._as_text(
            diagnosis_norm.get("completion_condition")
        )
        diagnosis_norm["root_failure"] = self._normalize_root_failure(
            diagnosis_norm.get("root_failure")
        )
        diagnosis_norm["supporting_evidence"] = self._normalize_supporting_evidence(
            diagnosis_norm.get("supporting_evidence", [])
        )
        diagnosis_norm["downstream_symptoms"] = self._normalize_downstream_symptoms(
            diagnosis_norm.get("downstream_symptoms", [])
        )
        diagnosis_norm["impact_on_node_contract"] = self._as_text(
            diagnosis_norm.get("impact_on_node_contract")
        )

        # Normalize optional diagnosis-only fields.
        diagnosis_norm["failure_timeline"] = self._normalize_failure_timeline(
            diagnosis_norm.get("failure_timeline", [])
        )
        diagnosis_norm["failure_chain"] = self._normalize_failure_chain(
            diagnosis_norm.get("failure_chain", [])
        )
        diagnosis_norm["usable_outputs"] = self._normalize_output_items(
            diagnosis_norm.get("usable_outputs", []),
            expected_usable=True,
        )
        diagnosis_norm["unusable_outputs"] = self._normalize_output_items(
            diagnosis_norm.get("unusable_outputs", []),
            expected_usable=False,
        )
        diagnosis_norm["diagnosis_confidence"] = self._normalize_confidence(
            diagnosis_norm.get("diagnosis_confidence", "medium")
        )

        recovery_norm = {
            "recovery_goal": self._as_text(recovery_suggestion.get("recovery_goal")),
            "actions": self._normalize_list_of_text(
                recovery_suggestion.get("actions", [])
            ),
            "do_not": self._normalize_list_of_text(
                recovery_suggestion.get("do_not", [])
            ),
            "handoff_payload": self._normalize_handoff_payload(
                recovery_suggestion.get("handoff_payload", {})
            ),
        }

        # Attach decision metadata. This is useful for debugging and evaluation.
        # If you must keep the exact old schema only, remove this field.
        recovery_norm["decision_metadata"] = {
            "decision_rationale": self._as_text(
                decision.get("decision_rationale")
            ),
            "severity": self._as_text(decision.get("severity")),
            "downstream_risk": self._as_text(decision.get("downstream_risk")),
            "recoverability": self._as_text(decision.get("recoverability")),
        }

        if recovery_flag:
            if not recovery_norm["recovery_goal"]:
                recovery_norm["recovery_goal"] = (
                    "restore the node to a state where its node_contract can be satisfied"
                )
            if not recovery_norm["actions"]:
                recovery_norm["actions"] = [
                    "inspect the node diagnosis",
                    "repair the root failure before downstream use",
                ]
        else:
            if not recovery_norm["recovery_goal"]:
                recovery_norm["recovery_goal"] = "no recovery required"
            if not recovery_norm["handoff_payload"]["notes_for_recovery_node"]:
                recovery_norm["handoff_payload"]["notes_for_recovery_node"] = (
                    "No recovery branch should be launched for this node."
                )

        return {
            "recovery_flag": recovery_flag,
            "diagnosis": diagnosis_norm,
            "recovery_suggestion": recovery_norm,
        }
        
    # -------------------------------------------------------------------------
    # Backend wrapper
    # -------------------------------------------------------------------------

    def _extract_json_object_candidates(self, text: str) -> List[str]:
        """
        Extract balanced top-level JSON-object-looking substrings from text.

        This is safer than taking text[text.find("{"):text.rfind("}")],
        because LLM output may contain:
        - a schema example plus a final answer
        - JSON inside the inspected trajectory step
        - multiple JSON objects
        - markdown or explanatory text around the JSON
        """
        candidates: List[str] = []

        depth = 0
        start: Optional[int] = None
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
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start:i + 1])
                        start = None

        return candidates

    def _looks_like_step_diagnosis(self, obj: Any) -> bool:
        return (
            isinstance(obj, dict)
            and "step" in obj
            and obj.get("step_status") in {"success", "warning", "failure"}
            and isinstance(obj.get("is_root_candidate"), bool)
            and isinstance(obj.get("evidence"), list)
        )

    def _looks_like_node_verification(self, obj: Any) -> bool:
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("recovery_flag"), bool)
            and isinstance(obj.get("diagnosis"), dict)
            and isinstance(obj.get("recovery_suggestion"), dict)
        )

    def _looks_like_node_diagnosis_only(self, obj: Any) -> bool:
        """
        Intermediate output from Node Diagnosis Aggregation Module.

        This module must not contain recovery_flag or recovery_suggestion.
        """
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("diagnosis"), dict)
            and "recovery_flag" not in obj
            and "recovery_suggestion" not in obj
            and "recovery_decision" not in obj
        )

    def _looks_like_recovery_plan(self, obj: Any) -> bool:
        """
        Intermediate output from Recovery Decision and Suggestion Module.
        """
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("recovery_decision"), dict)
            and isinstance(obj.get("recovery_suggestion"), dict)
        )

    def _looks_like_recovery_suggestion_only(self, obj: Any) -> bool:
        return (
            isinstance(obj, dict)
            and isinstance(obj.get("recovery_goal"), str)
            and isinstance(obj.get("primary_fault_hypothesis"), str)
            and isinstance(obj.get("recommended_next_actions"), list)
        )

    def _looks_like_expected_json_output(self, obj: Any) -> bool:
        return (
            self._looks_like_step_diagnosis(obj)
            or self._looks_like_node_diagnosis_only(obj)
            or self._looks_like_recovery_plan(obj)
            or self._looks_like_node_verification(obj)
            or self._looks_like_recovery_suggestion_only(obj)
        )

    def _extract_expected_json_output_dict(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        if isinstance(raw, dict):
            if self._looks_like_expected_json_output(raw):
                return raw
            raise ValueError(
                f"dict parsed but schema mismatch; keys={sorted(raw.keys())}"
            )

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

        if not isinstance(raw, str):
            raise ValueError(
                f"expected json output must be dict/str/bytes, got {type(raw).__name__}"
            )

        text = raw.strip()
        if not text:
            raise ValueError("expected json output text is empty")

        scan_text = text

        m = re.search(r"assistantfinal\s*\{", text, flags=re.IGNORECASE)
        if m:
            brace_index = text.find("{", m.start())
            if brace_index != -1:
                scan_text = text[brace_index:]

        # 1. full text
        obj = self._try_load_json_dict(scan_text)
        if obj is not None and self._looks_like_expected_json_output(obj):
            return obj

        # 2. fenced JSON blocks, from the end
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            scan_text,
            flags=re.DOTALL,
        )
        for fenced in reversed(fenced_blocks):
            obj = self._try_load_json_dict(fenced)
            if obj is not None and self._looks_like_expected_json_output(obj):
                return obj

        # 3. balanced {...} candidates, from the end
        print(f"scan_text: {scan_text}", flush=True)
        candidates = list(self._iter_balanced_json_object_candidates(scan_text))
        print(f"candidates: {candidates}", flush=True)
        for candidate in reversed(candidates):
            obj = self._try_load_json_dict(candidate)
            if obj is not None and self._looks_like_expected_json_output(obj):
                return obj

        preview = text[:300].replace("\n", "\\n")
        raise ValueError(
            "Could not recover schema-matching verifier JSON; "
            f"prefix={preview!r}"
        )
    

    def _wrap_llm_generate_as_json_call(self, llm_generate):
        def _call(prompt: str) -> Dict[str, Any]:
            # print("=" * 80, flush=True)
            # print("[_wrap_llm_generate_as_json_call] start", flush=True)
            # print(
            #     f"[_wrap_llm_generate_as_json_call] llm_generate={llm_generate}",
            #     flush=True,
            # )
            # print(
            #     f"[_wrap_llm_generate_as_json_call] prompt_len={len(prompt)}",
            #     flush=True,
            # )
            # print(
            #     f"[_wrap_llm_generate_as_json_call] prompt_preview={self._debug_preview(prompt, limit=1500)}",
            #     flush=True,
            # )
            # print(
            #     f"[_wrap_llm_generate_as_json_call] llm_generate={llm_generate}",
            #     flush=True,
            # )

            raw = llm_generate(prompt)

            # print(
            #     f"[_wrap_llm_generate_as_json_call] raw_type_before_tuple={type(raw).__name__}",
            #     flush=True,
            # )
            # print(
            #     f"[_wrap_llm_generate_as_json_call] raw_repr_before_tuple={repr(raw)[:2000]}",
            #     flush=True,
            # )

            # time.sleep(3)

            if isinstance(raw, tuple):
                # print(
                #     f"[_wrap_llm_generate_as_json_call] tuple_len={len(raw)}",
                #     flush=True,
                # )
                raw = raw[0]
                # print(
                #     f"[_wrap_llm_generate_as_json_call] raw_type_after_tuple={type(raw).__name__}",
                #     flush=True,
                # )
                # print(
                #     f"[_wrap_llm_generate_as_json_call] raw_repr_after_tuple={repr(raw)[:2000]}",
                #     flush=True,
                # )
                # time.sleep(3)

            raw_text = (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, bytes)
                else raw if isinstance(raw, str)
                else repr(raw)
            )

            # print(
            #     f"[_wrap_llm_generate_as_json_call] raw_text_type={type(raw_text).__name__}",
            #     flush=True,
            # )
            # print(
            #     f"[_wrap_llm_generate_as_json_call] raw_text_len={len(raw_text)}",
            #     flush=True,
            # )
            # print(
            #     f"[_wrap_llm_generate_as_json_call] raw_text_preview={self._debug_preview(raw_text, limit=3000)}",
            #     flush=True,
            # )

            # logger.warning(
            #     "RAW_VERIFIER_OUTPUT_BEFORE_PARSE prefix=%r",
            #     self._debug_preview(raw_text, limit=2000),
            # )

            # time.sleep(3)

            try:
                # print(
                #     "[_wrap_llm_generate_as_json_call] calling _extract_expected_json_output_dict ...",
                #     flush=True,
                # )
                parsed = self._extract_expected_json_output_dict(raw_text)

                # print(
                #     f"[_wrap_llm_generate_as_json_call] parsed_type={type(parsed).__name__}",
                #     flush=True,
                # )
                # if isinstance(parsed, dict):
                #     print(
                #         f"[_wrap_llm_generate_as_json_call] parsed_keys={list(parsed.keys())}",
                #         flush=True,
                #     )
                # else:
                #     print(
                #         f"[_wrap_llm_generate_as_json_call] parsed_repr={repr(parsed)[:2000]}",
                #         flush=True,
                #     )

                # time.sleep(3)

                if isinstance(parsed, dict):
                    parsed = dict(parsed)
                    parsed["_verifier_raw_output"] = raw_text
                    parsed["_verifier_prompt"] = prompt

                    # print(
                    #     f"[_wrap_llm_generate_as_json_call] return_keys={list(parsed.keys())}",
                    #     flush=True,
                    # )
                    # print(
                    #     "[_wrap_llm_generate_as_json_call] success",
                    #     flush=True,
                    # )
                    # print("=" * 80, flush=True)
                    return parsed

                raise TypeError(
                    f"_extract_expected_json_output_dict returned non-dict: {type(parsed).__name__}"
                )

            except Exception as exc:
                # print(
                #     f"[_wrap_llm_generate_as_json_call] parse_exception_type={type(exc).__name__}",
                #     flush=True,
                # )
                # print(
                #     f"[_wrap_llm_generate_as_json_call] parse_exception={repr(exc)}",
                #     flush=True,
                # )

                candidates = []
                try:
                    candidates = list(self._iter_balanced_json_object_candidates(raw_text))
                    # print(
                    #     f"[_wrap_llm_generate_as_json_call] candidate_count={len(candidates)}",
                    #     flush=True,
                    # )
                    # if candidates:
                    #     print(
                    #         f"[_wrap_llm_generate_as_json_call] first_candidate_preview={self._debug_preview(candidates[0], limit=1000)}",
                    #         flush=True,
                    #     )
                    print(
                        f"candidates: {candidates}",
                        flush=True,
                    )
                except Exception as candidate_exc:
                    # print(
                    #     f"[_wrap_llm_generate_as_json_call] candidate_scan_exception={repr(candidate_exc)}",
                    #     flush=True,
                    # )
                    # time.sleep(3)
                    raise ValueError(
                        "Extractor failed while scanning balanced JSON candidates. "
                        f"root_error={repr(candidate_exc)}; "
                        f"raw_prefix={raw_text[:1000]!r}"
                    ) from candidate_exc

                # time.sleep(3)

                raise ValueError(
                    "Could not recover schema-matching verifier JSON. "
                    f"root_error={repr(exc)}; "
                    f"candidate_count={len(candidates)}; "
                    f"last_candidate_preview={self._debug_preview(candidates[-1], limit=400) if candidates else '<none>'}; "
                    f"raw_prefix={raw_text[:1000]!r}"
                ) from exc

        return _call
    
    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    def _truncate_text(self, text: Any, max_chars: int) -> str:
        s = self._as_text(text)
        if max_chars <= 0:
            return s
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + "...<TRUNCATED>"

    def _normalize_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(node, dict):
            raise TypeError("node must be a dict")

        return {
            "id": self._as_text(node.get("id")),
            "task": self._as_text(node.get("task")),
            "agent": self._as_text(node.get("agent")),
            "deps": self._normalize_list_of_text(node.get("deps", [])),
            "node_contract": self._as_text(node.get("node_contract")),
        }

    def _normalize_logs(self, logs: Any) -> Dict[str, Any]:
        if logs is None:
            return {"final_answer": ""}

        if isinstance(logs, dict):
            normalized = dict(logs)
            if "final_answer" not in normalized:
                normalized["final_answer"] = ""
            if "reviews" not in normalized:
                normalized["reviews"] = []
            return normalized

        return {
            "final_answer": self._as_text(logs),
            "reviews": [],
        }

    # -------------------------------------------------------------------------
    # Step level helpers
    # -------------------------------------------------------------------------

    def _extract_step_records_from_logs(self, logs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract step records from logs["trajectroy_log"] and normalize them.

        Returns:
            [
                {
                    "step": "1",
                    "thought": "...",
                    "action": "...",
                    "action_input": "...",
                    "observation": "...",
                    "state": "Valid Action"
                },
                ...
            ]
        """
        if not isinstance(logs, dict):
            return []

        raw_steps = logs.get("trajectroy_log", [])
        if not isinstance(raw_steps, list):
            return []

        step_records: List[Dict[str, Any]] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            step_records.append(self._normalize_step_record(raw_step))

        return step_records

    def _normalize_step_record(self, raw_step: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step": self._as_text(raw_step.get("step")),
            "thought": self._as_text(raw_step.get("thought")),
            "action": self._as_text(raw_step.get("action")),
            "action_input": self._as_text(raw_step.get("action_input")),
            "observation": self._truncate_text(
                raw_step.get("observation"),
                self.max_step_observation_chars,
            ),
            "state": self._as_text(raw_step.get("state")),
        }

    def _build_step_context(
        self,
        node: Dict[str, Any],
        step_record: Dict[str, Any],
        previous_step_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if previous_step_summaries is None:
            previous_step_summaries = []

        return {
            "node": {
                "id": self._as_text(node.get("id")),
                "task": self._as_text(node.get("task")),
                "agent": self._as_text(node.get("agent")),
                "deps": self._normalize_list_of_text(node.get("deps", [])),
                "node_contract": self._as_text(node.get("node_contract")),
            },
            "current_step": step_record,
            "previous_step_summaries": previous_step_summaries,
        }

    # -------------------------------------------------------------------------
    # Step level LLM analysis
    # -------------------------------------------------------------------------

    def _analyze_step(
        self,
        node: Dict[str, Any],
        step_record: Dict[str, Any],
        previous_step_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        step_context = self._build_step_context(
            node=node,
            step_record=step_record,
            previous_step_summaries=previous_step_summaries,
        )

        prompt = self._build_step_analysis_prompt(step_context)

        print("\n" + "=" * 80)
        print(f"[STEP-VERIFY] start step={step_record.get('step', '')}")
        print(f"[STEP-VERIFY] node_id={node.get('id', '')}")
        print(f"[STEP-VERIFY] task={node.get('task', '')}")
        print(f"[STEP-VERIFY] thought={step_record.get('thought', '')}")
        print(f"[STEP-VERIFY] action={step_record.get('action', '')}")
        print(f"[STEP-VERIFY] action_input={step_record.get('action_input', '')}")
        print(f"[STEP-VERIFY] observation={step_record.get('observation', '')}")
        print(f"[STEP-VERIFY] state={step_record.get('state', '')}")
        print(
            f"[STEP-VERIFY] previous_step_summaries="
            f"{json.dumps(previous_step_summaries or [], ensure_ascii=False, indent=2)}"
        )

        

        last_error: Optional[Exception] = None
        last_raw_text: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                print(f"[STEP-VERIFY] attempt={attempt + 1} calling step_llm_json_call ...")
                raw = self.step_llm_json_call(prompt)
                # time.sleep(100)
                
                print(f"[_analyze_step] raw: {raw}", flush=True)
                # time.sleep(100)

                raw_text = ""
                raw_prompt = prompt
                if isinstance(raw, dict):
                    raw_text = self._as_text(raw.get("_verifier_raw_output", ""))
                    raw_prompt = self._as_text(raw.get("_verifier_prompt", prompt)) or prompt
                else:
                    raw_text = self._as_text(raw)

                last_raw_text = raw_text

                print(f"[STEP-VERIFY] raw_output_type={type(raw).__name__}", flush=True)
                print("[STEP-VERIFY] raw type:", type(raw), flush=True)
                print("[STEP-VERIFY] raw repr:", repr(raw)[:1000], flush=True)
                try:
                    print(
                        f"[STEP-VERIFY] raw_output="
                        f"{json.dumps(raw, ensure_ascii=False, indent=2, default=str)}"
                    )
                except Exception:
                    print(f"[STEP-VERIFY] raw_output={raw}")

                validated = self._validate_step_diagnosis(raw, step_record=step_record)

                # raw / prompt / input step を sidecar として保持
                validated = dict(validated)
                validated["_verification_log"] = {
                    "kind": "step_analysis",
                    "step": step_record.get("step", ""),
                    "input_step_record": step_record,
                    "previous_step_summaries": previous_step_summaries or [],
                    "prompt": raw_prompt,
                    "raw_llm_output": raw_text,
                    "parsed_step_diagnosis": {
                        k: v for k, v in validated.items()
                        if k != "_verification_log"
                    },
                }

                print(
                    f"[STEP-VERIFY] validated_step_diagnosis="
                    f"{json.dumps(validated, ensure_ascii=False, indent=2, default=str)}"
                )
                print(f"[STEP-VERIFY] done step={step_record.get('step', '')}")
                print("=" * 80 + "\n")

                return validated

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "_analyze_step attempt %d failed for step=%s: %s",
                    attempt + 1,
                    step_record.get("step", ""),
                    repr(exc),
                )
                print(
                    f"[STEP-VERIFY] attempt={attempt + 1} failed "
                    f"for step={step_record.get('step', '')}"
                )
                print(f"[STEP-VERIFY] error={repr(exc)}")

        logger.error(
            "_analyze_step exhausted retries for step=%s. Returning fallback step diagnosis.",
            step_record.get("step", ""),
        )
        print(
            f"[STEP-VERIFY] exhausted retries for step={step_record.get('step', '')}, "
            "returning fallback"
        )

        fallback = self._build_fallback_step_diagnosis(step_record, error=last_error)
        fallback = dict(fallback)
        fallback["_verification_log"] = {
            "kind": "step_analysis",
            "step": step_record.get("step", ""),
            "input_step_record": step_record,
            "previous_step_summaries": previous_step_summaries or [],
            "prompt": prompt,
            "raw_llm_output": last_raw_text,
            "parsed_step_diagnosis": {
                k: v for k, v in fallback.items()
                if k != "_verification_log"
            },
            "fallback_error": self._as_text(repr(last_error) if last_error is not None else ""),
        }

        print(
            f"[STEP-VERIFY] fallback_step_diagnosis="
            f"{json.dumps(fallback, ensure_ascii=False, indent=2, default=str)}"
        )
        print("=" * 80 + "\n")

        return fallback

    def _build_step_analysis_prompt(self, step_context: Dict[str, Any]) -> str:
        step_context_json = self._safe_json_dumps(step_context)

        return f"""
You are a step level verification agent.

You will be given:
- node metadata
- one current step from the node trajectory
- a short summary of previous steps

Your job is to analyze only the current step.

You must determine:
1. whether this step is a success, warning, or failure
2. whether this step is a plausible root cause candidate for the node
3. what evidence appears in this step
4. what the local diagnosis is
5. what the suggested next step should be

Evidence Type Assignment Guide:

Use "Failed Tool Execution" when the step contains an attempted tool call that does not execute successfully.
Typical signals:
- the observation contains an explicit runtime error
- the tool crashes
- the tool returns an execution failure message
- the step state is invalid because the tool call failed
Do not use this label for a missing prerequisite alone, unless the main issue is that the tool itself actually failed during execution.

Use "Unsupported Thought or Assumption" when the reasoning shifts from evidence-based solving to assumption-based solving.
Typical signals:
- the thought says "I will assume ..."
- the thought says "I will use my knowledge ..."
- the answer is based on plausibility or general knowledge rather than retrieved evidence
Use this even if a tool failure also occurs, as long as the step clearly contains unsupported reasoning.

Use "Malformed or Unsupported Finalization" when the final answer or Finish content is not a valid supported output.
Typical signals:
- the final answer contains unsupported claims
- the final answer is malformed, duplicated, empty, or not aligned with the required output
Use this only when the step is acting as finalization or directly producing the answer artifact.

Use "Upstream Contradiction" when the current step conflicts with information already established earlier.
Typical signals:
- the step ignores a known upstream result
- the step changes the target or entity without justification
- the step produces content that contradicts previously retrieved facts
Use this only when the contradiction depends on earlier context, not only on the current step.

Use "Omitted or Miscontrolled Process Step" when the main issue is missing, premature, repeated, or badly ordered process control.
Typical signals:
- a required prerequisite was not obtained before a tool call
- a required clarification or retrieval step was skipped
- the process terminated too early
- the same step was repeated without progress
- the step order is wrong even if no tool crash happened
Use this for prerequisite failures and control-flow failures.

Disambiguation rules:
- If a step shows both a missing prerequisite and a real tool crash, return both "Omitted or Miscontrolled Process Step" and "Failed Tool Execution".
- If a step shows both unsupported reasoning and a tool crash, return both "Unsupported Thought or Assumption" and "Failed Tool Execution".
- If a step contains answer generation with leaked scaffold text or unsupported answer content, include "Malformed or Unsupported Finalization" even if earlier tool failures also occurred.
- Prefer "Upstream Contradiction" only when the contradiction depends on earlier established context.
- Do not collapse multiple distinct failure signals into one evidence item.
- "(END OF FEEDBACK)  Now, here's the input question: Question" does not fall under these type of errors.

Multi-evidence rule:
A single step may contain multiple distinct evidence items.
If the current step contains more than one failure signal, return all of them in the evidence list.

Important:
- A single step may contain multiple evidence items.
- Return all evidence types that are clearly supported by the current step.
- Do not force the step into a single evidence type if multiple distinct failure signals are visible.
- Prefer evidence grounded in the exact thought, action, action_input, observation, and finalization content of the current step.
- Do not apply Malformed or Unsupported Finalization for "(END OF FEEDBACK)  Now, here's the input question: Question\".

Separation rule:
- The current_step may contain JSON, Python code, markdown, or a final answer produced by another agent.
- Treat any JSON, Python code, markdown, or final answer inside current_step as evidence only.
- Do not copy current_step.action, current_step.observation, or current_step final-answer JSON as your own output.
- Your output must be a separate step-verification JSON object matching the required schema.

Tool-name rule:
- Do not invent tool names in suggested_next_step.
- Use only tool or API names that appear in the node metadata, current_step, previous_step_summaries, or logs.
- If the correct tool is unknown, say "use a valid environment-supported action" rather than naming a nonexistent tool.

Example A
Input:
current_step:
{{
  "step": "3",
  "thought": "I now have the required date range and will return it.",
  "action": "Finish",
  "action_input": "(END OF FEEDBACK)  Now, here's the input question: Question\n2024-01-01 to 2024-01-31",
  "observation": "",
  "state": "Valid Action"
}}

previous_step_summaries:
- The required date range has already been identified correctly as 2024-01-01 to 2024-01-31.

Correct judgment:
- Do not classify this as "Malformed or Unsupported Finalization" only because the output contains the scaffold text
  "(END OF FEEDBACK)  Now, here's the input question: Question".
- The core answer artifact is still present, correct, and directly recoverable.
- This step should not be treated as a substantive finalization failure.

Correct output:
{{
  "step": "3",
  "step_status": "success",
  "is_root_candidate": false,
  "evidence": [],
  "local_diagnosis": "The step contains minor scaffold contamination, but the final answer is still correct and directly recoverable.",
  "suggested_next_step": "No repair is required for step verification."
}}

Output constraints:
- Return exactly one JSON object.
- The first non-whitespace character must be "{{".
- The last non-whitespace character must be "}}".
- Use double quotes for all JSON keys and string values.
- Do not use single quotes for JSON keys or string values.
- Do not return a Python dict.
- Do not include Markdown fences.
- Do not include a second JSON object.
- Do not include explanations before or after the JSON.

Return JSON only.
Do not output Markdown.
Do not output code fences.
Do not apply Malformed or Unsupported Finalization for "(END OF FEEDBACK)  Now, here's the input question: Question\".


Required output schema:
{{
  "step": "3",
  "step_status": "success | warning | failure",
  "is_root_candidate": true,
  "evidence": [
    {{
      "evidence_type": "Failed Tool Execution",
      "snippet": "short snippet from thought or action or action_input or observation",
      "interpretation": "why this snippet matters locally"
    }}
  ],
  "local_diagnosis": "short local diagnosis for this step",
  "suggested_next_step": "short suggestion for what should happen next"
}}

Step context:
{step_context_json}
""".strip()

    def _extract_step_diagnosis_dict(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        """
        Accepts:
        - dict
        - str containing only JSON
        - str containing extra text before/after JSON
        - str containing fenced JSON

        Strategy:
        - Prefer the last schema-matching JSON object in the text.
        - Restrict matches to the step-diagnosis shape only.

        Returns:
        - parsed dict for the step diagnosis

        Raises:
        - ValueError if no step-diagnosis-shaped dict can be recovered
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_step_diagnosis_dict: start type=%s preview=%s",
                type(raw).__name__,
                self._debug_preview(raw),
            )

        if isinstance(raw, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_step_diagnosis_dict: raw is already dict keys=%s",
                    sorted(raw.keys()),
                )
            if self._looks_like_step_diagnosis(raw):
                return raw
            raise ValueError(
                "step diagnosis dict does not match required schema; "
                f"keys={sorted(raw.keys())}"
            )

        if isinstance(raw, bytes):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_step_diagnosis_dict: decoding bytes len=%d",
                    len(raw),
                )
            raw = raw.decode("utf-8", errors="replace")

        if not isinstance(raw, str):
            raise ValueError(
                f"step diagnosis must be a dict or str, got {type(raw).__name__}"
            )

        text = raw.strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_step_diagnosis_dict: normalized text len=%d preview=%s",
                len(text),
                self._debug_preview(text),
            )

        if not text:
            raise ValueError("step diagnosis text is empty")

        # 1. Fast path: the entire text is already a JSON object
        obj = self._try_load_json_dict(text)
        if obj is not None:
            if self._looks_like_step_diagnosis(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_step_diagnosis_dict: full-text parse succeeded keys=%s",
                        sorted(obj.keys()),
                    )
                return obj
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_step_diagnosis_dict: full-text parse produced dict but schema mismatch keys=%s",
                    sorted(obj.keys()),
                )

        # 2. Fenced JSON blocks, searched from the end
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_step_diagnosis_dict: found %d fenced block(s)",
                len(fenced_blocks),
            )

        for reverse_idx, fenced in enumerate(reversed(fenced_blocks), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_step_diagnosis_dict: trying fenced candidate from end #%d len=%d preview=%s",
                    reverse_idx,
                    len(fenced),
                    self._debug_preview(fenced),
                )
            obj = self._try_load_json_dict(fenced)
            if obj is not None and self._looks_like_step_diagnosis(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_step_diagnosis_dict: fenced parse succeeded keys=%s",
                        sorted(obj.keys()),
                    )
                return obj

        # 3. Balanced {...} candidates, searched from the end
        candidates = list(self._iter_balanced_json_object_candidates(text))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_step_diagnosis_dict: found %d balanced candidate(s)",
                len(candidates),
            )

        for reverse_idx, candidate in enumerate(reversed(candidates), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_step_diagnosis_dict: trying candidate from end #%d preview=%s",
                    reverse_idx,
                    self._debug_preview(candidate),
                )

            obj = self._try_load_json_dict(candidate)
            if obj is None:
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_step_diagnosis_dict: candidate from end #%d parsed as dict keys=%s",
                    reverse_idx,
                    sorted(obj.keys()),
                )

            if self._looks_like_step_diagnosis(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_step_diagnosis_dict: selected candidate from end #%d as step diagnosis",
                        reverse_idx,
                    )
                return obj

        preview = text[:300].replace("\n", "\\n")
        raise ValueError(
            "could not recover a step-diagnosis-shaped dict from text; "
            f"prefix={preview!r}"
        )


    def _validate_step_diagnosis(
        self,
        raw: Any,
        step_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_dict = self._extract_step_diagnosis_dict(raw)

        step = self._as_text(raw_dict.get("step")) or self._as_text(step_record.get("step"))

        step_status = self._as_text(raw_dict.get("step_status")).lower()
        if step_status not in {"success", "warning", "failure"}:
            raise ValueError("step_status must be one of: success, warning, failure")

        is_root_candidate = raw_dict.get("is_root_candidate")
        if not isinstance(is_root_candidate, bool):
            raise ValueError("is_root_candidate must be a bool")

        evidence_raw = raw_dict.get("evidence", [])
        if not isinstance(evidence_raw, list):
            evidence_raw = []

        evidence: List[Dict[str, str]] = []
        for item in evidence_raw:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "evidence_type": self._as_text(item.get("evidence_type")),
                    "snippet": self._as_text(item.get("snippet")),
                    "interpretation": self._as_text(item.get("interpretation")),
                }
            )

        return {
            "step": step,
            "step_status": step_status,
            "is_root_candidate": is_root_candidate,
            "evidence": evidence,
            "local_diagnosis": self._as_text(raw_dict.get("local_diagnosis")),
            "suggested_next_step": self._as_text(raw_dict.get("suggested_next_step")),
        }

    def _build_fallback_step_diagnosis(
        self,
        step_record: Dict[str, Any],
        error: Optional[Exception],
    ) -> Dict[str, Any]:
        return {
            "step": self._as_text(step_record.get("step")),
            "step_status": "failure",
            "is_root_candidate": True,
            "verifier_failed": True,
            "evidence": [
                {
                    "evidence_type": "Omitted or Miscontrolled Process Step",
                    "snippet": self._as_text(
                        step_record.get("observation")
                        or step_record.get("action")
                        or repr(error)
                        or "unknown step verifier failure"
                    ),
                    "interpretation": (
                        "The step-level verifier failed to parse its own output. "
                        "This is a verification-infrastructure failure, so the step "
                        "requires manual inspection or a rerun with strict JSON parsing."
                    ),
                }
            ],
            "local_diagnosis": (
                "Verifier parse failure prevented reliable step diagnosis."
            ),
            "suggested_next_step": (
                "Rerun verification after fixing JSON parsing, then inspect this step "
                "against the original trajectory evidence."
            ),
        }

    # -------------------------------------------------------------------------
    # Node level orchestration
    # -------------------------------------------------------------------------

    def _analyze_node_steps(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        step_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        step_diagnoses: List[Dict[str, Any]] = []
        previous_step_summaries: List[Dict[str, Any]] = []

        for step_record in step_records:
            step_result = self._analyze_step(
                node=node,
                step_record=step_record,
                previous_step_summaries=previous_step_summaries,
            )

            # print(f"step_result: {step_result}", flush=True)
            # time.sleep(100)


            step_diagnoses.append(step_result)

            # 次 step に渡す summary には raw sidecar を含めない
            previous_step_summaries.append(
                {
                    "step": self._as_text(step_result.get("step")),
                    "step_status": self._as_text(step_result.get("step_status")),
                    "is_root_candidate": bool(step_result.get("is_root_candidate", False)),
                    "local_diagnosis": self._as_text(step_result.get("local_diagnosis")),
                    "suggested_next_step": self._as_text(step_result.get("suggested_next_step")),
                }
            )

        return step_diagnoses

    def _summarize_step_diagnoses(
        self,
        step_diagnoses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        first_root_candidate = ""
        failure_steps: List[Dict[str, Any]] = []
        warning_steps: List[Dict[str, Any]] = []
        success_steps: List[Dict[str, Any]] = []

        for d in step_diagnoses:
            item = {
                "step": d.get("step", ""),
                "step_status": d.get("step_status", ""),
                "is_root_candidate": d.get("is_root_candidate", False),
                "local_diagnosis": d.get("local_diagnosis", ""),
                "suggested_next_step": d.get("suggested_next_step", ""),
                "evidence": d.get("evidence", []),
            }

            if item["is_root_candidate"] and not first_root_candidate:
                first_root_candidate = self._as_text(item["step"])

            if item["step_status"] == "failure":
                failure_steps.append(item)
            elif item["step_status"] == "warning":
                warning_steps.append(item)
            else:
                success_steps.append(item)

        return {
            "first_root_candidate_step": first_root_candidate,
            "num_steps": len(step_diagnoses),
            "num_failure_steps": len(failure_steps),
            "num_warning_steps": len(warning_steps),
            "num_success_steps": len(success_steps),
            "failure_steps": failure_steps,
            "warning_steps": warning_steps,
            "success_steps": success_steps,
        }

    def _build_node_aggregation_context(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        step_diagnoses: List[Dict[str, Any]],
        summarized_steps: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "node": {
                "id": self._as_text(node.get("id")),
                "task": self._as_text(node.get("task")),
                "agent": self._as_text(node.get("agent")),
                "deps": self._normalize_list_of_text(node.get("deps", [])),
                "node_contract": self._as_text(node.get("node_contract")),
            },
            "final_answer": self._as_text(logs.get("final_answer", "")),
            "reviews": logs.get("reviews", []),
            "step_diagnoses": step_diagnoses,
            "summarized_steps": summarized_steps,
        }

    def _aggregate_node_diagnosis(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        step_diagnoses: List[Dict[str, Any]],
        summarized_steps: Dict[str, Any],
    ) -> Dict[str, Any]:
        aggregation_context = self._build_node_aggregation_context(
            node=node,
            logs=logs,
            step_diagnoses=step_diagnoses,
            summarized_steps=summarized_steps,
        )

        prompt = self._build_node_aggregation_prompt(aggregation_context)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_json_call(prompt)
                return self._validate_verify_output(raw, node=node, logs=logs)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "_aggregate_node_diagnosis attempt %d failed: %s",
                    attempt + 1,
                    repr(exc),
                )

        logger.error("_aggregate_node_diagnosis exhausted retries. Returning safe fallback.")
        return self._build_fallback_result(
            node=node,
            logs=logs,
            error=last_error,
        )

    def _build_node_aggregation_prompt(
        self,
        aggregation_context: Dict[str, Any],
    ) -> str:
        aggregation_context_json = self._safe_json_dumps(aggregation_context)

        return f"""
You are a node level verification aggregation agent.

You will be given:
- node metadata
- the node final answer
- optional review text
- step level diagnoses for each step in the node trajectory
- a compact summary of those step level diagnoses

Your job is to aggregate the step level findings into one node level verification result.

You must determine:
1. whether recovery is needed before downstream use
2. the task intent and completion condition
3. the primary root failure
4. downstream symptoms
5. the impact on the node contract
6. a concrete recovery suggestion

Important:
- Use the step level diagnoses as evidence.
- Prefer the earliest structural break when identifying the main failure chain.
- Distinguish between root failure and downstream symptoms.
- If part of the node output is still usable, reflect that in impact_on_node_contract.
- Return the same final schema used by the current verifier API.

Return JSON only.
Do not output Markdown.
Do not output code fences.

Required output schema:
{{
  "recovery_flag": true,
  "diagnosis": {{
    "task_intent": "short description of what the node was supposed to achieve",
    "completion_condition": "what must be true for the node_contract to be satisfied",
    "root_failure": {{
      "category": "one of the five evidence categories, or 'none'",
      "where": ["step ids if available"],
      "why": "why the failure occurred",
      "how": "how it manifested in the node execution"
    }},
    "supporting_evidence": [
      {{
        "step": "step id or empty string",
        "evidence_type": "one of the five evidence categories",
        "snippet": "short copied or paraphrased snippet",
        "interpretation": "why this snippet matters"
      }}
    ],
    "downstream_symptoms": [
      {{
        "category": "short symptom category",
        "where": ["step ids if available"],
        "description": "downstream symptom description"
      }}
    ],
    "impact_on_node_contract": "how the failure affected satisfaction of node_contract"
  }},
  "recovery_suggestion": {{
    "recovery_goal": "what should be restored",
    "actions": [
      "first concrete action",
      "second concrete action"
    ],
    "do_not": [
      "do not do X",
      "do not do Y"
    ],
    "handoff_payload": {{
      "retry_from_step": "short anchor such as 'sensor retrieval' or ''",
      "required_inputs": ["..."],
      "notes_for_recovery_node": "brief note"
    }}
  }}
}}

Aggregation context:
{aggregation_context_json}
""".strip()

    # -------------------------------------------------------------------------
    # Whole node prompt
    # -------------------------------------------------------------------------

    def _build_verify_prompt(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> str:
        logs_json = self._safe_json_dumps(logs)
        if len(logs_json) > self.max_log_chars:
            logs_json = logs_json[: self.max_log_chars] + "\n...<TRUNCATED_LOG>"

        node_json = self._safe_json_dumps(node)

        return f"""
You are a verification failure mode analysis agent.

You will be given one executed node and its local execution log.
Your job is to decide whether recovery is needed, and if so, produce a diagnosis and a recovery suggestion.

This is not a simple label classification task.
You must read the full node log and infer the failure mechanism from the actual trajectory.

Analyze the node in the following way:

1. Reconstruct what the node was supposed to achieve.
2. Infer the completion condition from the task and node_contract.
3. Read the local log step by step.
4. Use the following evidence categories as analysis lenses:
   - Failed Tool Execution
   - Unsupported Thought or Assumption
   - Malformed or Unsupported Finalization
   - Upstream Contradiction
   - Omitted or Miscontrolled Process Step
5. Identify the root failure if one exists.
6. Separate root failure from downstream symptoms.
7. Decide whether the node needs recovery before downstream use.
8. Produce an actionable recovery suggestion for a recovery node.

Important rules:
- Do not rely only on abstract labels.
- Ground the diagnosis in the provided log.
- Whenever possible, copy or quote short snippets from the log into supporting_evidence.snippet.
- If the node appears successful and no recovery is needed, set recovery_flag=false.
- If recovery_flag=false, still provide a brief diagnosis, but recovery_suggestion may remain lightweight.
- Be concrete. The recovery suggestion must tell a recovery node what to do next.
- Prefer a single primary root failure over many vague ones.

Return JSON only.
Do not output Markdown.
Do not output code fences.
Do not output explanations before or after the JSON.

Required output schema:
{{
  "recovery_flag": true,
  "diagnosis": {{
    "task_intent": "short description of what the node was supposed to achieve",
    "completion_condition": "what must be true for the node_contract to be satisfied",
    "root_failure": {{
      "category": "one of the five evidence categories, or 'none'",
      "where": ["step ids such as 2 or 3 if available, otherwise []"],
      "why": "why the failure occurred",
      "how": "how it manifested in the node execution"
    }},
    "supporting_evidence": [
      {{
        "step": "step id or empty string",
        "evidence_type": "one of the five evidence categories",
        "snippet": "short copied or paraphrased snippet from the log",
        "interpretation": "why this snippet matters"
      }}
    ],
    "downstream_symptoms": [
      {{
        "category": "short symptom category",
        "where": ["step ids if available"],
        "description": "downstream symptom description"
      }}
    ],
    "impact_on_node_contract": "how the failure affected satisfaction of node_contract"
  }},
  "recovery_suggestion": {{
    "recovery_goal": "what should be restored",
    "actions": [
      "first concrete action",
      "second concrete action"
    ],
    "do_not": [
      "do not do X",
      "do not do Y"
    ],
    "handoff_payload": {{
      "retry_from_step": "short anchor such as 'sensor retrieval' or ''",
      "required_inputs": ["..."],
      "notes_for_recovery_node": "brief note"
    }}
  }}
}}

Node:
{node_json}

Local log:
{logs_json}
""".strip()

    # -------------------------------------------------------------------------
    # Final output validation
    # -------------------------------------------------------------------------

    def _extract_node_verification_dict(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        """
        Accepts:
        - dict
        - str containing only JSON
        - str containing extra text before/after JSON
        - str containing fenced JSON

        Strategy:
        - Prefer the last schema-matching JSON object in the text.
        - Restrict matches to the whole-node verification shape only.

        Returns:
        - parsed dict for the whole-node verification output

        Raises:
        - ValueError if no node-verification-shaped dict can be recovered
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_verification_dict: start type=%s preview=%s",
                type(raw).__name__,
                self._debug_preview(raw),
            )

        if isinstance(raw, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_verification_dict: raw is already dict keys=%s",
                    sorted(raw.keys()),
                )
            if self._looks_like_node_verification(raw):
                return raw
            raise ValueError(
                "node verification dict does not match required schema; "
                f"keys={sorted(raw.keys())}"
            )

        if isinstance(raw, bytes):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_verification_dict: decoding bytes len=%d",
                    len(raw),
                )
            raw = raw.decode("utf-8", errors="replace")

        if not isinstance(raw, str):
            raise ValueError(
                f"node verification output must be a dict or str, got {type(raw).__name__}"
            )

        text = raw.strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_verification_dict: normalized text len=%d preview=%s",
                len(text),
                self._debug_preview(text),
            )

        if not text:
            raise ValueError("node verification text is empty")

        # 1. Fast path: the entire text is already a JSON object
        obj = self._try_load_json_dict(text)
        if obj is not None:
            if self._looks_like_node_verification(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_node_verification_dict: full-text parse succeeded keys=%s",
                        sorted(obj.keys()),
                    )
                return obj
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_verification_dict: full-text parse produced dict but schema mismatch keys=%s",
                    sorted(obj.keys()),
                )

        # 2. Fenced JSON blocks, searched from the end
        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_verification_dict: found %d fenced block(s)",
                len(fenced_blocks),
            )

        for reverse_idx, fenced in enumerate(reversed(fenced_blocks), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_verification_dict: trying fenced candidate from end #%d len=%d preview=%s",
                    reverse_idx,
                    len(fenced),
                    self._debug_preview(fenced),
                )
            obj = self._try_load_json_dict(fenced)
            if obj is not None and self._looks_like_node_verification(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_node_verification_dict: fenced parse succeeded keys=%s",
                        sorted(obj.keys()),
                    )
                return obj

        # 3. Balanced {...} candidates, searched from the end
        candidates = list(self._iter_balanced_json_object_candidates(text))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_extract_node_verification_dict: found %d balanced candidate(s)",
                len(candidates),
            )

        for reverse_idx, candidate in enumerate(reversed(candidates), start=1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_verification_dict: trying candidate from end #%d preview=%s",
                    reverse_idx,
                    self._debug_preview(candidate),
                )

            obj = self._try_load_json_dict(candidate)
            if obj is None:
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "_extract_node_verification_dict: candidate from end #%d parsed as dict keys=%s",
                    reverse_idx,
                    sorted(obj.keys()),
                )

            if self._looks_like_node_verification(obj):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "_extract_node_verification_dict: selected candidate from end #%d as node verification",
                        reverse_idx,
                    )
                return obj

        preview = text[:300].replace("\n", "\\n")
        raise ValueError(
            "could not recover a node-verification-shaped dict from text; "
            f"prefix={preview!r}"
        )


    def _validate_verify_output(
        self,
        raw: Any,
        node: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_dict = self._extract_node_verification_dict(raw)

        if "recovery_flag" not in raw_dict:
            raise ValueError("verification output must contain 'recovery_flag'")
        if not isinstance(raw_dict["recovery_flag"], bool):
            raise ValueError("'recovery_flag' must be a bool")

        diagnosis = raw_dict.get("diagnosis", {})
        if not isinstance(diagnosis, dict):
            diagnosis = {}

        recovery_suggestion = raw_dict.get("recovery_suggestion", {})
        if not isinstance(recovery_suggestion, dict):
            recovery_suggestion = {}

        diagnosis_norm = {
            "task_intent": self._as_text(diagnosis.get("task_intent")),
            "completion_condition": self._as_text(diagnosis.get("completion_condition")),
            "root_failure": self._normalize_root_failure(diagnosis.get("root_failure")),
            "supporting_evidence": self._normalize_supporting_evidence(
                diagnosis.get("supporting_evidence", [])
            ),
            "downstream_symptoms": self._normalize_downstream_symptoms(
                diagnosis.get("downstream_symptoms", [])
            ),
            "impact_on_node_contract": self._as_text(
                diagnosis.get("impact_on_node_contract")
            ),
        }

        recovery_norm = {
            "recovery_goal": self._as_text(recovery_suggestion.get("recovery_goal")),
            "actions": self._normalize_list_of_text(
                recovery_suggestion.get("actions", [])
            ),
            "do_not": self._normalize_list_of_text(
                recovery_suggestion.get("do_not", [])
            ),
            "handoff_payload": self._normalize_handoff_payload(
                recovery_suggestion.get("handoff_payload", {})
            ),
        }

        if not diagnosis_norm["task_intent"]:
            diagnosis_norm["task_intent"] = self._infer_task_intent_from_node(node)
        if not diagnosis_norm["completion_condition"]:
            diagnosis_norm["completion_condition"] = self._infer_completion_condition_from_node(node)

        if not diagnosis_norm["root_failure"]["category"]:
            diagnosis_norm["root_failure"]["category"] = "none"

        if not diagnosis_norm["impact_on_node_contract"]:
            diagnosis_norm["impact_on_node_contract"] = (
                "the verifier could not determine the contract impact reliably"
            )

        recovery_flag = raw_dict["recovery_flag"]

        if recovery_flag:
            if not recovery_norm["recovery_goal"]:
                recovery_norm["recovery_goal"] = (
                    "restore the node to a state where its node_contract can be satisfied"
                )
            if not recovery_norm["actions"]:
                recovery_norm["actions"] = [
                    "inspect the node diagnosis",
                    "repair the root failure before reusing this node output downstream",
                ]
        else:
            if not recovery_norm["recovery_goal"]:
                recovery_norm["recovery_goal"] = "no recovery required"
            if not recovery_norm["handoff_payload"]["notes_for_recovery_node"]:
                recovery_norm["handoff_payload"]["notes_for_recovery_node"] = (
                    "No recovery branch should be launched for this node."
                )

        return {
            "recovery_flag": recovery_flag,
            "diagnosis": diagnosis_norm,
            "recovery_suggestion": recovery_norm,
        }

    def _normalize_root_failure(self, obj: Any) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            obj = {}

        return {
            "category": self._as_text(obj.get("category")),
            "where": self._normalize_list_of_text(obj.get("where", [])),
            "why": self._as_text(obj.get("why")),
            "how": self._as_text(obj.get("how")),
        }

    def _normalize_supporting_evidence(self, items: Any) -> List[Dict[str, str]]:
        if not isinstance(items, list):
            return []

        out: List[Dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "step": self._as_text(item.get("step")),
                    "evidence_type": self._as_text(item.get("evidence_type")),
                    "snippet": self._as_text(item.get("snippet")),
                    "interpretation": self._as_text(item.get("interpretation")),
                }
            )
        return out

    def _normalize_downstream_symptoms(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        out: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "category": self._as_text(item.get("category")),
                    "where": self._normalize_list_of_text(item.get("where", [])),
                    "description": self._as_text(item.get("description")),
                }
            )
        return out
    
    def _normalize_failure_timeline(self, items: Any) -> List[Dict[str, str]]:
        if not isinstance(items, list):
            return []

        allowed_roles = {
            "prerequisite_success",
            "first_structural_break",
            "downstream_symptom",
            "finalization",
            "not_relevant",
        }

        out: List[Dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            role = self._as_text(item.get("role_in_failure_chain"))
            if role not in allowed_roles:
                role = "not_relevant"

            status = self._as_text(item.get("step_status")).lower()
            if status not in {"success", "warning", "failure"}:
                status = ""

            out.append(
                {
                    "step": self._as_text(item.get("step")),
                    "step_status": status,
                    "role_in_failure_chain": role,
                    "summary": self._as_text(item.get("summary")),
                }
            )

        return out

    def _normalize_failure_chain(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        allowed_stages = {"root", "propagation", "symptom", "finalization"}

        out: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            stage = self._as_text(item.get("stage"))
            if stage not in allowed_stages:
                stage = "symptom"

            out.append(
                {
                    "stage": stage,
                    "steps": self._normalize_list_of_text(item.get("steps", [])),
                    "description": self._as_text(item.get("description")),
                }
            )

        return out

    def _normalize_output_items(
        self,
        items: Any,
        expected_usable: bool,
    ) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        out: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            usable = item.get("usable")
            if not isinstance(usable, bool):
                usable = expected_usable

            out.append(
                {
                    "name": self._as_text(item.get("name")),
                    "value": self._as_text(item.get("value")),
                    "usable": usable,
                    "reason": self._as_text(item.get("reason")),
                }
            )

        return out

    def _normalize_confidence(self, value: Any) -> str:
        confidence = self._as_text(value).lower()
        if confidence not in {"high", "medium", "low"}:
            return "medium"
        return confidence
    
    def _normalize_handoff_payload(self, obj: Any) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            obj = {}

        return {
            "retry_from_step": self._as_text(obj.get("retry_from_step")),
            "required_inputs": self._normalize_list_of_text(obj.get("required_inputs", [])),
            "notes_for_recovery_node": self._as_text(obj.get("notes_for_recovery_node")),
        }

    # -------------------------------------------------------------------------
    # Safe fallback
    # -------------------------------------------------------------------------

    def _build_fallback_result(
        self,
        node: Dict[str, Any],
        logs: Dict[str, Any],
        error: Optional[Exception],
    ) -> Dict[str, Any]:
        return {
            "recovery_flag": True,
            "diagnosis": {
                "task_intent": self._infer_task_intent_from_node(node),
                "completion_condition": self._infer_completion_condition_from_node(node),
                "root_failure": {
                    "category": "Omitted or Miscontrolled Process Step",
                    "where": [],
                    "why": "the verification step itself failed to produce a reliable diagnosis",
                    "how": self._as_text(repr(error) if error is not None else "unknown verifier failure"),
                },
                "supporting_evidence": [
                    {
                        "step": "",
                        "evidence_type": "Omitted or Miscontrolled Process Step",
                        "snippet": self._as_text(repr(error) if error is not None else "unknown verifier failure"),
                        "interpretation": "the verification agent could not complete a reliable analysis",
                    }
                ],
                "downstream_symptoms": [
                    {
                        "category": "verification_unavailable",
                        "where": [],
                        "description": "downstream use of this node would proceed without a reliable verification result",
                    }
                ],
                "impact_on_node_contract": "the node may or may not satisfy node_contract, but the verifier could not determine that reliably",
            },
            "recovery_suggestion": {
                "recovery_goal": "obtain a reliable diagnosis before downstream use",
                "actions": [
                    "inspect the parent node log directly",
                    "repair the verifier or rerun verification with complete logs",
                ],
                "do_not": [
                    "do not treat the missing verifier diagnosis as a successful verification pass",
                ],
                "handoff_payload": {
                    "retry_from_step": "verification",
                    "required_inputs": [],
                    "notes_for_recovery_node": "verifier fallback triggered because the verification call failed",
                },
            },
        }

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------

    def _infer_task_intent_from_node(self, node: Dict[str, Any]) -> str:
        task = self._as_text(node.get("task"))
        if task:
            return task
        return "analyze whether the node achieved its intended task"

    def _infer_completion_condition_from_node(self, node: Dict[str, Any]) -> str:
        contract = self._as_text(node.get("node_contract"))
        if contract:
            return f"the node output satisfies the following contract: {contract}"
        return "the node output satisfies its expected contract"

    def _normalize_list_of_text(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [self._as_text(x) for x in value if self._as_text(x)]

    def _as_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _safe_json_dumps(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(obj)
    











import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Iterator

# -----------------------------------------------------------------------------
# Basic I/O
# -----------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def as_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


# -----------------------------------------------------------------------------
# Trajectory parsing
# -----------------------------------------------------------------------------

def _looks_like_verifier_record(record: Dict[str, Any]) -> bool:
    agent_name = as_text(record.get("agent_name"))
    if agent_name == "verifier":
        return True

    logs = record.get("logs")
    if isinstance(logs, dict):
        workflow_info = logs.get("_workflow", {})
        if isinstance(workflow_info, dict):
            return as_text(workflow_info.get("kind")) == "verifier"

    return False


def _normalize_logs(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep logs as rich as possible.
    If logs are missing or not a dict, fall back to the final response.
    """
    logs = record.get("logs", {})
    if isinstance(logs, dict):
        out = dict(logs)
        if "final_answer" not in out:
            out["final_answer"] = record.get("response", "")
        return out

    return {"final_answer": record.get("response", "")}


def extract_nodes_from_trajectory(trajectory_obj: Any) -> List[Dict[str, Any]]:
    """
    Expected input:
      - most commonly: a list of history records from ConditionalWorkflow.generate_history()
      - optionally: a dict containing "history" or "trajectory"

    Output:
      [
        {
          "node": {...},
          "logs": {...},
          "raw_record": {...}
        }
      ]
    """
    if isinstance(trajectory_obj, list):
        records = trajectory_obj
    elif isinstance(trajectory_obj, dict):
        if isinstance(trajectory_obj.get("history"), list):
            records = trajectory_obj["history"]
        elif isinstance(trajectory_obj.get("trajectory"), list):
            records = trajectory_obj["trajectory"]
        else:
            raise ValueError(
                "Trajectory JSON must be a list, or a dict with key 'history' or 'trajectory'."
            )
    else:
        raise TypeError("Trajectory JSON must be a list or dict.")

    extracted: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        # Skip verifier records if they already exist in the trajectory.
        if _looks_like_verifier_record(record):
            continue

        task_number = record.get("task_number", idx + 1)
        node_id = as_text(record.get("node_id")) or f"S{task_number}"

        node = {
            "id": node_id,
            "task": as_text(record.get("task_description") or record.get("task")),
            "agent": as_text(record.get("agent_name") or record.get("agent")),
            "deps": record.get("deps", []) if isinstance(record.get("deps", []), list) else [],
            "node_contract": as_text(record.get("node_contract", "")),
        }

        logs = _normalize_logs(record)

        extracted.append(
            {
                "node": node,
                "logs": logs,
                "raw_record": record,
            }
        )

    return extracted


def _normalize_cached_diagnosis_record(
    record: Dict[str, Any],
    parent_trajectory_path: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Normalize one node-level record from a diagnosis-only cache.

    Expected node record shape:
        {
            "node_id": "S1",
            "task": "...",
            "agent": "...",
            "node_contract": "",
            "step_diagnoses": [...],
            "node_diagnosis": {...}
        }

    Returns:
        {
            "trajectory_path": "...",
            "node": {
                "id": "...",
                "task": "...",
                "agent": "...",
                "deps": [],
                "node_contract": "..."
            },
            "step_diagnoses": [...],
            "node_diagnosis": {...}
        }
    """
    if not isinstance(record, dict):
        return None

    node_diagnosis = record.get("node_diagnosis")
    if not isinstance(node_diagnosis, dict):
        return None

    node = {
        "id": as_text(record.get("node_id")),
        "task": as_text(record.get("task")),
        "agent": as_text(record.get("agent")),
        "deps": record.get("deps", []) if isinstance(record.get("deps"), list) else [],
        "node_contract": as_text(record.get("node_contract")),
    }

    return {
        "trajectory_path": as_text(record.get("trajectory_path")) or parent_trajectory_path,
        "node": node,
        "step_diagnoses": record.get("step_diagnoses", []) if isinstance(record.get("step_diagnoses"), list) else [],
        "node_diagnosis": node_diagnosis,
    }


def extract_nodes_from_diagnosis_cache(cache_obj: Any) -> List[Dict[str, Any]]:
    """
    Extract node-level diagnosis records from a diagnosis-only cache.

    Supported inputs:

    1) trajectory-level cache:
        {
            "trajectory_path": "...",
            "output_mode": "diagnosis_only",
            "results": [
                {
                    "node_id": "S1",
                    "task": "...",
                    "agent": "...",
                    "node_contract": "",
                    "step_diagnoses": [...],
                    "node_diagnosis": {...}
                }
            ]
        }

    2) folder-level cache:
        {
            "trajectory_dir": "...",
            "output_mode": "diagnosis_only",
            "results": [
                {
                    "trajectory_path": "...",
                    "output_mode": "diagnosis_only",
                    "results": [...]
                },
                ...
            ]
        }

    Returns:
        [
            {
                "trajectory_path": "...",
                "node": {...},
                "step_diagnoses": [...],
                "node_diagnosis": {...}
            },
            ...
        ]
    """
    if not isinstance(cache_obj, dict):
        raise TypeError("Diagnosis cache must be a JSON object.")

    results = cache_obj.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Diagnosis cache must contain a list field named 'results'.")

    extracted: List[Dict[str, Any]] = []
    parent_trajectory_path = as_text(cache_obj.get("trajectory_path"))

    for record in results:
        if not isinstance(record, dict):
            continue

        # Case A:
        # folder-level cache -> each record is itself a trajectory-level cache
        if isinstance(record.get("results"), list):
            sub_items = extract_nodes_from_diagnosis_cache(record)

            # inherit trajectory_path if sub-items do not have one
            record_traj_path = as_text(record.get("trajectory_path"))
            for item in sub_items:
                if not item.get("trajectory_path"):
                    item["trajectory_path"] = record_traj_path
                extracted.append(item)
            continue

        # Case B:
        # direct node-level diagnosis record
        normalized = _normalize_cached_diagnosis_record(
            record,
            parent_trajectory_path=parent_trajectory_path,
        )
        if normalized is not None:
            extracted.append(normalized)

    return extracted


def extract_nodes_from_diagnosis_cache_for_trajectory(
    cache_obj: Any,
    trajectory_path: Path,
) -> List[Dict[str, Any]]:
    """
    From a diagnosis-only cache, extract only the cached node diagnoses that
    belong to the specified trajectory_path.
    """
    all_items = extract_nodes_from_diagnosis_cache(cache_obj)

    target_path = str(trajectory_path)
    matched = [
        item for item in all_items
        if str(item.get("trajectory_path", "")) == target_path
    ]

    # Fallback: match by basename if absolute paths differ between environments
    if not matched:
        target_name = trajectory_path.name
        matched = [
            item for item in all_items
            if Path(str(item.get("trajectory_path", ""))).name == target_name
        ]

    if not matched:
        raise ValueError(
            f"No cached diagnosis records found for trajectory_path={trajectory_path}"
        )

    return matched


# -----------------------------------------------------------------------------
# LLM backend adapter
# -----------------------------------------------------------------------------

def build_llm_json_call() -> Callable[[str], Dict[str, Any]]:
    """
    Replace the body of this function with your actual LLM backend.

    The function must accept a single prompt string and return a parsed JSON dict.
    """
    def _call(prompt: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "Connect this to your actual LLM backend. "
            "It must accept a prompt string and return a parsed JSON dict."
        )
    return _call

from reactxen.utils.model_inference import watsonx_llm


def build_tracking_llm_generate(token_counter):
    def _llm_generate(prompt: str) -> str:
        resp = watsonx_llm(prompt, model_id=20)

        # token accounting
        if isinstance(resp, dict):
            token_counter["input"] += int(resp.get("input_token_count", 0) or 0)
            token_counter["output"] += int(resp.get("generated_token_count", 0) or 0)

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


# -----------------------------------------------------------------------------
# Verification runner
# -----------------------------------------------------------------------------

def run_verification_on_trajectory(
    trajectory_path: Path,
    output_path: Optional[Path] = None,
    output_mode: str = "diagnosis_only",
) -> Dict[str, Any]:
    trajectory_obj = read_json(trajectory_path)
    extracted_nodes = extract_nodes_from_trajectory(trajectory_obj)

    execution_token_counter = {"input": 0, "output": 0}
    execution_llm_generate = build_tracking_llm_generate(execution_token_counter)

    verification_agent = VerificationAgent(
        llm_generate=execution_llm_generate,
        max_retries=2,
        max_log_chars=30000,
    )

    results: List[Dict[str, Any]] = []

    for item in extracted_nodes:
        node = item["node"]
        logs = item["logs"]

        if output_mode == "diagnosis_only":
            diagnosis_result = verification_agent.diagnose_only(
                node=node,
                logs=logs,
            )
            results.append(
                {
                    "node_id": node["id"],
                    "task": node["task"],
                    "agent": node["agent"],
                    "node_contract": node["node_contract"],
                    "step_diagnoses": diagnosis_result["step_diagnoses"],
                    "node_diagnosis": diagnosis_result["diagnosis"],
                }
            )

        elif output_mode == "full_verification":
            verify_result = verification_agent.verify(
                node=node,
                logs=logs,
            )
            results.append(
                {
                    "node_id": node["id"],
                    "task": node["task"],
                    "agent": node["agent"],
                    "node_contract": node["node_contract"],
                    "verification_result": verify_result,
                }
            )

        else:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

    summary = {
        "trajectory_path": str(trajectory_path),
        "output_mode": output_mode,
        "num_nodes_analyzed": len(results),
        "execution_token_counter": execution_token_counter,
        "results": results,
    }

    if output_path is not None:
        write_json(summary, output_path)

    return summary

def run_verification_on_folder(
    trajectory_dir: Path,
    output_path: Optional[Path] = None,
    output_mode: str = "diagnosis_only",
    file_pattern: str = "*.json",
) -> Dict[str, Any]:
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"trajectory_dir does not exist: {trajectory_dir}")
    if not trajectory_dir.is_dir():
        raise NotADirectoryError(f"trajectory_dir is not a directory: {trajectory_dir}")

    trajectory_files = sorted(trajectory_dir.glob(file_pattern))

    folder_results: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, str]] = []

    for trajectory_file in trajectory_files:
        try:
            result = run_verification_on_trajectory(
                trajectory_path=trajectory_file,
                output_path=None,  # folder-level cache is written here, not per-file
                output_mode=output_mode,
            )
            folder_results.append(result)

        except Exception as exc:
            logger.exception("Failed to process trajectory file: %s", trajectory_file)
            failed_files.append(
                {
                    "trajectory_path": str(trajectory_file),
                    "error": repr(exc),
                }
            )

        # Incremental cache write after each trajectory file
        summary = {
            "trajectory_dir": str(trajectory_dir),
            "output_mode": output_mode,
            "file_pattern": file_pattern,
            "num_trajectory_files_found": len(trajectory_files),
            "num_trajectory_files_succeeded": len(folder_results),
            "num_trajectory_files_failed": len(failed_files),
            "failed_files": failed_files,
            "results": folder_results,
        }

        if output_path is not None:
            write_json(summary, output_path)

    return summary

def _build_recovery_output_path(output_dir: Path, trajectory_path: str) -> Path:
    stem = Path(trajectory_path).stem
    if stem.endswith("_trajectory"):
        stem = stem[: -len("_trajectory")]
    return output_dir / f"{stem}_recovery_only.json"


def _write_pretty_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent="\t")
        f.write("\n")


def run_recovery_on_diagnosis_cache(
    diagnosis_cache_path: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Read a diagnosis-only cache and run only the recovery decision/suggestion module.

    Output behavior:
    - one output JSON file per trajectory
    - after each node is processed, rewrite that trajectory's JSON file
    """
    cache_obj = read_json(diagnosis_cache_path)
    cached_nodes = extract_nodes_from_diagnosis_cache(cache_obj)

    # trajectory_path ごとにまとめる
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in cached_nodes:
        trajectory_path = str(item.get("trajectory_path", ""))
        grouped.setdefault(trajectory_path, []).append(item)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_summaries: List[Dict[str, Any]] = []

    for trajectory_path, items in grouped.items():
        execution_token_counter = {"input": 0, "output": 0}
        execution_llm_generate = build_tracking_llm_generate(execution_token_counter)

        verification_agent = VerificationAgent(
            llm_generate=execution_llm_generate,
            max_retries=2,
            max_log_chars=30000,
        )

        results: List[Dict[str, Any]] = []
        failed_nodes: List[Dict[str, str]] = []

        per_traj_output_path: Optional[Path] = None
        if output_dir is not None:
            per_traj_output_path = _build_recovery_output_path(output_dir, trajectory_path)

        for item in items:
            node = item["node"]
            node_id = node.get("id", "")

            try:
                node_diagnosis = item["node_diagnosis"]

                verification_result = verification_agent.recover_from_diagnosis(
                    node=node,
                    diagnosis=node_diagnosis,
                    logs={},
                )

                results.append(
                    {
                        "trajectory_path": trajectory_path,
                        "node_id": node["id"],
                        "task": node["task"],
                        "agent": node["agent"],
                        "node_contract": node["node_contract"],
                        "step_diagnoses": item.get("step_diagnoses", []),
                        "node_diagnosis": node_diagnosis,
                        "verification_result": verification_result,
                    }
                )

            except Exception as exc:
                logger.exception(
                    "Failed to process recovery_only for trajectory=%s node_id=%s",
                    trajectory_path,
                    node_id,
                )
                failed_nodes.append(
                    {
                        "trajectory_path": trajectory_path,
                        "node_id": str(node_id),
                        "error": repr(exc),
                    }
                )

            # each node ごとに、その trajectory 用 JSON 全体を書き直す
            summary = {
                "trajectory_path": trajectory_path,
                "diagnosis_cache_path": str(diagnosis_cache_path),
                "output_mode": "recovery_only",
                "num_nodes_analyzed": len(results),
                "execution_token_counter": execution_token_counter,
                "results": results,
            }

            if failed_nodes:
                summary["failed_nodes"] = failed_nodes

            if per_traj_output_path is not None:
                _write_pretty_json(per_traj_output_path, summary)

        trajectory_summaries.append(
            {
                "trajectory_path": trajectory_path,
                "output_path": str(per_traj_output_path) if per_traj_output_path is not None else "",
                "num_nodes_analyzed": len(results),
                "num_nodes_failed": len(failed_nodes),
                "execution_token_counter": execution_token_counter,
            }
        )

    return {
        "diagnosis_cache_path": str(diagnosis_cache_path),
        "output_mode": "recovery_only",
        "num_trajectories": len(grouped),
        "trajectories": trajectory_summaries,
        "output_dir": str(output_dir) if output_dir is not None else "",
    }

def run_recovery_on_trajectory(
    trajectory_path: Path,
    diagnosis_cache_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Read a diagnosis-only cache, select the records corresponding to one
    trajectory_path, and run only the recovery decision/suggestion module.

    This does not re-run:
    - step-level diagnosis
    - node-level diagnosis aggregation

    Incremental write behavior:
    - After each node is processed, the current summary is written to output_path.
    """
    cache_obj = read_json(diagnosis_cache_path)
    cached_nodes = extract_nodes_from_diagnosis_cache_for_trajectory(
        cache_obj=cache_obj,
        trajectory_path=trajectory_path,
    )

    execution_token_counter = {"input": 0, "output": 0}
    execution_llm_generate = build_tracking_llm_generate(execution_token_counter)

    verification_agent = VerificationAgent(
        llm_generate=execution_llm_generate,
        max_retries=2,
        max_log_chars=30000,
    )

    results: List[Dict[str, Any]] = []
    failed_nodes: List[Dict[str, str]] = []

    for item in cached_nodes:
        node = item["node"]
        node_id = node.get("id", "")

        try:
            node_diagnosis = item["node_diagnosis"]

            verification_result = verification_agent.recover_from_diagnosis(
                node=node,
                diagnosis=node_diagnosis,
                logs={},  # recovery-only from cache does not need raw trajectory logs
            )

            results.append(
                {
                    "trajectory_path": item.get("trajectory_path", ""),
                    "node_id": node["id"],
                    "task": node["task"],
                    "agent": node["agent"],
                    "node_contract": node["node_contract"],
                    "step_diagnoses": item.get("step_diagnoses", []),
                    "node_diagnosis": node_diagnosis,
                    "verification_result": verification_result,
                }
            )

        except Exception as exc:
            logger.exception(
                "Failed to process recovery_only for trajectory=%s node_id=%s",
                trajectory_path,
                node_id,
            )
            failed_nodes.append(
                {
                    "trajectory_path": str(item.get("trajectory_path", trajectory_path)),
                    "node_id": str(node_id),
                    "error": repr(exc),
                }
            )

        # each node ごとに現在の全体 summary を保存
        summary = {
            "trajectory_path": str(trajectory_path),
            "diagnosis_cache_path": str(diagnosis_cache_path),
            "output_mode": "recovery_only",
            "num_nodes_analyzed": len(results),
            "execution_token_counter": execution_token_counter,
            "results": results,
        }

        if failed_nodes:
            summary["failed_nodes"] = failed_nodes

        if output_path is not None:
            write_json(summary, output_path)

    return summary


# 434, 1014
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VerificationAgent in diagnosis-only mode, full-verification mode, "
            "or recovery-only mode from either a diagnosis cache or a single trajectory file."
        )
    )

    parser.add_argument(
        "--trajectory_path",
        type=str,
        default="",
        help="Path to a single trajectory JSON file.",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="",
        help="Path to a folder containing trajectory JSON files.",
    )
    parser.add_argument(
        "--diagnosis_cache_path",
        type=str,
        default="",
        help="Path to a diagnosis-only cache JSON file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path to a single output JSON file.",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default="diagnosis_only",
        choices=[
            "diagnosis_only",
            "full_verification",
            "recovery_only",
        ],
        help="Choose the output mode.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.json",
        help="Glob pattern used for trajectory files in the folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to write one recovery_only JSON file per trajectory.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    print("[CLI] main() entered", flush=True)

    args = parse_args()
    print(
        "[CLI] parsed args: "
        f"output_mode={args.output_mode}, "
        f"trajectory_path={args.trajectory_path!r}, "
        f"trajectory_dir={args.trajectory_dir!r}, "
        f"diagnosis_cache_path={args.diagnosis_cache_path!r}, "
        f"output_path={args.output_path!r}, "
        f"output_dir={args.output_dir!r}",
        flush=True,
    )

    output_path = Path(args.output_path) if args.output_path else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[CLI] output file parent ready: {output_path.parent}", flush=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CLI] output directory ready: {output_dir}", flush=True)

    if args.output_mode == "recovery_only":
        if args.trajectory_path:
            if not args.diagnosis_cache_path:
                raise ValueError(
                    "When output_mode=recovery_only and --trajectory_path is given, "
                    "--diagnosis_cache_path is also required."
                )
            if output_path is None:
                raise ValueError(
                    "When output_mode=recovery_only and --trajectory_path is given, "
                    "--output_path is required."
                )

            summary = run_recovery_on_trajectory(
                trajectory_path=Path(args.trajectory_path),
                diagnosis_cache_path=Path(args.diagnosis_cache_path),
                output_path=output_path,
            )

        elif args.diagnosis_cache_path:
            if output_dir is None:
                raise ValueError(
                    "When output_mode=recovery_only and only --diagnosis_cache_path is given, "
                    "--output_dir is required."
                )

            summary = run_recovery_on_diagnosis_cache(
                diagnosis_cache_path=Path(args.diagnosis_cache_path),
                output_dir=output_dir,
            )

        else:
            raise ValueError(
                "When output_mode=recovery_only, provide either "
                "--diagnosis_cache_path alone, or both --trajectory_path and --diagnosis_cache_path."
            )

    else:
        print(f"[CLI] branch={args.output_mode}", flush=True)

        if output_path is None:
            raise ValueError(
                "When output_mode is diagnosis_only or full_verification, "
                "--output_path is required."
            )

        if args.trajectory_dir:
            print("[CLI] using trajectory_dir", flush=True)
            summary = run_verification_on_folder(
                trajectory_dir=Path(args.trajectory_dir),
                output_path=output_path,
                output_mode=args.output_mode,
                file_pattern=args.file_pattern,
            )
        elif args.trajectory_path:
            print("[CLI] using trajectory_path", flush=True)
            summary = run_verification_on_trajectory(
                trajectory_path=Path(args.trajectory_path),
                output_path=output_path,
                output_mode=args.output_mode,
            )
        else:
            raise ValueError(
                "When output_mode is diagnosis_only or full_verification, "
                "either --trajectory_dir or --trajectory_path is required."
            )

    print("[CLI] summary ready", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__": 
    main()