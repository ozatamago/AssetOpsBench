from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import time

JsonDict = Dict[str, Any]


class ReplanningAgentError(Exception):
    """Raised when replanning input/output is invalid."""


@dataclass
class ReplanningAgentConfig:
    max_reason_chars: int = 2000
    allow_answer_contract_revision: bool = False
    require_executor_native_plan: bool = True
    include_json_schema_in_prompt: bool = True
    strict_restart_scope_check: bool = True

    # New flags for the revised workflow semantics
    require_material_change_after_post_recovery_failure: bool = True
    allow_restart_into_avoided_node_only_if_plan_changed: bool = True
    
    max_current_whole_plan_chars: int = 5000
    max_parent_trajectory_chars: int = 4000
    max_review_result_chars: int = 1200
    max_avoidance_guidance_chars: int = 800

class ReplanningAgent:
    """
    Callable replanning handler for the revised ConditionalWorkflow API.

    Expected input:
        user_query: str
        current_whole_plan: {
            "answer_contract": str,
            "tasks": list,
            "node_map": dict,
            "start_node_id": str,
            ...
        }
        parent_trajectory: {
            "node_id": str,
            "task": str,
            "agent_name": str,
            "user_input": str,
            "node_output": str,
            "review_result": dict,
            "normalized_status": str,
            "failure_stage": str,  # e.g. "post_recovery_execution"
            "original_task": str,
            "current_task": str,
            "task_prime": str | None,
            "local_recovery_reason": str | None,
            "recovery_result": dict,
            "avoid_node_ids": list[str],
            "avoid_task_texts": list[str],
            "answer_contract": str,
            "replanning_round_count": int,
            ...
        }
        review_result: {
            "status": str,
            "reason": str
        }

    Expected output:
        {
            "changed_whole_plan": dict,
            "replanning_scope": "node" | "subgraph" | "ancestor",
            "restart_from_node_id": str,
            "reason": str
        }
    """

    ALLOWED_SCOPES = {"node", "subgraph", "ancestor"}

    def __init__(
        self,
        llm_generate: Callable[[str], Any],
        config: Optional[ReplanningAgentConfig] = None,
    ) -> None:
        self.llm_generate = llm_generate
        self.config = config or ReplanningAgentConfig()

    def __call__(
        self,
        *,
        user_query: str,
        current_whole_plan: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        available_agent_catalog: List[JsonDict],
    ) -> JsonDict:
        self._validate_inputs(
            user_query=user_query,
            current_whole_plan=current_whole_plan,
            parent_trajectory=parent_trajectory,
            review_result=review_result,
            available_agent_catalog=available_agent_catalog,
        )

        review_status = self._normalize_review_status(review_result.get("status", ""))
        current_node_id = str(parent_trajectory["node_id"]).strip()
        failure_stage = str(parent_trajectory.get("failure_stage", "")).strip()

        # If the review is already successful and we are not in a known failure-stage
        # path, keep the current plan unchanged.
        if review_status == "PASS" and failure_stage not in {
            "post_recovery_execution",
            "recovery_generation_failure",
        }:
            return {
                "changed_whole_plan": self._executor_plan_to_raw_plan(current_whole_plan),
                "replanning_scope": "node",
                "restart_from_node_id": current_node_id,
                "reason": "Review status already indicates success; plan-level rewriting was unnecessary.",
            }

        prompt = self._build_prompt(
            user_query=user_query,
            current_whole_plan=current_whole_plan,
            parent_trajectory=parent_trajectory,
            review_result=review_result,
            available_agent_catalog=available_agent_catalog,
        )

        llm_ret = self.llm_generate(prompt)
        raw_text, in_tok, out_tok = self._unpack_llm_return(llm_ret)

        parsed = self._parse_model_output(raw_text)

        print(f"parsed['changed_whole_plan']: {parsed['changed_whole_plan']}", flush=True)
        # time.sleep(10)

        self._validate_replanning_output(
            output=parsed,
            current_whole_plan=current_whole_plan,
            parent_trajectory=parent_trajectory,
            available_agent_catalog=available_agent_catalog,
        )


        return {
            "changed_whole_plan": parsed["changed_whole_plan"],
            "replanning_scope": parsed["replanning_scope"],
            "restart_from_node_id": parsed["restart_from_node_id"],
            "reason": self._sanitize_text(parsed["reason"]),
            "in_tok": in_tok,
            "out_tok": out_tok,
        }

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        *,
        user_query: str,
        current_whole_plan: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        available_agent_catalog: List[JsonDict],
    ) -> None:
        if not isinstance(user_query, str) or not user_query.strip():
            raise ReplanningAgentError("user_query must be a non-empty string.")

        if not isinstance(current_whole_plan, dict):
            raise ReplanningAgentError("current_whole_plan must be a dict.")

        if not isinstance(parent_trajectory, dict):
            raise ReplanningAgentError("parent_trajectory must be a dict.")

        if not isinstance(review_result, dict):
            raise ReplanningAgentError("review_result must be a dict.")
        
        if not isinstance(available_agent_catalog, list) or len(available_agent_catalog) == 0:
            raise ReplanningAgentError("available_agent_catalog must be a non-empty list.")

        for i, agent_info in enumerate(available_agent_catalog):
            if not isinstance(agent_info, dict):
                raise ReplanningAgentError(f"available_agent_catalog[{i}] must be a dict.")
            if not isinstance(agent_info.get("name"), str) or not agent_info["name"].strip():
                raise ReplanningAgentError(f"available_agent_catalog[{i}] must contain non-empty 'name'.")

        node_id = parent_trajectory.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ReplanningAgentError("parent_trajectory must contain non-empty 'node_id'.")

        status = review_result.get("status")
        if not isinstance(status, str) or not status.strip():
            raise ReplanningAgentError("review_result must contain non-empty 'status'.")

        if "reason" in review_result and not isinstance(review_result["reason"], str):
            raise ReplanningAgentError("review_result['reason'] must be a string if present.")

        if "avoid_node_ids" in parent_trajectory and not isinstance(parent_trajectory["avoid_node_ids"], list):
            raise ReplanningAgentError("parent_trajectory['avoid_node_ids'] must be a list if present.")

        if "avoid_task_texts" in parent_trajectory and not isinstance(parent_trajectory["avoid_task_texts"], list):
            raise ReplanningAgentError("parent_trajectory['avoid_task_texts'] must be a list if present.")

        if self.config.require_executor_native_plan:
            self._validate_executor_native_plan(current_whole_plan)

    def _validate_executor_native_plan(self, plan: JsonDict) -> None:
        required = ["answer_contract", "tasks", "node_map", "start_node_id"]
        missing = [k for k in required if k not in plan]
        if missing:
            raise ReplanningAgentError(
                f"current_whole_plan must contain executor-native keys {required}. "
                f"Missing: {missing}"
            )

        if not isinstance(plan["answer_contract"], str) or not plan["answer_contract"].strip():
            raise ReplanningAgentError("current_whole_plan['answer_contract'] must be a non-empty string.")

        if not isinstance(plan["tasks"], list) or len(plan["tasks"]) == 0:
            raise ReplanningAgentError("current_whole_plan['tasks'] must be a non-empty list.")

        if not isinstance(plan["node_map"], dict) or len(plan["node_map"]) == 0:
            raise ReplanningAgentError("current_whole_plan['node_map'] must be a non-empty dict.")

        if not isinstance(plan["start_node_id"], str) or not plan["start_node_id"].strip():
            raise ReplanningAgentError("current_whole_plan['start_node_id'] must be a non-empty string.")
        
    def _truncate_text(self, text: str, max_chars: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        if len(text) <= max_chars:
            return text
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head - 32
        return (
            text[:keep_head]
            + "\n... [TRUNCATED FOR PROMPT] ...\n"
            + text[-keep_tail:]
        )

    def _truncate_json_for_prompt(self, value: Any, max_chars: int) -> str:
        """
        Convert possibly non-JSON-serializable objects into a JSON-safe form
        before truncating for prompt use.
        """
        safe_value = self._json_safe(value)
        text = json.dumps(safe_value, ensure_ascii=False, indent=2)
        return self._truncate_text(text, max_chars)
    
    def _compress_parent_trajectory_for_prompt(self, parent_trajectory: JsonDict) -> JsonDict:
        compressed = {
            "node_id": parent_trajectory.get("node_id"),
            "task": parent_trajectory.get("task"),
            "agent_name": parent_trajectory.get("agent_name"),
            "normalized_status": parent_trajectory.get("normalized_status"),
            "decision": parent_trajectory.get("decision"),

            # replanning-critical fields
            "failure_stage": parent_trajectory.get("failure_stage"),
            "original_task": parent_trajectory.get("original_task"),
            "current_task": parent_trajectory.get("current_task"),
            "task_prime": parent_trajectory.get("task_prime"),
            "local_recovery_reason": parent_trajectory.get("local_recovery_reason"),
            "avoid_node_ids": parent_trajectory.get("avoid_node_ids"),
            "avoid_task_texts": parent_trajectory.get("avoid_task_texts"),
            "recovery_result": parent_trajectory.get("recovery_result"),

            # still useful grounding fields
            "review_result": parent_trajectory.get("review_result"),
            "node_output": parent_trajectory.get("node_output"),
            "user_input": parent_trajectory.get("user_input"),
        }

        # aggressively trim large free-text fields
        text_budget = max(200, self.config.max_parent_trajectory_chars // 4)

        for key in [
            "task",
            "original_task",
            "current_task",
            "task_prime",
            "local_recovery_reason",
            "node_output",
            "user_input",
        ]:
            if isinstance(compressed.get(key), str):
                compressed[key] = self._truncate_text(compressed[key], text_budget)

        # review_result / decision / recovery_result can also become large
        if compressed.get("review_result") is not None:
            compressed["review_result"] = self._json_safe(compressed["review_result"])
            compressed["review_result"] = self._truncate_text(
                json.dumps(compressed["review_result"], ensure_ascii=False, indent=2),
                text_budget,
            )

        if compressed.get("decision") is not None:
            compressed["decision"] = self._json_safe(compressed["decision"])
            compressed["decision"] = self._truncate_text(
                json.dumps(compressed["decision"], ensure_ascii=False, indent=2),
                text_budget,
            )

        if compressed.get("recovery_result") is not None:
            compressed["recovery_result"] = self._json_safe(compressed["recovery_result"])
            compressed["recovery_result"] = self._truncate_text(
                json.dumps(compressed["recovery_result"], ensure_ascii=False, indent=2),
                text_budget,
            )

        return compressed
    
    def _compress_plan_for_prompt(self, current_whole_plan: JsonDict) -> JsonDict:
        """
        Compress an executor-native plan into a smaller raw-plan-like summary
        suitable for prompt use.

        Goal:
        - keep only the structural information the LLM needs for replanning
        - avoid sending large executor-internal objects directly
        """
        compressed: JsonDict = {
            "answer_contract": current_whole_plan.get("answer_contract"),
            "start_node_id": current_whole_plan.get("start_node_id"),
            "_replanning_meta": current_whole_plan.get("_replanning_meta", {}),
            "nodes": [],
        }

        node_map = current_whole_plan.get("node_map", {})

        if not isinstance(node_map, dict):
            return compressed

        for node_id, node in node_map.items():
            if hasattr(node, "to_dict"):
                try:
                    d = node.to_dict()
                except Exception:
                    d = {}
            elif isinstance(node, dict):
                d = dict(node)
            else:
                try:
                    d = vars(node)
                except Exception:
                    d = {}

            raw_id = d.get("node_id", d.get("id", str(node_id)))
            raw_task = d.get("task", "")
            raw_agent = d.get("agent_name", d.get("agent", ""))
            raw_deps = d.get("deps", []) or []
            raw_branches = d.get("branches", []) or []

            # Keep only essential branch fields
            compact_branches = []
            for branch in raw_branches:
                if hasattr(branch, "to_dict"):
                    try:
                        b = branch.to_dict()
                    except Exception:
                        b = {}
                elif isinstance(branch, dict):
                    b = dict(branch)
                else:
                    try:
                        b = vars(branch)
                    except Exception:
                        b = {}

                compact_branches.append({
                    "expect": self._truncate_text(str(b.get("expect", "")), 300),
                    "next": str(b.get("next", "")),
                })

            compressed["nodes"].append({
                "id": str(raw_id),
                "task": self._truncate_text(str(raw_task), 600),
                "agent": str(raw_agent),
                "deps": [str(x) for x in raw_deps],
                "branches": compact_branches,
            })

        return compressed
    
    def _executor_plan_to_raw_plan(self, plan: JsonDict) -> JsonDict:
        """
        Convert an executor-native plan into raw conditional-plan format.

        Input (executor-native):
        {
            "answer_contract": ...,
            "tasks": ...,
            "node_map": ...,
            "start_node_id": ...
        }

        Output (raw):
        {
            "answer_contract": ...,
            "nodes": [...],
            "start_node_id": ...
        }
        """
        raw_plan: JsonDict = {
            "answer_contract": plan.get("answer_contract", ""),
            "nodes": [],
        }

        if isinstance(plan.get("start_node_id"), str) and plan["start_node_id"].strip():
            raw_plan["start_node_id"] = plan["start_node_id"]

        node_map = plan.get("node_map", {})
        if not isinstance(node_map, dict):
            return raw_plan

        for node_id, node in node_map.items():
            if hasattr(node, "to_dict"):
                try:
                    d = node.to_dict()
                except Exception:
                    d = {}
            elif isinstance(node, dict):
                d = dict(node)
            else:
                try:
                    d = vars(node)
                except Exception:
                    d = {}

            raw_node = {
                "id": str(d.get("node_id", d.get("id", node_id))),
                "task": str(d.get("task", "")),
                "agent": str(d.get("agent_name", d.get("agent", ""))),
                "deps": [str(x) for x in (d.get("deps", []) or [])],
                "branches": [],
            }

            raw_branches = d.get("branches", []) or []
            for branch in raw_branches:
                if hasattr(branch, "to_dict"):
                    try:
                        b = branch.to_dict()
                    except Exception:
                        b = {}
                elif isinstance(branch, dict):
                    b = dict(branch)
                else:
                    try:
                        b = vars(branch)
                    except Exception:
                        b = {}

                raw_node["branches"].append({
                    "expect": str(b.get("expect", "")),
                    "next": str(b.get("next", "")),
                })

            raw_plan["nodes"].append(raw_node)

        return raw_plan

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        *,
        user_query: str,
        current_whole_plan: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        available_agent_catalog: List[JsonDict],
    ) -> str:
        schema_text = """
        Output JSON schema:
        {
        "changed_whole_plan": {
            "answer_contract": "<string>",
            "nodes": [
            {
                "id": "<node id>",
                "task": "<task text>",
                "agent": "<agent name>",
                "deps": ["<node id>", "..."],
                "branches": [
                {"expect": "<expected output condition>", "next": "<node id | TERMINATE | RECOVERY>"}
                ]
            }
            ],
            "start_node_id": "<optional node id>"
        },
        "replanning_scope": "node" | "subgraph" | "ancestor",
        "restart_from_node_id": "<node id>",
        "reason": "<short explanation of why local recovery was insufficient and why plan-level rewriting was needed>"
        }
        """.strip()

        answer_contract_rule = (
            "Preserve the original answer_contract exactly."
            if not self.config.allow_answer_contract_revision
            else
            "Preserve the original answer_contract unless revision is strictly necessary. "
            "If you revise it, explain why in the reason."
        )

        failure_stage = str(parent_trajectory.get("failure_stage", "")).strip() or "unknown"
        avoid_node_ids, avoid_task_texts = self._collect_avoidance_constraints(
            current_whole_plan=current_whole_plan,
            parent_trajectory=parent_trajectory,
        )

        strong_avoidance_text = (
            "There are no explicit avoid constraints."
            if not avoid_node_ids and not avoid_task_texts
            else (
                "Treat the following as strong negative constraints unless the failure cause is "
                "explicitly removed by a materially rewritten plan:\n"
                f"- avoid_node_ids: {json.dumps(avoid_node_ids, ensure_ascii=False)}\n"
                f"- avoid_task_texts: {json.dumps(avoid_task_texts, ensure_ascii=False)}"
            )
        )

        plan_for_prompt = self._compress_plan_for_prompt(current_whole_plan)
        trajectory_for_prompt = self._compress_parent_trajectory_for_prompt(parent_trajectory)

        plan_text = self._truncate_json_for_prompt(
            plan_for_prompt,
            self.config.max_current_whole_plan_chars,
        )
        trajectory_text = self._truncate_json_for_prompt(
            trajectory_for_prompt,
            self.config.max_parent_trajectory_chars,
        )
        review_text = self._truncate_json_for_prompt(
            review_result,
            self.config.max_review_result_chars,
        )
        avoidance_text = self._truncate_text(
            strong_avoidance_text,
            self.config.max_avoidance_guidance_chars,
        )
        available_agents_text = self._truncate_json_for_prompt(
            available_agent_catalog,
            2000,
        )

        system_rules = f"""
You are a ReplanningAgent for a conditional DAG executor.

Your job is to rewrite part or all of the current plan so that the system can still answer the user query after the parent node failed.

Hard constraints:
1. The changed plan must still aim to answer the original user query.
2. {answer_contract_rule}
3. The replanning must be grounded in the parent trajectory and the review result.
4. Prefer the smallest sufficient change scope:
   node-level < subgraph-level < ancestor-level.
5. Return a changed whole plan in raw conditional-plan format with top-level fields such as answer_contract and nodes. Do not return executor-internal fields such as tasks or node_map.
6. Return an explicit replanning_scope.
7. Return an explicit restart_from_node_id consistent with the chosen scope.
8. Return JSON only. Do not return markdown. Do not return explanatory text outside JSON. Changed_whole_plan must be returned in raw conditional-plan JSON format, not in executor-native internal format.
9. If failure_stage == "post_recovery_execution", local recovery has already been tried once and still failed after executing task_prime.
10. In that case, do not simply reuse the same failed local strategy.
11. Avoid routing execution back through the same failed node unless that node is materially rewritten and the failure cause is explicitly removed.
12. Prefer an alternative node/subgraph/ancestor rewrite that bypasses the failed local strategy while still satisfying the original answer_contract.
13. Use avoid_node_ids and avoid_task_texts as strong negative constraints unless you explicitly remove the failure cause.
14. Minimize plan edits. Keep every unchanged node exactly unchanged, including node_id, task, agent, deps, and branches, unless modification is strictly necessary.
15. Prefer preserving the already-executed prefix of the plan.
16. Set restart_from_node_id to the earliest node whose execution may be invalidated by the plan changes.
17. Do not rename unchanged nodes.
18. Do not rewrite the whole plan if a node-level or small subgraph rewrite is sufficient.
19. Do not invent new agent names.
20. Every node in changed_whole_plan must use an agent that already exists in the available agent catalog.
""".strip()

        prompt = f"""
{system_rules}

Failure stage:
{failure_stage}

User query:
{user_query}

Current whole plan:
{plan_text}

Parent trajectory:
{trajectory_text}

Review result:
{review_text}

Available agent catalog:
{available_agents_text}

Avoidance guidance:
{avoidance_text}

You must decide whether the failure is best handled by node-level, subgraph-level, or ancestor-level replanning.
Do not return a plan diff.
Return the changed whole plan itself.

Additional guidance:
- If failure_stage is "post_recovery_execution", the previous task and its local rewrite were both insufficient.
- If you keep the same failed node, rewrite it materially enough that the original failure cause is explicitly addressed.
- If you choose subgraph or ancestor replanning, ensure restart_from_node_id reflects the smallest node from which the revised logic should be re-executed.

{schema_text}
""".strip()

        return prompt

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_model_output(self, raw_text: str) -> JsonDict:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ReplanningAgentError("Replanning model returned empty text.")

        text = raw_text.strip()

        # Direct JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fenced JSON
        fenced = self._extract_json_block(text)
        if fenced is not None:
            try:
                parsed = json.loads(fenced)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise ReplanningAgentError(
            "Replanning model output was not valid JSON. "
            "Return a single JSON object matching the required schema."
        )

    def _extract_json_block(self, text: str) -> Optional[str]:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    def _validate_raw_replanned_plan(self, plan: JsonDict) -> None:
        if not isinstance(plan, dict):
            raise ReplanningAgentError("changed_whole_plan must be a dict.")

        if "answer_contract" not in plan:
            raise ReplanningAgentError("changed_whole_plan is missing top-level field 'answer_contract'.")
        if "nodes" not in plan:
            raise ReplanningAgentError("changed_whole_plan is missing top-level field 'nodes'.")

        answer_contract = plan["answer_contract"]
        nodes = plan["nodes"]
        raw_start_node_id = plan.get("start_node_id")

        if not isinstance(answer_contract, str) or not answer_contract.strip():
            raise ReplanningAgentError("changed_whole_plan['answer_contract'] must be a non-empty string.")

        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ReplanningAgentError("changed_whole_plan['nodes'] must be a non-empty list.")

        node_ids = set()
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise ReplanningAgentError(f"Node at index {i} must be a dict.")

            node_id = node.get("id")
            task = node.get("task")
            agent = node.get("agent")
            deps = node.get("deps", [])
            branches = node.get("branches", [])

            if not isinstance(node_id, str) or not node_id.strip():
                raise ReplanningAgentError(f"Node at index {i} has invalid 'id'.")
            if node_id in node_ids:
                raise ReplanningAgentError(f"Duplicate node id detected: '{node_id}'")
            node_ids.add(node_id)

            if not isinstance(task, str) or not task.strip():
                raise ReplanningAgentError(f"Node '{node_id}' has invalid 'task'.")
            if not isinstance(agent, str) or not agent.strip():
                raise ReplanningAgentError(f"Node '{node_id}' has invalid 'agent'.")
            if not isinstance(deps, list):
                raise ReplanningAgentError(f"Node '{node_id}' has invalid 'deps'; it must be a list.")
            if not isinstance(branches, list):
                raise ReplanningAgentError(f"Node '{node_id}' has invalid 'branches'; it must be a list.")

        # second pass for deps / next validation
        allowed_special_next = {"TERMINATE", "RECOVERY"}
        for node in nodes:
            node_id = node["id"]
            for dep in node.get("deps", []) or []:
                if dep not in node_ids:
                    raise ReplanningAgentError(
                        f"Node '{node_id}' depends on unknown node '{dep}'."
                    )

            for j, branch in enumerate(node.get("branches", []) or []):
                if not isinstance(branch, dict):
                    raise ReplanningAgentError(
                        f"Node '{node_id}' has non-dict branch at index {j}."
                    )
                expect = branch.get("expect")
                nxt = branch.get("next")
                if not isinstance(expect, str) or not expect.strip():
                    raise ReplanningAgentError(
                        f"Node '{node_id}' has invalid branch.expect at index {j}."
                    )
                if not isinstance(nxt, str) or not nxt.strip():
                    raise ReplanningAgentError(
                        f"Node '{node_id}' has invalid branch.next at index {j}."
                    )
                if nxt not in node_ids and nxt not in allowed_special_next:
                    raise ReplanningAgentError(
                        f"Node '{node_id}' has branch to unknown next node '{nxt}'."
                    )

        if raw_start_node_id is not None:
            if not isinstance(raw_start_node_id, str) or not raw_start_node_id.strip():
                raise ReplanningAgentError("changed_whole_plan['start_node_id'] must be a non-empty string if provided.")
            if raw_start_node_id not in node_ids:
                raise ReplanningAgentError(
                    f"changed_whole_plan['start_node_id'] ('{raw_start_node_id}') is not one of the defined node ids."
                )

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------
    def _validate_replanning_output(
        self,
        *,
        output: JsonDict,
        current_whole_plan: JsonDict,
        parent_trajectory: JsonDict,
        available_agent_catalog: List[JsonDict],
    ) -> None:
        if not isinstance(output, dict):
            raise ReplanningAgentError("Replanning output must be a dict.")

        required = ["changed_whole_plan", "replanning_scope", "restart_from_node_id", "reason"]
        missing = [k for k in required if k not in output]
        if missing:
            raise ReplanningAgentError(f"Replanning output missing required keys: {missing}")

        changed_whole_plan = output["changed_whole_plan"]
        replanning_scope = output["replanning_scope"]
        restart_from_node_id = output["restart_from_node_id"]
        reason = output["reason"]

        if not isinstance(changed_whole_plan, dict):
            raise ReplanningAgentError("'changed_whole_plan' must be a dict.")

        if not isinstance(replanning_scope, str) or replanning_scope not in self.ALLOWED_SCOPES:
            raise ReplanningAgentError(
                f"'replanning_scope' must be one of {sorted(self.ALLOWED_SCOPES)}."
            )

        if not isinstance(restart_from_node_id, str) or not restart_from_node_id.strip():
            raise ReplanningAgentError("'restart_from_node_id' must be a non-empty string.")

        if not isinstance(reason, str) or not reason.strip():
            raise ReplanningAgentError("'reason' must be a non-empty string.")

        if len(reason) > self.config.max_reason_chars:
            raise ReplanningAgentError(
                f"'reason' is too long ({len(reason)} chars > {self.config.max_reason_chars})."
            )

        self._validate_raw_replanned_plan(changed_whole_plan)
        allowed_agent_names = {
            str(agent_info.get("name", "")).strip()
            for agent_info in available_agent_catalog
            if isinstance(agent_info, dict)
        }

        raw_nodes = changed_whole_plan.get("nodes", [])
        for i, node in enumerate(raw_nodes):
            if not isinstance(node, dict):
                continue
            agent_name = str(node.get("agent", "")).strip()
            if agent_name not in allowed_agent_names:
                raise ReplanningAgentError(
                    f"Node '{node.get('id', f'index {i}')}' uses unknown agent '{agent_name}'. "
                    f"Allowed agents are: {sorted(allowed_agent_names)}"
                )

        old_contract = current_whole_plan.get("answer_contract")
        new_contract = changed_whole_plan.get("answer_contract")

        raw_nodes = changed_whole_plan.get("nodes", [])
        raw_node_ids = {
            str(node.get("id", "")).strip()
            for node in raw_nodes
            if isinstance(node, dict)
        }

        if restart_from_node_id not in raw_node_ids:
            raise ReplanningAgentError(
                f"restart_from_node_id '{restart_from_node_id}' not found in changed_whole_plan['nodes']."
            )

        if self.config.strict_restart_scope_check:
            self._check_scope_restart_consistency(
                replanning_scope=replanning_scope,
                restart_from_node_id=restart_from_node_id,
                parent_trajectory=parent_trajectory,
                current_whole_plan=current_whole_plan,
                changed_whole_plan=changed_whole_plan,
            )

        failure_stage = str(parent_trajectory.get("failure_stage", "")).strip()
        plans_equal = self._plans_equivalent(current_whole_plan, changed_whole_plan)

        if plans_equal:
            if (
                self.config.require_material_change_after_post_recovery_failure
                and failure_stage == "post_recovery_execution"
            ):
                raise ReplanningAgentError(
                    "For post_recovery_execution failure, replanning must make a material plan change; "
                    "returning an unchanged plan is not allowed."
                )

            if replanning_scope != "node" or restart_from_node_id != parent_trajectory["node_id"]:
                raise ReplanningAgentError(
                    "Plan is unchanged, but replanning_scope/restart_from_node_id suggest broader rewriting."
                )

        avoid_node_ids, _ = self._collect_avoidance_constraints(
            current_whole_plan=current_whole_plan,
            parent_trajectory=parent_trajectory,
        )

        if (
            self.config.allow_restart_into_avoided_node_only_if_plan_changed
            and failure_stage == "post_recovery_execution"
            and restart_from_node_id in avoid_node_ids
            and plans_equal
        ):
            raise ReplanningAgentError(
                "restart_from_node_id points to an avoided failed node, but the changed plan is unchanged."
            )

    def _check_scope_restart_consistency(
        self,
        *,
        replanning_scope: str,
        restart_from_node_id: str,
        parent_trajectory: JsonDict,
        current_whole_plan: JsonDict,
        changed_whole_plan: JsonDict,
    ) -> None:
        current_node_id = str(parent_trajectory["node_id"]).strip()

        if replanning_scope == "node":
            if restart_from_node_id != current_node_id:
                raise ReplanningAgentError(
                    "For node-level replanning, restart_from_node_id must equal the current parent node id."
                )
            return

        if replanning_scope == "subgraph":
            if (
                restart_from_node_id == changed_whole_plan.get("start_node_id")
                and restart_from_node_id != current_node_id
            ):
                raise ReplanningAgentError(
                    "For subgraph-level replanning, restart_from_node_id should not jump to the global start node "
                    "unless the current node itself is the global start."
                )
            return

        if replanning_scope == "ancestor":
            return

        raise ReplanningAgentError(f"Unknown replanning_scope: {replanning_scope}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_avoidance_constraints(
        self,
        *,
        current_whole_plan: JsonDict,
        parent_trajectory: JsonDict,
    ) -> tuple[List[str], List[str]]:
        node_ids: List[str] = []
        task_texts: List[str] = []

        if isinstance(parent_trajectory.get("avoid_node_ids"), list):
            node_ids.extend([str(x) for x in parent_trajectory["avoid_node_ids"] if str(x).strip()])

        if isinstance(parent_trajectory.get("avoid_task_texts"), list):
            task_texts.extend([str(x) for x in parent_trajectory["avoid_task_texts"] if str(x).strip()])

        meta = current_whole_plan.get("_replanning_meta", {})
        if isinstance(meta, dict):
            if isinstance(meta.get("avoid_node_ids"), list):
                node_ids.extend([str(x) for x in meta["avoid_node_ids"] if str(x).strip()])
            if isinstance(meta.get("avoid_task_texts"), list):
                task_texts.extend([str(x) for x in meta["avoid_task_texts"] if str(x).strip()])

        node_ids = sorted(set(node_ids))
        task_texts = list(dict.fromkeys(task_texts))
        return node_ids, task_texts

    def _unpack_llm_return(self, llm_ret: Any) -> tuple[str, Optional[int], Optional[int]]:
        if isinstance(llm_ret, tuple):
            if len(llm_ret) == 3:
                return llm_ret[0], llm_ret[1], llm_ret[2]
            if len(llm_ret) == 1:
                return llm_ret[0], None, None
        if isinstance(llm_ret, str):
            return llm_ret, None, None
        raise ReplanningAgentError(
            "llm_generate must return either a string or a tuple "
            "(raw_text, in_tok, out_tok)."
        )

    def _plans_equivalent(self, p1: JsonDict, p2: JsonDict) -> bool:
        try:
            raw1 = self._normalize_plan_for_comparison(p1)
            raw2 = self._normalize_plan_for_comparison(p2)
            s1 = json.dumps(self._json_safe(raw1), ensure_ascii=False, sort_keys=True)
            s2 = json.dumps(self._json_safe(raw2), ensure_ascii=False, sort_keys=True)
            return s1 == s2
        except Exception:
            return False
        
    def _normalize_plan_for_comparison(self, plan: JsonDict) -> JsonDict:
        """
        Normalize either
        - executor-native plan, or
        - raw conditional-plan
        into raw conditional-plan format for semantic comparison.
        """
        if not isinstance(plan, dict):
            return {"answer_contract": "", "nodes": []}

        # Case 1: already raw conditional-plan
        if "answer_contract" in plan and "nodes" in plan:
            normalized = {
                "answer_contract": str(plan.get("answer_contract", "")).strip(),
                "nodes": [],
            }
            if isinstance(plan.get("start_node_id"), str) and plan["start_node_id"].strip():
                normalized["start_node_id"] = plan["start_node_id"].strip()

            raw_nodes = plan.get("nodes", [])
            if isinstance(raw_nodes, list):
                for node in raw_nodes:
                    if not isinstance(node, dict):
                        continue
                    normalized_node = {
                        "id": str(node.get("id", "")).strip(),
                        "task": str(node.get("task", "")).strip(),
                        "agent": str(node.get("agent", "")).strip(),
                        "deps": [str(x) for x in (node.get("deps", []) or [])],
                        "branches": [],
                    }

                    raw_branches = node.get("branches", []) or []
                    for branch in raw_branches:
                        if not isinstance(branch, dict):
                            continue
                        normalized_node["branches"].append({
                            "expect": str(branch.get("expect", "")).strip(),
                            "next": str(branch.get("next", "")).strip(),
                        })

                    normalized["nodes"].append(normalized_node)

            # sort nodes for order-insensitive comparison
            normalized["nodes"] = sorted(normalized["nodes"], key=lambda x: x["id"])
            for node in normalized["nodes"]:
                node["deps"] = sorted(node["deps"])
                node["branches"] = sorted(
                    node["branches"],
                    key=lambda b: (b["expect"], b["next"])
                )

            return normalized

        # Case 2: executor-native plan
        if {"answer_contract", "tasks", "node_map", "start_node_id"}.issubset(plan.keys()):
            normalized = self._executor_plan_to_raw_plan(plan)

            # canonicalize ordering
            raw_nodes = normalized.get("nodes", [])
            canonical_nodes = []
            for node in raw_nodes:
                if not isinstance(node, dict):
                    continue
                canonical_node = {
                    "id": str(node.get("id", "")).strip(),
                    "task": str(node.get("task", "")).strip(),
                    "agent": str(node.get("agent", "")).strip(),
                    "deps": sorted([str(x) for x in (node.get("deps", []) or [])]),
                    "branches": [],
                }

                raw_branches = node.get("branches", []) or []
                for branch in raw_branches:
                    if not isinstance(branch, dict):
                        continue
                    canonical_node["branches"].append({
                        "expect": str(branch.get("expect", "")).strip(),
                        "next": str(branch.get("next", "")).strip(),
                    })

                canonical_node["branches"] = sorted(
                    canonical_node["branches"],
                    key=lambda b: (b["expect"], b["next"])
                )
                canonical_nodes.append(canonical_node)

            normalized["answer_contract"] = str(normalized.get("answer_contract", "")).strip()
            if isinstance(normalized.get("start_node_id"), str):
                normalized["start_node_id"] = normalized["start_node_id"].strip()
            normalized["nodes"] = sorted(canonical_nodes, key=lambda x: x["id"])

            return normalized

        # Unknown shape: return json-safe fallback
        return self._json_safe(plan)

    def _json_safe(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if hasattr(obj, "to_dict"):
            try:
                return self._json_safe(obj.to_dict())
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return self._json_safe(vars(obj))
            except Exception:
                pass
        return str(obj)

    def _sanitize_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _normalize_review_status(self, status: Any) -> str:
        if not isinstance(status, str):
            return "UNKNOWN"
        s = status.strip().lower()
        if s in {"pass", "passed", "accomplished", "success", "succeeded"}:
            return "PASS"
        if s in {"fail", "failed", "error", "not accomplished", "partially accomplished"}:
            return "FAIL"
        return "UNKNOWN"