from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


JsonDict = Dict[str, Any]


class RecoveryAgentError(Exception):
    """Raised when recovery input/output is invalid."""


@dataclass
class RecoveryAgentConfig:
    max_output_chars: int = 4000
    enforce_same_objective: bool = True
    include_json_schema_in_prompt: bool = True
    strict_capability_guard: bool = False
    max_parent_trajectory_chars: int = 3000
    max_node_chars: int = 1500
    max_review_result_chars: int = 1000
    max_agent_description_chars: int = 1000


class RecoveryAgent:
    """
    Callable recovery handler for ConditionalWorkflow.

    Expected input:
        node: {
            "id" or "node_id": str,
            "task": str,
            "agent" or "agent_name": str,
            "deps": list,
            "branches": list,
            ...
        }

        parent_trajectory: {
            "node_id": str,
            "input_task": str,          # preferred
            "trajectory": list,         # preferred
            "final_output": str,        # preferred
            ...
        }

        review_result: {
            "status": "Accomplished" | "Partially Accomplished" | "Not Accomplished",
            "reason": str
        }

        agent_description: {
            "name": str,
            "capabilities": list[str],  # preferred
            "constraints": list[str]    # preferred
        }

    Returns:
        {
            "task_prime": str,
            "reason": str
        }

    This class deliberately does NOT return:
        - next
        - plan
        - changed_whole_plan
        - restart_from_node_id
        - replanning_scope
    because recovery is node-local only.
    """

    FORBIDDEN_OUTPUT_KEYS = {
        "next",
        "plan",
        "changed_whole_plan",
        "restart_from_node_id",
        "replanning_scope",
    }

    def __init__(
        self,
        llm_generate: Callable[[str], str],
        config: Optional[RecoveryAgentConfig] = None,
    ) -> None:
        """
        Args:
            llm_generate:
                A function that receives a prompt string and returns model text.
                Example:
                    def llm_generate(prompt: str) -> str: ...
            config:
                Optional configuration.
        """
        self.llm_generate = llm_generate
        self.config = config or RecoveryAgentConfig()

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
        text = json.dumps(value, ensure_ascii=False, indent=2)
        return self._truncate_text(text, max_chars)
    
    def _compress_parent_trajectory_for_prompt(self, parent_trajectory: JsonDict) -> JsonDict:
        compressed = {
            "node_id": parent_trajectory.get("node_id"),
            "task": parent_trajectory.get("task"),
            "agent_name": parent_trajectory.get("agent_name"),
            "normalized_status": parent_trajectory.get("normalized_status"),
            "decision": parent_trajectory.get("decision"),
            "review_result": parent_trajectory.get("review_result"),
            "node_output": parent_trajectory.get("node_output"),
            "user_input": parent_trajectory.get("user_input"),
        }

        # aggressively trim large free-text fields
        if isinstance(compressed.get("node_output"), str):
            compressed["node_output"] = self._truncate_text(
                compressed["node_output"],
                self.config.max_parent_trajectory_chars // 3,
            )
        if isinstance(compressed.get("user_input"), str):
            compressed["user_input"] = self._truncate_text(
                compressed["user_input"],
                self.config.max_parent_trajectory_chars // 3,
            )

        return compressed
    
    def __call__(
        self,
        *,
        node: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        agent_description: JsonDict,
    ) -> JsonDict:
        self._validate_inputs(
            node=node,
            parent_trajectory=parent_trajectory,
            review_result=review_result,
            agent_description=agent_description,
        )

        original_task = self._extract_original_task(node, parent_trajectory)
        review_status = str(review_result.get("status", "")).strip()

        # If review already says accomplished, recovery should usually not rewrite.
        # We still return a valid task_prime, but preserve the original task.
        if review_status == "Accomplished":
            return {
                "task_prime": original_task,
                "reason": "Review status is 'Accomplished'; no node-local correction was necessary.",
            }

        prompt = self._build_prompt(
            node=node,
            parent_trajectory=parent_trajectory,
            review_result=review_result,
            agent_description=agent_description,
            original_task=original_task,
        )

        raw_text, in_tok, out_tok = self.llm_generate(prompt)
        parsed = self._parse_model_output(raw_text)

        task_prime = parsed.get("task_prime")
        reason = parsed.get("reason")

        if not isinstance(task_prime, str) or not task_prime.strip():
            raise RecoveryAgentError("Recovery model output did not contain a valid non-empty 'task_prime'.")

        if not isinstance(reason, str) or not reason.strip():
            raise RecoveryAgentError("Recovery model output did not contain a valid non-empty 'reason'.")

        task_prime = self._sanitize_text(task_prime)
        reason = self._sanitize_text(reason)

        # self._validate_recovery_output(
        #     output=parsed,
        #     original_task=original_task,
        #     node=node,
        #     parent_trajectory=parent_trajectory,
        #     review_result=review_result,
        #     agent_description=agent_description,
        # )

        return {
            "task_prime": task_prime,
            "reason": reason,
            "in_tok": in_tok,
            "out_tok": out_tok,
        }


    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        *,
        node: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        agent_description: JsonDict,
    ) -> None:
        if not isinstance(node, dict):
            raise RecoveryAgentError("node must be a dict.")
        if not isinstance(parent_trajectory, dict):
            raise RecoveryAgentError("parent_trajectory must be a dict.")
        if not isinstance(review_result, dict):
            raise RecoveryAgentError("review_result must be a dict.")
        if not isinstance(agent_description, dict):
            raise RecoveryAgentError("agent_description must be a dict.")

        node_id = node.get("id", node.get("node_id"))
        if not isinstance(node_id, str) or not node_id.strip():
            raise RecoveryAgentError("node must contain a non-empty 'id' or 'node_id'.")

        node_task = node.get("task")
        if not isinstance(node_task, str) or not node_task.strip():
            raise RecoveryAgentError("node must contain a non-empty 'task'.")

        status = review_result.get("status")
        if not isinstance(status, str) or not status.strip():
            raise RecoveryAgentError("review_result must contain a non-empty 'status'.")

        if "reason" in review_result and not isinstance(review_result["reason"], str):
            raise RecoveryAgentError("review_result['reason'] must be a string if present.")

        agent_name = agent_description.get("name")
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise RecoveryAgentError("agent_description must contain a non-empty 'name'.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        *,
        node: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        agent_description: JsonDict,
        original_task: str,
    ) -> str:
        schema_text = ""
        if self.config.include_json_schema_in_prompt:
            schema_text = """
        Output JSON schema:
        {
        "task_prime": "<a revised task for the SAME node and SAME agent. It must explicitly carry forward the relevant error/failure context from the previous execution, state or imply what expected output was not achieved, and provide the concrete next-step guidance, missing information, or corrective instruction needed for the next execution to better achieve the expected output.>",
        "reason": "<a short explanation of which part of the expected output was not achieved previously, what caused the failure or gap, and why the revised task_prime is a better next instruction>"
        }
        """.strip()

        system_rules = """
        You are a RecoveryAgent for a conditional DAG executor.

        Your job is ONLY node-local recovery.
        You must revise the task for the SAME node and the SAME agent.

        Hard constraints:
        1. Preserve the original node objective. Do not change what this node is fundamentally trying to accomplish.
        2. Use the given node, parent trajectory, node output, review result, and any failure or error information from the previous attempt.
        3. Determine what part of the expected output was not achieved in the previous execution.
        4. Determine what was missing, incorrect, blocked, ambiguous, unverified, or based on an invalid assumption.
        5. Rewrite the task into task_prime so that the next execution directly addresses those specific gaps or failures.
        6. task_prime must explicitly incorporate the relevant failure/error context and the concrete guidance needed for the next execution to better achieve the expected output.
        7. task_prime must be an immediately executable next instruction for the same agent in the same node.
        8. Stay within the assigned agent's capabilities and constraints.
        9. Do NOT change the plan structure.
        10. Do NOT output any next node, branch decision, new plan, restart point, or replanning scope.
        11. Do NOT merely restate or lightly paraphrase the original task. The revision must be targeted and responsive to the specific failure of the previous execution.
        12. If the previous execution failed due to missing information, ambiguity, incomplete verification, tool error, invalid assumption, or unmet intermediate conditions, reflect that explicitly in task_prime and instruct the agent how to address it in the next execution if possible within the same node.
        13. task_prime should help the next execution achieve the expected output, not merely summarize what went wrong.
        14. Return JSON only.
        """.strip()

        node_text = self._truncate_json_for_prompt(
            node, self.config.max_node_chars
        )
        parent_trajectory_for_prompt = self._compress_parent_trajectory_for_prompt(parent_trajectory)
        parent_trajectory_text = self._truncate_json_for_prompt(
            parent_trajectory_for_prompt,
            self.config.max_parent_trajectory_chars,
        )
        review_result_text = self._truncate_json_for_prompt(
            review_result, self.config.max_review_result_chars
        )
        agent_description_text = self._truncate_json_for_prompt(
            agent_description, self.config.max_agent_description_chars
        )

        prompt = f"""
        {system_rules}

        Node:
        {node_text}

        Parent trajectory:
        {parent_trajectory_text}

        Review result:
        {review_result_text}

        Agent description:
        {agent_description_text}

        Original task:
        {original_task}

        Write a corrected task for the same node.
        The revised task should preserve the goal, but make the execution instruction more precise, safer, and more directly responsive to the failure indicated by the trajectory and review.

        {schema_text}
        """.strip()

        return prompt

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_model_output(self, raw_text: str) -> JsonDict:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise RecoveryAgentError("Recovery model returned empty text.")

        text = raw_text.strip()

        # First try direct JSON parse.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Then try fenced JSON extraction.
        fenced = self._extract_json_block(text)
        if fenced is not None:
            try:
                parsed = json.loads(fenced)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Last-resort heuristic extraction.
        return self._heuristic_parse(text)

    def _extract_json_block(self, text: str) -> Optional[str]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _heuristic_parse(self, text: str) -> JsonDict:
        """
        Fallback parser for outputs like:
            task_prime: ...
            reason: ...
        """
        task_prime = None
        reason = None

        task_match = re.search(
            r'(?:^|\n)\s*"?(task_prime)"?\s*:\s*"?(.*?)"?(?:\n\s*"?(reason)"?\s*:|$)',
            text,
            flags=re.DOTALL,
        )
        reason_match = re.search(
            r'(?:^|\n)\s*"?(reason)"?\s*:\s*"?(.*?)"?\s*$',
            text,
            flags=re.DOTALL,
        )

        if task_match:
            task_prime = task_match.group(2).strip().strip(",").strip()
        if reason_match:
            reason = reason_match.group(2).strip().strip(",").strip()

        if task_prime is None:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                task_prime = lines[0]

        if reason is None:
            reason = "Parsed from non-JSON output."

        return {
            "task_prime": task_prime,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------
    def _validate_recovery_output(
        self,
        *,
        output: JsonDict,
        original_task: str,
        node: JsonDict,
        parent_trajectory: JsonDict,
        review_result: JsonDict,
        agent_description: JsonDict,
    ) -> None:
        if not isinstance(output, dict):
            raise RecoveryAgentError("Recovery output must be a dict.")

        illegal = self.FORBIDDEN_OUTPUT_KEYS.intersection(output.keys())
        if illegal:
            raise RecoveryAgentError(
                f"Recovery output must not change plan structure. Forbidden keys found: {sorted(illegal)}"
            )

        task_prime = output.get("task_prime")
        reason = output.get("reason")

        if not isinstance(task_prime, str) or not task_prime.strip():
            raise RecoveryAgentError("Recovery output must contain non-empty 'task_prime'.")
        if not isinstance(reason, str) or not reason.strip():
            raise RecoveryAgentError("Recovery output must contain non-empty 'reason'.")

        if len(task_prime) > self.config.max_output_chars:
            raise RecoveryAgentError(
                f"task_prime is too long ({len(task_prime)} chars > {self.config.max_output_chars})."
            )

        if self.config.enforce_same_objective:
            self._check_same_objective(original_task=original_task, task_prime=task_prime)

        if self.config.strict_capability_guard:
            self._check_capability_guard(
                task_prime=task_prime,
                agent_description=agent_description,
            )

        # Minimal grounding check: the review reason should not be completely ignored.
        review_reason = str(review_result.get("reason", "")).strip().lower()
        if review_reason:
            overlap = self._soft_overlap(review_reason, reason.lower() + " " + task_prime.lower())
            if overlap == 0:
                # Do not hard-fail here because LLM wording can vary.
                # But at least signal poor grounding.
                raise RecoveryAgentError(
                    "Recovery output appears insufficiently grounded in the review_result.reason."
                )

    def _check_same_objective(self, *, original_task: str, task_prime: str) -> None:
        """
        Conservative guard:
        - task_prime may add clarification and error-avoidance instructions
        - but it should not discard the original task objective entirely
        """
        original_keywords = self._content_words(original_task)
        revised_keywords = self._content_words(task_prime)

        if not original_keywords:
            return

        overlap_count = len(original_keywords.intersection(revised_keywords))
        if overlap_count == 0:
            raise RecoveryAgentError(
                "task_prime does not appear to preserve the original node objective."
            )

    def _check_capability_guard(self, *, task_prime: str, agent_description: JsonDict) -> None:
        """
        Optional conservative guard.
        Only checks explicit textual mismatch signals.
        """
        constraints = agent_description.get("constraints", [])
        if not isinstance(constraints, list):
            return

        lowered_task = task_prime.lower()
        for item in constraints:
            if not isinstance(item, str):
                continue
            item_l = item.lower()
            if item_l.startswith("cannot ") and item_l[7:] in lowered_task:
                raise RecoveryAgentError(
                    f"task_prime may violate an explicit agent constraint: {item}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_original_task(self, node: JsonDict, parent_trajectory: JsonDict) -> str:
        input_task = parent_trajectory.get("input_task")
        if isinstance(input_task, str) and input_task.strip():
            return input_task.strip()

        node_task = node.get("task")
        if isinstance(node_task, str) and node_task.strip():
            return node_task.strip()

        raise RecoveryAgentError("Could not recover original task from node or parent_trajectory.")

    def _sanitize_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _content_words(self, text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z0-9_/-]+", text.lower())
        stop = {
            "the", "a", "an", "to", "of", "for", "and", "or", "if", "then",
            "use", "check", "make", "be", "is", "are", "with", "on", "in",
            "same", "node", "task", "agent", "output", "result",
        }
        return {t for t in tokens if t not in stop and len(t) >= 3}

    def _soft_overlap(self, source: str, target: str) -> int:
        s = self._content_words(source)
        t = self._content_words(target)
        return len(s.intersection(t))


# # ----------------------------------------------------------------------
# # Example LLM adapter
# # ----------------------------------------------------------------------
# def dummy_llm_generate(prompt: str) -> str:
#     """
#     Example stub.
#     Replace this with your actual LLM call.

#     Expected return format: JSON string
#     """
#     return json.dumps(
#         {
#             "task_prime": (
#                 "Re-execute the same node objective, but correct the previous mistake. "
#                 "Follow the original goal exactly, use the appropriate interface/tool, "
#                 "and verify the relevant scope or parameter before terminating."
#             ),
#             "reason": (
#                 "The previous attempt appears not to satisfy the node objective according "
#                 "to the review result, so the task was rewritten to make the execution "
#                 "instruction more explicit while preserving the same goal."
#             ),
#         },
#         ensure_ascii=False,
#     )


# # ----------------------------------------------------------------------
# # Example usage
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     recovery_handler = RecoveryAgent(llm_generate=dummy_llm_generate)

#     node = {
#         "id": "S1",
#         "task": "List the IoT sites",
#         "agent": "IoT Data Download",
#         "deps": [],
#         "branches": [
#             {"expect": "A list of IoT sites is provided", "next": "S2"},
#             {"expect": "Failure to retrieve the list", "next": "TERMINATE"},
#         ],
#     }

#     parent_trajectory = {
#         "node_id": "S1",
#         "input_task": "List the IoT sites",
#         "trajectory": [
#             {"kind": "message", "content": "Start retrieval."},
#             {"kind": "tool_call", "content": "device-status API"},
#             {"kind": "observation", "content": "Returned device status, not site list."},
#         ],
#         "final_output": "No site list was returned.",
#     }

#     review_result = {
#         "status": "Not Accomplished",
#         "reason": "The execution used the wrong API and did not return the requested site list.",
#     }

#     agent_description = {
#         "name": "IoT Data Download",
#         "capabilities": ["Call IoT listing APIs", "Validate returned site scope"],
#         "constraints": ["Do not fabricate unavailable site names"],
#     }

#     result = recovery_handler(
#         node=node,
#         parent_trajectory=parent_trajectory,
#         review_result=review_result,
#         agent_description=agent_description,
#     )

#     print(json.dumps(result, ensure_ascii=False, indent=2))