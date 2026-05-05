from reactxen.utils.model_inference import watsonx_llm
import json
import re
from typing import Any, Dict, List, Optional


BRANCHING_HEADER = (
    "The branching decision below is part of conditional plan execution. "
    "Select exactly one next node from the provided branches or output RECOVERY.\n"
)


system_prompt_template = """You are a branching agent in a conditional planning system.
Your job is local routing only.
You are NOT a reviewer, replanner, or plan generator.

You must inspect the current node output and choose exactly one next action.
You must use ONLY the provided inputs:
- answer_contract
- current_node
- node_output
- branches

Routing rule:
1. Compare node_output against the expect field of each provided branch.
2. If one branch is sufficiently supported by the observed output, return that branch's next value.
3. If multiple branches look plausible, choose the single best-supported branch.
4. If no branch is supported, return RECOVERY.

Hard constraints:
- Do not invent new branches.
- Do not invent new node identifiers.
- Your "next" value must be exactly one of the provided branch next values or "RECOVERY".
- Prefer "RECOVERY" when the match is unclear.
- Output JSON only.
- Do not output Markdown, code fences, or extra commentary.

Input:
answer_contract:
{answer_contract}

current_node:
{current_node}

node_output:
{node_output}

branches:
{branches}

Allowed next values:
{allowed_next_values}

Output format:
{{
  "next": "<one allowed next value or RECOVERY>",
  "matched_expect": "<optional matched branch expectation>",
  "reason": "<optional short explanation>"
}}
(END OF RESPONSE)
"""


class BranchingAgent:
    """
    Local routing agent for conditional planning.

    Given the global answer contract, the current node specification,
    the current node output, and the node's branches, this agent selects
    exactly one next node identifier or returns RECOVERY.
    """

    def __init__(
        self,
        branching_prompt: str = system_prompt_template,
        llm=watsonx_llm,
        model_id: int = 6,
        max_retries: int = 3,
    ):
        self.branching_prompt = branching_prompt
        self.llm = llm
        self.model_id = model_id
        self.max_retries = max_retries

    def _to_json_string(self, value: Any) -> str:
        """Serialize values for prompt injection without losing structure."""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)

    def _allowed_next_values(self, branches: List[Dict[str, Any]]) -> List[str]:
        """Return the allowed next node identifiers from the branch list."""
        allowed = []
        for branch in branches or []:
            next_value = branch.get("next")
            if isinstance(next_value, str) and next_value not in allowed:
                allowed.append(next_value)
        return allowed

    def _validate_parsed_result(
        self,
        parsed_result: Dict[str, Any],
        allowed_next_values: List[str],
    ) -> Dict[str, Any]:
        """Validate that parsed JSON satisfies the routing contract."""
        if not isinstance(parsed_result, dict):
            return {
                "status": "Error",
                "reasoning": "Parsed response is not a JSON object.",
                "suggestions": "Return a single JSON object with a valid 'next' field.",
            }

        next_value = parsed_result.get("next")
        if not isinstance(next_value, str) or not next_value.strip():
            return {
                "status": "Error",
                "reasoning": "Missing or empty 'next' field in branching output.",
                "suggestions": "Set 'next' to one of the provided branch next values or RECOVERY.",
            }

        next_value = next_value.strip()
        allowed_set = set(allowed_next_values) | {"RECOVERY"}
        if next_value not in allowed_set:
            return {
                "status": "Error",
                "reasoning": (
                    f"Invalid next value '{next_value}'. "
                    f"Allowed values are {sorted(allowed_set)}."
                ),
                "suggestions": "Do not invent node identifiers. Choose one allowed value or RECOVERY.",
            }

        normalized = {"next": next_value}
        if "matched_expect" in parsed_result and parsed_result["matched_expect"] is not None:
            normalized["matched_expect"] = str(parsed_result["matched_expect"])
        if "reason" in parsed_result and parsed_result["reason"] is not None:
            normalized["reason"] = str(parsed_result["reason"])
        return normalized

    def extract_and_parse_json_using_manual_parser(self, response: str) -> Dict[str, Any]:
        """
        Fallback parser for loosely formatted model responses.
        """
        cleaned = response.strip().replace("\n", " ").replace("\\n", " ")

        next_match = re.search(r'"next"\s*:\s*"([^"]+)"', cleaned)
        matched_expect_match = re.search(r'"matched_expect"\s*:\s*"([^"]*)"', cleaned)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)

        if not next_match:
            return {
                "status": "Error",
                "reasoning": "The extracted JSON block could not be parsed.",
                "suggestions": "Ensure the model outputs valid JSON with a 'next' field.",
            }

        result = {"next": next_match.group(1)}
        if matched_expect_match:
            result["matched_expect"] = matched_expect_match.group(1)
        if reason_match:
            result["reason"] = reason_match.group(1)
        return result

    def extract_and_parse_json(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from the model response.
        """
        try:
            match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            json_block = match.group(0).strip() if match else response.strip()

            if not json_block:
                raise ValueError("Extracted JSON block is empty.")

            return json.loads(json_block)

        except json.JSONDecodeError as ex:
            return {
                "status": "Error",
                "reasoning": f"The extracted JSON block could not be parsed. {ex}",
                "suggestions": "Ensure the model outputs a valid JSON object.",
            }
        except ValueError as ex:
            return {
                "status": "Error",
                "reasoning": str(ex),
                "suggestions": "Check if the extracted JSON block is empty or improperly formatted.",
            }

    def refine_response(
        self,
        answer_contract: Any,
        current_node: Any,
        node_output: Any,
        branches: List[Dict[str, Any]],
        allowed_next_values: List[str],
        error_details: Dict[str, Any],
        it_index: int,
        raw_result: Any,
    ) -> str:
        """
        Ask the model to repair an invalid routing response.
        """
        refinement_prompt = (
            "Your previous routing output violated the JSON or routing constraints. "
            "Regenerate the answer as exactly one JSON object.\n"
            f"Allowed next values: {json.dumps(allowed_next_values + ['RECOVERY'], ensure_ascii=False)}\n"
            f"Error details: {json.dumps(error_details, ensure_ascii=False, indent=2)}\n"
            "Do not invent node identifiers. Prefer RECOVERY if the match is unclear."
        )
        base_prompt = self.branching_prompt.format(
            answer_contract=self._to_json_string(answer_contract),
            current_node=self._to_json_string(current_node),
            node_output=self._to_json_string(node_output),
            branches=self._to_json_string(branches),
            allowed_next_values=self._to_json_string(allowed_next_values + ["RECOVERY"]),
        )
        return (
            f"{base_prompt}\n\n"
            f"Previous Response {it_index}: {self._to_json_string(raw_result)}\n\n"
            f"Feedback {it_index}: {refinement_prompt}"
        )

    def build_prompt(
        self,
        answer_contract: Any,
        current_node: Any,
        node_output: Any,
        branches: List[Dict[str, Any]],
    ) -> str:
        """
        Construct the model prompt for one routing decision.
        """
        allowed_next_values = self._allowed_next_values(branches)
        return self.branching_prompt.format(
            answer_contract=self._to_json_string(answer_contract),
            current_node=self._to_json_string(current_node),
            node_output=self._to_json_string(node_output),
            branches=self._to_json_string(branches),
            allowed_next_values=self._to_json_string(allowed_next_values + ["RECOVERY"]),
        )

    def decide_next(
        self,
        answer_contract: Any,
        current_node: Dict[str, Any],
        node_output: Any,
        branches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Decide the next node for conditional execution.

        Args:
            answer_contract: Global answer requirement for the whole task.
            current_node: Current node specification.
            node_output: Output produced by the current node.
            branches: Branch definitions for the current node.

        Returns:
            dict: JSON-like routing decision, for example:
                  {"next": "N3"}
                  or
                  {"next": "RECOVERY", "reason": "No branch was clearly supported."}
        """
        allowed_next_values = self._allowed_next_values(branches)
        prompt = self.build_prompt(
            answer_contract=answer_contract,
            current_node=current_node,
            node_output=node_output,
            branches=branches,
        )

        for it_index in range(self.max_retries):
            raw_result = self.llm(
                prompt,
                model_id=self.model_id,
                stop=["\n(END OF RESPONSE)"],
            )
            response_text = raw_result.get("generated_text", "")

            parsed_result = self.extract_and_parse_json(response_text)
            validated_result = self._validate_parsed_result(
                parsed_result, allowed_next_values
            )
            if validated_result.get("status") != "Error":
                return validated_result

            parsed_result = self.extract_and_parse_json_using_manual_parser(
                response_text
            )
            validated_result = self._validate_parsed_result(
                parsed_result, allowed_next_values
            )
            if validated_result.get("status") != "Error":
                return validated_result

            prompt = self.refine_response(
                answer_contract=answer_contract,
                current_node=current_node,
                node_output=node_output,
                branches=branches,
                allowed_next_values=allowed_next_values,
                error_details=validated_result,
                it_index=it_index,
                raw_result=raw_result,
            )

        return {
            "next": "RECOVERY",
            "reason": (
                f"Failed to produce a valid routing decision after {self.max_retries} attempts."
            ),
        }