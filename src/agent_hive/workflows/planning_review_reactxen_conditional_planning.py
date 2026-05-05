from agent_hive.task import Task, ConditionalTask
from pydantic import Field
from typing import List
from agent_hive.enum import ContextType
import json
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
from agent_hive.workflows.sequential import ConditionalWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.workflows.branching_agent import BranchingAgent
from agent_hive.workflows.recovery_agent import RecoveryAgent
from agent_hive.workflows.replanning_agent import ReplanningAgent
from agent_hive.workflows.planning import PlanningWorkflow
from agent_hive.workflows.verification_agent import VerificationAgent

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

# =========================================================
# TODO: Participants can edit this section ONLY
# Add variable, dict. no more any import just any inline code
# =========================================================
# END OF EDITABLE SECTION

from typing import Any, Dict, List, Optional, Set


def extract_conditional_plan_json_text(text: str) -> str:
    """
    Extract exactly one JSON object for the conditional plan.

    Priority:
    1. JSON fenced blocks
    2. Balanced-brace JSON object scan
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Plan text is empty.")

    # ---------------------------------------------------------
    # 1) Try fenced ```json ... ```
    # ---------------------------------------------------------
    fenced_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for block in reversed(fenced_blocks):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "answer_contract" in obj and "nodes" in obj:
                return block
        except Exception:
            pass

    # ---------------------------------------------------------
    # 2) Scan balanced JSON objects from raw text
    # ---------------------------------------------------------
    candidates = []
    depth = 0
    start = None

    for i, ch in enumerate(text):
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

    # Prefer the last valid plan-like object
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "answer_contract" in obj and "nodes" in obj:
                return candidate
        except Exception:
            pass

    raise ValueError("Could not extract a valid conditional plan JSON object.")


def _normalize_branch_dict(branch: Dict[str, Any], *, node_id: str, branch_index: int) -> Dict[str, str]:
    """
    Normalize one branch entry.

    Accepted input formats:
      1) {"expect": "...", "next": "S2"}          # old format
      2) {"label": "execution", "next": "B1_S1"}  # new format

    Returned canonical format:
      {"expect": "<string>", "next": "<string>"}

    We intentionally normalize `label` -> `expect` so that the existing
    ConditionalTask / Branching logic can keep working with minimal changes.
    """
    if not isinstance(branch, dict):
        raise ValueError(
            f"Node '{node_id}' branch at index {branch_index} must be a JSON object."
        )

    if "next" not in branch:
        raise ValueError(
            f"Node '{node_id}' branch at index {branch_index} is missing 'next'."
        )

    next_value = branch["next"]
    if not isinstance(next_value, str) or not next_value.strip():
        raise ValueError(
            f"Node '{node_id}' branch at index {branch_index} has invalid 'next'."
        )
    next_value = next_value.strip()

    expect_raw = branch.get("expect")
    label_raw = branch.get("label")

    expect_str = expect_raw.strip() if isinstance(expect_raw, str) else None
    label_str = label_raw.strip() if isinstance(label_raw, str) else None

    if expect_str and label_str:
        if expect_str != label_str:
            raise ValueError(
                f"Node '{node_id}' branch at index {branch_index} has both "
                f"'expect' and 'label', but they differ: '{expect_str}' vs '{label_str}'."
            )
        canonical_expect = expect_str
    elif label_str:
        canonical_expect = label_str
    elif expect_str:
        canonical_expect = expect_str
    else:
        raise ValueError(
            f"Node '{node_id}' branch at index {branch_index} must have either "
            f"'expect' or 'label'."
        )

    return {
        "expect": canonical_expect,
        "next": next_value,
    }


def _normalize_node_dict(node_data: Dict[str, Any], *, node_index: int) -> Dict[str, Any]:
    """
    Normalize one node dict before passing it into ConditionalTask.from_dict().

    Compatibility policy:
    - old branch format {"expect": "...", "next": "..."} is accepted
    - new branch format {"label": "...", "next": "..."} is also accepted
    - extra fields such as 'verification' are preserved unless they interfere
      with downstream parsing
    """
    if not isinstance(node_data, dict):
        raise ValueError(f"Node at index {node_index} must be a JSON object.")

    normalized = dict(node_data)

    node_id = normalized.get("id", f"<node_{node_index}>")
    if "branches" in normalized:
        branches = normalized["branches"]
        if branches is None:
            normalized["branches"] = []
        elif not isinstance(branches, list):
            raise ValueError(f"Node '{node_id}' field 'branches' must be a list.")
        else:
            normalized["branches"] = [
                _normalize_branch_dict(branch, node_id=node_id, branch_index=i)
                for i, branch in enumerate(branches)
            ]

    return normalized


def load_conditional_plan(
    plan_source: Any,
    allowed_special_next: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Load a minimal conditional plan from JSON text or dict.

    Accepted branch formats inside each node:
      1) old format:
         {"expect": "...", "next": "S2"}

      2) new label-based format:
         {"label": "success", "next": "S2"}
         {"label": "execution", "next": "B1_S1"}
         {"label": "terminate", "next": "TERMINATE"}

    The loader normalizes `label` -> `expect` before constructing
    ConditionalTask objects, so existing execution code can remain largely
    unchanged.

    Expected top-level format:
    {
      "answer_contract": "...",
      "nodes": [
        {
          "id": "S1",
          "task": "...",
          "agent": "...",
          "deps": [],
          "branches": [
            {"label": "success", "next": "S2"},
            {"label": "execution", "next": "B1_S1"},
            {"label": "terminate", "next": "TERMINATE"}
          ]
        },
        ...
      ]
    }

    Args:
        plan_source:
            - JSON string, or
            - already-parsed dict
        allowed_special_next:
            Optional special transition labels allowed in branch.next.
            If omitted, {"TERMINATE"} is allowed by default.

    Returns:
        dict with:
        - answer_contract
        - tasks: List[ConditionalTask]
        - node_map: Dict[str, ConditionalTask]
        - start_node_id: str

    Raises:
        TypeError, ValueError on malformed input.
    """
    if allowed_special_next is None:
        allowed_special_next = {"TERMINATE"}
    else:
        allowed_special_next = set(allowed_special_next)

    # ---------------------------------------------------------
    # 1. Parse top-level JSON
    # ---------------------------------------------------------
    if isinstance(plan_source, str):
        text = plan_source.strip()
        if not text:
            raise ValueError("Plan text is empty.")

        extracted = extract_conditional_plan_json_text(text)
        raw_plan = json.loads(extracted)

    elif isinstance(plan_source, dict):
        raw_plan = plan_source

    else:
        raise TypeError("plan_source must be a JSON string or a dict.")

    if not isinstance(raw_plan, dict):
        raise ValueError("Top-level plan must be a JSON object.")

    # ---------------------------------------------------------
    # 2. Validate top-level fields
    # ---------------------------------------------------------
    if "answer_contract" not in raw_plan:
        raise ValueError("Missing top-level field 'answer_contract'.")
    if "nodes" not in raw_plan:
        raise ValueError("Missing top-level field 'nodes'.")

    answer_contract = raw_plan["answer_contract"]
    nodes_raw = raw_plan["nodes"]

    if not isinstance(answer_contract, str) or not answer_contract.strip():
        raise ValueError("'answer_contract' must be a non-empty string.")

    if not isinstance(nodes_raw, list) or len(nodes_raw) == 0:
        raise ValueError("'nodes' must be a non-empty list.")

    # ---------------------------------------------------------
    # 3. Normalize nodes before building ConditionalTask objects
    # ---------------------------------------------------------
    normalized_nodes_raw: List[Dict[str, Any]] = []
    for i, node_data in enumerate(nodes_raw):
        normalized_nodes_raw.append(_normalize_node_dict(node_data, node_index=i))

    # ---------------------------------------------------------
    # 4. Build ConditionalTask objects
    # ---------------------------------------------------------
    tasks: List[ConditionalTask] = []
    node_map: Dict[str, ConditionalTask] = {}

    for i, node_data in enumerate(normalized_nodes_raw):
        try:
            ctask = ConditionalTask.from_dict(node_data)
            ctask.validate()
        except Exception as e:
            raise ValueError(f"Invalid node at index {i}: {e}") from e

        if ctask.node_id in node_map:
            raise ValueError(f"Duplicate node id detected: '{ctask.node_id}'")

        tasks.append(ctask)
        node_map[ctask.node_id] = ctask

    node_ids = set(node_map.keys())

    # ---------------------------------------------------------
    # 5. Validate deps and branch targets
    # ---------------------------------------------------------
    for ctask in tasks:
        for dep_id in ctask.deps:
            if dep_id not in node_ids:
                raise ValueError(
                    f"Node '{ctask.node_id}' depends on unknown node '{dep_id}'."
                )

        for branch in ctask.branches:
            if branch.next not in node_ids and branch.next not in allowed_special_next:
                raise ValueError(
                    f"Node '{ctask.node_id}' has branch to unknown next node "
                    f"'{branch.next}'."
                )

    # ---------------------------------------------------------
    # 6. Infer start node
    # ---------------------------------------------------------
    no_dep_nodes = [t.node_id for t in tasks if len(t.deps) == 0]

    if len(no_dep_nodes) == 1:
        start_node_id = no_dep_nodes[0]
    elif len(tasks) > 0:
        # deterministic fallback: first node in the list
        start_node_id = tasks[0].node_id
    else:
        raise ValueError("Plan has no executable nodes.")

    # ---------------------------------------------------------
    # 7. Return normalized plan object
    # ---------------------------------------------------------
    return {
        "answer_contract": answer_contract.strip(),
        "tasks": tasks,
        "node_map": node_map,
        "start_node_id": start_node_id,
    }


class NewPlanningWorkflow(Workflow):
    """
    Thin wrapper around agent_hive.workflows.planning.PlanningWorkflow.

    Responsibilities here:
    - validate incoming Task list
    - translate Task / agent objects into planner inputs
    - call staged PlanningWorkflow.generate_plan_object()
    - adapt the returned final plan into ConditionalWorkflow input format
    - keep token accounting for watsonx
    """

    llm: str = Field(description="LLM used by the task planning.")
    mode: str = Field(default="force_verify", description="planner ablation mode")    

    def __init__(self, tasks: List[Task], llm: str, mode: str = "force_verify", enable_module1_review=True, module1_review_retries=5):
        self.tasks = tasks
        self.memory = []
        self.max_memory = 10
        self.llm = llm    
        self.mode = mode
        self.max_retries = 5
        self.enable_module1_review = enable_module1_review
        self.module1_review_retries = module1_review_retries
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        if len(self.tasks) != 1:
            raise ValueError("Planning only supports one task")
        task = self.tasks[0]
        if task.agents is None or len(task.agents) < 1:
            raise ValueError("Task must have at least one agent")

    def _build_tracking_llm_generate(self, token_counter):
        def _llm_generate(prompt: str) -> str:
            resp = watsonx_llm(prompt, model_id=self.llm)

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

    def _build_agent_catalog(self, task: Task) -> List[Dict[str, Any]]:
        agent_catalog: List[Dict[str, Any]] = []

        for aagent in task.agents:
            entry: Dict[str, Any] = {
                "name": getattr(aagent, "name", ""),
                "description": getattr(aagent, "description", ""),
            }

            task_examples = getattr(aagent, "task_examples", None)
            if task_examples:
                entry["task_examples"] = list(task_examples)

            agent_catalog.append(entry)

        return agent_catalog

    def run(self, save_plan=False, saved_plan_prefix="", qid=None):
        input_tokens_count = 0
        generated_tokens_count = 0

        plan_obj, user_q, in_tok, out_tok = self.generate_plan_object(
            save_plan=save_plan,
            saved_plan_filename=saved_plan_prefix,
            qid=qid,
        )
        input_tokens_count += in_tok
        generated_tokens_count += out_tok

        # normal task nodes が agent 名から実体 agent を引けるようにする
        plan_obj["available_agents"] = self.tasks[0].agents

        # verifier 用の LLM 呼び出しだけを計測する
        execution_token_counter = {"input": 0, "output": 0}
        execution_llm_generate = self._build_tracking_llm_generate(execution_token_counter)

        verification_agent = VerificationAgent(
            llm_generate=execution_llm_generate,
            max_retries=2,
            max_log_chars=30000,
        )

        conditional_workflow = ConditionalWorkflow(
            plan=plan_obj,
            verification_agent=verification_agent,
            user_q=user_q,
            context_type=ContextType.SELECTED,
        )

        history = conditional_workflow.run()

        # execution 側で verifier が使った分だけ加算
        input_tokens_count += execution_token_counter["input"]
        generated_tokens_count += execution_token_counter["output"]

        return history, input_tokens_count, generated_tokens_count

    def generate_plan_object(self, save_plan=False, saved_plan_filename="", qid=None):
        task = self.tasks[0]

        input_tokens_count = 0
        generated_tokens_count = 0
        planner_token_counter = {"input": 0, "output": 0}

        # =========================================================
        # TODO: Participants can edit this section ONLY
        # Purpose: Prepare planner inputs for staged PlanningWorkflow
        # =========================================================

        agent_catalog = self._build_agent_catalog(task)

        step_type_library = [
            {
                "name": "EntityGroundingStep",
                "purpose": "identify, disambiguate, or pin down the exact entity / variable / target / scope needed for later work",
            },
            {
                "name": "RetrievalStep",
                "purpose": "obtain data, evidence, documents, or external outputs",
            },
        ]

        constraints = None
        
        exception_taxonomy = [
            {
                "label": "availability",
                "description": "Required files, data, assets, thresholds, or parameters are missing, inaccessible, or not yet provided, so the task cannot proceed on a valid basis.",
                "representative_signals": [
                    "required file is missing",
                    "data source is not accessible",
                    "input artifact is not found",
                    "required parameter is missing",
                    "threshold or configuration value is unavailable"
                ]
            },
            {
                "label": "execution",
                "description": "The intended operation fails to run correctly because of tool misuse, missing invocation, runtime errors, context overflow, or unhandled exceptions.",
                "representative_signals": [
                    "tool was not called",
                    "tool invocation failed",
                    "runtime error occurred",
                    "context length exceeded",
                    "exception was not handled"
                ]
            },
            {
                "label": "grounding",
                "description": "The task target, entity, scope, identifier, or intended meaning is misunderstood or incorrectly selected.",
                "representative_signals": [
                    "wrong entity was selected",
                    "task scope was misread",
                    "incorrect dataset or asset was used",
                    "identifier does not match the requested target",
                    "the response addresses a different objective"
                ]
            },
            {
                "label": "core_action_missing",
                "description": "A required core action such as analysis, retrieval, classification, generation, or decision making was never actually performed.",
                "representative_signals": [
                    "required analysis is missing",
                    "core retrieval step was skipped",
                    "classification was not performed",
                    "expected artifact was never created",
                    "the response describes steps without executing them"
                ]
            },
            {
                "label": "coverage",
                "description": "Only part of the required scope, entities, assets, cases, or subtasks was handled, so the result is incomplete.",
                "representative_signals": [
                    "only some assets were checked",
                    "part of the requested scope is missing",
                    "not all subtasks were completed",
                    "one or more required cases were omitted",
                    "the result is only partially complete"
                ]
            },
            {
                "label": "contract",
                "description": "The final output does not satisfy the answer contract because it is missing, empty, malformed, or lacks required fields or artifacts.",
                "representative_signals": [
                    "final output is empty",
                    "required field is missing",
                    "output format is invalid",
                    "expected artifact is absent",
                    "schema requirements are not satisfied"
                ]
            },
            {
                "label": "verification",
                "description": "A claim, conclusion, or downstream decision is presented without sufficient supporting evidence, validation, or confirmation.",
                "representative_signals": [
                    "no supporting evidence is provided",
                    "claim is not verified",
                    "downstream use occurs before validation",
                    "result is asserted without confirmation",
                    "evidence-binding step is missing"
                ]
            },
            {
                "label": "quality",
                "description": "The output exists but is too vague, generic, incorrect, hallucinated, or based on undisclosed assumptions.",
                "representative_signals": [
                    "answer is vague or generic",
                    "hallucinated content is present",
                    "unsupported assumption is made",
                    "important uncertainty is not disclosed",
                    "the response sounds plausible but is not grounded"
                ]
            },
            {
                "label": "clarification_control",
                "description": "The workflow asks for clarification unnecessarily or routes to clarification before checking whether the needed information is already available.",
                "representative_signals": [
                    "clarification was requested despite sufficient input",
                    "existing input was not checked first",
                    "the system asked redundant follow-up questions",
                    "clarification was used as a premature fallback",
                    "missing-information judgment was incorrect"
                ]
            },
            {
                "label": "termination_recovery",
                "description": "The workflow stops too early or fails to recover appropriately from empty results, missing data, or intermediate failures.",
                "representative_signals": [
                    "process terminated prematurely",
                    "empty result was not handled properly",
                    "no recovery branch was attempted",
                    "the workflow stopped after a recoverable failure",
                    "fallback handling is missing"
                ]
            }
        ]

        agent_env_specs_json = {
          "unit_system_name": "ReActXen",
          "purpose": "Improve ReAct robustness by iterating ReAct(Self-Ask) -> Review -> Reflect with optional Distillation and memory updates until success or budget exhaustion.",
          "io": {
            "input": "query Q + optional context/memory",
            "output": "final answer or escalation/clarification"
          },
          "nodes": [
            {"id": "Env", "type": "environment"},
            {"id": "Distill", "type": "distillation_agent_optional"},
            {"id": "Mem", "type": "trajectory_store_or_memory"},
            {"id": "Obs", "type": "observation_formatter"},
            {"id": "ReAct", "type": "react_executor_with_self_ask"},
            {"id": "Review", "type": "review_verifier"},
            {"id": "Reflect", "type": "reflection_agent"}
          ],
          "edges": [
            {"from": "Env", "to": "Obs", "label": "raw_observation"},
            {"from": "Obs", "to": "ReAct", "label": "observation"},
            {"from": "Env", "to": "Distill", "label": "query_Q"},
            {"from": "Distill", "to": "ReAct", "label": "distilled_query_Q_prime"},
            {"from": "Mem", "to": "ReAct", "label": "memory_or_examples(TTS)_as_context"},
            {"from": "ReAct", "to": "Mem", "label": "append_trajectory_T"},
            {"from": "ReAct", "to": "ReAct", "label": "self_ask_internal_subquestion(r_i)_and_answer"},
            {"from": "ReAct", "to": "Env", "label": "action(A): external_act_triggers_observation"},
            {"from": "ReAct", "to": "Review", "label": "triple(Q_or_Q_prime, trajectory_T, final_output_o_final)"},
            {"from": "Review", "to": "Reflect", "label": "status_and_suggestions_if_needed"},
            {"from": "Reflect", "to": "Mem", "label": "feedback_to_update_prompt_or_memory"},
            {"from": "Review", "to": "Mem", "label": "review_feedback_to_update_prompt_or_memory"}
          ],
          "control": {
            "outer_loop": {
              "name": "verbal_feedback_loop",
              "max_iters": "T_max",
              "state_var": "t",
              "init": [
                "Receive query Q",
                "Optional: Q_prime <- Distill(Q)",
                "Initialize memory mem <- [TTS/examples]; t <- 0"
              ]
            },
            "inner_react_execution": {
              "react_step_budget": "max_react_step",
              "behavior": [
                "ReAct generates a trajectory T in response to (Q_prime, mem)",
                "Before executing any tool/action a_i in T, emit a self-ask sub-question r_i and answer it internally to improve trajectory coherence"
              ],
              "invariants": [
                "self_ask(r_i) and reasoning traces do not affect external environment (no new observation)",
                "only action(A) sent to Env can trigger new observation"
              ]
            },
            "review_and_break": {
              "inputs": ["Q_or_Q_prime", "ans_or_o_final", "trajectory_T"],
              "review_status_space": ["Accomplished", "Partially Accomplished", "Not Accomplished", "Error"],
              "rule": [
                {
                  "when": "final_output_exists",
                  "do": "review_t <- Review(Q_or_Q_prime, trajectory_T, o_final)"
                },
                {
                  "when": "review_t.status == Accomplished",
                  "transition": "TERMINATE",
                  "emit": "final_answer"
                },
                {
                  "when": "review_t.status == Error",
                  "transition": "TERMINATE",
                  "emit": "escalate_to_planner"
                }
              ]
            },
            "reflect_and_update": {
              "trigger": [
                "ReAct fails to finish",
                "review_t.status in {Partially Accomplished, Not Accomplished}",
                "or Review explicitly requests reflection"
              ],
              "do": [
                "reflect_t <- Reflect(Q_or_Q_prime, trajectory_T, review_t)",
                "Update mem with feedback from reflect_t and review_t",
                "t <- t + 1"
              ]
            },
            "termination": {
              "success": ["review.status == Accomplished"],
              "non_success": [
                "review.status == Error",
                "budget_exceeded (t reaches T_max)",
                "unrecoverable_failure"
              ],
              "on_budget_exceeded": "emit escalate_to_planner"
            }
          }
        }

        empirical_failure_profile = {
          "unit_system_name": "ReActXen",
          "planner_guardrail_profile": [
            {
              "failure_mode": "Step Repetition",
              "risk": "high",
              "why_it_remains": "Local repair is available, but explicit completed-step tracking remains weak.",
              "planner_countermeasure": "Track milestone completion explicitly and forbid repeated attempts on the same milestone without escalation."
            },
            {
              "failure_mode": "Unaware of Termination Conditions",
              "risk": "high",
              "why_it_remains": "Review provides local outcome judgment, not a global stopping policy.",
              "planner_countermeasure": "Add planner-level stop conditions and require milestone completion before termination."
            },
            {
              "failure_mode": "Premature Termination",
              "risk": "high",
              "why_it_remains": "A locally satisfactory result may be mistaken for overall task completion.",
              "planner_countermeasure": "Gate the final answer on verified evidence and explicit completion of required milestones."
            },
            {
              "failure_mode": "Weak Verification",
              "risk": "medium",
              "why_it_remains": "Verification is present, but its strictness may still be insufficient.",
              "planner_countermeasure": "Require stronger final checks for answer-evidence consistency and unresolved contradictions."
            },
            {
              "failure_mode": "Residual Task Derailment",
              "risk": "medium",
              "why_it_remains": "Distillation and reflection reduce drift, but they do not replace planner-level task anchoring.",
              "planner_countermeasure": "Use milestone-based dependencies and require each step to advance a named artifact."
            }
          ],
        }

        task_type_failure_priors = [
        {
          "task_label": "data_analysis",
          "failure_reason_labels": [
            "action_not_executed",
            "availability",
            "contract",
            "contract_misread",
            "core_action_plan_missing",
            "execution",
            "grounding",
            "premature_termination",
            "recovery_control_missing",
            "verification_plan_missing"
          ]
        },
        {
          "task_label": "data_handling",
          "failure_reason_labels": [
            "action_not_executed",
            "availability",
            "premature_termination",
            "recovery_control_missing",
            "unverified_intermediate_use",
            "verification_plan_missing"
          ]
        },
        {
          "task_label": "entity_or_metadata",
          "failure_reason_labels": [
            "availability",
            "execution",
            "premature_termination",
            "recovery_control_missing"
          ]
        },
        {
          "task_label": "event",
          "failure_reason_labels": [
            "availability",
            "contract_misread",
            "core_action_plan_missing",
            "coverage",
            "grounding",
            "premature_termination"
          ]
        },
        {
          "task_label": "failure_mode",
          "failure_reason_labels": [
            "availability",
            "contract_misread",
            "execution",
            "premature_termination",
            "recovery_control_missing",
            "unverified_intermediate_use",
            "verification_plan_missing"
          ]
        },
        {
          "task_label": "model",
          "failure_reason_labels": [
            "availability",
            "execution"
          ]
        },
        {
          "task_label": "sensor_data_retrieval",
          "failure_reason_labels": [
            "action_not_executed",
            "availability",
            "core_action_plan_missing",
            "coverage",
            "grounding",
            "premature_termination",
            "recovery_control_missing",
            "unresolved_target_ambiguity",
            "verification_plan_missing"
          ]
        },
        {
          "task_label": "sensor_identification",
          "failure_reason_labels": [
            "action_not_executed",
            "availability",
            "coverage",
            "recovery_control_missing",
            "unsupported_conclusion",
            "verification_plan_missing"
          ]
        },
        {
          "task_label": "work_order",
          "failure_reason_labels": [
            "action_not_executed",
            "availability",
            "contract",
            "contract_misread",
            "core_action_plan_missing",
            "coverage",
            "execution",
            "grounding",
            "premature_termination",
            "recovery_control_missing",
            "unsupported_conclusion",
            "verification_plan_missing"
          ]
        }
      ]
        

        # additional_agent_catalog = ["recovery", "replanning"]
        additional_agent_catalog = []

        # =========================================================
        # END OF EDITABLE SECTION
        # =========================================================

        planner_llm_generate = self._build_tracking_llm_generate(planner_token_counter)

        planner = PlanningWorkflow(
            llm_generate=planner_llm_generate,
            model_name=self.llm,
            validate_json=True,
            enable_module1_review=self.enable_module1_review,
            module1_review_retries=self.module1_review_retries,
        )

        final_plan = planner.generate_plan_object(
            task_description=task.description,
            agent_catalog=agent_catalog,
            step_type_library=step_type_library,
            output_format="json",
            constraints=constraints,
            exception_taxonomy=exception_taxonomy,
            agent_env_specs_json=agent_env_specs_json,
            empirical_failure_profile=empirical_failure_profile,
            additional_agent_catalog=additional_agent_catalog,
            task_type_failure_priors=task_type_failure_priors,
            mode=self.mode,
            qid=qid,
        )

        input_tokens_count += planner_token_counter["input"]
        generated_tokens_count += planner_token_counter["output"]

        logger.info(
            "Generated staged final plan JSON:\n%s",
            json.dumps(final_plan, ensure_ascii=False, indent=2)
        )

        self.memory = []

        if save_plan:
            if not saved_plan_filename.endswith(".txt"):
                saved_plan_filename += ".txt"

            saved_plan_text = (
                f"Question: {task.description}\n"
                f"Conditional Plan JSON:\n"
                f"{json.dumps(final_plan, ensure_ascii=False, indent=2)}"
            )
            with open(saved_plan_filename, "w") as f:
                f.write(saved_plan_text)


        plan_obj = load_conditional_plan(
            final_plan,
            allowed_special_next={"TERMINATE"}
        )

        logger.info(f"Loaded Conditional Plan: \n{plan_obj}")

        return plan_obj, task.description, input_tokens_count, generated_tokens_count