import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Union, Set
from pathlib import Path
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

def _load_first_valid_json_dict(plan_path: Path) -> Dict[str, Any]:
    text = plan_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not find a valid JSON object in: {plan_path}")

class PlanningWorkflow:
    """
    A staged offline planner with four modules:

    Module 1: Base Plan Generation
    Module 2: Possible Exception Annotation
    Module 3: Risk-Cost Branch Generation (node-level output)
    Module 4: Sub-plan Generation / Synthesis (branch-level output)

    Final integration is done by generate_plan_object().
    """

    def __init__(
        self,
        llm_generate: Callable[[str], Union[str, Dict[str, Any]]],
        model_name: str = "default",
        validate_json: bool = True,
        strict_branch_label_uniqueness: bool = True,
        enable_module1_review: bool = True,
        module1_review_retries: int = 2,
    ) -> None:
        self.llm_generate = llm_generate
        self.model_name = model_name
        self.validate_json = validate_json
        self.strict_branch_label_uniqueness = strict_branch_label_uniqueness

        self.enable_module1_review = bool(enable_module1_review)
        self.module1_review_retries = max(0, int(module1_review_retries))

    # -------------------------------------------------------------------------
    # Module 1
    # -------------------------------------------------------------------------
    def module1_generate_base_plan(
        self,
        task_description: str,
        agent_catalog: Union[List[Any], Dict[str, Any]],
        step_type_library: Union[List[Any], Dict[str, Any]],
        output_format: str = "json",
        constraints: Optional[Union[List[Any], Dict[str, Any], str]] = None,
        review_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate the normal-path base plan.

        Output schema:
        {
        "answer_contract": "<one concise sentence>",
        "nodes": [
            {
            "id": "S1",
            "task": "...",
            "agent": "...",
            "deps": [],
            "node_contract": "..."
            }
        ]
        }
        """
        if review_retries is None:
            review_retries = self.module1_review_retries
        review_retries = max(0, int(review_retries))

        prev_plan = None
        prev_review = None
        last_result = None

        total_rounds = 1
        if self.enable_module1_review and review_retries > 0:
            total_rounds = review_retries + 1

        for review_round in range(total_rounds):
            invalid_plan_description = ""
            if prev_plan is not None and prev_review is not None:
                invalid_reason = prev_review.get("reasoning", "")
                invalid_suggestions = prev_review.get("suggestions", "")
                invalid_plan_description = f"""
    Here is one invalid base plan. Learn from it and do not repeat its mistakes.

    Invalid base plan:
    {json.dumps(prev_plan, ensure_ascii=False, indent=2)}

    Reason why this base plan is invalid:
    {invalid_reason}

    Suggestion for improvement:
    {invalid_suggestions}
                """.strip()

            prompt = f"""
    You are Module 1 of a staged planner.

    Your job is to generate a normal-path base plan DAG only.
    Do not include exceptions.
    Do not include verification.
    Do not include branches.

    Inputs:
    - task_description:
    {task_description}

    - agent_catalog:
    {json.dumps(agent_catalog, ensure_ascii=False, indent=2, default=str)}

    - output_format:
    {output_format}

    - constraints:
    {json.dumps(constraints, ensure_ascii=False, indent=2, default=str)}

    Output JSON only.
    Do not output Markdown.
    Do not output code fences.
    Do not output explanations before or after the JSON.

    Return exactly one JSON object in the following format:
    {{
    "answer_contract": "<one concise sentence>",
    "nodes": [
        {{
        "id": "S1",
        "task": "<local task description with explicit constraints if needed>",
        "agent": "<one agent name from the agent_catalog>",
        "deps": [],
        "node_contract": "..."
        }}
    ]
    }}

    {invalid_plan_description}
            """.strip()

            allowed_agent_names = self._extract_agent_names(agent_catalog)

            result = self._call_llm_json_with_retry(
                base_prompt=prompt,
                kind="plan",
                validate_kind="plan",
                validate_kwargs={
                    "allowed_agent_names": allowed_agent_names,
                },
            )

            last_result = result

            if not self.enable_module1_review or review_retries == 0:
                return result

            review = self._review_module1_plan(
                task_description=task_description,
                agent_catalog=agent_catalog,
                plan_obj=result,
            )

            status = str(review.get("status", "")).strip().lower()
            if status == "valid":
                return result

            prev_plan = result
            prev_review = review

        return last_result

    # -------------------------------------------------------------------------
    # Module 2
    # -------------------------------------------------------------------------
    def module2_annotate_expected_exceptions(
        self,
        base_plan: Dict[str, Any],
        exception_taxonomy: Union[List[Any], Dict[str, Any]],
        task_type_failure_priors: Optional[Union[Dict[str, Any], str]] = None,
    ) -> Dict[str, Any]:
        """
        Annotate each node in the base plan with expected_exception.

        Output schema:
        {
        "answer_contract": "...",
        "nodes": [
            {
            "id": "S1",
            "task": "...",
            "agent": "...",
            "deps": [],
            "node_contract": "...",
            "expected_exception": [
                {
                "label": "execution",
                "signals": ["tool not called", "wrong tool selected"]
                }
            ]
            }
        ]
        }
        """
        task_type_failure_priors_json = json.dumps(
            task_type_failure_priors,
            ensure_ascii=False,
            indent=2,
            default=str,
        ) if task_type_failure_priors is not None else "null"

        prompt = f"""
    You are Module 2 of a staged planner.

    This is an in-place annotation task, not a plan regeneration task.

    Your only job is to add or update the `expected_exception` field for each existing node in the given base plan.

    You must preserve the base plan structure exactly.

    Strict preservation rules:
    - Copy `answer_contract` from the input `base_plan` exactly as-is.
    - Preserve the number of nodes exactly.
    - Preserve `answer_contract` exactly.
    - Preserve each node's `id` exactly.
    - Preserve each node's `task` exactly.
    - Preserve each node's `agent` exactly.
    - Preserve each node's `deps` exactly.
    - Preserve each node's `node_contract` exactly.
    - Do not reorder nodes.
    - Do not add new nodes.
    - Do not delete nodes.
    - Do not rename any field.
    - Do not modify any field other than adding or updating `expected_exception`.

    Exception annotation rules:
    - For each node, create `expected_exception` by using both the exception taxonomy and the task-type failure priors.
    - Use `task_type_failure_priors` as guidance about which exception labels are more likely for a given task type, but do not copy them blindly.
    - Assign labels that are genuinely plausible for the node given its task, agent, dependencies, and node_contract.
    - Avoid over-assigning labels. Do not attach many weakly supported labels just because they appear in the taxonomy or the priors.
    - At the same time, do not omit a clearly relevant failure label.
    - If multiple labels are plausible, include all of them.
    - Include the main realistic exception labels that could prevent the node from satisfying its `node_contract`.
    - Think in terms of branch usefulness: include a label if it would support a meaningful downstream recovery or verification branch.
    - For each included label, provide short representative signals that a later verifier could use.
    - If no taxonomy label is plausibly applicable, return an empty list for `expected_exception`.
    - When failures occur, multiple exception labels may co-occur for the same node; if 1 to 5 labels are simultaneously plausible and well-supported, include all of them.
    
    Important:
    - The output must have exactly the same top-level structure as the input `base_plan`.
    - The output must preserve `answer_contract`.
    - The output must preserve every original node and every original node field.
    - The only allowed semantic change is the addition or update of `expected_exception`.

    Inputs:
    - base_plan:
    {json.dumps(base_plan, ensure_ascii=False, indent=2)}

    - exception_taxonomy:
    {json.dumps(exception_taxonomy, ensure_ascii=False, indent=2, default=str)}

    - task_type_failure_priors:
    {task_type_failure_priors_json}

    Output JSON only.
    Do not output Markdown.
    Do not output code fences.
    Do not output explanations before or after the JSON.

    Return exactly one JSON object with the same structure as `base_plan`.
    That means:
    - the top-level object must contain `answer_contract` and `nodes`
    - `answer_contract` must be copied exactly from `base_plan`
    - each node must preserve `id`, `task`, `agent`, `deps`, and `node_contract`
    - each node must additionally contain `expected_exception`

    Example output shape:
    {{
    "answer_contract": base_plan["answer_contract"],
    "nodes": [
        {{
        "id": "<copied from base_plan>",
        "task": "<copied from base_plan>",
        "agent": "<copied from base_plan>",
        "deps": "<copied from base_plan>",
        "node_contract": "<copied from base_plan>",
        "expected_exception": [
            {{
            "label": "execution",
            "signals": [
                "tool not called",
                "wrong tool selected",
                "tool output absent",
                "tool output invalid"
            ]
            }}
        ]
        }}
    ]
    }}
    """.strip()

        result = self._call_llm_json_with_retry(
            base_prompt=prompt,
            kind="plan",
            validate_kind="plan",
        )

        return result


    # -------------------------------------------------------------------------
    # Module 3 (node-level)
    # -------------------------------------------------------------------------
    def _json(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)

    def _expected_exception_labels(self, node: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        for item in node.get("expected_exception", []):
            if isinstance(item, dict):
                label = item.get("label")
                if isinstance(label, str) and label.strip():
                    labels.append(label.strip())
        return labels

    def _normalize_branch_skeletons(
        self,
        node: Dict[str, Any],
        verification: bool,
        branches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not verification:
            return []

        allowed = set(self._expected_exception_labels(node))
        seen = set()
        out: List[Dict[str, Any]] = []

        for branch in branches:
            if not isinstance(branch, dict):
                continue
            label = branch.get("label")
            if not isinstance(label, str):
                continue
            label = label.strip()
            if not label:
                continue
            if label not in allowed:
                continue
            if label in seen:
                continue
            seen.add(label)
            out.append(
                {
                    "next": f"B_{node['id']}_{len(out) + 1}",
                    "label": label,
                }
            )
        return out

    def module3_interpret_execution_system(
        self,
        agent_env_specs_json: Union[List[Any], Dict[str, Any], str],
    ) -> Dict[str, Any]:
        few_shot_examples_data = [
            {
                "online_execution_system": {
                    "unit_system_name": "Reflexion",
                    "purpose": "Improve agent behavior across repeated trials by converting task feedback into verbal self-reflection, storing reflections in episodic memory, and conditioning subsequent attempts on that memory instead of updating model weights.",
                    "io": {
                        "input": "task + optional context + episodic memory",
                        "output": "task result or clarification request"
                    },
                    "nodes": [
                        {"id": "Env", "type": "environment_or_external_feedback_source"},
                        {"id": "Actor", "type": "trajectory_generator"},
                        {"id": "Evaluator", "type": "trajectory_feedback_provider"},
                        {"id": "SelfReflection", "type": "verbal_reflection_generator"},
                        {"id": "Memory", "type": "episodic_memory_buffer"}
                    ],
                    "edges": [
                        {"from": "Memory", "to": "Actor", "label": "prior_reflections_as_context"},
                        {"from": "Actor", "to": "Env", "label": "trajectory_or_action_sequence"},
                        {"from": "Env", "to": "Evaluator", "label": "task_outcome_or_feedback_signal"},
                        {"from": "Actor", "to": "Evaluator", "label": "trajectory_log"},
                        {"from": "Evaluator", "to": "SelfReflection", "label": "reward_or_feedback_signal"},
                        {"from": "Actor", "to": "SelfReflection", "label": "trajectory_log"},
                        {"from": "SelfReflection", "to": "Memory", "label": "new_verbal_reflection"}
                    ],
                    "control": {
                        "state": "Actor",
                        "loop_budget": {
                            "max_trials": 5,
                            "max_steps_per_trial": 10
                        },
                        "mode": {
                            "name": "trial_level_reflection",
                            "options": ["EXTERNAL_FEEDBACK", "INTERNAL_SELF_EVAL"]
                        },
                        "termination": {
                            "success": ["output_satisfied"],
                            "non_success": [
                                "clarification_needed",
                                "failure_after_max_trials",
                                "unrecoverable_failure",
                                "budget_exceeded"
                            ]
                        },
                        "invariants": [
                            "Reflections are stored as natural language in episodic memory",
                            "Learning occurs by updating context through memory, not by updating model weights",
                            "Reflection is generated after a trial from trajectory plus feedback",
                            "Subsequent trials are conditioned on prior reflections",
                            "The core improvement mechanism is cross-trial adaptation rather than in-trial local rollback"
                        ]
                    }
                },
                "output": {
                    "independent_check_strength": "moderate",
                    "local_repair_strength": "weak",
                    "evidence_confirmation_strength": "weak",
                    "termination_reliability": "moderate",
                    "summary": "Reflexion improves behavior by evaluating a completed trial, generating verbal self-reflection, and storing that reflection in episodic memory for subsequent trials. It is stronger at cross-trial improvement than at within-trial local repair."
                }
            },
            {
                "online_execution_system": {
                    "unit_system_name": "ReWOO",
                    "purpose": "Decouple reasoning from observations by generating a complete reasoning and evidence acquisition plan first, executing evidence-producing steps with variable binding, and synthesizing the final answer only after evidence collection.",
                    "io": {
                        "input": "subtask + optional context",
                        "output": "final answer or clarification request"
                    },
                    "nodes": [
                        {"id": "Planner", "type": "reasoning_planner"},
                        {"id": "Worker", "type": "tool_executor_over_evidence_steps"},
                        {"id": "Memory", "type": "evidence_store_with_variable_bindings"},
                        {"id": "Env", "type": "external_tools_or_knowledge_sources"},
                        {"id": "Obs", "type": "observation_formatter"},
                        {"id": "Solver", "type": "answer_synthesizer"}
                    ],
                    "edges": [
                        {"from": "Planner", "to": "Worker", "label": "plan_steps + evidence_slots"},
                        {"from": "Worker", "to": "Env", "label": "tool_call_for_current_evidence_step"},
                        {"from": "Env", "to": "Obs", "label": "raw_observation"},
                        {"from": "Obs", "to": "Worker", "label": "formatted_observation"},
                        {"from": "Worker", "to": "Memory", "label": "bind(Ei := result)"},
                        {"from": "Memory", "to": "Worker", "label": "variable_reference(E1,...,E{i-1})"},
                        {"from": "Memory", "to": "Solver", "label": "collected_evidence"}
                    ],
                    "control": {
                        "state": "Planner",
                        "loop_budget": {
                            "max_plan_passes": 1,
                            "max_worker_steps": 8,
                            "max_solver_passes": 1
                        },
                        "mode": {
                            "name": "plan_then_execute_then_solve",
                            "options": ["SERIAL_EVIDENCE_EXECUTION", "DEPENDENCY_AWARE_EXECUTION"]
                        },
                        "termination": {
                            "success": ["output_satisfied"],
                            "non_success": [
                                "clarification_needed_before_planning",
                                "critical_evidence_missing",
                                "plan_invalid_or_dependency_cycle_detected",
                                "budget_exceeded"
                            ]
                        },
                        "invariants": [
                            "Planner does not consume new external observations after execution begins",
                            "Reasoning is generated before observation collection, not interleaved with each observation",
                            "Each evidence slot Ei is bound once its step is executed or computed",
                            "Later evidence steps may reference earlier bindings",
                            "Solver receives collected evidence and produces the final answer"
                        ]
                    }
                },
                "output": {
                    "independent_check_strength": "moderate",
                    "local_repair_strength": "strong",
                    "evidence_confirmation_strength": "strong",
                    "termination_reliability": "moderate",
                    "summary": "ReWOO separates planning, evidence collection, and final synthesis. It supports strong local repair through step-level evidence rewriting and variable rebinding, and it is relatively strong at confirming evidence because intermediate results are explicitly stored."
                }
            },
            {
                "online_execution_system": {
                    "unit_system_name": "TreeOfThoughts",
                    "purpose": "Perform deliberate problem solving by exploring multiple thought candidates as tree nodes, evaluating partial reasoning states, and using search to select, expand, prune, or backtrack before committing to a final answer.",
                    "io": {
                        "input": "problem statement + optional context",
                        "output": "final answer or failure / clarification request"
                    },
                    "nodes": [
                        {"id": "Task", "type": "problem_instance"},
                        {"id": "SearchCtrl", "type": "tree_search_controller"},
                        {"id": "ThoughtGen", "type": "candidate_thought_generator"},
                        {"id": "StateEval", "type": "partial_state_evaluator"},
                        {"id": "Frontier", "type": "frontier_and_scored_state_store"},
                        {"id": "Answer", "type": "final_answer_renderer"}
                    ],
                    "edges": [
                        {"from": "Task", "to": "SearchCtrl", "label": "problem_specification"},
                        {"from": "SearchCtrl", "to": "ThoughtGen", "label": "selected_partial_state_for_expansion"},
                        {"from": "ThoughtGen", "to": "StateEval", "label": "candidate_thoughts"},
                        {"from": "StateEval", "to": "Frontier", "label": "scored_partial_states"},
                        {"from": "Frontier", "to": "SearchCtrl", "label": "frontier_snapshot + scores + parent_links"},
                        {"from": "SearchCtrl", "to": "SearchCtrl", "label": "select_expand_prune_backtrack"},
                        {"from": "SearchCtrl", "to": "Answer", "label": "best_complete_reasoning_path"}
                    ],
                    "control": {
                        "state": "SearchCtrl",
                        "loop_budget": {
                            "max_depth": 5,
                            "max_expansions": 20,
                            "beam_width": 5
                        },
                        "mode": {
                            "name": "tree_search_policy",
                            "options": ["BFS", "DFS"]
                        },
                        "termination": {
                            "success": ["solution_state_verified"],
                            "non_success": [
                                "all_frontier_states_pruned",
                                "problem_underspecified",
                                "budget_exceeded"
                            ]
                        },
                        "invariants": [
                            "A thought is a coherent intermediate text unit rather than a single next token",
                            "Multiple candidate reasoning paths may coexist simultaneously in the frontier",
                            "Search decisions are made over partial states after evaluation",
                            "Low quality states may be pruned before full rollout",
                            "DFS mode may backtrack; BFS mode may keep top states at each depth"
                        ]
                    }
                },
                "output": {
                    "independent_check_strength": "moderate",
                    "local_repair_strength": "strong",
                    "evidence_confirmation_strength": "weak",
                    "termination_reliability": "strong",
                    "summary": "Tree of Thoughts explores multiple reasoning branches and evaluates intermediate states before continuing. This makes local repair strong because poor branches can be pruned and promising branches can be extended or revisited."
                }
            }
        ]

        few_shot_examples = self._json(few_shot_examples_data)

        prompt = f"""
        You are the System Interpreter for Module 3 of a staged planner.

        Your job is to read the online execution system specification and summarize the following points.

        1. Whether this execution system has a strong independent mechanism for checking correctness and sufficiency, or whether it mainly relies on the executing agent's own judgment.
        2. Whether this execution system can locally repair small failures or minor errors on its own.
        3. Whether this execution system is strong or weak at confirming that the necessary information or evidence has actually been obtained before producing an output.
        4. Whether this execution system can terminate safely at an appropriate point, or whether it is prone to premature or insufficient termination.

        Below are a few-shot examples.
        Infer the underlying judgment rules from these examples.
        Use the same label semantics, decision criteria, and summary style when analyzing the target system.
        Do not invent new labels, new keys, or new evaluation dimensions.

        Return JSON only.
        The output must be a single valid JSON object and nothing else.

        Return exactly one JSON object in the following format:
        {{
        "independent_check_strength": "strong|moderate|weak|absent",
        "local_repair_strength": "strong|moderate|weak|absent",
        "evidence_confirmation_strength": "strong|moderate|weak|absent",
        "termination_reliability": "strong|moderate|weak|absent",
        "summary": "short natural-language summary"
        }}

        Few-shot examples:
        {few_shot_examples}

        Input:
        - online execution system:
        {self._json(agent_env_specs_json)}
        """.strip()

        return self._call_llm_json(prompt)

    def module3_infer_task_type_and_exposure(
        self,
        node: Dict[str, Any],
        execution_system_summary: Dict[str, Any],
        empirical_failure_profile: Union[List[Any], Dict[str, Any], str],
        task_type_failure_priors: Optional[Union[Dict[str, Any], str]] = None,
    ) -> Dict[str, Any]:
        priors_json = self._json(task_type_failure_priors) if task_type_failure_priors is not None else "null"

        few_shot_examples_data = [
  {
    "input": {
      "node": {
        "id": "S3",
        "task": "Analyze the downloaded sensor data to predict the risk of air leak failure within the next 7 days.",
        "agent": "Time Series Analytics and Forecasting",
        "deps": ["S1", "S2"],
        "node_contract": "Use the identified relevant sensors' data to forecast the risk of failure."
      },
      "execution_system_summary": {
        "independent_check_strength": "moderate",
        "local_repair_strength": "weak",
        "evidence_confirmation_strength": "weak",
        "termination_reliability": "moderate",
        "summary": "Reflexion improves behavior by evaluating a completed trial, generating verbal self-reflection, and storing that reflection in episodic memory for subsequent trials. It is stronger at cross-trial improvement than at within-trial local repair."
      },
      "empirical_failure_profile": {
        "unit_system_name": "Reflexion",
        "planner_guardrail_profile": [
          {
            "failure_mode": "Weak In-Trial Repair",
            "risk": "high",
            "why_it_remains": "The core improvement mechanism is cross-trial reflection through episodic memory, not in-trial local rollback or step-level repair.",
            "planner_countermeasure": "Insert explicit verification or checkpoint nodes before downstream use of intermediate outputs, and avoid assuming that reflection alone can repair within-trial mistakes."
          },
          {
            "failure_mode": "Step Repetition",
            "risk": "high",
            "why_it_remains": "Reflections may reduce repeated mistakes across trials, but explicit completed-step tracking and same-milestone repetition control remain weak within a trial.",
            "planner_countermeasure": "Track milestone completion explicitly and forbid repeated attempts on the same milestone unless the branch clearly represents a different continuation."
          },
          {
            "failure_mode": "Premature Termination",
            "risk": "high",
            "why_it_remains": "Evidence confirmation is weak, and success can be declared by output satisfaction without a strong independent final evidence gate.",
            "planner_countermeasure": "Gate the final answer on verified evidence artifacts and explicit completion of all required milestones."
          },
          {
            "failure_mode": "Unaware of Termination Conditions",
            "risk": "medium",
            "why_it_remains": "Termination reliability is moderate, but the system does not specify a strong planner-level global stopping controller tied to milestone completion.",
            "planner_countermeasure": "Add planner-level stop conditions and require milestone completion checks before termination."
          },
          {
            "failure_mode": "Weak Verification",
            "risk": "medium",
            "why_it_remains": "An evaluator exists, but independent check strength is only moderate and evidence confirmation remains weak; the evaluation is primarily trial-level rather than strict local verification.",
            "planner_countermeasure": "Require stronger answer-evidence consistency checks and explicit contradiction checks before accepting final outputs."
          },
          {
            "failure_mode": "Residual Task Derailment",
            "risk": "medium",
            "why_it_remains": "Cross-trial reflection helps future attempts, but it does not strongly anchor within-trial progress to named milestones or artifacts.",
            "planner_countermeasure": "Use milestone-based dependencies and require each node to advance a named artifact, decision, or evidence object."
          },
          {
            "failure_mode": "Fail to Ask for Clarification",
            "risk": "medium",
            "why_it_remains": "Clarification is an allowed non-success outcome, but ambiguity resolution is not strongly enforced before acting within a trial.",
            "planner_countermeasure": "Insert clarification-oriented nodes whenever required entities, constraints, or assumptions are missing or ambiguous."
          }
        ]
      }
    },
    "output": {
      "predicted_task_type": "data_analysis",
      "task_type_basis": "The node consumes already obtained sensor data and asks for forecasting and analysis of future failure risk, so it is an analysis task rather than retrieval or identification.",
      "task_prior_failure_labels": ["availability", "unsupported_conclusion"],
      "system_residual_weaknesses": [
        "Weak In-Trial Repair",
        "Step Repetition",
        "Premature Termination",
        "Unaware of Termination Conditions",
        "Weak Verification",
        "Residual Task Derailment",
        "Fail to Ask for Clarification"
      ],
      "node_exposed_failure_labels": ["availability", "unsupported_conclusion"],
      "exposure_judgment_summary": "This is a data_analysis node that depends on upstream sensor identification and downloaded data, so availability remains plausibly exposed. Reflexion is also weak at in-trial repair and has only moderate independent checking with weak evidence confirmation, so unsupported_conclusion is exposed for a forecasting step that may produce a weakly supported risk prediction."
    }
  },
  {
    "input": {
      "node": {
        "id": "S2",
        "task": "Identify relevant sensors for monitoring valve condition in asset hp_1.",
        "agent": "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
        "deps": ["S1"],
        "node_contract": "List sensors relevant to valve condition failure mode for hp_1."
      },
      "execution_system_summary": {
        "independent_check_strength": "moderate",
        "local_repair_strength": "strong",
        "evidence_confirmation_strength": "strong",
        "termination_reliability": "moderate",
        "summary": "ReWOO separates planning, evidence collection, and final synthesis. It supports strong local repair through step-level evidence rewriting and variable rebinding, and it is relatively strong at confirming evidence because intermediate results are explicitly stored."
      },
      "empirical_failure_profile": {
        "unit_system_name": "ReWOO",
        "planner_guardrail_profile": [
          {
            "failure_mode": "Front-Loaded Plan Error Propagation",
            "risk": "high",
            "why_it_remains": "Reasoning is generated before observation collection begins, so an early planning error or wrong decomposition can propagate through later evidence steps and final synthesis.",
            "planner_countermeasure": "Insert explicit plan validation before execution and require each evidence step to be traceably linked to a named subgoal or required output slot."
          },
          {
            "failure_mode": "Residual Clarification Failure",
            "risk": "medium",
            "why_it_remains": "Clarification can occur before planning, but ambiguity that is not resolved at plan time may propagate through the whole execution because planning is front-loaded.",
            "planner_countermeasure": "Add clarification-oriented nodes before planning whenever entities, constraints, assumptions, or required output slots are underspecified."
          },
          {
            "failure_mode": "Unaware of Termination Conditions",
            "risk": "medium",
            "why_it_remains": "Termination reliability is moderate, but the system does not specify a strong planner-level global stopping controller beyond output satisfaction and several non-success conditions.",
            "planner_countermeasure": "Require explicit completion checks over all required evidence slots and final output requirements before termination."
          },
          {
            "failure_mode": "Premature Termination",
            "risk": "medium",
            "why_it_remains": "Evidence confirmation is strong, but final success may still be declared once output appears satisfactory even if some required evidence slots or constraints were not fully discharged.",
            "planner_countermeasure": "Gate final answer emission on verified completion of all critical evidence bindings and answer-evidence consistency checks."
          },
          {
            "failure_mode": "Weak Independent Verification",
            "risk": "medium",
            "why_it_remains": "Evidence is explicitly stored and reused, but there is no clearly separate strict verifier beyond the Planner-Worker-Solver pipeline; independent check strength is only moderate.",
            "planner_countermeasure": "Add an explicit verification node before downstream use of synthesized outputs, especially for evidence-heavy or high-risk nodes."
          },
          {
            "failure_mode": "Step Repetition",
            "risk": "low",
            "why_it_remains": "Evidence slots are explicitly bound and later steps reference prior bindings, which reduces repeated execution, though no fully explicit milestone tracker is specified.",
            "planner_countermeasure": "Preserve slot-level execution logs and forbid re-executing an already satisfied evidence slot unless the branch explicitly revises that slot."
          },
          {
            "failure_mode": "Residual Task Derailment",
            "risk": "medium",
            "why_it_remains": "The plan-work-solve structure reduces local drift during execution, but if the original plan is mis-scoped or incomplete, the whole pipeline can remain anchored to the wrong decomposition.",
            "planner_countermeasure": "Use milestone-based plan structure and require each evidence step and solver claim to map back to named task requirements."
          }
        ]
      }
    },
    "output": {
      "predicted_task_type": "sensor_identification",
      "task_type_basis": "The node is explicitly about selecting and listing sensors that are relevant to a target asset and failure mode, so the core work is sensor identification.",
      "task_prior_failure_labels": ["grounding", "coverage"],
      "system_residual_weaknesses": [
        "Front-Loaded Plan Error Propagation",
        "Residual Clarification Failure",
        "Unaware of Termination Conditions",
        "Premature Termination",
        "Weak Independent Verification",
        "Step Repetition",
        "Residual Task Derailment"
      ],
      "node_exposed_failure_labels": ["grounding", "coverage"],
      "exposure_judgment_summary": "This is a sensor_identification node, so choosing the correct asset and failure-mode context is central, which makes grounding exposed under ReWOO because an early planning or clarification error can propagate through the whole pipeline. Coverage is also exposed because a front-loaded or mis-scoped decomposition can omit relevant sensors even when later execution is structured and evidence-aware."
    }
  },
  {
    "input": {
      "node": {
        "id": "S1",
        "task": "Identify assets located at the MAIN facility",
        "agent": "IoT Data Download",
        "deps": [],
        "node_contract": "The list of assets at the MAIN facility is retrieved."
      },
      "execution_system_summary": {
        "independent_check_strength": "moderate",
        "local_repair_strength": "strong",
        "evidence_confirmation_strength": "weak",
        "termination_reliability": "strong",
        "summary": "Tree of Thoughts explores multiple reasoning branches and evaluates intermediate states before continuing. This makes local repair strong because poor branches can be pruned and promising branches can be extended or revisited."
      },
      "empirical_failure_profile": {
        "unit_system_name": "TreeOfThoughts",
        "planner_guardrail_profile": [
          {
            "failure_mode": "Weak Evidence Grounding",
            "risk": "high",
            "why_it_remains": "The system is strong at evaluating partial reasoning states, but evidence confirmation is weak and the search operates primarily over internal thought states rather than explicit externally verified evidence bindings.",
            "planner_countermeasure": "Insert explicit evidence-check nodes before downstream use of factual or environment-dependent claims, and require answer-evidence consistency checks before final output."
          },
          {
            "failure_mode": "Weak Independent Verification",
            "risk": "medium",
            "why_it_remains": "State evaluation exists, but it is mainly used for search control over partial reasoning states rather than as a separate strict verifier of final answer correctness against external evidence.",
            "planner_countermeasure": "Add an explicit verifier node before accepting final outputs, especially for evidence-heavy, retrieval-heavy, or high-stakes nodes."
          },
          {
            "failure_mode": "Residual Search Misguidance",
            "risk": "medium",
            "why_it_remains": "Search decisions depend on the quality of partial-state evaluation; if the evaluator scores misleading branches too highly, the controller may keep expanding the wrong subtree.",
            "planner_countermeasure": "Require milestone-based evaluation criteria and explicit checks that frontier scoring aligns with required task outputs, not only local reasoning plausibility."
          },
          {
            "failure_mode": "Clarification Failure",
            "risk": "medium",
            "why_it_remains": "Problem underspecification is an explicit non-success condition, but clarification is not strongly enforced before the search begins, so the tree can still expand over an underspecified objective.",
            "planner_countermeasure": "Insert clarification-oriented nodes before search whenever entities, constraints, or target criteria are ambiguous."
          },
          {
            "failure_mode": "Premature Termination",
            "risk": "low",
            "why_it_remains": "Termination reliability is strong and success is tied to a verified solution state, which reduces but does not completely eliminate the risk of stopping on a locally strong but globally unsupported answer.",
            "planner_countermeasure": "Retain final completion checks over all required conditions and require explicit support for any evidence-dependent claims."
          },
          {
            "failure_mode": "Unaware of Termination Conditions",
            "risk": "low",
            "why_it_remains": "The search controller has explicit non-success outcomes such as all frontier states pruned, underspecification, and budget exhaustion, which makes stopping policy stronger than in many reactive systems.",
            "planner_countermeasure": "Preserve explicit search-stop rules and ensure that solution_state_verified is defined against all required task conditions."
          },
          {
            "failure_mode": "Step Repetition",
            "risk": "low",
            "why_it_remains": "The frontier store, scored partial states, and prune/backtrack operations reduce naive repetition, though redundant expansion of equivalent thought states can still occur if duplicate control is weak.",
            "planner_countermeasure": "Track explored partial states and forbid re-expanding semantically equivalent branches unless they contain genuinely new information."
          },
          {
            "failure_mode": "Residual Task Derailment",
            "risk": "low",
            "why_it_remains": "The combination of branching search and partial-state evaluation gives stronger local correction than single-path systems, but evaluator bias can still keep the search centered on the wrong objective.",
            "planner_countermeasure": "Tie search scoring to named task requirements and require each retained branch to preserve alignment with the original problem specification."
          }
        ]
      }
    },
    "output": {
      "predicted_task_type": "entity_or_metadata",
      "task_type_basis": "The node is about identifying and retrieving the correct asset set associated with a named facility, so it is an entity or metadata lookup task.",
      "task_prior_failure_labels": ["grounding", "availability"],
      "system_residual_weaknesses": [
        "Weak Evidence Grounding",
        "Weak Independent Verification",
        "Residual Search Misguidance",
        "Clarification Failure",
        "Premature Termination",
        "Unaware of Termination Conditions",
        "Step Repetition",
        "Residual Task Derailment"
      ],
      "node_exposed_failure_labels": ["grounding"],
      "exposure_judgment_summary": "This is an entity_or_metadata node whose main difficulty is identifying the correct facility-linked asset set, so grounding is exposed. Under Tree of Thoughts, weak evidence grounding and only moderate independent verification make it plausible that the search remains internally coherent while still selecting an externally unsupported interpretation of the MAIN facility, whereas availability is not especially amplified by this system profile for this node."
    }
  },
  {
  "input": {
    "node": {
      "id": "S1",
      "task": "Download compressor sensor data for asset mp_1 from May 29 to June 4, 2020.",
      "agent": "IoT Data Download",
      "deps": [],
      "node_contract": "Retrieve historical sensor data for the specified asset and date range."
    },
    "execution_system_summary": {
      "independent_check_strength": "moderate",
      "local_repair_strength": "weak",
      "evidence_confirmation_strength": "weak",
      "termination_reliability": "moderate",
      "summary": "Reflexion improves behavior by evaluating a completed trial, generating verbal self-reflection, and storing that reflection in episodic memory for subsequent trials. It is stronger at cross-trial improvement than at within-trial local repair."
    },
    "empirical_failure_profile": {
      "unit_system_name": "Reflexion",
      "planner_guardrail_profile": [
        {
          "failure_mode": "Weak In-Trial Repair",
          "risk": "high",
          "why_it_remains": "The core improvement mechanism is cross-trial reflection through episodic memory, not in-trial local rollback or step-level repair.",
          "planner_countermeasure": "Insert explicit verification or checkpoint nodes before downstream use of intermediate outputs, and avoid assuming that reflection alone can repair within-trial mistakes."
        },
        {
          "failure_mode": "Step Repetition",
          "risk": "high",
          "why_it_remains": "Reflections may reduce repeated mistakes across trials, but explicit completed-step tracking and same-milestone repetition control remain weak within a trial.",
          "planner_countermeasure": "Track milestone completion explicitly and forbid repeated attempts on the same milestone unless the branch clearly represents a different continuation."
        },
        {
          "failure_mode": "Premature Termination",
          "risk": "high",
          "why_it_remains": "Evidence confirmation is weak, and success can be declared by output satisfaction without a strong independent final evidence gate.",
          "planner_countermeasure": "Gate the final answer on verified evidence artifacts and explicit completion of all required milestones."
        },
        {
          "failure_mode": "Unaware of Termination Conditions",
          "risk": "medium",
          "why_it_remains": "Termination reliability is moderate, but the system does not specify a strong planner-level global stopping controller tied to milestone completion.",
          "planner_countermeasure": "Add planner-level stop conditions and require milestone completion checks before termination."
        },
        {
          "failure_mode": "Weak Verification",
          "risk": "medium",
          "why_it_remains": "An evaluator exists, but independent check strength is only moderate and evidence confirmation remains weak; the evaluation is primarily trial-level rather than strict local verification.",
          "planner_countermeasure": "Require stronger answer-evidence consistency checks and explicit contradiction checks before accepting final outputs."
        },
        {
          "failure_mode": "Residual Task Derailment",
          "risk": "medium",
          "why_it_remains": "Cross-trial reflection helps future attempts, but it does not strongly anchor within-trial progress to named milestones or artifacts.",
          "planner_countermeasure": "Use milestone-based dependencies and require each node to advance a named artifact, decision, or evidence object."
        },
        {
          "failure_mode": "Fail to Ask for Clarification",
          "risk": "medium",
          "why_it_remains": "Clarification is an allowed non-success outcome, but ambiguity resolution is not strongly enforced before acting within a trial.",
          "planner_countermeasure": "Insert clarification-oriented nodes whenever required entities, constraints, or assumptions are missing or ambiguous."
        }
      ]
    }
  },
  "output": {
    "predicted_task_type": "sensor_data_retrieval",
    "task_type_basis": "The node is explicitly about downloading historical sensor data for a specified asset and date range, so the core work is sensor data retrieval rather than analysis or identification.",
    "task_prior_failure_labels": [
      "availability",
      "execution"
    ],
    "system_residual_weaknesses": [
      "Weak In-Trial Repair",
      "Step Repetition",
      "Premature Termination",
      "Unaware of Termination Conditions",
      "Weak Verification",
      "Residual Task Derailment",
      "Fail to Ask for Clarification"
    ],
    "node_exposed_failure_labels": [
      "availability",
      "execution"
    ],
    "exposure_judgment_summary": "This is a sensor_data_retrieval node that depends on access to the correct historical data source for a specific asset and date range, so availability is directly exposed. Under Reflexion, weak in-trial repair also makes execution exposed, because a failed or malformed retrieval step is not strongly corrected within the same trial and may remain unresolved until a later retry."
  }
}
]

        few_shot_examples = self._json(few_shot_examples_data)

        prompt = f"""
You are the Task Type / Failure Prior / Exposure Interpreter for Module 3 of a staged planner.

Your job is to analyze exactly ONE node.

Before making the final judgment, use the following output fields in the precise sense defined below.

Field meanings:

1. predicted_task_type
This is the task category that best matches what the node is trying to do.
It should describe the kind of work the node performs, not the topic or domain.

2. task_type_basis
This is a short explanation of why the node belongs to that task category.
Use the node's task, node contract, dependencies, and expected exceptions as evidence.

3. task_prior_failure_labels
These are the failure labels that are generally likely for this task type.
They come from task type failure priors and should be treated as prior expectations, not as final exposure judgments.

4. system_residual_weaknesses
These are the failure modes that this execution system is weak at preventing, detecting, or safely absorbing on its own.
They come from the empirical failure profile and describe system-level weaknesses, not node-specific exception labels.

5. node_exposed_failure_labels
These are the failure labels that this specific node is actually exposed to under this execution system.
Only include labels that are plausible for this node after combining:
(a) the task-type-level likely failures, and
(b) the residual weaknesses of the execution system.

6. exposure_judgment_summary
This is a short natural-language summary of the final exposure judgment.
It should connect the task type, likely task failures, system weaknesses, and node-specific exposed versus ruled-out labels.

Reasoning rules:

1. Read the node and determine its most likely task category.

2. Read the task type failure priors and identify the failure labels that are generally likely for that task category.
   - These are not failures that automatically apply to this node.
   - They are the general difficulties or failure tendencies associated with that task category.

3. Read the empirical failure profile and identify the residual weaknesses of this execution system.
   - These are not failures that automatically apply to this node.
   - They are execution-system-specific weaknesses that may affect many task categories, rather than weaknesses tied to one task category alone.

4. Combine the task-category-level likely failures with the system-level residual weaknesses, and decide which likely failure labels are actually exposed for this node and which should be ruled out.
   - A failure label should be marked as exposed only if it is supported both by the task prior and by the node's actual situation under this system.
   - Keep the reasoning conservative and specific to this node.
   - Focus on actual failure manifestation risk under this execution system, not on abstract possibility.

Additional constraint:
- If task_type_failure_priors is null, infer the task category and likely task failures from the node and the empirical failure profile alone.

Return JSON only.

Return exactly one JSON object in the following format:
{{
  "predicted_task_type": "string",
  "task_type_basis": "short explanation",
  "task_prior_failure_labels": ["label1", "label2"],
  "system_residual_weaknesses": ["mode1", "mode2"],
  "node_exposed_failure_labels": ["label1"],
  "exposure_judgment_summary": "short natural-language summary"
}}

Few-shot examples:
{few_shot_examples}

Inputs:
- node:
{self._json(node)}

- execution system summary:
{self._json(execution_system_summary)}

- empirical_failure_profile:
{self._json(empirical_failure_profile)}

- task_type_failure_priors:
{priors_json}
        """.strip()

        result = self._call_llm_json(prompt)

        allowed = set(self._expected_exception_labels(node))

        likely = result.get("likely_failure_labels", [])
        exposed = result.get("exposed_failure_labels", [])
        non_exposed = result.get("non_exposed_failure_labels", [])

        if not isinstance(likely, list):
            likely = []
        if not isinstance(exposed, list):
            exposed = []
        if not isinstance(non_exposed, list):
            non_exposed = []

        result["likely_failure_labels"] = [
            x.strip() for x in likely if isinstance(x, str) and x.strip()
        ]
        result["exposed_failure_labels"] = [
            x.strip() for x in exposed if isinstance(x, str) and x.strip() in allowed
        ]
        result["non_exposed_failure_labels"] = [
            x.strip() for x in non_exposed if isinstance(x, str) and x.strip() in allowed
        ]
        return result

    def module3_decide_risk_cost(
        self,
        node: Dict[str, Any],
        execution_system_summary: Dict[str, Any],
        task_and_exposure_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        few_shot_examples_data = [
        {
            "node": {
                "id": "S2",
                "task": "Analyze the downloaded IoT data to assess the compressor's condition",
                "agent": "Time Series Analytics and Forecasting",
                "deps": ["S1"],
                "node_contract": "The compressor's condition is evaluated based on the IoT data.",
                "expected_exception": [
                    {
                        "label": "availability",
                        "signals": [
                            "downloaded IoT data is missing or unusable",
                            "required input data for condition assessment is unavailable"
                        ]
                    },
                    {
                        "label": "action_not_executed",
                        "signals": [
                            "analysis step is not actually performed",
                            "output is produced without running the intended analytic procedure"
                        ]
                    },
                    {
                        "label": "unsupported_conclusion",
                        "signals": [
                            "condition judgment is stated without sufficient support from the data",
                            "analysis output looks plausible but is not grounded in evidence"
                        ]
                    }
                ]
            },
            "execution_system_summary": {
                "unit_system_name": "TreeOfThoughts",
                "independent_check_strength": "moderate",
                "local_repair_strength": "strong",
                "evidence_confirmation_strength": "weak",
                "termination_reliability": "strong",
                "summary": "Tree of Thoughts explores multiple reasoning branches and evaluates intermediate states before continuing. This makes local repair strong because poor branches can be pruned and promising branches can be extended or revisited."
            },
            "task_and_exposure_summary": {
                "task_type": "data_analysis",
                "task_type_failure_priors": [
                    "availability",
                    "action_not_executed",
                    "recovery_control_missing",
                    "execution",
                    "unsupported_conclusion"
                ],
                "plausible_failure_labels_for_this_node": [
                    "availability",
                    "action_not_executed",
                    "unsupported_conclusion"
                ],
                "exposure_summary": "This node makes an environment-dependent condition judgment from downloaded IoT data. If the data is unavailable or the analytic step is not actually executed, the node can still produce a superficially plausible but unsupported condition assessment.",
                "downstream_impact": "high",
                "system_absorbability": "limited",
                "system_reason": "TreeOfThoughts is strong at exploring and pruning reasoning branches, but evidence confirmation is weak and independent verification is only moderate for externally grounded claims."
            },
            "output": {
                "estimated_failure_risk": "high",
                "risk_summary": "The node is exposed to data unavailability, skipped analysis, and unsupported conclusions, and these failures can yield a plausible but ungrounded condition assessment.",
                "verification_cost": "one_additional_node",
                "cost_summary": "The added cost is only one verifier node, which is small relative to the cost of propagating an unsupported condition judgment.",
                "verification": True,
                "decision_summary": "Add verification because this is an evidence-dependent analysis node with high propagation risk, and the assigned system is not strong enough at independent evidence confirmation to safely absorb the likely failures."
            }
        },
        {
            "node": {
                "id": "S1",
                "task": "List all failure modes of Chiller 6 and identify the relevant sensors for chiller trip failure.",
                "agent": "Failure Mode and Sensor Relevancy Expert for Industrial Asset",
                "deps": [],
                "node_contract": "A list of failure modes and relevant sensors for Chiller 6.",
                "expected_exception": [
                    {
                        "label": "execution",
                        "signals": [
                            "failure modes or sensors are only partially listed",
                            "the intended identification step is incomplete or incorrectly carried out"
                        ]
                    },
                    {
                        "label": "availability",
                        "signals": [
                            "required asset knowledge is not available",
                            "needed information about Chiller 6 cannot be retrieved or used"
                        ]
                    },
                    {
                        "label": "contract_misread",
                        "signals": [
                            "the output omits either failure modes or relevant sensors",
                            "the node answers a narrower or different question than the contract requires"
                        ]
                    }
                ]
            },
            "execution_system_summary": {
                "unit_system_name": "ReWOO",
                "independent_check_strength": "moderate",
                "local_repair_strength": "strong",
                "evidence_confirmation_strength": "strong",
                "termination_reliability": "moderate",
                "summary": "ReWOO separates planning, evidence collection, and final synthesis. It supports strong local repair through step-level evidence rewriting and variable rebinding, and it is relatively strong at confirming evidence because intermediate results are explicitly stored."
            },
            "task_and_exposure_summary": {
                "task_type": "failure_mode",
                "task_type_failure_priors": [
                    "availability",
                    "execution",
                    "contract_misread",
                    "recovery_control_missing",
                    "verification_plan_missing",
                    "premature_termination"
                ],
                "plausible_failure_labels_for_this_node": [
                    "execution",
                    "availability",
                    "contract_misread"
                ],
                "exposure_summary": "This node must identify failure modes and relevant sensors, so omission or misreading is plausible. However, it is an initial evidence-building node rather than a final analytic judgment, and its output is naturally representable as explicit evidence slots.",
                "downstream_impact": "medium",
                "system_absorbability": "good",
                "system_reason": "ReWOO explicitly stores intermediate evidence, supports variable binding and step-level correction, and is relatively strong at evidence confirmation compared with the other assigned systems."
            },
            "output": {
                "estimated_failure_risk": "medium",
                "risk_summary": "Some omission or contract misreading is plausible, but this node mainly builds an evidence inventory and the likely failures are comparatively absorbable within the assigned system.",
                "verification_cost": "one_additional_node",
                "cost_summary": "Verification still costs one additional node, and here that cost is not clearly justified because the system can often contain or revise this kind of intermediate evidence error.",
                "verification": False,
                "decision_summary": "Do not add verification because the node is only medium risk and ReWOO is relatively strong at explicit evidence handling and local correction, so the likely failures do not justify paying an extra node."
            }
        },
        {
            "node": {
                "id": "S2",
                "task": "Download sensor data for relevant sensors of asset hp_1 from days leading up to and including 2023-01-28.",
                "agent": "IoT Data Download",
                "deps": ["S1"],
                "node_contract": "Retrieve historical sensor data for identified relevant sensors of asset hp_1 up to 2023-01-28.",
                "expected_exception": [
                    {
                        "label": "availability",
                        "signals": [
                            "requested sensor data is missing or inaccessible",
                            "historical records for the required period are unavailable"
                        ]
                    },
                    {
                        "label": "grounding",
                        "signals": [
                            "the retrieved data does not correspond to the intended sensors",
                            "sensor identity or asset mapping is incorrect"
                        ]
                    },
                    {
                        "label": "coverage",
                        "signals": [
                            "the returned data does not cover the requested time window",
                            "some relevant sensors or dates are missing from the retrieval"
                        ]
                    }
                ]
            },
            "execution_system_summary": {
                "unit_system_name": "Reflexion",
                "independent_check_strength": "moderate",
                "local_repair_strength": "weak",
                "evidence_confirmation_strength": "weak",
                "termination_reliability": "moderate",
                "summary": "Reflexion improves behavior by evaluating a completed trial, generating verbal self-reflection, and storing that reflection in episodic memory for subsequent trials. It is stronger at cross-trial improvement than at within-trial local repair."
            },
            "task_and_exposure_summary": {
                "task_type": "sensor_data_retrieval",
                "task_type_failure_priors": [
                    "availability",
                    "grounding",
                    "coverage"
                ],
                "plausible_failure_labels_for_this_node": [
                    "availability",
                    "grounding",
                    "coverage"
                ],
                "exposure_summary": "This node retrieves date-bounded historical sensor data for sensors identified upstream. It is directly exposed to missing data, wrong sensor grounding, and incomplete temporal coverage.",
                "downstream_impact": "medium",
                "system_absorbability": "weak",
                "system_reason": "Reflexion is better at cross-trial improvement than at in-trial local rollback, and its evidence confirmation and strict local verification are only weak to moderate."
            },
            "output": {
                "estimated_failure_risk": "medium",
                "risk_summary": "The node is exposed to missing data, wrong sensor grounding, and incomplete coverage, and those failures can silently corrupt downstream analysis even when the retrieval appears superficially successful.",
                "verification_cost": "one_additional_node",
                "cost_summary": "The added cost is one verifier node, which is justified because the assigned system is weak at in-trial local repair and weak at strict evidence confirmation.",
                "verification": True,
                "decision_summary": "Add verification because this retrieval node has meaningful environment-dependent failure exposure and Reflexion is not strong enough at within-trial recovery to safely absorb those failures without an explicit check."
            }
        }
    ]
        
        few_shot_examples = self._json(few_shot_examples_data)

        prompt = f"""
You are the Risk-Cost Decision module for Module 3 of a staged planner.

Your job is to decide whether verification should be added to ONE node.

Reason as follows:
- Read the execution system summary.
- Read the task-type and exposure analysis.
- Estimate the risk of not adding verification.
- In particular, judge whether, if the likely failure occurs, this execution system could safely absorb it on its own without explicit verification.
- Estimate the cost of adding verification.
- Treat verification as the cost of executing one additional node.

Use the following tradeoff:
- without verification: execute one node and accept the risk that failure may occur and propagate;
- with verification: spend one additional node of budget to reduce or contain that failure risk.

Decision rule:
- Add verification only when the estimated risk of leaving this node entirely to online execution is greater than the cost of spending one additional node of budget on verification.
- Otherwise, do not add verification.

A possible failure alone is not enough to justify verification.
Only add verification when the likely failure is both plausible for this node and costly enough that paying for one additional node is justified.

Return JSON only.

Return exactly one JSON object in the following format:
{{
  "estimated_failure_risk": "high|medium|low",
  "risk_summary": "short explanation",
  "verification_cost": "one_additional_node",
  "cost_summary": "short explanation",
  "verification": true,
  "decision_summary": "short explanation"
}}

Few-shot examples:
{few_shot_examples}

Inputs:
- node:
{self._json(node)}

- execution system summary:
{self._json(execution_system_summary)}

- task and exposure summary:
{self._json(task_and_exposure_summary)}
        """.strip()

        return self._call_llm_json(prompt)

    def module3_select_branches(
        self,
        node: Dict[str, Any],
        task_and_exposure_summary: Dict[str, Any],
        risk_cost_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = f"""
You are the Branch Selector for Module 3 of a staged planner.

Your job is to decide which minimal exception branch skeletons should be created for ONE node.

Rules:
- If verification is false, return branches=[].
- If verification is true, create branches only for exception labels that may need repair or alternative handling.
- Do not create branches for "success" or "terminate".
- Use only exception labels that are justified by the node and its likely exposure.
- The branch labels must be unique within this node.
- The final "next" ids will be normalized later, so you only need to return the labels that should branch.

Return JSON only.

Return exactly one JSON object in the following format:
{{
  "verification": true,
  "branch_labels": ["label1", "label2"],
  "selection_summary": "short explanation"
}}

Inputs:
- node:
{self._json(node)}

- task and exposure summary:
{self._json(task_and_exposure_summary)}

- risk cost decision:
{self._json(risk_cost_decision)}
        """.strip()

        result = self._call_llm_json(prompt)

        verification = bool(result.get("verification", False))
        branch_labels = result.get("branch_labels", [])
        if not isinstance(branch_labels, list):
            branch_labels = []

        branches = self._normalize_branch_skeletons(
            node=node,
            verification=verification,
            branches=[{"label": x} for x in branch_labels if isinstance(x, str)],
        )

        return {
            "verification": verification,
            "branches": branches,
            "selection_summary": result.get("selection_summary", ""),
        }

    def module3_generate_verification_for_node(
        self,
        node: Dict[str, Any],
        agent_env_specs_json: Union[List[Any], Dict[str, Any], str],
        empirical_failure_profile: Union[List[Any], Dict[str, Any], str],
        task_type_failure_priors: Optional[Union[Dict[str, Any], str]] = None,
    ) -> Dict[str, Any]:
        """
        Final output shape:
        {
          "id": "S1",
          "task": "...",
          "agent": "...",
          "deps": [],
          "node_contract": "...",
          "expected_exception": [...],
          "verification": true|false,
          "branches": [...]
        }
        """
        system_summary = self.module3_interpret_execution_system(
            agent_env_specs_json=agent_env_specs_json
        )

        task_and_exposure_summary = self.module3_infer_task_type_and_exposure(
            node=node,
            execution_system_summary=system_summary,
            empirical_failure_profile=empirical_failure_profile,
            task_type_failure_priors=task_type_failure_priors,
        )

        risk_cost_decision = self.module3_decide_risk_cost(
            node=node,
            execution_system_summary=system_summary,
            task_and_exposure_summary=task_and_exposure_summary,
        )

        # branch_selection = self.module3_select_branches(
        #     node=node,
        #     task_and_exposure_summary=task_and_exposure_summary,
        #     risk_cost_decision=risk_cost_decision,
        # )

        final_node = copy.deepcopy(node)

        verification = bool(risk_cost_decision.get("verification", False))
        final_node["verification"] = verification

        raw_branch_labels = task_and_exposure_summary.get("node_exposed_failure_labels", [])
        if not isinstance(raw_branch_labels, list):
            raw_branch_labels = []

        # expected_exception に含まれる label だけを許可するなら、ここで絞る
        allowed_labels = {
            ex.get("label")
            for ex in node.get("expected_exception", [])
            if isinstance(ex, dict) and isinstance(ex.get("label"), str)
        }

        branch_dicts = []
        seen = set()

        if verification:
            for x in raw_branch_labels:
                if not isinstance(x, str):
                    continue
                label = x.strip()
                if not label:
                    continue
                if label in {"success", "terminate"}:
                    continue
                if allowed_labels and label not in allowed_labels:
                    continue
                if label in seen:
                    continue
                seen.add(label)
                branch_dicts.append({"label": label})

        final_node["branches"] = self._normalize_branch_skeletons(
            node=node,
            verification=verification,
            branches=branch_dicts,
        )
        # final_node["verification"] = bool(branch_selection["verification"])
        # final_node["branches"] = branch_selection["branches"] if final_node["verification"] else []

        if "expected_exception" in node:
            final_node["expected_exception"] = node["expected_exception"]

        return final_node


    def module3_generate_verifications_for_plan(
        self,
        annotated_plan: Dict[str, Any],
        agent_env_specs_json: Union[List[Any], Dict[str, Any], str],
        empirical_failure_profile: Union[List[Any], Dict[str, Any], str],
    ) -> List[Dict[str, Any]]:
        """
        Apply Module 3 node-by-node across the plan.

        Returns:
            List[Dict[str, Any]]: list of updated node objects using the minimal schema
        """
        nodes = annotated_plan.get("nodes", [])
        updated_nodes: List[Dict[str, Any]] = []

        for node in nodes:
            # updated_node = self.module3_generate_verification_for_node(
            #     node=node,
            #     agent_env_specs_json=agent_env_specs_json,
            #     empirical_failure_profile=empirical_failure_profile,
            # )
            # if self.validate_json:
            #     self._validate_plan_object(updated_node, kind="node")

            for attempt in range(3):
                try:
                    updated_node = self.module3_generate_verification_for_node(
                        node=node,
                        agent_env_specs_json=agent_env_specs_json,
                        empirical_failure_profile=empirical_failure_profile,
                    )
                    if self.validate_json:
                        self._validate_plan_object(updated_node, kind="node")
                    break
                except Exception as e:
                    last_error = e
                    if attempt == 2:
                        raise
            else:
                raise last_error
            
            updated_nodes.append(updated_node)

        return updated_nodes
    
    def module3_force_verifications_for_plan(
        self,
        annotated_plan: Dict[str, Any],
        fallback_exception_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fixed-verification ablation for Module 3.

        Unlike the adaptive Module 3, this function does NOT compare
        estimated failure risk against verification/subplan cost.
        Instead, it forces verification on every node.

        Input:
            annotated_plan = {
                "answer_contract": ...,
                "nodes": [
                    {
                        "id": "S1",
                        "task": ...,
                        "agent": ...,
                        "deps": ...,
                        "node_contract": ...,
                        "expected_exception": [
                            {"label": "execution", ...},
                            {"label": "coverage", ...},
                            ...
                        ]
                    },
                    ...
                ]
            }

        Output:
            module3_nodes = [
                {
                    ... original node fields ...,
                    "verification": True,
                    "branches": [
                        {"label": "execution", "next": "B_S1_1"},
                        {"label": "coverage",  "next": "B_S1_2"},
                        ...
                    ]
                },
                ...
            ]

        Notes:
        - success / terminate branches are NOT created here.
        They are handled later by merge/runtime verifier logic.
        - If expected_exception is empty, fallback_exception_labels
        are used so that every node still gets at least one repair branch.
        """
        if fallback_exception_labels is None:
            fallback_exception_labels = ["execution"]

        if not isinstance(annotated_plan, dict):
            raise TypeError("annotated_plan must be a dict")

        nodes = annotated_plan.get("nodes", None)
        if not isinstance(nodes, list):
            raise ValueError("annotated_plan must contain a list field 'nodes'")

        # sanitize fallback labels once
        fallback_labels: List[str] = []
        for lbl in fallback_exception_labels:
            if isinstance(lbl, str) and lbl.strip():
                if lbl not in fallback_labels:
                    fallback_labels.append(lbl.strip())

        if len(fallback_labels) == 0:
            raise ValueError("fallback_exception_labels must contain at least one non-empty label")

        module3_nodes: List[Dict[str, Any]] = []

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise TypeError(f"annotated_plan['nodes'][{idx}] must be a dict")

            updated = copy.deepcopy(node)

            node_id = updated.get("id", None)
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(f"annotated_plan['nodes'][{idx}] is missing a valid string field 'id'")

            expected_exception = updated.get("expected_exception", [])
            if expected_exception is None:
                expected_exception = []
            if not isinstance(expected_exception, list):
                raise ValueError(
                    f"annotated_plan['nodes'][{idx}]['expected_exception'] must be a list"
                )

            labels: List[str] = []

            for j, exc in enumerate(expected_exception):
                if not isinstance(exc, dict):
                    continue

                label = exc.get("label", None)
                if isinstance(label, str) and label.strip():
                    label = label.strip()
                    if label not in labels:
                        labels.append(label)

            # Force at least one exception branch for every node.
            if len(labels) == 0:
                labels = list(fallback_labels)

            updated["verification"] = True
            updated["branches"] = [
                {
                    "label": label,
                    "next": f"B_{node_id}_{k+1}",
                }
                for k, label in enumerate(labels)
            ]

            if getattr(self, "validate_json", False):
                self._validate_plan_object(updated, kind="node")

            module3_nodes.append(updated)

        return module3_nodes

    # -------------------------------------------------------------------------
    # Module 4 (branch-level)
    # -------------------------------------------------------------------------
    def module4_generate_subplan_for_branch(
        self,
        parent_node: Dict[str, Any],
        branch: Dict[str, Any],
        agent_catalog: Union[List[Any], Dict[str, Any]],
        additional_agent_catalog: Optional[Union[List[Any], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a minimal branch fragment for one branch.

        Minimal output schema:
        {
        "id": "B1",
        "nodes": [
            {
            "id": "B1_S1",
            "task": "...",
            "agent": "...",
            "deps": ["V_S1"]
            }
        ]
        }

        Notes:
        - This is a branch fragment, not a detached answer-producing sub-plan.
        - The fragment may assume that a verifier node V_<parent_id> will be inserted later.
        - The verifier will return runtime JSON:
            {"label": "<string>", "rationale": "<free-text explanation>"}
        - The branch fragment may use that runtime rationale as diagnostic context.
        - Reconnection to the original downstream continuation is handled later by merge/composition.
        """
        parent_id = parent_node["id"]
        branch_id = branch["next"]
        branch_label = branch["label"]
        verifier_id = f"V_{parent_id}"

        prompt = f"""
    You are Module 4 of a staged planner.

    Your job is to synthesize a minimal executable branch fragment for ONE branch of ONE parent node.

    This branch fragment must handle the exception label of the branch and must be written so that,
    at execution time, it may use the verifier's runtime rationale as diagnostic context.

    Important semantics:
    - The parent node is "{parent_id}".
    - A later composition step may insert a verifier node with id "{verifier_id}" after the parent node.
    - That verifier will return runtime JSON:
    {{
    "label": "<string>",
    "rationale": "<free-text explanation>"
    }}
    - Your fragment should be compatible with that setup.
    - Do NOT generate a detached final answer for the branch.
    - Do NOT specify how to reconnect to the original downstream continuation.
    That is handled later by merge/composition.
    - Do NOT generate verifier nodes here.
    - You may use recovery-style local correction or replanning-style repair logic if needed,
    but your output must always be a normal minimal branch fragment object.

    Inputs:
    - parent_node:
    {json.dumps(parent_node, ensure_ascii=False, indent=2)}

    - branch:
    {json.dumps(branch, ensure_ascii=False, indent=2)}

    - agent_catalog:
    {json.dumps(agent_catalog, ensure_ascii=False, indent=2, default=str)}

    - additional_agent_catalog:
    {json.dumps(additional_agent_catalog, ensure_ascii=False, indent=2, default=str)}

    Core objective:
    - The branch fragment must not stop at diagnosis or at proposing a repair action.
    - The branch fragment must include the actual execution of a repair / retry / substitute action.
    - The terminal node of the branch fragment must produce a concrete output that can serve as a usable substitute for the failed parent node's intended output.
    - In other words, the branch fragment should leave the workflow in a state from which the original downstream continuation can consume the repaired result.

    Hard constraints:
    - The branch id must be "{branch_id}".
    - The branch label is "{branch_label}".
    - The fragment must specifically address the exception label "{branch_label}".
    - The fragment may instruct its first node to use the verifier rationale from "{verifier_id}" as diagnostic context.
    - Node ids inside the branch must avoid collision with the base plan.
    - A naming convention like "{branch_id}_S1", "{branch_id}_S2", "{branch_id}_S3", ... is preferred.
    - Return a minimal schema:
    - top-level: "id", "nodes"
    - each node: "id", "task", "agent", "deps"
    - "node_contract" is optional. Include it only if it materially helps.
    - The fragment should usually contain three stages when appropriate:
    1. diagnose the failure,
    2. repair or choose a concrete corrective action,
    3. re-execute the parent capability (or an equivalent substitute action) so that a downstream node can consume the result.
    - If diagnosis is unnecessary, you may omit it, but you must still include an actual corrective execution step.
    - A branch fragment is invalid if it only explains what to do, recommends an action, or creates a work order without also producing a repaired executable result that substitutes for the failed parent output.
    - The last node of the fragment should normally be an execution / retry / substitute node, not a diagnosis-only node and not a recommendation-only node.
    - The output of the last node should be written so that, after later merge/composition, downstream nodes that originally depended on "{parent_id}" can consume the repaired result together with their other existing dependencies.
    - Therefore, the terminal node should try to re-materialize the parent node's intended contract or a downstream-usable equivalent of it.

    Additional guidance:
    - Prefer branch fragments whose final node directly performs the missing or corrected action.
    - Prefer using the same agent as the parent node for the final re-execution step when feasible.
    - Use a different expert/repair agent only when it clearly helps diagnosis or repair before the final execution step.
    - If the failure is about grounding, missing extraction, contract mismatch, or execution error, the branch fragment should typically end by re-running the failed parent task (or a corrected equivalent) rather than merely analyzing it.
    - If the failure requires an intermediate repair node, make that repair node feed into a final execution node.

    Suggested shape:
    - First node: read verifier rationale and diagnose the failure.
    - Second node: repair parameters, entity choice, tool usage, or file handling as needed.
    - Final node: actually execute the corrected retrieval / extraction / write / transformation step and produce a downstream-usable result.

    Output JSON only.
    Do not output Markdown.
    Do not output code fences.
    Do not output explanations before or after the JSON.

    Return exactly one JSON object in the following minimal format:
    {{
    "id": "{branch_id}",
    "nodes": [
        {{
        "id": "{branch_id}_S1",
        "task": "Use the verifier rationale from {verifier_id} if needed, diagnose the {branch_label} failure.",
        "agent": "<one agent name from the catalog>",
        "deps": ["{verifier_id}"]
        }},
        {{
        "id": "{branch_id}_S2",
        "task": "Repair the {branch_label} failure with a concrete corrective action.",
        "agent": "<one agent name from the catalog>",
        "deps": ["{branch_id}_S1"]
        }},
        {{
        "id": "{branch_id}_S3",
        "task": "Actually re-execute the failed capability of parent node {parent_id} (or an equivalent substitute action) so that downstream nodes can consume a repaired result.",
        "agent": "<one agent name from the catalog>",
        "deps": ["{branch_id}_S2"]
        }}
    ]
    }}
        """.strip()

        allowed_agent_names = self._extract_agent_names(
            agent_catalog,
            additional_agent_catalog,
        )

        result = self._call_llm_json_with_retry(
            base_prompt=prompt,
            kind="branch",
            validate_kind="branch",
            validate_kwargs={
                "allowed_agent_names": allowed_agent_names,
            },
        )

        return result


    def module4_generate_subplans_for_plan(
        self,
        module3_nodes: List[Dict[str, Any]],
        agent_catalog: Union[List[Any], Dict[str, Any]],
        additional_agent_catalog: Optional[Union[List[Any], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan all Module 3-updated nodes, find minimal branch skeletons, and synthesize
        minimal branch fragments.

        Returns:
            List[Dict[str, Any]]: list of branch fragment objects
        """
        subplans: List[Dict[str, Any]] = []

        for node in module3_nodes:
            verification = bool(node.get("verification", False))
            branches = node.get("branches", [])

            if not verification or not branches:
                continue

            for branch in branches:
                # subplan = self.module4_generate_subplan_for_branch(
                #     parent_node=node,
                #     branch=branch,
                #     agent_catalog=agent_catalog,
                #     additional_agent_catalog=additional_agent_catalog,
                # )
                # if self.validate_json:
                #     self._validate_plan_object(subplan, kind="branch")

                for attempt in range(3):
                    try:
                        subplan = self.module4_generate_subplan_for_branch(
                            parent_node=node,
                            branch=branch,
                            agent_catalog=agent_catalog,
                            additional_agent_catalog=additional_agent_catalog,
                        )
                        if self.validate_json:
                            self._validate_plan_object(subplan, kind="branch")
                        break
                    except Exception as e:
                        last_error = e
                        if attempt == 2:
                            raise
                else:
                    raise last_error
                
                subplans.append(subplan)

        return subplans

    def module_attach_verification_flags(
        self,
        task_description: str,
        base_plan: Dict[str, Any],
        constraints: Optional[Union[List[Any], Dict[str, Any], str]] = None,
    ) -> Dict[str, Any]:
        prompt = f"""
    You are a planner module that adds verification=true/false to each node in an existing base plan.

    Your only job:
    - preserve the base_plan exactly
    - add one boolean field "verification" to each node

    Do not add branches.
    Do not add expected_exception.
    Do not modify any existing field.

    Set verification=true only if verifying this node's output would be meaningfully useful before downstream use or before final answer construction.
    Otherwise set verification=false.

    Inputs:
    - task_description:
    {task_description}

    - base_plan:
    {json.dumps(base_plan, ensure_ascii=False, indent=2)}

    - constraints:
    {json.dumps(constraints, ensure_ascii=False, indent=2, default=str)}

    Output JSON only.
    Return exactly the same base_plan structure, with only "verification" added to each node.
        """.strip()

        return self._call_llm_json_with_retry(
            base_prompt=prompt,
            kind="plan",
            validate_kind="plan_with_verification",
        )
    
    # -------------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------------
    def generate_plan_object(
        self,
        task_description: str,
        agent_catalog: Union[List[Any], Dict[str, Any]],
        step_type_library: Union[List[Any], Dict[str, Any]],
        output_format: str = "json",
        constraints: Optional[Union[List[Any], Dict[str, Any], str]] = None,
        exception_taxonomy: Optional[Union[List[Any], Dict[str, Any]]] = None,
        agent_env_specs_json: Optional[Union[List[Any], Dict[str, Any], str]] = None,
        empirical_failure_profile: Optional[Union[List[Any], Dict[str, Any], str]] = None,
        additional_agent_catalog: Optional[Union[List[Any], Dict[str, Any]]] = None,
        task_type_failure_priors: Optional[Union[Dict[str, Any], str]] = None,
        mode: str = "force_verify",
        qid: int = None,
    ) -> Dict[str, Any]:
        """
        Run Module 1 -> Module 2 -> Module 3 -> Module 4 and merge everything.

        Final output shape:
        {
          "answer_contract": "...",
          "nodes": [...],                # updated nodes after Module 3
          "branch_subplans": {
            "B1": {...},
            "B2": {...}
          }
        }
        """
        if exception_taxonomy is None:
            exception_taxonomy = []
        if agent_env_specs_json is None:
            agent_env_specs_json = {}
        if empirical_failure_profile is None:
            empirical_failure_profile = {}

        if mode.startswith("oracle_verify_recovery"):
            rewritten_dir = Path(
                f"/home/track1_result/plan/[ReAct_CD][oracle_verify_recovery]Model_16/Rewritten files"
            )

            if qid is None:
                raise ValueError("qid is required for mode=oracle_verify_recovery")

            # まずは filename に Q_{qid} を含むものを探す
            candidates = sorted(rewritten_dir.glob(f"*Q_{qid}_plan*"))

            if candidates:
                chosen_plan_path = candidates[0]
                rewritten_plan = _load_first_valid_json_dict(chosen_plan_path)
                print(
                    f"Loaded rewritten oracle_verify_recovery plan from {chosen_plan_path}: {rewritten_plan}",
                    flush=True
                )
                return rewritten_plan

            raise FileNotFoundError(
                f"No rewritten plan found for qid={qid} in {rewritten_dir}"
            )

        if mode.startswith("no_verify"):
            plan_path = Path(
                f"/home/track1_result/plan/[ReAct_CD][no_verify]Model_16/Model_16_Q_{qid}_plan.txt"
            )

            if plan_path.exists():
                text = plan_path.read_text(encoding="utf-8")
                base_plan_text = self._extract_json_text(text, kind="plan")
                base_plan = json.loads(base_plan_text)
                print(f"Loaded existing no_verify plan from {plan_path}: {base_plan}", flush=True)
                return base_plan

            base_plan = self.module1_generate_base_plan(
                task_description=task_description,
                agent_catalog=agent_catalog,
                step_type_library=step_type_library,
                output_format=output_format,
                constraints=constraints,
            )
            print(f"final_plan(no_verify): {base_plan}", flush=True)
            return base_plan
        
        if mode.startswith("allocation_only"):
            if qid is not None:
                plan_path = Path(
                    f"/home/track1_result/plan/[ReAct_CD][annotated_only_w_few_shot_1]Model_16/Model_16_Q_{qid}_plan.txt"
                )

                if not plan_path.exists():
                    raise FileNotFoundError(f"Plan file not found: {plan_path}")

                text = plan_path.read_text(encoding="utf-8")
                decoder = json.JSONDecoder()
                annotated_plan = None

                for i, ch in enumerate(text):
                    if ch not in "{[":
                        continue
                    try:
                        obj, _ = decoder.raw_decode(text[i:])
                        if isinstance(obj, dict):
                            annotated_plan = obj
                            break
                    except json.JSONDecodeError:
                        continue

                if annotated_plan is None:
                    raise ValueError(f"Could not find a valid JSON object in: {plan_path}")
        else:
            if qid is not None:
                plan_path = Path(
                    f"/home/track1_result/plan/[ReAct_CD][no_verify]Model_16/Model_16_Q_{qid}_plan.txt"
                )

                if not plan_path.exists():
                    raise FileNotFoundError(f"Plan file not found: {plan_path}")

                text = plan_path.read_text(encoding="utf-8")
                decoder = json.JSONDecoder()
                base_plan = None

                for i, ch in enumerate(text):
                    if ch not in "{[":
                        continue
                    try:
                        obj, _ = decoder.raw_decode(text[i:])
                        if isinstance(obj, dict):
                            base_plan = obj
                            break
                    except json.JSONDecodeError:
                        continue

                if base_plan is None:
                    raise ValueError(f"Could not find a valid JSON object in: {plan_path}")

            print(f"base_plan: {base_plan}", flush=True)

            annotated_plan = self.module2_annotate_expected_exceptions(
                base_plan=base_plan,
                exception_taxonomy=exception_taxonomy,
                task_type_failure_priors=task_type_failure_priors,
            )

        if mode.startswith("annotated_only"):
            return annotated_plan

        print(f"annotated_plan: {annotated_plan}", flush=True)

        if mode.startswith("adaptive"):
            module3_nodes = self.module3_generate_verifications_for_plan(
                annotated_plan=annotated_plan,
                agent_env_specs_json=agent_env_specs_json,
                empirical_failure_profile=empirical_failure_profile,
            )
        elif mode.startswith("force_verify"):
            module3_nodes = self.module3_force_verifications_for_plan(
                annotated_plan=annotated_plan,
                fallback_exception_labels=["execution"],
            )
        elif mode.startswith("allocation_only"):
            module3_nodes = self.module3_generate_verifications_for_plan(
                annotated_plan=annotated_plan,
                agent_env_specs_json=agent_env_specs_json,
                empirical_failure_profile=empirical_failure_profile,
            )
            return module3_nodes
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"module3_nodes: {module3_nodes}", flush=True)

        module4_subplans = self.module4_generate_subplans_for_plan(
            module3_nodes=module3_nodes,
            agent_catalog=agent_catalog,
            additional_agent_catalog=additional_agent_catalog,
        )

        print(f"module4_subplans: {module4_subplans}", flush=True)

        final_plan = self._merge_module_outputs(
            annotated_plan=annotated_plan,
            module3_nodes=module3_nodes,
            module4_subplans=module4_subplans,
        )

        print(f"final_plan: {final_plan}", flush=True)

        # import time
        # # time.sleep(10)

        if self.validate_json:
            self._validate_plan_object(final_plan, kind="final_plan")

        return final_plan
        # return base_plan

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    def _call_llm_json(self, prompt: str, kind: str = "plan") -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        last_text: str = ""

        for attempt in range(1, 4):
            text = self.llm_generate(prompt)
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)

            last_text = text
            print(f"[planning] kind={kind} attempt={attempt} raw repr={text!r}", flush=True)

            if not text.strip():
                last_error = ValueError(
                    f"Empty LLM output for kind={kind} on attempt={attempt}"
                )
                continue

            try:
                json_text = self._extract_json_text(text, kind=kind)
                obj = json.loads(json_text)
                if not isinstance(obj, dict):
                    raise ValueError(f"Parsed JSON is not an object for kind={kind}")
                return obj
            except Exception as e:
                last_error = e
                continue

        # one final stricter retry for plan generation
        if kind == "plan":
            repair_prompt = (
                prompt
                + "\n\nYour previous response was invalid or empty.\n"
                + "Return EXACTLY ONE JSON object.\n"
                + "Do not return prose.\n"
                + "Do not return markdown.\n"
                + "Do not return code fences.\n"
                + 'The output must start with "{" and end with "}".'
            )

            text = self.llm_generate(repair_prompt)
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)

            print(f"[planning] kind={kind} repair raw repr={text!r}", flush=True)

            if text.strip():
                try:
                    json_text = self._extract_json_text(text, kind=kind)
                    obj = json.loads(json_text)
                    if isinstance(obj, dict):
                        return obj
                except Exception as e:
                    last_error = e
                    last_text = text
            else:
                last_text = text

        raise ValueError(
            f"LLM output does not contain a JSON object for kind={kind}.\n"
            f"Raw output repr:\n{last_text!r}\n"
            f"Last error:\n{last_error}"
        )
    
    def _build_retry_prompt_with_error(
        self,
        base_prompt: str,
        error: Exception,
        attempt_index: int,
        max_error_chars: int = 1200,
    ) -> str:
        error_text = " ".join(str(error).split())
        if len(error_text) > max_error_chars:
            error_text = error_text[:max_error_chars] + " ...[truncated]"

        retry_note = f"""

    Previous attempt failed.

    Retry attempt: {attempt_index + 1}

    You must correct the output so that it satisfies all required constraints, schema rules, and validation checks.

    Error from the previous attempt:
    {error_text}

    Retry instructions:
    - Fix the specific problem indicated by the error above.
    - If the error mentions a missing key, ensure that key is present.
    - If the error mentions a schema mismatch, conform exactly to the requested schema.
    - If the error mentions invalid JSON, repair the JSON formatting first.
    - Preserve the required top-level structure exactly.
    - Preserve all required keys and field names exactly.
    - Return only valid JSON.
    - Do not output Markdown.
    - Do not output code fences.
    - Do not output explanations before or after the JSON.
    """
        return base_prompt + retry_note


    def _call_llm_json_with_retry(
        self,
        *,
        base_prompt: str,
        kind: str,
        validate_kind: Optional[str] = None,
        validate_kwargs: Optional[dict] = None,
        max_attempts: int = 3,
    ):
        validate_kind = validate_kind or kind
        validate_kwargs = validate_kwargs or {}

        last_error = None

        for attempt in range(max_attempts):
            prompt_for_attempt = (
                base_prompt
                if attempt == 0
                else self._build_retry_prompt_with_error(
                    base_prompt=base_prompt,
                    error=last_error,
                    attempt_index=attempt,
                )
            )

            try:
                result = self._call_llm_json(prompt_for_attempt, kind=kind)
                if self.validate_json:
                    self._validate_plan_object(
                        result,
                        kind=validate_kind,
                        **validate_kwargs,
                    )
                return result
            except Exception as e:
                last_error = e
                if attempt == max_attempts - 1:
                    raise

        raise last_error
    
    def _review_module1_plan(
        self,
        task_description: str,
        agent_catalog: Union[List[Any], Dict[str, Any]],
        plan_obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        reviewer = PlanReviewerAgent(llm=self.model_name)

        agent_lines: List[str] = []
        if isinstance(agent_catalog, dict):
            catalog_items = list(agent_catalog.values())
        else:
            catalog_items = list(agent_catalog)

        for ii, agent in enumerate(catalog_items, start=1):
            if isinstance(agent, dict):
                name = agent.get("name", "")
                description = agent.get("description", "")
                task_examples = agent.get("task_examples", []) or []
            else:
                name = getattr(agent, "name", "")
                description = getattr(agent, "description", "")
                task_examples = getattr(agent, "task_examples", []) or []

            agent_lines.append(f"({ii}) Agent name: {name}")
            agent_lines.append(f"Agent description: {description}")
            if task_examples:
                agent_lines.append("Tasks that agent can solve:")
                for idx, task_example in enumerate(task_examples, start=1):
                    agent_lines.append(f"{idx}. {task_example}")
            agent_lines.append("")

        review, in_tok, out_tok = reviewer.execute_task(
            question=task_description,
            agent_descriptions="\n".join(agent_lines).strip(),
            plan=json.dumps(plan_obj, ensure_ascii=False, indent=2),
        )

        if not isinstance(review, dict):
            raise ValueError("Plan reviewer must return a dict.")

        logger.info(
            "Module 1 review tokens: input=%s output=%s",
            in_tok,
            out_tok,
        )

        return review

    def _extract_json_text(self, text: str, kind: str) -> str:
        text = text.strip()

        if not text:
            raise ValueError(f"Empty text for kind={kind}")

        # Case 1: whole response is already a JSON object
        if text.startswith("{") and text.endswith("}"):
            return text

        # Case 2: fenced JSON block
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        # Case 3: find balanced {...} candidates and return the last valid dict-shaped JSON
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

        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return candidate
            except Exception:
                pass

        raise ValueError(f"No JSON object found for kind={kind}")

    def _canonicalize_branch_ids(self, node_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rewrite branch next ids to be globally unique per parent node:
        B_<parent_id>_1, B_<parent_id>_2, ...
        """
        if not isinstance(node_obj, dict):
            return node_obj

        out = dict(node_obj)
        parent_id = str(out.get("id", "")).strip()
        branches = out.get("branches", [])

        if not isinstance(branches, list) or not parent_id:
            return out

        new_branches = []
        for i, branch in enumerate(branches, start=1):
            if not isinstance(branch, dict):
                new_branches.append(branch)
                continue

            b = dict(branch)
            b["next"] = f"B_{parent_id}_{i}"
            new_branches.append(b)

        out["branches"] = new_branches
        return out

    def _validate_plan_object(
        self,
        obj: Dict[str, Any],
        kind: str,
        allowed_agent_names: Optional[Set[str]] = None,
    ) -> None:
        """
        Lightweight structural validation with optional agent-name validation.

        kind:
        - "plan"
        - "node"
        - "branch"
        - "final_plan"

        Minimal-schema policy:
        - node.branches require only: {"next", "label"}
        - branch requires only: {"id", "nodes"}
        - node_contract is optional (but if present, must be a string)
        - final_plan is a flattened single-DAG object:
            {"answer_contract": "...", "nodes": [...]}

        If allowed_agent_names is provided, every node.agent must be a member of it.
        """

        if not isinstance(obj, dict):
            raise TypeError(f"{kind} must be a dict, got {type(obj)}")

        if allowed_agent_names is not None and not isinstance(allowed_agent_names, set):
            allowed_agent_names = set(allowed_agent_names)

        if kind == "plan":
            required = ["answer_contract", "nodes"]
            for key in required:
                if key not in obj:
                    raise ValueError(f"plan missing required key: {key}")

            if not isinstance(obj["answer_contract"], str):
                raise TypeError("plan.answer_contract must be a string")
            if not isinstance(obj["nodes"], list):
                raise TypeError("plan.nodes must be a list")

            node_ids = set()
            for node in obj["nodes"]:
                self._validate_plan_object(
                    node,
                    kind="node",
                    allowed_agent_names=allowed_agent_names,
                )
                node_id = node["id"]
                if node_id in node_ids:
                    raise ValueError(f"Duplicate node id in plan: {node_id}")
                node_ids.add(node_id)
            return

        if kind == "node":
            required = ["id", "task", "agent", "deps"]
            for key in required:
                if key not in obj:
                    raise ValueError(f"node missing required key: {key}")

            if not isinstance(obj["id"], str):
                raise TypeError("node.id must be a string")
            if not isinstance(obj["task"], str):
                raise TypeError("node.task must be a string")
            if not isinstance(obj["agent"], str):
                raise TypeError("node.agent must be a string")
            if not isinstance(obj["deps"], list):
                raise TypeError("node.deps must be a list")

            agent_name = obj["agent"].strip()
            if not agent_name:
                raise ValueError("node.agent must be a non-empty string")

            if allowed_agent_names is not None and agent_name not in allowed_agent_names:
                allowed_sorted = sorted(allowed_agent_names)
                raise ValueError(
                    f"node.agent '{agent_name}' is not in the allowed agent catalog. "
                    f"Allowed agents: {allowed_sorted}"
                )

            if "node_contract" in obj and not isinstance(obj["node_contract"], str):
                raise TypeError("node.node_contract must be a string if present")

            if "expected_exception" in obj:
                if not isinstance(obj["expected_exception"], list):
                    raise TypeError("node.expected_exception must be a list")
                for exc in obj["expected_exception"]:
                    if not isinstance(exc, dict):
                        raise TypeError("each expected_exception item must be a dict")
                    if "label" not in exc or "signals" not in exc:
                        raise ValueError("each expected_exception item must have label and signals")
                    if not isinstance(exc["label"], str):
                        raise TypeError("expected_exception.label must be a string")
                    if not isinstance(exc["signals"], list):
                        raise TypeError("expected_exception.signals must be a list")
                    for s in exc["signals"]:
                        if not isinstance(s, str):
                            raise TypeError("each expected_exception.signal must be a string")

            if "verification" in obj:
                if not isinstance(obj["verification"], bool):
                    raise TypeError("node.verification must be a bool")

            if "branches" in obj:
                if not isinstance(obj["branches"], list):
                    raise TypeError("node.branches must be a list")

                seen_labels = set()
                seen_next_ids = set()

                for branch in obj["branches"]:
                    if not isinstance(branch, dict):
                        raise TypeError("each branch must be a dict")

                    for key in ["next", "label"]:
                        if key not in branch:
                            raise ValueError(f"branch missing required key: {key}")

                    if not isinstance(branch["next"], str):
                        raise TypeError("branch.next must be a string")
                    if not isinstance(branch["label"], str):
                        raise TypeError("branch.label must be a string")

                    if "signals" in branch:
                        if not isinstance(branch["signals"], list):
                            raise TypeError("branch.signals must be a list if present")
                        for s in branch["signals"]:
                            if not isinstance(s, str):
                                raise TypeError("each branch.signal must be a string")

                    if branch["next"] in seen_next_ids and branch["next"] != "TERMINATE":
                        raise ValueError(f"Duplicate branch.next within node {obj['id']}: {branch['next']}")
                    seen_next_ids.add(branch["next"])

                    if self.strict_branch_label_uniqueness:
                        if branch["label"] in seen_labels:
                            raise ValueError(
                                f"Duplicate branch.label within node {obj['id']}: {branch['label']}"
                            )
                        seen_labels.add(branch["label"])
            return

        if kind == "branch":
            required = ["id", "nodes"]
            for key in required:
                if key not in obj:
                    raise ValueError(f"branch missing required key: {key}")

            if not isinstance(obj["id"], str):
                raise TypeError("branch.id must be a string")
            if not isinstance(obj["nodes"], list):
                raise TypeError("branch.nodes must be a list")

            node_ids = set()
            for node in obj["nodes"]:
                self._validate_plan_object(
                    node,
                    kind="node",
                    allowed_agent_names=allowed_agent_names,
                )
                node_id = node["id"]
                if node_id in node_ids:
                    raise ValueError(f"Duplicate node id inside branch {obj['id']}: {node_id}")
                node_ids.add(node_id)
            return

        if kind == "final_plan":
            required = ["answer_contract", "nodes"]
            for key in required:
                if key not in obj:
                    raise ValueError(f"final_plan missing required key: {key}")

            if not isinstance(obj["answer_contract"], str):
                raise TypeError("final_plan.answer_contract must be a string")
            if not isinstance(obj["nodes"], list):
                raise TypeError("final_plan.nodes must be a list")

            node_ids = set()
            for node in obj["nodes"]:
                self._validate_plan_object(
                    node,
                    kind="node",
                    allowed_agent_names=allowed_agent_names,
                )
                node_id = node["id"]
                if node_id in node_ids:
                    raise ValueError(f"Duplicate node id in final_plan: {node_id}")
                node_ids.add(node_id)

            for node in obj["nodes"]:
                for branch in node.get("branches", []):
                    next_id = branch["next"]
                    if next_id != "TERMINATE" and next_id not in node_ids:
                        raise ValueError(
                            f"Node '{node['id']}' has branch target '{next_id}' "
                            f"which is not present in final_plan.nodes."
                        )

            return

        raise ValueError(f"Unknown validation kind: {kind}")

    def _merge_module_outputs(
        self,
        annotated_plan: Dict[str, Any],
        module3_nodes: List[Dict[str, Any]],
        module4_subplans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge:
        - plan-level output from Module 2
        - node-level outputs from Module 3
        - branch-level outputs from Module 4

        Final shape (flattened single DAG):
        {
        "answer_contract": "...",
        "nodes": [...]
        }

        Semantics:
        - If verification=true on node S, insert verifier node V_S.
        - Verifier runtime output is assumed to be:
            {"label": "<string>", "rationale": "<free-text explanation>"}
        - Verifier control labels are interpreted as:
            success   -> original normal continuation
            <label>   -> matching branch fragment entry
            terminate -> TERMINATE
        - Branch fragments are flattened into the final DAG.
        - Sink nodes of each branch fragment are rewired back to the original
        normal continuation, with an explicit terminate path as well.
        - This version intentionally allows both:
            success   -> TERMINATE
            terminate -> TERMINATE
        so validator must allow duplicate branch.next == "TERMINATE".
        """

        def _deepcopy_node(n: Dict[str, Any]) -> Dict[str, Any]:
            return copy.deepcopy(n)

        def _get_entry_node_ids(fragment: Dict[str, Any]) -> List[str]:
            nodes = fragment.get("nodes", [])
            internal_ids = {n["id"] for n in nodes}
            entry_ids: List[str] = []

            for n in nodes:
                deps = n.get("deps", [])
                internal_deps = [d for d in deps if d in internal_ids]
                if len(internal_deps) == 0:
                    entry_ids.append(n["id"])
            return entry_ids

        def _get_sink_node_ids(fragment: Dict[str, Any]) -> List[str]:
            nodes = fragment.get("nodes", [])
            internal_ids = {n["id"] for n in nodes}
            used_as_internal_dep = set()

            for n in nodes:
                for d in n.get("deps", []):
                    if d in internal_ids:
                        used_as_internal_dep.add(d)

            return [n["id"] for n in nodes if n["id"] not in used_as_internal_dep]

        def _replace_dep_with_many(node_obj: Dict[str, Any], old_dep: str, new_deps: List[str]) -> None:
            deps = node_obj.get("deps", [])
            replaced: List[str] = []

            for d in deps:
                if d == old_dep:
                    replaced.extend(new_deps)
                else:
                    replaced.append(d)

            seen = set()
            deduped = []
            for d in replaced:
                if d not in seen:
                    deduped.append(d)
                    seen.add(d)

            node_obj["deps"] = deduped

        # ------------------------------------------------------------------
        # 1) Replace base nodes by id using Module 3 outputs
        # ------------------------------------------------------------------
        base_plan = copy.deepcopy(annotated_plan)
        module3_map = {node["id"]: node for node in module3_nodes}

        merged_base_nodes: List[Dict[str, Any]] = []
        for old_node in base_plan.get("nodes", []):
            node_id = old_node["id"]
            merged_base_nodes.append(_deepcopy_node(module3_map.get(node_id, old_node)))

        # ------------------------------------------------------------------
        # 2) Validate and index Module 4 branch fragments
        # ------------------------------------------------------------------
        branch_map: Dict[str, Dict[str, Any]] = {}
        for subplan in module4_subplans:
            branch_id = subplan["id"]
            if branch_id in branch_map:
                raise ValueError(f"Duplicate branch subplan id during merge: {branch_id}")
            branch_map[branch_id] = _deepcopy_node(subplan)

        # ------------------------------------------------------------------
        # 3) Build immediate downstream map from merged base DAG
        #    downstream[parent_id] = [child node ids that directly depend on parent_id]
        # ------------------------------------------------------------------
        downstream_map: Dict[str, List[str]] = {}
        for node in merged_base_nodes:
            for dep in node.get("deps", []):
                downstream_map.setdefault(dep, []).append(node["id"])

        # ------------------------------------------------------------------
        # 4) Start flattened final plan
        # ------------------------------------------------------------------
        final_plan: Dict[str, Any] = {
            "answer_contract": base_plan["answer_contract"],
            "nodes": [],
        }

        final_nodes: List[Dict[str, Any]] = []
        final_node_ids = set()

        def _append_node(node_obj: Dict[str, Any]) -> None:
            node_id = node_obj["id"]
            if node_id in final_node_ids:
                raise ValueError(f"Duplicate node id during merge: {node_id}")
            final_nodes.append(node_obj)
            final_node_ids.add(node_id)

        # ------------------------------------------------------------------
        # 5) Preserve original Module 3 branches BEFORE stripping them from the
        #    execution nodes. This is the key fix.
        # ------------------------------------------------------------------
        branch_source_map: Dict[str, List[Dict[str, Any]]] = {
            node["id"]: copy.deepcopy(node.get("branches", []))
            for node in merged_base_nodes
        }

        rewritten_base_nodes: Dict[str, Dict[str, Any]] = {}
        for node in merged_base_nodes:
            node_copy = _deepcopy_node(node)
            if node_copy.get("verification", False):
                # Original execution node should no longer branch directly.
                node_copy["branches"] = []
            rewritten_base_nodes[node_copy["id"]] = node_copy

        # ------------------------------------------------------------------
        # 6) Insert verifier / join nodes and flatten branch fragments
        # ------------------------------------------------------------------
        synthetic_nodes_to_append: List[Dict[str, Any]] = []
        branch_fragment_nodes_to_append: List[Dict[str, Any]] = []

        for parent_id, parent_node in rewritten_base_nodes.items():
            if not bool(parent_node.get("verification", False)):
                continue

            verifier_id = f"V_{parent_id}"
            original_children = list(downstream_map.get(parent_id, []))

            # ------------------------------------------------------------
            # A) determine normal continuation target and gate dependency
            # ------------------------------------------------------------
            if len(original_children) == 0:
                normal_next = "TERMINATE"
                gate_dep = verifier_id
            elif len(original_children) == 1:
                normal_next = original_children[0]
                gate_dep = verifier_id
            else:
                join_id = f"J_{parent_id}"
                join_node = {
                    "id": join_id,
                    "task": f"Rejoin normal continuation after verification/repair of {parent_id}.",
                    "agent": "system",
                    "deps": [verifier_id],
                    "branches": [],
                }
                synthetic_nodes_to_append.append(join_node)
                normal_next = join_id
                gate_dep = join_id

            verifier_branches: List[Dict[str, Any]] = []
            verifier_branches.append({
                "label": "success",
                "next": normal_next,
            })

            parent_branches = branch_source_map.get(parent_id, [])
            all_branch_sink_ids: List[str] = []

            # ------------------------------------------------------------
            # B) flatten each fragment, record its sink ids
            # ------------------------------------------------------------
            for branch in parent_branches:
                branch_label = branch["label"]
                branch_id = branch["next"]

                if branch_id not in branch_map:
                    raise ValueError(
                        f"Parent node '{parent_id}' refers to missing branch fragment '{branch_id}'."
                    )

                fragment = _deepcopy_node(branch_map[branch_id])
                fragment_nodes = fragment.get("nodes", [])
                if not fragment_nodes:
                    raise ValueError(f"Branch fragment '{branch_id}' has no nodes.")

                entry_ids = _get_entry_node_ids(fragment)
                if len(entry_ids) != 1:
                    raise ValueError(
                        f"Branch fragment '{branch_id}' must have exactly one entry node, but got {entry_ids}."
                    )
                entry_id = entry_ids[0]

                sink_ids = _get_sink_node_ids(fragment)
                if len(sink_ids) == 0:
                    raise ValueError(f"Branch fragment '{branch_id}' has no sink node.")

                for sink_id in sink_ids:
                    if sink_id not in all_branch_sink_ids:
                        all_branch_sink_ids.append(sink_id)

                internal_ids = {n["id"] for n in fragment_nodes}

                # normalize fragment deps:
                # - entry node: depends only on verifier_id
                # - non-entry node: depends only on internal fragment deps
                for frag_node in fragment_nodes:
                    deps = frag_node.get("deps", [])
                    internal_deps = [d for d in deps if d in internal_ids]

                    if len(internal_deps) == 0:
                        frag_node["deps"] = [verifier_id]
                    else:
                        frag_node["deps"] = internal_deps

                for frag_node in fragment_nodes:
                    bad_external_deps = [
                        d for d in frag_node.get("deps", [])
                        if d != verifier_id and d not in internal_ids
                    ]
                    if bad_external_deps:
                        raise ValueError(
                            f"Branch fragment '{branch_id}' node '{frag_node['id']}' "
                            f"still has external deps after normalization: {bad_external_deps}"
                        )

                # sink nodes branch back to normal continuation
                for frag_node in fragment_nodes:
                    if frag_node["id"] in sink_ids:
                        frag_node["branches"] = [
                            {"label": "success", "next": normal_next},
                            {"label": "terminate", "next": "TERMINATE"},
                        ]

                verifier_branches.append({
                    "label": branch_label,
                    "next": entry_id,
                })

                for frag_node in fragment_nodes:
                    branch_fragment_nodes_to_append.append(frag_node)

            # ------------------------------------------------------------
            # C) now rewrite original children deps
            # ------------------------------------------------------------
            for child_id in original_children:
                child_node = rewritten_base_nodes[child_id]

                replacement_deps = [gate_dep, parent_id, *all_branch_sink_ids]
                _replace_dep_with_many(child_node, parent_id, replacement_deps)

            # ------------------------------------------------------------
            # D) verifier node
            # ------------------------------------------------------------
            verifier_branches.append({
                "label": "terminate",
                "next": "TERMINATE",
            })

            verifier_node = {
                "id": verifier_id,
                "task": (
                    f"Verify whether the output of {parent_id} is sufficient for normal continuation. "
                    f"Return a runtime JSON object with fields 'label' and 'rationale'."
                ),
                "agent": "verifier",
                "deps": [parent_id],
                "branches": verifier_branches,
            }
            synthetic_nodes_to_append.append(verifier_node)

        # ------------------------------------------------------------------
        # 7) Append everything into the final flattened node list
        # ------------------------------------------------------------------
        for node_id in [n["id"] for n in merged_base_nodes]:
            _append_node(rewritten_base_nodes[node_id])

        for node_obj in synthetic_nodes_to_append:
            _append_node(node_obj)

        for node_obj in branch_fragment_nodes_to_append:
            _append_node(node_obj)

        final_plan["nodes"] = final_nodes
        return final_plan
    
    def _extract_agent_names(
        self,
        agent_catalog: Optional[Union[List[Any], Dict[str, Any]]] = None,
        additional_agent_catalog: Optional[Union[List[Any], Dict[str, Any]]] = None,
    ) -> Set[str]:
        names: Set[str] = set()

        def _add_from_catalog(catalog):
            if catalog is None:
                return
            if isinstance(catalog, dict):
                items = catalog.values()
            else:
                items = catalog

            for item in items:
                if isinstance(item, str):
                    if item.strip():
                        names.add(item.strip())
                elif isinstance(item, dict):
                    name = item.get("name")
                    if isinstance(name, str) and name.strip():
                        names.add(name.strip())
                else:
                    name = getattr(item, "name", None)
                    if isinstance(name, str) and name.strip():
                        names.add(name.strip())

        _add_from_catalog(agent_catalog)
        _add_from_catalog(additional_agent_catalog)
        return names

import json
import unittest
from typing import Any, Dict, List, Union


class FakeLLM:
    """
    A simple fake LLM that returns pre-configured outputs in sequence.
    It also stores prompts for optional inspection.
    """

    def __init__(self, outputs: List[Union[str, Dict[str, Any]]]) -> None:
        self.outputs = list(outputs)
        self.calls: List[str] = []

    def __call__(self, prompt: str) -> Union[str, Dict[str, Any]]:
        self.calls.append(prompt)
        if not self.outputs:
            raise RuntimeError("FakeLLM has no more configured outputs.")
        return self.outputs.pop(0)


class TestPlanningWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_base_plan = {
            "answer_contract": "Return the available IoT site.",
            "nodes": [
                {
                    "id": "S1",
                    "task": "Retrieve the list of available IoT sites.",
                    "agent": "iot",
                    "deps": [],
                    "node_contract": "Return the available site list.",
                }
            ],
        }

        self.sample_annotated_plan = {
            "answer_contract": "Return the available IoT site.",
            "nodes": [
                {
                    "id": "S1",
                    "task": "Retrieve the list of available IoT sites.",
                    "agent": "iot",
                    "deps": [],
                    "node_contract": "Return the available site list.",
                    "expected_exception": [
                        {
                            "label": "execution",
                            "signals": [
                                "tool not called",
                                "wrong tool selected",
                            ],
                        },
                        {
                            "label": "availability",
                            "signals": [
                                "no data loaded",
                                "empty result",
                            ],
                        },
                    ],
                }
            ],
        }

        self.sample_module3_node_with_branches = {
            "id": "S1",
            "task": "Retrieve the list of available IoT sites.",
            "agent": "iot",
            "deps": [],
            "node_contract": "Return the available site list.",
            "expected_exception": [
                {
                    "label": "execution",
                    "signals": [
                        "tool not called",
                        "wrong tool selected",
                    ],
                },
                {
                    "label": "availability",
                    "signals": [
                        "no data loaded",
                        "empty result",
                    ],
                },
            ],
            "verification": True,
            "branches": [
                {
                    "next": "B1",
                    "label": "execution",
                },
                {
                    "next": "B2",
                    "label": "availability",
                },
            ],
        }

        self.sample_module3_node_without_branches = {
            "id": "S1",
            "task": "Retrieve the list of available IoT sites.",
            "agent": "iot",
            "deps": [],
            "node_contract": "Return the available site list.",
            "expected_exception": [],
            "verification": False,
            "branches": [],
        }

        self.sample_branch_fragment_b1 = {
            "id": "B1",
            "nodes": [
                {
                    "id": "B1_S1",
                    "task": "Use the verifier rationale from V_S1 if needed, diagnose the execution failure, and retry correctly.",
                    "agent": "iot",
                    "deps": ["V_S1"],
                }
            ],
        }

        self.sample_branch_fragment_b2 = {
            "id": "B2",
            "nodes": [
                {
                    "id": "B2_S1",
                    "task": "Use the verifier rationale from V_S1 if needed, diagnose the availability failure, and recover a non-empty result.",
                    "agent": "iot",
                    "deps": ["V_S1"],
                }
            ],
        }

    def test_module1_generate_base_plan_success(self) -> None:
        fake_llm = FakeLLM([json.dumps(self.sample_base_plan)])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        result = wf.module1_generate_base_plan(
            task_description="Find the available IoT site.",
            agent_catalog=["iot", "fmsr"],
            step_type_library=["retrieve", "synthesis"],
            output_format="json",
            constraints=None,
        )

        self.assertEqual(result["answer_contract"], "Return the available IoT site.")
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["nodes"][0]["id"], "S1")
        self.assertEqual(result["nodes"][0]["agent"], "iot")

    def test_module2_annotate_expected_exceptions_success(self) -> None:
        fake_llm = FakeLLM([json.dumps(self.sample_annotated_plan)])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        result = wf.module2_annotate_expected_exceptions(
            base_plan=self.sample_base_plan,
            exception_taxonomy=["execution", "availability", "contract"],
        )

        self.assertIn("expected_exception", result["nodes"][0])
        self.assertEqual(len(result["nodes"][0]["expected_exception"]), 2)
        self.assertEqual(result["nodes"][0]["expected_exception"][0]["label"], "execution")

    def test_module3_generate_verification_for_node_with_branches(self) -> None:
        fake_llm = FakeLLM([json.dumps(self.sample_module3_node_with_branches)])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        input_node = self.sample_annotated_plan["nodes"][0]
        result = wf.module3_generate_verification_for_node(
            node=input_node,
            agent_env_specs_json={"autonomy": "high"},
            empirical_failure_profile={"execution": 0.6, "availability": 0.5},
        )

        self.assertTrue(result["verification"])
        self.assertEqual(len(result["branches"]), 2)
        self.assertEqual(result["branches"][0]["label"], "execution")
        self.assertEqual(result["branches"][1]["label"], "availability")
        self.assertNotIn("signals", result["branches"][0])
        self.assertNotIn("signals", result["branches"][1])

    def test_module3_generate_verification_for_node_without_branches(self) -> None:
        fake_llm = FakeLLM([json.dumps(self.sample_module3_node_without_branches)])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        input_node = {
            "id": "S1",
            "task": "Retrieve the list of available IoT sites.",
            "agent": "iot",
            "deps": [],
            "node_contract": "Return the available site list.",
            "expected_exception": [],
        }

        result = wf.module3_generate_verification_for_node(
            node=input_node,
            agent_env_specs_json={"autonomy": "high"},
            empirical_failure_profile={},
        )

        self.assertFalse(result["verification"])
        self.assertEqual(result["branches"], [])

    def test_module4_generate_subplan_for_branch_success(self) -> None:
        fake_llm = FakeLLM([json.dumps(self.sample_branch_fragment_b1)])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        result = wf.module4_generate_subplan_for_branch(
            parent_node=self.sample_module3_node_with_branches,
            branch=self.sample_module3_node_with_branches["branches"][0],
            agent_catalog=["iot", "fmsr"],
            additional_agent_catalog=["recovery", "replanning"],
        )

        self.assertEqual(result["id"], "B1")
        self.assertEqual(result["nodes"][0]["id"], "B1_S1")
        self.assertEqual(result["nodes"][0]["agent"], "iot")
        self.assertEqual(result["nodes"][0]["deps"], ["V_S1"])
        self.assertNotIn("answer_contract", result)

    def test_call_llm_json_accepts_code_fence(self) -> None:
        raw = """```json
{
  "answer_contract": "Return the available IoT site.",
  "nodes": [
    {
      "id": "S1",
      "task": "Retrieve the list of available IoT sites.",
      "agent": "iot",
      "deps": [],
      "node_contract": "Return the available site list."
    }
  ]
}
```"""
        fake_llm = FakeLLM([raw])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        result = wf._call_llm_json("dummy prompt", kind="plan")
        self.assertIn("answer_contract", result)
        self.assertIn("nodes", result)

    def test_validate_plan_object_rejects_duplicate_branch_labels(self) -> None:
        wf = PlanningWorkflow(llm_generate=lambda _: "{}")
        bad_node = {
            "id": "S1",
            "task": "Retrieve the list of available IoT sites.",
            "agent": "iot",
            "deps": [],
            "verification": True,
            "branches": [
                {
                    "next": "B1",
                    "label": "execution",
                },
                {
                    "next": "B2",
                    "label": "execution",
                },
            ],
        }

        with self.assertRaises(ValueError):
            wf._validate_plan_object(bad_node, kind="node")

    def test_validate_plan_object_allows_duplicate_terminate_next(self) -> None:
        wf = PlanningWorkflow(llm_generate=lambda _: "{}")
        node = {
            "id": "V_S1",
            "task": "Verifier node",
            "agent": "verifier",
            "deps": ["S1"],
            "branches": [
                {
                    "label": "success",
                    "next": "TERMINATE",
                },
                {
                    "label": "terminate",
                    "next": "TERMINATE",
                },
            ],
        }

        # This should pass once validator allows duplicate next only for TERMINATE.
        wf._validate_plan_object(node, kind="node")

    def test_validate_branch_accepts_minimal_fragment_schema(self) -> None:
        wf = PlanningWorkflow(llm_generate=lambda _: "{}")
        wf._validate_plan_object(self.sample_branch_fragment_b1, kind="branch")

    def test_merge_module_outputs_success_flattened(self) -> None:
        wf = PlanningWorkflow(llm_generate=lambda _: "{}")

        final_plan = wf._merge_module_outputs(
            annotated_plan=self.sample_annotated_plan,
            module3_nodes=[self.sample_module3_node_with_branches],
            module4_subplans=[self.sample_branch_fragment_b1, self.sample_branch_fragment_b2],
        )

        self.assertEqual(final_plan["answer_contract"], self.sample_annotated_plan["answer_contract"])
        self.assertIn("nodes", final_plan)
        self.assertNotIn("branch_subplans", final_plan)

        node_ids = {node["id"] for node in final_plan["nodes"]}
        self.assertIn("S1", node_ids)
        self.assertIn("V_S1", node_ids)
        self.assertIn("B1_S1", node_ids)
        self.assertIn("B2_S1", node_ids)

        verifier_node = next(node for node in final_plan["nodes"] if node["id"] == "V_S1")
        verifier_branches = verifier_node["branches"]
        verifier_branch_labels = {b["label"] for b in verifier_branches}
        self.assertIn("success", verifier_branch_labels)
        self.assertIn("execution", verifier_branch_labels)
        self.assertIn("availability", verifier_branch_labels)
        self.assertIn("terminate", verifier_branch_labels)

        # Because S1 has no downstream child in this toy setup,
        # both success and terminate may legitimately point to TERMINATE.
        terminate_targets = [b for b in verifier_branches if b["next"] == "TERMINATE"]
        self.assertGreaterEqual(len(terminate_targets), 2)

        b1_node = next(node for node in final_plan["nodes"] if node["id"] == "B1_S1")
        self.assertEqual(b1_node["deps"], ["V_S1"])

        b1_branch_labels = {b["label"] for b in b1_node["branches"]}
        self.assertIn("success", b1_branch_labels)
        self.assertIn("terminate", b1_branch_labels)

        b1_terminate_targets = [b for b in b1_node["branches"] if b["next"] == "TERMINATE"]
        self.assertGreaterEqual(len(b1_terminate_targets), 2)

    def test_generate_plan_object_success_flattened(self) -> None:
        fake_llm = FakeLLM([
            json.dumps(self.sample_base_plan),
            json.dumps(self.sample_annotated_plan),
            json.dumps(self.sample_module3_node_with_branches),
            json.dumps(self.sample_branch_fragment_b1),
            json.dumps(self.sample_branch_fragment_b2),
        ])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        result = wf.generate_plan_object(
            task_description="Find the available IoT site.",
            agent_catalog=["iot", "fmsr"],
            step_type_library=["retrieve", "synthesis"],
            output_format="json",
            constraints={"max_nodes": 5},
            exception_taxonomy=["execution", "availability", "contract"],
            agent_env_specs_json={"autonomy": "high"},
            empirical_failure_profile={"execution": 0.6, "availability": 0.5},
            additional_agent_catalog=["recovery", "replanning"],
        )

        self.assertEqual(result["answer_contract"], "Return the available IoT site.")
        self.assertIn("nodes", result)
        self.assertNotIn("branch_subplans", result)

        node_ids = {node["id"] for node in result["nodes"]}
        self.assertIn("S1", node_ids)
        self.assertIn("V_S1", node_ids)
        self.assertIn("B1_S1", node_ids)
        self.assertIn("B2_S1", node_ids)

        verifier_node = next(node for node in result["nodes"] if node["id"] == "V_S1")
        verifier_branch_labels = {b["label"] for b in verifier_node["branches"]}
        self.assertIn("success", verifier_branch_labels)
        self.assertIn("execution", verifier_branch_labels)
        self.assertIn("availability", verifier_branch_labels)
        self.assertIn("terminate", verifier_branch_labels)

    def test_generate_plan_object_fails_on_invalid_branch_id_mismatch(self) -> None:
        bad_branch_fragment = {
            "id": "B9",
            "nodes": [
                {
                    "id": "B9_S1",
                    "task": "Bad branch.",
                    "agent": "iot",
                    "deps": ["V_S1"],
                }
            ],
        }

        fake_llm = FakeLLM([
            json.dumps(self.sample_base_plan),
            json.dumps(self.sample_annotated_plan),
            json.dumps(self.sample_module3_node_with_branches),
            json.dumps(bad_branch_fragment),
            json.dumps(self.sample_branch_fragment_b2),
        ])
        wf = PlanningWorkflow(llm_generate=fake_llm)

        with self.assertRaises(ValueError):
            wf.generate_plan_object(
                task_description="Find the available IoT site.",
                agent_catalog=["iot", "fmsr"],
                step_type_library=["retrieve", "synthesis"],
                output_format="json",
                constraints=None,
                exception_taxonomy=["execution", "availability"],
                agent_env_specs_json={"autonomy": "high"},
                empirical_failure_profile={"execution": 0.6},
                additional_agent_catalog=["recovery", "replanning"],
            )


def main() -> None:
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()