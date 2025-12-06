from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import json
import os
import sys
import re

HOME_DIR = "/home"
if HOME_DIR not in sys.path:
    sys.path.insert(0, HOME_DIR)

from summarize import watsonx_llm  # type: ignore


GROUND_TRUTH_TRAJECTORIES_JSON: List[str] = [
    '''{
    "uuid": "c50d83ca-6082-447d-a030-70b38a4e4a4b",
    "id": 1,
    "type": "IoT",
    "text": "What IoT sites are available?",
    "category": "Knowledge Query",
    "characteristic_form": "first call action sites with no parameters",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "list all the sites"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "sites",
            "agent": "IoTAgent",
            "arguments": {},
            "outputs": []
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The available IoT site is MAIN.  The final answer is The available IoT site is MAIN.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}''', 
'''{
    "uuid": "9f645a51-c34b-46c8-b166-b23d4a1acada",
    "id": 2,
    "type": "IoT",
    "text": "Can you list the IoT sites?",
    "category": "Knowledge Query",
    "characteristic_form": "The expected response should be the return value of all sites, either as text or as a reference to a file",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "list all the sites"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "sites",
            "agent": "IoTAgent",
            "arguments": {},
            "outputs": []
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The available IoT sites are: MAIN.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}''', 
'''{
    "uuid": "efc94d35-5236-410c-9e4f-5dcdfee818cc",
    "id": 3,
    "type": "IoT",
    "text": "What assets can be found at the MAIN site?",
    "category": "Knowledge Query",
    "characteristic_form": "The expected response should be the return value from querying the assets at the MAIN site. The response should be a reference to a file containing the list of assets",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "list assets for site MAIN"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "assets",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN"
            },
            "outputs": []
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The assets at the MAIN site are: CQPA AHU 1, CQPA AHU 2B, Chiller 4, Chiller 6, Chiller 9, Chiller 3.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}''',
'''{
    "uuid": "bd83d19e-ca09-43e7-89ac-51dfc5088588",
    "id": 4,
    "type": "IoT",
    "text": "Which assets are located at the MAIN facility?",
    "category": "Knowledge Query",
    "characteristic_form": "The expected response should be the return value from querying the assets at the MAIN site. The response should be a reference to a file containing the list of assets",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "list assets for site MAIN"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "assets",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN"
            },
            "outputs": []
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The assets for site MAIN are: CQPA AHU 1, CQPA AHU 2B, Chiller 4, Chiller 6, Chiller 9, Chiller 3.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}''',
'''{
    "uuid": "0a220b4f-2c2e-4dcd-adc2-0a2f7f15853e",
    "id": 5,
    "type": "IoT",
    "text": "Retrieve metadata for Chiller 6 located at the MAIN site.",
    "category": "Data Query",
    "characteristic_form": "The expected response should be the metadata for asset 'Chiller 6' at the MAIN site.  The metadata may be in the return value, or the may be returned as a reference to a file containing the metadata",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "retrieve metadata for Chiller 6 at MAIN site"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "assets",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN"
            },
            "outputs": []
        },
        {
            "name": "step2",
            "action": "jsonreader",
            "agent": "IoTAgent",
            "arguments": {
                "file_name": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/6d1c069b-39f3-4849-9c23-defd183367a5.json"
            },
            "outputs": []
        },
        {
            "name": "step3",
            "action": "sensors",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN",
                "assetnum": "Chiller 6"
            },
            "outputs": []
        },
        {
            "name": "step4",
            "action": "jsonreader",
            "agent": "IoTAgent",
            "arguments": {
                "file_name": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/1cf556ab-b911-422b-9b9e-daf67076a38f.json"
            },
            "outputs": [
                "The metadata for Chiller 6 at MAIN site includes: Chiller 6 Condenser Water Return To Tower Temperature, Chiller 6 Chiller Efficiency, Chiller 6 Tonnage, Chiller 6 Supply Temperature, Chiller 6 Return Temperature, Chiller 6 Run Status, Chiller 6 Condenser Water Flow, Chiller 6 Schedule, Chiller 6 Power Input, Chiller 6 Chiller % Loaded, Chiller 6 Liquid Refrigerant Evaporator Temperature, Chiller 6 Setpoint Temperature."
            ]
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The metadata for Chiller 6 at MAIN site includes: Chiller 6 Condenser Water Return To Tower Temperature, Chiller 6 Chiller Efficiency, Chiller 6 Tonnage, Chiller 6 Supply Temperature, Chiller 6 Return Temperature, Chiller 6 Run Status, Chiller 6 Condenser Water Flow, Chiller 6 Schedule, Chiller 6 Power Input, Chiller 6 Chiller % Loaded, Chiller 6 Liquid Refrigerant Evaporator Temperature, Chiller 6 Setpoint Temperature.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "step2"
        },
        {
            "source": "step2",
            "target": "step3"
        },
        {
            "source": "step3",
            "target": "step4"
        },
        {
            "source": "step5",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}''',
'''{
    "uuid": "592c6bf8-9866-44b6-808e-8db4564ac8ad",
    "id": 6,
    "type": "IoT",
    "text": "Get the asset details for Chiller 9 at the MAIN site.",
    "category": "Data Query",
    "characteristic_form": "The expected response should be the details for asset 'Chiller 9' at the MAIN site.  The details may be in the return value, or the may be returned as a reference to a file containing the details",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "get asset details for Chiller 9 at MAIN site"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "assets",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN"
            },
            "outputs": []
        },
        {
            "name": "step2",
            "action": "jsonreader",
            "agent": "IoTAgent",
            "arguments": {
                "file_name": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/81ea7d5b-4667-452e-9c0f-eff30353858b.json"
            },
            "outputs": []
        },
        {
            "name": "step3",
            "action": "sensors",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN",
                "assetnum": "Chiller 9"
            },
            "outputs": []
        },
        {
            "name": "step4",
            "action": "jsonreader",
            "agent": "IoTAgent",
            "arguments": {
                "file_name": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/81ea7d5b-4667-452e-9c0f-eff30353858b.json"
            },
            "outputs": [
                "The asset details for Chiller 9 at MAIN site include the following sensors: Chiller 9 Setpoint Temperature, Chiller 9 Supply Temperature, Chiller 9 Tonnage, Chiller 9 Run Status, Chiller 9 Return Temperature, Chiller Efficiency, Chiller 9 Schedule, Chiller 9 Power Input, Chiller 9 Chiller % Loaded, Chiller 9 Condenser Water Flow, Chiller 9 Liquid Refrigerant Evaporator Temperature, Chiller 9 Condenser Water Supply To Chiller Temperature."
            ]
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The metadata for Chiller 6 at MAIN site includes: Chiller 6 Condenser Water Return To Tower Temperature, Chiller 6 Chiller Efficiency, Chiller 6 Tonnage, Chiller 6 Supply Temperature, Chiller 6 Return Temperature, Chiller 6 Run Status, Chiller 6 Condenser Water Flow, Chiller 6 Schedule, Chiller 6 Power Input, Chiller 6 Chiller % Loaded, Chiller 6 Liquid Refrigerant Evaporator Temperature, Chiller 6 Setpoint Temperature.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "step2"
        },
        {
            "source": "step2",
            "target": "step3"
        },
        {
            "source": "step3",
            "target": "step4"
        },
        {
            "source": "step4",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}''',
'''{
    "uuid": "4e825afb-3b00-473c-b19f-52284d5c08f4",
    "id": 7,
    "type": "IoT",
    "text": "Download the metadata for Chiller 3 at the MAIN facility.",
    "category": "Data Query",
    "characteristic_form": "The expected response should be the metadata for asset 'Chiller 3' as the MAIN site.  The metadata may be in the return value, or the may be returned as a reference to a file containing the metadata",
    "expected_result": null,
    "data": {},
    "planning_steps": [
        {
            "agent": "IoTAgent",
            "instruction": "download metadata for Chiller 3 at MAIN facility"
        }
    ],
    "execution_steps": [
        {
            "name": "step1",
            "action": "sensors",
            "agent": "IoTAgent",
            "arguments": {
                "site_name": "MAIN",
                "assetnum": "Chiller 3"
            },
            "outputs": []
        },
        {
            "name": "step2",
            "action": "jsonreader",
            "agent": "IoTAgent",
            "arguments": {
                "file_name": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/bbbb2e97-77d5-4376-8a21-32a67dda0169.json"
            },
            "outputs": [
                "The metadata for Chiller 3 at MAIN facility has been downloaded and includes the following sensors: Chiller 3 Condenser Water Flow, Chiller 3 Chiller Efficiency, Chiller 3 Liquid Refrigerant Evaporator Temperature, Chiller 3 Run Status, Chiller 3 Tonnage, Chiller 3 Chiller % Loaded, Chiller 3 Supply Temperature, Chiller 3 Condenser Water Supply To Chiller Temperature, Chiller 3 Schedule, Chiller 3 Setpoint Temperature, Chiller 3 Power Input, Chiller 3 Return Temperature."
            ]
        },
        {
            "name": "finish",
            "action": "Finish",
            "agent": "IoTAgent",
            "argument": "The metadata for Chiller 3 at MAIN facility has been downloaded and includes the following sensors: Chiller 3 Condenser Water Flow, Chiller 3 Chiller Efficiency, Chiller 3 Liquid Refrigerant Evaporator Temperature, Chiller 3 Run Status, Chiller 3 Tonnage, Chiller 3 Chiller % Loaded, Chiller 3 Supply Temperature, Chiller 3 Condenser Water Supply To Chiller Temperature, Chiller 3 Schedule, Chiller 3 Setpoint Temperature, Chiller 3 Power Input, Chiller 3 Return Temperature.",
            "deterministic": {
                "name": false,
                "action": true,
                "arguments": true,
                "outputs": true
            }
        }
    ],
    "execution_links": [
        {
            "source": "step1",
            "target": "step2"
        },
        {
            "source": "step2",
            "target": "finish"
        }
    ],
    "possible_alternatives": {}
}'''
]


def build_few_shot_from_ground_truth(
    traj_json_list: Sequence[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for raw in traj_json_list:
        traj = json.loads(raw)

        user_q = traj.get("text", "").strip()
        exec_steps = traj.get("execution_steps", []) or []

        # Build a simple DAG prefix string list: "Agent -> action(args)"
        dag_prefix: list[str] = []
        for step in exec_steps:
            agent = step.get("agent", "")
            action = step.get("action", "")
            arguments = step.get("arguments", {})
            if action == "Finish":
                # We still include it in the prefix so Critic sees the full trajectory
                arg_str = step.get("argument", "")
                dag_prefix.append(
                    f"{agent} -> Finish(argument={arg_str!r})"
                )
            else:
                dag_prefix.append(
                    f"{agent} -> {action}({json.dumps(arguments, ensure_ascii=False)})"
                )

        # Candidate answer = Finish.argument (if any)
        finish_step = next(
            (s for s in exec_steps if s.get("action") == "Finish"), None
        )
        if finish_step is None:
            # If no Finish step, skip this trajectory as a few-shot example
            continue

        candidate_answer = str(finish_step.get("argument", "")).strip()
        if not user_q or not candidate_answer:
            continue

        examples.append(
            {
                "name": f"Trajectory {traj.get('id', traj.get('uuid', 'unknown'))}",
                "user_question": user_q,
                "dag_prefix": dag_prefix,
                "candidate_answer": candidate_answer,
                "label": {
                    "status": "Accomplished",
                    "can_answer_now": True,
                    "rationale": (
                        "The candidate answer matches the final Finish.argument "
                        "of the ground-truth execution_steps for this trajectory."
                    ),
                },
            }
        )

    return examples

FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = build_few_shot_from_ground_truth(
    GROUND_TRUTH_TRAJECTORIES_JSON
)



# -------------------------------------------------------------------
# CriticAgent 本体
# -------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """
You are a CRITIC AGENT for DAG-based multi-agent workflows.

Your job:
- You receive a user question, a DAG prefix (steps that have been planned
  or executed so far), and a candidate answer produced by another agent.
- You must judge how well the candidate answer responds to the user question,
  given the DAG prefix.

Your decision must follow this schema (JSON, single top-level object):

{
  "status": "Accomplished" | "Partially accomplished" | "Not accomplished",
  "can_answer_now": true | false,
  "rationale": "short natural-language explanation"
}

Semantics:
- "Accomplished": the answer is essentially correct and complete for the question.
- "Partially accomplished": the answer is on-topic but clearly incomplete or missing
  some important details.
- "Not accomplished": the answer is incorrect, off-topic, or fundamentally misaligned.

IMPORTANT:
- You MUST output valid JSON only.
- Do NOT wrap the JSON in markdown, backticks, or any extra text.
- Do NOT add extra fields.
""".strip()


class CriticAgent:
    """
    Minimal Critic agent that uses:
    - watsonx LLM for judgement, and
    - a small set of few-shot rubric examples (FEW_SHOT_EXAMPLES)
      that encode how to evaluate DAG prefixes and answers.

    It does NOT see the ground-truth answer of the current question.
    Ground truth is only used offline to create the few-shot examples.
    """

    def __init__(
        self,
        *,
        system_prompt: str = CRITIC_SYSTEM_PROMPT,
        few_shot_examples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        system_prompt : str
            High-level instruction and JSON schema for the Critic.
        few_shot_examples : Optional[Sequence[Dict[str, Any]]]
            A list of rubric examples of the form:
            {
              "user_question": str,
              "dag_prefix": List[str] or List[dict],
              "candidate_answer": str,
              "label": {
                  "status": "...",
                  "can_answer_now": bool,
                  "rationale": "..."
              }
            }
        """
        self.system_prompt = system_prompt
        self.few_shot_examples: List[Dict[str, Any]] = list(
            few_shot_examples or FEW_SHOT_EXAMPLES
        )
        self.llm = watsonx_llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        user_question: str,
        candidate_answer: str,
        dag_prefix: Sequence[Any],
        scenario_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate answer for a given DAG prefix and user question.

        Parameters
        ----------
        user_question : str
            Original user question.
        candidate_answer : str
            Answer produced by the Simulator or executing agents.
        dag_prefix : Sequence[Any]
            Prefix of the DAG (steps executed so far). Each element can be
            - a string, or
            - a dict with keys like {task, agent, dependency, expected_output}.
        scenario_context : Optional[Dict[str, Any]]
            Optional extra context (e.g. scenario metadata).

        Returns
        -------
        result : dict
            {
              "status": "Accomplished" | "Partially accomplished" | "Not accomplished",
              "can_answer_now": bool,
              "rationale": str,
            }
        """
        context_text = self._build_context(
            user_question=user_question,
            candidate_answer=candidate_answer,
            dag_prefix=dag_prefix,
            scenario_context=scenario_context,
        )

        # Call watsonx LLM (model_id=16 は summarize.py 側の設定に合わせる)
        resp = self.llm(context_text, model_id=16)
        raw_text = resp.get("generated_text", "")
        in_tok = resp.get("input_token_count", 0)
        out_tok = resp.get("generated_token_count", 0)

        # Parse JSON
        parsed = self._parse_llm_json(raw_text)

        return parsed, in_tok, out_tok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_context(
        self,
        *,
        user_question: str,
        candidate_answer: str,
        dag_prefix: Sequence[Any],
        scenario_context: Optional[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.append(self.system_prompt)
        lines.append("")
        lines.append("=== Few-shot evaluation examples ===")

        # Few-shot 例をすべて出す（量が増えたら類似度で絞るロジックを入れてもよい）
        for idx, ex in enumerate(self.few_shot_examples, start=1):
            label = ex["label"]
            lines.append(f"Example {idx}: {ex.get('name', '')}")
            lines.append(f"User question: {ex['user_question']}")
            lines.append("DAG prefix:")
            for step in ex["dag_prefix"]:
                lines.append(self._format_step(step))
            lines.append(f"Candidate answer: {ex['candidate_answer']}")
            lines.append("Expected evaluation JSON:")
            lines.append(json.dumps(label, ensure_ascii=False))
            lines.append("")

        # New case
        lines.append("=== New case to evaluate ===")
        lines.append(f"User question: {user_question}")
        lines.append("DAG prefix:")
        for step in dag_prefix:
            lines.append(self._format_step(step))
        lines.append(f"Candidate answer: {candidate_answer}")

        if scenario_context:
            lines.append("")
            lines.append("Additional scenario context (for reference):")
            lines.append(json.dumps(scenario_context, ensure_ascii=False))

        lines.append("")
        lines.append(
            "Now, based on the evaluation examples above, decide the status, "
            "can_answer_now, and rationale for THIS new case. "
            "Output ONLY a single JSON object with keys "
            "\"status\", \"can_answer_now\", and \"rationale\"."
        )

        return "\n".join(lines)

    @staticmethod
    def _format_step(step: Any) -> str:
        """
        DAG の 1 ステップを、人間が読める 1 行にフォーマットする。

        - step が文字列ならそのまま出す
        - dict の場合は {task, agent, dependency, expected_output} を優先的に使う
        """
        if isinstance(step, str):
            return f"  - {step}"

        if isinstance(step, dict):
            task = step.get("task")
            agent = step.get("agent")
            dep = step.get("dependency")
            exp = step.get("expected_output")

            parts = []
            if agent:
                parts.append(f"Agent={agent}")
            if task:
                parts.append(f"Task={task}")
            if dep:
                parts.append(f"Dependency={dep}")
            if exp:
                parts.append(f"ExpectedOutput={exp}")
            if not parts:
                parts.append(str(step))

            return "  - " + ", ".join(parts)

        # その他の型はそのまま文字列化
        return "  - " + str(step)

    @staticmethod
    def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
        """
        Extract and parse a JSON object from the LLM's raw text output.

        The model is instructed to output JSON only, but in case it
        adds extra text, we try to find the first '{' and last '}'.
        """
        text = raw_text.strip()

        # Try direct parse first
        try:
            obj = json.loads(text)
            return CriticAgent._sanitize_result(obj)
        except Exception:
            pass

        # Fallback: attempt to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return CriticAgent._sanitize_result(obj)
            except Exception:
                pass

        # Final fallback: return a default Not accomplished
        return {
            "status": "Not accomplished",
            "can_answer_now": False,
            "rationale": f"Failed to parse JSON from model output: {raw_text[:200]}",
        }

    @staticmethod
    def _sanitize_result(obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the result has the required keys and valid values.
        """
        status = obj.get("status", "Not accomplished")
        if status not in {
            "Accomplished",
            "Partially accomplished",
            "Not accomplished",
        }:
            status = "Not accomplished"

        can_answer_now = bool(obj.get("can_answer_now", False))
        rationale = str(obj.get("rationale", "")).strip()
        if not rationale:
            rationale = "No rationale provided by the model."

        return {
            "status": status,
            "can_answer_now": can_answer_now,
            "rationale": rationale,
        }


# -------------------------------------------------------------------
# 「#Task1: ...」形式の final_plan を 4 属性 DAG に変換するヘルパ
#   （簡易パーサ：Task/Agent/Dependency/ExpectedOutput を抜き出す）
# -------------------------------------------------------------------
TASK_RE = re.compile(
    r"#Task(?P<idx>\d+):\s*(?P<task>.+?)(?=\n#Agent|\Z)", re.DOTALL
)
AGENT_RE = re.compile(
    r"#Agent(?P<idx>\d+):\s*(?P<agent>.+?)(?=\n#Dependency|\n#ExpectedOutput|\Z)",
    re.DOTALL,
)
DEP_RE = re.compile(
    r"#Dependency(?P<idx>\d+):\s*(?P<dep>.+?)(?=\n#ExpectedOutput|\Z)",
    re.DOTALL,
)
EXP_RE = re.compile(
    r"#ExpectedOutput(?P<idx>\d+):\s*(?P<exp>.+?)(?=\n#Task|\Z)",
    re.DOTALL,
)


def parse_plan_text_to_dag_prefix(plan_text: str) -> List[Dict[str, Any]]:
    """
    Example of plan_text:
        "#Task1: List all available IoT sites.\n"
        "#Agent1: IoT Data Download\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: A list of available IoT sites.\n"
        "#Task2: ..."

    Return:
        [
          {
            "task": "List all available IoT sites.",
            "agent": "IoT Data Download",
            "dependency": "None",
            "expected_output": "A list of available IoT sites.",
          },
          ...
        ]
    """
    tasks: Dict[str, Dict[str, Any]] = {}

    def _ensure_slot(idx: str) -> Dict[str, Any]:
        if idx not in tasks:
            tasks[idx] = {}
        return tasks[idx]

    for m in TASK_RE.finditer(plan_text):
        d = _ensure_slot(m.group("idx"))
        d["task"] = m.group("task").strip()

    for m in AGENT_RE.finditer(plan_text):
        d = _ensure_slot(m.group("idx"))
        d["agent"] = m.group("agent").strip()

    for m in DEP_RE.finditer(plan_text):
        d = _ensure_slot(m.group("idx"))
        d["dependency"] = m.group("dep").strip()

    for m in EXP_RE.finditer(plan_text):
        d = _ensure_slot(m.group("idx"))
        d["expected_output"] = m.group("exp").strip()

    # idx の数値順に並べる
    dag_prefix: List[Dict[str, Any]] = []
    for idx in sorted(tasks.keys(), key=lambda s: int(s)):
        dag_prefix.append(tasks[idx])

    return dag_prefix


def main() -> None:
    """
    Minimal CLI-style test for CriticAgent.

    We simulate the situation where a SimulatorAgent has already produced
    a predicted answer for a given DAG prefix and user question.

    Test 1:
        - simulated_predicted_answer_1 is a full natural-language list
          of assets at MAIN.
        - We expect the Critic to judge this as "Accomplished".

    Test 2:
        - simulated_predicted_answer_2 is only a file path that (presumably)
          contains the list of assets.
        - We expect the Critic to judge this as "Partially accomplished".
    """
    critic = CriticAgent()

    user_question = "Which assets are located at the MAIN facility?"

    # DAG prefix built from the ground-truth execution_steps:
    #   step1: IoTAgent -> assets(site_name='MAIN')
    #   finish: IoTAgent -> Finish(argument='The assets for site MAIN are: ...')
    dag_prefix_full = [
        "IoTAgent -> assets(site_name='MAIN')",
        (
            "IoTAgent -> Finish("
            "argument='The assets for site MAIN are: "
            "CQPA AHU 1, CQPA AHU 2B, Chiller 4, "
            "Chiller 6, Chiller 9, Chiller 3.')"
        ),
    ]

    print("=== CriticAgent manual test (few-shot based) ===")
    print(f"User Question: {user_question}")
    print("DAG Prefix (full):")
    for s in dag_prefix_full:
        print("  ", s)
    print()

    # ------------------------------------------------------------------
    # Test 1: SimulatorAgent hypothetical output (good natural-language list)
    # ------------------------------------------------------------------
    simulated_predicted_answer_1 = (
        "The assets for site MAIN are: "
        "CQPA AHU 1, CQPA AHU 2B, Chiller 4, "
        "Chiller 6, Chiller 9, Chiller 3."
    )

    print(">> Test 1: Hypothetical SimulatorAgent predicted answer (full list)")
    print("simulated_predicted_answer_1:")
    print("  ", simulated_predicted_answer_1)
    print()

    result1 = critic.evaluate(
        user_question=user_question,
        candidate_answer=simulated_predicted_answer_1,
        dag_prefix=dag_prefix_full,
    )
    print("Critic result for Test 1:")
    print(json.dumps(result1, indent=2))
    print()

    # ------------------------------------------------------------------
    # Test 2: SimulatorAgent hypothetical output (file reference only)
    #   Here we only assume the first step of the DAG has run (assets call),
    #   and the simulated predicted answer is a file path.
    # ------------------------------------------------------------------
    dag_prefix_partial = [
        "IoTAgent -> assets(site_name='MAIN')",
    ]
    simulated_predicted_answer_2 = "/tmp/cbmdir/MAIN_assets.json"

    print(">> Test 2: Hypothetical SimulatorAgent predicted answer (file path only)")
    print("DAG Prefix (partial):")
    for s in dag_prefix_partial:
        print("  ", s)
    print("simulated_predicted_answer_2:")
    print("  ", simulated_predicted_answer_2)
    print()

    result2 = critic.evaluate(
        user_question=user_question,
        candidate_answer=simulated_predicted_answer_2,
        dag_prefix=dag_prefix_partial,
    )
    print("Critic result for Test 2:")
    print(json.dumps(result2, indent=2))
    print()

    print("=== End CriticAgent test ===")


# if __name__ == "__main__":
#     main()
