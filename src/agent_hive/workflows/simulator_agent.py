from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict
from uuid import UUID
from datetime import datetime
import psycopg  # psycopg 3.x
import sys
import os

# Ensure /home (where summarize.py is mounted) is on Python's module search path
HOME_DIR = "/home"
if HOME_DIR not in sys.path:
    sys.path.insert(0, HOME_DIR)

# Now we can import from summarize.py
from summarize import get_watsonx_embedder_for_summary, watsonx_llm

# ------------------------------------------------------------------
# Ground-truth trajectories for few-shot prompting
# ------------------------------------------------------------------
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


# --------------------------------------------------------------------
# Types
# --------------------------------------------------------------------
class TaskSummaryHit(TypedDict):
    """
    Minimal representation of a retrieved task-level summary row.
    """
    doc_id: UUID
    task_id: UUID
    task_number: int
    summary: str
    user_question: str
    task_description: str
    agent_name: str
    status: Optional[str]
    created_at: datetime
    similarity_score: float
    rank: int


class SimulatorAgent:
    """
    Minimal Simulator agent.

    Responsibilities:
    - Given (user_question, task_description, agent_name),
      1) retrieve similar tasks by querying traj_task_summaries
         with vector similarity (summary_vec),
      2) build a simple LLM prompt using those summaries as context,
      3) call Watsonx LLM to predict the output for this task.

    This class does NOT handle uncertainty, Critic decisions, etc.
    It only predicts the output text for a single target task.
    """

    def __init__(
        self,
        *,
        db_url: Optional[str] = None,
        system_prompt: str,
        max_similar_tasks: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        db_url : Optional[str]
            PostgreSQL connection string. If None, it will be read from
            environment variables DB_URL or DATABASE_URL.
        system_prompt : str
            High-level instruction for the Simulator agent. This string
            is placed at the start of the LLM prompt.
        max_similar_tasks : int
            Maximum number of similar tasks to retrieve and include in
            the LLM context.
        """
        self.db_url = db_url or os.getenv("DB_URL") or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise RuntimeError("DB_URL / DATABASE_URL must be set for SimulatorAgent.")

        self.system_prompt = system_prompt
        self.max_similar_tasks = max_similar_tasks

        # Embedder for traj_task_summaries.summary_vec
        # This function should map text -> list[float] with dimension
        # matching your Postgres vector column (e.g., vector(1536)).
        self.embed_fn = get_watsonx_embedder_for_summary()

        # LLM client wrapper for simulation
        self.watsonx_llm = watsonx_llm

    @staticmethod
    def _to_pgvector_literal(vec: Sequence[float]) -> str:
        """
        Convert a Python list[float] into a pgvector literal string.

        Example:
            [0.1, 0.2, 0.3] -> "[0.1,0.2,0.3]"

        We will then use it in SQL as:
            summary_vec <-> %(q_vec)s::vector
        """
        # You can choose the precision; 6 decimals is usually enough.
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        user_question: str,
        task_description: str,
        agent_name: str,
        dag_prefix: Optional[Sequence[Any]] = None,
    ) -> str:
        """
        Predict the output for a single target task.

        Parameters
        ----------
        user_question : str
            Original user query.
        task_description : str
            Target task description to simulate.
        agent_name : str
            Name of the agent that will execute this task.
        dag_prefix : Optional[Sequence[Any]]
            Optional list of previous DAG steps. For now we stringify
            them directly; you can refine the formatting later.

        Returns
        -------
        predicted_output : str
            LLM-predicted output for this task.
        """
        user_question = str(user_question).strip()
        task_description = str(task_description).strip()
        agent_name = str(agent_name).strip()

        # 1) Build query text for similarity search
        query_text = self._build_query_text(
            user_question=user_question,
            task_description=task_description,
            agent_name=agent_name,
        )

        # 2) Retrieve similar tasks (success-only for now)
        similar_tasks = self.search_task_summaries(
            query_text=query_text,
            top_k=self.max_similar_tasks,
            agent_names=[agent_name] if agent_name else None,
            status_filter=["Accomplished"],
        )

        # 3) Build prompt for the LLM
        prompt = self._build_context(
            user_question=user_question,
            task_description=task_description,
            agent_name=agent_name,
            dag_prefix=dag_prefix,
            similar_tasks=similar_tasks,
        )

        # 4) Call Watsonx LLM to simulate the output
        #    We assume model_id=16 is defined in your environment.
        llm_result = self.watsonx_llm(prompt, model_id=16)
        predicted_answer = llm_result.get("generated_text", "").strip()
        in_tok = llm_result.get("input_token_count", 0)
        out_tok = llm_result.get("generated_token_count", 0)

        return predicted_answer, in_tok, out_tok

    def search_task_summaries(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        agent_names: Optional[Sequence[str]] = None,
        status_filter: Optional[Sequence[str]] = None,
    ) -> List[TaskSummaryHit]:
        """
        Search traj_task_summaries using pgvector similarity on summary_vec.
        """
        # 1) Embed query_text into the same vector space as summary_vec
        q_vec_list = self.embed_fn(query_text)

        # DEBUG (optional): check length
        # print(f"[SimulatorAgent] query embedding dim={len(q_vec_list)}")

        # Convert to pgvector literal string, e.g. "[0.1,0.2,0.3]"
        q_vec_literal = self._to_pgvector_literal(q_vec_list)

        agent_list: Optional[List[str]] = list(agent_names) if agent_names else None
        status_list: Optional[List[str]] = list(status_filter) if status_filter else None

        hits: List[TaskSummaryHit] = []

        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT
                        doc_id,
                        task_id,
                        task_number,
                        summary,
                        user_question,
                        task_description,
                        agent_name,
                        status,
                        created_at,
                        1.0 / (1.0 + (summary_vec <-> %(q_vec)s::vector)) AS similarity_score
                    FROM traj_task_summaries
                    WHERE summary_vec IS NOT NULL
                """
                params: Dict[str, Any] = {
                    "q_vec": q_vec_literal,
                    "top_k": top_k,
                }

                if agent_list:
                    sql += " AND agent_name = ANY(%(agent_names)s)"
                    params["agent_names"] = agent_list

                if status_list:
                    sql += " AND status = ANY(%(status_filter)s)"
                    params["status_filter"] = status_list

                sql += """
                    ORDER BY summary_vec <-> %(q_vec)s::vector
                    LIMIT %(top_k)s
                """

                cur.execute(sql, params)
                rows = cur.fetchall()

        for idx, row in enumerate(rows):
            (
                doc_id,
                task_id,
                task_number,
                summary_text,
                user_q,
                task_desc,
                agent_name_row,
                status,
                created_at,
                sim_score,
            ) = row

            hit: TaskSummaryHit = {
                "doc_id": doc_id,
                "task_id": task_id,
                "task_number": task_number,
                "summary": summary_text,
                "user_question": user_q,
                "task_description": task_desc,
                "agent_name": agent_name_row,
                "status": status,
                "created_at": created_at,
                "similarity_score": float(sim_score),
                "rank": idx + 1,
            }
            hits.append(hit)

        return hits


    # ------------------------------------------------------------------
    # Internal prompt-building helpers
    # ------------------------------------------------------------------
    def _build_query_text(
        self,
        *,
        user_question: str,
        task_description: str,
        agent_name: str,
    ) -> str:
        """
        Build a compact query string for similarity search.

        You can later tweak the weighting (e.g., repeat certain fields)
        if you want to emphasize agent_name or task_description.
        """
        return (
            f"User question: {user_question}\n"
            f"Agent: {agent_name}\n"
            f"Task: {task_description}"
        )

    def _build_context(
        self,
        *,
        user_question: str,
        task_description: str,
        agent_name: str,
        dag_prefix: Optional[Sequence[Any]],
        similar_tasks: List[TaskSummaryHit],
    ) -> str:
        """
        Build the prompt text given to the LLM.

        The format is:
        - system instructions
        - user question
        - DAG prefix (stringified)
        - ground-truth example trajectories (full JSON, few-shot)
        - target task
        - similar past task summaries
        - final instruction to "output only the predicted result"
        """
        lines: List[str] = []

        # System instructions
        lines.append(self.system_prompt.strip())
        lines.append("")
        lines.append("=== User Question ===")
        lines.append(user_question)

        # Optional DAG prefix
        if dag_prefix:
            lines.append("")
            lines.append("=== DAG Prefix (high-level) ===")
            for i, step in enumerate(dag_prefix, start=1):
                # For now we simply cast step to string.
                # You can replace this with a custom formatter if step is a dataclass or dict.
                lines.append(f"Step {i}: {step}")

        # ------------------------------------------------------------------
        # Ground-truth trajectories (few-shot examples, full JSON as-is)
        # ------------------------------------------------------------------
        if GROUND_TRUTH_TRAJECTORIES_JSON:
            lines.append("")
            lines.append("=== Ground-Truth Trajectories (Few-Shot Examples) ===")
            lines.append(
                "Below are full ground-truth trajectories in JSON format, "
                "including planning_steps and execution_steps. "
                "Use them as exemplars of how an IoT agent behaves and what "
                "its final outputs look like."
            )
            for idx, traj_json in enumerate(GROUND_TRUTH_TRAJECTORIES_JSON, start=1):
                lines.append("")
                lines.append(f"--- Ground-Truth Trajectory {idx} ---")
                # Insert the JSON exactly as stored, no compression or rewriting.
                lines.append(traj_json)

        # Target task
        lines.append("")
        lines.append("=== Target Task ===")
        lines.append(f"Agent: {agent_name}")
        lines.append(f"Task description: {task_description}")

        # Similar past tasks (from DB summaries)
        if similar_tasks:
            lines.append("")
            lines.append("=== Similar Past Tasks ===")
            for hit in similar_tasks:
                # A compact line: status + summary
                status_str = hit.get("status") or "Unknown"
                lines.append(
                    f"- [status={status_str}] {hit.get('summary', '')}"
                )

        # Final instruction
        lines.append("")
        lines.append("=== Instruction ===")
        lines.append(
            "Using the patterns from the Ground-Truth Trajectories above, the "
            "current User Question, the Target Task, the DAG Prefix (if any), "
            "and the Similar Past Tasks, predict the output that the agent "
            "should produce for the Target Task. "
            "Respond ONLY with that output, without explanations or JSON."
        )

        return "\n".join(lines)


# If SimulatorAgent is in the same file, you don't need this import.
# If it's in another module, adjust the import path accordingly:
# from simulator_agent import SimulatorAgent


SIMULATOR_SYSTEM_PROMPT = """
You are a simulator agent. Given:
- a user question,
- an optional DAG prefix (previous steps),
- a target task (agent + description),
- and a few similar past tasks with their summaries,

you must PREDICT the output that the agent will produce for this target task.
Do NOT explain your reasoning. Respond ONLY with the predicted output.
""".strip()

def _make_dummy_dag_prefix() -> Sequence[Any]:
    """
    Build a minimal DAG prefix from the provided 'truncated_plan' text.

    Constraint: Use only functions/methods already appearing in this file.
    Therefore, we do inline parsing with basic string operations (no new helpers).
    """
    example = {
        "name": "Example A (expected: Not accomplished)",
        "user_question": "What assets can be found at the MAIN site?",
        "stop_index": 1,
        "truncated_plan": (
            "#Task1: List the assets available at SiteX.\n"
            "#Agent1: IoT Data Download\n"
            "#Dependency1: None\n"
            "#ExpectedOutput1: A list of assets at SiteX.\n"
        ),
        "candidate_answer": "",
        "expected_status": "Not accomplished",
        "expected_can_answer_now": False,
    }

    plan_text = example["truncated_plan"]

    step: dict = {}
    for line in plan_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#Task1:"):
            step["task"] = line.split(":", 1)[1].strip()
        elif line.startswith("#Agent1:"):
            step["agent"] = line.split(":", 1)[1].strip()
        elif line.startswith("#Dependency1:"):
            step["dependency"] = line.split(":", 1)[1].strip()
        elif line.startswith("#ExpectedOutput1:"):
            step["expected_output"] = line.split(":", 1)[1].strip()

    # Return as a list of dict steps (allowed by downstream code/comments).
    return [step]


def main() -> None:
    # 1) Load DB URL from environment (standard behavior of os.getenv()).
    db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DB_URL / DATABASE_URL is not set in the environment.")

    # 2) Use the provided Example A as the test case
    example =         {
            "name": "Example B (expected: Partially accomplished)",
            "user_question": "What assets can be found at the MAIN site?",
            "stop_index": 1,
            "truncated_plan": (
                "#Task1: List the assets available at the MAIN site.\n"
                "#Agent1: IoT Data Download\n"
                "#Dependency1: None\n"
                "#ExpectedOutput1: A list of assets at the MAIN site.\n"
            ),
            # Not provided by the user; placeholder (replace with Simulator output)
            "candidate_answer": "",
            "expected_status": "Partially accomplished",
            "expected_can_answer_now": True,
        }

    user_question = example["user_question"]
    dag_prefix = _make_dummy_dag_prefix()

    # 3) Derive agent_name and task_description from the parsed prefix (no new helper funcs)
    agent_name = ""
    task_description = ""
    if dag_prefix and isinstance(dag_prefix[0], dict):
        agent_name = str(dag_prefix[0].get("agent", "")).strip()
        task_description = str(dag_prefix[0].get("task", "")).strip()

    # Fallbacks (should not trigger for Example A)
    if not agent_name:
        agent_name = "IoT Data Download"
    if not task_description:
        task_description = "List assets using the IoT Data Download agent."

    print("=== SimulatorAgent manual test (Example A) ===")
    print(f"DB_URL: {db_url}")
    print(f"Test case: {example.get('name','')}")
    print(f"User question: {user_question}")
    print(f"Agent name: {agent_name}")
    print(f"Task description: {task_description}")
    print("DAG prefix:")
    for s in dag_prefix:
        print("  ", s)
    print()

    # 4) Instantiate the SimulatorAgent
    sim = SimulatorAgent(
        db_url=db_url,
        system_prompt=SIMULATOR_SYSTEM_PROMPT,
        max_similar_tasks=5,
    )

    # 5) (Optional) directly test search_task_summaries before full run
    print(">> Running search_task_summaries()...")
    query_text = sim._build_query_text(
        user_question=user_question,
        task_description=task_description,
        agent_name=agent_name,
    )
    hits = sim.search_task_summaries(
        query_text=query_text,
        top_k=5,
        agent_names=[agent_name],
        status_filter=["Accomplished"],
    )
    print(f"Found {len(hits)} similar tasks.")
    for h in hits[:3]:
        print(
            f"  - doc_id={h['doc_id']} task_id={h['task_id']} "
            f"status={h.get('status')} score={h['similarity_score']:.3f}"
        )
        print(f"    summary={h['summary'][:200]}{'...' if len(h['summary']) > 200 else ''}")
    print()

    # 6) Run the full simulation (search + LLM)
    print(">> Running sim.run(...) to get predicted output...")
    predicted_output = sim.run(
        user_question=user_question,
        task_description=task_description,
        agent_name=agent_name,
        dag_prefix=dag_prefix,
    )

    print("\n=== Predicted Output ===")
    print(predicted_output)
    print("=== End ===")


if __name__ == "__main__":
    main()
