import json
from collections import deque
from typing import Any, Dict, List, Optional

from pydantic import Field

from agent_hive.enum import ContextType
from agent_hive.workflows.base_workflow import Workflow
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ConditionalWorkflow(Workflow):
    """
    Minimal conditional executor aligned with:
    - planner-produced conditional DAG
    - SequentialWorkflow-style logging

    Main principles:
    - store one response string per executed node in self.memory
    - store one log object per executed node in self.logs
    - for normal nodes, logs are agent.agent_executor.trajectory if available
    - if response is empty, fall back to logs["final_answer"]
    - no recovery / no replanning
    """

    context_type: ContextType = Field(
        default=ContextType.DISABLED,
        description="Type of context to use."
    )

    def __init__(
        self,
        plan: Dict[str, Any],
        verification_agent: Any,
        user_q: str = "",
        context_type: ContextType = ContextType.SELECTED,
        max_steps: int = 100,
    ):
        self.plan = plan
        self.verification_agent = verification_agent
        self.user_query = user_q
        self.context_type = context_type
        self.max_steps = max_steps

        self.answer_contract = plan["answer_contract"]
        self.tasks = plan["tasks"]
        self.node_map = plan["node_map"]
        self.start_node_id = plan["start_node_id"]
        self.available_agents = plan.get("available_agents", [])

        self.memory: List[Any] = []
        self.logs: List[Any] = []
        self.executed_node_ids: List[str] = []

        self.step_output: Dict[str, Any] = {}
        self.step_payload: Dict[str, Any] = {}

        self._verify_plan()

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    def _verify_plan(self) -> None:
        if not isinstance(self.plan, dict):
            raise ValueError("plan must be a dict")

        if not isinstance(self.answer_contract, str) or not self.answer_contract.strip():
            raise ValueError("plan['answer_contract'] must be a non-empty string")

        if not isinstance(self.tasks, list) or len(self.tasks) == 0:
            raise ValueError("plan['tasks'] must be a non-empty list")

        if not isinstance(self.node_map, dict) or len(self.node_map) == 0:
            raise ValueError("plan['node_map'] must be a non-empty dict")

        if not isinstance(self.start_node_id, str) or self.start_node_id not in self.node_map:
            raise ValueError("plan['start_node_id'] must point to a known node")

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _get_task_agent_name(self, task: Any) -> str:
        return str(getattr(task, "agent_name", getattr(task, "agent", "")))


    def _get_task_node_id(self, task: Any) -> str:
        return str(getattr(task, "node_id", getattr(task, "id", "")))


    def _is_verifier_node(self, task: Any) -> bool:
        agent_name = self._get_task_agent_name(task)
        node_id = self._get_task_node_id(task)
        return agent_name == "verifier" or node_id.startswith("V_")


    def _is_recovery_node(self, task: Any) -> bool:
        agent_name = self._get_task_agent_name(task)
        node_id = self._get_task_node_id(task)
        return agent_name == "recovery" or node_id.startswith("R_")


    def _is_normal_node(self, task: Any) -> bool:
        return not self._is_verifier_node(task) and not self._is_recovery_node(task)

    def _resolve_agent_for_task(self, task: Any) -> Any:
        task_agent_name = str(getattr(task, "agent_name", getattr(task, "agent", "")))
        node_id = str(getattr(task, "node_id", getattr(task, "id", "?")))

        if self._is_verifier_node(task):
            raise ValueError(
                f"_resolve_agent_for_task should not be called on verifier node '{node_id}'"
            )

        if self._is_recovery_node(task):
            raise ValueError(
                f"_resolve_agent_for_task should not be called on recovery node '{node_id}'. "
                "Resolve the subject normal node's agent instead."
            )

        if "available_agents" in self.plan and isinstance(self.plan["available_agents"], list):
            for agent in self.plan["available_agents"]:
                if getattr(agent, "name", None) == task_agent_name:
                    return agent

        raise ValueError(
            f"Could not resolve agent for node '{node_id}' "
            f"with agent_name='{task_agent_name}'"
        )

    def _build_user_input(self, task: Any) -> str:
        base_text = getattr(task, "task", "")
        deps = getattr(task, "deps", []) or []

        if self.context_type == ContextType.DISABLED or len(deps) == 0:
            return base_text

        dep_chunks: List[str] = []
        for dep_id in deps:
            if dep_id in self.step_output:
                dep_val = self.step_output[dep_id]
                if isinstance(dep_val, str):
                    dep_chunks.append(f"[{dep_id}]\n{dep_val}")
                else:
                    dep_chunks.append(f"[{dep_id}]\n{json.dumps(dep_val, ensure_ascii=False, indent=2)}")

        if self.context_type == ContextType.PREVIOUS:
            if len(dep_chunks) == 0:
                return base_text
            return f"{base_text}\n\nContext:\n{dep_chunks[-1]}"

        if self.context_type in {ContextType.ALL, ContextType.SELECTED}:
            context_text = "\n\n".join(dep_chunks)
            if not context_text:
                return base_text
            return f"{base_text}\n\nContext:\n{context_text}"

        return base_text

    def _extract_agent_logs(self, agent: Any) -> Any:
        """
        SequentialWorkflow-style:
        if agent.agent_executor exists, store agent.agent_executor.trajectory.
        otherwise store "".
        """
        if agent is None:
            return ""

        agent_executor = getattr(agent, "agent_executor", None)
        if agent_executor is None:
            return ""

        return getattr(agent_executor, "trajectory", "")

    def _normalize_response(self, response: Any) -> Any:
        if isinstance(response, str):
            return response.split("Final Answer:")[0].strip()
        return response

    def _get_branches(self, task: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for b in getattr(task, "branches", []) or []:
            if hasattr(b, "to_dict"):
                d = b.to_dict()
            elif isinstance(b, dict):
                d = dict(b)
            else:
                d = {
                    "expect": getattr(b, "expect", None),
                    "next": getattr(b, "next", None),
                }

            label = d.get("label", d.get("expect"))
            out.append(
                {
                    "label": str(label) if label is not None else "",
                    "next": str(d.get("next", "")),
                }
            )
        return out

    def _is_soft_dep(self, dep_id: str) -> bool:
        return isinstance(dep_id, str) and dep_id.startswith("B_")


    def _get_required_deps(self, task: Any) -> List[str]:
        deps = getattr(task, "deps", []) or []
        return [dep for dep in deps if not self._is_soft_dep(dep)]


    def _is_node_ready(self, node_id: str) -> bool:
        if node_id == "TERMINATE":
            return False
        if node_id not in self.node_map:
            return False
        if node_id in self.executed_node_ids:
            return False

        task = self.node_map[node_id]
        required_deps = self._get_required_deps(task)
        executed = set(self.executed_node_ids)
        return all(dep in executed for dep in required_deps)


    def _all_ready_nodes(self) -> List[str]:
        ready: List[str] = []
        for node_id in self.node_map.keys():
            if self._is_node_ready(node_id):
                ready.append(node_id)
        return ready


    def _enqueue_if_ready(
        self,
        node_id: Optional[str],
        queue: deque[str],
        queued: set[str],
    ) -> None:
        if node_id is None or node_id == "TERMINATE":
            return
        if node_id in queued:
            return
        if node_id in self.executed_node_ids:
            return
        if self._is_node_ready(node_id):
            queue.append(node_id)
            queued.add(node_id)


    def _enqueue_all_ready_nodes(
        self,
        queue: deque[str],
        queued: set[str],
    ) -> None:
        for node_id in self._all_ready_nodes():
            if node_id not in queued and node_id not in self.executed_node_ids:
                queue.append(node_id)
                queued.add(node_id)

    # ---------------------------------------------------------
    # Node execution
    # ---------------------------------------------------------

    def _execute_normal_node(self, task: Any) -> Dict[str, Any]:
        node_id = getattr(task, "node_id", None)
        if node_id is None:
            raise ValueError("Task has no node_id")

        agent = self._resolve_agent_for_task(task)
        user_input = self._build_user_input(task)

        response = agent.execute_task(user_input)
        response = self._normalize_response(response)
        logs = self._extract_agent_logs(agent)

        result = {
            "node_id": node_id,
            "task_description": getattr(task, "task", ""),
            "agent_name": getattr(agent, "name", getattr(task, "agent_name", "UNKNOWN_AGENT")),
            "response": response,
            "logs": logs,
        }

        if hasattr(task, "mark_executed"):
            task.mark_executed(result)

        self.memory.append(response)
        self.logs.append(logs)
        self.executed_node_ids.append(node_id)

        # normal downstream context 用: 文字列 response を保持
        self.step_output[node_id] = response

        # verifier / recovery 用: response と logs をまとめて保持
        self.step_payload[node_id] = {
            "response": response,
            "logs": logs,
            "task_description": getattr(task, "task", ""),
            "agent_name": getattr(agent, "name", getattr(task, "agent_name", "UNKNOWN_AGENT")),
        }

        return result

    def _execute_verifier_node(self, task: Any) -> Dict[str, Any]:
        node_id = getattr(task, "node_id", None)
        if node_id is None:
            raise ValueError("Verifier node has no node_id")

        deps = getattr(task, "deps", []) or []
        if len(deps) != 1:
            raise ValueError(f"Verifier node '{node_id}' must have exactly one parent dep")

        parent_id = deps[0]
        if parent_id not in self.node_map:
            raise ValueError(f"Unknown verifier parent '{parent_id}'")

        parent_task = self.node_map[parent_id]

        # parent normal node の保存形式:
        # self.step_payload[parent_id] = {
        #     "response": ...,
        #     "logs": ...,
        #     ...
        # }
        parent_payload = self.step_payload.get(parent_id, {})
        if not isinstance(parent_payload, dict):
            parent_payload = {}

        parent_response = parent_payload.get("response", self.step_output.get(parent_id, ""))
        parent_logs = parent_payload.get("logs", {})

        # verify() は logs 引数しか受けないので、
        # 親 response は logs["final_answer"] に埋め込んで渡す
        if isinstance(parent_logs, dict):
            verify_logs_input = dict(parent_logs)
        else:
            verify_logs_input = {}

        if "final_answer" not in verify_logs_input:
            verify_logs_input["final_answer"] = parent_response
        if "parent_response" not in verify_logs_input:
            verify_logs_input["parent_response"] = parent_response

        verify_result = self.verification_agent.verify(
            node={
                "id": getattr(parent_task, "node_id", getattr(parent_task, "id", "")),
                "task": getattr(parent_task, "task", ""),
                "agent": getattr(parent_task, "agent_name", getattr(parent_task, "agent", "")),
                "deps": getattr(parent_task, "deps", []) or [],
                "node_contract": getattr(parent_task, "node_contract", ""),
            },
            logs=verify_logs_input,
        )

        if not isinstance(verify_result, dict):
            raise ValueError(
                f"Verifier node '{node_id}' must return a dict, got {type(verify_result).__name__}"
            )

        # 新しい verify() の返り値 schema:
        # {
        #   "diagnosis": {...},
        #   "recovery_suggestion": {...},
        #   "verification_logs": {...}
        # }
        diagnosis = verify_result.get("diagnosis", {})
        recovery_suggestion = verify_result.get("recovery_suggestion", {})
        verification_logs = verify_result.get("verification_logs", {})

        if not isinstance(diagnosis, dict):
            raise ValueError(
                f"Verifier node '{node_id}' returned non-dict diagnosis"
            )
        if not isinstance(recovery_suggestion, dict):
            raise ValueError(
                f"Verifier node '{node_id}' returned non-dict recovery_suggestion"
            )
        if not isinstance(verification_logs, dict):
            raise ValueError(
                f"Verifier node '{node_id}' returned non-dict verification_logs"
            )

        # downstream recovery node が使う本体
        response = {
            "diagnosis": diagnosis,
            "recovery_suggestion": recovery_suggestion,
        }

        # verifier node 自体の logs
        logs = {
            "final_answer": response,
            "verification_logs": verification_logs,
            "_workflow": {
                "kind": "verifier",
                "node_id": node_id,
                "parent_id": parent_id,
            },
        }

        if hasattr(task, "mark_executed"):
            task.mark_executed(response)

        self.memory.append(response)
        self.logs.append(logs)
        self.executed_node_ids.append(node_id)

        # 通常の downstream context 用
        self.step_output[node_id] = response

        # recovery node が response と logs の両方を読めるように保持
        self.step_payload[node_id] = {
            "response": response,
            "logs": logs,
            "parent_id": parent_id,
            "parent_response": parent_response,
            "parent_logs": verify_logs_input,
        }

        return {
            "node_id": node_id,
            "task_description": getattr(task, "task", ""),
            "agent_name": "verifier",
            "response": response,
            "logs": logs,
        }

    def _execute_recovery_node(self, task: Any) -> Dict[str, Any]:
        node_id = getattr(task, "node_id", getattr(task, "id", None))
        if node_id is None:
            raise ValueError("Recovery node has no node_id")

        subject_node_id = self._get_subject_node_id_from_recovery_node(task)

        if subject_node_id not in self.node_map:
            raise ValueError(
                f"Recovery node '{node_id}' refers to unknown subject node '{subject_node_id}'"
            )

        subject_task = self.node_map[subject_node_id]

        verifier_dep, upstream_dep_ids = self._split_recovery_deps(task)

        if verifier_dep not in self.step_payload:
            raise ValueError(
                f"Recovery node '{node_id}' cannot find verifier payload for '{verifier_dep}'"
            )

        verifier_payload = self.step_payload.get(verifier_dep, {})
        if not isinstance(verifier_payload, dict):
            raise ValueError(
                f"Recovery node '{node_id}' got non-dict verifier payload for '{verifier_dep}'"
            )

        verification_response = verifier_payload.get("response", {})
        verification_logs = verifier_payload.get("logs", {})

        if not isinstance(verification_response, dict):
            raise ValueError(
                f"Recovery node '{node_id}' got non-dict verification response from '{verifier_dep}'"
            )

        diagnosis = verification_response.get("diagnosis", {})
        recovery_suggestion = verification_response.get("recovery_suggestion", {})

        if not isinstance(diagnosis, dict):
            raise ValueError(
                f"Recovery node '{node_id}' got non-dict diagnosis from '{verifier_dep}'"
            )
        if not isinstance(recovery_suggestion, dict):
            raise ValueError(
                f"Recovery node '{node_id}' got non-dict recovery_suggestion from '{verifier_dep}'"
            )

        subject_previous_payload = self.step_payload.get(subject_node_id, {})
        if not isinstance(subject_previous_payload, dict):
            subject_previous_payload = {}

        subject_previous_response = subject_previous_payload.get(
            "response",
            self.step_output.get(subject_node_id, ""),
        )
        subject_previous_logs = subject_previous_payload.get("logs", {})

        agent = self._resolve_agent_for_task(subject_task)

        user_input = self._build_recovery_user_input(
            subject_task=subject_task,
            upstream_dep_ids=upstream_dep_ids,
            verification_response=verification_response,
            subject_previous_response=subject_previous_response,
        )

        response = agent.execute_task(user_input)
        response = self._normalize_response(response)
        logs = self._extract_agent_logs(agent)

        result = {
            "node_id": node_id,
            "task_description": getattr(subject_task, "task", ""),
            "agent_name": getattr(
                agent,
                "name",
                getattr(subject_task, "agent_name", getattr(subject_task, "agent", "UNKNOWN_AGENT")),
            ),
            "response": response,
            "logs": logs,
        }

        if hasattr(task, "mark_executed"):
            task.mark_executed(response)

        self.memory.append(response)
        self.logs.append(logs)
        self.executed_node_ids.append(node_id)

        self.step_output[node_id] = response

        self.step_payload[node_id] = {
            "response": response,
            "logs": logs,
            "subject_node_id": subject_node_id,
            "verifier_dep": verifier_dep,
            "upstream_dep_ids": upstream_dep_ids,
            "verification_response": verification_response,
            "verification_logs": verification_logs,
            "subject_previous_response": subject_previous_response,
            "subject_previous_logs": subject_previous_logs,
            "_workflow": {
                "kind": "recovery",
                "node_id": node_id,
                "subject_node_id": subject_node_id,
            },
        }

        return result

    def _execute_node(self, task: Any) -> Dict[str, Any]:
        if self._is_verifier_node(task):
            return self._execute_verifier_node(task)

        if self._is_recovery_node(task):
            return self._execute_recovery_node(task)

        return self._execute_normal_node(task)

    # ---------------------------------------------------------
    # Routing
    # ---------------------------------------------------------

    def _route_verifier_node(self, task: Any) -> Optional[str]:
        node_id = getattr(task, "node_id", None)
        if node_id is None:
            raise ValueError("Verifier node has no node_id")

        payload = self.step_payload.get(node_id)
        if not isinstance(payload, dict):
            raise ValueError(f"Verifier node '{node_id}' returned non-dict payload")

        label = payload.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Verifier node '{node_id}' returned invalid label")

        for b in self._get_branches(task):
            if b["label"] == label:
                return b["next"]

        raise ValueError(
            f"Verifier node '{node_id}' returned label '{label}', "
            f"but no matching outgoing branch exists"
        )

    def _route_regular_node_with_branches(self, task: Any) -> Optional[str]:
        branches = self._get_branches(task)
        by_label = {b["label"]: b["next"] for b in branches}

        if "success" in by_label:
            return by_label["success"]
        if "terminate" in by_label:
            return by_label["terminate"]
        if len(branches) == 1:
            return branches[0]["next"]

        raise ValueError(
            f"Cannot route node '{getattr(task, 'node_id', '')}' with branches={branches}"
        )


    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _get_explicit_targets(self, task: Any) -> List[str]:
        """
        No explicit routing.
        Execution order is determined only by dependency readiness.
        """
        return []
    
    def _get_subject_node_id_from_recovery_node(self, task: Any) -> str:
        node_id = getattr(task, "node_id", getattr(task, "id", ""))
        if not isinstance(node_id, str) or not node_id.startswith("R_"):
            raise ValueError(f"Recovery node id must start with 'R_': {node_id}")
        return node_id.split("R_", 1)[1]


    def _split_recovery_deps(self, task: Any) -> tuple[str, list[str]]:
        """
        Recovery deps are:
        deps(R_Sx) = rewritten upstream final deps + [V_Sx]

        Returns:
        verifier_dep: "V_Sx"
        upstream_dep_ids: rewritten upstream final deps such as ["R_S1", ...]
        """
        node_id = getattr(task, "node_id", getattr(task, "id", ""))
        if not isinstance(node_id, str) or not node_id.startswith("R_"):
            raise ValueError(f"Recovery node id must start with 'R_': {node_id}")

        subject_node_id = node_id.split("R_", 1)[1]
        expected_verifier_id = f"V_{subject_node_id}"

        deps = getattr(task, "deps", []) or []
        if not isinstance(deps, list):
            raise ValueError(f"Recovery node '{node_id}' has invalid deps: {deps}")

        verifier_deps = [d for d in deps if d == expected_verifier_id]
        upstream_deps = [d for d in deps if d != expected_verifier_id]

        if len(verifier_deps) != 1:
            raise ValueError(
                f"Recovery node '{node_id}' must have exactly one verifier dep "
                f"'{expected_verifier_id}', but got deps={deps}"
            )

        return verifier_deps[0], upstream_deps


    def _serialize_context_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)


    def _build_recovery_user_input(
        self,
        subject_task: Any,
        upstream_dep_ids: list[str],
        verification_response: dict,
        subject_previous_response: Any,
    ) -> str:
        """
        Build recovery-aware input for re-executing the subject task Sx.

        Inputs are separated into:
        1. upstream final outputs needed to execute the task
        2. verification diagnosis / recovery suggestion describing how to repair
        3. optional previous failed response for direct comparison
        """
        base_text = getattr(subject_task, "task", "")

        dep_chunks: list[str] = []
        for dep_id in upstream_dep_ids:
            dep_val = self.step_output.get(dep_id, "")
            dep_chunks.append(f"[{dep_id}]\n{self._serialize_context_value(dep_val)}")

        upstream_context = "\n\n".join(dep_chunks).strip()

        diagnosis = verification_response.get("diagnosis", {})
        recovery_suggestion = verification_response.get("recovery_suggestion", {})

        parts: list[str] = [base_text]

        if upstream_context:
            parts.append(f"Upstream context:\n{upstream_context}")

        if subject_previous_response not in ("", None, {}):
            parts.append(
                "Previous execution result:\n"
                f"{self._serialize_context_value(subject_previous_response)}"
            )

        parts.append(
            "Previous execution diagnosis:\n"
            f"{self._serialize_context_value(diagnosis)}"
        )
        parts.append(
            "Recovery guidance:\n"
            f"{self._serialize_context_value(recovery_suggestion)}"
        )
        parts.append(
            "Please execute the same task again, correcting the previous failure "
            "based on the diagnosis and recovery guidance."
        )

        return "\n\n".join(parts)

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    def generate_history(self) -> List[Dict[str, Any]]:
        history = []

        for i, node_id in enumerate(self.executed_node_ids):
            task = self.node_map[node_id]
            response = self.memory[i]
            logs = self.logs[i]

            if (response is None or response == "") and isinstance(logs, dict):
                response = logs["final_answer"] if "final_answer" in logs else ""

            if self._is_verifier_node(task):
                agent_name = "verifier"
            elif self._is_recovery_node(task):
                agent_name = "recovery"
            else:
                agent_name = getattr(task, "agent_name", getattr(task, "agent", "UNKNOWN_AGENT"))

            history.append(
                {
                    "task_number": i + 1,
                    "node_id": node_id,
                    "task_description": getattr(task, "task", ""),
                    "agent_name": agent_name,
                    "response": response,
                    "logs": logs,
                }
            )

        return history

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        self.memory = []
        self.logs = []
        self.executed_node_ids = []
        self.step_output = {}
        self.step_payload = {}

        steps = 0
        terminated = False

        queue: deque[str] = deque()
        queued: set[str] = set()

        # seed with all currently ready roots
        self._enqueue_all_ready_nodes(queue, queued)

        # fallback: if nothing was enqueued, at least try the declared start node
        if not queue:
            self._enqueue_if_ready(self.start_node_id, queue, queued)

        while queue:
            if steps >= self.max_steps:
                raise RuntimeError(f"Exceeded max_steps={self.max_steps}")

            current_node_id = queue.popleft()
            queued.discard(current_node_id)

            if current_node_id == "TERMINATE":
                terminated = True
                break

            if current_node_id not in self.node_map:
                raise ValueError(f"Unknown node id during execution: {current_node_id}")

            # queue に入った後で他のノード進行により状態が変わる場合に備える
            if current_node_id in self.executed_node_ids:
                continue
            if not self._is_node_ready(current_node_id):
                continue

            task = self.node_map[current_node_id]
            logger.info(f"Conditional node {steps+1}: {getattr(task, 'task', '')}")

            self._execute_node(task)
            steps += 1

            # dependency-based DAG progress only
            self._enqueue_all_ready_nodes(queue, queued)

        # stalled DAG check: nothing left in queue, but some nodes remain unexecuted
        if not terminated:
            remaining = [
                node_id for node_id in self.node_map.keys()
                if node_id not in self.executed_node_ids
            ]
            still_ready = [node_id for node_id in remaining if self._is_node_ready(node_id)]

            if still_ready:
                raise RuntimeError(
                    f"Execution ended with ready-but-unexecuted nodes: {still_ready}"
                )

        history = self.generate_history()
        print(json.dumps(history, indent=4, ensure_ascii=False, default=str))
        return history