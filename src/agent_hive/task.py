from typing import List, Optional

from pydantic import Field

from agent_hive.agents.base_agent import BaseAgent


class Task:
    description: str = Field(description="Description of the actual task.")
    agents: List[BaseAgent] = Field(description="Agents responsible for execution the task.")
    expected_output: Optional[str] = Field(default=None,
                                           description="Clear definition of expected output for the task.")
    context: Optional[List["Task"]] = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=None,
    )

    def __init__(self, description: str, agents: List[BaseAgent], expected_output: Optional[str] = None,
                 context: Optional[List['Task']] = None):
        self.description = description
        self.agents = agents
        self.expected_output = expected_output
        self.context = context

    def __str__(self):
        return f"Task(description={self.description}, agents={self.agents}, expected_output={self.expected_output}, context={self.context})"


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConditionalBranch:
    """
    One outgoing branch from a conditional plan node.
    """
    expect: str
    next: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalBranch":
        if not isinstance(data, dict):
            raise TypeError("Each branch must be a dict.")

        expect = data.get("expect", "")
        next_node = data.get("next", "")

        if not isinstance(expect, str) or not expect.strip():
            raise ValueError("Branch field 'expect' must be a non-empty string.")
        if not isinstance(next_node, str) or not next_node.strip():
            raise ValueError("Branch field 'next' must be a non-empty string.")

        return cls(
            expect=expect.strip(),
            next=next_node.strip(),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "expect": self.expect,
            "next": self.next,
        }


@dataclass
class ConditionalTask:
    """
    One node in the minimal conditional plan.

    This corresponds to a node with fields:
    - id
    - task
    - agent
    - deps
    - branches
    """
    node_id: str
    task: str
    agent_name: str
    deps: List[str] = field(default_factory=list)
    branches: List[ConditionalBranch] = field(default_factory=list)

    # runtime fields
    output: Optional[str] = None
    executed: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalTask":
        if not isinstance(data, dict):
            raise TypeError("Node must be a dict.")

        node_id = data.get("id", "")
        task = data.get("task", "")
        agent_name = data.get("agent", "")
        deps = data.get("deps", [])
        branches_raw = data.get("branches", [])

        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("Node field 'id' must be a non-empty string.")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Node field 'task' must be a non-empty string.")
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Node field 'agent' must be a non-empty string.")
        if deps is None:
            deps = []
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ValueError("Node field 'deps' must be a list of strings.")
        if branches_raw is None:
            branches_raw = []
        if not isinstance(branches_raw, list):
            raise ValueError("Node field 'branches' must be a list.")

        branches = [ConditionalBranch.from_dict(b) for b in branches_raw]

        return cls(
            node_id=node_id.strip(),
            task=task.strip(),
            agent_name=agent_name.strip(),
            deps=[d.strip() for d in deps if isinstance(d, str) and d.strip()],
            branches=branches,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "task": self.task,
            "agent": self.agent_name,
            "deps": list(self.deps),
            "branches": [b.to_dict() for b in self.branches],
        }

    def validate(self) -> None:
        if not self.node_id:
            raise ValueError("ConditionalTask.node_id is empty.")
        if not self.task:
            raise ValueError(f"ConditionalTask[{self.node_id}] has empty task.")
        if not self.agent_name:
            raise ValueError(f"ConditionalTask[{self.node_id}] has empty agent_name.")

        # seen_next = set()
        # for branch in self.branches:
        #     if branch.next in seen_next:
        #         raise ValueError(
        #             f"ConditionalTask[{self.node_id}] has duplicate branch target: {branch.next}"
        #         )
        #     seen_next.add(branch.next)

    def resolve_agent(self, available_agents: List[Any]) -> Any:
        """
        Find the concrete agent object whose name matches self.agent_name.
        Fallback is NOT done here; mismatch is treated as an error.
        """
        for agent in available_agents:
            if getattr(agent, "name", None) == self.agent_name:
                return agent
        raise ValueError(
            f"Agent '{self.agent_name}' for node '{self.node_id}' was not found."
        )

    def build_context_text(self, completed_outputs: Dict[str, str]) -> str:
        """
        Build a simple text context from dependency outputs.
        Only executed dependency nodes are included.
        """
        if not self.deps:
            return ""

        chunks: List[str] = []
        for dep_id in self.deps:
            if dep_id in completed_outputs:
                chunks.append(f"[{dep_id}]\n{completed_outputs[dep_id]}")

        return "\n\n".join(chunks)

    def build_user_input(self, completed_outputs: Dict[str, str]) -> str:
        """
        Build the input text passed to the assigned agent at execution time.
        """
        context_text = self.build_context_text(completed_outputs)

        if not context_text:
            return self.task

        return f"{self.task}\n\nContext:\n{context_text}"

    def mark_executed(self, output: str) -> None:
        self.executed = True
        self.output = output

    def candidate_next_nodes(self) -> List[str]:
        return [b.next for b in self.branches]

    def has_branching(self) -> bool:
        return len(self.branches) > 0