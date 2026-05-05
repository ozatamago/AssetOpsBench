import re
from typing import List, Optional

from langchain.tools import BaseTool
from reactxen.agents.rafa.agents import RAFAAgent
from reactxen.agents.react.prompts.fewshots import MPE_SIMPLE4

from agent_hive.agents.base_agent import BaseAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ReactReflectAgent(BaseAgent):
    """
    Wrapper around ReActXen RAFAAgent.
    Keeps the external Agent Hive interface similar to the old wrapper.
    """
    few_shots: Optional[str] = None
    task_examples: Optional[List[str]] = None

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[BaseTool],
        llm: str,
        few_shots: str = MPE_SIMPLE4,
        task_examples: Optional[List[str]] = None,
        reflect_step: int = 5,
        search_width: int = 2,
        search_depth: int = 1,
        max_steps: int = 6,
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.llm = llm
        self.memory = []
        self.few_shots = few_shots

        # RAFA uses run(..., k=...) rather than num_reflect_iteration
        self.reflect_step = reflect_step
        self.search_width = search_width
        self.search_depth = search_depth
        self.max_steps = max_steps

        if task_examples:
            self.task_examples = task_examples
        else:
            self.task_examples = re.findall(r"^Question:(.*)$", self.few_shots, re.MULTILINE)
            self.task_examples = [ex.strip() for ex in self.task_examples]

        self.agent_executor: Optional[RAFAAgent] = None

    def execute_task(self, user_input):
        logger.info(
            "RAFAAgent is executing task: %s, with Tools %s",
            user_input,
            self.tools,
        )

        self.agent_executor = RAFAAgent(
            question=user_input,
            key="",
            cbm_tools=self.tools,
            max_steps=self.max_steps,
            react_llm_model_id=self.llm,
            react_example=self.few_shots,
            log_structured_messages=True,
            handle_context_length_overflow=True,
            apply_loop_detection_check=True,
            early_stop=True,
        )

        # RAFAAgent.run() mutates internal state and does not return the final answer
        self.agent_executor.run(
            reset=True,
            name=self.name,
            k=self.reflect_step,
            search_width=self.search_width,
            search_depth=self.search_depth,
        )

        answer = self.agent_executor.answer
        trajectory = self.agent_executor.export_trajectory()

        # Optional: attach review text if your downstream code uses it
        if isinstance(trajectory, dict):
            trajectory["review_str"] = getattr(self.agent_executor, "review_str", "")

        print(f"type(answer): {type(answer)}", flush=True)
        print(f"answer: {answer}", flush=True)

        print(f"type(trajectory): {type(trajectory)}", flush=True)
        print(f"trajectory: {trajectory}", flush=True)

        return answer, trajectory