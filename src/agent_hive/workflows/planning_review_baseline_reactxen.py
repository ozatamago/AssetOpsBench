from agent_hive.task import Task
from pydantic import Field
from typing import List
from agent_hive.enum import ContextType
import json
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.logger import get_custom_logger
import time

logger = get_custom_logger(__name__)

import json
from typing import Any, Dict, List, Optional, Set


def load_conditional_plan(
    plan_source: Any,
    allowed_special_next: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Load a minimal conditional plan from JSON text or dict.

    Expected format:
    {
      "answer_contract": "...",
      "nodes": [
        {
          "id": "S1",
          "task": "...",
          "agent": "...",
          "deps": [],
          "branches": [
            {"expect": "...", "next": "S2"}
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
            Optional special transition labels allowed in branch.next
            in addition to real node IDs, e.g. {"TERMINATE"}.
            By default, no special labels are allowed.

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
        allowed_special_next = set()

    # ---------------------------------------------------------
    # 1. Parse top-level JSON
    # ---------------------------------------------------------
    if isinstance(plan_source, str):
        text = plan_source.strip()
        if not text:
            raise ValueError("Plan text is empty.")

        try:
            raw_plan = json.loads(text)
        except json.JSONDecodeError:
            # optional fallback: extract the outermost JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Could not find a valid JSON object in plan text.")
            candidate = text[start:end + 1]
            raw_plan = json.loads(candidate)

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
    # 3. Build ConditionalTask objects
    # ---------------------------------------------------------
    tasks: List[ConditionalTask] = []
    node_map: Dict[str, ConditionalTask] = {}

    for i, node_data in enumerate(nodes_raw):
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
    # 4. Validate deps and branch targets
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
    # 5. Infer start node
    # ---------------------------------------------------------
    # This is an implementation convention, because the minimal format
    # itself does not explicitly define "start_node".
    no_dep_nodes = [t.node_id for t in tasks if len(t.deps) == 0]

    if len(no_dep_nodes) == 1:
        start_node_id = no_dep_nodes[0]
    elif len(tasks) > 0:
        # deterministic fallback: first node in the list
        start_node_id = tasks[0].node_id
    else:
        raise ValueError("Plan has no executable nodes.")

    # ---------------------------------------------------------
    # 6. Return normalized plan object
    # ---------------------------------------------------------
    return {
        "answer_contract": answer_contract.strip(),
        "tasks": tasks,
        "node_map": node_map,
        "start_node_id": start_node_id,
    }


class PlanningReviewWorkflow(Workflow):
    """
    This class represents a planning review workflow, where the (parent) task is decomposed into a list of subtasks.
    The workflow is designed to review the planning process and ensure that the plan is correct and feasible,
     all subtasks are correctly defined and executed.

    Example:
        agent1 = ...
        agent2 = ...

        task = Task(..., agents=[agent1, agent2], ...)

        workflow = PlanningReviewWorkflow(task=[task1], ...)
        workflow.run()
    """

    llm: str = Field(description="LLM used by the task planning.")

    def __init__(self, tasks: List[Task], llm: str):
        self.tasks = tasks
        self.memory = []
        self.max_memory = 10
        self.llm = llm
        self.max_retries = 3
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        if len(self.tasks) != 1:
            raise ValueError("Planning only supports one task")
        task = self.tasks[0]
        if task.agents is None or len(task.agents) < 1:
            raise ValueError("Task must have at least one agent")

    def run(self, save_plan=False, saved_plan_prefix="", qid=None, enable_summarization=False):
        generated_steps, input_tokens_count, generated_tokens_count = self.generate_steps(
            save_plan=save_plan,
            saved_plan_filename=saved_plan_prefix,
            qid=qid
        )
        print(f"=====================\n{generated_steps}\n===================", flush=True)
        # time.sleep(10)
        if enable_summarization:
            from agent_hive.agents.summarization_agent import SummarizationAgent

            summarization_task = Task(
                description=self.tasks[0].description,
                expected_output=self.tasks[0].expected_output,
                agents=[SummarizationAgent(llm=self.llm)],
                context=generated_steps[:],
            )
            generated_steps.append(summarization_task)

        sequential_workflow = SequentialWorkflow(
            tasks=generated_steps, context_type=ContextType.SELECTED
        )

        return sequential_workflow.run(), input_tokens_count, generated_tokens_count

    def generate_steps(self, save_plan=False, saved_plan_filename="", qid=None):
        task = self.tasks[0]
        agent_descriptions = ""
        input_tokens_count=0
        generated_tokens_count=0

        print("!!!!!!!!!!!!!!!!!!!!!", flush=True)
        # time.sleep(10)

        for ii, aagent in enumerate(task.agents):
            agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
            agent_descriptions += f"\nAgent description: {aagent.description}"
            if "task_examples" in aagent.__dict__ and aagent.task_examples:
                agent_descriptions += f"\nTasks that agent can solve:"
                for idx, task_example in enumerate(aagent.task_examples, start=1):
                    agent_descriptions += f"\n{idx}. {task_example}"  # Numbering each task example on a new line
            agent_descriptions += "\n"

        retry = 0
        final_plan = ""
        prev_plan = ""
        prev_review = ""
        while retry < self.max_retries:
            try:
                prompt = self.get_prompt(task.description, agent_descriptions, prev_plan, prev_review)
                logger.info(f"Plan Generation Prompt: \n{prompt}")
                resp = watsonx_llm(prompt, model_id=self.llm)
                llm_response=resp.get("generated_text", "")
                in_tok = resp.get("input_token_count", 0)
                out_tok = resp.get("generated_token_count", 0)
                print(f"in_tok1: {in_tok}")
                print(f"type(in_tok): {type(in_tok)}")
                print(f"out_tok1: {out_tok}")
                print(f"type(out_tok): {type(out_tok)}")
                input_tokens_count+=in_tok
                generated_tokens_count+=out_tok
                logger.info(f"Plan: \n{llm_response}")

                plan_reviewer_agent = PlanReviewerAgent(llm=self.llm)
                review, in_tok, out_tok = plan_reviewer_agent.execute_task(
                    question=task.description,
                    agent_descriptions=agent_descriptions,
                    plan=llm_response,
                )
                print(f"in_tok3: {in_tok}")
                print(f"type(in_tok): {type(in_tok)}")
                print(f"out_tok3: {out_tok}")
                print(f"type(out_tok): {type(out_tok)}")
                # time.sleep(10)
                input_tokens_count+=in_tok
                generated_tokens_count+=out_tok
                prev_review = review
                prev_plan = llm_response
                logger.info(f"Plan Review: \n{review}")
                if review["status"].lower() == "valid":
                    logger.info(f"Plan {retry + 1} is valid.")
                    final_plan = llm_response
                    break
                else:
                    logger.info(f"Plan {retry + 1} is invalid.")
                    retry += 1
            except Exception as e:
                logger.warning(f"Error during plan review: {e}. Retrying...")
                retry += 1

        if final_plan == "":
            logger.info(
                "No valid plan found after multiple retries. Use the plan from the last retry."
            )
            final_plan = prev_plan

        self.memory = []

        task_pattern = r"#Task\d+: (.+)"
        agent_pattern = r"#Agent\d+: (.+)"
        dependency_pattern = r"#Dependency\d+: (.+)"
        output_pattern = r"#ExpectedOutput\d+: (.+)"

        tasks = re.findall(task_pattern, final_plan)
        agents = re.findall(agent_pattern, final_plan)
        dependencies = re.findall(dependency_pattern, final_plan)
        outputs = re.findall(output_pattern, final_plan)

        print(f"=====================\nfinal_plan: {final_plan}\n===================", flush=True)
        # time.sleep(10)

        if save_plan:
            if not saved_plan_filename.endswith(".txt"):
                saved_plan_filename += ".txt"

            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename, "w") as f:
                f.write(saved_plan_text)

        planned_tasks = []
        for i in range(len(tasks)):
            task_description = tasks[i]
            if i == len(agents):
                break
            agent_name = agents[i]
            if i < len(dependencies):
                dependency = dependencies[i]
            else:
                dependency = "None"
            if i < len(outputs):
                expected_output = outputs[i]
            else:
                expected_output = ""

            # identify the agent
            selected_agent = None
            for agent in task.agents:
                if agent.name == agent_name:
                    selected_agent = agent
                    break

            if selected_agent is None:
                # raise ValueError(f"Agent {agent_name} not found in the task.agents")
                selected_agent = task.agents[0]

            # identify the dependency
            if dependency != "None":
                numbers = re.findall(r"#S(\d+)", dependency)
                numbers = list(map(int, numbers))
                context = [planned_tasks[i - 1] for i in numbers]
            else:
                context = []

            a_task = Task(
                description=task_description,
                expected_output=expected_output,
                agents=[selected_agent],
                context=context,
            )
            planned_tasks.append(a_task)

        return planned_tasks, input_tokens_count, generated_tokens_count


    def get_prompt(self, task_description, agent_descriptions, prev_plan, prev_review):
        invalid_plan_description = ''
        if prev_plan:
            invalid_reason = prev_review['reasoning']
            invalid_suggestions = prev_review['suggestions']
            invalid_plan_description = f'''
## Here is one invalid plan, please learn from it and do not repeat its mistakes: ##
Invalid plan:
{prev_plan}
Reason why this plan is invalid: {invalid_reason}
Suggestion for improvement: {invalid_suggestions}
'''
        prompt = f"""
You are an AI assistant who makes step-by-step plan to solve a complicated problem under the help of external agents. 
For each step, make one task followed by one agent-call.
Each step denoted by #S1, #S2, #S3 ... can be referred to in later steps as a dependency.

Each step must contain Task, Agent, Dependency and ExpectedOutput. 
1. **Task**: A detailed description of what needs to be done in this step. It should include all necessary details and requirements.
2. **Agent**: The external agent to be used for solving this task. Agent needs to be selected from the available agents.
3. **Dependency**: A list of previous steps (denoted as `#S1`, `#S2`, etc.) that this step depends on. If no previous steps are required, use `None`.
4. **ExpectedOutput**: The anticipated result from the agent's execution.

## Output Format (Replace '<...>') ##

## Step 1
#Task1: <describe your task here>
#Agent1: <agent_name>
#Dependency1: None
#ExpectedOutput1: <describe the expected output of the call>

## Step 2
#Task2: <describe next task>
#Agent2: <agent_name>
#Dependency2: [<you can use #S1 and more to represent previous outputs as a dependency>]
#ExpectedOutput2: <describe the expected output of the call>

And so on...

## Here are the available agents: ##
{agent_descriptions}

## You are going to solve the following complicated problem: ##
{task_description}

## Guidelines: ##
- Task should be something that can be solved by the agent. Task needs to be clear and unambiguous and contain all the information needed to solve it.
- A plan usually contains less than 5 steps.
- Only output the generated plan, do not output any other text.

{invalid_plan_description}

Output (your generated plan):
"""     
        print(f"agent_descriptions: {agent_descriptions}", flush=True)
        # time.sleep(10)
        return prompt
