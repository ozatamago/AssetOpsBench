from agent_hive.task import Task
from pydantic import Field
from typing import List
from agent_hive.enum import ContextType
import json
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
import os
from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

# =========================================================
# TODO: Participants can edit this section ONLY
# Add variable, dict. no more any import just any inline code
# =========================================================
# END OF EDITABLE SECTION


class NewPlanningWorkflow(Workflow):
    """
    Participant Template for Planning Review Workflow.
    ---------------------------------------------------
    📝 Instructions for participants:
    - Only modify the section marked with "TODO: Edit prompt here"
    - Do NOT change any workflow logic, agents, or execution components
    - Keep all retry, memory, and sequential execution intact
    """

    llm: str = Field(description="LLM used by the task planning.")

    def __init__(self, tasks: List[Task], llm: str):
        self.tasks = tasks
        self.memory = []
        self.max_memory = 10
        self.llm = llm
        self.max_retries = 5
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        if len(self.tasks) != 1:
            raise ValueError("Planning only supports one task")
        task = self.tasks[0]
        if task.agents is None or len(task.agents) < 1:
            raise ValueError("Task must have at least one agent")

    def run(self, enable_summarization=False, qid=None):
        generated_steps, input_tokens_count, generated_tokens_count = self.generate_steps(qid=qid)

        sequential_workflow = SequentialWorkflow(
            tasks=generated_steps, context_type=ContextType.SELECTED
        )

        return sequential_workflow.run(), input_tokens_count, generated_tokens_count
    
    MAX_ORIGINAL_PLAN_CHARS = 2000

    def _truncate_text(
        self,
        text: str,
        max_chars: int | None = None,
        *,
        mode: str = "head",          # "head" | "tail" | "middle"
        placeholder: str = "\n...[TRUNCATED]...\n",
    ) -> str:
        """
        Truncate `text` to a constant character length (<= max_chars), preserving whitespace/newlines.

        mode:
          - "head":   keep the first max_chars
          - "tail":   keep the last max_chars
          - "middle": keep head+tail with a placeholder in the middle
        """
        if text is None:
            return ""

        if max_chars is None:
            max_chars = self.MAX_ORIGINAL_PLAN_CHARS

        if max_chars <= 0:
            return ""

        if len(text) <= max_chars:
            return text

        # If placeholder alone exceeds budget, hard-cut it.
        if len(placeholder) >= max_chars:
            return placeholder[:max_chars]

        if mode == "head":
            keep = max_chars - len(placeholder)
            return text[:keep] + placeholder

        if mode == "tail":
            keep = max_chars - len(placeholder)
            return placeholder + text[-keep:]

        if mode == "middle":
            keep = max_chars - len(placeholder)
            left = keep // 2
            right = keep - left
            return text[:left] + placeholder + text[-right:]

        raise ValueError(f"Unknown mode='{mode}'. Use 'head', 'tail', or 'middle'.")

    def _validate_plan_text(self, plan_text: str, agents_allowed):
        TASK_RE   = re.compile(r"^#Task(\d+): (.+)$", re.M)
        AGENT_RE  = re.compile(r"^#Agent(\d+): (.+)$", re.M)
        DEP_RE    = re.compile(r"^#Dependency(\d+): (.+)$", re.M)
        OUT_RE    = re.compile(r"^#ExpectedOutput(\d+): (.+)$", re.M)
        DEP_TOKEN = re.compile(r"#S(\d+)")

        errors = []
        tks  = TASK_RE.findall(plan_text)
        ags  = AGENT_RE.findall(plan_text)
        deps = DEP_RE.findall(plan_text)
        outs = OUT_RE.findall(plan_text)

        def _check_seq(pairs, label):
            if not pairs:
                errors.append(f"{label} lines missing")
                return
            nums = [int(n) for n, _ in pairs]
            if nums != list(range(1, len(nums) + 1)):
                errors.append(f"{label} numbers must be 1..N in order; got {nums}")

        _check_seq(tks,  "Task")
        _check_seq(ags,  "Agent")
        _check_seq(deps, "Dependency")
        _check_seq(outs, "ExpectedOutput")

        if len({len(tks), len(ags), len(deps), len(outs)}) != 1:
            errors.append("Counts of Task/Agent/Dependency/ExpectedOutput must match")

        if tks and deps:
            total = len(tks)
            for n, dep in deps:
                n = int(n)
                dep = dep.strip()
                if dep == "None":
                    continue
                nums = [int(x) for x in DEP_TOKEN.findall(dep)]
                if not nums:
                    errors.append(f"Dependency{n} must be 'None' or '#S1 #S2 ...'; got '{dep}'")
                    continue
                bad = [k for k in nums if k < 1 or k > total]
                if bad:
                    errors.append(f"Dependency{n} out of range {bad}; valid 1..{total}")
                fwd = [k for k in nums if k >= n]
                if fwd:
                    errors.append(f"Dependency{n} forward reference {fwd}; only past steps allowed")

        valid = set(agents_allowed)
        for n, name in AGENT_RE.findall(plan_text):
            if name not in valid:
                errors.append(f"Agent{n} unknown '{name}'. Allowed: {sorted(valid)}")

        # # single-line enforcement between tag lines
        # lines = plan_text.splitlines()
        # TAG = re.compile(r"^#(Task|Agent|Dependency|ExpectedOutput)\d+:", re.M)
        # idxs = [i for i, l in enumerate(lines) if TAG.match(l)] + [len(lines)]
        # for i in range(len(idxs) - 1):
        #     head = idxs[i]
        #     for j in range(head + 1, idxs[i + 1]):
        #         if lines[j].strip() and not TAG.match(lines[j]):
        #             errors.append(f"Field after '{lines[head]}' must be single-line")
        #             break

        return (len(errors) == 0, errors)



    def _build_repair_prompt(
        self,
        base_prompt: str,
        original_plan: str,
        errors: list[str],
        agents_allowed,
        spiral_feedback: dict | None = None,
        truncated_plan_text: str | None = None,
    ) -> str:
        """
        Build a repair prompt for the planner LLM.

        - base_prompt: the original planning prompt (with OUTPUT_MARKER in it).
        - original_plan: the current DAG plan text.
        - errors: issues found by _validate_plan_text (format / structure problems).
        - agents_allowed: list of valid agent names.
        - spiral_feedback: dict returned by _spiral_evaluate_plan (or None).
          We use 'status', 'rationale', and, if present, 'can_answer_now'
          and 'stop_index'.

        The repaired prompt explicitly asks the planner to construct a DAG
        with the bare minimum set of tasks needed to answer the user’s
        question, guided by SPIRAL feedback. It also briefly explains how
        to interpret the main SPIRAL fields (status, can_answer_now,
        stop_index) for planning.
        """
        OUTPUT_MARKER = "Output (your generated plan) ⬇️:"

        # Remove the output marker once so we don't duplicate it
        base_wo_marker = base_prompt.replace(OUTPUT_MARKER, "", 1).rstrip()
            
        # ---- Human / parser errors from _validate_plan_text ----
        if errors:
            issues_text = "Issues detected by the validator:\n- " + "\n- ".join(errors) + "\n"
        else:
            # Even if there are no structural errors, we still allow semantic refinement
            issues_text = (
                "No structural issues were detected by the validator. However, you should still "
                "consider the SPIRAL evaluation feedback above and improve the DAG minimally if needed.\n"
            )

        # ---- Repair rules ----
        # We explicitly instruct the planner to:
        # - interpret SPIRAL fields briefly (status, can_answer_now, stop_index),
        # - build a DAG with the bare minimum number of tasks needed,
        # - and only make minimal changes if the current plan is already good.
        rules = (
            "Repair rules:\n"
            "Avoid redundant or unnecessary tasks.\n"
            "- Make the minimal changes necessary; if there is no problem, you MUST output the Original Plan as-is.\n"
            "- Output ONLY lines in this exact format (no extra prose):\n"
            "  #TaskN: <one-line>\n"
            "  #AgentN: <exact agent name>\n"
            "  #DependencyN: None | #S1 #S2 ... (past steps only)\n"
            "  #ExpectedOutputN: <one-line>\n"
            f"- Agents allowed: {', '.join(agents_allowed)}\n"
            "- Use N = 1..K sequentially; counts across all tags must match.\n"
            "- Do NOT output any explanation, comments, or markdown.\n"
        )

        trunc_original = self._truncate_text(original_plan, max_chars=4000, mode="head")

        return (
            f"{base_wo_marker}\n\n"
            "=== Detected Issues ===\n"
            f"{issues_text}\n"
            "=== Original Plan ===\n"
            f"{trunc_original}\n\n"
            f"{rules}\n\n"
            f"{OUTPUT_MARKER}"
        )

    def generate_steps(self, save_plan=False, saved_plan_filename="", qid=None):
        task = self.tasks[0]
        agent_descriptions = ""
        input_tokens_count=0
        generated_tokens_count=0

        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Customize how agent information is collected and formatted
        # ✅ Allowed: 
        #     - Change numbering style or bullet points
        #     - Include additional metadata (e.g., agent capabilities, tags)
        #     - Provide examples in a different format
        #     - Add emojis or formatting to make the prompt clearer 
        #     - More thinking
        # ❌ Not allowed: 
        #     - Modify workflow execution
        #     - Replace the base ReAct agent or Executor
        #     - Change memory or retry logic
        # =========================================================

        for ii, aagent in enumerate(task.agents):
            agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
            agent_descriptions += f"\nAgent description: {aagent.description}"
            if "task_examples" in aagent.__dict__ and aagent.task_examples:
                agent_descriptions += f"\nTasks that agent can solve:"
                for idx, task_example in enumerate(aagent.task_examples, start=1):
                    agent_descriptions += f"\n{idx}. {task_example}"
            agent_descriptions += "\n"

        # =========================================================
        # END OF EDITABLE SECTION
        # 🚫 Participants should not modify code below this line
        # ❌ No new variables, functions, or workflow logic allowed
        # ✅ Only modify the section marked as TODO above
        # =========================================================

        prompt = self.get_prompt(task.description, agent_descriptions)
        logger.info(f"Plan Generation Prompt: \n{prompt}")
        resp = watsonx_llm(prompt, model_id=self.llm)
        llm_response=resp.get("generated_text", "")
        in_tok = resp.get("input_token_count", 0)
        out_tok = resp.get("generated_token_count", 0)
        input_tokens_count+=in_tok
        generated_tokens_count+=out_tok
        logger.info(f"Plan: \n{llm_response}")

        final_plan = llm_response
        print(f"DAG round_0: {final_plan}")
        saved_plan_filename_0 = saved_plan_filename + "_0.txt"
        saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
        with open(saved_plan_filename_0, "w") as f:
            f.write(saved_plan_text)

        # 0.5) Pre-parse validation + reflexive repair (up to 3 rounds)
        agents_allowed = [a.name for a in task.agents]
        print(f"agents_allowed: {agents_allowed}")

        T = 3
        for t in range(T):
            ok, errs = self._validate_plan_text(final_plan, agents_allowed)
            
            if ok:
                break

            repair_prompt = self._build_repair_prompt(
                base_prompt=prompt,
                original_plan=final_plan,
                errors=errs,
                agents_allowed=agents_allowed,
            )
            print(f"repair_prompt: {repair_prompt}")

            resp = watsonx_llm(repair_prompt, model_id=self.llm)
            final_plan=resp.get("generated_text", "")
            in_tok = resp.get("input_token_count", 0)
            out_tok = resp.get("generated_token_count", 0)
            input_tokens_count+=in_tok
            generated_tokens_count+=out_tok
            print(f"DAG round_{t+1}: {final_plan}")
            saved_plan_filename_t = saved_plan_filename + f"_{t+1}.txt"
            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename_t, "w") as f:
                f.write(saved_plan_text)
            logger.info("Plan was repaired based on issues:\n- " + "\n- ".join(errs or ["(no structural issues)"]))

        if save_plan:
            RESULT_DIR = "/home/track1_result/"
            PLAN_DIR = RESULT_DIR + "plan/"
            plan_subdir = os.path.join(PLAN_DIR, f"[VALID]Model_{self.llm}")
            saved_plan_prefix = os.path.join(plan_subdir, f"Model_{self.llm}_Q_{qid}_plan")
            saved_plan_filename_final = saved_plan_prefix + ".txt"


            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename_final, "w") as f:
                f.write(saved_plan_text)
        
        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Customize LLM response post-processing
        # ❌ Not allowed: 
        #     - Modify workflow execution
        #     - Replace the base ReAct agent or Executor or Task
        #     - Change memory or retry logic
        # =========================================================
        
        self.memory = []

        task_pattern = r"#Task\d+: (.+)"
        agent_pattern = r"#Agent\d+: (.+)"
        dependency_pattern = r"#Dependency\d+: (.+)"
        output_pattern = r"#ExpectedOutput\d+: (.+)"

        tasks = re.findall(task_pattern, final_plan)
        agents = re.findall(agent_pattern, final_plan)
        dependencies = re.findall(dependency_pattern, final_plan)
        outputs = re.findall(output_pattern, final_plan)

        planned_tasks = []
        task_description = ""
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

            selected_agent = None
            for agent in task.agents:
                if agent.name == agent_name:
                    selected_agent = agent
                    break
            if selected_agent is None:
                selected_agent = task.agents[0]

            dependency = "None"
            context = []
            if dependency != "None":
                try:
                    # Extract step numbers like "#S12" -> ["12", ...]
                    numbers = re.findall(r"#S(\d+)", dependency)
                    numbers = list(map(int, numbers))

                    # If any index would be invalid, treat as "no context"
                    n = len(planned_tasks)
                    if (n == 0) or any(i < 1 or i > n for i in numbers):
                        context = []
                    else:
                        context = [planned_tasks[i - 1] for i in numbers]

                except (ValueError, IndexError, TypeError):
                    # ValueError: int conversion failed (unexpected)
                    # IndexError: out-of-range index
                    # TypeError: planned_tasks or dependency unexpected type
                    context = []
            else:
                context = []

            a_task = Task(
                description=task_description,
                expected_output=expected_output,
                agents=[selected_agent],
                context=context,
            )
            planned_tasks.append(a_task)

        logger.info(f"Planned Tasks: \n{planned_tasks}")

        # =========================================================
        # END OF EDITABLE SECTION
        # =========================================================

        return planned_tasks, input_tokens_count, generated_tokens_count

    def get_prompt(self, task_description, agent_descriptions):
        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Improve prompt clarity, formatting, emojis, guidance
        # ✅ Allowed: Wording, structure, examples, emojis
        # ❌ Not allowed: Changing workflow, ReAct agent, Executor, or memory logic
        # =========================================================

        prompt = f"""
🚀 You are an AI assistant tasked with creating a step-by-step plan to solve a complex problem using the external agents provided.  

⚠️ Constraints:
- Only use the agents listed below. No new agents may be added.
- The base ReAct agent and Executor component are fixed. Do not change them.
- Produce a plan with fewer than 5 steps.
- Include Task, Agent, Dependency, and ExpectedOutput for each step.
- Make instructions clear, unambiguous, and actionable.

Each step must follow this format:
#Task<N>: <Describe your task here>
#Agent<N>: <agent_name>
#Dependency<N>: <use #S1, #S2, ... or None>
#ExpectedOutput<N>: <Expected output>

## Here are the available agents: ##
{agent_descriptions}

## Problem to solve: ##
{task_description}

Output (your generated plan) ⬇️:
"""
        # =========================================================
        # End of participant editable section
        # =========================================================
        return prompt
