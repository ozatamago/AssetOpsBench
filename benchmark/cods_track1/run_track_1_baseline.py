import argparse
import json
import os
import time  

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from huggingface_hub import login

login(os.getenv("HF_APIKEY", None))

from agent_hive.task import Task
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
    iot_task_examples,
)
from agent_hive.tools.tsfm import (
    tsfm_tools,
    tsfm_fewshots,
    tsfm_agent_name,
    tsfm_agent_description,
    tsfm_task_examples,
)
from agent_hive.tools.wo import (
    wo_agent_description,
    wo_agent_name,
    wo_fewshots,
    wo_tools,
    wo_task_examples,
)
from agent_hive.agents.react_reflect_agent import ReactReflectAgent
from agent_hive.logger import get_custom_logger
from agent_hive.agents.wo_agent import WorderOrderAgent
from agent_hive.workflows.track1_planning import NewPlanningWorkflow

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

import warnings

warnings.filterwarnings("ignore")

RESULT_DIR = "/home/track1_result/"
PLAN_DIR = RESULT_DIR + "plan/"
TRAJECTORY_DIR = RESULT_DIR + "trajectory/"
EXP_DIR = RESULT_DIR + "exp/"


def _write_time_token_file(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not path.endswith(".txt"):
        path += ".txt"
    with open(path, "w") as f:
        # Human-readable + machine-readable
        f.write(json.dumps(payload, indent=2, sort_keys=True))
        f.write("\n")

def load_scenarios(utterance_ids):
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    train_ds = ds["train"]
    df = train_ds.to_pandas()

    filtered_df = df[df["id"].isin(utterance_ids)]

    return filtered_df.to_dict(orient="records")


def run_planning_workflow(
        question, qid, llm_model=16, generate_steps_only=False
):
    iot_r_agent = ReactReflectAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
        task_examples=iot_task_examples,
        reflect_step=1,
    )

    fmsr_r_agent = ReactReflectAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
        reflect_step=1,
    )

    tsfm_rr_agent = ReactReflectAgent(
        name=tsfm_agent_name,
        description=tsfm_agent_description,
        tools=tsfm_tools,
        llm=llm_model,
        few_shots=tsfm_fewshots,
        task_examples=tsfm_task_examples,
        reflect_step=1,
    )
    
    wo_rr_agent = WorderOrderAgent(
        name=wo_agent_name,
        description=wo_agent_description,
        tools=wo_tools,
        llm=llm_model,
        few_shots=wo_fewshots,
        reflect_step=1,
        task_examples=wo_task_examples,
    )

    task = Task(
        description=question,
        expected_output="",
        agents=[iot_r_agent, fmsr_r_agent, tsfm_rr_agent, wo_rr_agent],
    )

    wf = NewPlanningWorkflow(
        tasks=[task],
        llm=llm_model,
    )

    plan_subdir = os.path.join(PLAN_DIR, f"[BASE]Model_{llm_model}")
    question_subdir = os.path.join(plan_subdir, f"Q_{qid}")
    os.makedirs(question_subdir, exist_ok=True)

    saved_plan_prefix = os.path.join(question_subdir, f"Model_{llm_model}_Q_{qid}_plan")

    if generate_steps_only:
        os.makedirs(PLAN_DIR, exist_ok=True)

        return wf.generate_steps(
            save_plan=True,
            saved_plan_filename=saved_plan_prefix,
            qid=qid
        )
    history, input_tokens_count, generated_tokens_count = wf.run(qid=qid)

    return history, input_tokens_count, generated_tokens_count

def run(utterances, generate_steps_only=False, llm_model=16):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    exp_subdir = os.path.join(EXP_DIR, f"[BASE]Model_{llm_model}")
    os.makedirs(exp_subdir, exist_ok=True)

    for utterance in utterances:
        start_time = time.perf_counter()
        input_tokens_count=0
        generated_tokens_count=0
        
        logger.info("=" * 10)
        logger.info(f"ID: {utterance['id']}, Task: {utterance['text']}")
        trajectory_file = f"{TRAJECTORY_DIR}Q_{utterance['id']}_trajectory.json"

        ans, input_tokens, generated_tokens= run_planning_workflow(
            utterance["text"],
            utterance["id"],
            llm_model=llm_model,
            generate_steps_only=generate_steps_only,
        )
        input_tokens_count+=input_tokens
        generated_tokens_count+=generated_tokens

        if generate_steps_only:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(f"[run] total elapsed time: {elapsed:.2f} seconds")
            print(f"[run] total input_tokens: {input_tokens_count}")
            print(f"[run] total generated_tokens: {generated_tokens_count}")

            payload = {
                "llm_model": llm_model,
                "qid": utterance["id"],  # if this is per-question; otherwise remove
                "elapsed_seconds": elapsed,
                "total_input_tokens": input_tokens_count,
                "total_generated_tokens": generated_tokens_count,
            }

            time_token_path = os.path.join(exp_subdir, f"Model_{llm_model}_Q_{utterance['id']}_time_token.txt")
            _write_time_token_file(time_token_path, payload)
            continue

        output = {"id": utterance["id"], "text": utterance["text"], "trajectory": ans}

        with open(trajectory_file, "w") as f:
            json.dump(output, f, indent=4)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[run] total elapsed time: {elapsed:.2f} seconds")
        print(f"[run] total input_tokens: {input_tokens_count}")
        print(f"[run] total generated_tokens: {generated_tokens_count}")

        payload = {
            "llm_model": llm_model,
            "qid": utterance["id"],  # if this is per-question; otherwise remove
            "elapsed_seconds": elapsed,
            "total_input_tokens": input_tokens_count,
            "total_generated_tokens": generated_tokens_count,
        }

        time_token_path = os.path.join(exp_subdir, f"Model_{llm_model}_Q_{utterance['id']}_time_token.txt")
        _write_time_token_file(time_token_path, payload)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_ids", type=str, default="1,106")
    parser.add_argument("--generate_steps_only", type=bool, default=False)
    parser.add_argument("--llm_model", type=int, default=16)

    args = parser.parse_args()
    utterance_ids = [int(uid.strip()) for uid in args.utterance_ids.split(",")]
    utterances = load_scenarios(utterance_ids)

    run(
        utterances,
        generate_steps_only=args.generate_steps_only,
        llm_model=args.llm_model
    )
