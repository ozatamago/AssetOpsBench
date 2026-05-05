import argparse
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

from dotenv import load_dotenv

load_dotenv()

from agent_hive.task import Task

from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
)
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.tsfm import (
    tsfm_tools,
    tsfm_fewshots,
    tsfm_agent_name,
    tsfm_agent_description,
)
from agent_hive.tools.wo import (
    wo_agent_description,
    wo_agent_name,
    wo_fewshots,
    wo_tools,
)
from agent_hive.workflows.planning_review import PlanningReviewWorkflow

from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.react_reflect_agent import ReactReflectAgent
from agent_hive.agents.wo_agent import WorderOrderAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

RESULT_DIR = "/home/track1_result/"
PLAN_DIR = RESULT_DIR + "plan/"
TRAJECTORY_DIR = RESULT_DIR + "trajectory/"
EXP_DIR = RESULT_DIR + "exp/"

import warnings

warnings.filterwarnings("ignore")

PLAN_PREFIX = os.path.dirname(os.path.abspath(__file__)) + "/plan/"

from datasets import load_dataset
from huggingface_hub import login
import os

def load_scenarios_from_hf(utterance_ids):
    # 1) 認証（HF_TOKEN を推奨。既に login 済みなら省略しても良い）
    token = os.getenv("HF_TOKEN") or os.getenv("HF_APIKEY")
    if token:
        login(token)

    # 2) シナリオ取得（"scenarios" config, split は通常 "train"）
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")["train"]

    # 3) IDで絞り込み
    idset = set(utterance_ids)
    filtered = ds.filter(lambda x: x["id"] in idset)

    # 4) run.py が必要とする最小形に落とす
    return [{"id": ex["id"], "text": ex["text"]} for ex in filtered]


def load_scenarios_local(utterance_ids, jsonl_path="/home/scenarios/all_utterance.jsonl"):
    want = set(map(str, utterance_ids))  # id は string 扱いが安全
    out = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("id")) in want:
                out.append(rec)

    # 見つからないIDがあるときに即気付けるようにする
    found = {str(r["id"]) for r in out if "id" in r}
    missing = sorted(want - found)
    if missing:
        raise ValueError(f"Missing scenario ids in {jsonl_path}: {missing[:20]} ... (total {len(missing)})")

    return out

def _write_time_token_file(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not path.endswith(".txt"):
        path += ".txt"
    with open(path, "w") as f:
        # Human-readable + machine-readable
        f.write(json.dumps(payload, indent=2, sort_keys=True))
        f.write("\n")

@retry(stop=stop_after_attempt(7), wait=wait_fixed(2))
def run_planning_workflow(question, llm_model, qid):
    iot_rr_agent = ReactReflectAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
        # reflect_step=3
    )

    fmsr_rr_agent = ReactReflectAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
        # reflect_step=3
    )

    tsfm_rr_agent = ReactReflectAgent(
        name=tsfm_agent_name,
        description=tsfm_agent_description,
        tools=tsfm_tools,
        llm=llm_model,
        few_shots=tsfm_fewshots,
        # reflect_step=3
    )
    
    wo_rr_agent = WorderOrderAgent(
        name=wo_agent_name,
        description=wo_agent_description,
        tools=wo_tools,
        llm=llm_model,
        few_shots=wo_fewshots,
        # reflect_step=3
    )

    task_1 = Task(
        description=question,
        expected_output="",
        agents=[iot_rr_agent, fmsr_rr_agent, tsfm_rr_agent, wo_rr_agent],
    )

    wf = PlanningReviewWorkflow(
        tasks=[task_1],
        llm=llm_model
    )

    plan_subdir = os.path.join(PLAN_DIR, f"[BASE_RAX]Model_{llm_model}")
    os.makedirs(plan_subdir, exist_ok=True)

    saved_plan_prefix = os.path.join(plan_subdir, f"Model_{llm_model}_Q_{qid}_plan")

    history, input_tokens_count, generated_tokens_count = wf.run(save_plan=True,saved_plan_prefix=saved_plan_prefix,qid=qid,enable_summarization=False)

    return history, input_tokens_count, generated_tokens_count



def run_react_reflect(utterances, react_llm_model_id, reverse=False):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)
    exp_subdir = os.path.join(EXP_DIR, f"[BASE_RAX]Model_{react_llm_model_id}")
    os.makedirs(exp_subdir, exist_ok=True)

    print("####################", flush=True)
    
    data = utterances[::-1] if reverse else utterances

    for utterance in data:
        print("$$$$$$$$$$$$$$", flush=True)
        input_tokens_count=0
        generated_tokens_count=0
        logger.info("=" * 10)
        print(
            f"ID: {utterance['id']}, Text: {utterance['text']}, model: {react_llm_model_id}, ReactReflectAgent...", flush=True
        )
        trajectory_subdir = os.path.join(TRAJECTORY_DIR, f"[BASE_RAX]Model_{react_llm_model_id}")
        os.makedirs(trajectory_subdir, exist_ok=True)
        trajectory_file = os.path.join(trajectory_subdir, f"Q_{utterance['id']}_trajectory.json")

        if os.path.exists(trajectory_file):
            print(f"Skipping {utterance['id']}")
            continue

        start_time = time.time()
        ans, input_tokens, generated_tokens = run_planning_workflow(
            utterance["text"],
            react_llm_model_id,
            utterance["id"],
        )
        input_tokens_count+=input_tokens
        generated_tokens_count+=generated_tokens

        end_time = time.time()
        runtime = end_time - start_time
        elapsed = end_time - start_time
        print(f"[run] total elapsed time: {elapsed:.2f} seconds")
        print(f"[run] total input_tokens: {input_tokens_count}")
        print(f"[run] total generated_tokens: {generated_tokens_count}")

        output = {
            "id": utterance["id"],
            "text": utterance["text"],
            "runtime": runtime,
            "trajectory": ans,
        }

        with open(trajectory_file, "w") as file:
            json.dump(output, file, indent=4)

        payload = {
            "llm_model": react_llm_model_id,
            "qid": utterance["id"],  # if this is per-question; otherwise remove
            "elapsed_seconds": elapsed,
            "total_input_tokens": input_tokens_count,
            "total_generated_tokens": generated_tokens_count,
        }
        
        time_token_path = os.path.join(exp_subdir, f"Model_{react_llm_model_id}_Q_{utterance['id']}_time_token.txt")
        _write_time_token_file(time_token_path, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_ids", type=str, default="1,106")
    parser.add_argument("--llm", type=int, default=16)
    parser.add_argument("--reverse", action="store_true")

    args = parser.parse_args()

    utterance_ids = [int(uid.strip()) for uid in args.utterance_ids.split(",")]

    # utterances = load_scenarios_from_hf(utterance_ids)
    utterances = load_scenarios_local(utterance_ids)

    run_react_reflect(
        utterances,
        react_llm_model_id=args.llm,
        reverse=args.reverse,
    )

    # utterances = load_scenarios_local(utterance_ids)

