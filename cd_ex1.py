from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_ROOT = Path(
    "/Users/yusuke/Desktop/Program/codabench/AssetOpsBench/benchmark/cods_track1/track1_result"
)

MODE_TO_DIR = {
    "adaptive": {
        "trajectory": BASE_ROOT / "trajectory" / "[ReAct_CD][adaptive]Model_16",
        "exp": BASE_ROOT / "exp" / "[ReAct_CD][adaptive]Model_16",
        "plan": BASE_ROOT / "plan" / "[ReAct_CD][adaptive]Model_16",
    },
    "force_verify_conservative": {
        "trajectory": BASE_ROOT / "trajectory" / "[ReAct_CD][force_verify_conservative]Model_16",
        "exp": BASE_ROOT / "exp" / "[ReAct_CD][force_verify_conservative]Model_16",
        "plan": BASE_ROOT / "plan" / "[ReAct_CD][force_verify_conservative]Model_16",
    },
    "no_verify": {
        "trajectory": BASE_ROOT / "trajectory" / "[ReAct_CD][no_verify]Model_16",
        "exp": BASE_ROOT / "exp" / "[ReAct_CD][no_verify]Model_16",
        "plan": BASE_ROOT / "plan" / "[ReAct_CD][no_verify]Model_16",
    },
}

STATUS_MAP = {
    "accomplished": "Accomplished",
    "partially accomplished": "Partially accomplished",
    "not accomplished": "Not accomplished",
    "error": "Error",
}


@dataclass
class SplitStats:
    non_verifier_tasks: int = 0
    accomplished: int = 0
    partially_accomplished: int = 0
    not_accomplished: int = 0
    error: int = 0
    unknown_status: int = 0

    @property
    def accomplished_rate(self) -> float:
        return self.accomplished / self.non_verifier_tasks if self.non_verifier_tasks else 0.0

    @property
    def partially_accomplished_rate(self) -> float:
        return self.partially_accomplished / self.non_verifier_tasks if self.non_verifier_tasks else 0.0

    @property
    def not_accomplished_rate(self) -> float:
        return self.not_accomplished / self.non_verifier_tasks if self.non_verifier_tasks else 0.0

    @property
    def error_rate(self) -> float:
        return self.error / self.non_verifier_tasks if self.non_verifier_tasks else 0.0


@dataclass
class MainMissedStats:
    partially_count: int = 0
    not_count: int = 0
    error_count: int = 0

    no_verifier_missed_partially_count: int = 0
    verifier_success_missed_partially_count: int = 0

    no_verifier_missed_not_count: int = 0
    verifier_success_missed_not_count: int = 0

    no_verifier_missed_error_count: int = 0
    verifier_success_missed_error_count: int = 0

    @property
    def total_missed_partially_count(self) -> int:
        return (
            self.no_verifier_missed_partially_count
            + self.verifier_success_missed_partially_count
        )

    @property
    def total_missed_not_count(self) -> int:
        return (
            self.no_verifier_missed_not_count
            + self.verifier_success_missed_not_count
        )

    @property
    def total_missed_error_count(self) -> int:
        return (
            self.no_verifier_missed_error_count
            + self.verifier_success_missed_error_count
        )

    @property
    def no_verifier_missed_partially_rate(self) -> Optional[float]:
        if self.partially_count == 0:
            return None
        return self.no_verifier_missed_partially_count / self.partially_count

    @property
    def verifier_success_missed_partially_rate(self) -> Optional[float]:
        if self.partially_count == 0:
            return None
        return self.verifier_success_missed_partially_count / self.partially_count

    @property
    def total_missed_partially_rate(self) -> Optional[float]:
        if self.partially_count == 0:
            return None
        return self.total_missed_partially_count / self.partially_count

    @property
    def no_verifier_missed_not_rate(self) -> Optional[float]:
        if self.not_count == 0:
            return None
        return self.no_verifier_missed_not_count / self.not_count

    @property
    def verifier_success_missed_not_rate(self) -> Optional[float]:
        if self.not_count == 0:
            return None
        return self.verifier_success_missed_not_count / self.not_count

    @property
    def total_missed_not_rate(self) -> Optional[float]:
        if self.not_count == 0:
            return None
        return self.total_missed_not_count / self.not_count

    @property
    def no_verifier_missed_error_rate(self) -> Optional[float]:
        if self.error_count == 0:
            return None
        return self.no_verifier_missed_error_count / self.error_count

    @property
    def verifier_success_missed_error_rate(self) -> Optional[float]:
        if self.error_count == 0:
            return None
        return self.verifier_success_missed_error_count / self.error_count

    @property
    def total_missed_error_rate(self) -> Optional[float]:
        if self.error_count == 0:
            return None
        return self.total_missed_error_count / self.error_count


@dataclass
class ModeSummary:
    mode: str
    num_files: int
    total_tasks_executed: int
    verification_count: int

    main: SplitStats
    branch: SplitStats
    main_missed: MainMissedStats

    exp_elapsed_seconds_sum: float
    exp_total_input_tokens_sum: int
    exp_total_generated_tokens_sum: int

    traj_tokens_sent_sum: int
    traj_tokens_received_sum: int
    traj_api_calls_sum: int
    traj_total_cost_sum: float
    traj_instance_cost_sum: float

    combined_token_total: int


def load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def extract_qid_from_name(path: Path) -> Optional[int]:
    m = re.search(r"Q_(\d+)", path.name)
    return int(m.group(1)) if m else None


def normalize_status(raw: Optional[str]) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    return STATUS_MAP.get(raw.strip().lower())


def is_verifier_task(task: Dict[str, Any]) -> bool:
    if task.get("agent_name") == "verifier":
        return True
    logs = task.get("logs", {})
    if isinstance(logs, dict):
        wf = logs.get("_workflow", {})
        if isinstance(wf, dict) and wf.get("kind") == "verifier":
            return True
    return False


def is_system_task(task: Dict[str, Any]) -> bool:
    if task.get("agent_name") == "system":
        return True
    logs = task.get("logs", {})
    if isinstance(logs, dict):
        wf = logs.get("_workflow", {})
        if isinstance(wf, dict) and wf.get("kind") == "system":
            return True
    return False


def has_explicit_error_marker(task: Dict[str, Any]) -> bool:
    response = task.get("response")
    logs = task.get("logs", {})

    def contains_error(obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                if key in {"error", "exception", "traceback"} and v:
                    return True
                if contains_error(v):
                    return True
        elif isinstance(obj, list):
            for x in obj:
                if contains_error(x):
                    return True
        elif isinstance(obj, str):
            s = obj.lower()
            if "traceback" in s or "exception" in s or "error" in s:
                return True
        return False

    return contains_error(response) or contains_error(logs)


def extract_status_from_response(task: Dict[str, Any]) -> Optional[str]:
    response = task.get("response")

    if isinstance(response, dict):
        if "status" in response:
            status = normalize_status(response.get("status"))
            if status is not None:
                return status
        if has_explicit_error_marker(task):
            return "Error"

    if isinstance(response, list):
        for item in reversed(response):
            if isinstance(item, dict) and "status" in item:
                status = normalize_status(item.get("status"))
                if status is not None:
                    return status

    reviews = task.get("reviews")
    if isinstance(reviews, list):
        for review in reviews:
            if not isinstance(review, str):
                continue
            m = re.search(
                r"Task Status:\s*(Accomplished|Partially accomplished|Not accomplished|Error)",
                review,
                flags=re.IGNORECASE,
            )
            if m:
                return normalize_status(m.group(1))

    logs = task.get("logs", {})
    if isinstance(logs, dict):
        revs = logs.get("reviews")
        if isinstance(revs, list):
            for review in revs:
                if not isinstance(review, str):
                    continue
                m = re.search(
                    r"Task Status:\s*(Accomplished|Partially accomplished|Not accomplished|Error)",
                    review,
                    flags=re.IGNORECASE,
                )
                if m:
                    return normalize_status(m.group(1))

    if has_explicit_error_marker(task):
        return "Error"

    return None


def extract_verifier_label(task: Dict[str, Any]) -> Optional[str]:
    response = task.get("response")
    if isinstance(response, dict) and isinstance(response.get("label"), str):
        return response["label"].strip()

    logs = task.get("logs", {})
    if isinstance(logs, dict):
        final_answer = logs.get("final_answer")
        if isinstance(final_answer, dict) and isinstance(final_answer.get("label"), str):
            return final_answer["label"].strip()

    return None


def extract_model_stats(task: Dict[str, Any]) -> Dict[str, Any]:
    logs = task.get("logs", {})
    if isinstance(logs, dict):
        info = logs.get("info", {})
        if isinstance(info, dict):
            stats = info.get("model_stats", {})
            if isinstance(stats, dict):
                return stats
    return {}


def paired_verifier_label(tasks: List[Dict[str, Any]], idx: int) -> Optional[str]:
    j = idx + 1
    while j < len(tasks):
        nxt = tasks[j]
        if is_verifier_task(nxt):
            return extract_verifier_label(nxt)
        if not is_system_task(nxt):
            return None
        j += 1
    return None


def extract_conditional_plan_json_text(text: str) -> str:
    marker = "Conditional Plan JSON:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()

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
            if isinstance(obj, dict) and "answer_contract" in obj and "nodes" in obj:
                return candidate
        except Exception:
            pass

    raise ValueError("Could not extract plan JSON from saved plan text.")


def load_plan_file(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    extracted = extract_conditional_plan_json_text(raw)
    return json.loads(extracted)


def find_plan_file(plan_dir: Path, qid: int) -> Optional[Path]:
    patterns = [
        f"*Q_{qid}*plan*.txt",
        f"*Q_{qid}*.txt",
    ]
    for pattern in patterns:
        matches = sorted(plan_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def classify_plan_node(node: Dict[str, Any]) -> str:
    node_id = str(node.get("id", ""))
    agent = str(node.get("agent", ""))

    if node_id.startswith("V_") or agent == "verifier":
        return "verifier"
    if node_id.startswith("B_"):
        return "branch"
    if node_id.startswith("J_") or agent == "system":
        return "system"
    return "main"


def build_plan_lookup(plan_obj: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, str], int]]:
    lookup: Dict[str, Dict[Tuple[str, str], int]] = {
        "main": {},
        "branch": {},
    }

    nodes = plan_obj.get("nodes", [])
    if not isinstance(nodes, list):
        return lookup

    for node in nodes:
        if not isinstance(node, dict):
            continue

        kind = classify_plan_node(node)
        if kind not in {"main", "branch"}:
            continue

        task_desc = str(node.get("task", ""))
        agent_name = str(node.get("agent", ""))
        key = (task_desc, agent_name)
        lookup[kind][key] = lookup[kind].get(key, 0) + 1

    return lookup


def heuristic_kind_from_task(task: Dict[str, Any]) -> str:
    desc = str(task.get("task_description", "")).lower()
    if (
        "diagnose" in desc
        or "repair" in desc
        or "retry" in desc
        or "re-download" in desc
        or "re-execute" in desc
        or "failure" in desc
    ):
        return "branch"
    return "main"


def update_split_counts(split: SplitStats, status: Optional[str]) -> None:
    split.non_verifier_tasks += 1

    if status == "Accomplished":
        split.accomplished += 1
    elif status == "Partially accomplished":
        split.partially_accomplished += 1
    elif status == "Not accomplished":
        split.not_accomplished += 1
    elif status == "Error":
        split.error += 1
    else:
        split.unknown_status += 1


def update_main_missed(
    main_missed: MainMissedStats,
    status: Optional[str],
    verifier_label: Optional[str],
) -> None:
    if status == "Partially accomplished":
        main_missed.partially_count += 1
        if verifier_label is None:
            main_missed.no_verifier_missed_partially_count += 1
        elif verifier_label == "success":
            main_missed.verifier_success_missed_partially_count += 1

    elif status == "Not accomplished":
        main_missed.not_count += 1
        if verifier_label is None:
            main_missed.no_verifier_missed_not_count += 1
        elif verifier_label == "success":
            main_missed.verifier_success_missed_not_count += 1

    elif status == "Error":
        main_missed.error_count += 1
        if verifier_label is None:
            main_missed.no_verifier_missed_error_count += 1
        elif verifier_label == "success":
            main_missed.verifier_success_missed_error_count += 1


def summarize_mode(mode: str, traj_dir: Path, exp_dir: Path, plan_dir: Path) -> ModeSummary:
    traj_files = sorted(traj_dir.glob("Q_*_trajectory.json"))
    exp_files = sorted(exp_dir.glob("*Q_*_time_token.txt"))

    exp_by_qid: Dict[int, Dict[str, Any]] = {}
    for p in exp_files:
        qid = extract_qid_from_name(p)
        if qid is None:
            continue
        try:
            exp_by_qid[qid] = load_json_file(p)
        except Exception as e:
            print(f"[WARN] Failed to read exp file: {p} ({e})")

    total_tasks_executed = 0
    verification_count = 0

    main_split = SplitStats()
    branch_split = SplitStats()
    main_missed = MainMissedStats()

    exp_elapsed_seconds_sum = 0.0
    exp_total_input_tokens_sum = 0
    exp_total_generated_tokens_sum = 0

    traj_tokens_sent_sum = 0
    traj_tokens_received_sum = 0
    traj_api_calls_sum = 0
    traj_total_cost_sum = 0.0
    traj_instance_cost_sum = 0.0

    for traj_path in traj_files:
        try:
            payload = load_json_file(traj_path)
        except Exception as e:
            print(f"[WARN] Failed to read trajectory file: {traj_path} ({e})")
            continue

        qid = payload.get("id")
        if isinstance(qid, int) and qid in exp_by_qid:
            exp_payload = exp_by_qid[qid]
            exp_elapsed_seconds_sum += safe_float(exp_payload.get("elapsed_seconds"))
            exp_total_input_tokens_sum += safe_int(exp_payload.get("total_input_tokens"))
            exp_total_generated_tokens_sum += safe_int(exp_payload.get("total_generated_tokens"))

        plan_lookup = {"main": {}, "branch": {}}
        if isinstance(qid, int):
            plan_path = find_plan_file(plan_dir, qid)
            if plan_path is not None:
                try:
                    plan_obj = load_plan_file(plan_path)
                    plan_lookup = build_plan_lookup(plan_obj)
                except Exception as e:
                    print(f"[WARN] Failed to read plan file for qid={qid}: {plan_path} ({e})")
            else:
                print(f"[WARN] No plan file found for qid={qid} in {plan_dir}")

        tasks = payload.get("trajectory", [])
        if not isinstance(tasks, list):
            continue

        total_tasks_executed += len(tasks)

        used_lookup = {
            "main": {},
            "branch": {},
        }

        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue

            stats = extract_model_stats(task)
            traj_tokens_sent_sum += safe_int(stats.get("tokens_sent"))
            traj_tokens_received_sum += safe_int(stats.get("tokens_received"))
            traj_api_calls_sum += safe_int(stats.get("api_calls"))
            traj_total_cost_sum += safe_float(stats.get("total_cost"))
            traj_instance_cost_sum += safe_float(stats.get("instance_cost"))

            if is_verifier_task(task):
                verification_count += 1
                continue

            if is_system_task(task):
                continue

            key = (str(task.get("task_description", "")), str(task.get("agent_name", "")))

            main_used = used_lookup["main"].get(key, 0)
            main_cap = plan_lookup["main"].get(key, 0)

            branch_used = used_lookup["branch"].get(key, 0)
            branch_cap = plan_lookup["branch"].get(key, 0)

            if main_used < main_cap:
                kind = "main"
                used_lookup["main"][key] = main_used + 1
            elif branch_used < branch_cap:
                kind = "branch"
                used_lookup["branch"][key] = branch_used + 1
            else:
                kind = heuristic_kind_from_task(task)

            status = extract_status_from_response(task)
            verifier_label = paired_verifier_label(tasks, i)

            if kind == "branch":
                update_split_counts(branch_split, status)
            else:
                update_split_counts(main_split, status)
                update_main_missed(main_missed, status, verifier_label)

    combined_token_total = (
        exp_total_input_tokens_sum
        + exp_total_generated_tokens_sum
        + traj_tokens_sent_sum
        + traj_tokens_received_sum
    )

    return ModeSummary(
        mode=mode,
        num_files=len(traj_files),
        total_tasks_executed=total_tasks_executed,
        verification_count=verification_count,

        main=main_split,
        branch=branch_split,
        main_missed=main_missed,

        exp_elapsed_seconds_sum=exp_elapsed_seconds_sum,
        exp_total_input_tokens_sum=exp_total_input_tokens_sum,
        exp_total_generated_tokens_sum=exp_total_generated_tokens_sum,

        traj_tokens_sent_sum=traj_tokens_sent_sum,
        traj_tokens_received_sum=traj_tokens_received_sum,
        traj_api_calls_sum=traj_api_calls_sum,
        traj_total_cost_sum=traj_total_cost_sum,
        traj_instance_cost_sum=traj_instance_cost_sum,

        combined_token_total=combined_token_total,
    )


def fmt_rate(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.4f}"


def print_summary_table(rows: List[ModeSummary]) -> None:
    headers = [
        "mode",
        "num_files",
        "total_tasks_executed",
        "verification_count",

        "main_non_verifier_tasks",
        "main_Accomplished",
        "main_Partially",
        "main_Not",
        "main_Error",
        "main_Accomplished_rate",
        "main_Partially_rate",
        "main_Not_rate",
        "main_Error_rate",

        "main_no_verifier_missed_partially_count",
        "main_verifier_success_missed_partially_count",
        "main_total_missed_partially_count",
        "main_partially_count",
        "main_no_verifier_missed_partially_rate",
        "main_verifier_success_missed_partially_rate",
        "main_total_missed_partially_rate",

        "main_no_verifier_missed_not_count",
        "main_verifier_success_missed_not_count",
        "main_total_missed_not_count",
        "main_not_count",
        "main_no_verifier_missed_not_rate",
        "main_verifier_success_missed_not_rate",
        "main_total_missed_not_rate",

        "main_no_verifier_missed_error_count",
        "main_verifier_success_missed_error_count",
        "main_total_missed_error_count",
        "main_error_count",
        "main_no_verifier_missed_error_rate",
        "main_verifier_success_missed_error_rate",
        "main_total_missed_error_rate",

        "branch_non_verifier_tasks",
        "branch_Accomplished",
        "branch_Partially",
        "branch_Not",
        "branch_Error",
        "branch_Accomplished_rate",
        "branch_Partially_rate",
        "branch_Not_rate",
        "branch_Error_rate",

        "exp_input_tokens",
        "exp_generated_tokens",
        "traj_tokens_sent",
        "traj_tokens_received",
        "traj_api_calls",
        "traj_total_cost",
        "traj_instance_cost",
        "combined_token_total",
        "elapsed_seconds_sum",
    ]

    print("\t".join(headers))
    for r in rows:
        print(
            "\t".join([
                r.mode,
                str(r.num_files),
                str(r.total_tasks_executed),
                str(r.verification_count),

                str(r.main.non_verifier_tasks),
                str(r.main.accomplished),
                str(r.main.partially_accomplished),
                str(r.main.not_accomplished),
                str(r.main.error),
                f"{r.main.accomplished_rate:.4f}",
                f"{r.main.partially_accomplished_rate:.4f}",
                f"{r.main.not_accomplished_rate:.4f}",
                f"{r.main.error_rate:.4f}",

                str(r.main_missed.no_verifier_missed_partially_count),
                str(r.main_missed.verifier_success_missed_partially_count),
                str(r.main_missed.total_missed_partially_count),
                str(r.main_missed.partially_count),
                fmt_rate(r.main_missed.no_verifier_missed_partially_rate),
                fmt_rate(r.main_missed.verifier_success_missed_partially_rate),
                fmt_rate(r.main_missed.total_missed_partially_rate),

                str(r.main_missed.no_verifier_missed_not_count),
                str(r.main_missed.verifier_success_missed_not_count),
                str(r.main_missed.total_missed_not_count),
                str(r.main_missed.not_count),
                fmt_rate(r.main_missed.no_verifier_missed_not_rate),
                fmt_rate(r.main_missed.verifier_success_missed_not_rate),
                fmt_rate(r.main_missed.total_missed_not_rate),

                str(r.main_missed.no_verifier_missed_error_count),
                str(r.main_missed.verifier_success_missed_error_count),
                str(r.main_missed.total_missed_error_count),
                str(r.main_missed.error_count),
                fmt_rate(r.main_missed.no_verifier_missed_error_rate),
                fmt_rate(r.main_missed.verifier_success_missed_error_rate),
                fmt_rate(r.main_missed.total_missed_error_rate),

                str(r.branch.non_verifier_tasks),
                str(r.branch.accomplished),
                str(r.branch.partially_accomplished),
                str(r.branch.not_accomplished),
                str(r.branch.error),
                f"{r.branch.accomplished_rate:.4f}",
                f"{r.branch.partially_accomplished_rate:.4f}",
                f"{r.branch.not_accomplished_rate:.4f}",
                f"{r.branch.error_rate:.4f}",

                str(r.exp_total_input_tokens_sum),
                str(r.exp_total_generated_tokens_sum),
                str(r.traj_tokens_sent_sum),
                str(r.traj_tokens_received_sum),
                str(r.traj_api_calls_sum),
                f"{r.traj_total_cost_sum:.6f}",
                f"{r.traj_instance_cost_sum:.6f}",
                str(r.combined_token_total),
                f"{r.exp_elapsed_seconds_sum:.4f}",
            ])
        )


def main() -> None:
    rows: List[ModeSummary] = []

    for mode, paths in MODE_TO_DIR.items():
        traj_dir = paths["trajectory"]
        exp_dir = paths["exp"]
        plan_dir = paths["plan"]

        if not traj_dir.exists():
            print(f"[WARN] trajectory dir not found for {mode}: {traj_dir}")
            continue
        if not exp_dir.exists():
            print(f"[WARN] exp dir not found for {mode}: {exp_dir}")
            continue
        if not plan_dir.exists():
            print(f"[WARN] plan dir not found for {mode}: {plan_dir}")
            continue

        row = summarize_mode(mode, traj_dir, exp_dir, plan_dir)
        rows.append(row)

    print_summary_table(rows)


if __name__ == "__main__":
    main()