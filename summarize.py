# summarize_pipeline.py — extract + concise (choice/word-limited) summaries, no trimming

import os
import re
import json
from reactxen.utils.model_inference import watsonx_llm

RESULT_DIR = "/home/track1_result/"
TRAJECTORY_DIR = os.path.join(RESULT_DIR, "trajectory")

# ======== word limits (tune as needed) ========
WORD_LIMITS = {
    "task_description": 12,   # words
    "final_answer": 10,       # only used for generic path
    "review": 20,             # words for body after status
}

# ======== utils ========
def _word_count(s: str) -> int:
    return len((s or "").strip().split())

def _llm(prompt: str, model_id: int) -> str:
    return (watsonx_llm(prompt, model_id=model_id) or {}).get("generated_text", "").strip()

def _enforce_limit_with_regen(initial_prompt: str, word_limit: int, model_id: int) -> str:
    """1回生成→語数・禁句チェック→超過なら一度だけやり直し（task_description等・一般用途）。"""
    out = _llm(initial_prompt, model_id)
    ok = bool(out) and _word_count(out) <= word_limit and ("Input:" not in out) and ("Output" not in out)
    if ok:
        return out
    over = _word_count(out) if out else 0
    regen_prompt = (
        f"Your previous answer had {over} words. Rewrite to ≤ {word_limit} words.\n"
        "One line. No preamble or quotes. Do not include the words 'Input' or 'Output'.\n\n"
        "Previous answer:\n"
        f"{out or ''}\n\n"
        "Return ONLY the final line:"
    )
    return _llm(regen_prompt, model_id)

def _enforce_choice_with_regen(initial_prompt: str, allowed: set[str], model_id: int) -> str:
    """集合 {EXPORT, NO_EXPORT} などに“厳密一致”させるための強制器（final_answer向け）。"""
    out = _llm(initial_prompt, model_id).strip()
    if out in allowed:
        return out
    # 再生成
    regen_prompt = (
        "Your previous answer did not exactly match one of the allowed tokens.\n"
        f"Return EXACTLY one of: {', '.join(sorted(allowed))}.\n"
        "No punctuation, no spaces, no preamble, one line only.\n\n"
        "Previous answer:\n"
        f"{out}\n\n"
        "Respond with ONLY the token:"
    )
    out2 = _llm(regen_prompt, model_id).strip()
    return out2

# ======== prompt builders ========

# --- task_description（そのまま：例を1つ、タグ区切り、禁句） ---
def _prompt_task_description(content: str, limit: int) -> str:
    ex_in  = "Identify the available IoT sites and confirm data availability."
    ex_out = "List available IoT sites and confirm data availability"
    return (
        "Rewrite as a short imperative headline.\n"
        f"Rules: one line; ≤ {limit} words; start with a verb; no ending punctuation; natural and specific.\n"
        "Do not include the words 'Input' or 'Output'. Return only the headline.\n\n"
        "<examples>\n"
        "<example>\n"
        f"<input>{ex_in}</input>\n"
        f"<output>{ex_out}</output>\n"
        "</example>\n"
        "</examples>\n\n"
        "<task>\n"
        f"{content}\n"
        "</task>\n\n"
        "Respond with ONLY:\n"
        "<response>"
    )

# --- final_answer（厳格二択：EXPORT/NO_EXPORT 版） ---
def _prompt_final_answer_export_rule(content: str) -> str:
    """
    “exported”（大文字小文字無視）が入力に含まれるかで、
    出力を {EXPORT, NO_EXPORT} の二択に厳格化。
    """
    return (
        "Binary classification. Output must be EXACTLY one of: EXPORT, NO_EXPORT.\n"
        "Decision rule: If the input contains the word 'exported' (case-insensitive), output EXPORT; otherwise NO_EXPORT.\n"
        "No preamble, no explanation, no punctuation, no spaces, one line only.\n"
        "Do not include the words 'Input' or 'Output'.\n\n"
        "<examples>\n"
        "<example>\n"
        "<input>The record was exported successfully.</input>\n"
        "<output>EXPORT</output>\n"
        "</example>\n"
        "<example>\n"
        "<input>No export flag found in the text.</input>\n"
        "<output>NO_EXPORT</output>\n"
        "</example>\n"
        "</examples>\n\n"
        "<task>\n"
        f"{content}\n"
        "</task>\n\n"
        "Respond with ONLY:\n"
        "<response>"
    )

# --- final_answer（汎用：原子値優先の極短回答） ---
def _prompt_final_answer_generic(content: str, limit: int) -> str:
    return (
        "Return the atomic answer ONLY.\n"
        "If exactly one value exists (e.g., a site name), output that value alone.\n"
        f"Otherwise reply with a single line ≤ {limit} words; no preamble or quotes.\n"
        "Do not include the words 'Input' or 'Output'.\n\n"
        "<examples>\n"
        "<example>\n"
        "<input>The available IoT site is MAIN.</input>\n"
        "<output>MAIN</output>\n"
        "</example>\n"
        "</examples>\n\n"
        "<task>\n"
        f"{content}\n"
        "</task>\n\n"
        "Respond with ONLY:\n"
        "<response>"
    )

# --- review（先頭ステータス保持、本文だけ短縮） ---
def _prompt_review_body(content_wo_status: str, limit: int) -> str:
    return (
        "Compress into one pipe-delimited line: `key action | improvement` (omit missing parts).\n"
        f"Keep it clear and ≤ {limit} words. Do not include any status label.\n"
        "Do not include the words 'Input' or 'Output'.\n\n"
        "<examples>\n"
        "<example>\n"
        "<input>The agent used the sites tool correctly and answered precisely. Suggestions: none.</input>\n"
        "<output>used sites tool correctly | none</output>\n"
        "</example>\n"
        "</examples>\n\n"
        "<task>\n"
        f"{content_wo_status}\n"
        "</task>\n\n"
        "Respond with ONLY:\n"
        "<response>"
    )

# ======== extraction ========
def extract_trajectory_fields(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {"text": data.get("text", ""), "tasks": []}
    for traj in data.get("trajectory", []):
        logs = traj.get("logs", {}) or {}
        reviews = logs.get("reviews")
        review = ""
        if isinstance(reviews, list) and reviews:
            review = reviews[0]
        elif isinstance(reviews, str):
            review = reviews
        out["tasks"].append({
            "task_description": traj.get("task_description", ""),
            "agent_name": traj.get("agent_name", ""),       # no summary
            "final_answer": logs.get("final_answer", ""),
            "review": review,
        })
    return out

# ======== summarization ========
def summarize_field(field_content: str, field_name: str, model_id: int = 16) -> str:
    if not field_content:
        return ""
    if field_name in ("text", "agent_name"):
        return field_content

    if field_name == "task_description":
        limit = WORD_LIMITS["task_description"]
        prompt = _prompt_task_description(field_content, limit)
        return _enforce_limit_with_regen(prompt, limit, model_id)

    if field_name == "final_answer":
        # “exported” に関する二択ルールが匂う場合は厳格分類プロンプトを使う
        lower = field_content.lower()
        looks_like_export_rule = "exported" in lower or "export" in lower
        if looks_like_export_rule:
            prompt = _prompt_final_answer_export_rule(field_content)
            out = _enforce_choice_with_regen(prompt, {"EXPORT", "NO_EXPORT"}, model_id).strip()
            # まだ不一致ならルールで最終決定（安全フォールバック）
            if out not in {"EXPORT", "NO_EXPORT"}:
                out = "EXPORT" if "exported" in lower else "NO_EXPORT"
            return out
        # 汎用（原子値を返す）
        limit = WORD_LIMITS["final_answer"]
        prompt = _prompt_final_answer_generic(field_content, limit)
        return _enforce_limit_with_regen(prompt, limit, model_id)

    if field_name == "review":
        m = re.search(r"(Task Status:\s*[^\n\r]+)", field_content)
        status_prefix = m.group(1).strip() if m else "Task Status: Unknown"
        body = field_content.replace(status_prefix, "", 1).strip()
        body_prompt = _prompt_review_body(body or "", WORD_LIMITS["review"])
        body_out = _enforce_limit_with_regen(body_prompt, WORD_LIMITS["review"], model_id).strip()
        return f"{status_prefix} | {body_out}" if body_out else status_prefix

    # fallback
    limit = 50
    prompt = (
        f"Summarize in one concise line (≤ {limit} words). Keep key facts; no preamble.\n\n"
        f"{field_content}\n\n"
        "Respond with ONLY:\n"
        "<response>"
    )
    return _enforce_limit_with_regen(prompt, limit, model_id)

def summarise_all_fields(data: dict, model_id: int = 16) -> dict:
    summary = {"text": data.get("text", "")}
    summary["tasks"] = []
    for task in data.get("tasks", []):
        summary["tasks"].append({
            "task_description": summarize_field(task.get("task_description",""), "task_description", model_id),
            "agent_name": task.get("agent_name",""),
            "final_answer": summarize_field(task.get("final_answer",""), "final_answer", model_id),
            "review": summarize_field(task.get("review",""), "review", model_id),
        })
    return summary

# ======== runner ========
def process_trajectory(q_id: str, model_id: int = 16, save_summary: bool = True) -> dict:
    filename = f"{q_id}_trajectory.json"
    filepath = os.path.join(TRAJECTORY_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file for {q_id} not found: {filepath}")
    extracted = extract_trajectory_fields(filepath)
    summary = summarise_all_fields(extracted, model_id=model_id)
    if save_summary:
        out_dir = os.path.join(RESULT_DIR, "summary")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{q_id}_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary for {q_id} to {out_path}")
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract trajectory and produce concise summaries (choice/word-limited, no trimming).")
    parser.add_argument("q_id", type=str, help="Question ID, e.g., Q_1, Q_42")
    parser.add_argument("--model_id", type=int, default=16, help="LLM model ID")
    parser.add_argument("--no_save", action="store_true", help="Do not save summary file")
    args = parser.parse_args()
    result = process_trajectory(args.q_id, model_id=args.model_id, save_summary=not args.no_save)
    print(json.dumps(result, indent=2, ensure_ascii=False))
