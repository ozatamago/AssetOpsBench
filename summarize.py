# summarize_pipeline.py — extract + concise (choice/word-limited) summaries, no trimming

import os
import re
import json
from typing import Any, Dict
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

def _llm(prompt: str, model_id: int) -> Dict[str, Any]:
    """
    Thin wrapper around watsonx_llm that returns:
      {
        "generated_text": str,
        "input_token_count": int,
        "generated_token_count": int,
        "stop_reason": str | None,
      }
    """
    resp = watsonx_llm(prompt, model_id=model_id) or {}

    text = (resp.get("generated_text") or "").strip()
    in_tok = resp.get("input_token_count", 0)
    out_tok = resp.get("generated_token_count", 0)
    stop_reason = resp.get("stop_reason")

    return {
        "generated_text": text,
        "input_token_count": in_tok,
        "generated_token_count": out_tok,
        "stop_reason": stop_reason,
    }

from typing import Tuple, Set, Any, Dict

def _enforce_limit_with_regen(initial_prompt: str, word_limit: int, model_id: int) -> tuple[str, int, int]:
    """
    1回生成→語数・禁句チェック→超過なら一度だけやり直し（task_description等・一般用途）。

    Returns:
        (final_text, total_input_tokens, total_generated_tokens)
    """
    # 1st generation
    first = _llm(initial_prompt, model_id)  # dict from watsonx_llm wrapper
    out = (first.get("generated_text") or "").strip()
    in_tok_total = int(first.get("input_token_count", 0))
    out_tok_total = int(first.get("generated_token_count", 0))

    ok = (
        bool(out)
        and _word_count(out) <= word_limit
        and ("Input:" not in out)
        and ("Output" not in out)
    )
    if ok:
        return out, in_tok_total, out_tok_total

    # Need regen
    over = _word_count(out) if out else 0
    regen_prompt = (
        f"Your previous answer had {over} words. Rewrite to ≤ {word_limit} words.\n"
        "One line. No preamble or quotes. Do not include the words 'Input' or 'Output'.\n\n"
        "Previous answer:\n"
        f"{out or ''}\n\n"
        "Return ONLY the final line:"
    )

    second = _llm(regen_prompt, model_id)
    out2 = (second.get("generated_text") or "").strip()

    # accumulate token usage from regen
    in_tok_total += int(second.get("input_token_count", 0))
    out_tok_total += int(second.get("generated_token_count", 0))

    return out2, in_tok_total, out_tok_total


def _enforce_choice_with_regen(initial_prompt: str, allowed: set[str], model_id: int) -> tuple[str, int, int]:
    """
    集合 {EXPORT, NO_EXPORT} などに“厳密一致”させるための強制器（final_answer向け）。

    Returns:
        (final_token, total_input_tokens, total_generated_tokens)
    """
    # 1st generation
    first = _llm(initial_prompt, model_id)
    out = (first.get("generated_text") or "").strip()
    in_tok_total = int(first.get("input_token_count", 0))
    out_tok_total = int(first.get("generated_token_count", 0))

    if out in allowed:
        return out, in_tok_total, out_tok_total

    # 再生成
    regen_prompt = (
        "Your previous answer did not exactly match one of the allowed tokens.\n"
        f"Return EXACTLY one of: {', '.join(sorted(allowed))}.\n"
        "No punctuation, no spaces, no preamble, one line only.\n\n"
        "Previous answer:\n"
        f"{out}\n\n"
        "Respond with ONLY the token:"
    )

    second = _llm(regen_prompt, model_id)
    out2 = (second.get("generated_text") or "").strip()

    in_tok_total += int(second.get("input_token_count", 0))
    out_tok_total += int(second.get("generated_token_count", 0))

    return out2, in_tok_total, out_tok_total


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
from typing import Tuple
import re

def summarize_field_with_tokens(
    field_content: str,
    field_name: str,
    model_id: int = 16,
) -> tuple[str, int, int]:
    """
    Token-aware version of summarize_field.

    Returns:
        (summary_text, input_token_count, generated_token_count)
    """
    if not field_content:
        return "", 0, 0

    # These fields are not summarized at all, so no tokens are used.
    if field_name in ("text", "agent_name"):
        return field_content, 0, 0

    # ---- task_description ----
    if field_name == "task_description":
        limit = WORD_LIMITS["task_description"]
        prompt = _prompt_task_description(field_content, limit)
        text, in_tok, out_tok = _enforce_limit_with_regen(prompt, limit, model_id)
        return text, in_tok, out_tok

    # ---- final_answer ----
    if field_name == "final_answer":
        lower = field_content.lower()
        looks_like_export_rule = "exported" in lower or "export" in lower

        # Branch 1: EXPORT / NO_EXPORT classification
        if looks_like_export_rule:
            prompt = _prompt_final_answer_export_rule(field_content)
            token, in_tok, out_tok = _enforce_choice_with_regen(
                prompt,
                {"EXPORT", "NO_EXPORT"},
                model_id,
            )
            token = token.strip()
            if token not in {"EXPORT", "NO_EXPORT"}:
                token = "EXPORT" if "exported" in lower else "NO_EXPORT"
            return token, in_tok, out_tok

        # Branch 2: generic summarization
        limit = WORD_LIMITS["final_answer"]
        prompt = _prompt_final_answer_generic(field_content, limit)
        text, in_tok, out_tok = _enforce_limit_with_regen(prompt, limit, model_id)
        return text, in_tok, out_tok

    # ---- review ----
    if field_name == "review":
        # Preserve "Task Status: ..." and only summarize the rest.
        m = re.search(r"(Task Status:\s*[^\n\r]+)", field_content)
        status_prefix = m.group(1).strip() if m else "Task Status: Unknown"
        body = field_content.replace(status_prefix, "", 1).strip()

        body_prompt = _prompt_review_body(body or "", WORD_LIMITS["review"])
        body_out, in_tok, out_tok = _enforce_limit_with_regen(
            body_prompt,
            WORD_LIMITS["review"],
            model_id,
        )
        body_out = body_out.strip()
        full = f"{status_prefix} | {body_out}" if body_out else status_prefix
        return full, in_tok, out_tok

    # ---- fallback ----
    limit = 50
    prompt = (
        f"Summarize in one concise line (≤ {limit} words). Keep key facts; no preamble.\n\n"
        f"{field_content}\n\n"
        "Respond with ONLY:\n"
        "<response>"
    )
    text, in_tok, out_tok = _enforce_limit_with_regen(prompt, limit, model_id)
    return text, in_tok, out_tok


from typing import Dict, Any, Tuple

def summarise_all_fields(
    data: dict,
    model_id: int = 16,
) -> tuple[dict, int, int]:
    """
    Summarize all relevant fields in a trajectory JSON.

    Returns:
        summary_dict, total_input_tokens, total_generated_tokens

    summary_dict has the same shape as before:
        {
          "text": ...,
          "tasks": [
            {
              "task_description": ...,
              "agent_name": ...,
              "final_answer": ...,
              "review": ...,
            },
            ...
          ]
        }
    """
    total_in_tokens = 0
    total_out_tokens = 0

    summary: Dict[str, Any] = {"text": data.get("text", "")}
    summary["tasks"] = []

    for task in data.get("tasks", []) or []:
        # task_description
        td_summary, td_in, td_out = summarize_field_with_tokens(
            task.get("task_description", "") or "",
            "task_description",
            model_id,
        )
        total_in_tokens += td_in
        total_out_tokens += td_out

        # final_answer
        fa_summary, fa_in, fa_out = summarize_field_with_tokens(
            task.get("final_answer", "") or "",
            "final_answer",
            model_id,
        )
        total_in_tokens += fa_in
        total_out_tokens += fa_out

        # review
        review_summary, rv_in, rv_out = summarize_field_with_tokens(
            task.get("review", "") or "",
            "review",
            model_id,
        )
        total_in_tokens += rv_in
        total_out_tokens += rv_out

        summary["tasks"].append(
            {
                "task_description": td_summary,
                "agent_name": task.get("agent_name", "") or "",
                "final_answer": fa_summary,
                "review": review_summary,
            }
        )

    return summary, total_in_tokens, total_out_tokens

# ======== runner ========
def process_trajectory(q_id: str, model_id: int = 16, save_summary: bool = True) -> dict:
    filename = f"Q_{q_id}_trajectory.json"
    filepath = os.path.join(TRAJECTORY_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file for {q_id} not found: {filepath}")
    extracted = extract_trajectory_fields(filepath)
    summary, total_in_tokens, total_out_tokens = summarise_all_fields(extracted, model_id=model_id)
    if save_summary:
        out_dir = os.path.join(RESULT_DIR, "summary")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{q_id}_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary for {q_id} to {out_path}")
    return summary, total_in_tokens, total_out_tokens

# summarize.py などに追加

import os
import re
from typing import Callable, Sequence, Optional

import psycopg  # psycopg3 を想定

# Watsonx
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

# グローバルキャッシュ
_WX_EMBED_FN: Optional[Callable[[str], list[float]]] = None


from typing import Callable, List, Optional
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure

# グローバルキャッシュ（必要なら）
_WX_SUMMARY_EMBED_FN: Optional[Callable[[str], List[float]]] = None


def _init_watsonx_embedder_from_env_for_summary() -> Callable[[str], List[float]]:
    """
    Watsonx.ai の埋め込みクライアントを初期化し、
    text -> list[float] (長さ = DB_EMBEDDING_DIM) を返す関数を作る。

    WATSONX_EMBEDDING_MODEL は 2 通りを受け付ける:
      1. EmbeddingTypes の属性名 (例: IBM_SLATE_30M_ENG)
      2. API の model_id 文字列 (例: ibm/slate-30m-english-rtrvr-v2)
    """
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    # デフォルトは v2 モデルを想定
    raw_model = os.getenv(
        "WATSONX_EMBEDDING_MODEL",
        "ibm/slate-30m-english-rtrvr-v2",
    )

    if not (api_key and project_id):
        raise RuntimeError("WATSONX_APIKEY / WATSONX_PROJECT_ID が未設定です。")

    # 1) "ibm/..." で始まっていたら「API の model_id そのもの」として解釈
    if raw_model.lower().startswith("ibm/"):
        model_id = raw_model
    else:
        # 2) それ以外は EmbeddingTypes の属性名とみなす
        try:
            enum_obj = getattr(EmbeddingTypes, raw_model)
        except AttributeError:
            raise RuntimeError(
                f"未知の埋め込みモデル指定: {raw_model} "
                f"(EmbeddingTypes.* か 'ibm/...-english-rtrvr(-v2)' で指定して下さい)"
            )
        # enum.value に本当の model_id 文字列が入っていることが多い
        model_id = getattr(enum_obj, "value", enum_obj)

    creds = Credentials(api_key=api_key, url=url)
    emb = Embeddings(model_id=model_id, credentials=creds, project_id=project_id)

    db_dim = int(os.getenv("DB_EMBEDDING_DIM", "1536"))

    def _coerce_to_db_dim(vec: List[float]) -> List[float]:
        """
        Watsonx の生の埋め込みベクトル（例: 384 or 768 次元）を、
        Postgres 側の vector(DB_EMBEDDING_DIM) に合わせるために
        pad/truncate する。
        """
        if len(vec) == db_dim:
            return vec
        if len(vec) > db_dim:
            return vec[:db_dim]
        return vec + [0.0] * (db_dim - len(vec))

    def _wx_embed(text: str) -> List[float]:
        v = emb.embed_query(text)   # Watsonx から list[float] が返る
        return _coerce_to_db_dim(v)

    # 一応、ping で疎通確認（ここで 404 ならすぐ分かる）
    try:
        _ = _wx_embed("ping")
    except ApiRequestFailure as e:
        raise RuntimeError(
            f"[summary] watsonx.ai 埋め込みモデル '{model_id}' がこの環境ではサポートされていません。\n"
            f"- IBM Cloud の watsonx.ai プロジェクトで利用可能な Embedding モデル一覧を確認し、\n"
            f"  そこに載っている model_id を WATSONX_EMBEDDING_MODEL に設定してください。"
        ) from e

    return _wx_embed

def get_watsonx_embedder_for_summary() -> Callable[[str], list[float]]:
    """
    summarize.py から呼ぶためのラッパー。
    最初の一回だけ Watsonx を初期化し、以降は同じ関数を返す。
    """
    global _WX_EMBED_FN
    if _WX_EMBED_FN is None:
        _WX_EMBED_FN = _init_watsonx_embedder_from_env_for_summary()
    return _WX_EMBED_FN

def build_task_summary(user_question: str,
                       task_description: str,
                       agent_name: str,
                       final_answer: str,
                       review: str) -> str:
    """
    1つのタスクについて、人間が読める1〜数文の要約を作る。
    task_description, final_answer, review は既に summarize_field で短くなっている前提。
    """
    status = None
    if review:
        m = re.search(r"Task Status:\s*([^|]+)", review)
        if m:
            status = m.group(1).strip()

    parts = []

    if status:
        parts.append(f"Status: {status}.")
    if agent_name and task_description:
        parts.append(f"Agent {agent_name} executed: {task_description}.")
    elif task_description:
        parts.append(task_description)

    if final_answer:
        parts.append(f"Output: {final_answer}.")

    # 軽く質問のコンテキストも入れておく（検索・類似度に効く）
    if user_question:
        parts.append(f"User asked: {user_question}")

    return " ".join(p.strip() for p in parts if p)

def save_summarized_task(summary: dict,
                         json_id: int,
                         db_url: str | None = None) -> None:
    """
    task-summary JSON（process_trajectory の出力）を受け取り、
    Watsonx 埋め込みを使って traj_task_summaries に保存する。
    """
    user_question = summary.get("text", "")
    tasks = summary.get("tasks", []) or []

    if db_url is None:
        db_url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DB_URL / DATABASE_URL が環境変数に設定されていません。")

    # Watsonx 埋め込み関数を初期化（1回目だけ API 呼び出し）
    wx_embed = get_watsonx_embedder_for_summary()

    print(f"[save_summarized_task] START json_id={json_id}, num_tasks_in_summary={len(tasks)}")
    # Postgres 接続
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # 1) json_id から doc_id を引く
            cur.execute(
                "SELECT doc_id FROM traj_docs WHERE json_id = %s",
                (json_id,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"[save_summarized_task] ERROR: No traj_docs row found for json_id={json_id}")
                raise ValueError(f"No traj_docs row found for json_id={json_id}")
            (doc_id,) = row
            print(f"[save_summarized_task] Found doc_id={doc_id} for json_id={json_id}")

            # 2) その doc_id に対応する traj_tasks を task_number 順に取得
            cur.execute(
                """
                SELECT task_id, task_number
                FROM traj_tasks
                WHERE doc_id = %s
                ORDER BY task_number ASC
                """,
                (doc_id,),
            )
            db_tasks = cur.fetchall()
            print(
                f"[save_summarized_task] DB has {len(db_tasks)} traj_tasks rows "
                f"for doc_id={doc_id}"
            )

        if len(db_tasks) != len(tasks):
            print(
                f"[save_summarized_task] WARNING: DB has {len(db_tasks)} tasks "
                f"but summary has {len(tasks)} tasks for json_id={json_id}"
            )

        upsert_count = 0  # 何件 upsert したかカウント

        # 3) 各タスクについて traj_task_summaries に upsert
        with conn.cursor() as cur:
            for idx, (task_id, task_number) in enumerate(db_tasks):
                if idx >= len(tasks):
                    print(
                        f"[save_summarized_task] INFO: Reached end of summary tasks "
                        f"(idx={idx}), stopping loop."
                    )
                    break

                t = tasks[idx]
                task_desc = t.get("task_description", "") or ""
                agent_name = t.get("agent_name", "") or ""
                final_answer = t.get("final_answer", "") or ""
                review = t.get("review", "") or ""

                # review から status を抽出 ("Task Status: X | ...")
                m = re.search(r"Task Status:\s*([^|]+)", review)
                status = m.group(1).strip() if m else None

                # 1行の summary 文を作る
                summary_text = build_task_summary(
                    user_question=user_question,
                    task_description=task_desc,
                    agent_name=agent_name,
                    final_answer=final_answer,
                    review=review,
                )
                MAX_SUMMARY_CHARS = 1000
                if len(summary_text) > MAX_SUMMARY_CHARS:
                    summary_text = summary_text[:MAX_SUMMARY_CHARS]

                # Watsonx 埋め込み → DB_EMBEDDING_DIM に揃った list[float]
                summary_vec = wx_embed(summary_text)

                # 短くログに出したいので summary_text は先頭だけ
                print(
                    f"[save_summarized_task] Upserting task idx={idx}, "
                    f"task_number={task_number}, task_id={task_id}, "
                    f"agent_name='{agent_name}', "
                    f"summary_preview='{summary_text[:80]}...'"
                )

                cur.execute(
                    """
                    INSERT INTO traj_task_summaries (
                        doc_id,
                        task_id,
                        task_number,
                        user_question,
                        task_description,
                        agent_name,
                        final_answer,
                        status,
                        review,
                        summary,
                        summary_vec
                    )
                    VALUES (
                        %(doc_id)s,
                        %(task_id)s,
                        %(task_number)s,
                        %(user_question)s,
                        %(task_description)s,
                        %(agent_name)s,
                        %(final_answer)s,
                        %(status)s,
                        %(review)s,
                        %(summary)s,
                        %(summary_vec)s
                    )
                    ON CONFLICT (doc_id, task_id) DO UPDATE
                    SET
                        task_number      = EXCLUDED.task_number,
                        user_question    = EXCLUDED.user_question,
                        task_description = EXCLUDED.task_description,
                        agent_name       = EXCLUDED.agent_name,
                        final_answer     = EXCLUDED.final_answer,
                        status           = EXCLUDED.status,
                        review           = EXCLUDED.review,
                        summary          = EXCLUDED.summary,
                        summary_vec      = EXCLUDED.summary_vec
                    """,
                    {
                        "doc_id": doc_id,
                        "task_id": task_id,
                        "task_number": task_number,
                        "user_question": user_question,
                        "task_description": task_desc,
                        "agent_name": agent_name,
                        "final_answer": final_answer,
                        "status": status,
                        "review": review,
                        "summary": summary_text,
                        "summary_vec": summary_vec,
                    },
                )
                upsert_count += 1

            conn.commit()

    print(
        f"[save_summarized_task] DONE json_id={json_id}, doc_id={doc_id}, "
        f"upserted_rows={upsert_count}"
    )



# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Extract trajectory and produce concise summaries (choice/word-limited, no trimming).")
#     parser.add_argument("q_id", type=str, help="Question ID, e.g., Q_1, Q_42")
#     parser.add_argument("--model_id", type=int, default=16, help="LLM model ID")
#     parser.add_argument("--no_save", action="store_true", help="Do not save summary file")
#     args = parser.parse_args()
#     result = process_trajectory(args.q_id, model_id=args.model_id, save_summary=not args.no_save)
#     print(json.dumps(result, indent=2, ensure_ascii=False))

