from __future__ import annotations
from agent_hive.task import Task
from pydantic import Field
from typing import Dict, Any, List
from agent_hive.enum import ContextType
from pathlib import Path
from datetime import datetime
import json, os, tempfile, logging
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.workflows.simulator_agent import SimulatorAgent, SIMULATOR_SYSTEM_PROMPT
from agent_hive.workflows.critic_agent import CriticAgent
from agent_hive.logger import get_custom_logger
import os
import uuid
import psycopg  # psycopg3
from psycopg.rows import dict_row
import json
import os
from typing import Any, Dict, List, Optional
from decimal import Decimal
from uuid import UUID
from datetime import date, datetime
import psycopg
from psycopg.rows import dict_row
# from simulator_agent import Simulator
# from critic_agent import Critic
logger = get_custom_logger(__name__)

# =========================================================
# TODO: Participants can edit this section ONLY
# Add variable, dict. no more any import just any inline code
# =========================================================
# END OF EDITABLE SECTION

RESULT_DIR = "/home/track1_result/"
PLAN_DIR = RESULT_DIR + "plan/"

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
        
    def _init_watsonx_embedder_from_env(self) -> None:
        """
        Create self.watsonx_embed(text, model_id=None) -> list[float (len=DB_EMBEDDING_DIM)].
        Uses IBM watsonx.ai Embeddings API and pads/truncates to DB_EMBEDDING_DIM.

        WATSONX_EMBEDDING_MODEL は 2 通りを受け付ける:
        1. EmbeddingTypes の属性名 (例: IBM_SLATE_30M_ENG)
        2. API の model_id 文字列 (例: ibm/slate-30m-english-rtrvr-v2)
        """
        import os
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings
        from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
        from ibm_watsonx_ai.wml_client_error import ApiRequestFailure

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

        # 1. "ibm/..." で始まっていたら「API の model_id そのもの」と解釈
        if raw_model.lower().startswith("ibm/"):
            model_id = raw_model
        else:
            # 2. そうでなければ EmbeddingTypes の属性名だとみなす
            try:
                enum_obj = getattr(EmbeddingTypes, raw_model)
            except AttributeError:
                raise RuntimeError(
                    f"未知の埋め込みモデル指定: {raw_model} "
                    f"(EmbeddingTypes.* か 'ibm/...-english-rtrvr(-v2)' で指定して下さい)"
                )
            # enum の .value に実際の model_id が入っていることが多い
            model_id = getattr(enum_obj, "value", enum_obj)

        creds = Credentials(api_key=api_key, url=url)
        emb = Embeddings(model_id=model_id, credentials=creds, project_id=project_id)

        db_dim = int(os.getenv("DB_EMBEDDING_DIM", "1536"))

        def _coerce_to_db_dim(vec: list[float]) -> list[float]:
            if len(vec) == db_dim:
                return vec
            if len(vec) > db_dim:
                return vec[:db_dim]
            return vec + [0.0] * (db_dim - len(vec))

        def _wx_embed(text: str, model_id=None) -> list[float]:
            v = emb.embed_query(text)   # returns list[float]
            return _coerce_to_db_dim(v)

        # wire up
        self.watsonx_embed = _wx_embed
        self.embed_model_id = model_id

        # quick probe: ここで 404 等が出たら、わかりやすいメッセージで止める
        try:
            _ = self.watsonx_embed("ping")
        except ApiRequestFailure as e:
            raise RuntimeError(
                f"watsonx.ai 埋め込みモデル '{model_id}' がこの環境ではサポートされていません。\n"
                f"- IBM Cloud コンソールの watsonx.ai プロジェクトで、利用可能な Embedding モデル一覧を確認して下さい。\n"
                f"- その上で WATSONX_EMBEDDING_MODEL を、一覧にある model_id "
                f"(例: 'ibm/slate-30m-english-rtrvr-v2' や 'ibm/slate-125m-english-rtrvr') に変更して下さい。"
            ) from e


    def run(self, save_plan=False, saved_plan_prefix="", qid=None):
        generated_steps, run_id, plan_id, input_tokens_count, generated_tokens_count = self.generate_steps(
            save_plan=save_plan,
            saved_plan_filename=saved_plan_prefix,
            qid=qid
        )

        sequential_workflow = SequentialWorkflow(
            tasks=generated_steps, context_type=ContextType.SELECTED
        )

        return sequential_workflow.run(), run_id, plan_id, input_tokens_count, generated_tokens_count

    # def generate_steps(self, save_plan=False, saved_plan_filename=""):
    #     task = self.tasks[0]
    #     agent_descriptions = ""

    #     # =========================================================
    #     # TODO: Participants can edit this section ONLY
    #     # 🎨 Purpose: Customize how agent information is collected and formatted
    #     # ✅ Allowed: 
    #     #     - Change numbering style or bullet points
    #     #     - Include additional metadata (e.g., agent capabilities, tags)
    #     #     - Provide examples in a different format
    #     #     - Add emojis or formatting to make the prompt clearer 
    #     #     - More thinking
    #     # ❌ Not allowed: 
    #     #     - Modify workflow execution
    #     #     - Replace the base ReAct agent or Executor
    #     #     - Change memory or retry logic
    #     # =========================================================

    #     for ii, aagent in enumerate(task.agents):
    #         agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
    #         agent_descriptions += f"\nAgent description: {aagent.description}"
    #         if "task_examples" in aagent.__dict__ and aagent.task_examples:
    #             agent_descriptions += f"\nTasks that agent can solve:"
    #             for idx, task_example in enumerate(aagent.task_examples, start=1):
    #                 agent_descriptions += f"\n{idx}. {task_example}"
    #         agent_descriptions += "\n"

    #     # =========================================================
    #     # END OF EDITABLE SECTION
    #     # 🚫 Participants should not modify code below this line
    #     # ❌ No new variables, functions, or workflow logic allowed
    #     # ✅ Only modify the section marked as TODO above
    #     # =========================================================

    #     prompt = self.get_prompt(task.description, agent_descriptions)
    #     logger.info(f"Plan Generation Prompt: \n{prompt}")
    #     llm_response = watsonx_llm(
    #         prompt, model_id=self.llm,
    #     )["generated_text"]
    #     logger.info(f"Plan: \n{llm_response}")

    #     final_plan = llm_response
    #     self.memory = []

    #     task_pattern = r"#Task\d+: (.+)"
    #     agent_pattern = r"#Agent\d+: (.+)"
    #     dependency_pattern = r"#Dependency\d+: (.+)"
    #     output_pattern = r"#ExpectedOutput\d+: (.+)"

    #     tasks = re.findall(task_pattern, final_plan)
    #     agents = re.findall(agent_pattern, final_plan)
    #     dependencies = re.findall(dependency_pattern, final_plan)
    #     outputs = re.findall(output_pattern, final_plan)

    #     if save_plan:
    #         if not saved_plan_filename.endswith(".txt"):
    #             saved_plan_filename += ".txt"

    #         saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
    #         with open(saved_plan_filename, "w") as f:
    #             f.write(saved_plan_text)

    #     planned_tasks = []
    #     for i in range(len(tasks)):
    #         task_description = tasks[i]
    #         if i == len(agents):
    #             break
    #         agent_name = agents[i]
    #         if i < len(dependencies):
    #             dependency = dependencies[i]
    #         else:
    #             dependency = "None"
    #         if i < len(outputs):
    #             expected_output = outputs[i]
    #         else:
    #             expected_output = ""

    #         selected_agent = None
    #         for agent in task.agents:
    #             if agent.name == agent_name:
    #                 selected_agent = agent
    #                 break
    #         if selected_agent is None:
    #             selected_agent = task.agents[0]

    #         if dependency != "None":
    #             numbers = re.findall(r"#S(\d+)", dependency)
    #             numbers = list(map(int, numbers))
    #             context = [planned_tasks[i - 1] for i in numbers]
    #         else:
    #             context = []

    #         a_task = Task(
    #             description=task_description,
    #             expected_output=expected_output,
    #             agents=[selected_agent],
    #             context=context,
    #         )
    #         planned_tasks.append(a_task)

    #     logger.info(f"Planned Tasks: \n{planned_tasks}")

    #     return planned_tasks

    # search_triplets.py

    # --- 可搬なヘルパー ---

    def _vector_literal(self, vec: List[float]) -> str:
        """pgvector の文字列表現 [v1, v2, ...] を生成"""
        return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"

    from typing import List, Optional, Tuple

    def _extract_keywords_with_llm(
            self, 
            user_question: str,
            *,
            llm_fn,                  # 例: watsonx_llm
            llm_model_id: str,       # 例: "16"
            known: Optional[List[str]] = None
        ) -> Tuple[List[str], int, int]:
        """
        LLMでキーワード抽出（websearch_to_tsquery 用の語彙を返す）

        Returns:
            (keywords, input_token_count, generated_token_count)
        """
        ctx = ""
        if known:
            ctx = "Known related terms: " + ", ".join(known[:6]) + "\n"

        prompt = (
            "You extract search keywords for a Postgres retrieval system with full-text search "
            "(websearch_to_tsquery, simple config) and pgvector.\n"
            "Return ONLY a JSON array of 5–12 strings. No prose, no markdown.\n"
            "Rules: keep domain terms, short 2–3 word phrases, lowercase except acronyms, "
            "avoid stopwords, deduplicate.\n\n"
            f"{ctx}"
            f"QUESTION:\n{user_question}\n"
        )

        # ★ llm_fn をちゃんと使う or self.llm のままでもOK（今の実装に合わせる）
        resp = self.llm(prompt, model_id=int(llm_model_id))
        text = resp.get("generated_text", "").strip()
        in_tok = int(resp.get("input_token_count", 0))
        out_tok = int(resp.get("generated_token_count", 0))

        # --- JSON で返ってきた場合 ---
        try:
            kws = json.loads(text)
            if isinstance(kws, list) and all(isinstance(x, str) for x in kws):
                out, seen = [], set()
                for k in (s.strip() for s in kws if s and s.strip()):
                    key = k.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(k)
                return out, in_tok, out_tok
        except Exception:
            pass

        # --- フォールバック：カンマ区切り ---
        parts = [p.strip() for p in text.replace("\n", " ").split(",")]
        out, seen = [], set()
        for k in parts:
            if not k:
                continue
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(k)

        if not out:
            out = [user_question]

        return out, in_tok, out_tok


    def _get_embedding(self, text: str, *, embed_fn) -> List[float]:
        """埋め込みベクトルを取得（1536次元など、実装に依存）"""
        if not callable(embed_fn):
            raise RuntimeError("embed_fn is required and must be callable")
        vec = embed_fn(text)
        if not isinstance(vec, (list, tuple)):
            raise TypeError("embed_fn must return list[float]")
        return list(vec)

    # --- DBハイドレーション ---

    def _fetch_traj_doc(self, conn, doc_id: str) -> Optional[dict]:
        """traj_docs + traj_tasks を取得"""
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT d.doc_id, d.json_id, d.text, d.tsv_all, d.text_vec
                FROM traj_docs d
                WHERE d.doc_id = %s
                LIMIT 1;
            """, (doc_id,))
            doc = cur.fetchone()
            if not doc:
                return None
            cur.execute("""
                SELECT task_id, doc_id, task_number, task_description, final_answer
                FROM traj_tasks
                WHERE doc_id = %s
                ORDER BY task_number ASC;
            """, (doc_id,))
            tasks = cur.fetchall()
        doc["tasks"] = tasks
        return doc

    def _fetch_latest_score(self, conn, doc_id: str) -> Optional[dict]:
        """dag_trajectory_score から doc_id 最新スコア（created_at DESC）"""
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT score_id, plan_id, doc_id,
                    num_edit, correct, num_partially, num_not, error_analysis, created_at
                FROM dag_trajectory_score
                WHERE doc_id = %s
                ORDER BY created_at DESC
                LIMIT 1;
            """, (doc_id,))
            return cur.fetchone()

    def _fetch_dag_by_plan(self, conn, plan_id: str) -> Optional[dict]:
        """plan_dag_rounds から DAG を取得（最新 1 件想定）"""
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT
                plan_id,
                scenario_id,
                created_at AS generated_at,
                dag        AS final_plan
                FROM plan_dag_rounds
                WHERE plan_id = %s
                ORDER BY round_t DESC
                LIMIT 1;
            """, (plan_id,))
            return cur.fetchone()

    # --- 主関数：検索して (DAG, trajectory, score) を束ねる ---

    from typing import Any, Dict, List, Optional, Tuple

    def search_similar_triplets(
        self, 
        db_url: str,
        user_question: str,
        *,
        llm_fn,                # 例: watsonx_llm（今は self.llm を使っているが型は残す）
        llm_model_id: str,     # 例: "16"
        embed_fn=None,         # 例: lambda s: watsonx_embed(s, model_id=...)
        fts_limit: int = 3,
        vector_k: int = 1,
        max_triplets: int = 5,
    ) -> tuple[dict, int, int]:
        """
        1) キーワード抽出 -> websearch_to_tsquery(simple) で traj_tasks + traj_docs を FTS
        2) pgvector で traj_docs.text_vec のベクトル近傍
        3) doc_id をユニーク化して、(trajectory, 最新 score, DAG) をハイドレート

        戻り値:
            out_dict, total_input_tokens, total_generated_tokens

        out_dict の構造:
        {
        "keywords": [...],
        "fts_hits": [ {src, doc_id, rank, ...}, ... ],
        "vector_hits": [ {doc_id, sim, ...} ] | [],
        "triplets": [
            {
            "doc_id": "...",
            "plan_id": "...",         # スコアがあれば
            "trajectory": {...},      # traj_docs + tasks
            "dag": {"final_plan": "...", ...} | None,
            "score": {num_edit, correct, num_partially, num_not, error_analysis, ...} | None
            }, ...
        ]
        }
        """
        total_in_tok = 0
        total_out_tok = 0

        # 0) キーワード（ここで LLM を 1 回使う）
        keywords, kw_in_tok, kw_out_tok = self._extract_keywords_with_llm(
            user_question, llm_fn=llm_fn, llm_model_id=llm_model_id
        )
        total_in_tok += kw_in_tok
        total_out_tok += kw_out_tok

        # websearch_to_tsquery 用 OR 連結
        tsquery_str = " OR ".join(
            f"\"{k.replace('\"','')}\"" for k in keywords if k.strip()
        )

        # 1) ベクトル用（ここは embedding API なので、LLM トークンとは別扱い）
        vec_lit = None
        if callable(embed_fn):
            q_vec = self._get_embedding(user_question, embed_fn=embed_fn)
            vec_lit = self._vector_literal(q_vec)

        out: Dict[str, Any] = {
            "keywords": keywords,
            "fts_hits": [],
            "vector_hits": [],
            "triplets": [],
        }

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            # --- 1) FTS（traj_tasks + traj_docs） ---
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq),
                    t_hits AS (
                        SELECT
                            'traj_tasks' AS src,
                            tt.doc_id,
                            tt.task_id,
                            tt.task_number,
                            ts_rank(tt.tsv_task, (SELECT tsq FROM q)) AS rank
                        FROM traj_tasks tt
                        WHERE tt.tsv_task @@ (SELECT tsq FROM q)
                        ORDER BY rank DESC
                        LIMIT 10
                    ),
                    d_hits AS (
                        SELECT
                            'traj_docs' AS src,
                            td.doc_id,
                            NULL::uuid    AS task_id,
                            NULL::int     AS task_number,
                            ts_rank(td.tsv_all, (SELECT tsq FROM q)) AS rank
                        FROM traj_docs td
                        WHERE td.tsv_all @@ (SELECT tsq FROM q)
                        ORDER BY rank DESC
                        LIMIT 10
                    )
                    SELECT * FROM (
                        SELECT * FROM t_hits
                        UNION ALL
                        SELECT * FROM d_hits
                    ) u
                    ORDER BY rank DESC
                    LIMIT %s;
                    """,
                    (tsquery_str, fts_limit),
                )
                out["fts_hits"] = [dict(r) for r in cur.fetchall()]

            # --- 2) ベクトル検索（traj_docs.text_vec） ---
            if vec_lit is not None and vector_k > 0:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH params AS (SELECT CAST(%s AS vector) AS qv)
                        SELECT
                            td.doc_id, td.json_id, td.text,
                            (1 - (td.text_vec <=> (SELECT qv FROM params))) AS sim
                        FROM traj_docs td
                        WHERE td.text_vec IS NOT NULL
                        ORDER BY td.text_vec <=> (SELECT qv FROM params)
                        LIMIT %s;
                        """,
                        (vec_lit, vector_k),
                    )
                    out["vector_hits"] = [dict(r) for r in cur.fetchall()]

            # --- 3) ハイドレート（doc_id をユニーク化して上位 max_triplets 件） ---
            doc_order: List[str] = []
            for h in out["fts_hits"]:
                did = h.get("doc_id")
                if did and did not in doc_order:
                    doc_order.append(did)
            for h in out["vector_hits"]:
                did = h.get("doc_id")
                if did and did not in doc_order:
                    doc_order.append(did)

            for doc_id in doc_order[:max_triplets]:
                trip: Dict[str, Any] = {
                    "doc_id": doc_id,
                    "plan_id": None,
                    "trajectory": None,
                    "dag": None,
                    "score": None,
                }
                traj = self._fetch_traj_doc(conn, doc_id)
                if traj:
                    trip["trajectory"] = traj

                score = self._fetch_latest_score(conn, doc_id)
                if score:
                    trip["score"] = score
                    trip["plan_id"] = score["plan_id"]
                    dag = self._fetch_dag_by_plan(conn, score["plan_id"])
                    if dag:
                        trip["dag"] = dag

                out["triplets"].append(trip)

        # LLM を使っているのは現状キーワード抽出だけなので、
        # total_in_tok / total_out_tok は kw_in_tok / kw_out_tok の値になる。
        # 将来、ここで追加の LLM 呼び出しをするときは、同じように加算していけばOK。
        return out, total_in_tok, total_out_tok


    def _clip(self, s: Optional[str], n: int) -> str:
        if not s:
            return ""
        s = " ".join(str(s).split())
        return s if len(s) <= n else (s[: n - 1] + "…")

    def _json_default(self, o):
        # json.dumps(..., default=_json_default) から呼ばれるフォールバック
        if isinstance(o, Decimal):
            # 数値として扱いたいなら float へ（丸め/桁落ちが困るなら str に）
            return float(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, UUID):
            return str(o)
        # 最後の手段（未知型は文字列化）
        return str(o)

    def _coerce_for_json(self, obj):
        # 再帰的に Decimal 等を潰す（default だけでも可だが二重に堅くする）
        if isinstance(obj, dict):
            return {k: self._coerce_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._coerce_for_json(v) for v in obj]
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return obj


    # 落ちている箇所（例：line 487 付近）の置き換え
    def _minify_json(self, obj, max_chars: int = 2000) -> str:
        obj = self._coerce_for_json(obj)  # ← 先に潰す
        s = json.dumps(
            obj,
            ensure_ascii=False,
            separators=(",", ":"),
            default=self._json_default,     # ← 取りこぼしを拾う保険
        )
        return s[:max_chars]


    def _trajectory_summary_for_prompt(
        self,
        traj_doc: Dict[str, Any],
        *,
        max_chars: int = 1200,
        prefer_summarizer: bool = True,
    ) -> str:
        """
        traj_docs (+ tasks) をプロンプト向けに短縮。
        既存の self.summarise_all_fields があればそれを使用（textはそのまま抽出）し、
        なければ簡易要約（text + 先頭タスク数件の要点のみ）。
        """
        try:
            # 既存の抽出フォーマットに似せて生成（text と tasks 配列）
            extracted = {
                "text": traj_doc.get("text", ""),
                "tasks": [
                    {
                        "task_description": t.get("task_description", ""),
                        "final_answer": t.get("final_answer", ""),
                        "review": t.get("review", ""),
                    }
                    for t in (traj_doc.get("tasks") or [])
                ],
            }

            if prefer_summarizer and hasattr(self, "summarise_all_fields"):
                # text は要約せずそのまま、他フィールドは短縮（あなたの実装仕様）
                summed = self.summarise_all_fields(extracted, model_id=getattr(self, "llm_model_id", 16))
                return self._minify_json(summed, max_chars)
            else:
                # フォールバック：手作業で短縮
                tasks = extracted["tasks"][:3]  # 先頭3件
                for t in tasks:
                    t["task_description"] = self._clip(t["task_description"], 140)
                    t["final_answer"]     = self._clip(t["final_answer"], 120)
                    t["review"]           = self._clip(t["review"], 160)
                preview = {"text": extracted["text"], "tasks": tasks}
                return self._minify_json(preview, max_chars)
        except Exception:
            # どんな入力でも落ちないように
            return self._minify_json({"text": traj_doc.get("text", "")}, max_chars)

    # ================== [NEW] formatter: triplets -> few-shot prompt block ==================
    def _format_triplets_for_prompt(
        self,
        search_hits: Dict[str, Any],
        *,
        per_triplets: int = 3,
        dag_chars: int = 1500,
        traj_chars: int = 1200,
        score_chars: int = 600,
    ) -> str:
        """
        search_similar_triplets() の返り値 search_hits["triplets"] を
        Few-shot 用 <examples> ブロックに整形する。

        生成形式（XML風の明確な区切り）:
        <examples>
          <example>
            <dag>...</dag>
            <trajectory>...</trajectory>
            <gold_score_json>{"num_edit":..,"correct":..,"error_analysis":"..."}</gold_score_json>
          </example>
          ...
        </examples>
        """
        trips: List[Dict[str, Any]] = search_hits.get("triplets") or []
        blocks: List[str] = []

        for ex in trips[:per_triplets]:
            # --- DAG ---
            dag_obj = None
            if isinstance(ex.get("dag"), dict) and ex["dag"].get("final_plan"):
                # plan_dag_rounds.final_plan は文字列 (DAGスクリプト) の想定
                # 可能なら JSON として minify、無理ならそのまま切り詰め
                raw = ex["dag"]["final_plan"]
                try:
                    dag_obj = json.loads(raw)
                    dag_min = self._minify_json(dag_obj, dag_chars)
                except Exception:
                    dag_min = self._clip(raw, dag_chars)
            else:
                dag_min = ""

            # --- trajectory (要約) ---
            traj_min = ""
            if isinstance(ex.get("trajectory"), dict):
                traj_min = self._trajectory_summary_for_prompt(ex["trajectory"], max_chars=traj_chars, prefer_summarizer=True)

            # --- score (gold) ---
            score_obj = ex.get("score") or {}
            gold = {
                "num_edit": score_obj.get("num_edit", 0),
                "correct": score_obj.get("correct", 0),
                "error_analysis": score_obj.get("error_analysis", ""),
            }
            gold_txt = self._minify_json(gold, score_chars)

            block = (
                "<example>\n"
                "<dag>\n" + dag_min + "\n</dag>\n"
                "<trajectory>\n" + traj_min + "\n</trajectory>\n"
                "<gold_score_json>\n" + gold_txt + "\n</gold_score_json>\n"
                "</example>\n"
            )
            blocks.append(block)

        return "<examples>\n" + "".join(blocks) + "</examples>"

    # ================== [/NEW formatter] =====================================

    def build_prediction_prompt(
        self,
        user_question: str,
        candidate_dag_text: str,
        hits_block: str,
    ) -> str:
        return (
            "You are an impartial evaluator for planning DAGs.\n"
            "Using the few-shot examples and retrieved context, judge ONLY the candidate.\n"
            "Return a JSON with keys: correct (0.0–1.0), error_analysis (newline-separated; each line starts with [OK]/[PARTIAL]/[NOT], ≤15 words).\n"
            "No extra keys, no prose.\n\n"
            + hits_block + "\n\n"
            "<candidate>\n"
            "<user_question>\n" + user_question + "\n</user_question>\n"
            "<dag_json_or_text>\n" + candidate_dag_text + "\n</dag_json_or_text>\n"
            "</candidate>\n\n"
            'Respond with ONLY this JSON: {"correct": float, "error_analysis": "<lines>"}'
        )
    
    def _parse_llm_json_strict(self, text: str) -> Dict[str, Any]:
        """
        LLMの出力から JSON を堅牢に取得。
        - そのまま json.loads
        - 失敗したら {...} を最短・最長一致でサーチ
        """
        text = text.strip()
        # まず素直に
        try:
            return json.loads(text)
        except Exception:
            pass
        # コードフェンス除去
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        try:
            return json.loads(text)
        except Exception:
            pass
        # 最後の手段：{...} を抜き出す
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Failed to parse JSON from LLM output")

    def predict_score(
        self,
        db_url: str,
        user_question: str,
        candidate_dag_text: str,
        *,
        fts_limit: int = 3,
        vector_k: int = 1,
        per_triplets: int = 3,
    ) -> Dict[str, Any]:
        hits = self.search_similar_triplets(
            db_url=db_url,
            user_question=user_question,
            llm_fn=watsonx_llm,
            llm_model_id=16,
            # embed_fn=self.embed_fn,
            fts_limit=fts_limit,
            vector_k=vector_k,
            max_triplets=max(per_triplets, 1),
        )

        examples_block = self._format_triplets_for_prompt(
            hits,
            per_triplets=per_triplets,
            dag_chars=1500,
            traj_chars=1200,
            score_chars=600,
        )

        prompt = self.build_prediction_prompt(
            user_question=user_question,
            candidate_dag_text=candidate_dag_text,
            hits_block=examples_block,
        )

        resp = watsonx_llm(prompt, model_id=16)
        llm_text=resp.get("generated_text", "").strip()
        in_tok = resp.get("input_token_count", 0)
        out_tok = resp.get("generated_token_count", 0)
        print(f"llm_text: {llm_text}")

        try:
            parsed = self._parse_llm_json_strict(llm_text)
            print(f"parsed: {parsed}")

        except Exception:
            parsed = {"correct": 0.0, "error_analysis": ""}

        correct = parsed.get("correct", 0.0)
        try:
            correct = float(correct)
        except Exception:
            correct = 0.0
        correct = max(0.0, min(1.0, correct))

        ea = parsed.get("error_analysis", "")
        if isinstance(ea, list):
            ea = "\n".join(str(x) for x in ea)
        else:
            ea = str(ea or "")

        return {
            "user_question": user_question,
            "correct": correct,
            "error_analysis": ea,
        }

    # ================== [NEW] keyword & vector search helper ==================
    def search_task_description(
        self,
        db_url: str,
        question_text: str,
        *,
        llm_fn,                
        llm_model_id: str,     # e.g. self.llm
        embed_fn=None,         # e.g. self.embedder or wrapper around watsonx_embed
        fts_limit: int = 3,
    ) -> dict:
        """
        From task.description (question_text):
        1) extract keywords via watsonx LLM,
        2) build an embedding vector,
        3) run FTS (top-N) and vector search (top-1),
        4) return a dict with results.

        Targets:
        - FTS: traj_tasks.tsv_task + traj_docs.tsv_all
        - Vector: traj_docs.text_vec

        Ranking (Option B):
        - FTS: order by is_accomplished DESC, rank DESC
        - Vector: order by is_accomplished_doc DESC, distance
            where is_accomplished(_doc) is derived from traj_tasks.status = 'Accomplished'.
        """
        import json
        import psycopg
        from psycopg.rows import dict_row

        # ----------------------- 1) Keyword extraction -----------------------
        def _extract_keywords_with_watsonx(q: str, search_hits: dict | None = None) -> list[str]:
            ctx = ""
            if search_hits:
                kw = ", ".join(search_hits.get("keywords", [])[:6])
                ctx = f"Known related terms: {kw}\n"

            prompt = (
                "You are extracting search keywords for a retrieval system backed by PostgreSQL full-text search "
                "(simple config and jsonb_to_tsvector), pg_trgm fuzzy matching, and vector embeddings.\n"
                "Return ONLY a JSON array of 5–12 strings. No prose, no markdown, no trailing comma.\n\n"
                "Rules:\n"
                "1) Include domain-specific terms and 2–3 word short phrases when helpful (e.g., 'IoT sites').\n"
                "2) Avoid stopwords and very short tokens (<3 chars) unless indispensable (e.g., 'AI').\n"
                "3) Prefer canonical tokens usable by Postgres simple FTS (no quotes, no punctuation inside terms).\n"
                "4) Deduplicate; include useful morphological or naming variants (singular/plural, hyphen/no-hyphen) only if meaningful.\n"
                "5) Use lowercase except for well-known acronyms (e.g., IoT, HVAC).\n\n"
                "Examples (format only):\n"
                'QUESTION: What IoT sites are available?\n'
                'OUTPUT: ["IoT", "IoT sites", "site list", "available sites", "facility sites", "HVAC"]\n'
                'QUESTION: Get sensor history for Chiller 4 at MAIN site\n'
                'OUTPUT: ["sensor history", "Chiller 4", "MAIN site", "asset history", "time series", "observation data"]\n\n'
                f"{ctx}"
                f"QUESTION:\n{q}\n"
            )

            resp = watsonx_llm(prompt, model_id=16)
            kw_text = resp.get("generated_text", "")
            in_tok = resp.get("input_token_count", 0)
            out_tok = resp.get("generated_token_count", 0)

            # Primary path: parse as JSON list[str]
            try:
                kws = json.loads(kw_text)
                if isinstance(kws, list) and all(isinstance(x, str) for x in kws):
                    out: list[str] = []
                    seen: set[str] = set()
                    for k in (s.strip() for s in kws if s and s.strip()):
                        if len(k) < 3 and not (k.isupper() and len(k) in (2, 3)):
                            continue
                        key = k.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(k)
                    return out
            except Exception:
                pass

            # Fallback: comma-split heuristic
            parts = [p.strip() for p in kw_text.replace("\n", " ").split(",")]
            out: list[str] = []
            seen: set[str] = set()
            for k in parts:
                if not k:
                    continue
                if len(k) < 3 and not (k.isupper() and len(k) in (2, 3)):
                    continue
                key = k.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(k)
            return out

        # ----------------------- 2) Embedding helper ------------------------
        if embed_fn is None:
            # try class attributes if not provided explicitly
            if hasattr(self, "embedder") and callable(self.embedder):
                embed_fn = self.embedder
            elif hasattr(self, "watsonx_embed") and callable(self.watsonx_embed):
                embed_model_id = getattr(self, "embed_model_id", None)
                embed_fn = lambda s: self.watsonx_embed(s, model_id=embed_model_id)
            else:
                raise RuntimeError(
                    "No embedding function provided. "
                    "Pass embed_fn or define self.embedder/self.watsonx_embed."
                )

        def _get_embedding(
            text: str,
            *,
            embed_fn=None,
            watsonx_embed_fn=None,
            embed_model_id=None,
        ) -> list[float]:
            """
            Returns a 1536-dim vector (list[float]) for pgvector vector(1536).
            Prefer `embed_fn(text)`. If not provided, try `watsonx_embed_fn(text, model_id=...)`.
            """
            if callable(embed_fn):
                vec = embed_fn(text)
            elif callable(watsonx_embed_fn):
                vec = watsonx_embed_fn(text, model_id=embed_model_id)
            else:
                raise RuntimeError("No embedding function provided. Pass embed_fn or watsonx_embed_fn.")
            if not isinstance(vec, (list, tuple)):
                raise TypeError("Embedding must be list[float] or tuple[float].")
            return list(vec)

        def _vector_literal(vec: list[float]) -> str:
            # pgvector textual literal: [v1, v2, ...]
            return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"

        # ----------------------- 3) Build queries ---------------------------
        keywords = _extract_keywords_with_watsonx(question_text)
        if not keywords:
            keywords = [question_text]

        # websearch tsquery string: OR-join of quoted keywords for recall
        # e.g. "iot" OR "site list" OR "chiller 4"
        tsquery_str = " OR ".join(
            f"\"{k.replace('\"', '')}\"" for k in keywords if k.strip()
        )

        # Embedding
        q_vec = _get_embedding(question_text, embed_fn=embed_fn)
        q_vec_lit = _vector_literal(q_vec)

        results: dict[str, object] = {
            "keywords": keywords,
            "fts_top3": [],
            "vector_top1": None,
        }

        # ----------------------- 4) Run DB queries --------------------------
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # 4-a) FTS across traj_tasks + traj_docs
                #     Boost tasks with status='Accomplished' (Option B).
                cur.execute(
                    """
                    WITH q AS (
                        SELECT websearch_to_tsquery('simple', %s) AS tsq
                    ),
                    t_hits AS (
                        SELECT
                            'traj_tasks' AS src,
                            tt.task_id,
                            tt.doc_id,
                            NULL::uuid    AS run_id,
                            NULL::uuid    AS plan_id,
                            NULL::int     AS scenario_id,
                            NULL::int     AS json_id,
                            tt.task_number,
                            tt.task_description,
                            tt.final_answer,
                            ts_rank(tt.tsv_task, (SELECT tsq FROM q)) AS rank,
                            CASE
                                WHEN tt.status = 'Accomplished' THEN 1
                                ELSE 0
                            END AS is_accomplished
                        FROM traj_tasks tt
                        WHERE tt.tsv_task @@ (SELECT tsq FROM q)
                        ORDER BY rank DESC
                        LIMIT 10
                    ),
                    d_hits AS (
                        SELECT
                            'traj_docs' AS src,
                            NULL::uuid      AS task_id,
                            td.doc_id,
                            NULL::uuid      AS run_id,
                            NULL::uuid      AS plan_id,
                            NULL::int       AS scenario_id,
                            td.json_id,
                            NULL::int       AS task_number,
                            td.text         AS task_description,
                            NULL::text      AS final_answer,
                            ts_rank(td.tsv_all, (SELECT tsq FROM q)) AS rank,
                            0 AS is_accomplished
                        FROM traj_docs td
                        WHERE td.tsv_all @@ (SELECT tsq FROM q)
                        ORDER BY rank DESC
                        LIMIT 10
                    )
                    SELECT * FROM (
                        SELECT * FROM t_hits
                        UNION ALL
                        SELECT * FROM d_hits
                    ) u
                    ORDER BY is_accomplished DESC, rank DESC
                    LIMIT %s;
                    """,
                    (tsquery_str, fts_limit),
                )
                results["fts_top3"] = [dict(r) for r in cur.fetchall()]

                # 4-b) Vector top-1 (traj_docs.text_vec)
                #      Prefer docs that have at least one Accomplished task.
                cur.execute(
                    """
                    WITH params AS (
                        SELECT CAST(%s AS vector) AS qv
                    ),
                    accomplished_docs AS (
                        SELECT DISTINCT tt.doc_id
                        FROM traj_tasks tt
                        WHERE tt.status = 'Accomplished'
                    )
                    SELECT
                        td.doc_id,
                        td.json_id,
                        td.text,
                        (1 - (td.text_vec <=> (SELECT qv FROM params))) AS sim,
                        CASE
                            WHEN ad.doc_id IS NOT NULL THEN 1
                            ELSE 0
                        END AS is_accomplished_doc
                    FROM traj_docs td
                    LEFT JOIN accomplished_docs ad
                        ON ad.doc_id = td.doc_id
                    WHERE td.text_vec IS NOT NULL
                    ORDER BY is_accomplished_doc DESC, td.text_vec <=> (SELECT qv FROM params)
                    LIMIT 1;
                    """,
                    (q_vec_lit,),
                )
                vrow = cur.fetchone()
                if vrow:
                    results["vector_top1"] = dict(vrow)

        return results
    # ================== [/NEW helper] =======================================

    def _fetch_trajectory_previews(
        self,
        db_url: str,
        doc_ids: set[str],
        task_ids: set[str],
        *,
        per_doc_tasks: int = 2,
        per_task_steps: int = 2,
    ) -> dict:
        """
        doc_ids に対しては traj_tasks から task 概要（desc/response/final）を先頭から取得。
        task_ids に対しては traj_log_steps を時系列で数件取得。
        戻り値:
        {
            "doc":  {doc_id: [ {task_number, task_description, response_head, final_head}, ... ]},
            "task": {task_id: [ {step, thought, action, observation}, ... ]}
        }
        """
        out = {"doc": {}, "task": {}}
        if not (doc_ids or task_ids):
            return out

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                if doc_ids:
                    cur.execute(
                        """
                        SELECT
                        t.doc_id,
                        t.task_number,
                        t.task_description,
                        LEFT(t.response, 200)      AS response_head,
                        LEFT(t.final_answer, 200)  AS final_head
                        FROM traj_tasks t
                        WHERE t.doc_id = ANY(%s::uuid[])
                        ORDER BY t.doc_id, t.task_number
                        """,
                        (list(doc_ids),),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        out["doc"].setdefault(str(r["doc_id"]), []).append({
                            "task_number": r["task_number"],
                            "task_description": r["task_description"],
                            "response_head": r["response_head"],
                            "final_head": r["final_head"],
                        })
                    # 上限を適用
                    for k in list(out["doc"].keys()):
                        out["doc"][k] = out["doc"][k][:per_doc_tasks]

                if task_ids:
                    cur.execute(
                        """
                        SELECT
                        l.task_id,
                        s.step,
                        s.thought,
                        s.action,
                        s.observation
                        FROM traj_logs l
                        JOIN traj_log_steps s ON s.log_id = l.log_id
                        WHERE l.task_id = ANY(%s::uuid[])
                        ORDER BY l.task_id, s.step
                        """,
                        (list(task_ids),),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        out["task"].setdefault(str(r["task_id"]), []).append({
                            "step": r["step"],
                            "thought": r["thought"],
                            "action": r["action"],
                            "observation": r["observation"],
                        })
                    # 上限を適用
                    for k in list(out["task"].keys()):
                        out["task"][k] = out["task"][k][:per_task_steps]

        return out


    # ================== [NEW] formatter: inject search hits into the prompt ==================
    def _format_search_hits_for_prompt(
        self,
        search_hits: dict,
        max_chars: int = 1800,
        *,
        db_url: str | None = None,
        per_doc_tasks: int = 2,
        per_task_steps: int = 2,
    ) -> str:
        """
        検索結果の要約に trajectory の短縮プレビューも追加する。
        - keywords
        - FTS (top-3) summary
        - Vector (top-1) summary
        - trajectory preview（doc基準とtask基準、それぞれ数件）
        """
        def _clip(s: str | None, n: int = 240) -> str:
            if not s:
                return ""
            s = " ".join(str(s).split())
            return s if len(s) <= n else (s[: n - 1] + "…")

        if not isinstance(search_hits, dict):
            return ""

        parts: list[str] = []

        # 1) keywords
        kws = search_hits.get("keywords") or []
        if kws:
            parts.append("Keywords: " + ", ".join(_clip(k, 60) for k in kws[:12]))

        # 2) FTS hits (top-3)
        fts = search_hits.get("fts_top3") or []
        if fts:
            lines = []
            for r in fts[:3]:
                src  = r.get("src") or "unknown"
                rank = r.get("rank")
                if src == "traj_tasks":
                    desc = _clip(r.get("task_description") or "")
                    ans  = _clip(r.get("final_answer") or "")
                    lines.append(f"- [{src}] rank={rank:.4f} desc={desc} | final={ans}")
                else:
                    txt = _clip(r.get("task_description") or r.get("text") or "")
                    lines.append(f"- [{src}] rank={rank:.4f} text={txt}")
            parts.append("FTS hits (top-3):\n" + "\n".join(lines))

        # 3) Vector hit (top-1)
        v1 = search_hits.get("vector_top1")
        if v1:
            sim = v1.get("sim")
            txt = _clip(v1.get("text") or "")
            parts.append(f"Vector hit (top-1): sim={sim:.4f} text={txt}")

        # 4) trajectory preview（DB から doc/task の抜粋を取得）
        doc_ids: set[str] = set()
        task_ids: set[str] = set()

        # FTS 結果に doc_id / task_id が入っている想定（検索関数側で返す）
        for r in fts:
            if r.get("doc_id"):
                doc_ids.add(str(r["doc_id"]))
            if r.get("task_id"):
                task_ids.add(str(r["task_id"]))

        # Vector top-1 が traj_docs なら doc_id を追加
        if v1 and v1.get("doc_id"):
            doc_ids.add(str(v1["doc_id"]))

        if db_url and (doc_ids or task_ids):
            previews = self._fetch_trajectory_previews(
                db_url, doc_ids, task_ids,
                per_doc_tasks=per_doc_tasks,
                per_task_steps=per_task_steps,
            )
            # doc 単位（タスク見出し + response/final の頭出し）
            if previews.get("doc"):
                lines = []
                for did, rows in list(previews["doc"].items())[:2]:
                    for row in rows:
                        tn = row.get("task_number")
                        td = _clip(row.get("task_description"), 90)
                        rs = _clip(row.get("response_head"), 100)
                        fn = _clip(row.get("final_head"), 100)
                        lines.append(f"- [doc {did[:8]}] S{tn}: {td}"
                                    + (f" | resp={rs}" if rs else "")
                                    + (f" | final={fn}" if fn else ""))
                if lines:
                    parts.append("Trajectory preview (by doc):\n" + "\n".join(lines))

            # task 単位（step の thought/action/observation 抜粋）
            if previews.get("task"):
                lines = []
                for tid, steps in list(previews["task"].items())[:2]:
                    for s in steps:
                        th = _clip(s.get("thought"), 80)
                        ac = _clip(s.get("action"), 40)
                        ob = _clip(s.get("observation"), 80)
                        lines.append(f"- [task {tid[:8]}] step {s.get('step')}: {th} | {ac} | {ob}")
                if lines:
                    parts.append("Trajectory preview (by task):\n" + "\n".join(lines))

        out = "Retrieved context:\n" + "\n".join(parts)
        print(f"parts: {parts}")
        return out[:max_chars]

    # ================== [/NEW formatter] =====================================================

    # ================== [NEW] helper: save user question ==================
    def _save_user_question(
        self,
        db_url: str,
        *,
        text: str,
        plan_id: str | None,
        scenario_id: int | None = None,
        trajectory_doc_id: str | None = None,
        trajectory_path: str | None = None,
    ) -> tuple[str, str]:
        if plan_id is None:
            plan_id = str(uuid.uuid4())
        if scenario_id is None:
            scenario_id = 0
        if not trajectory_path:
            trajectory_path = "unknown.json"

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_runs
                    (scenario_id, plan_id, text, trajectory_path, trajectory_doc_id)
                    VALUES
                    (%s, %s, %s, %s, %s)
                    ON CONFLICT (scenario_id, plan_id) DO UPDATE
                    SET text = EXCLUDED.text,
                        trajectory_path =
                            COALESCE(query_runs.trajectory_path, EXCLUDED.trajectory_path),
                        trajectory_doc_id =
                            COALESCE(query_runs.trajectory_doc_id, EXCLUDED.trajectory_doc_id)
                    RETURNING run_id, plan_id
                    """,
                    (scenario_id, plan_id, text, trajectory_path, trajectory_doc_id),
                )
                row = cur.fetchone()
                return row["run_id"], row["plan_id"]
    # ================== [/NEW helper] =====================================

    def _save_plan_round(self, db_url: str, *, plan_id, scenario_id: int, llm: str,
                    round_t: int, prompt: str, answer: str, dag: dict) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        sql = """
        INSERT INTO plan_dag_rounds
        (plan_id, scenario_id, llm, round_t, prompt, answer, dag)
        VALUES (%(plan_id)s, %(scenario_id)s, %(llm)s, %(round_t)s, %(prompt)s, %(answer)s, %(dag)s)
        ON CONFLICT (plan_id, round_t) DO UPDATE
        SET prompt = EXCLUDED.prompt,
            answer = EXCLUDED.answer,
            dag    = EXCLUDED.dag
        """
        params = {
            "plan_id": plan_id,
            "scenario_id": scenario_id,
            "llm": llm,
            "round_t": round_t,
            "prompt": prompt,
            "answer": answer,
            "dag": Jsonb(dag),  # ← psycopg3 の JSONB アダプタで安全に送る
        }
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

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
    
    def _truncate_plan_text(self, plan_text: str, stop_index: int) -> str:
        """
        Truncate a plan to steps 1..stop_index (inclusive) and rebuild it in the canonical
        4-lines-per-step format used by your validator:

          #TaskN: ...
          #AgentN: ...
          #DependencyN: None | "#S1 #S2 ..."
          #ExpectedOutputN: ...

        Notes:
        - Always truncates (does NOT depend on can_answer_now).
        - Filters dependencies to only include past steps (<N) within the truncated prefix.
        - If parsing is inconsistent, falls back to returning the original plan_text.
        """
        if not isinstance(stop_index, int) or stop_index < 1:
            return plan_text

        TASK_RE   = re.compile(r"^#Task(\d+): (.+)$", re.M)
        AGENT_RE  = re.compile(r"^#Agent(\d+): (.+)$", re.M)
        DEP_RE    = re.compile(r"^#Dependency(\d+): (.+)$", re.M)
        OUT_RE    = re.compile(r"^#ExpectedOutput(\d+): (.+)$", re.M)
        DEP_TOKEN = re.compile(r"#S(\d+)")

        # Keep any header/prefix text before the first tag line (optional but safe).
        FIRST_TAG = re.compile(r"(?m)^#(?:Task|Agent|Dependency|ExpectedOutput)\d+:\s")
        m0 = FIRST_TAG.search(plan_text)
        prefix = plan_text[: m0.start()] if m0 else ""

        tasks: Dict[int, str]  = {int(n): s.strip() for n, s in TASK_RE.findall(plan_text)}
        agents: Dict[int, str] = {int(n): s.strip() for n, s in AGENT_RE.findall(plan_text)}
        deps: Dict[int, str]   = {int(n): s.strip() for n, s in DEP_RE.findall(plan_text)}
        outs: Dict[int, str]   = {int(n): s.strip() for n, s in OUT_RE.findall(plan_text)}

        if not tasks or not agents or not deps or not outs:
            return plan_text

        # Only truncate as far as we have complete step data for all 4 fields.
        total_complete = min(len(tasks), len(agents), len(deps), len(outs))
        k = min(stop_index, total_complete)
        if k < 1:
            return plan_text

        # Require a contiguous prefix 1..k; otherwise, safest fallback is no-op.
        for i in range(1, k + 1):
            if i not in tasks or i not in agents or i not in deps or i not in outs:
                return plan_text

        rebuilt: List[str] = []
        for i in range(1, k + 1):
            rebuilt.append(f"#Task{i}: {tasks[i]}")
            rebuilt.append(f"#Agent{i}: {agents[i]}")

            dep_raw = deps[i].strip()
            if dep_raw == "None":
                dep_fixed = "None"
            else:
                nums = [int(x) for x in DEP_TOKEN.findall(dep_raw)]
                # Keep only valid, past-step dependencies within the truncated prefix.
                nums = [x for x in nums if 1 <= x <= k and x < i]
                dep_fixed = "None" if not nums else " ".join(f"#S{x}" for x in nums)

            rebuilt.append(f"#Dependency{i}: {dep_fixed}")
            rebuilt.append(f"#ExpectedOutput{i}: {outs[i]}")
            rebuilt.append("")  # blank line between steps (optional)

        return prefix + "\n".join(rebuilt).rstrip() + "\n"

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


        # ---- SPIRAL-style evaluation summary (status, rationale, etc.) ----
        if spiral_feedback:
            status = spiral_feedback.get("status", "Unknown")
            rationale = spiral_feedback.get("rationale", "")
            can_answer_now = spiral_feedback.get("can_answer_now", None)
            stop_index = spiral_feedback.get("stop_index", None)

            lines_sf: list[str] = []
            lines_sf.append("SPIRAL-style evaluation of the current plan:")
            lines_sf.append(f"- Status: {status}")
            if can_answer_now is not None:
                lines_sf.append(f"- can_answer_now: {can_answer_now}")
            if stop_index is not None:
                lines_sf.append(
                    f"- stop_index: {stop_index} "
                    "(earliest step index after which SPIRAL believes the plan can already answer)"
                )
            lines_sf.append(f"- Critic rationale: {rationale}")

            # NEW: include truncated plan text (if provided)
            if truncated_plan_text:
                lines_sf.append("")
                lines_sf.append("Truncated plan (use this as the current plan context for repair):")
                lines_sf.append(truncated_plan_text.rstrip())

            spiral_text = "\n".join(lines_sf) + "\n"
        else:
            lines_sf: list[str] = []
            lines_sf.append(
                "SPIRAL-style evaluation of the current plan is not available for this round.\n"
                "Assume the current plan may still be suboptimal and try to improve it based "
                "on the issues and the planning instructions."
            )

            # NEW: still include truncated plan text if you have it
            if truncated_plan_text:
                lines_sf.append("")
                lines_sf.append("Truncated plan (use this as the current plan context for repair):")
                lines_sf.append(truncated_plan_text.rstrip())

            spiral_text = "\n".join(lines_sf) + "\n"

            
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
            "- Interpret the SPIRAL feedback as follows in your planning:\n"
            "  * status: how complete and correct the current answer is.\n"
            "  * can_answer_now=True: you may safely stop planning and keep the plan minimal.\n"
            "  * stop_index: earliest step index after which the plan already supports answering.\n"
            "- Use the SPIRAL evaluation feedback and the issues above to decide how to fix the plan.\n"
            "- Construct the DAG using the bare minimum number of tasks required to satisfy the user question and constraints. "
            "Avoid redundant or unnecessary tasks.\n"
            "- Make the minimal changes necessary; if there is no problem, you MUST output the Original Plan as-is.\n"
            # "- Output ONLY lines in this exact format (no extra prose):\n"
            # "  #TaskN: <one-line>\n"
            # "  #AgentN: <exact agent name>\n"
            # "  #DependencyN: None | #S1 #S2 ... (past steps only)\n"
            # "  #ExpectedOutputN: <one-line>\n"
            # f"- Agents allowed: {', '.join(agents_allowed)}\n"
            # "- Use N = 1..K sequentially; counts across all tags must match.\n"
            "- Do NOT output any explanation, comments, or markdown.\n"
        )

        return (
            f"{base_wo_marker}\n\n"
            "=== SPIRAL Evaluation Feedback ===\n"
            f"{spiral_text}\n"
            "=== Detected Issues ===\n"
            f"{issues_text}\n"
            "=== Original Plan ===\n"
            f"{original_plan}\n\n"
            f"{rules}\n\n"
            f"{OUTPUT_MARKER}"
        )



    def _extract_planned_tasks_and_dag(self, final_plan: str, task) -> tuple[list, dict]:
        """
        Parse `final_plan` text and:
        - build `planned_tasks` (existing Task objects)  ← 既存ロジックのまま
        - build normalized DAG JSON: {"nodes":[...], "edges":[...]}
        """
        import re

        # === 既存の正規表現（そのまま） ===
        task_pattern        = r"#Task\d+: (.+)"
        agent_pattern       = r"#Agent\d+: (.+)"
        dependency_pattern  = r"#Dependency\d+: (.+)"
        output_pattern      = r"#ExpectedOutput\d+: (.+)"

        tasks         = re.findall(task_pattern,       final_plan)
        agents        = re.findall(agent_pattern,      final_plan)
        dependencies  = re.findall(dependency_pattern, final_plan)
        outputs       = re.findall(output_pattern,     final_plan)

        # === 既存の planned_tasks 構築（そのまま） ===
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

            selected_agent = None
            for agent in task.agents:
                if agent.name == agent_name:
                    selected_agent = agent
                    break
            if selected_agent is None:
                selected_agent = task.agents[0]

            if dependency != "None":
                numbers = re.findall(r"#S(\d+)", dependency)
                numbers = list(map(int, numbers))
                # 既存コードのまま（外側 i を上書きする内側の i を使用）
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

        # === DAG JSON を組み立て（CHECK: dag ? 'nodes' AND dag ? 'edges' を満たす） ===
        nodes = []
        edges = []
        for i in range(len(tasks)):
            nodes.append({
                "id": f"S{i+1}",
                "task": tasks[i],
                "agent": agents[i] if i < len(agents) else None,
                "expected_output": outputs[i] if i < len(outputs) else ""
            })
            if i < len(dependencies) and dependencies[i] != "None":
                nums = re.findall(r"#S(\d+)", dependencies[i])
                for n in nums:
                    try:
                        nn = int(n)
                        edges.append({"from": f"S{nn}", "to": f"S{i+1}"})
                    except ValueError:
                        pass

        dag_json = {"nodes": nodes, "edges": edges}
        return planned_tasks, dag_json
    
    def search_plan_rounds_prompt_answer(db_url: str, q: str, limit: int = 10) -> list[dict]:
        """
        plan_dag_rounds の (prompt + answer) を FTS で検索。
        websearch_to_tsquery を使うので、AND/OR/ダブルクオートのフレーズにも対応。
        例: 'iot sites'  / '"iot sites"' / 'iot -legacy OR hvac'
        """
        sql = """
        SELECT
        plan_id, round_t, llm,
        ts_rank(
            to_tsvector('simple', coalesce(prompt,'') || ' ' || coalesce(answer,'')),
            websearch_to_tsquery('simple', %(q)s)
        ) AS rank,
        left(prompt, 200) AS prompt_head,
        left(answer, 200) AS answer_head,
        created_at
        FROM plan_dag_rounds
        WHERE to_tsvector('simple', coalesce(prompt,'') || ' ' || coalesce(answer,''))
            @@ websearch_to_tsquery('simple', %(q)s)
        ORDER BY rank DESC, created_at DESC
        LIMIT %(limit)s;
        """
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"q": q, "limit": limit})
                return cur.fetchall()

    def search_plan_rounds_dag(db_url: str, q: str, limit: int = 10) -> list[dict]:
        """
        plan_dag_rounds.dag を jsonb_to_tsvector で FTS 検索。
        JSON 内の文字列/数値/真偽にヒットします（'simple' 辞書）。
        """
        sql = """
        SELECT
        plan_id, round_t, llm,
        ts_rank(
            jsonb_to_tsvector('simple', dag, '["string","numeric","boolean"]'),
            websearch_to_tsquery('simple', %(q)s)
        ) AS rank,
        jsonb_array_length(dag->'nodes') AS node_cnt,
        jsonb_array_length(dag->'edges') AS edge_cnt,
        created_at
        FROM plan_dag_rounds
        WHERE jsonb_to_tsvector('simple', dag, '["string","numeric","boolean"]')
            @@ websearch_to_tsquery('simple', %(q)s)
        ORDER BY rank DESC, created_at DESC
        LIMIT %(limit)s;
        """
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"q": q, "limit": limit})
                return cur.fetchall()

    def search_traj_docs_text_to_latest_round(db_url: str, q: str, limit_docs: int = 5) -> list[dict]:
        """
        1) traj_docs.tsv_all に対して FTS
        2) ヒットした doc に紐づく query_runs を JOIN
        3) その plan_id の最新 round_t を LATERAL で 1件取得
        """
        sql = """
        WITH hits AS (
        SELECT
            td.doc_id,
            td.text,
            ts_rank(td.tsv_all, websearch_to_tsquery('simple', %(q)s)) AS rank
        FROM traj_docs td
        WHERE td.tsv_all @@ websearch_to_tsquery('simple', %(q)s)
        ORDER BY rank DESC
        LIMIT %(limit_docs)s
        )
        SELECT
        h.doc_id, h.text AS doc_text, h.rank AS doc_rank,
        qr.plan_id, qr.run_id, qr.created_at AS run_created_at,
        pr.round_t, pr.llm,
        pr.prompt, pr.answer, pr.dag, pr.created_at AS round_created_at
        FROM hits h
        JOIN query_runs qr
        ON qr.trajectory_doc_id = h.doc_id
        JOIN LATERAL (
        SELECT plan_id, round_t, llm, prompt, answer, dag, created_at
        FROM plan_dag_rounds
        WHERE plan_id = qr.plan_id
        ORDER BY round_t DESC
        LIMIT 1
        ) pr ON TRUE
        ORDER BY h.rank DESC, pr.round_t DESC;
        """
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"q": q, "limit_docs": limit_docs})
                return cur.fetchall()
            
    def _spiral_evaluate_plan(
        self,
        db_url: str,
        user_question: str,
        final_plan: str,
        task,
        max_prefix_steps: int | None = None,
    ) -> dict:
        """
        Run a SPIRAL-style evaluation on the current 'final_plan':
          1) parse it into planned_tasks,
          2) iterate over prefixes (Step 1, Steps 1–2, ...),
          3) at each prefix:
             - use SimulatorAgent to predict the output for the current step,
             - use CriticAgent to judge the candidate answer,
             - stop early if Critic says Accomplished or Not accomplished.

        Returns a dict like:
        {
            "stop_index": int,     # 1-based index of the prefix where we stopped
            "status": "Accomplished" | "Partially accomplished" | "Not accomplished",
            "can_answer_now": bool,
            "rationale": str,
            "predicted_answer": str,
        }
        """
        input_tokens_count=0
        generated_tokens_count=0
        # 1) Parse plan into tasks (re-use your existing logic)
        planned_tasks, dag_json = self._extract_planned_tasks_and_dag(final_plan, task)
        if not planned_tasks:
            # nothing to evaluate
            return {
                "stop_index": 0,
                "status": "Not accomplished",
                "can_answer_now": False,
                "rationale": "No tasks parsed from plan.",
                "predicted_answer": "",
            }

        # Optional: limit how many steps we consider
        if max_prefix_steps is None:
            max_prefix_steps = len(planned_tasks)

        # 2) Instantiate Simulator + Critic (simple: fresh per call)
        sim = SimulatorAgent(
            db_url=db_url,
            system_prompt=SIMULATOR_SYSTEM_PROMPT,
            max_similar_tasks=5,
        )
        critic = CriticAgent()  # uses few-shot rubric internally

        dag_prefix_for_critic: list[str] = []
        last_result: dict | None = None

        for idx, pl_task in enumerate(planned_tasks, start=1):
            if idx > max_prefix_steps:
                break

            # Build a human-readable prefix line for Critic
            agent_name = pl_task.agents[0].name if pl_task.agents else "<unknown>"
            prefix_line = (
                f"Step{idx}: Agent={agent_name}, "
                f"Task={pl_task.description}, "
                f"ExpectedOutput={pl_task.expected_output}"
            )
            dag_prefix_for_critic.append(prefix_line)

            # 3) Simulator: predict output for this step
            predicted_output, input_tokens, generated_tokens = sim.run(
                user_question=user_question,
                task_description=pl_task.description,
                agent_name=agent_name,
                dag_prefix=dag_prefix_for_critic,
            )
            input_tokens_count+=input_tokens
            generated_tokens_count+=generated_tokens
            print(f"[Simulator] in_tok: {input_tokens}, out_tok: {generated_tokens}")


            # 4) Critic: evaluate whether this prefix + answer is enough
            critic_res, input_tokens, generated_tokens  = critic.evaluate(
                user_question=user_question,
                candidate_answer=predicted_output,
                dag_prefix=dag_prefix_for_critic,
            )
            input_tokens_count+=input_tokens
            generated_tokens_count+=generated_tokens
            print(f"[Critic] in_tok: {input_tokens}, out_tok: {generated_tokens}")

            last_result = {
                "stop_index": idx,
                "status": critic_res["status"],
                "can_answer_now": critic_res["can_answer_now"],
                "rationale": critic_res["rationale"],
                "predicted_answer": predicted_output,
            }

            # SPIRAL-style stopping rule:
            #   - If accomplished → good plan.
            #   - If not accomplished → plan is structurally bad; no need to go deeper.
            if (critic_res["status"] == "Accomplished") or (
                critic_res["status"] == "Not accomplished"
            ):
                break

        if last_result is None:
            # Should not normally happen, but be safe.
            return {
                "stop_index": 0,
                "status": "Not accomplished",
                "can_answer_now": False,
                "rationale": "SPIRAL evaluation did not run any step.",
                "predicted_answer": "",
            }

        return last_result, input_tokens_count, generated_tokens_count


    def generate_steps(self, save_plan=False, saved_plan_filename="", qid=None):
        task = self.tasks[0]
        agent_descriptions = ""
        input_tokens_count=0
        generated_tokens_count=0

        # --- ensure embedder before any search / similarity ---
        if not hasattr(self, "watsonx_embed"):
            self._init_watsonx_embedder_from_env()
        # ------------------------------------------------------


        # ===== Editable section (collect agent info) =====
        for ii, aagent in enumerate(task.agents):
            agent_descriptions += f"\n({ii + 1}) Agent name: {aagent.name}"
            agent_descriptions += f"\nAgent description: {aagent.description}"
            if "task_examples" in aagent.__dict__ and aagent.task_examples:
                agent_descriptions += f"\nTasks that agent can solve:"
                for idx, task_example in enumerate(aagent.task_examples, start=1):
                    agent_descriptions += f"\n{idx}. {task_example}"
            agent_descriptions += "\n"
        # ===== End editable section =====

        # ------------------ [NEW] call: save the user question -----------------
        # Decide DB URL, plan_id, scenario_id. If your class has attributes for these,
        # they will be reused; otherwise they are defaulted (plan_id auto, scenario_id=0).
        DB_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@host:port/db
        current_plan_id = getattr(self, "plan_id", None)        # may be None first time
        scenario_id = getattr(self, "scenario_id", None)        # may be None -> will be 0
        trajectory_doc_id = getattr(self, "trajectory_doc_id", None)
        trajectory_path = getattr(self, "trajectory_path", None)

        run_id, persisted_plan_id = self._save_user_question(
            DB_URL,
            text=task.description,              # <-- User question
            plan_id=current_plan_id,
            scenario_id=scenario_id,
            trajectory_doc_id=trajectory_doc_id,
            trajectory_path=trajectory_path,
        )
        # keep them on self for later rounds / saves
        self.plan_id = persisted_plan_id
        self.run_id = run_id
        # ------------------ [/NEW call] ---------------------------------------

        # ------------------ [NEW] call search BEFORE asking LLM ------------------
        search_hits = self.search_task_description(
            DB_URL,
            task.description,
            llm_fn=watsonx_llm,
            llm_model_id=self.llm,
            # どちらかを渡す（優先: self.embedder → watsonx_embed）
            embed_fn = (self.embedder
                        if hasattr(self, "embedder") and callable(self.embedder)
                        else (lambda s: self.watsonx_embed(
                                s,
                                model_id=getattr(self, "embed_model_id", None)
                            ))),
            fts_limit=3,
        )


        logger.info(
            "Search hits based on task.description:\n%s",
            json.dumps(search_hits, ensure_ascii=False, indent=2, default=str),
        )

        self.search_hits = search_hits
        # ------------------ [/NEW call] -----------------------------------------

        # 0) Ask LLM for a plan  (augment prompt with retrieved context)
        base_prompt = self.get_prompt(task.description, agent_descriptions)

        # retrieved_context = self._format_search_hits_for_prompt(
        #     search_hits,
        #     max_chars=1800,
        #     db_url=DB_URL,               # これを渡すと trajectory のプレビューも引いてくる
        #     per_doc_tasks=2,
        #     per_task_steps=2,
        # )

        # print(f"retrieved_context: {retrieved_context}")
        # if retrieved_context:
        #     prompt = (
        #         f"{base_prompt}\n\n"
        #         "### Retrieved Context (from prior runs and trajectory DB)\n"
        #         f"{retrieved_context}\n\n"
        #         "### Instruction\n"
        #         "- Use the context above to align your task breakdown and agent selection.\n"
        #         "- If similar tasks appear, re-use their structure/expected outputs when sensible.\n"
        #         "- Do NOT copy irrelevant content; keep the required output format strictly.\n"
        #     )
        # else:
        #     prompt = base_prompt
        prompt = base_prompt

        logger.info(f"Plan Generation Prompt (augmented): \n{prompt}")
        resp = watsonx_llm(prompt, model_id=self.llm)
        llm_response=resp.get("generated_text", "")
        in_tok = resp.get("input_token_count", 0)
        out_tok = resp.get("generated_token_count", 0)
        input_tokens_count+=in_tok
        generated_tokens_count+=out_tok
        logger.info(f"=============End of Prompt===============\n\n\nPlan (raw): \n{llm_response}")
        print(f"[Prompt 0] in_tok: {in_tok}, out_tok: {out_tok}")


        final_plan = llm_response
        print(f"DAG round_0: {final_plan}")
        saved_plan_filename_0 = saved_plan_filename + "_plan_0.txt"
        saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
        with open(saved_plan_filename_0, "w") as f:
            f.write(saved_plan_text)
        self.memory = []

        planned_tasks, dag_json = self._extract_planned_tasks_and_dag(final_plan, task)
        self._save_plan_round(
            DB_URL,
            plan_id=persisted_plan_id,                 # ← 既存の _save_user_question() の戻りを使用
            scenario_id=getattr(self, "scenario_id", 0),
            llm=self.llm,
            round_t=0,
            prompt=prompt,
            answer=final_plan,
            dag=dag_json
        )

        # 0.5) Pre-parse validation + reflexive repair (up to 3 rounds)
        agents_allowed = [a.name for a in task.agents]
        print(f"agents_allowed: {agents_allowed}")

        T = 3
        spiral_res: Optional[dict] = None

        for t in range(T):
            ok, errs = self._validate_plan_text(final_plan, agents_allowed)
            print(f"ok: {ok}, errs: {errs}")
            saved_valid_filename_t = saved_plan_filename + f"_valid_{t}.txt"
            saved_valid_text = f"ok: {ok}\nerrs:\n{errs}"
            with open(saved_valid_filename_t, "w") as f:
                f.write(saved_valid_text)

            spiral_res = None

            # 1) If the format is valid, run SPIRAL (Simulator + Critic) to evaluate the DAG
            if ok:
                spiral_res, in_tok, out_tok = self._spiral_evaluate_plan(
                    db_url=DB_URL,
                    user_question=task.description,
                    final_plan=final_plan,
                    task=task,
                    max_prefix_steps=None,  # or e.g. min(len(tasks), 5)
                )
                print(f"[SPIRAL] result: {spiral_res}")
                input_tokens_count+=in_tok
                generated_tokens_count+=out_tok
                print(f"[SPIRAL {t}] in_tok: {in_tok}, out_tok: {out_tok}")

                stop_index = spiral_res.get("stop_index", None)
                truncated_plan = self._truncate_plan_text(final_plan, stop_index)

                status = spiral_res.get("status", "Not accomplished")
                can_answer_now = bool(spiral_res.get("can_answer_now", False))

                saved_spiral_filename_t = saved_plan_filename + f"_spiral_{t}.txt"
                saved_spiral_text = f"stop_index: {stop_index}\n\ntruncated_plan: {truncated_plan}\nstatus: {status}\ncan_answer_now:{can_answer_now}"
                with open(saved_spiral_filename_t, "w") as f:
                    f.write(saved_spiral_text)
                
                # 2) Acceptance rule:
                #    - format OK
                #    - Critic says Accomplished
                #    - can_answer_now == True
                if (status == "Accomplished" or status=="Partially accomplished") and can_answer_now:
                    final_plan=truncated_plan
                    print("[SPIRAL] Plan accepted.")
                    break

            # 3) If we reach here, we need to repair:
            #    - either format is broken (ok == False)
            #    - or SPIRAL says it's not good enough yet
            spiral_hint = spiral_res if (ok and spiral_res is not None) else None
            truncated_plan_text = truncated_plan if (ok and truncated_plan is not None) else None

            repair_prompt = self._build_repair_prompt(
                base_prompt=prompt,
                original_plan=final_plan,
                errors=errs,
                agents_allowed=agents_allowed,
                spiral_feedback=spiral_hint,
                truncated_plan_text=truncated_plan_text
            )
            print(f"\n\n\n!!!!!!!!!!!repair_prompt!!!!!!!!!!!!!!!!!: {repair_prompt}\n!!!!!!!!!!!End of Prompt!!!!!!!!!!!!!!\n\n\n")

            resp = watsonx_llm(repair_prompt, model_id=self.llm)
            final_plan=resp.get("generated_text", "")
            in_tok = resp.get("input_token_count", 0)
            out_tok = resp.get("generated_token_count", 0)
            input_tokens_count+=in_tok
            generated_tokens_count+=out_tok
            print(f"DAG round_{t}: {final_plan}")
            print(f"[Prompt {t}] in_tok: {in_tok}, out_tok: {out_tok}")
            logger.info("Plan was repaired based on issues:\n- " + "\n- ".join(errs or ["(no structural issues)"]))

            # Re-parse and log this repaired plan round
            planned_tasks, dag_json = self._extract_planned_tasks_and_dag(final_plan, task)
            self._save_plan_round(
                DB_URL,
                plan_id=persisted_plan_id,
                scenario_id=getattr(self, "scenario_id", 0),
                llm=self.llm,
                round_t=t + 1,
                prompt=repair_prompt,
                answer=final_plan,
                dag=dag_json,
            )

        if save_plan:
            RESULT_DIR = "/home/track1_result/"
            PLAN_DIR = RESULT_DIR + "plan/"
            plan_subdir = os.path.join(PLAN_DIR, f"[SPIRAL]Model_{self.llm}")
            saved_plan_prefix = os.path.join(plan_subdir, f"Model_{self.llm}_Q_{qid}_plan")
            saved_plan_filename_final = saved_plan_prefix + ".txt"


            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename_final, "w") as f:
                f.write(saved_plan_text)

        # 1) Parse validated text into fields
        task_pattern = r"#Task\d+: (.+)"
        agent_pattern = r"#Agent\d+: (.+)"
        dependency_pattern = r"#Dependency\d+: (.+)"
        output_pattern = r"#ExpectedOutput\d+: (.+)"

        tasks = re.findall(task_pattern, final_plan)
        agents = re.findall(agent_pattern, final_plan)
        dependencies = re.findall(dependency_pattern, final_plan)
        outputs = re.findall(output_pattern, final_plan)

        # 3) Build planned_tasks
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
        return planned_tasks, self.run_id, self.plan_id, input_tokens_count, generated_tokens_count


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