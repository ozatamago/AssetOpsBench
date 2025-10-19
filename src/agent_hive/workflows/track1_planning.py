from agent_hive.task import Task
from pydantic import Field
from typing import Dict, Any, List
from agent_hive.enum import ContextType
import json
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.logger import get_custom_logger
import os
import uuid
import psycopg  # psycopg3
from psycopg.rows import dict_row

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
        
    def _init_watsonx_embedder_from_env(self) -> None:
        """
        Create self.watsonx_embed(text, model_id=None) -> list[float (len=DB_EMBEDDING_DIM)].
        Uses IBM watsonx.ai Embeddings API and pads/truncates to DB_EMBEDDING_DIM.
        """
        import os
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings
        from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

        api_key = os.getenv("WATSONX_APIKEY")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_name = os.getenv("WATSONX_EMBEDDING_MODEL", "IBM_SLATE_30M_ENG")
        if not (api_key and project_id):
            raise RuntimeError("WATSONX_API_KEY / WATSONX_PROJECT_ID が未設定です。")

        # EmbeddingTypes から属性で引く（例: IBM_SLATE_30M_ENG）
        try:
            model_id = getattr(EmbeddingTypes, model_name)
        except AttributeError:
            raise RuntimeError(f"未知の埋め込みモデル: {model_name}（SDKの EmbeddingTypes を確認してください）")

        creds = Credentials(api_key=api_key, url=url)
        emb = Embeddings(model_id=model_id, credentials=creds, project_id=project_id)

        db_dim = int(os.getenv("DB_EMBEDDING_DIM", "1536"))

        def _coerce_to_db_dim(vec: list[float]) -> list[float]:
            # Granite/Slate は 384/768 が多い。DB列の vector(1536) に合わせる。
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
        self.embed_model_id = model_name

        # quick probe
        _ = self.watsonx_embed("ping")


    def run(self, enable_summarization=False):
        generated_steps, run_id, plan_id = self.generate_steps()

        sequential_workflow = SequentialWorkflow(
            tasks=generated_steps, context_type=ContextType.SELECTED
        )

        return sequential_workflow.run(), run_id, plan_id

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

    # ================== [NEW] keyword & vector search helper ==================
    def search_task_description(
        self,
        db_url: str,
        question_text: str,
        *,
        llm_fn,                # 例: watsonx_llm
        llm_model_id: str,     # 例: self.llm
        embed_fn=None,              # 例: self.embedder もしくはラッパー(lambdaで watsonx_embed を包む)
        fts_limit: int = 3
    ) -> dict:
        """
        From task.description:
        1) extract keywords via watsonx LLM,
        2) build embedding vector,
        3) run FTS (top-3) and vector search (top-1),
        4) return a dict with results.
        Targets:
        - FTS: traj_tasks.tsv_task + traj_docs.tsv_all
        - Vector: traj_docs.text_vec
        """
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

            kw_text = watsonx_llm(prompt, model_id=llm_model_id)['generated_text']

            try:
                kws = json.loads(kw_text)
                if isinstance(kws, list) and all(isinstance(x, str) for x in kws):
                    out = []
                    seen = set()
                    for k in (s.strip() for s in kws if s and s.strip()):
                        if len(k) < 3 and not (k.isupper() and len(k) in (2,3)):
                            continue
                        key = k.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(k)
                    return out
            except Exception:
                pass
            parts = [p.strip() for p in kw_text.replace("\n", " ").split(",")]
            out, seen = [], set()
            for k in parts:
                print(f"k: {k}")
                if not k:
                    continue
                if len(k) < 3 and not (k.isupper() and len(k) in (2,3)):
                    continue
                key = k.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(k)
            return out

        if embed_fn is None:
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

        # ---- 2) embedding (project-specific; implement one of the hooks) ----
        def _get_embedding(
            text: str,
            *,
            embed_fn=None,               # ex) self.embedder
            watsonx_embed_fn=None,       # ex) self.watsonx_embed
            embed_model_id=None,         # ex) self.embed_model_id
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

        keywords = _extract_keywords_with_watsonx(question_text)
        if not keywords:
            keywords = [question_text]

        # Build a websearch tsquery string: OR-join of quoted keywords for recall
        # (e.g., "iot" OR "site" OR "chiller 4")
        tsquery_str = " OR ".join(
            f"\"{k.replace('\"','')}\"" for k in keywords if k.strip()
        )

        # Embedding
        q_vec = _get_embedding(question_text, embed_fn=embed_fn)
        q_vec_lit = _vector_literal(q_vec)

        # ---- 3) Run searches ----
        results = {"keywords": keywords, "fts_top3": [], "vector_top1": None}

        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # 3-a) FTS across traj_tasks + traj_docs, rank and take top-3
                cur.execute("""
                    WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq),
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
                        ts_rank(tt.tsv_task, (SELECT tsq FROM q)) AS rank
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
                    LIMIT 3;
                """, (tsquery_str,))
                results["fts_top3"] = [dict(r) for r in cur.fetchall()]

                # 3-b) Vector top-1 (traj_docs.text_vec)
                # use a CTE to CAST the vector literal safely
                cur.execute(f"""
                    WITH params AS (SELECT CAST(%s AS vector) AS qv)
                    SELECT
                    td.doc_id, td.json_id, td.text,
                    (1 - (td.text_vec <=> (SELECT qv FROM params))) AS sim  -- similarity in [0..1] if vectors are normalized
                    FROM traj_docs td
                    WHERE td.text_vec IS NOT NULL
                    ORDER BY td.text_vec <=> (SELECT qv FROM params)
                    LIMIT 1;
                """, (q_vec_lit,))
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

        # single-line enforcement between tag lines
        lines = plan_text.splitlines()
        TAG = re.compile(r"^#(Task|Agent|Dependency|ExpectedOutput)\d+:", re.M)
        idxs = [i for i, l in enumerate(lines) if TAG.match(l)] + [len(lines)]
        for i in range(len(idxs) - 1):
            head = idxs[i]
            for j in range(head + 1, idxs[i + 1]):
                if lines[j].strip() and not TAG.match(lines[j]):
                    errors.append(f"Field after '{lines[head]}' must be single-line")
                    break

        return (len(errors) == 0, errors)

    def _build_repair_prompt(self, base_prompt: str, original_plan: str, errors: list[str], agents_allowed):
        OUTPUT_MARKER = "Output (your generated plan) ⬇️:"

        # base_prompt から該当文を1回だけ除去
        base_wo_marker = base_prompt.replace(OUTPUT_MARKER, "", 1).rstrip()

        rules = (
            "Fix the issues with minimal edits. If there are no problems, you must output Original Plan as is. "
            "Output ONLY lines in this exact format:\n"
            "#TaskN: <one-line>\n"
            "#AgentN: <exact agent name>\n"
            "#DependencyN: None | #S1 #S2 ... (past steps only)\n"
            "#ExpectedOutputN: <one-line>\n"
            f"Agents allowed: {', '.join(agents_allowed)}\n"
            "Use N=1..K sequentially; counts across all tags must match.\n"
            "No extra prose."
        )

        return (
            f"{base_wo_marker}\n\n"
            "Issues:\n- " + "\n- ".join(errors) + "\n\n"
            "Original Plan:\n" + original_plan + "\n\n" + rules + "\n\n" + OUTPUT_MARKER
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


    def generate_steps(self, save_plan=False, saved_plan_filename=""):
        task = self.tasks[0]
        agent_descriptions = ""

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

        retrieved_context = self._format_search_hits_for_prompt(
            search_hits,
            max_chars=1800,
            db_url=DB_URL,               # これを渡すと trajectory のプレビューも引いてくる
            per_doc_tasks=2,
            per_task_steps=2,
        )

        print(f"retrieved_context: {retrieved_context}")
        if retrieved_context:
            prompt = (
                f"{base_prompt}\n\n"
                "### Retrieved Context (from prior runs and trajectory DB)\n"
                f"{retrieved_context}\n\n"
                "### Instruction\n"
                "- Use the context above to align your task breakdown and agent selection.\n"
                "- If similar tasks appear, re-use their structure/expected outputs when sensible.\n"
                "- Do NOT copy irrelevant content; keep the required output format strictly.\n"
            )
        else:
            prompt = base_prompt

        logger.info(f"Plan Generation Prompt (augmented): \n{prompt}")
        llm_response = watsonx_llm(prompt, model_id=self.llm)["generated_text"]
        logger.info(f"Plan (raw): \n{llm_response}")

        final_plan = llm_response
        print(f"initial_plan: {final_plan}")
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

        T = 3
        for t in range(T):
            ok, errs = self._validate_plan_text(final_plan, agents_allowed)
            print(f"ok: {ok}, errs: {errs}")
            if ok:
                break

            repair_prompt = self._build_repair_prompt(prompt, final_plan, errs, agents_allowed)
            print(f"repair_prompt: {repair_prompt}")
            final_plan = watsonx_llm(repair_prompt, model_id=self.llm)["generated_text"]
            print(f"final_plan: {final_plan}")
            logger.info("Plan was repaired based on issues:\n- " + "\n- ".join(errs))

            planned_tasks, dag_json = self._extract_planned_tasks_and_dag(final_plan, task)
            self._save_plan_round(
                DB_URL,
                plan_id=persisted_plan_id,
                scenario_id=getattr(self, "scenario_id", 0),
                llm=self.llm,
                round_t=t + 1,
                prompt=repair_prompt,   
                answer=final_plan,
                dag=dag_json
            )

        # 1) Parse validated text into fields
        task_pattern = r"#Task\d+: (.+)"
        agent_pattern = r"#Agent\d+: (.+)"
        dependency_pattern = r"#Dependency\d+: (.+)"
        output_pattern = r"#ExpectedOutput\d+: (.+)"

        tasks = re.findall(task_pattern, final_plan)
        agents = re.findall(agent_pattern, final_plan)
        dependencies = re.findall(dependency_pattern, final_plan)
        outputs = re.findall(output_pattern, final_plan)

        # 2) Optionally save plan text
        if save_plan:
            if not saved_plan_filename.endswith(".txt"):
                saved_plan_filename += ".txt"
            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename, "w") as f:
                f.write(saved_plan_text)

        # 3) Build planned_tasks
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

        logger.info(f"Planned Tasks: \n{planned_tasks}")
        return planned_tasks, self.run_id, self.plan_id


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
