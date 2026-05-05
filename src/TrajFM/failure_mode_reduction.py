# import pandas as pd
# from pathlib import Path
# from typing import Optional, Dict, Any, Tuple, List


# def failure_mode_reduction(
#     combined_pickle_path: str,
#     out_dir: str = "summary",
#     model_name: str = "all-MiniLM-L6-v2",
#     k: Optional[int] = None,
#     k_min: int = 2,
#     k_max: int = 7,
#     verbose: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Reduce additional failure modes by exploding, embedding, clustering, and labeling.

#     Returns
#     -------
#     dict with:
#       - 'df_expanded': tidy dataframe with ['title','description']
#       - 'df_clustered': clustered dataframe with ['cluster','failure mode','title','description']
#       - 'k': number of clusters used
#       - 'silhouette_scores': List[Tuple[int, float]] if k was auto-selected, else []
#       - 'paths': {'addtional_fm_csv', 'additional_fm_clustered_csv'}
#     """
#     if verbose:
#         print(f"Loading combined pickle: {combined_pickle_path}")
#     df = pd.read_pickle(combined_pickle_path)
#     print (df)

#     # --- Step 3: explode addi_fm_list -> title/description ---
#     if verbose:
#         print("Exploding additional failure modes...")
#     if "addi_fm_cnt" not in df.columns or "addi_fm_list" not in df.columns:
#         raise KeyError("Expected columns 'addi_fm_cnt' and 'addi_fm_list' not found.")

#     df_new_fm = df[df["addi_fm_cnt"] > 0][["addi_fm_cnt", "addi_fm_list"]].copy()
#     df_new_fm.reset_index(drop=True, inplace=True)

#     df_exploded = df_new_fm.explode("addi_fm_list", ignore_index=True)
#     df_expanded = pd.concat(
#         [
#             df_exploded.drop(columns=["addi_fm_list"]),
#             pd.json_normalize(df_exploded["addi_fm_list"]),
#         ],
#         axis=1,
#     )

#     keep_cols = [c for c in ["title", "description"] if c in df_expanded.columns]
#     if not keep_cols:
#         raise KeyError(
#             "No 'title'/'description' columns found inside 'addi_fm_list' items."
#         )
#     df_expanded = df_expanded[keep_cols].copy()

#     # Save the “addtional_fm.csv” (typo preserved to match notebook)
#     out = Path(out_dir)
#     out.mkdir(parents=True, exist_ok=True)
#     addtional_csv = out / "addtional_fm.csv"
#     df_expanded.to_csv(addtional_csv, index=False)
#     if verbose:
#         print(f"Saved: {addtional_csv} (rows={len(df_expanded)})")

#     # --- Step 4/5: embeddings + clustering with small-sample handling ---
#     titles = df_expanded["title"].fillna("").astype(str).tolist()
#     n = len(titles)

#     # n == 0: nothing to do
#     if n == 0:
#         if verbose:
#             print("No titles to cluster. Returning early.")
#         return {
#             "df_expanded": df_expanded,
#             "df_clustered": pd.DataFrame(
#                 columns=["cluster", "failure mode", "title", "description"]
#             ),
#             "k": 0,
#             "silhouette_scores": [],
#             "paths": {
#                 "addtional_fm_csv": str(addtional_csv),
#                 "additional_fm_clustered_csv": None,
#             },
#         }

#     # n == 1: assign a single cluster without embeddings
#     if n == 1:
#         df_clustered = df_expanded.copy()
#         df_clustered["cluster"] = 0
#         df_clustered["failure mode"] = df_clustered["title"]
#         clustered_csv = out / "additional_fm_clustered.csv"
#         df_clustered[["cluster", "failure mode", "title", "description"]].to_csv(
#             clustered_csv, index=False
#         )
#         if verbose:
#             print(f"Single item: saved {clustered_csv}")
#         return {
#             "df_expanded": df_expanded,
#             "df_clustered": df_clustered[
#                 ["cluster", "failure mode", "title", "description"]
#             ],
#             "k": 1,
#             "silhouette_scores": [],
#             "paths": {
#                 "addtional_fm_csv": str(addtional_csv),
#                 "additional_fm_clustered_csv": str(clustered_csv),
#             },
#         }

#     # n >= 2: embed
#     if verbose:
#         print(f"Embedding {n} titles with {model_name} ...")
#     from sentence_transformers import SentenceTransformer
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score
#     from sklearn.metrics.pairwise import euclidean_distances
#     import numpy as np

#     model = SentenceTransformer(model_name)
#     embeddings = model.encode(titles, convert_to_numpy=True, show_progress_bar=False)

#     silhouette_scores: List[Tuple[int, float]] = []

#     # n == 2: only valid K is 2 for silhouette constraints
#     if n == 2:
#         k = 2
#         if verbose:
#             print("Only two samples detected; using K=2.")
#     else:
#         if k is None:
#             lo = max(2, k_min)
#             hi = min(k_max, n - 1)  # silhouette requires k <= n-1
#             if lo > hi:
#                 # Not enough samples for a range; fall back to a valid K
#                 k = min(2, n - 1)
#                 if verbose:
#                     print(f"Insufficient samples for a K range; using K={k}.")
#             else:
#                 if verbose:
#                     print(f"Selecting K by silhouette over [{lo}..{hi}]")
#                 best_k, best_score = None, -1.0
#                 for cand in range(lo, hi + 1):
#                     km = KMeans(n_clusters=cand, random_state=42, n_init="auto")
#                     labels = km.fit_predict(embeddings)
#                     # If all points fall into one cluster (identical embeddings), silhouette is invalid
#                     if len(set(labels)) <= 1:
#                         score = -1.0
#                     else:
#                         score = float(silhouette_score(embeddings, labels))
#                     silhouette_scores.append((cand, score))
#                     if score > best_score:
#                         best_k, best_score = cand, score
#                 k = best_k or min(2, n - 1)
#                 if verbose:
#                     print("Silhouette scores:", silhouette_scores)
#                     print(f"Chosen K = {k}")
#         else:
#             # user-provided K → clamp safely
#             if n <= 2:
#                 k = 2
#             else:
#                 k = max(2, min(int(k), n - 1))
#             if verbose:
#                 print(f"Using K = {k} (validated for n={n})")

#     # Final clustering
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
#     clusters = kmeans.fit_predict(embeddings)

#     df_clustered = df_expanded.copy()
#     df_clustered["cluster"] = clusters

#     # Representative (closest to centroid) title per cluster
#     if verbose:
#         print("Selecting representative title for each cluster...")
#     representative_titles: List[Tuple[int, str]] = []
#     for cl in range(k):
#         idxs = df_clustered.index[df_clustered["cluster"] == cl].tolist()
#         if not idxs:
#             continue
#         dists = euclidean_distances(
#             embeddings[idxs], [kmeans.cluster_centers_[cl]]
#         ).flatten()
#         closest_local = int(np.argmin(dists))
#         rep_idx = idxs[closest_local]
#         representative_titles.append((cl, df_clustered.loc[rep_idx, "title"]))

#     if verbose and representative_titles:
#         print("\nRepresentative titles:")
#         for cl, title in representative_titles:
#             print(f"  Cluster {cl}: {title}")

#     cluster_to_title = dict(representative_titles)
#     df_clustered["failure mode"] = df_clustered["cluster"].map(cluster_to_title)

#     # final column order
#     cols = ["cluster", "failure mode", "title", "description"]
#     df_clustered = df_clustered[cols].copy()

#     clustered_csv = out / "additional_fm_clustered.csv"
#     df_clustered.to_csv(clustered_csv, index=False)
#     if verbose:
#         print(f"Saved: {clustered_csv} (rows={len(df_clustered)})")

#     return {
#         "df_expanded": df_expanded,
#         "df_clustered": df_clustered,
#         "k": k,
#         "silhouette_scores": silhouette_scores,
#         "paths": {
#             "addtional_fm_csv": str(addtional_csv),
#             "additional_fm_clustered_csv": str(clustered_csv),
#         },
#     }


import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import time
import os
import numpy as np
import os, time, socket, platform

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        print(f"[{_ts()}] [TIMER-START] {self.name}", flush=True)
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        print(f"[{_ts()}] [TIMER-END]   {self.name} -> {dt:.3f}s", flush=True)
        # 例外は握りつぶさない
        return False
    
def vlog(msg):
    print(f"[{_ts()}] {msg}", flush=True)

def show_env_debug():
    vlog(f"[HOST] hostname={socket.gethostname()} platform={platform.platform()}")
    vlog(f"[ENV] HF_HOME={os.getenv('HF_HOME')}")
    vlog(f"[ENV] HF_HUB_CACHE={os.getenv('HF_HUB_CACHE')}")
    vlog(f"[ENV] TRANSFORMERS_CACHE={os.getenv('TRANSFORMERS_CACHE')}")
    vlog(f"[ENV] SENTENCE_TRANSFORMERS_HOME={os.getenv('SENTENCE_TRANSFORMERS_HOME')}")

    # 既定キャッシュ候補（HFの既定は ~/.cache/huggingface/hub）:contentReference[oaicite:1]{index=1}
    home = str(Path.home())
    default_cache = Path(home) / ".cache" / "huggingface" / "hub"
    vlog(f"[CACHE] default_candidate={default_cache} exists={default_cache.exists()}")

def show_torch_debug():
    try:
        import torch
        vlog(f"[TORCH] version={torch.__version__}")
        vlog(f"[TORCH] cuda.is_available={torch.cuda.is_available()}")  # :contentReference[oaicite:2]{index=2}
        if torch.cuda.is_available():
            vlog(f"[TORCH] cuda.device_count={torch.cuda.device_count()}")
            vlog(f"[TORCH] cuda.current_device={torch.cuda.current_device()}")
            vlog(f"[TORCH] cuda.device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        vlog(f"[TORCH] import/check failed: {e}")


def failure_mode_reduction(
    combined_pickle_path: str,
    out_dir: str = "summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Reduce additional failure modes by exploding, embedding, clustering, and labeling.
    """

    vlog(f"[START] failure_mode_reduction")
    vlog(f"[INPUT] combined_pickle_path={combined_pickle_path}")
    vlog(f"[INPUT] out_dir={out_dir} model_name={model_name} k={k} k_min={k_min} k_max={k_max}")

    # --- Step 1: load pickle ---
    with Timer("read_pickle"):
        if not os.path.exists(combined_pickle_path):
            raise FileNotFoundError(f"Pickle not found: {combined_pickle_path}")
        df = pd.read_pickle(combined_pickle_path)  # docs: load pickled object :contentReference[oaicite:1]{index=1}

    vlog(f"[DF] shape={df.shape} cols={list(df.columns)}")
    vlog(f"[DF] head(2)=\n{df.head(2)}")

    # --- Step 2: column validation ---
    with Timer("validate_columns"):
        missing = [c for c in ["addi_fm_cnt", "addi_fm_list"] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns: {missing}")
        vlog("[OK] required columns exist: addi_fm_cnt, addi_fm_list")

    # --- Step 3: explode addi_fm_list -> title/description ---
    with Timer("explode_additional_failure_modes"):
        vlog("Exploding additional failure modes (filter addi_fm_cnt > 0)...")
        df_new_fm = df[df["addi_fm_cnt"] > 0][["addi_fm_cnt", "addi_fm_list"]].copy()
        df_new_fm.reset_index(drop=True, inplace=True)
        vlog(f"[DF_NEW_FM] shape={df_new_fm.shape}")

        if len(df_new_fm) > 0:
            sample = df_new_fm["addi_fm_list"].iloc[0]
            vlog(f"[DF_NEW_FM] sample addi_fm_list[0] type={type(sample)} value_head={str(sample)[:200]}")

        df_exploded = df_new_fm.explode("addi_fm_list", ignore_index=True)
        vlog(f"[DF_EXPLODED] shape={df_exploded.shape}")

        df_expanded = pd.concat(
            [
                df_exploded.drop(columns=["addi_fm_list"]),
                pd.json_normalize(df_exploded["addi_fm_list"]),
            ],
            axis=1,
        )
        vlog(f"[DF_EXPANDED_RAW] shape={df_expanded.shape} cols={list(df_expanded.columns)}")

        keep_cols = [c for c in ["title", "description"] if c in df_expanded.columns]
        if not keep_cols:
            raise KeyError("No 'title'/'description' columns found inside 'addi_fm_list' items.")
        df_expanded = df_expanded[keep_cols].copy()
        vlog(f"[DF_EXPANDED] shape={df_expanded.shape} keep_cols={keep_cols}")

    # --- Step 3.5: save addtional_fm.csv ---
    with Timer("save_addtional_fm_csv"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        addtional_csv = out / "addtional_fm.csv"
        df_expanded.to_csv(addtional_csv, index=False)  # index=False etc. documented :contentReference[oaicite:2]{index=2}
        vlog(f"[SAVED] {addtional_csv} (rows={len(df_expanded)})")
        vlog(f"[SAVED] exists={addtional_csv.exists()} size={addtional_csv.stat().st_size if addtional_csv.exists() else 'NA'}")

    # --- Step 4/5: embeddings + clustering with small-sample handling ---
    titles = df_expanded["title"].fillna("").astype(str).tolist()
    n = len(titles)
    vlog(f"[TITLES] n={n} first3={[t[:80] for t in titles[:3]]}")

    # n == 0: nothing to do
    if n == 0:
        vlog("[EARLY-RETURN] No titles to cluster.")
        return {
            "df_expanded": df_expanded,
            "df_clustered": pd.DataFrame(columns=["cluster", "failure mode", "title", "description"]),
            "k": 0,
            "silhouette_scores": [],
            "paths": {
                "addtional_fm_csv": str(addtional_csv),
                "additional_fm_clustered_csv": None,
            },
        }

    # n == 1: assign a single cluster without embeddings
    if n == 1:
        with Timer("single_item_write_clustered_csv"):
            df_clustered = df_expanded.copy()
            df_clustered["cluster"] = 0
            df_clustered["failure mode"] = df_clustered["title"]
            clustered_csv = out / "additional_fm_clustered.csv"
            df_clustered[["cluster", "failure mode", "title", "description"]].to_csv(clustered_csv, index=False)
            vlog(f"[SINGLE] saved {clustered_csv} exists={clustered_csv.exists()}")
        return {
            "df_expanded": df_expanded,
            "df_clustered": df_clustered[["cluster", "failure mode", "title", "description"]],
            "k": 1,
            "silhouette_scores": [],
            "paths": {
                "addtional_fm_csv": str(addtional_csv),
                "additional_fm_clustered_csv": str(clustered_csv),
            },
        }
    
    show_env_debug()
    show_torch_debug()

    # n >= 2: embed
    with Timer("import_and_embed"):
        vlog(f"Embedding {n} titles with SentenceTransformer('{model_name}') ...")

    with Timer("SentenceTransformer_init"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)   # ここが「DL/ロード」で詰まることがある

    vlog(f"[MODEL] type={type(model)}")

    with Timer("encode"):
        embeddings = model.encode(
            titles,
            convert_to_numpy=True,
            show_progress_bar=True,   # まず True にして進捗を出す
            batch_size=64,            # 明示（環境により調整）
        )

    vlog(f"[EMB] shape={embeddings.shape} dtype={embeddings.dtype}")


    # clustering imports
    with Timer("import_clustering_modules"):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import euclidean_distances

    silhouette_scores: List[Tuple[int, float]] = []

    # n == 2: only valid K is 2 for silhouette constraints
    if n == 2:
        k = 2
        vlog("[K] Only two samples; using K=2 (silhouette selection not applicable).")
    else:
        if k is None:
            lo = max(2, k_min)
            hi = min(k_max, n - 1)  # silhouette requires k <= n-1 :contentReference[oaicite:3]{index=3}
            vlog(f"[K] auto-select by silhouette: candidate range lo={lo} hi={hi} (n={n})")

            if lo > hi:
                k = min(2, n - 1)
                vlog(f"[K] insufficient range; fallback K={k}")
            else:
                best_k, best_score = None, -1.0
                with Timer("silhouette_sweep"):
                    for cand in range(lo, hi + 1):
                        km = KMeans(n_clusters=cand, random_state=42, n_init="auto")
                        labels = km.fit_predict(embeddings)
                        if len(set(labels)) <= 1:
                            score = -1.0
                        else:
                            score = float(silhouette_score(embeddings, labels))
                        silhouette_scores.append((cand, score))
                        vlog(f"[K-SWEEP] cand={cand} score={score:.6f}")
                        if score > best_score:
                            best_k, best_score = cand, score
                k = best_k or min(2, n - 1)
                vlog(f"[K] chosen={k} best_score={best_score:.6f}")
        else:
            if n <= 2:
                k = 2
            else:
                k = max(2, min(int(k), n - 1))
            vlog(f"[K] user-provided -> validated K={k} for n={n}")

    # Final clustering
    with Timer("kmeans_fit_predict"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(embeddings)
        vlog(f"[CLUSTERS] k={k} unique={sorted(set(map(int, clusters)))}")

    df_clustered = df_expanded.copy()
    df_clustered["cluster"] = clusters

    # Representative (closest to centroid) title per cluster
    with Timer("select_representatives"):
        vlog("Selecting representative title for each cluster...")
        representative_titles: List[Tuple[int, str]] = []
        for cl in range(k):
            idxs = df_clustered.index[df_clustered["cluster"] == cl].tolist()
            vlog(f"[REP] cluster={cl} size={len(idxs)}")
            if not idxs:
                continue
            dists = euclidean_distances(embeddings[idxs], [kmeans.cluster_centers_[cl]]).flatten()
            closest_local = int(dists.argmin())
            rep_idx = idxs[closest_local]
            representative_titles.append((cl, df_clustered.loc[rep_idx, "title"]))
        vlog(f"[REP] representatives={representative_titles}")

    cluster_to_title = dict(representative_titles)
    df_clustered["failure mode"] = df_clustered["cluster"].map(cluster_to_title)

    cols = ["cluster", "failure mode", "title", "description"]
    df_clustered = df_clustered[cols].copy()

    # Save clustered csv
    with Timer("save_clustered_csv"):
        clustered_csv = out / "additional_fm_clustered.csv"
        df_clustered.to_csv(clustered_csv, index=False)
        vlog(f"[SAVED] {clustered_csv} (rows={len(df_clustered)})")
        vlog(f"[SAVED] exists={clustered_csv.exists()} size={clustered_csv.stat().st_size if clustered_csv.exists() else 'NA'}")

    vlog("[DONE] failure_mode_reduction")
    return {
        "df_expanded": df_expanded,
        "df_clustered": df_clustered,
        "k": k,
        "silhouette_scores": silhouette_scores,
        "paths": {
            "addtional_fm_csv": str(addtional_csv),
            "additional_fm_clustered_csv": str(clustered_csv),
        },
    }
