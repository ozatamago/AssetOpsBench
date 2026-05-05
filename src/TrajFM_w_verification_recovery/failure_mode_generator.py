import faulthandler
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils import get_llm_answer_from_json, extract_json_from_response


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Timer:
    def __init__(self, label: str, extra: str = ""):
        self.label = label
        self.extra = extra
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[{_ts()}] [TIMER-START] {self.label} {self.extra}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        print(f"[{_ts()}] [TIMER-END]   {self.label} {self.extra} -> {dt:.3f}s", flush=True)
        return False


def _load_all_json_files(root_path: str) -> Dict[str, Any]:
    """Load JSON files recursively under root_path."""
    json_data: Dict[str, Any] = {}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    json_data[file_path] = data
            except Exception:
                # Skip non-JSON or unreadable files
                pass
    return json_data


_QID_RE = re.compile(r"(Q_\d+)")


def _extract_ut_id(path: str) -> str:
    """
    Extract a unit-test / query id from filename.

    Examples:
      Q_102_trajectory.json -> Q_102
      abc_trajectory.json   -> abc
      foo.json              -> foo
    """
    name = Path(path).name
    match = _QID_RE.search(name)
    if match:
        return match.group(1)

    stem = Path(path).stem
    if stem.endswith("_trajectory"):
        return stem[: -len("_trajectory")]
    return stem


def _resolve_timestamp_roots(
    timestamps: Optional[Sequence[str]],
    traj_root_base: str,
) -> List[Tuple[str, str]]:
    """
    Resolve timestamp labels and corresponding directories.

    Behavior:
    - If `timestamps` is provided, treat each item as a subdirectory under traj_root_base
      when it exists there; otherwise treat it as a direct path.
    - If `timestamps` is None and traj_root_base has subdirectories, auto-discover them.
    - If `timestamps` is None and traj_root_base has no subdirectories, process traj_root_base
      itself as a single timestamp named "1".
    """
    base = Path(traj_root_base)

    if timestamps:
        resolved: List[Tuple[str, str]] = []
        for ts in timestamps:
            candidate = base / ts
            if candidate.exists() and candidate.is_dir():
                resolved.append((str(ts), str(candidate)))
            else:
                resolved.append((str(ts), str(Path(ts))))
        return resolved

    if base.exists() and base.is_dir():
        subdirs = sorted([p for p in base.iterdir() if p.is_dir()])
        if subdirs:
            return [(p.name, str(p)) for p in subdirs]

    return [("1", str(base))]


def process_trajectories(
    timestamps: Optional[Sequence[str]] = None,
    traj_root_base: str = ".",
    model_id: int = 18,
    out_dir: str = "processed_trajectories",
):
    """
    Process trajectories using an LLM and save per-timestamp + combined pickles.

    This version only keeps the predefined failure modes (1.1 ~ 3.3).
    It does NOT process or store additional failure modes.
    """
    failure_mode_keys = [
        "1.1 Disobey Task Specification",
        "1.2 Disobey Role Specification",
        "1.3 Step Repetition",
        "1.4 Loss of Conversation History",
        "1.5 Unaware of Termination Conditions",
        "2.1 Conversation Reset",
        "2.2 Fail to Ask for Clarification",
        "2.3 Task Derailment",
        "2.4 Information Withholding",
        "2.5 Ignored Other Agent's Input",
        "2.6 Action-Reasoning Mismatch",
        "3.1 Failure Signal Miss or Misdetection",
        "3.2 Failure Root Not Isolated",
        "3.3 Failure Representation Breakdown",
        "4.1 Diagnosis Compression Mismatch",
        "4.2 Unsupported Fault Hypothesis",
        "4.3 Missing Upstream Repair Signal",
        "5.1 Fault Misidentification",
        "5.2 Incorrect Probe Selection",
        "5.3 Unsafe or Improper Termination",
    ]

    df_columns = [
        "model_id",
        "counter",
        "timestamp",
        "vendor",
        "model",
        "ut_id",
    ] + failure_mode_keys

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    per_timestamp_paths: List[str] = []
    all_dfs: List[pd.DataFrame] = []

    timestamp_roots = _resolve_timestamp_roots(timestamps, traj_root_base)
    print(f"[{_ts()}] Resolved timestamps: {timestamp_roots}", flush=True)

    for timestamp, root_directory in timestamp_roots:
        print(f"\n[{_ts()}] Processing timestamp={timestamp} root={root_directory}", flush=True)

        with Timer("load_all_json_files", extra=f"root={root_directory}"):
            all_jsons = _load_all_json_files(root_directory)

        print(f"[{_ts()}] Loaded {len(all_jsons)} files", flush=True)

        rows: List[Dict[str, Any]] = []
        counter = 1

        for path, content in all_jsons.items():
            print(f"\n[{_ts()}] [FILE] counter={counter} path={path}", flush=True)

            with Timer("path_parse"):
                ut_id = _extract_ut_id(path)

            model = model_id
            vendor = ""

            max_trial = 2
            success = False

            for cur_trial in range(1, max_trial + 1):
                print(f"[{_ts()}] [TRY] {cur_trial}/{max_trial} ut_id={ut_id}", flush=True)

                faulthandler.dump_traceback_later(30, repeat=False)

                try:
                    with Timer("get_llm_answer_from_json"):
                        raw_output = get_llm_answer_from_json(data=content, model_id=model_id)

                    faulthandler.cancel_dump_traceback_later()

                    with Timer("response_text_normalize"):
                        if isinstance(raw_output, dict):
                            response_text = raw_output.get("generated_text", "")
                            print(
                                f"[{_ts()}] [LLM] stop_reason={raw_output.get('stop_reason')} "
                                f"in_tok={raw_output.get('input_token_count')} "
                                f"out_tok={raw_output.get('generated_token_count')}",
                                flush=True,
                            )
                        else:
                            response_text = str(raw_output)

                        print(f"[{_ts()}] [LLM] response_chars={len(response_text)}", flush=True)

                    with Timer("extract_json_from_response"):
                        response_json = extract_json_from_response(response_text)

                    with Timer("row_build"):
                        failure_modes = response_json.get("failure_modes", {})

                        row = {
                            "model_id": model_id,
                            "counter": counter,
                            "timestamp": timestamp,
                            "vendor": vendor,
                            "model": model,
                            "ut_id": ut_id,
                        }

                        for key in failure_mode_keys:
                            row[key] = bool(failure_modes.get(key, False))

                    rows.append(row)
                    success = True
                    break

                except Exception as e:
                    faulthandler.cancel_dump_traceback_later()
                    print(f"[{_ts()}] [ERROR] Failed to process {path}: {e}", flush=True)

            if not success:
                print(f"[{_ts()}] [WARN] Giving up on {path} after {max_trial} trials", flush=True)

            counter += 1

        df = pd.DataFrame(rows, columns=df_columns)

        df_file_path = str(Path(out_dir) / f"{timestamp}_m{model_id}_db.pkl")
        with Timer("to_pickle_per_timestamp", extra=f"rows={len(df)} path={df_file_path}"):
            df.to_pickle(df_file_path)

        per_timestamp_paths.append(df_file_path)
        all_dfs.append(df)
        print(f"[{_ts()}] Saved {df_file_path} with {len(df)} rows", flush=True)

    with Timer("concat_all_dfs"):
        combined_df = (
            pd.concat(all_dfs, ignore_index=True)
            if all_dfs
            else pd.DataFrame(columns=df_columns)
        )

    combined_file_path = str(Path(out_dir) / f"combined_m{model_id}_db.pkl")
    with Timer("to_pickle_combined", extra=f"rows={len(combined_df)} path={combined_file_path}"):
        combined_df.to_pickle(combined_file_path)

    print(
        f"\n[{_ts()}] Saved combined DataFrame: {combined_file_path} ({len(combined_df)} rows)",
        flush=True,
    )

    return {
        "per_timestamp_paths": per_timestamp_paths,
        "combined_path": combined_file_path,
        "combined_df": combined_df,
    }