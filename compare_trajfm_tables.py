#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


DEFAULT_KEYS = ["ut_id", "qid", "id", "counter"]


def load_pickle_as_df(path: str) -> pd.DataFrame:
    obj = pd.read_pickle(path)

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, pd.Series):
        df = obj.to_frame()
    elif isinstance(obj, dict):
        df = pd.DataFrame(obj)
    else:
        try:
            df = pd.DataFrame(obj)
        except Exception as e:
            raise TypeError(f"Unsupported pickle content type: {type(obj)}") from e

    return df


def choose_key(left: pd.DataFrame, right: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for key in preferred:
        if key in left.columns and key in right.columns:
            return key
    return None


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return out


def export_table(input_path: str, output_csv: str) -> None:
    df = load_pickle_as_df(input_path)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_for_csv(df).to_csv(output_path, index=False)
    print(f"[export] rows={len(df)} cols={len(df.columns)} -> {output_path}")


def build_column_summary(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_cols = set(left.columns)
    right_cols = set(right.columns)
    all_cols = sorted(left_cols | right_cols)

    rows = []
    for col in all_cols:
        rows.append({
            "column": col,
            "in_left": col in left_cols,
            "in_right": col in right_cols,
            "left_dtype": str(left[col].dtype) if col in left_cols else "",
            "right_dtype": str(right[col].dtype) if col in right_cols else "",
        })
    return pd.DataFrame(rows)


def build_bool_count_summary(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    common_cols = [c for c in left.columns if c in right.columns]

    rows = []
    for col in common_cols:
        left_is_bool_like = pd.api.types.is_bool_dtype(left[col]) or set(left[col].dropna().unique()).issubset({True, False})
        right_is_bool_like = pd.api.types.is_bool_dtype(right[col]) or set(right[col].dropna().unique()).issubset({True, False})

        if left_is_bool_like and right_is_bool_like:
            left_true = int(left[col].fillna(False).astype(bool).sum())
            right_true = int(right[col].fillna(False).astype(bool).sum())
            rows.append({
                "column": col,
                "left_true_count": left_true,
                "right_true_count": right_true,
                "left_false_count": int(len(left) - left_true),
                "right_false_count": int(len(right) - right_true),
                "delta_true_count": left_true - right_true,
            })

    return pd.DataFrame(rows).sort_values("column") if rows else pd.DataFrame(
        columns=[
            "column",
            "left_true_count",
            "right_true_count",
            "left_false_count",
            "right_false_count",
            "delta_true_count",
        ]
    )


def build_row_diff(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    common_cols = [c for c in left.columns if c in right.columns]
    common_cols = [c for c in common_cols if c != key]

    left2 = left[[key] + common_cols].copy()
    right2 = right[[key] + common_cols].copy()

    left2 = left2.sort_values(key).drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    right2 = right2.sort_values(key).drop_duplicates(subset=[key], keep="first").reset_index(drop=True)

    merged = left2.merge(
        right2,
        on=key,
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )

    diff_rows = []
    for _, row in merged.iterrows():
        status = row["_merge"]
        if status != "both":
            diff_rows.append({
                key: row[key],
                "difference_type": f"row_only_in_{status}",
                "column": "",
                "left_value": "",
                "right_value": "",
            })
            continue

        for col in common_cols:
            lv = row[f"{col}_left"]
            rv = row[f"{col}_right"]

            equal = (pd.isna(lv) and pd.isna(rv)) or (lv == rv)
            if not equal:
                diff_rows.append({
                    key: row[key],
                    "difference_type": "value_mismatch",
                    "column": col,
                    "left_value": lv,
                    "right_value": rv,
                })

    row_diff_df = pd.DataFrame(diff_rows)

    direct_compare_df = None
    left_cmp = left2.set_index(key)
    right_cmp = right2.set_index(key)

    common_index = left_cmp.index.intersection(right_cmp.index)
    left_cmp = left_cmp.loc[common_index].sort_index()
    right_cmp = right_cmp.loc[common_index].sort_index()

    if list(left_cmp.columns) == list(right_cmp.columns) and list(left_cmp.index) == list(right_cmp.index):
        direct_compare_df = left_cmp.compare(
            right_cmp,
            keep_shape=False,
            keep_equal=False,
            result_names=("left", "right"),
        ).reset_index()

    return row_diff_df, direct_compare_df


def compare_tables(left_path: str, right_path: str, outdir: str, left_name: str, right_name: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    left = load_pickle_as_df(left_path)
    right = load_pickle_as_df(right_path)

    sanitize_for_csv(left).to_csv(out / f"{left_name}.csv", index=False)
    sanitize_for_csv(right).to_csv(out / f"{right_name}.csv", index=False)

    summary = {
        "left_name": left_name,
        "right_name": right_name,
        "left_path": left_path,
        "right_path": right_path,
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "left_cols": int(len(left.columns)),
        "right_cols": int(len(right.columns)),
        "left_columns": list(map(str, left.columns)),
        "right_columns": list(map(str, right.columns)),
        "common_columns": sorted(list(set(left.columns) & set(right.columns))),
        "left_only_columns": sorted(list(set(left.columns) - set(right.columns))),
        "right_only_columns": sorted(list(set(right.columns) - set(left.columns))),
    }

    key = choose_key(left, right, DEFAULT_KEYS)
    summary["comparison_key"] = key

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    col_summary = build_column_summary(left, right)
    col_summary.to_csv(out / "column_summary.csv", index=False)

    bool_summary = build_bool_count_summary(left, right)
    bool_summary.to_csv(out / "bool_count_summary.csv", index=False)

    if key is not None:
        row_diff_df, direct_compare_df = build_row_diff(left, right, key=key)
        sanitize_for_csv(row_diff_df).to_csv(out / "row_diff.csv", index=False)

        if direct_compare_df is not None:
            sanitize_for_csv(direct_compare_df).to_csv(out / "direct_compare.csv", index=False)
    else:
        print("[compare] No common key found among:", DEFAULT_KEYS)

    print(f"[compare] wrote outputs to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_export = subparsers.add_parser("export", help="Export one pickle table to CSV")
    p_export.add_argument("--input", required=True)
    p_export.add_argument("--output", required=True)

    p_compare = subparsers.add_parser("compare", help="Compare two pickle tables")
    p_compare.add_argument("--left", required=True)
    p_compare.add_argument("--right", required=True)
    p_compare.add_argument("--outdir", required=True)
    p_compare.add_argument("--left-name", default="left")
    p_compare.add_argument("--right-name", default="right")

    args = parser.parse_args()

    if args.command == "export":
        export_table(args.input, args.output)
    elif args.command == "compare":
        compare_tables(
            left_path=args.left,
            right_path=args.right,
            outdir=args.outdir,
            left_name=args.left_name,
            right_name=args.right_name,
        )


if __name__ == "__main__":
    main()