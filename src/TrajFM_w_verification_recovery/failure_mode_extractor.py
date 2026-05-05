import argparse
from failure_mode_generator import process_trajectories


def main():
    """
    Run the failure mode extraction pipeline.

    This simplified version only:
      1) Reads trajectories
      2) Labels the predefined failure modes (1.1 ~ 3.3)
      3) Writes the combined pickle output

    It does NOT:
      - cluster additional failure modes
      - run sentence-transformer embeddings
      - export clustered CSV summaries
    """
    parser = argparse.ArgumentParser(
        description="Analyze LLM execution trajectories and label predefined failure modes."
    )
    parser.add_argument(
        "--traj_directory",
        type=str,
        default="./localtemp/trajectory/",
        help="Path to the root directory containing per-timestamp trajectory folders.",
    )
    parser.add_argument(
        "--backstage_directory",
        type=str,
        default=".",
        help="(Optional) Path to auxiliary resources (unused, kept for compatibility).",
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=18,
        help="Model ID passed to the generator step.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="processed_trajectories",
        help="Directory to write the combined pickle output.",
    )
    parser.add_argument(
        "--timestamps",
        nargs="*",
        default=None,
        help="Optional list of timestamps to process. If omitted, auto-discovers all subfolders.",
    )

    args = parser.parse_args()

    print(f"args: {args}", flush=True)

    # Step 1: Generate combined pickle (auto-discovers timestamps if not provided)
    gen = process_trajectories(
        timestamps=args.timestamps,   # None => auto-discover
        traj_root_base=args.traj_directory,
        model_id=args.model_id,
        out_dir=args.out_dir,
    )

    print("\n[Step 1] Combined pickle:", gen["combined_path"], flush=True)

    if "combined_df" in gen and gen["combined_df"] is not None:
        print("[Step 1] Preview:", flush=True)
        print(gen["combined_df"].head(), flush=True)

    print("\nFailure mode extraction completed successfully.", flush=True)


if __name__ == "__main__":
    main()