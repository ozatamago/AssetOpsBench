from failure_mode_generator import process_trajectories  # Step 1 (generation only)


def run_failure_mode_pipeline(
    traj_root_base: str,
    model_id: int = 18,
    timestamps=None,  # None => auto-discover subfolders
    out_dir: str = "processed_trajectories",
):
    """
    Run the simplified failure mode pipeline.

    This version only performs:
      1) trajectory processing
      2) labeling of predefined failure modes (1.1 ~ 3.3)
      3) saving the combined pickle output

    It does NOT perform:
      - additional failure mode reduction
      - title embedding
      - clustering
      - summary CSV generation
    """

    # Step 1: generate + save combined pickle
    gen = process_trajectories(
        timestamps=timestamps,  # None => auto-discover
        traj_root_base=traj_root_base,
        model_id=model_id,
        out_dir=out_dir,
    )

    print("Combined pickle:", gen["combined_path"])
    if "combined_df" in gen and gen["combined_df"] is not None:
        print(gen["combined_df"].head())

    # Return generation result only
    return {"generation": gen}