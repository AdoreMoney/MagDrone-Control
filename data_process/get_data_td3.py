import os
import sys
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def find_event_file(log_dir: str) -> Path:
    p = Path(log_dir)
    if not p.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
    candidates = sorted(p.rglob("events.out.tfevents.*"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No events.out.tfevents.* files found in the directory: {log_dir}")
    print(f"[INFO] Automatically selected event file: {candidates[0]}")
    return candidates[0]


def read_tb_scalars_to_df(event_path: str, tags=None, size_guidance=None):
    path = Path(event_path)
    if path.is_dir():
        path = find_event_file(str(path))

    if not path.is_file():
        raise FileNotFoundError(f"Event file not found: {path}")

    ea = event_accumulator.EventAccumulator(str(path), size_guidance=size_guidance or {"scalars": 0})
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if not available_tags:
        print("[WARN] No scalar data exists in the event file.")
        return pd.DataFrame()

    if tags is None:
        tags = available_tags
    else:
        tags = [t.strip() for t in tags]
        missing = [t for t in tags if t not in available_tags]
        if missing:
            print(f"[WARN] The following tags do not exist: {missing}")
        tags = [t for t in tags if t in available_tags]
        if not tags:
            print("[WARN] No valid tags have been selected.")
            return pd.DataFrame()

    # Read and aggregate by step (retain the last entry for each step)
    dfs = []
    for tag in tags:
        items = ea.Scalars(tag)
        df = pd.DataFrame(items)[["step", "wall_time", "value"]].rename(columns={"value": tag})
        df = df.drop_duplicates(subset=["step"], keep="last")  # Retain only the last entry for each step
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Outer join with step as the primary key: only merge the value column on the right to avoid wall_time conflicts
    df_all = dfs[0]
    for df in dfs[1:]:
        tag_name = df.columns[-1]  # Column name of the current tag
        df_right = df[["step", tag_name]]
        df_all = df_all.merge(df_right, on="step", how="outer")

    df_all = df_all.sort_values("step").reset_index(drop=True)
    return df_all


if __name__ == "__main__":
    #event_path = r"C:\Users\a1fla\Desktop\MagDrone\logs\td3\MagDrone_td3_tensorboard\td3_20251208_1921\TD3_0"
    #event_path = r"C:\Users\a1fla\Desktop\MagDrone\logs\td3\MagDrone_td3_tensorboard\td3_20251209_1757\TD3_0"
    #event_path = r"C:\Users\a1fla\Desktop\MagDrone\logs\td3\MagDrone_td3_tensorboard\td3_20251209_1901\TD3_0"
    event_path = r"C:\Users\a1fla\Desktop\MagDrone\logs\td3\MagDrone_td3_tensorboard\td3_20251210_1416\TD3_0"

    target_tags = [
        "eval/mean_reward",
        "rollout/ep_rew_mean",
        "train/learning_rate",
        "train/actor_loss",
        "train/critic_loss"
    ]

    size_guidance = {"scalars": 0}

    try:
        df_export = read_tb_scalars_to_df(event_path, tags=target_tags, size_guidance=size_guidance)

        parts = Path(event_path).parts
        run_name = None
        # Find folder names in the format of td3_20251202_2010
        for p in parts:
            if p.startswith("td3_") and len(p) >= 13 and p[4:12].isdigit() and p[12] == '_' and p[13:17].isdigit():
                run_name = p
                break
        if run_name is None:
            # If not found, use the second last folder name as a fallback
            run_name = Path(event_path).parts[-2]

        out_name = f"{run_name}.csv"  
        out_path = Path(event_path).parents[1] / out_name

        df_export.to_csv(out_path, index=False)
        print(f"Successfully exported to: {out_path}")
        print(df_export.head())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)