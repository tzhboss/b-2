#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from reference_frame_probe import (
    feature_columns,
    fit_eval,
    make_within_speaker_folds,
    paired_deltas,
    prepare_features,
    summarize_metrics,
    write_run_metadata,
)


BASE_COLUMNS = [
    "sample_id",
    "dataset",
    "speaker_id",
    "gender_canonical",
    "emotion_5class_candidate",
    "task_eligible",
    "emotion_training_usable",
    "f0_median_hz",
    "pitch_relative_st",
    "speech_lufs",
    "loudness_relative_lu",
    "phoneme_articulation_rate",
    "rate_relative_ratio",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    data_env = cfg["dataset"]["source_env"]
    if data_env not in os.environ:
        raise RuntimeError(f"missing required environment variable: {data_env}")
    data_path = Path(os.environ[data_env]).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_parquet(data_path, columns=BASE_COLUMNS)
    df = df[df["dataset"].isin(cfg["dataset"]["datasets"])].copy()
    df = prepare_features(df)

    artifact_root = Path(cfg["outputs"]["artifact_root"])
    if args.dry_run:
        print(
            {
                "data_path": str(data_path),
                "datasets": df["dataset"].value_counts().to_dict(),
                "artifact_root": str(artifact_root),
            }
        )
        return

    artifact_root.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    metric_rows = []
    row_hash_checks = []

    attribute_sets = cfg["parameters"]["attribute_sets"]
    reps = cfg["parameters"]["representations"]
    n_splits = int(cfg["parameters"]["n_splits"])
    clf_cfg = cfg["parameters"]["classifier"]

    for dataset in cfg["dataset"]["datasets"]:
        dset = df[df["dataset"] == dataset].copy()
        for task, task_cfg in cfg["parameters"]["tasks"].items():
            label_col = task_cfg["label"]
            task_df = dset.copy()
            if task == "emotion":
                task_df = task_df[
                    task_df["emotion_training_usable"].fillna(False)
                    & task_df[label_col].fillna("").ne("")
                ].copy()
            elif task == "gender":
                task_df = task_df[
                    task_df["task_eligible"].fillna(False)
                    & task_df[label_col].isin(["male", "female"])
                ].copy()
            else:
                raise ValueError(task)

            for aset, attrs in attribute_sets.items():
                needed = sorted(
                    set(feature_columns(attrs, "absolute") + feature_columns(attrs, "relative"))
                )
                sub = task_df.dropna(subset=needed + [label_col, "speaker_id"]).copy()
                for a in attrs:
                    if a == "pitch":
                        sub = sub[sub["f0_median_hz"] > 0]
                    elif a == "rate":
                        sub = sub[
                            (sub["phoneme_articulation_rate"] > 0)
                            & (sub["rate_relative_ratio"] > 0)
                        ]
                sub = sub.sort_values("sample_id").reset_index(drop=True)

                row_id_hash = hashlib.sha256(
                    "\n".join(sub["sample_id"].astype(str)).encode("utf-8")
                ).hexdigest()
                inventory_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "attribute_set": aset,
                        "n_rows": len(sub),
                        "n_speakers": sub["speaker_id"].nunique(),
                        "n_classes": sub[label_col].nunique(),
                        "row_id_sha256": row_id_hash,
                    }
                )

                for seed in cfg["seed"]:
                    folds = make_within_speaker_folds(
                        sub, label_col=label_col, n_splits=n_splits, seed=int(seed)
                    )
                    for fold in range(n_splits):
                        train = sub[folds != fold]
                        test = sub[folds == fold]
                        if train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
                            raise RuntimeError(
                                f"insufficient classes: {dataset}/{task}/{aset}/seed={seed}/fold={fold}"
                            )
                        for rep in reps:
                            cols = feature_columns(attrs, rep)
                            scores = fit_eval(
                                train,
                                test,
                                cols=cols,
                                label_col=label_col,
                                C=float(clf_cfg["C"]),
                                class_weight=clf_cfg["class_weight"],
                                max_iter=int(clf_cfg["max_iter"]),
                            )
                            metric_rows.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "attribute_set": aset,
                                    "representation": rep,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "n_rows": len(sub),
                                    "n_train": len(train),
                                    "n_test": len(test),
                                    **scores,
                                }
                            )
                            row_hash_checks.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "attribute_set": aset,
                                    "representation": rep,
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "row_id_sha256": row_id_hash,
                                }
                            )

    inventory = pd.DataFrame(inventory_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    deltas = paired_deltas(metrics, bootstrap_reps=int(cfg["parameters"]["bootstrap_reps"]))

    inventory.to_csv(artifact_root / "data_inventory.csv", index=False)
    metrics.to_csv(artifact_root / "metrics_by_fold.csv", index=False)
    summary.to_csv(artifact_root / "summary.csv", index=False)
    deltas.to_csv(artifact_root / "paired_deltas.csv", index=False)
    pd.DataFrame(row_hash_checks).to_csv(artifact_root / "row_hash_checks.csv", index=False)

    metadata = {
        "experiment_id": cfg["experiment_id"],
        "data_env": data_env,
        "data_path_observed": str(data_path),
        "data_sha256": sha256_file(data_path),
        "data_rows_loaded": int(len(df)),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": git_value(["git", "branch", "--show-current"]),
        "git_status_short": git_value(["git", "status", "--short"]),
        "python": os.sys.version,
        "config": str(config_path),
        "seeds": cfg["seed"],
        "n_splits": n_splits,
        "row_identity_control_file": "row_hash_checks.csv",
    }
    write_run_metadata(artifact_root / "run_metadata.json", metadata)
    print(summary.to_string(index=False))
    print("\nPaired deltas:")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
