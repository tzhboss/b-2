from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ABS_COLUMNS = {
    "pitch": "abs_pitch_semitone",
    "loudness": "speech_lufs",
    "rate": "abs_log_rate",
}
REL_COLUMNS = {
    "pitch": "pitch_relative_st",
    "loudness": "loudness_relative_lu",
    "rate": "rel_log_rate",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_pitch_semitone"] = 12.0 * np.log2(out["f0_median_hz"].astype(float))
    out["abs_log_rate"] = np.log(out["phoneme_articulation_rate"].astype(float))
    out["rel_log_rate"] = np.log(out["rate_relative_ratio"].astype(float))
    return out


def feature_columns(attributes: Iterable[str], representation: str) -> list[str]:
    attrs = list(attributes)
    if representation == "absolute":
        return [ABS_COLUMNS[a] for a in attrs]
    if representation == "relative":
        return [REL_COLUMNS[a] for a in attrs]
    if representation == "hybrid":
        return [ABS_COLUMNS[a] for a in attrs] + [REL_COLUMNS[a] for a in attrs]
    raise ValueError(f"unknown representation: {representation}")


def make_within_speaker_folds(
    df: pd.DataFrame,
    label_col: str,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.full(len(df), -1, dtype=int)
    work = df.reset_index(drop=True)
    grouped = work.groupby(["speaker_id", label_col], dropna=False, sort=True).indices
    for _, idx in grouped.items():
        idx = np.asarray(idx, dtype=int)
        rng.shuffle(idx)
        assigned = np.arange(len(idx), dtype=int) % n_splits
        rng.shuffle(assigned)
        folds[idx] = assigned
    if (folds < 0).any():
        raise RuntimeError("unassigned fold rows")
    return folds


def fit_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    label_col: str,
    C: float,
    class_weight: str | None,
    max_iter: int,
    labels: list[str],
) -> dict[str, float]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train[cols].to_numpy(), train[label_col].to_numpy())
    pred = model.predict(test[cols].to_numpy())
    y = test[label_col].to_numpy()
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def paired_bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    inds = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[inds].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "task", "attribute_set", "representation"]
    return (
        metrics.groupby(keys, as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            n_eval=("macro_f1", "size"),
            n_rows=("n_rows", "max"),
        )
    )


def paired_deltas(metrics: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    idx = ["dataset", "task", "attribute_set", "seed", "fold"]
    pivot = metrics.pivot_table(index=idx, columns="representation", values="macro_f1")
    rows = []
    comparisons = [("relative", "absolute"), ("hybrid", "absolute"), ("hybrid", "relative")]
    for (dataset, task, attribute_set), sub in pivot.groupby(level=[0, 1, 2]):
        for a, b in comparisons:
            if a not in sub.columns or b not in sub.columns:
                continue
            delta = (sub[a] - sub[b]).dropna().to_numpy()
            stable_seed = int.from_bytes(
                f"{dataset}|{task}|{attribute_set}|{a}|{b}".encode("utf-8"), "little", signed=False
            ) % (2**32)
            lo, hi = paired_bootstrap_ci(delta, reps=bootstrap_reps, seed=stable_seed)
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "attribute_set": attribute_set,
                    "comparison": f"{a}-{b}",
                    "delta_macro_f1_mean": float(delta.mean()) if len(delta) else math.nan,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_pairs": int(len(delta)),
                }
            )
    return pd.DataFrame(rows)


def write_run_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
