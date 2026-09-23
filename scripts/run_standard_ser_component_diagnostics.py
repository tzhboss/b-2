#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def ccc_from_moments(m):
    m = np.asarray(m, dtype=float)
    my, mp, my2, mp2, myp = [m[..., i] for i in range(5)]
    vy = np.maximum(my2 - my * my, 0.0)
    vp = np.maximum(mp2 - mp * mp, 0.0)
    cov = myp - my * mp
    den = vy + vp + (my - mp) ** 2
    out = np.divide(2.0 * cov, den, out=np.full_like(den, np.nan, dtype=float), where=den > 0)
    return out


def speaker_moments(df, pred_col):
    x = df[["speaker_id", "y_true", pred_col]].rename(columns={pred_col: "pred"}).copy()
    x["y2"] = x.y_true * x.y_true
    x["p2"] = x.pred * x.pred
    x["yp"] = x.y_true * x.pred
    g = x.groupby("speaker_id", sort=True).agg(
        n=("y_true", "size"),
        my=("y_true", "mean"),
        mp=("pred", "mean"),
        my2=("y2", "mean"),
        mp2=("p2", "mean"),
        myp=("yp", "mean"),
    )
    overall = g[["my", "mp", "my2", "mp2", "myp"]].to_numpy(float)
    between = np.column_stack([
        g.my.to_numpy(float),
        g.mp.to_numpy(float),
        g.my.to_numpy(float) ** 2,
        g.mp.to_numpy(float) ** 2,
        g.my.to_numpy(float) * g.mp.to_numpy(float),
    ])
    within = np.column_stack([
        np.zeros(len(g)),
        np.zeros(len(g)),
        np.maximum(g.my2.to_numpy(float) - g.my.to_numpy(float) ** 2, 0.0),
        np.maximum(g.mp2.to_numpy(float) - g.mp.to_numpy(float) ** 2, 0.0),
        g.myp.to_numpy(float) - g.my.to_numpy(float) * g.mp.to_numpy(float),
    ])
    return list(g.index.astype(str)), {"overall": overall, "between": between, "within": within}


def point_ccc(moment_matrix):
    return float(ccc_from_moments(moment_matrix.mean(axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    src = Path(cfg["source_oof"])
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    lam = float(cfg.get("lambda_filter", 1.0))
    reps = int(cfg.get("bootstrap_reps", 5000))
    seed = int(cfg.get("bootstrap_seed", 20260923))

    d = pd.read_parquet(src)
    d = d[np.isclose(d["lambda"].astype(float), lam)].copy()
    required = {"sample_id","speaker_id","model_family","target","y_true",
                "pred_absolute","pred_relative","pred_hybrid"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    duplicate_keys = ["sample_id","model_family","target"]
    if d.duplicated(duplicate_keys).any():
        raise RuntimeError("duplicate OOF rows after lambda filter")

    representations = {
        "absolute": "pred_absolute",
        "relative": "pred_relative",
        "hybrid": "pred_hybrid",
    }
    point_rows, delta_rows, rank_rows = [], [], []
    rng = np.random.default_rng(seed)

    for (model, target), cell in d.groupby(["model_family","target"], sort=True):
        by_rep = {}
        speakers_ref = None
        for rep, pred_col in representations.items():
            speakers, moments = speaker_moments(cell, pred_col)
            if speakers_ref is None:
                speakers_ref = speakers
            elif speakers != speakers_ref:
                raise RuntimeError(f"speaker mismatch in {model}/{target}/{rep}")
            by_rep[rep] = moments

        ns = len(speakers_ref)
        counts = rng.multinomial(ns, np.full(ns, 1.0/ns), size=reps).astype(np.float64)

        boot = {}
        for rep in representations:
            boot[rep] = {}
            for metric in ["overall","between","within"]:
                moments = by_rep[rep][metric]
                pt = point_ccc(moments)
                bm = (counts @ moments) / ns
                bccc = ccc_from_moments(bm)
                boot[rep][metric] = bccc
                lo, hi = np.nanquantile(bccc, [0.025,0.975])
                point_rows.append({
                    "model_family": model, "target": target, "representation": rep,
                    "metric": metric, "ccc": pt, "ci95_low": float(lo), "ci95_high": float(hi),
                    "n_speakers": ns, "bootstrap_reps": reps,
                })

        for metric in ["overall","between","within"]:
            pts = {rep: point_ccc(by_rep[rep][metric]) for rep in representations}
            winner = max(pts, key=pts.get)
            rank_rows.append({
                "model_family": model, "target": target, "metric": metric,
                "winner": winner,
                "absolute": pts["absolute"], "relative": pts["relative"], "hybrid": pts["hybrid"],
            })
            for a,b in [("relative","absolute"),("hybrid","relative"),("hybrid","absolute")]:
                delta = boot[a][metric] - boot[b][metric]
                lo,hi = np.nanquantile(delta,[0.025,0.975])
                delta_rows.append({
                    "model_family": model, "target": target, "metric": metric,
                    "comparison": f"{a}-{b}",
                    "delta_ccc": pts[a]-pts[b],
                    "ci95_low": float(lo), "ci95_high": float(hi),
                    "n_speakers": ns, "bootstrap_reps": reps,
                })

    point = pd.DataFrame(point_rows)
    delta = pd.DataFrame(delta_rows)
    ranks = pd.DataFrame(rank_rows)
    point.to_csv(root/"component_metrics.csv", index=False)
    delta.to_csv(root/"paired_component_deltas.csv", index=False)
    ranks.to_csv(root/"rank_summary.csv", index=False)

    checks = []
    for (model,target), g in ranks.groupby(["model_family","target"]):
        win = dict(zip(g.metric,g.winner))
        checks.append({
            "model_family":model, "target":target,
            "overall_winner":win["overall"], "between_winner":win["between"], "within_winner":win["within"],
            "overall_vs_within_rank_change": bool(win["overall"] != win["within"]),
        })
    checks = pd.DataFrame(checks)
    checks.to_csv(root/"gating_checks.csv", index=False)

    metadata = {
        "experiment_id": cfg["experiment_id"],
        "source_oof": str(src),
        "lambda_filter": lam,
        "interpretation": "standard absolute-label OOF predictions; relative is diagnostic only",
        "metrics": {
            "overall": "equal-speaker-weighted pooled CCC",
            "between": "CCC across speaker mean truth/prediction",
            "within": "equal-speaker-weighted pooled CCC after centering truth/prediction within speaker",
        },
        "bootstrap_unit": "speaker",
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "rows_after_filter": int(len(d)),
        "speakers": int(d.speaker_id.nunique()),
    }
    (root/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")

    print("COMPONENT METRICS")
    print(point.to_string(index=False))
    print("\nDELTAS")
    print(delta.to_string(index=False))
    print("\nGATING")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
