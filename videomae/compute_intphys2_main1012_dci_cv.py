#!/usr/bin/env python3
"""Compute cross-validated DCI-style metrics for full IntPhys2 Main (1012)."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute CV DCI-style metrics for IntPhys2 Main 1012")
    parser.add_argument(
        "--embeddings_bundle",
        type=str,
        default="outputs/intphys2_main1012_pipeline/aggregated/intphys2_main1012_embeddings_bundle.pt",
    )
    parser.add_argument(
        "--joined_metadata_csv",
        type=str,
        default="outputs/intphys2_main1012_pipeline/aggregated/intphys2_main1012_joined.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/intphys2_main1012_pipeline/aggregated",
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def entropy_normalized(probs: np.ndarray, axis: int, log_base: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0, np.log(probs), 0.0)
    ent = -np.sum(probs * logp, axis=axis)
    if log_base <= 1:
        return np.zeros_like(ent)
    return ent / np.log(log_base)


def main() -> None:
    args = parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for DCI-CV script "
            "(LogisticRegression, StratifiedKFold, LabelEncoder)."
        ) from exc

    bundle_path = Path(args.embeddings_bundle).resolve()
    joined_path = Path(args.joined_metadata_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    emb = bundle["embeddings"]
    if torch.is_tensor(emb):
        emb = emb.detach().cpu().to(torch.float32).numpy()
    else:
        emb = np.asarray(emb, dtype=np.float32)
    bundle_ids = list(bundle["video_ids"])

    joined_rows = load_csv(joined_path)
    joined_by_id = {r["video_id"]: r for r in joined_rows}

    aligned_rows = []
    aligned_idx = []
    dropped_ids = []
    for i, vid in enumerate(bundle_ids):
        row = joined_by_id.get(vid)
        if row is None:
            dropped_ids.append(vid)
            continue
        aligned_rows.append(row)
        aligned_idx.append(i)

    x = emb[np.array(aligned_idx, dtype=np.int64)]
    n, d = x.shape

    if n == 0:
        raise RuntimeError("No aligned rows after video_id alignment.")
    if n != len(aligned_rows):
        raise RuntimeError("Alignment mismatch between embeddings and joined rows.")

    candidate_factors = [
        ("condition", "condition"),
        ("type", "type"),
        ("type_raw", "type"),
        ("occluder", "occluder"),
        ("difficulty", "difficulty"),
        ("camera", "camera"),
    ]

    used_factors = []
    skipped_factors = []
    factor_reports = {}
    importance_vectors = []

    seen_out_name = set()
    for source_col, out_name in candidate_factors:
        if out_name in seen_out_name:
            continue
        if source_col not in aligned_rows[0]:
            skipped_factors.append(
                {"factor": out_name, "source_column": source_col, "reason": "column_not_found"}
            )
            continue

        values = [
            (r.get(source_col, "").strip() if r.get(source_col, "") is not None else "")
            for r in aligned_rows
        ]
        values = [v if v != "" else "__MISSING__" for v in values]

        class_counts = Counter(values)
        if len(class_counts) < 2:
            skipped_factors.append(
                {
                    "factor": out_name,
                    "source_column": source_col,
                    "reason": "fewer_than_2_classes",
                    "class_counts": dict(class_counts),
                }
            )
            continue

        min_class_count = min(class_counts.values())
        folds = min(args.cv_folds, min_class_count)
        if folds < 2:
            skipped_factors.append(
                {
                    "factor": out_name,
                    "source_column": source_col,
                    "reason": "insufficient_samples_for_cv",
                    "class_counts": dict(class_counts),
                    "requested_folds": int(args.cv_folds),
                    "usable_folds": int(folds),
                }
            )
            continue

        le = LabelEncoder()
        y = le.fit_transform(values)

        clf_cv = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=args.seed,
        )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(clf_cv, x, y, cv=cv, scoring="accuracy")

        clf_full = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=args.seed,
        )
        clf_full.fit(x, y)
        coef = clf_full.coef_
        imp = np.mean(np.abs(coef), axis=0)

        used_factors.append(out_name)
        seen_out_name.add(out_name)
        importance_vectors.append(imp)
        factor_reports[out_name] = {
            "source_column": source_col,
            "folds_used": int(folds),
            "class_counts": dict(class_counts),
            "n_classes": int(len(class_counts)),
            "cv_mean_accuracy": float(np.mean(cv_scores)),
            "cv_std_accuracy": float(np.std(cv_scores)),
        }

    if not used_factors:
        raise RuntimeError("No factors were eligible for CV DCI computation.")

    R = np.stack(importance_vectors, axis=1).astype(np.float64)
    total_importance = float(R.sum())

    if total_importance <= 0:
        disentanglement = float("nan")
        completeness = float("nan")
    else:
        row_sums = R.sum(axis=1, keepdims=True)
        p_f_given_d = np.divide(R, row_sums, out=np.zeros_like(R), where=row_sums > 0)
        ent_d = entropy_normalized(p_f_given_d, axis=1, log_base=R.shape[1])
        disent_per_dim = 1.0 - ent_d
        w_d = row_sums[:, 0] / total_importance
        disentanglement = float(np.sum(w_d * disent_per_dim))

        col_sums = R.sum(axis=0, keepdims=True)
        p_d_given_f = np.divide(R, col_sums, out=np.zeros_like(R), where=col_sums > 0)
        ent_f = entropy_normalized(p_d_given_f, axis=0, log_base=R.shape[0])
        comp_per_factor = 1.0 - ent_f
        w_f = col_sums[0] / total_importance
        completeness = float(np.sum(w_f * comp_per_factor))

    cv_means = [factor_reports[f]["cv_mean_accuracy"] for f in used_factors]
    informativeness_cv = float(np.mean(cv_means))

    results = {
        "embedding_shape": [int(n), int(d)],
        "embeddings_rows_total": int(emb.shape[0]),
        "joined_rows_total": int(len(joined_rows)),
        "aligned_rows_used": int(len(aligned_rows)),
        "dropped_video_ids_count": int(len(dropped_ids)),
        "dropped_video_ids": dropped_ids,
        "requested_cv_folds": int(args.cv_folds),
        "factors_used": used_factors,
        "factors_skipped": skipped_factors,
        "factor_reports": factor_reports,
        "dci_scores": {
            "disentanglement_fullfit": disentanglement,
            "completeness_fullfit": completeness,
            "informativeness_cv": informativeness_cv,
        },
        "modeling_note": (
            "Informativeness uses StratifiedKFold CV accuracies; "
            "disentanglement/completeness use full-fit linear probes for importance weights."
        ),
        "limitations_note": (
            "Full IntPhys2 Main (1012) estimate; linear probes measure separability, not causal factors."
        ),
    }

    json_out = out_dir / "dci_results_cv.json"
    csv_out = out_dir / "dci_results_cv.csv"
    txt_out = out_dir / "dci_results_cv.txt"

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    flat = {
        "embedding_rows": int(n),
        "embedding_dims": int(d),
        "factors_used_count": len(used_factors),
        "factors_used": "|".join(used_factors),
        "disentanglement_fullfit": disentanglement,
        "completeness_fullfit": completeness,
        "informativeness_cv": informativeness_cv,
    }
    for factor in used_factors:
        flat[f"folds_{factor}"] = factor_reports[factor]["folds_used"]
        flat[f"cv_mean_accuracy_{factor}"] = factor_reports[factor]["cv_mean_accuracy"]
        flat[f"cv_std_accuracy_{factor}"] = factor_reports[factor]["cv_std_accuracy"]

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    with txt_out.open("w", encoding="utf-8") as f:
        f.write(f"Embedding shape: {x.shape}\n")
        f.write(f"Aligned rows used: {len(aligned_rows)} | dropped: {len(dropped_ids)}\n")
        f.write(f"Factors used ({len(used_factors)}): {used_factors}\n")
        f.write(
            "Folds used per factor: "
            + json.dumps({k: v["folds_used"] for k, v in factor_reports.items()})
            + "\n"
        )
        f.write(
            "CV mean/std accuracy per factor: "
            + json.dumps(
                {
                    k: {"mean": v["cv_mean_accuracy"], "std": v["cv_std_accuracy"]}
                    for k, v in factor_reports.items()
                }
            )
            + "\n"
        )
        f.write(f"Disentanglement (full-fit): {disentanglement:.6f}\n")
        f.write(f"Completeness (full-fit): {completeness:.6f}\n")
        f.write(f"Informativeness (CV): {informativeness_cv:.6f}\n")
        if skipped_factors:
            f.write("Skipped factors: " + json.dumps(skipped_factors) + "\n")

    print(f"Embedding shape: {x.shape}")
    print(f"Aligned rows used: {len(aligned_rows)} | dropped: {len(dropped_ids)}")
    print(f"Factors used ({len(used_factors)}): {used_factors}")
    print("Folds used per factor:", {k: v["folds_used"] for k, v in factor_reports.items()})
    print(
        "CV mean/std accuracy per factor:",
        {
            k: {"mean": v["cv_mean_accuracy"], "std": v["cv_std_accuracy"]}
            for k, v in factor_reports.items()
        },
    )
    print(f"Disentanglement (full-fit): {disentanglement:.6f}")
    print(f"Completeness (full-fit): {completeness:.6f}")
    print(f"Informativeness (CV): {informativeness_cv:.6f}")
    if skipped_factors:
        print("Skipped factors:", skipped_factors)
    print(f"Saved: {json_out}")
    print(f"Saved: {csv_out}")
    print(f"Saved: {txt_out}")


if __name__ == "__main__":
    main()
