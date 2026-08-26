"""Train LR, calibrated linear SVM and MLP on cached strict-256 sentence features.

This script performs no RoBERTa inference. It reads the final strict-256 cached
train/test feature tables, reproduces the source-stratified 80/20 meta split,
selects a threshold on validation balanced accuracy, refits each classifier on
all 12,640 training rows, and evaluates once on the held-out 5,393-row test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 42
SENTENCE_FEATURES = [
    "sent_mean",
    "sent_max",
    "sent_std",
    "sent_prop_gt_05",
    "sent_min",
    "sent_range",
    "sent_median",
    "sent_q25",
    "sent_q75",
    "sent_top2_mean",
    "sent_bottom2_mean",
    "sent_diff_mean",
    "sent_high_low_coexist",
    "n_sentences",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "improved_fusion_results_strict_256"
    default_output = script_dir / "strict_sentence_baseline_results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--validation-size", type=float, default=0.2)
    return parser.parse_args()


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = [
        balanced_accuracy_score(y_true, probabilities >= threshold)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def make_models() -> dict[str, Pipeline]:
    return {
        "LR": Pipeline(
            [
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
                        cv=5,
                        method="sigmoid",
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
    }


def source_aurocs(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    output = {}
    for source in ("generation", "polish", "fusion"):
        mask = frame["source"].isin(["human", source]).to_numpy()
        output[source] = float(
            roc_auc_score(frame.loc[mask, "label"], probabilities[mask])
        )
    return output


def main() -> None:
    args = parse_args()
    train_path = args.input_dir / "train_features.csv"
    test_path = args.input_dir / "test_features.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected strict feature files at {train_path} and {test_path}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required = {"id", "label", "source", *SENTENCE_FEATURES}
    for name, frame in (("train", train), ("test", test)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} features missing columns: {sorted(missing)}")
    if len(train) != 12640 or len(test) != 5393:
        raise ValueError(
            f"Unexpected strict split sizes: train={len(train)}, test={len(test)}"
        )

    meta_train, validation = train_test_split(
        train,
        test_size=args.validation_size,
        random_state=SEED,
        stratify=train["source"],
    )
    print(
        f"Loaded strict features: meta-train={len(meta_train)}, "
        f"validation={len(validation)}, test={len(test)}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    x_meta = meta_train[SENTENCE_FEATURES].to_numpy()
    y_meta = meta_train["label"].to_numpy()
    x_val = validation[SENTENCE_FEATURES].to_numpy()
    y_val = validation["label"].to_numpy()
    x_full = train[SENTENCE_FEATURES].to_numpy()
    y_full = train["label"].to_numpy()
    x_test = test[SENTENCE_FEATURES].to_numpy()
    y_test = test["label"].to_numpy()

    for model_name, model in make_models().items():
        print(f"\nTraining Sentence + {model_name} for validation...")
        model.fit(x_meta, y_meta)
        val_prob = model.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, val_prob)
        validation_auroc = float(roc_auc_score(y_val, val_prob))

        print(f"Refitting Sentence + {model_name} on all training rows...")
        final_model = make_models()[model_name]
        final_model.fit(x_full, y_full)
        test_prob = final_model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        metrics = calculate_metrics(y_test, test_prob, threshold)
        per_source = source_aurocs(test, test_prob)

        row = {
            "model": f"Sentence + {model_name}",
            "n_features": len(SENTENCE_FEATURES),
            "validation_auroc": validation_auroc,
            "threshold": threshold,
            **metrics,
            **{f"auc_{key}": value for key, value in per_source.items()},
        }
        summary.append(row)

        predictions = test[["id", "source", "label"]].copy()
        predictions["prob_ai"] = test_prob
        predictions["prediction"] = test_pred
        predictions.to_csv(
            args.output_dir / f"{model_name.lower()}_predictions.csv", index=False
        )
        print(
            f"{model_name}: AUROC={metrics['auroc']:.4f}, "
            f"Generation={per_source['generation']:.4f}, "
            f"Polish={per_source['polish']:.4f}, "
            f"Fusion={per_source['fusion']:.4f}, threshold={threshold:.4f}"
        )

    pd.DataFrame(summary).to_csv(args.output_dir / "results_summary.csv", index=False)
    pd.DataFrame({"feature": SENTENCE_FEATURES}).to_csv(
        args.output_dir / "features_used.csv", index=False
    )
    print(f"\nAll outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
