"""
Improved fusion-aware detector
==============================

Adds the missing document-level RoBERTa probability to sentence-distribution
features, uses a validation split for model selection, and evaluates the test
set only after the best fusion model has been selected.

Run from the repository root after strict RoBERTa training:
    python improved_fusion_stacking.py
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


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
DOC_FEATURES = ["doc_ai_prob"]
ALL_FEATURES = DOC_FEATURES + SENTENCE_FEATURES
HETEROGENEITY_FEATURES = [
    "sent_std",
    "sent_range",
    "sent_diff_mean",
    "sent_high_low_coexist",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="cheat_unified.jsonl",
        help="Unified JSONL containing split, id, text, label and source.",
    )
    parser.add_argument(
        "--checkpoint",
        default="roberta_results_strict/final_model",
        help="Fine-tuned RoBERTa checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="improved_fusion_results_strict_256",
        help="Output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--doc-max-length", type=int, default=256)
    parser.add_argument("--sent-max-length", type=int, default=128)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore cached features and score all texts again.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_nltk():
    try:
        sent_tokenize("A sentence.")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            required = {"id", "text", "label", "source", "split"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"Missing fields {missing} in row {len(rows)}")
            rows.append(row)

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not test_rows:
        # Preserve compatibility with datasets where every non-train row is test.
        test_rows = [r for r in rows if r["split"] != "train"]

    train_ids = {str(r["id"]) for r in train_rows}
    test_ids = {str(r["id"]) for r in test_rows}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} IDs occur in both splits.")
    print(f"Loaded train={len(train_rows)}, test={len(test_rows)}")
    return train_rows, test_rows


def load_detector(checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict_ai_prob(texts, model, tokenizer, device, batch_size, max_length):
    probabilities = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        probabilities.extend(
            torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
        )
    return np.asarray(probabilities, dtype=np.float32)


def sentence_summary(probs):
    probs = np.asarray(probs, dtype=np.float32)
    count = len(probs)
    sorted_probs = np.sort(probs)
    k = min(2, count)
    return {
        "sent_mean": float(np.mean(probs)),
        "sent_max": float(np.max(probs)),
        "sent_std": float(np.std(probs)),
        "sent_prop_gt_05": float(np.mean(probs > 0.5)),
        "sent_min": float(np.min(probs)),
        "sent_range": float(np.max(probs) - np.min(probs)),
        "sent_median": float(np.median(probs)),
        "sent_q25": float(np.percentile(probs, 25)),
        "sent_q75": float(np.percentile(probs, 75)),
        "sent_top2_mean": float(np.mean(sorted_probs[-k:])),
        "sent_bottom2_mean": float(np.mean(sorted_probs[:k])),
        "sent_diff_mean": (
            float(np.mean(np.abs(np.diff(probs)))) if count > 1 else 0.0
        ),
        # Captures documents that contain clearly human-like and AI-like sentences.
        "sent_high_low_coexist": float(
            np.any(probs >= 0.8) and np.any(probs <= 0.2)
        ),
        "n_sentences": float(count),
    }


def extract_features(
    rows,
    model,
    tokenizer,
    device,
    batch_size,
    doc_max_length,
    sent_max_length,
    description,
):
    texts = [r["text"] for r in rows]
    print(f"\nScoring {description} documents...")
    doc_probs = predict_ai_prob(
        texts, model, tokenizer, device, batch_size, doc_max_length
    )

    records = []
    for row, doc_prob in tqdm(
        zip(rows, doc_probs), total=len(rows), desc=f"{description} sentences"
    ):
        sentences = [
            s.strip() for s in sent_tokenize(row["text"]) if len(s.strip()) > 5
        ]
        if not sentences:
            sentences = [row["text"]]
        sent_probs = predict_ai_prob(
            sentences, model, tokenizer, device, batch_size, sent_max_length
        )
        record = {
            "id": str(row["id"]),
            "label": int(row["label"]),
            "source": row["source"],
            "doc_ai_prob": float(doc_prob),
        }
        record.update(sentence_summary(sent_probs))
        records.append(record)
    return pd.DataFrame(records)


def load_or_extract_features(
    cache_path,
    rows,
    model,
    tokenizer,
    device,
    args,
    description,
):
    expected_ids = [str(r["id"]) for r in rows]
    if cache_path.exists() and not args.force_features:
        cached = pd.read_csv(cache_path, dtype={"id": str})
        if (
            list(cached["id"]) == expected_ids
            and all(c in cached.columns for c in ALL_FEATURES)
        ):
            print(f"Using cached features: {cache_path}")
            return cached
        print(f"Cache does not match current data; rebuilding {cache_path}")

    frame = extract_features(
        rows,
        model,
        tokenizer,
        device,
        args.batch_size,
        args.doc_max_length,
        args.sent_max_length,
        description,
    )
    frame.to_csv(cache_path, index=False)
    print(f"Saved features: {cache_path}")
    return frame


def make_models():
    return {
        "LR": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=SEED,
        ),
    }


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predictions),
        "auroc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def source_aurocs(frame, probabilities):
    result = {}
    sources = frame["source"].to_numpy()
    labels = frame["label"].to_numpy()
    for source in ["generation", "polish", "fusion"]:
        mask = np.isin(sources, ["human", source])
        if mask.sum() and len(np.unique(labels[mask])) == 2:
            result[source] = roc_auc_score(labels[mask], probabilities[mask])
    return result


def choose_threshold(y_true, probabilities):
    # Threshold is selected on validation by maximum balanced accuracy.
    candidates = np.unique(
        np.concatenate(([0.0], probabilities, [1.0]))
    )
    scores = [
        balanced_accuracy_score(y_true, probabilities >= threshold)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def bootstrap_auc_difference(
    y_true, proposed_probs, baseline_probs, n_bootstrap=2000
):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    differences = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        differences.append(
            roc_auc_score(y_true[indices], proposed_probs[indices])
            - roc_auc_score(y_true[indices], baseline_probs[indices])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(np.mean(differences)), float(low), float(high)


def fit_and_validate(train_frame, val_frame, feature_names):
    x_train = train_frame[feature_names].to_numpy()
    y_train = train_frame["label"].to_numpy()
    x_val = val_frame[feature_names].to_numpy()
    y_val = val_frame["label"].to_numpy()

    candidates = []
    for name, model in make_models().items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)[:, 1]
        metrics = calculate_metrics(y_val, probabilities)
        per_source = source_aurocs(val_frame, probabilities)
        candidates.append((metrics["auroc"], name, model, probabilities, per_source))
        print(
            f"  {name:<21} validation AUROC={metrics['auroc']:.4f}, "
            f"fusion={per_source.get('fusion', float('nan')):.4f}"
        )
    return max(candidates, key=lambda item: item[0])


def print_test_result(name, metrics, per_source, threshold):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"AUROC            : {metrics['auroc']:.4f}")
    print(f"PR-AUC           : {metrics['pr_auc']:.4f}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1               : {metrics['f1']:.4f}")
    print(f"MCC              : {metrics['mcc']:.4f}")
    print(f"Threshold        : {threshold:.4f}")
    for source, auc in per_source.items():
        print(f"human vs {source:<10}: {auc:.4f}")


def main():
    args = parse_args()
    set_seed(SEED)
    ensure_nltk()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rows, test_rows = load_rows(args.data)
    model, tokenizer = load_detector(args.checkpoint, device)
    train_features = load_or_extract_features(
        output_dir / "train_features.csv",
        train_rows,
        model,
        tokenizer,
        device,
        args,
        "train",
    )
    test_features = load_or_extract_features(
        output_dir / "test_features.csv",
        test_rows,
        model,
        tokenizer,
        device,
        args,
        "test",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stratifying by source preserves all four source types in validation.
    meta_train, validation = train_test_split(
        train_features,
        test_size=args.validation_size,
        random_state=SEED,
        stratify=train_features["source"],
    )
    print(
        f"\nMeta-classifier split: train={len(meta_train)}, "
        f"validation={len(validation)}, test={len(test_features)}"
    )

    experiments = {
        "Document RoBERTa baseline": DOC_FEATURES,
        "Sentence features only": SENTENCE_FEATURES,
        "Proposed document + sentence fusion": ALL_FEATURES,
        "Fusion without heterogeneity features": [
            f for f in ALL_FEATURES if f not in HETEROGENEITY_FEATURES
        ],
    }

    selections = {}
    print("\nSelecting each experiment's classifier using validation AUROC:")
    for experiment, features in experiments.items():
        print(f"\n{experiment} ({len(features)} features)")
        selections[experiment] = (
            features,
            fit_and_validate(meta_train, validation, features),
        )

    # The final proposed classifier is selected only from validation results.
    proposed_name = "Proposed document + sentence fusion"
    proposed_features, proposed_selection = selections[proposed_name]
    _, classifier_name, _, validation_probs, _ = proposed_selection
    validation_threshold = choose_threshold(
        validation["label"].to_numpy(), validation_probs
    )
    print(
        f"\nSelected proposed model: {classifier_name}; "
        f"validation threshold={validation_threshold:.4f}"
    )

    # Refit the preselected algorithm on all available training meta-features.
    final_model = make_models()[classifier_name]
    final_model.fit(
        train_features[proposed_features].to_numpy(),
        train_features["label"].to_numpy(),
    )
    proposed_test_probs = final_model.predict_proba(
        test_features[proposed_features].to_numpy()
    )[:, 1]
    proposed_metrics = calculate_metrics(
        test_features["label"].to_numpy(),
        proposed_test_probs,
        validation_threshold,
    )
    proposed_sources = source_aurocs(test_features, proposed_test_probs)
    print_test_result(
        "FINAL PROPOSED FUSION MODEL",
        proposed_metrics,
        proposed_sources,
        validation_threshold,
    )

    # Document probability is the original RoBERTa baseline and needs no fitting.
    baseline_probs = test_features["doc_ai_prob"].to_numpy()
    baseline_val_probs = validation["doc_ai_prob"].to_numpy()
    baseline_threshold = choose_threshold(
        validation["label"].to_numpy(), baseline_val_probs
    )
    baseline_metrics = calculate_metrics(
        test_features["label"].to_numpy(), baseline_probs, baseline_threshold
    )
    baseline_sources = source_aurocs(test_features, baseline_probs)
    print_test_result(
        "DOCUMENT-LEVEL ROBERTA BASELINE",
        baseline_metrics,
        baseline_sources,
        baseline_threshold,
    )

    # Evaluate the two predefined ablations using their validation-selected models.
    summary_rows = [
        {
            "experiment": proposed_name,
            "classifier": classifier_name,
            **proposed_metrics,
            **{f"auc_{k}": v for k, v in proposed_sources.items()},
        },
        {
            "experiment": "Document RoBERTa baseline",
            "classifier": "checkpoint probability",
            **baseline_metrics,
            **{f"auc_{k}": v for k, v in baseline_sources.items()},
        },
    ]
    for experiment in [
        "Sentence features only",
        "Fusion without heterogeneity features",
    ]:
        features, selection = selections[experiment]
        _, selected_name, _, val_probs, _ = selection
        threshold = choose_threshold(validation["label"].to_numpy(), val_probs)
        fitted = make_models()[selected_name]
        fitted.fit(
            train_features[features].to_numpy(),
            train_features["label"].to_numpy(),
        )
        probabilities = fitted.predict_proba(
            test_features[features].to_numpy()
        )[:, 1]
        metrics = calculate_metrics(
            test_features["label"].to_numpy(), probabilities, threshold
        )
        per_source = source_aurocs(test_features, probabilities)
        print_test_result(experiment.upper(), metrics, per_source, threshold)
        summary_rows.append(
            {
                "experiment": experiment,
                "classifier": selected_name,
                **metrics,
                **{f"auc_{k}": v for k, v in per_source.items()},
            }
        )

    mean_diff, ci_low, ci_high = bootstrap_auc_difference(
        test_features["label"].to_numpy(), proposed_test_probs, baseline_probs
    )
    print(
        "\nProposed minus document-baseline AUROC difference "
        f"(bootstrap 95% CI): {mean_diff:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]"
    )

    predictions = test_features[["id", "source", "label"]].copy()
    predictions["doc_roberta_prob"] = baseline_probs
    predictions["proposed_fusion_prob"] = proposed_test_probs
    predictions["proposed_prediction"] = (
        proposed_test_probs >= validation_threshold
    ).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "results_summary.csv", index=False)
    with open(output_dir / "final_fusion_model.pkl", "wb") as handle:
        pickle.dump(
            {
                "model": final_model,
                "features": proposed_features,
                "threshold": validation_threshold,
                "classifier": classifier_name,
            },
            handle,
        )

    print(f"\nAll outputs saved to: {output_dir.resolve()}")
    print("Important methodological note:")
    print(
        "The supplied RoBERTa checkpoint was trained on the original full training "
        "split. For the strongest thesis protocol, later generate out-of-fold "
        "RoBERTa predictions for meta-classifier training. The held-out test "
        "evaluation here is still untouched."
    )


if __name__ == "__main__":
    main()
