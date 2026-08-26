"""
baseline_svm.py
===============
TF-IDF + Linear SVM baseline.
LinearSVC wrapped with CalibratedClassifierCV for AUROC-compatible
probability output (Platt scaling, cv=5).

No GPU required. Runtime: under a minute on full 18056-sample dataset.

Usage (from project root):
    python "CHEAT_TEST/baseline_svm.py"
"""

import os
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_cache"
)
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)

INPUT_JSONL = "CHEAT_TEST/cheat_unified.jsonl"
OUTPUT_DIR  = "CHEAT_TEST/svm_results"

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    min_df=2,
)


def load_splits(jsonl_path):
    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []
    test_sources = []
    train_ids, test_ids = set(), set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "train":
                train_texts.append(r["text"])
                train_labels.append(r["label"])
                train_ids.add(r["id"])
            else:
                test_texts.append(r["text"])
                test_labels.append(r["label"])
                test_sources.append(r["source"])
                test_ids.add(r["id"])

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"Data leakage: {len(overlap)} paper ids in both train and test. "
            f"Re-run prepare_data.py."
        )
    print(f"Leakage check passed.  train={len(train_texts)}, test={len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels, test_sources


def evaluate(clf, X_test, y_test, test_sources, out_dir):
    preds  = clf.predict(X_test)
    probs  = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    auroc = roc_auc_score(y_test, probs)

    print(f"\n=== TF-IDF + Linear SVM ===")
    print(f"  accuracy : {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall   : {rec:.4f}")
    print(f"  f1       : {f1:.4f}")
    print(f"  auroc    : {auroc:.4f}")

    sources = pd.Series(test_sources)
    labels  = pd.Series(y_test)
    probs_s = pd.Series(probs)

    print(f"\n=== AUROC per AI source vs human ===")
    human_mask = sources == "human"
    for source in ["generation", "polish", "fusion"]:
        ai_mask     = sources == source
        subset_mask = human_mask | ai_mask
        if subset_mask.sum() == 0:
            continue
        auc = roc_auc_score(labels[subset_mask], probs_s[subset_mask])
        print(f"  human vs {source:12s}: AUROC = {auc:.4f}"
              f"  (n_human={human_mask.sum()}, n_{source}={ai_mask.sum()})")

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "svm_predictions.csv")
    pd.DataFrame({
        "source":  test_sources,
        "label":   y_test,
        "pred":    preds,
        "prob_ai": probs,
    }).to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading data from {INPUT_JSONL} ...")
    train_texts, train_labels, test_texts, test_labels, test_sources = \
        load_splits(INPUT_JSONL)

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    print(f"  Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"  X_train : {X_train.shape},  X_test : {X_test.shape}")

    print("\nTraining Linear SVM (CalibratedClassifierCV, cv=5) ...")
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=42),
        cv=5,
        method="sigmoid",
    )
    svm.fit(X_train, train_labels)
    evaluate(svm, X_test, test_labels, test_sources, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()