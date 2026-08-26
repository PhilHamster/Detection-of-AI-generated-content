"""Complete final non-neural baselines, Fast-DetectGPT audit, and core CIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 42
N_BOOTSTRAP = 2000
HERE = Path(__file__).resolve().parent
DATA = HERE / "cheat_unified.jsonl"
STRICT = HERE / "improved_fusion_results_strict_256"
FAST = HERE / "fastdetectgpt_scores_test.csv"
OUTPUT = HERE / "final_protocol_completion"

TFIDF_PARAMS = dict(
    ngram_range=(1, 2), max_features=50000, sublinear_tf=True,
    strip_accents="unicode", analyzer="word", min_df=2,
)


def load_data():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    train = frame[frame.split == "train"].reset_index(drop=True)
    test = frame[frame.split == "test"].reset_index(drop=True)
    if (len(train), len(test)) != (12640, 5393):
        raise ValueError(f"Unexpected split sizes: {len(train)}, {len(test)}")
    if set(train.id) & set(test.id):
        raise ValueError("Paper-ID leakage between train and test")
    return train, test


def choose_threshold(y, prob):
    candidates = np.unique(np.r_[0.0, prob, 1.0])
    scores = [balanced_accuracy_score(y, prob >= t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return dict(
        auroc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob),
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
    )


def source_aucs(frame, prob):
    result = {}
    for src in ("generation", "polish", "fusion"):
        mask = frame.source.isin(["human", src]).to_numpy()
        result[f"auc_{src}"] = roc_auc_score(frame.loc[mask, "label"], prob[mask])
    return result


def make_model(name):
    if name == "TF-IDF + LR":
        return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1, random_state=SEED)
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=SEED), cv=5, method="sigmoid"
    )


def run_tfidf(train, test):
    meta, val = train_test_split(
        train, test_size=0.2, random_state=SEED, stratify=train.source
    )
    rows = []
    for name in ("TF-IDF + LR", "TF-IDF + SVM"):
        print(f"\n{name}: validation fit")
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        x_meta = vectorizer.fit_transform(meta.text)
        x_val = vectorizer.transform(val.text)
        model = make_model(name)
        model.fit(x_meta, meta.label)
        val_prob = model.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(val.label.to_numpy(), val_prob)
        val_auc = roc_auc_score(val.label, val_prob)

        print(f"{name}: full refit and held-out test")
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        x_train = vectorizer.fit_transform(train.text)
        x_test = vectorizer.transform(test.text)
        model = make_model(name)
        model.fit(x_train, train.label)
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= threshold).astype(int)
        row = dict(model=name, validation_auroc=val_auc, threshold=threshold)
        row.update(metrics(test.label.to_numpy(), prob, threshold))
        row.update(source_aucs(test, prob))
        rows.append(row)
        out = test[["id", "source", "label"]].copy()
        out["prob_ai"] = prob
        out["prediction"] = pred
        slug = "tfidf_lr" if name.endswith("LR") else "tfidf_svm"
        out.to_csv(OUTPUT / f"{slug}_predictions.csv", index=False)
        print(f"{name}: AUROC={row['auroc']:.4f}, Fusion={row['auc_fusion']:.4f}")
    pd.DataFrame(rows).to_csv(OUTPUT / "tfidf_results_summary.csv", index=False)


def audit_fast(test):
    fast = pd.read_csv(FAST)
    if len(fast) != 5393:
        raise ValueError(f"Fast-DetectGPT row count is {len(fast)}, expected 5393")
    for frame in (test, fast):
        frame["record_key"] = frame.id.astype(str) + "::" + frame.source.astype(str)
    if fast.record_key.duplicated().any() or test.record_key.duplicated().any():
        raise ValueError("Duplicate id/source composite key")
    merged = test[["record_key", "id", "source", "label"]].merge(
        fast[["record_key", "label", "split", "fastdetect_score"]],
        on="record_key", suffixes=("_strict", "_fast"), validate="one_to_one"
    )
    ids_match = len(merged) == len(test)
    labels_match = bool((merged.label_strict == merged.label_fast).all())
    split_match = bool((merged.split == "test").all())
    finite = np.isfinite(merged.fastdetect_score.to_numpy())
    if not (ids_match and labels_match and split_match and finite.all()):
        raise ValueError("Fast-DetectGPT scores do not exactly match strict test records")
    y = merged.label_strict.to_numpy()
    prob = merged.fastdetect_score.to_numpy()
    row = dict(
        model="Fast-DetectGPT-style", rows=len(merged), ids_match=ids_match,
        labels_match=labels_match, all_split_test=split_match,
        finite_scores=int(finite.sum()), auroc=roc_auc_score(y, prob),
        pr_auc=average_precision_score(y, prob), **source_aucs(
            merged.rename(columns={"label_strict": "label"}), prob
        )
    )
    pd.DataFrame([row]).to_csv(OUTPUT / "fastdetectgpt_audit_and_results.csv", index=False)
    merged.to_csv(OUTPUT / "fastdetectgpt_verified_test_scores.csv", index=False)
    print(f"Fast-DetectGPT audit passed: 5393/5393 exact records, AUROC={row['auroc']:.4f}")


def paired_bootstrap(frame, scope):
    y = frame.label.to_numpy()
    proposed = frame.proposed_fusion_prob.to_numpy()
    baseline = frame.doc_roberta_prob.to_numpy()
    point = roc_auc_score(y, proposed) - roc_auc_score(y, baseline)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(N_BOOTSTRAP):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        differences.append(
            roc_auc_score(y[ix], proposed[ix]) - roc_auc_score(y[ix], baseline[ix])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return dict(
        scope=scope, n=len(frame), bootstrap_replicates=len(differences), seed=SEED,
        proposed_auroc=roc_auc_score(y, proposed), baseline_auroc=roc_auc_score(y, baseline),
        point_difference=point, bootstrap_mean=np.mean(differences),
        ci_95_low=low, ci_95_high=high,
        probability_difference_le_zero=np.mean(np.asarray(differences) <= 0),
    )


def export_core_bootstrap():
    pred = pd.read_csv(STRICT / "test_predictions.csv")
    rows = [paired_bootstrap(pred, "overall")]
    fusion = pred[pred.source.isin(["human", "fusion"])].reset_index(drop=True)
    rows.append(paired_bootstrap(fusion, "human_vs_fusion"))
    pd.DataFrame(rows).to_csv(OUTPUT / "hybrid_vs_roberta_bootstrap_ci.csv", index=False)
    for row in rows:
        print(
            f"{row['scope']}: delta={row['point_difference']:+.4f}, "
            f"95% CI [{row['ci_95_low']:+.4f}, {row['ci_95_high']:+.4f}]"
        )


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    train, test = load_data()
    print(f"Final paper-level split verified: train={len(train)}, test={len(test)}")
    run_tfidf(train, test)
    audit_fast(test)
    export_core_bootstrap()
    print(f"\nAll completion outputs saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
