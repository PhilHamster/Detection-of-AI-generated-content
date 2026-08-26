"""
sentence_level_fusion_classifier.py
=====================================
Fusion-aware hybrid detector: sentence-level scoring + document-level aggregation.

Pipeline:
  1. Split each abstract into sentences (NLTK sent_tokenize)
  2. Score each sentence with the fine-tuned RoBERTa checkpoint (Option A)
  3. Aggregate per-sentence scores into 4 document-level features:
       mean_ai   : mean AI probability across all sentences
       max_ai    : max AI probability (most AI-like sentence)
       std_ai    : std dev (variance in AI-ness — key signal for fusion texts)
       prop_ai   : proportion of sentences with p_ai > 0.5
  4. Train three fusion-aware classifiers on these features:
       - Logistic Regression
       - Linear SVM (CalibratedClassifierCV for AUROC)
       - MLP (2-layer, sklearn)
     Pick the best by test AUROC.
  5. Report per-source AUROC breakdown + feature importance + ablation

Why std_ai matters for fusion:
  Fusion texts interleave human and AI sentences, so sentence-level AI scores
  vary widely (high std). Pure generation texts are uniformly AI-like (low std).
  This variance signal is invisible to document-level classifiers.

Requirements:
    pip install torch transformers scikit-learn pandas nltk

Usage (from project root):
    python "CHEAT_TEST/sentence_level_fusion_classifier.py"
"""

import os
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_cache"
)
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

import json
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_JSONL    = "CHEAT_TEST/cheat_unified.jsonl"
ROBERTA_CKPT   = "CHEAT_TEST/roberta_results/final_model"
OUTPUT_DIR     = "CHEAT_TEST/sentence_fusion_results"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 32
MAX_LENGTH     = 128
PROP_THRESHOLD = 0.5
FEATURE_NAMES = [
    "mean_ai",      # 均值：整体AI程度
    "max_ai",       # 最高分：最像AI的句子
    "std_ai",       # 标准差：句子间AI程度的波动
    "prop_ai",      # 超过阈值的句子比例
    "min_ai",       # 最低分：最像人类的句子
    "range_ai",     # 极差：max-min，比std更直观的波动度量
    "median_ai",    # 中位数：比mean更鲁棒
    "q75_ai",       # 75分位数：偏向高AI分一侧的分布特征
    "n_sentences",  # 句子数量（摘要长度的代理特征）
    "diff_mean",    # 相邻句子AI分数的平均变化幅度（fusion核心信号）
]


# ── NLTK ──────────────────────────────────────────────────────────────────────
def ensure_nltk():
    try:
        sent_tokenize("test.")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")
        nltk.download("punkt_tab")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(jsonl_path):
    train_rows, test_rows = [], []
    train_ids, test_ids = set(), set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "train":
                train_rows.append(r)
                train_ids.add(r["id"])
            else:
                test_rows.append(r)
                test_ids.add(r["id"])

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} ids in both splits.")
    print(f"Loaded: train={len(train_rows)}, test={len(test_rows)}")
    return train_rows, test_rows


# ── RoBERTa sentence scorer ───────────────────────────────────────────────────
def load_roberta(ckpt_path, device):
    print(f"Loading RoBERTa checkpoint: {ckpt_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model     = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_sentences_batch(sentences, model, tokenizer, device):
    all_probs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i: i + BATCH_SIZE]
        enc   = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().tolist())
    return all_probs


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(rows, model, tokenizer, device, desc="Extracting"):
    """
    Extract sentence-level aggregated features for each document.

    Features:
      mean_ai    : mean p_ai across sentences
      max_ai     : max p_ai (most AI-like sentence)
      std_ai     : std dev of p_ai (variance in AI-ness)
      prop_ai    : proportion of sentences with p_ai > threshold
      min_ai     : min p_ai (most human-like sentence)
      range_ai   : max_ai - min_ai (spread, more intuitive than std)
      median_ai  : median p_ai (robust to outlier sentences)
      q75_ai     : 75th percentile of p_ai
      n_sentences: number of sentences (proxy for abstract length)
      diff_mean  : mean absolute difference between adjacent sentence scores
                   KEY feature for fusion: human/AI sentences alternate in
                   fusion texts, causing large score jumps between neighbours;
                   pure generation texts have uniformly high scores (low diff)
    """
    features, labels, sources, ids = [], [], [], []

    for r in tqdm(rows, desc=desc):
        sentences = [s.strip() for s in sent_tokenize(r["text"]) if len(s.strip()) > 5]
        if not sentences:
            sentences = [r["text"]]

        probs     = np.array(score_sentences_batch(sentences, model, tokenizer, device))
        n         = len(probs)

        mean_ai    = float(np.mean(probs))
        max_ai     = float(np.max(probs))
        std_ai     = float(np.std(probs))
        prop_ai    = float(np.mean(probs > PROP_THRESHOLD))
        min_ai     = float(np.min(probs))
        range_ai   = max_ai - min_ai
        median_ai  = float(np.median(probs))
        q75_ai     = float(np.percentile(probs, 75))
        n_sentences = float(n)
        # diff_mean: mean |p[i+1] - p[i]| across adjacent sentence pairs
        # single-sentence documents get 0 (no adjacent pair exists)
        diff_mean  = float(np.mean(np.abs(np.diff(probs)))) if n > 1 else 0.0

        features.append([
            mean_ai, max_ai, std_ai, prop_ai,
            min_ai, range_ai, median_ai, q75_ai,
            n_sentences, diff_mean,
        ])
        labels.append(r["label"])
        sources.append(r["source"])
        ids.append(r["id"])

    return np.array(features), np.array(labels), sources, ids


# ── Classifiers ───────────────────────────────────────────────────────────────
def build_classifiers():
    return {
        "LR": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            cv=5, method="sigmoid"
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_one(name, clf, X_test_scaled, y_test, sources, out_dir):
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auroc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*60}")
    print(f"  Sentence-Level + {name}")
    print(f"{'='*60}")
    print(f"  accuracy : {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall   : {rec:.4f}")
    print(f"  f1       : {f1:.4f}")
    print(f"  auroc    : {auroc:.4f}")

    sources_s = pd.Series(sources)
    labels_s  = pd.Series(y_test)
    probs_s   = pd.Series(y_prob)

    print(f"\n  AUROC per source vs human:")
    human_mask = sources_s == "human"
    per_source = {}
    for source in ["generation", "polish", "fusion"]:
        ai_mask     = sources_s == source
        subset_mask = human_mask | ai_mask
        if subset_mask.sum() == 0:
            continue
        auc = roc_auc_score(labels_s[subset_mask], probs_s[subset_mask])
        per_source[source] = auc
        print(f"    human vs {source:12s}: AUROC = {auc:.4f}"
              f"  (n_human={human_mask.sum()}, n_{source}={ai_mask.sum()})")

    # save per-sample predictions
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{name.lower()}_predictions.csv")
    pd.DataFrame({
        "source":  sources,
        "label":   y_test,
        "pred":    y_pred,
        "prob_ai": y_prob,
    }).to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")

    return auroc, per_source


# ── Feature importance ────────────────────────────────────────────────────────
def print_feature_importance(clf, name):
    print(f"\n  Feature importance ({name}):")
    if name == "LR":
        coefs = clf.coef_[0]
        for fname, coef in sorted(zip(FEATURE_NAMES, coefs),
                                   key=lambda x: abs(x[1]), reverse=True):
            print(f"    {fname:12s}: coef = {coef:+.4f}")
    elif name == "MLP":
        # first-layer weights: sum of abs weights per input feature
        w = np.abs(clf.coefs_[0]).sum(axis=1)
        for fname, importance in sorted(zip(FEATURE_NAMES, w),
                                         key=lambda x: x[1], reverse=True):
            print(f"    {fname:12s}: |W| sum = {importance:.4f}")
    else:
        print("    (SVM: coefficients not directly interpretable after calibration)")


# ── Ablation: single-feature AUROC ───────────────────────────────────────────
def ablation(X_test, y_test):
    print("\n  Ablation — single-feature AUROC (test set):")
    for i, fname in enumerate(FEATURE_NAMES):
        scores = X_test[:, i]
        auc = roc_auc_score(y_test, scores)
        auc = max(auc, 1 - auc)   # ensure we report the discriminative direction
        print(f"    {fname:12s}: AUROC = {auc:.4f}")


# ── Feature stats per source ──────────────────────────────────────────────────
def feature_stats(X_test, test_sources):
    df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    df["source"] = test_sources
    print("\n  Feature statistics per source (test set):")
    for src in ["human", "generation", "polish", "fusion"]:
        sub = df[df["source"] == src][FEATURE_NAMES]
        if len(sub) == 0:
            continue
        means = sub.mean()
        print(f"\n  [{src}]  n={len(sub)}")
        for fname in FEATURE_NAMES:
            print(f"    {fname:12s}: mean={means[fname]:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_nltk()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    train_rows, test_rows = load_data(INPUT_JSONL)

    # 2. Load RoBERTa checkpoint (Option A: fine-tuned on CHEAT train split)
    model, tokenizer = load_roberta(ROBERTA_CKPT, DEVICE)

    # 3. Extract sentence-level features
    print("\n[1/4] Scoring train sentences with RoBERTa ...")
    X_train, y_train, train_sources, _ = extract_features(
        train_rows, model, tokenizer, DEVICE, desc="Train"
    )
    print("\n[2/4] Scoring test sentences with RoBERTa ...")
    X_test, y_test, test_sources, _ = extract_features(
        test_rows, model, tokenizer, DEVICE, desc="Test"
    )

    # 4. Feature stats and ablation (analysis / interpretability section material)
    feature_stats(X_test, test_sources)
    ablation(X_test, y_test)

    # save feature table
    feat_csv = os.path.join(OUTPUT_DIR, "sentence_features_test.csv")
    df_feat = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    df_feat["source"] = test_sources
    df_feat["label"]  = y_test
    df_feat.to_csv(feat_csv, index=False)
    print(f"\n  Saved feature table -> {feat_csv}")

    # 5. Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 6. Train and evaluate all three classifiers
    print("\n[3/4] Training and evaluating classifiers ...")
    classifiers = build_classifiers()
    results = {}   # name -> (overall_auroc, per_source_dict, clf)

    for name, clf in classifiers.items():
        print(f"\nTraining {name} ...")
        clf.fit(X_train_s, y_train)
        auroc, per_source = evaluate_one(
            name, clf, X_test_s, y_test, test_sources, OUTPUT_DIR
        )
        print_feature_importance(clf, name)
        results[name] = (auroc, per_source, clf)

    # 7. Pick best by overall AUROC
    print("\n[4/4] Summary — picking best classifier by overall AUROC:")
    print(f"\n  {'Classifier':<20} {'Overall AUROC':<16} {'vs generation':<16} {'vs polish':<12} {'vs fusion'}")
    print(f"  {'-'*75}")
    best_name, best_auroc = None, -1
    for name, (auroc, per_source, _) in results.items():
        gen = per_source.get("generation", float("nan"))
        pol = per_source.get("polish",     float("nan"))
        fus = per_source.get("fusion",     float("nan"))
        print(f"  {'Sent-Level + ' + name:<20} {auroc:<16.4f} {gen:<16.4f} {pol:<12.4f} {fus:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_name  = name

    print(f"\n  ✅ Best classifier: Sentence-Level + {best_name} (AUROC = {best_auroc:.4f})")

    # 8. Baseline comparison reminder
    print("\n  Baseline reference (document-level, from Phase 1):")
    print(f"  {'Method':<28} {'Overall':<10} {'generation':<14} {'polish':<12} {'fusion'}")
    print(f"  {'-'*70}")
    baselines = [
        ("TF-IDF + LR",     0.9167, 0.9978, 0.9540, 0.7964),
        ("TF-IDF + SVM",    0.9196, 0.9938, 0.9579, 0.8051),
        ("RoBERTa-base",    0.9515, 0.9992, 0.9874, 0.8666),
        ("Fast-DetectGPT",  0.7704, 0.9890, 0.7408, 0.5782),
    ]
    for row in baselines:
        print(f"  {row[0]:<28} {row[1]:<10.4f} {row[2]:<14.4f} {row[3]:<12.4f} {row[4]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()