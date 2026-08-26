"""
baseline_fast_detect_gpt_test_only.py
======================================
Zero-shot Fast-DetectGPT-style evaluation on the CHEAT dataset.

Key change from the earlier full-scoring script:
    - By default, this script evaluates ONLY split == "test".
    - Fast-DetectGPT is a zero-shot detector, so it does not need to score the
      training split. This makes the comparison with RoBERTa / SVM / LR fairer
      and much faster.

Expected input schema from prepare_data.py:
    id, title, text, label, source, split

Default input:
    CHEAT_TEST/cheat_unified.jsonl

Default output:
    CHEAT_TEST/fastdetectgpt_scores_test.csv

Example usage from project root:
    python CHEAT_TEST/baseline_fast_detect_gpt_test_only.py

Explicit usage:
    python CHEAT_TEST/baseline_fast_detect_gpt_test_only.py \
        --data_path CHEAT_TEST/cheat_unified.jsonl \
        --eval_split test \
        --model_name gpt2-xl

Notes:
    - This script uses gpt2-xl as both scoring model and sampling/reference model
      by default, matching your local GPU constraint setup.
    - On RTX 3050 Ti Laptop GPU, gpt2-xl may still be slow. For a quick test, use:
          --model_name gpt2 --max_samples 20
"""

import argparse
import csv
import json
import math
import os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl_records(path: str, eval_split: str = "test", max_samples: int | None = None):
    """Load CHEAT unified jsonl and filter by split.

    eval_split:
        "test"  -> evaluate only held-out test split, recommended/default
        "train" -> evaluate train split only, normally unnecessary for zero-shot
        "all"   -> evaluate all rows, slow and usually not needed
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n_before = len(records)
    if eval_split != "all":
        records = [r for r in records if r.get("split") == eval_split]

    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    print(f"Loaded {len(records)} records from {path}")
    print(f"Evaluation split: {eval_split}  (before filtering: {n_before})")

    counts = Counter(r.get("source", "unknown") for r in records)
    print("Source distribution:")
    for src, cnt in sorted(counts.items()):
        print(f"  {src:12s}: {cnt}")

    return records


def load_model_and_tokenizer(model_name: str, device: str, cache_dir: str | None = None):
    print(f"Loading model: {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def fast_detect_gpt_score(text: str, model, tokenizer, device: str, max_length: int = 512) -> float:
    """Compute a Fast-DetectGPT-style analytic sampling-discrepancy score.

    This implementation uses the same model as scoring and reference/sampling model.
    For each next-token prediction position:
        observed = log p_score(actual token)
        expected = E_{x~p_ref}[log p_score(x)]
        variance = Var_{x~p_ref}[log p_score(x)]
    The final score is:
        (sum(observed) - sum(expected)) / sqrt(sum(variance))

    Higher scores are treated as more AI-like in the AUROC calculation.
    """
    if not isinstance(text, str) or not text.strip():
        return float("nan")

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Need at least two tokens for next-token scoring
    if input_ids.shape[1] < 2:
        return float("nan")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]          # predict next token
    labels = input_ids[:, 1:]                  # actual next token

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    observed = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    expected = (probs * log_probs).sum(dim=-1)
    expected_sq = (probs * (log_probs ** 2)).sum(dim=-1)
    var = expected_sq - expected ** 2

    numerator = (observed - expected).sum()
    denominator = torch.sqrt(var.sum().clamp_min(1e-12))
    score = numerator / denominator

    return float(score.detach().cpu().item())


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def write_scores_csv(records, scores, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "id", "title", "text", "label", "source", "split", "fastdetect_score"
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r, s in zip(records, scores):
            row = {
                "id": r.get("id", ""),
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "label": r.get("label", ""),
                "source": r.get("source", ""),
                "split": r.get("split", ""),
                "fastdetect_score": s,
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Fast-DetectGPT-style zero-shot evaluation on CHEAT. Default: test split only."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="cheat_unified.jsonl",
        help="Path to cheat_unified.jsonl produced by prepare_data.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save per-sample scores. Default depends on eval_split.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Which split to evaluate. Default is 'test'. Use 'all' only if you really want full scoring.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-xl",
        help="Causal LM used for Fast-DetectGPT-style scoring. Use gpt2 for quick debugging.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="model_cache",
        help="Model cache directory.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length. Lower values are faster but may truncate abstracts.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional debug limit, e.g. --max_samples 20.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu. Default uses cuda if available.",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"fastdetectgpt_scores_{args.eval_split}.csv"

    records = load_jsonl_records(
        args.data_path,
        eval_split=args.eval_split,
        max_samples=args.max_samples,
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    print(
        f"Scoring {len(records)} samples on {args.device} "
        f"with model={args.model_name}, split={args.eval_split} ..."
    )

    scores = []
    for r in tqdm(records):
        score = fast_detect_gpt_score(
            r.get("text", ""),
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_length=args.max_length,
        )
        scores.append(score)

    write_scores_csv(records, scores, args.output_path)
    print(f"Saved per-sample scores to {args.output_path}")

    labels = [int(r["label"]) for r in records]
    overall = safe_auroc(labels, scores)

    print("\n=== AUROC (human=0 vs AI=1), score as-is ===")
    print(f"Overall: {overall:.4f}")

    print("\n=== AUROC per AI source vs human (binary subset) ===")
    sources = sorted(set(r.get("source") for r in records if r.get("source") != "human"))
    for src in sources:
        subset_labels = []
        subset_scores = []
        n_human = 0
        n_src = 0
        for r, s in zip(records, scores):
            if r.get("source") == "human":
                subset_labels.append(0)
                subset_scores.append(s)
                n_human += 1
            elif r.get("source") == src:
                subset_labels.append(1)
                subset_scores.append(s)
                n_src += 1

        auc = safe_auroc(subset_labels, subset_scores)
        print(f"  human vs {src:12s}: AUROC = {auc:.4f}  (n_human={n_human}, n_{src}={n_src})")

    print("\nDone.")


if __name__ == "__main__":
    main()
