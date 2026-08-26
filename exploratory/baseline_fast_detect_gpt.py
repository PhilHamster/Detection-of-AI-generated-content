"""
run_fast_detect_gpt.py
=======================
Zero-shot detector: Fast-DetectGPT analytic sampling-discrepancy criterion.
Requires GPU + HuggingFace access.

Usage (from project root):
    python "CHEAT_TEST/run_fast_detect_gpt.py"
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
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

INPUT_JSONL = "CHEAT_TEST/cheat_unified.jsonl"
OUTPUT_CSV  = "CHEAT_TEST/fastdetectgpt_scores.csv"
CACHE_DIR   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_cache"
)

SCORING_MODEL  = "gpt2-xl"
SAMPLING_MODEL = "gpt2-xl"   # same model = self-comparison mode, only loads once
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH     = 512
FLOAT16_HINTS  = ["falcon", "llama", "gpt-j", "gpt-neo", "opt-13b", "bloom"]


def load_model_and_tokenizer(model_name, device, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "local." + model_name.replace("/", "_"))
    load_path  = local_path if os.path.exists(local_path) else model_name

    kwargs = {}
    if any(h in model_name.lower() for h in FLOAT16_HINTS):
        kwargs["torch_dtype"] = torch.float16

    print(f"Loading model: {load_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(load_path, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(load_path, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size  = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref  = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels      = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref    = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref    = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref     = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    return discrepancy.mean().item()


@torch.no_grad()
def score_text(text, scoring_model, scoring_tokenizer,
               sampling_model, sampling_tokenizer, device, same_model):
    tokenized = scoring_tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_LENGTH, return_token_type_ids=False,
    ).to(device)
    labels = tokenized.input_ids[:, 1:]
    if labels.shape[1] == 0:
        return None

    logits_score = scoring_model(**tokenized).logits[:, :-1]

    if same_model:
        logits_ref = logits_score
    else:
        tokenized_ref = sampling_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, return_token_type_ids=False,
        ).to(device)
        ref_labels = tokenized_ref.input_ids[:, 1:]
        if ref_labels.shape[1] != labels.shape[1] or not torch.equal(ref_labels, labels):
            return None
        logits_ref = sampling_model(**tokenized_ref).logits[:, :-1]

    return get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)


def main():
    same_model = (SCORING_MODEL == SAMPLING_MODEL)

    scoring_model, scoring_tokenizer = load_model_and_tokenizer(
        SCORING_MODEL, DEVICE, CACHE_DIR
    )
    if same_model:
        sampling_model, sampling_tokenizer = scoring_model, scoring_tokenizer
    else:
        sampling_model, sampling_tokenizer = load_model_and_tokenizer(
            SAMPLING_MODEL, DEVICE, CACHE_DIR
        )

    rows = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Scoring {len(rows)} samples on {DEVICE} ...")

    results, n_skipped = [], 0
    for r in tqdm(rows):
        score = score_text(
            r["text"], scoring_model, scoring_tokenizer,
            sampling_model, sampling_tokenizer, DEVICE, same_model,
        )
        if score is None:
            n_skipped += 1
            continue
        results.append({**r, "fastdetectgpt_score": score})

    if n_skipped:
        print(f"WARNING: skipped {n_skipped} samples.")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved scores -> {OUTPUT_CSV}")

    print("\n=== AUROC (human=0 vs AI=1) ===")
    print(f"Overall: {roc_auc_score(df['label'], df['fastdetectgpt_score']):.4f}")

    print("\n=== AUROC per AI source vs human ===")
    human_df = df[df["source"] == "human"]
    for source in ["generation", "polish", "fusion"]:
        ai_df   = df[df["source"] == source]
        if len(ai_df) == 0:
            continue
        subset  = pd.concat([human_df, ai_df])
        auc     = roc_auc_score(subset["label"], subset["fastdetectgpt_score"])
        print(f"  human vs {source:12s}: AUROC = {auc:.4f}"
              f"  (n_human={len(human_df)}, n_{source}={len(ai_df)})")


if __name__ == "__main__":
    main()