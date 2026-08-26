"""
baseline_RoBERTa.py
====================
Fine-tune RoBERTa-base as a binary classifier (human=0, AI=1).
Requires GPU + HuggingFace access.

Usage (from project root):
    python "CHEAT_TEST/baseline_RoBERTa.py"
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

INPUT_JSONL = "CHEAT_TEST/cheat_unified.jsonl"
OUTPUT_DIR  = "CHEAT_TEST/roberta_results"


class CheatTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs_ai = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds    = np.argmax(logits, axis=-1)
    acc      = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs_ai)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auc}


def load_split(jsonl_path, split_name):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split_name:
                rows.append(r)
    return rows


def main():
    parser_args = {
        "model_name": "roberta-base",
        "epochs":     3,
        "batch_size": 16,
        "lr":         2e-5,
        "max_length": 256,
        "seed":       42,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data ...")
    train_rows = load_split(INPUT_JSONL, "train")
    test_rows  = load_split(INPUT_JSONL, "test")
    print(f"  train: {len(train_rows)}, test: {len(test_rows)}")

    # leakage guard
    train_ids = {r["id"] for r in train_rows}
    test_ids  = {r["id"] for r in test_rows}
    overlap   = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"Data leakage: {len(overlap)} paper ids in both splits. "
            f"Re-run prepare_data.py."
        )
    print("  Leakage check passed.")

    tokenizer = AutoTokenizer.from_pretrained(parser_args["model_name"])
    model     = AutoModelForSequenceClassification.from_pretrained(
        parser_args["model_name"], num_labels=2
    )

    train_dataset = CheatTextDataset(
        [r["text"] for r in train_rows], [r["label"] for r in train_rows],
        tokenizer, parser_args["max_length"]
    )
    test_dataset = CheatTextDataset(
        [r["text"] for r in test_rows], [r["label"] for r in test_rows],
        tokenizer, parser_args["max_length"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=parser_args["epochs"],
        per_device_train_batch_size=parser_args["batch_size"],
        per_device_eval_batch_size=parser_args["batch_size"],
        learning_rate=parser_args["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        seed=parser_args["seed"],
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Training ...")
    trainer.train()

    print("\nFinal evaluation on test set:")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    print("\nGenerating per-source breakdown ...")
    predictions  = trainer.predict(test_dataset)
    probs_ai     = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    preds        = np.argmax(predictions.predictions, axis=-1)

    results_df = pd.DataFrame(test_rows)
    results_df["pred_label"]   = preds
    results_df["pred_prob_ai"] = probs_ai

    results_csv = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved -> {results_csv}")

    print("\n=== AUROC per AI source vs human ===")
    human_rows = results_df[results_df["source"] == "human"]
    for source in ["generation", "polish", "fusion"]:
        ai_rows = results_df[results_df["source"] == source]
        if len(ai_rows) == 0:
            continue
        subset = pd.concat([human_rows, ai_rows])
        try:
            auc = roc_auc_score(subset["label"], subset["pred_prob_ai"])
            print(f"  human vs {source:12s}: AUROC = {auc:.4f}"
                  f"  (n_human={len(human_rows)}, n_{source}={len(ai_rows)})")
        except ValueError:
            print(f"  human vs {source:12s}: AUROC undefined")

    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"\nModel saved -> {os.path.join(OUTPUT_DIR, 'final_model')}")


if __name__ == "__main__":
    main()