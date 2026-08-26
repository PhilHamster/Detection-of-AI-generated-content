"""Leakage-controlled RoBERTa training for the CHEAT experiment.

Protocol
--------
1. Keep the existing official/project train-test split unchanged.
2. Split only the original training rows into 80% development-training and
   20% validation at paper-ID level.
3. Fine-tune roberta-base for three epochs and choose the best epoch using
   validation F1. The held-out test set is not passed to Trainer in this stage.
4. Re-initialise roberta-base and train on all original training rows for the
   selected number of epochs.
5. Evaluate the held-out test set once and save probabilities and metrics.

The script deliberately writes to a new strict directory and never overwrites
the earlier RoBERTa experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "cheat_unified.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "roberta_results_strict",
    )
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--candidate-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"], truncation=True, max_length=self.max_length
        )
        item = {key: torch.tensor(value) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


def load_rows(path: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "text", "label", "source", "split"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            row["id"] = str(row["id"])
            rows.append(row)

    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    overlap = {row["id"] for row in train} & {row["id"] for row in test}
    if overlap:
        raise ValueError(f"Train/test leakage: {len(overlap)} paper IDs overlap")
    if not train or not test:
        raise ValueError("Both train and test splits must be non-empty")
    return train, test


def grouped_train_validation_split(
    rows: list[dict], validation_size: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Assign complete paper-ID groups to development train or validation."""
    ids = sorted({row["id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_validation_ids = max(1, round(len(ids) * validation_size))
    validation_ids = set(ids[:n_validation_ids])
    development_train = [row for row in rows if row["id"] not in validation_ids]
    validation = [row for row in rows if row["id"] in validation_ids]

    overlap = {row["id"] for row in development_train} & {
        row["id"] for row in validation
    }
    if overlap:
        raise AssertionError("Internal paper-ID split failed")
    return development_train, validation


def describe_split(name: str, rows: list[dict]) -> dict:
    counts = Counter(str(row["source"]) for row in rows)
    summary = {
        "name": name,
        "rows": len(rows),
        "paper_ids": len({row["id"] for row in rows}),
        "sources": dict(sorted(counts.items())),
    }
    print(
        f"{name}: rows={summary['rows']:,}, paper IDs={summary['paper_ids']:,}, "
        f"sources={summary['sources']}"
    )
    return summary


def metrics_from_arrays(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
        "auroc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
    }


def compute_metrics(eval_prediction) -> dict:
    logits, labels = eval_prediction
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return metrics_from_arrays(np.asarray(labels), probabilities)


def common_training_arguments(args: argparse.Namespace) -> dict:
    return {
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "logging_steps": 20,
    }


def select_best_epoch(
    args: argparse.Namespace,
    tokenizer,
    development_train: list[dict],
    validation: list[dict],
    selection_dir: Path,
) -> tuple[int, dict]:
    print("\nStage 1/2: selecting the epoch using internal validation only")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    train_dataset = TextDataset(development_train, tokenizer, args.max_length)
    validation_dataset = TextDataset(validation, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(selection_dir),
        num_train_epochs=args.candidate_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.candidate_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        **common_training_arguments(args),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    state = trainer.state
    if state.best_model_checkpoint is None:
        raise RuntimeError("Trainer did not record a best validation checkpoint")
    steps_per_epoch = int(np.ceil(len(development_train) / args.batch_size))
    checkpoint_step = int(Path(state.best_model_checkpoint).name.split("-")[-1])
    best_epoch = max(1, int(round(checkpoint_step / steps_per_epoch)))
    best_epoch = min(best_epoch, args.candidate_epochs)
    selection = {
        "best_epoch": best_epoch,
        "best_validation_f1": float(state.best_metric),
        "best_checkpoint": str(state.best_model_checkpoint),
        "steps_per_epoch": steps_per_epoch,
    }
    print(
        f"Selected epoch={best_epoch}; validation F1={state.best_metric:.4f}; "
        f"checkpoint={state.best_model_checkpoint}"
    )
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch, selection


def retrain_on_full_training_set(
    args: argparse.Namespace,
    tokenizer,
    full_train: list[dict],
    best_epoch: int,
    retraining_dir: Path,
) -> Trainer:
    print(
        f"\nStage 2/2: reinitialising {args.model_name} and training all "
        f"{len(full_train):,} training rows for {best_epoch} epoch(s)"
    )
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    training_args = TrainingArguments(
        output_dir=str(retraining_dir),
        num_train_epochs=best_epoch,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        **common_training_arguments(args),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TextDataset(full_train, tokenizer, args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    return trainer


def evaluate_test_once(
    trainer: Trainer,
    tokenizer,
    test: list[dict],
    max_length: int,
    output_dir: Path,
) -> dict:
    print("\nFinal held-out test evaluation (first use of test labels)")
    dataset = TextDataset(test, tokenizer, max_length)
    prediction_output = trainer.predict(dataset)
    probabilities = torch.softmax(
        torch.tensor(prediction_output.predictions), dim=-1
    )[:, 1].numpy()
    labels = np.asarray([int(row["label"]) for row in test])
    metrics = metrics_from_arrays(labels, probabilities)

    frame = pd.DataFrame(test)
    frame["pred_prob_ai"] = probabilities
    frame["pred_label"] = (probabilities >= 0.5).astype(int)
    frame.to_csv(output_dir / "test_predictions.csv", index=False)

    sources = frame["source"].astype(str).to_numpy()
    source_metrics = {}
    for source in ("generation", "polish", "fusion"):
        mask = np.isin(sources, ["human", source])
        source_metrics[f"human_vs_{source}_auroc"] = roc_auc_score(
            labels[mask], probabilities[mask]
        )
    metrics.update(source_metrics)
    return metrics


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty strict output directory: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    train, test = load_rows(args.data)
    development_train, validation = grouped_train_validation_split(
        train, args.validation_size, args.seed
    )
    split_summaries = [
        describe_split("development_train", development_train),
        describe_split("internal_validation", validation),
        describe_split("full_original_train", train),
        describe_split("held_out_test", test),
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    selection_dir = args.output / "epoch_selection"
    retraining_dir = args.output / "full_retraining_logs"
    final_model_dir = args.output / "final_model"

    best_epoch, selection = select_best_epoch(
        args, tokenizer, development_train, validation, selection_dir
    )
    trainer = retrain_on_full_training_set(
        args, tokenizer, train, best_epoch, retraining_dir
    )
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    test_metrics = evaluate_test_once(
        trainer, tokenizer, test, args.max_length, args.output
    )
    protocol = {
        "data": str(args.data),
        "model_name": args.model_name,
        "seed": args.seed,
        "internal_validation_size": args.validation_size,
        "candidate_epochs": args.candidate_epochs,
        "selected_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "split_summaries": split_summaries,
        "selection": selection,
        "test_metrics": test_metrics,
        "test_used_during_epoch_selection": False,
        "reinitialised_before_full_retraining": True,
    }
    (args.output / "strict_protocol_and_results.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\nSTRICT ROBERTA TEST RESULTS")
    for name, value in test_metrics.items():
        print(f"{name:<28}: {value:.4f}")
    print(f"\nFinal model: {final_model_dir}")
    print(f"All strict outputs: {args.output}")


if __name__ == "__main__":
    main()
