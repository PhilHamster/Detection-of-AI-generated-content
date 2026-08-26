"""Feature ablation study for the 15-feature lightweight hybrid detector.

This script reuses cached train_features.csv and test_features.csv. It does not
run RoBERTa again. It keeps the final HistGradientBoosting configuration fixed,
uses the same source-stratified 80/20 validation split as the original study,
selects each experiment's decision threshold on validation balanced accuracy,
then refits on the complete training feature set and evaluates the untouched
test set.

Two complementary ablation analyses are performed:
  1. Group ablation: remove one conceptually related feature group at a time.
  2. Leave-one-feature-out (LOFO): remove each of the 15 features in turn.

Paired, source-stratified bootstrap confidence intervals quantify the change
relative to the full model for overall AUROC and human-vs-fusion AUROC.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
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


SEED = 42

FEATURE_GROUPS = {
    "document": ["doc_ai_prob"],
    "distribution": [
        "sent_mean",
        "sent_max",
        "sent_prop_gt_05",
        "sent_min",
        "sent_median",
        "sent_q25",
        "sent_q75",
        "sent_top2_mean",
    ],
    "extremes": ["sent_bottom2_mean"],
    "heterogeneity": [
        "sent_std",
        "sent_range",
        "sent_diff_mean",
        "sent_high_low_coexist",
    ],
    "length": ["n_sentences"],
}

ALL_FEATURES = [feature for group in FEATURE_GROUPS.values() for feature in group]
SENTENCE_FEATURES = [f for f in ALL_FEATURES if f != "doc_ai_prob"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "improved_fusion_results_strict_256",
        help="Directory containing train_features.csv and test_features.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>/feature_ablation_results).",
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_model(seed: int) -> HistGradientBoostingClassifier:
    """Return exactly the HistGradientBoosting configuration of the final model."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=seed,
    )


def validate_frame(frame: pd.DataFrame, name: str) -> None:
    required = {"id", "label", "source", *ALL_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame[ALL_FEATURES].isna().any().any():
        bad = frame[ALL_FEATURES].columns[frame[ALL_FEATURES].isna().any()].tolist()
        raise ValueError(f"{name} contains missing values in: {bad}")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"{name} must contain binary labels 0 and 1")


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Select threshold only on validation, maximizing balanced accuracy."""
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = np.asarray(
        [balanced_accuracy_score(y_true, probabilities >= t) for t in candidates]
    )
    return float(candidates[int(np.argmax(scores))])


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predictions),
    }


def source_aurocs(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    labels = frame["label"].to_numpy()
    sources = frame["source"].astype(str).str.lower().to_numpy()
    result = {}
    for source in ("generation", "polish", "fusion"):
        mask = np.isin(sources, ["human", source])
        result[f"auc_{source}"] = roc_auc_score(labels[mask], probabilities[mask])
    return result


def build_experiments() -> list[dict]:
    experiments = [
        {
            "experiment": "Full model",
            "ablation_type": "reference",
            "removed": "none",
            "features": ALL_FEATURES,
        }
    ]

    # Removing all sentence features is a useful coarse-grained sanity check.
    group_definitions = {"all_sentence": SENTENCE_FEATURES, **FEATURE_GROUPS}
    for group_name, removed_features in group_definitions.items():
        experiments.append(
            {
                "experiment": f"Without group: {group_name}",
                "ablation_type": "group",
                "removed": group_name,
                "features": [f for f in ALL_FEATURES if f not in removed_features],
            }
        )

    for feature in ALL_FEATURES:
        experiments.append(
            {
                "experiment": f"Without feature: {feature}",
                "ablation_type": "single_feature",
                "removed": feature,
                "features": [f for f in ALL_FEATURES if f != feature],
            }
        )
    return experiments


def fit_experiment(
    experiment: dict,
    meta_train: pd.DataFrame,
    validation: pd.DataFrame,
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[dict, np.ndarray, HistGradientBoostingClassifier]:
    features = experiment["features"]

    validation_model = make_model(seed)
    validation_model.fit(
        meta_train[features].to_numpy(), meta_train["label"].to_numpy()
    )
    validation_probabilities = validation_model.predict_proba(
        validation[features].to_numpy()
    )[:, 1]
    threshold = choose_threshold(validation["label"].to_numpy(), validation_probabilities)

    final_model = make_model(seed)
    final_model.fit(full_train[features].to_numpy(), full_train["label"].to_numpy())
    test_probabilities = final_model.predict_proba(test[features].to_numpy())[:, 1]

    result = {
        "experiment": experiment["experiment"],
        "ablation_type": experiment["ablation_type"],
        "removed": experiment["removed"],
        "n_features": len(features),
        "features_used": ";".join(features),
        "threshold": threshold,
        **calculate_metrics(test["label"].to_numpy(), test_probabilities, threshold),
        **source_aurocs(test, test_probabilities),
    }
    return result, test_probabilities, final_model


def source_stratified_bootstrap_indices(
    sources: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample within each source so every replicate retains all four sources."""
    sampled = []
    for source in np.unique(sources):
        group_indices = np.flatnonzero(sources == source)
        sampled.append(rng.choice(group_indices, size=len(group_indices), replace=True))
    return np.concatenate(sampled)


def paired_bootstrap_differences(
    frame: pd.DataFrame,
    full_probs: np.ndarray,
    ablated_probs: np.ndarray,
    n_bootstrap: int,
    seed: int,
    description: str,
) -> dict[str, float]:
    """Return ablated-minus-full AUROC differences and paired 95% CIs."""
    y = frame["label"].to_numpy()
    sources = frame["source"].astype(str).str.lower().to_numpy()
    fusion_mask = np.isin(sources, ["human", "fusion"])
    rng = np.random.default_rng(seed)
    overall_differences = np.empty(n_bootstrap, dtype=float)
    fusion_differences = np.empty(n_bootstrap, dtype=float)

    for iteration in tqdm(
        range(n_bootstrap), desc=description, unit="replicate", leave=False
    ):
        indices = source_stratified_bootstrap_indices(sources, rng)
        overall_differences[iteration] = (
            roc_auc_score(y[indices], ablated_probs[indices])
            - roc_auc_score(y[indices], full_probs[indices])
        )

        selected_fusion_indices = indices[fusion_mask[indices]]
        fusion_differences[iteration] = (
            roc_auc_score(y[selected_fusion_indices], ablated_probs[selected_fusion_indices])
            - roc_auc_score(y[selected_fusion_indices], full_probs[selected_fusion_indices])
        )

    def summarise(values: np.ndarray, prefix: str) -> dict[str, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return {
            f"{prefix}_bootstrap_mean": float(values.mean()),
            f"{prefix}_ci_low": float(low),
            f"{prefix}_ci_high": float(high),
            f"{prefix}_p_decrease": float(np.mean(values < 0)),
        }

    return {
        **summarise(overall_differences, "delta_auroc"),
        **summarise(fusion_differences, "delta_fusion_auroc"),
    }


def classify_importance(delta: float, ci_low: float, ci_high: float) -> str:
    """Conservative descriptive label; statistical claims should use the CI itself."""
    loss = -delta
    if ci_high < 0 and loss >= 0.005:
        return "performance-critical"
    if ci_high < 0:
        return "complementary but small"
    if ci_low > 0:
        return "potentially harmful or unstable"
    return "weak or redundant"


def add_differences(results: pd.DataFrame) -> pd.DataFrame:
    full = results.loc[results["experiment"] == "Full model"].iloc[0]
    for metric in ("auroc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc",
                   "auc_generation", "auc_polish", "auc_fusion"):
        results[f"delta_{metric}"] = results[metric] - full[metric]
        results[f"loss_{metric}"] = full[metric] - results[metric]
    return results


def save_plot(results: pd.DataFrame, output_dir: Path, ablation_type: str) -> None:
    subset = results[results["ablation_type"] == ablation_type].copy()
    subset = subset.sort_values("loss_auc_fusion", ascending=True)
    labels = subset["removed"].str.replace("sent_", "", regex=False)
    height = max(5.0, 0.42 * len(subset) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), sharey=True)
    colours = ["#3b82f6" if value >= 0 else "#c17d11" for value in subset["loss_auroc"]]
    axes[0].barh(labels, subset["loss_auroc"], color=colours)
    colours = ["#3b82f6" if value >= 0 else "#c17d11" for value in subset["loss_auc_fusion"]]
    axes[1].barh(labels, subset["loss_auc_fusion"], color=colours)
    axes[0].set_title("Overall AUROC loss")
    axes[1].set_title("Fusion AUROC loss")
    for axis in axes:
        axis.axvline(0, color="#222222", linewidth=0.8)
        axis.grid(axis="x", alpha=0.2)
        axis.set_xlabel("Full model AUROC - ablated model AUROC")
    fig.suptitle(
        "Group ablation" if ablation_type == "group" else "Leave-one-feature-out ablation",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    stem = "group_ablation" if ablation_type == "group" else "single_feature_ablation"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, n_bootstrap: int) -> None:
    text = f"""# Feature ablation outputs

Protocol:
- Same 15 cached features and HistGradientBoosting configuration as the final model.
- Source-stratified 80/20 validation split, random seed 42.
- Each experiment selects its threshold on validation balanced accuracy.
- Each model is then refitted on all training rows and evaluated once on the held-out test set.
- {n_bootstrap:,} paired, source-stratified bootstrap replicates.
- Delta is **ablated minus full model**; loss is **full minus ablated**.
- A positive loss means that removing the feature/group reduced performance.

Files:
- `all_ablation_results.csv`: complete metrics, deltas and confidence intervals.
- `group_ablation_results.csv`: group-level thesis table.
- `single_feature_ablation_results.csv`: LOFO feature table.
- `test_probabilities.csv`: probabilities for reproducibility and further analysis.
- `group_ablation.png/pdf` and `single_feature_ablation.png/pdf`: figures.
- `models/`: fitted model for every experiment.

Interpretation warning:
The feature columns are correlated. LOFO measures the *unique conditional contribution*
of one feature given all remaining features; it is not a causal importance measure.
This analysis intentionally follows the existing non-OOF meta-feature protocol, which
must be reported as a methodological limitation.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "feature_ablation_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    train = pd.read_csv(input_dir / "train_features.csv", dtype={"id": str})
    test = pd.read_csv(input_dir / "test_features.csv", dtype={"id": str})
    validate_frame(train, "train_features.csv")
    validate_frame(test, "test_features.csv")

    meta_train, validation = train_test_split(
        train,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=train["source"],
    )
    print(
        f"Loaded train={len(train):,}, test={len(test):,}; "
        f"meta-train={len(meta_train):,}, validation={len(validation):,}"
    )

    experiments = build_experiments()
    rows = []
    probabilities = test[["id", "label", "source"]].copy()
    probability_arrays = {}

    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index:02d}/{len(experiments):02d}] {experiment['experiment']}")
        result, experiment_probs, fitted_model = fit_experiment(
            experiment, meta_train, validation, train, test, args.seed
        )
        rows.append(result)
        safe_name = (
            experiment["removed"].replace(" ", "_").replace("/", "_")
            if experiment["removed"] != "none"
            else "full_model"
        )
        probability_column = f"prob_{experiment['ablation_type']}_{safe_name}"
        probabilities[probability_column] = experiment_probs
        probability_arrays[experiment["experiment"]] = experiment_probs
        joblib.dump(
            {"model": fitted_model, "features": experiment["features"], "threshold": result["threshold"]},
            model_dir / f"{index:02d}_{safe_name}.joblib",
        )

    results = add_differences(pd.DataFrame(rows))
    full_probs = probability_arrays["Full model"]
    bootstrap_rows = []
    for index, experiment in enumerate(experiments[1:], start=1):
        print(f"Bootstrap [{index:02d}/{len(experiments)-1:02d}] {experiment['experiment']}")
        statistics = paired_bootstrap_differences(
            test,
            full_probs,
            probability_arrays[experiment["experiment"]],
            args.bootstrap,
            args.seed + index,
            experiment["removed"],
        )
        bootstrap_rows.append({"experiment": experiment["experiment"], **statistics})

    bootstrap = pd.DataFrame(bootstrap_rows)
    results = results.merge(bootstrap, on="experiment", how="left")
    results["overall_importance"] = results.apply(
        lambda row: "reference"
        if row["ablation_type"] == "reference"
        else classify_importance(
            row["delta_auroc"], row["delta_auroc_ci_low"], row["delta_auroc_ci_high"]
        ),
        axis=1,
    )
    results["fusion_importance"] = results.apply(
        lambda row: "reference"
        if row["ablation_type"] == "reference"
        else classify_importance(
            row["delta_auc_fusion"],
            row["delta_fusion_auroc_ci_low"],
            row["delta_fusion_auroc_ci_high"],
        ),
        axis=1,
    )

    results.to_csv(output_dir / "all_ablation_results.csv", index=False)
    results[results["ablation_type"].isin(["reference", "group"])].to_csv(
        output_dir / "group_ablation_results.csv", index=False
    )
    results[results["ablation_type"].isin(["reference", "single_feature"])].to_csv(
        output_dir / "single_feature_ablation_results.csv", index=False
    )
    probabilities.to_csv(output_dir / "test_probabilities.csv", index=False)
    save_plot(results, output_dir, "group")
    save_plot(results, output_dir, "single_feature")
    write_readme(output_dir, args.bootstrap)

    run_metadata = {
        "seed": args.seed,
        "validation_size": args.validation_size,
        "bootstrap_replicates": args.bootstrap,
        "train_rows": len(train),
        "test_rows": len(test),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    print("\nFull model and ablation summary:")
    display_columns = [
        "experiment", "auroc", "auc_fusion", "loss_auroc", "loss_auc_fusion",
        "overall_importance", "fusion_importance",
    ]
    print(results[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
