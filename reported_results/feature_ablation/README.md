# Feature ablation outputs

Protocol:
- Same 15 cached features and HistGradientBoosting configuration as the final model.
- Source-stratified 80/20 validation split, random seed 42.
- Each experiment selects its threshold on validation balanced accuracy.
- Each model is then refitted on all training rows and evaluated once on the held-out test set.
- 2,000 paired, source-stratified bootstrap replicates.
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
