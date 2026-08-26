# Detection of AI generated content

Source code and selected result summaries for an MSc dissertation that compares
traditional, zero-shot, and supervised Transformer detectors on academic abstracts,
and evaluates a lightweight document-sentence hybrid detector for Human-AI fusion text.

## Method overview

The experiments use four CHEAT conditions: Human, Generation, Polish, and Fusion.
All final comparisons use the same paper-level split, so different versions of the
same source paper cannot cross the train-test boundary. The proposed hybrid combines
one document-level RoBERTa probability with fourteen statistics derived from
sentence-level probabilities. A HistGradientBoosting classifier performs the final
decision-level fusion.

The final evaluation uses 18,033 records: 12,640 training records and 5,393 held-out
test records. Document inputs are limited to 256 tokens and sentence inputs to 128
tokens. Model and threshold selection use only the training partition and an internal
validation split; the held-out test set is evaluated after selection.

## Repository contents

- `prepare_data.py`: builds the unified paper-level CHEAT split.
- `strict_roberta_training.py`: selects the training duration internally, then fits a
  fresh RoBERTa model on the full training partition.
- `improved_fusion_stacking.py`: extracts the 15 document-sentence features and fits
  the final hybrid model.
- `feature_ablation_study.py`: group and leave-one-feature-out explanatory ablation.
- `strict_sentence_baselines.py`: sentence-only LR, SVM, and MLP comparisons.
- `baseline_fast_detect_gpt_test_only.py`: Fast-DetectGPT-style test-only scoring.
- `complete_final_baselines.py`: final TF-IDF baselines, Fast-DetectGPT score audit,
  and paired bootstrap confidence intervals.
- `run_strict_pipeline.ps1`: complete strict RoBERTa, fusion, and ablation pipeline.
- `rerun_strict_fusion_256.ps1`: regenerates the final downstream experiment from an
  existing strict RoBERTa checkpoint.
- `reported_results/`: small machine-readable summaries used in the dissertation.
- `exploratory/`: earlier development scripts retained for transparency; these are not
  the source of the final dissertation results.

Raw CHEAT text, model weights, downloaded model caches, per-sample feature matrices,
and prediction files are intentionally excluded.

## Dataset

Download CHEAT from its official repository:

<https://github.com/botianzhe/CHEAT>

Place the following files in a local `CHEAT_DATASET` directory:

- `ieee-init.jsonl`
- `ieee-chatgpt-generation.jsonl`
- `ieee-chatgpt-polish.jsonl`
- `ieee-chatgpt-fusion.jsonl`

The dataset is described in:

> Yu, P., Chen, J., Feng, X. and Xia, Z. (2023). CHEAT: A large-scale dataset
> for detecting ChatGPT-writtEn AbsTracts. arXiv:2304.12008.

## Environment

The final run used Python 3.11.15 on Windows 10 with an Intel Core i7-11800H,
an NVIDIA GeForce RTX 3050 Ti Laptop GPU, and 16 GB RAM. The recorded PyTorch
build was `2.12.1+cu130`. Install a PyTorch build appropriate for the local CUDA
driver, then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

An Anaconda environment can be used instead. Model downloads require internet access
on the first run.

## Reproducing the final protocol

### 1. Prepare the fixed paper-level split

From the repository root:

```powershell
python prepare_data.py --cheat_dir CHEAT_DATASET --out_dir . --n_per_class 4514 --test_frac 0.30 --seed 42
```

This should produce `cheat_unified.jsonl` and `cheat_unified.csv` with 12,640
training records and 5,393 test records.

### 2. Run strict RoBERTa training, fusion, and ablation

```powershell
powershell -ExecutionPolicy Bypass -File .\run_strict_pipeline.ps1
```

Alternatively, run each stage explicitly:

```powershell
python strict_roberta_training.py --data cheat_unified.jsonl --output roberta_results_strict

python improved_fusion_stacking.py `
  --data cheat_unified.jsonl `
  --checkpoint roberta_results_strict\final_model `
  --output improved_fusion_results_strict_256 `
  --doc-max-length 256 `
  --sent-max-length 128

python feature_ablation_study.py `
  --input-dir improved_fusion_results_strict_256 `
  --output-dir improved_fusion_results_strict_256\feature_ablation_results `
  --bootstrap 2000
```

The scripts refuse to overwrite populated final-output directories. Rename or archive
an existing output directory before an intentional rerun.

### 3. Run the sentence-only comparisons

```powershell
python strict_sentence_baselines.py `
  --input-dir improved_fusion_results_strict_256 `
  --output-dir strict_sentence_baseline_results
```

### 4. Generate Fast-DetectGPT-style scores

```powershell
python baseline_fast_detect_gpt_test_only.py `
  --data_path cheat_unified.jsonl `
  --output_path fastdetectgpt_scores_test.csv `
  --eval_split test `
  --model_name gpt2-xl `
  --max_length 512
```

This experiment requires substantially more GPU memory and storage than the tabular
baselines because GPT-2 XL is downloaded locally.

### 5. Complete the final baseline audit and confidence intervals

After the strict fusion and Fast-DetectGPT-style outputs exist:

```powershell
python complete_final_baselines.py
```

## Principal results

| Method | Overall AUROC | Human vs Fusion AUROC |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.9167 | 0.7964 |
| TF-IDF + calibrated Linear SVM | 0.9196 | 0.8051 |
| Fast-DetectGPT-style | 0.7704 | 0.5782 |
| Document-level RoBERTa | 0.9513 | 0.8633 |
| Sentence-only HistGradientBoosting | 0.9401 | 0.8458 |
| Proposed 15-feature hybrid | **0.9536** | **0.8720** |

Relative to document-level RoBERTa, the hybrid improved Overall AUROC by 0.0023
(95% paired-bootstrap CI: 0.0008 to 0.0037) and Human-versus-Fusion AUROC by
0.0087 (95% CI: 0.0047 to 0.0128). These gains are positive but modest.

## Methodological limitation

The held-out test set was excluded from model, feature, and threshold selection.
However, the meta-training probabilities used by the fusion classifier were not
generated out of fold: they came from a RoBERTa model fitted on the same official
training partition. The reported fusion gain should therefore be confirmed using
true out-of-fold meta-features and additional datasets and generators.

Detector scores indicate similarity to learned patterns; they do not establish
authorship, intent, or academic misconduct.
