param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataPath = Join-Path $ProjectDir "cheat_unified.jsonl"
$RoBertaOutput = Join-Path $ProjectDir "roberta_results_strict"
$FusionOutput = Join-Path $ProjectDir "improved_fusion_results_strict_256"

Write-Host "Stage 1: leakage-controlled RoBERTa training" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "strict_roberta_training.py") `
    --data $DataPath `
    --output $RoBertaOutput
if ($LASTEXITCODE -ne 0) { throw "Strict RoBERTa training failed." }

Write-Host "Stage 2: regenerate document and sentence features, then fit the hybrid model" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "improved_fusion_stacking.py") `
    --data $DataPath `
    --checkpoint (Join-Path $RoBertaOutput "final_model") `
    --output $FusionOutput `
    --doc-max-length 256 `
    --sent-max-length 128
if ($LASTEXITCODE -ne 0) { throw "Strict fusion experiment failed." }

Write-Host "Stage 3: group and leave-one-feature-out explanatory ablation" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "feature_ablation_study.py") `
    --input-dir $FusionOutput `
    --output-dir (Join-Path $FusionOutput "feature_ablation_results") `
    --bootstrap 2000
if ($LASTEXITCODE -ne 0) { throw "Strict feature ablation failed." }

Write-Host "Strict pipeline completed successfully." -ForegroundColor Green
Write-Host "RoBERTa results: $RoBertaOutput"
Write-Host "Fusion and ablation results: $FusionOutput"
