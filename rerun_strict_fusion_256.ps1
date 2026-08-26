param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataPath = Join-Path $ProjectDir "cheat_unified.jsonl"
$Checkpoint = Join-Path $ProjectDir "roberta_results_strict\final_model"
$FusionOutput = Join-Path $ProjectDir "improved_fusion_results_strict_256"
$AblationOutput = Join-Path $FusionOutput "feature_ablation_results"

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "Dataset not found: $DataPath"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Strict RoBERTa checkpoint not found: $Checkpoint"
}
if (Test-Path -LiteralPath $FusionOutput) {
    $existing = Get-ChildItem -LiteralPath $FusionOutput -Force
    if ($existing.Count -gt 0) {
        throw "Refusing to overwrite non-empty output directory: $FusionOutput"
    }
}

Write-Host "Stage 1/2: regenerate strict features with document max length 256" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "improved_fusion_stacking.py") `
    --data $DataPath `
    --checkpoint $Checkpoint `
    --output $FusionOutput `
    --doc-max-length 256 `
    --sent-max-length 128
if ($LASTEXITCODE -ne 0) { throw "Strict 256-token fusion experiment failed." }

Write-Host "Stage 2/2: rerun explanatory ablation with 2,000 paired bootstrap replicates" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "feature_ablation_study.py") `
    --input-dir $FusionOutput `
    --output-dir $AblationOutput `
    --bootstrap 2000
if ($LASTEXITCODE -ne 0) { throw "Strict 256-token feature ablation failed." }

$metadata = [ordered]@{
    protocol = "strict_256_token_document_inference"
    checkpoint = $Checkpoint
    document_max_length = 256
    sentence_max_length = 128
    bootstrap_replicates = 2000
    roberta_retrained = $false
    note = "Uses the already leakage-controlled strict RoBERTa checkpoint; only downstream features, hybrid model and explanatory ablation were rerun."
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $FusionOutput "strict_256_rerun_metadata.json") -Encoding UTF8

Write-Host "Strict 256-token downstream rerun completed successfully." -ForegroundColor Green
Write-Host "Final outputs: $FusionOutput"
