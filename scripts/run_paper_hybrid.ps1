# Run the paper's hybrid experiment (~35 min on i7-class CPU)
# Activate venv first: .\venv311\Scripts\Activate.ps1

param(
    [ValidateSet("paper", "quick")]
    [string]$Preset = "paper",
    [switch]$SkipPreprocess
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not $SkipPreprocess) {
    python main.py preprocess --preset $Preset
}

python main.py run hybrid --preset $Preset --evaluate

Write-Host ""
Write-Host "Done. Check outputs/runs/latest_run.txt for artifact location."
