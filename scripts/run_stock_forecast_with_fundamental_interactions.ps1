$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$forecastRun = "run_${stamp}_complete12_forecast"
$interactionRun = "run_${stamp}_fundamental_interactions"

& py -3.13 scripts\generate_guarded_weight_top10_forecast_v1.py --run-id $forecastRun
if ($LASTEXITCODE -ne 0) {
    throw "Complete twelve-factor forecast failed closed with exit code $LASTEXITCODE"
}

& py -3.13 scripts\research_twelve_factor_fundamental_price_interactions_v1.py --run-id $interactionRun
if ($LASTEXITCODE -ne 0) {
    throw "Fundamental interaction shadow run failed closed with exit code $LASTEXITCODE"
}

$interactionResult = Join-Path $root "outputs\edge_research\twelve_factor_fundamental_price_interactions_v1\$interactionRun\result.json"
if (-not (Test-Path -LiteralPath $interactionResult)) {
    throw "Fundamental interaction artifact is missing: $interactionResult"
}

& py -3.13 scripts\publish_fundamental_interaction_shadow.py --interaction-result $interactionResult
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard shadow merge failed closed with exit code $LASTEXITCODE"
}

Write-Output "Research-only complete 12-factor dashboard plus fundamental interaction shadow published."
exit 0
