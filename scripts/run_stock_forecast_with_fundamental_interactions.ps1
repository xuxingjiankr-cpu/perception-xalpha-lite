$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$rankingRun = "run_${stamp}_sixteen_factor_ranking"

& py -3.13 scripts\generate_sixteen_factor_interaction_top10_v1.py --run-id $rankingRun
if ($LASTEXITCODE -ne 0) {
    throw "Sixteen-factor interaction ranking failed closed with exit code $LASTEXITCODE"
}
Write-Output "Research-only sixteen-factor interaction ranking published to the dashboard."
exit 0
