$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$outputRoot = Join-Path $root "outputs\edge_research\perception_xalpha_all_ashares_v5_fundamental"
$logDirectory = Join-Path $outputRoot "run_logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory ("fundamental_discovery_" + $stamp + ".log")
$statusPath = Join-Path $logDirectory ("fundamental_discovery_" + $stamp + ".json")

$status = [ordered]@{
    schemaVersion = "perception_xalpha_fundamental_runner_v1"
    status = "research_only_not_trading"
    startedAt = (Get-Date).ToString("o")
    collectorStatus = "not_run"
    auditStatus = "not_run"
    minerStatus = "not_run"
    exitCode = $null
    finishedAt = $null
}

# Weekly force refresh is intentional: quarterly reports can be revised and UPDATE_DATE
# must be captured. Each security file is replaced atomically.
& py -3.13 scripts\collect_ashare_fundamentals.py --mode collect --workers 4 --force *>&1 |
  Tee-Object -FilePath $logPath
$status.collectorStatus = if ($LASTEXITCODE -in @(0, 3)) { "ok_or_isolated_failures" } else { "failed_$LASTEXITCODE" }
if ($LASTEXITCODE -notin @(0, 3)) {
    $status.exitCode = $LASTEXITCODE
    $status.finishedAt = (Get-Date).ToString("o")
    $status | ConvertTo-Json -Depth 4 | Out-File $statusPath -Encoding utf8
    exit $LASTEXITCODE
}

& py -3.13 scripts\collect_ashare_fundamentals.py --mode audit *>&1 |
  Tee-Object -FilePath $logPath -Append
$status.auditStatus = if ($LASTEXITCODE -eq 0) { "ok" } else { "failed_$LASTEXITCODE" }
if ($LASTEXITCODE -ne 0) {
    $status.exitCode = $LASTEXITCODE
    $status.finishedAt = (Get-Date).ToString("o")
    $status | ConvertTo-Json -Depth 4 | Out-File $statusPath -Encoding utf8
    exit $LASTEXITCODE
}

& py -3.13 scripts\research_perception_xalpha_autonomous.py `
  --config configs\research\perception_xalpha_all_ashares_v5_fundamental.json run `
  *>&1 | Tee-Object -FilePath $logPath -Append
$status.exitCode = $LASTEXITCODE
$status.minerStatus = if ($LASTEXITCODE -eq 0) { "ok_or_no_new_data" } else { "failed_$LASTEXITCODE" }
$status.finishedAt = (Get-Date).ToString("o")
$status | ConvertTo-Json -Depth 4 | Out-File $statusPath -Encoding utf8
exit $LASTEXITCODE

