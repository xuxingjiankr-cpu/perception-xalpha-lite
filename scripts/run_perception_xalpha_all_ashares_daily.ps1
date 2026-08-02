# Windows PowerShell 5 promotes any native-process stderr line inside `2>&1 |
# Tee-Object` to a terminating NativeCommandError when this is `Stop`. Market-data
# collectors legitimately emit warnings on stderr, so native exit codes below are the
# authority. Every stage is checked explicitly and still fails closed on non-zero exit.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$logDirectory = Join-Path $root "outputs\edge_research\perception_xalpha_all_ashares_v4_gross\run_logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory ("daily_gross_discovery_" + $stamp + ".log")
$statusPath = Join-Path $logDirectory ("daily_gross_discovery_" + $stamp + ".json")

$status = [ordered]@{
    schemaVersion = "perception_xalpha_daily_runner_v1"
    status = "research_only_not_trading"
    startedAt = (Get-Date).ToString("o")
    config = "configs/research/perception_xalpha_all_ashares_v4_gross_discovery.json"
    masterStatus = "not_run"
    collectorStatus = "not_run"
    auditStatus = "not_run"
    minerStatus = "not_run"
    exitCode = $null
    finishedAt = $null
}

& py -3.13 scripts\collect_ashare_research_daily.py --mode master *>&1 |
  Tee-Object -FilePath $logPath
$status.masterStatus = if ($LASTEXITCODE -eq 0) { "ok" } else { "failed_$LASTEXITCODE" }
if ($LASTEXITCODE -ne 0) {
    $status.exitCode = $LASTEXITCODE
    $status.finishedAt = (Get-Date).ToString("o")
    $status | ConvertTo-Json -Depth 4 | Out-File -FilePath $statusPath -Encoding utf8
    exit $LASTEXITCODE
}

& py -3.13 scripts\collect_ashare_research_daily.py --mode backfill --workers 4 --max-pages 4 *>&1 |
  Tee-Object -FilePath $logPath -Append
$status.collectorStatus = if ($LASTEXITCODE -eq 0) { "ok_or_within_tolerance" } else { "failed_$LASTEXITCODE" }
if ($LASTEXITCODE -ne 0) {
    $status.exitCode = $LASTEXITCODE
    $status.finishedAt = (Get-Date).ToString("o")
    $status | ConvertTo-Json -Depth 4 | Out-File -FilePath $statusPath -Encoding utf8
    exit $LASTEXITCODE
}

& py -3.13 scripts\collect_ashare_research_daily.py --mode audit *>&1 |
  Tee-Object -FilePath $logPath -Append
$status.auditStatus = if ($LASTEXITCODE -eq 0) { "ok" } else { "failed_$LASTEXITCODE" }
if ($LASTEXITCODE -ne 0) {
    $status.exitCode = $LASTEXITCODE
    $status.finishedAt = (Get-Date).ToString("o")
    $status | ConvertTo-Json -Depth 4 | Out-File -FilePath $statusPath -Encoding utf8
    exit $LASTEXITCODE
}

& py -3.13 scripts\research_perception_xalpha_autonomous.py `
  --config configs\research\perception_xalpha_all_ashares_v4_gross_discovery.json run `
  *>&1 | Tee-Object -FilePath $logPath -Append
$status.exitCode = $LASTEXITCODE
$status.minerStatus = if ($LASTEXITCODE -eq 0) { "ok_or_no_new_data" } else { "failed_$LASTEXITCODE" }
$status.finishedAt = (Get-Date).ToString("o")
$status | ConvertTo-Json -Depth 4 | Out-File -FilePath $statusPath -Encoding utf8
exit $LASTEXITCODE
