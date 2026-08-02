$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

# Unattended run must be observable and must not abort on isolated symbol failures.
# On 2026-08-01 five unreachable BJ symbols returned exit 3 from the backfill and the miner
# never started, while the weekly log gave no way to tell collection from mining apart.
# Coverage tolerance now lives in the collector (MINIMUM_BACKFILL_COVERAGE); this runner
# records which stage reached which state so a silent no-op is visible afterwards.
$logDirectory = Join-Path $root "outputs\edge_research\perception_xalpha_all_ashares\run_logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$statusPath = Join-Path $logDirectory ("weekly_status_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json")
$status = [ordered]@{
    startedAt       = (Get-Date).ToString("o")
    masterStatus    = "not_run"
    collectorStatus = "not_run"
    auditStatus     = "not_run"
    minerStarted    = $false
    minerCompleted  = $false
    exitCode        = $null
}
function Save-Status { $status | ConvertTo-Json -Depth 4 | Out-File -FilePath $statusPath -Encoding utf8 }

py -3.13 scripts\collect_ashare_research_daily.py --mode master
$status.masterStatus = if ($LASTEXITCODE -eq 0) { "ok" } else { "failed_$LASTEXITCODE" }
Save-Status
if ($LASTEXITCODE -ne 0) { $status.exitCode = $LASTEXITCODE; Save-Status; exit $LASTEXITCODE }

py -3.13 scripts\collect_ashare_research_daily.py --mode backfill --workers 4 --max-pages 4
$status.collectorStatus = if ($LASTEXITCODE -eq 0) { "ok_or_within_tolerance" } else { "insufficient_coverage_$LASTEXITCODE" }
Save-Status
if ($LASTEXITCODE -ne 0) { $status.exitCode = $LASTEXITCODE; Save-Status; exit $LASTEXITCODE }

py -3.13 scripts\collect_ashare_research_daily.py --mode audit
$status.auditStatus = if ($LASTEXITCODE -eq 0) { "ok" } else { "failed_$LASTEXITCODE" }
Save-Status
if ($LASTEXITCODE -ne 0) { $status.exitCode = $LASTEXITCODE; Save-Status; exit $LASTEXITCODE }

$status.minerStarted = $true
Save-Status
py -3.13 scripts\research_perception_xalpha_autonomous.py `
  --config configs\research\perception_xalpha_all_ashares_v2.json run
# Distinguish a real new cycle from an idempotent skip: exit 0 alone cannot tell them apart.
$latestResult = Get-ChildItem -Path (Join-Path $root "outputs\edge_research\perception_xalpha_all_ashares_v2") -Filter "result.json" -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$cycleStatus = "unknown"
if ($latestResult) {
    try {
        $parsed = Get-Content $latestResult.FullName -Raw | ConvertFrom-Json
        $cycleStatus = if ($parsed.status -eq "no_new_data") { "no_new_data" } else { "completed_with_new_cycle" }
    } catch { $cycleStatus = "unreadable_result" }
}
$status.cycleStatus = $cycleStatus
$status.minerCompleted = ($LASTEXITCODE -eq 0)
$status.exitCode = $LASTEXITCODE
$status.finishedAt = (Get-Date).ToString("o")
Save-Status
exit $LASTEXITCODE
