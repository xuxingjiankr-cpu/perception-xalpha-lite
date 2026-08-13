$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$date = Get-Date -Format "yyyy-MM-dd"
$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$logRoot = Join-Path $root "logs\sixteen_factor_daily"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot ("sixteen_factor_daily_" + $date + ".log")
$statusPath = Join-Path $logRoot ("sixteen_factor_daily_" + $date + ".json")
$rankingRun = "run_${stamp}_sixteen_factor_daily"

$status = [ordered]@{
    schemaVersion = "sixteen_factor_daily_runner_v1"
    status = "running_research_only"
    startedAt = (Get-Date).ToString("o")
    expectedSignalDate = $date
    pitAdjustedData = "not_run"
    ranking = "not_run"
    accountability = "not_run"
    choiceWatchlist = "not_run"
    rankingRun = $rankingRun
    rankingResult = $null
    exitCode = $null
    finishedAt = $null
    orders = @()
    automaticTradingChanges = @()
}

function Save-Status([string]$state, [int]$code) {
    $status.status = $state
    $status.exitCode = $code
    $status.finishedAt = (Get-Date).ToString("o")
    $status | ConvertTo-Json -Depth 6 | Set-Content -Path $statusPath -Encoding UTF8
}

# The model is intentionally built only from the isolated PIT-adjusted archive.  Run
# one parallel pass, then retry only failed shards serially so BaoStock is not asked to
# accept another burst of eight logins when a single connection was rejected.
$pitRun = "daily_${stamp}_parallel"
& powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `
  scripts\run_ashare_pit_adjusted_backfill.ps1 -ShardCount 8 -RunId $pitRun -EndDate $date `
  *>&1 | Tee-Object -FilePath $logPath -Append
$pitPassed = $LASTEXITCODE -eq 0
if (-not $pitPassed) {
    $launcher = Join-Path $root ("logs\ashare_pit_adjusted\" + $pitRun + "\launcher_manifest.json")
    try {
        $failedShards = @(
            (Get-Content -Raw $launcher | ConvertFrom-Json).workers |
              Where-Object { $_.exitCode -ne 0 } |
              ForEach-Object { [int]$_.shardIndex }
        )
    }
    catch {
        $failedShards = @()
    }
    if ($failedShards.Count -gt 0) {
        $serialPassed = $true
        foreach ($shard in $failedShards) {
            $retryRun = "daily_${stamp}_serial_shard${shard}"
            & py -3.13 scripts\collect_ashare_pit_adjusted_baostock.py `
              --mode backfill --shard-count 8 --shard-index $shard `
              --run-id $retryRun --end-date $date --progress-every 50 `
              *>&1 | Tee-Object -FilePath $logPath -Append
            if ($LASTEXITCODE -ne 0) {
                $serialPassed = $false
            }
        }
        if ($serialPassed) {
            & py -3.13 scripts\collect_ashare_pit_adjusted_baostock.py --mode audit `
              *>&1 | Tee-Object -FilePath $logPath -Append
            $pitPassed = $LASTEXITCODE -eq 0
        }
    }
}
if (-not $pitPassed) {
    $status.pitAdjustedData = "failed_closed_after_parallel_and_serial_retry"
    Save-Status "failed_closed" 20
    exit 20
}
$status.pitAdjustedData = "ok"

& py -3.13 scripts\generate_sixteen_factor_interaction_top10_v1.py --run-id $rankingRun `
  *>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    $status.ranking = "failed_closed_$LASTEXITCODE"
    Save-Status "failed_closed" 21
    exit 21
}
$resultPath = Join-Path $root ("outputs\edge_research\sixteen_factor_interaction_ranking_v1\" + $rankingRun + "\result.json")
if (-not (Test-Path $resultPath)) {
    $status.ranking = "missing_result"
    Save-Status "failed_closed" 22
    exit 22
}
try {
    $result = Get-Content -Raw $resultPath | ConvertFrom-Json
}
catch {
    $status.ranking = "invalid_result_json"
    Save-Status "failed_closed" 23
    exit 23
}
if ($result.signalDate -ne $date) {
    $status.ranking = "stale_signal_date_" + $result.signalDate
    Save-Status "failed_closed" 24
    exit 24
}
$status.ranking = "ok"
$status.rankingResult = $resultPath

& py -3.13 scripts\review_sixteen_factor_daily_v1.py `
  *>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    $status.accountability = "failed_closed_$LASTEXITCODE"
    Save-Status "failed_closed" 25
    exit 25
}
$status.accountability = "ok"

& py -3.13 scripts\export_eastmoney_choice_top10.py `
  --config configs\research\eastmoney_choice_sixteen_factor_top10_watchlist.json `
  --source $resultPath *>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    $status.choiceWatchlist = "failed_closed_$LASTEXITCODE"
    Save-Status "failed_closed" 26
    exit 26
}
$status.choiceWatchlist = "ok"
Save-Status "completed_research_only" 0
exit 0
