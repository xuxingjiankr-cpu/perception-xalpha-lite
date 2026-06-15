$ErrorActionPreference = "Stop"

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Log = Join-Path $Root "outputs\eastmoney_full_market\etf_june_backfill_resume.log"
$UniverseFile = Join-Path $Root "data\market\eastmoney\universe\eastmoney_universe_etf_20260615.jsonl"

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

function Write-BackfillLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $Log -Value "[$stamp] $Message" -Encoding UTF8
}

if (-not (Test-Path $UniverseFile)) {
    Write-BackfillLog "universe file missing: $UniverseFile"
    throw "universe file missing: $UniverseFile"
}

Write-BackfillLog "ETF June minute backfill resume start"
try {
    # Slow resume mode: skips existing files, uses Eastmoney only, and never calls Huatai.
    py -3.13 .\scripts\collect_eastmoney_full_market.py backfill-minute --scope etf --month 2026-06 --universe-file $UniverseFile --sleep-seconds 0.8 2>&1 |
        ForEach-Object { Write-BackfillLog $_ }
    Write-BackfillLog "ETF June minute backfill resume end"
}
catch {
    Write-BackfillLog "ETF June minute backfill resume failed: $($_.Exception.Message)"
    throw
}
