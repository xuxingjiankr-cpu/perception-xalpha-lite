$ErrorActionPreference = "Stop"

# Cross-border ETF premium research collector launcher (SHADOW / research only).
# Appends one aligned ETF+futures+CNH snapshot per run during the A-share session.
# No orders, no account/skill calls, no agent state. Safe to schedule freely.

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Log = Join-Path $Root "outputs\crossborder_premium\collect.log"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$Now = Get-Date
$Minutes = $Now.Hour * 60 + $Now.Minute
$IsWeekday = [int]$Now.DayOfWeek -ge 1 -and [int]$Now.DayOfWeek -le 5
$InMorning = $Minutes -ge (9 * 60 + 30) -and $Minutes -le (11 * 60 + 30)
$InAfternoon = $Minutes -ge (13 * 60) -and $Minutes -le (15 * 60)
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

if (-not ($IsWeekday -and ($InMorning -or $InAfternoon))) {
    Add-Content -Path $Log -Value "[$stamp] outside A-share session; skip" -Encoding UTF8
    exit 0
}

try {
    py -3.13 .\scripts\research_crossborder_premium.py 2>&1 | ForEach-Object { Add-Content -Path $Log -Value "[$stamp] $_" -Encoding UTF8 }
}
catch {
    Add-Content -Path $Log -Value "[$stamp] collect failed: $($_.Exception.Message)" -Encoding UTF8
}
