$ErrorActionPreference = "Stop"

$Workspace = "C:\Users\XU XINGJIAN\Documents\Codex"
Set-Location -LiteralPath $Workspace

$env:PYTHONIOENCODING = "utf-8"
$UserKey = [Environment]::GetEnvironmentVariable("HT_APIKEY", "User")
if ($UserKey) {
    $env:HT_APIKEY = $UserKey
}

$LogDir = Join-Path $Workspace "outputs\paper_trading_agent"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "scheduled_status_refresh.log"

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $LogPath -Value "[$Timestamp] refresh start"

$Now = Get-Date
$Minutes = $Now.Hour * 60 + $Now.Minute
$IsWeekday = [int]$Now.DayOfWeek -ge 1 -and [int]$Now.DayOfWeek -le 5
$InMorning = $Minutes -ge (9 * 60 + 30) -and $Minutes -le (11 * 60 + 30)
$InAfternoon = $Minutes -ge (13 * 60) -and $Minutes -le (15 * 60)
if (-not ($IsWeekday -and ($InMorning -or $InAfternoon))) {
    Add-Content -Path $LogPath -Value "[$Timestamp] outside regular A-share session; skip refresh"
    exit 0
}

try {
    py -3.13 scripts\run_etf_agent_status.py --refresh 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    $Done = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $LogPath -Value "[$Done] refresh exit_code=$ExitCode"
    exit $ExitCode
}
catch {
    $ErrTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $LogPath -Value "[$ErrTime] refresh error: $($_.Exception.Message)"
    exit 1
}
