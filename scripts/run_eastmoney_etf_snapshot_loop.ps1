$ErrorActionPreference = "Stop"

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Log = Join-Path $Root "outputs\eastmoney_full_market\etf_snapshot_loop.log"
$PidFile = Join-Path $Root "outputs\eastmoney_full_market\etf_snapshot_loop.pid"
$IntervalSeconds = 60

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

function Write-CollectorLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $Log -Value "[$stamp] $Message" -Encoding UTF8
}

function In-AshareSessionKst {
    $now = Get-Date
    if ($now.DayOfWeek -eq "Saturday" -or $now.DayOfWeek -eq "Sunday") {
        return $false
    }
    $mins = $now.Hour * 60 + $now.Minute
    # Host clock is Korea time. A-share sessions are 09:30-11:30 and 13:00-15:00 China time,
    # i.e. 10:30-12:30 and 14:00-16:00 Korea time.
    $morning = ($mins -ge (10 * 60 + 30)) -and ($mins -le (12 * 60 + 30))
    $afternoon = ($mins -ge (14 * 60)) -and ($mins -le (16 * 60))
    return ($morning -or $afternoon)
}

[System.Diagnostics.Process]::GetCurrentProcess().Id | Set-Content -Path $PidFile -Encoding ASCII
Write-CollectorLog "eastmoney ETF snapshot loop start pid=$([System.Diagnostics.Process]::GetCurrentProcess().Id) interval=${IntervalSeconds}s"

while ($true) {
    try {
        if (In-AshareSessionKst) {
            $started = Get-Date
            Write-CollectorLog "snapshot start"
            py -3.13 .\scripts\collect_eastmoney_full_market.py snapshot --scope etf --page-size 50 --page-retries 2 --retry-sleep-seconds 0.3 --only-session 2>&1 |
                ForEach-Object { Write-CollectorLog $_ }
            $elapsed = [int]((Get-Date) - $started).TotalSeconds
            Write-CollectorLog "snapshot end elapsed=${elapsed}s"
            $sleep = [Math]::Max(5, $IntervalSeconds - $elapsed)
            Start-Sleep -Seconds $sleep
        }
        else {
            Write-CollectorLog "outside A-share session; sleep"
            Start-Sleep -Seconds 60
        }
    }
    catch {
        Write-CollectorLog "snapshot loop error: $($_.Exception.Message)"
        Start-Sleep -Seconds 60
    }
}
