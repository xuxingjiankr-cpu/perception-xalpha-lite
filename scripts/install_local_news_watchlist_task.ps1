param(
    [string]$TaskName = "Local_ETF_News_Watchlist_Daily",
    [string]$LocalStartTime = "09:30",
    [switch]$KeepOpenAIWatchlistTask
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script = Join-Path $root "scripts\generate_local_news_etf_watchlist.py"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Generator script not found: $script"
}

$py = (Get-Command py.exe -ErrorAction Stop).Source
$action = New-ScheduledTaskAction `
    -Execute $py `
    -Argument "-3.13 `"$script`" --run-build" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At $LocalStartTime
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "No-model-API public RSS ETF news ranking; research-only observation pool input." `
    -Force | Out-Null

if (-not $KeepOpenAIWatchlistTask) {
    $legacy = Get-ScheduledTask -TaskName "ChatGPT_ETF_Watchlist_Daily" -ErrorAction SilentlyContinue
    if ($legacy) {
        Disable-ScheduledTask -TaskName "ChatGPT_ETF_Watchlist_Daily" | Out-Null
    }
}

$installed = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
[pscustomobject]@{
    TaskName = $installed.TaskName
    State = $installed.State
    NextRunTime = $info.NextRunTime
    Execute = $installed.Actions.Execute
    Arguments = $installed.Actions.Arguments
    OpenAIWatchlistTaskDisabled = -not $KeepOpenAIWatchlistTask
}
