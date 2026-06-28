$ErrorActionPreference = "Stop"

# Host time is Asia/Seoul: 10:28 KST = 09:28 China time.
$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Script = Join-Path $Root "scripts\collect_etf_option_pressure.py"
$TaskName = "ETF Option Pressure Collector"
$Python = (Get-Command py.exe -ErrorAction Stop).Source

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-3.13 `"$Script`" --until 15:00 --poll-seconds 300"
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "10:28"
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Collect SSE ETF-option contract pressure, implied volatility, skew and unsigned gamma state; research only." `
    -Force | Out-Null

$Task = Get-ScheduledTask -TaskName $TaskName
[pscustomobject]@{
    task_name = $Task.TaskName
    state = $Task.State
    execute = $Task.Actions.Execute
    arguments = $Task.Actions.Arguments
    trigger_start = $Task.Triggers.StartBoundary
    trigger_days = ($Task.Triggers.DaysOfWeek -join ",")
} | ConvertTo-Json -Depth 4
