$ErrorActionPreference = "Stop"

# Install/update the market-data-only collector. The host uses Asia/Seoul time:
# 10:25 KST = 09:25 China time, allowing the process to warm up before the open.
$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Script = Join-Path $Root "scripts\collect_l2_depth.py"
$TaskName = "ETF L2 Depth Collector"
$Python = (Get-Command py.exe -ErrorAction Stop).Source

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-3.13 `"$Script`" --until 15:00 --poll-seconds 5"
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "10:25"
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Collect confirmed full-universe ETF T0 Sina five-level depth and coverage; market data only, no orders." `
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
