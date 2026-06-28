$ErrorActionPreference = "Stop"

# Host time is Asia/Seoul: 10:20 KST = 09:20 China time. The process archives
# official SSE PCFs before the open, then polls vendor IOPV snapshots each minute.
$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Script = Join-Path $Root "scripts\collect_etf_iopv_pcf.py"
$TaskName = "ETF IOPV PCF Collector"
$Python = (Get-Command py.exe -ErrorAction Stop).Source

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-3.13 `"$Script`" --mode both --until 15:00 --poll-seconds 60"
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "10:20"
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Archive official SSE ETF PCFs and collect point-in-time vendor IOPV/premium snapshots; research only." `
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
