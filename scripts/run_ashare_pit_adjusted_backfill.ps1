param(
    [ValidateRange(1, 32)]
    [int]$ShardCount = 4,
    [string]$RunId = "",
    [string]$EndDate = "",
    [int]$ProgressEvery = 50,
    [ValidateRange(0, 100000)]
    [int]$MaxCodes = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Collector = Join-Path $Root "scripts\collect_ashare_pit_adjusted_baostock.py"
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMddTHHmmss"
}
if ($RunId -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "RunId may contain only letters, digits, dot, underscore and dash"
}

$LogRoot = Join-Path $Root "logs\ashare_pit_adjusted\$RunId"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$Workers = @()
for ($ShardIndex = 0; $ShardIndex -lt $ShardCount; $ShardIndex++) {
    $Stdout = Join-Path $LogRoot ("shard_{0:D2}_stdout.log" -f $ShardIndex)
    $Stderr = Join-Path $LogRoot ("shard_{0:D2}_stderr.log" -f $ShardIndex)
    $Arguments = @(
        "-3.13",
        ('"{0}"' -f $Collector),
        "--mode", "backfill",
        "--shard-count", "$ShardCount",
        "--shard-index", "$ShardIndex",
        "--run-id", "$RunId",
        "--progress-every", "$ProgressEvery"
    )
    if (-not [string]::IsNullOrWhiteSpace($EndDate)) {
        $Arguments += @("--end-date", $EndDate)
    }
    if ($MaxCodes -gt 0) {
        $Arguments += @("--max-codes", "$MaxCodes")
    }
    $Process = Start-Process -FilePath "py" `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru
    $Workers += [pscustomobject]@{
        shardIndex = $ShardIndex
        process = $Process
        pid = $Process.Id
        stdout = $Stdout
        stderr = $Stderr
    }
}

$InitialManifest = [ordered]@{
    schemaVersion = "ashare_pit_adjusted_backfill_launcher_v1"
    status = "running_research_only"
    runId = $RunId
    startedAt = (Get-Date).ToString("o")
    shardCount = $ShardCount
    workers = @($Workers | ForEach-Object {
        [ordered]@{
            shardIndex = $_.shardIndex
            pid = $_.pid
            stdout = $_.stdout
            stderr = $_.stderr
        }
    })
    orders = @()
    automaticTradingChanges = @()
}
$LauncherManifest = Join-Path $LogRoot "launcher_manifest.json"
$InitialManifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $LauncherManifest

$WorkerResults = @()
foreach ($Worker in $Workers) {
    $Worker.process.WaitForExit()
    $Worker.process.Refresh()
    $SummaryPath = Join-Path $Root (
        "outputs\edge_research\ashare_pit_adjusted_data_v1\collection_runs\{0}\shard_{1:D2}_of_{2:D2}_summary.json" -f `
            $RunId, $Worker.shardIndex, $ShardCount
    )
    $WorkerExitCode = 1
    if (Test-Path $SummaryPath) {
        try {
            $Summary = Get-Content -Raw $SummaryPath | ConvertFrom-Json
            if (@($Summary.failedSecurityIds).Count -eq 0) {
                $WorkerExitCode = 0
            }
        }
        catch {
            $WorkerExitCode = 1
        }
    }
    $WorkerResults += [ordered]@{
        shardIndex = $Worker.shardIndex
        pid = $Worker.pid
        exitCode = $WorkerExitCode
        summary = $SummaryPath
        stdout = $Worker.stdout
        stderr = $Worker.stderr
    }
}

$AuditStdout = Join-Path $LogRoot "audit_stdout.log"
$AuditStderr = Join-Path $LogRoot "audit_stderr.log"
& py -3.13 $Collector --mode audit 1> $AuditStdout 2> $AuditStderr
$AuditExitCode = $LASTEXITCODE

$AllWorkersPassed = -not ($WorkerResults | Where-Object { $_.exitCode -ne 0 })
$FinalManifest = [ordered]@{
    schemaVersion = "ashare_pit_adjusted_backfill_launcher_v1"
    status = if ($AllWorkersPassed -and $AuditExitCode -eq 0) { "completed_research_only" } else { "failed_research_only" }
    runId = $RunId
    startedAt = $InitialManifest.startedAt
    completedAt = (Get-Date).ToString("o")
    shardCount = $ShardCount
    workers = $WorkerResults
    auditExitCode = $AuditExitCode
    auditStdout = $AuditStdout
    auditStderr = $AuditStderr
    orders = @()
    automaticTradingChanges = @()
}
$FinalManifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $LauncherManifest

if ($FinalManifest.status -ne "completed_research_only") {
    exit 1
}
