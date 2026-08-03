param(
    [Parameter(Mandatory = $true)]
    [string]$BackfillRunId,
    [int]$PollSeconds = 60,
    [string]$ResearchRunId = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackfillManifest = Join-Path $Root (
    "logs\ashare_pit_adjusted\{0}\launcher_manifest.json" -f $BackfillRunId
)
$ResearchScript = Join-Path $Root (
    "scripts\research_perception_xalpha_market_opportunity_v8.py"
)
$Override = Join-Path $Root (
    "configs\research\perception_xalpha_pit_adjusted_robustness_v9.json"
)
$PriceAudit = Join-Path $Root (
    "outputs\edge_research\ashare_pit_adjusted_data_v1\latest_data_audit.json"
)
if ([string]::IsNullOrWhiteSpace($ResearchRunId)) {
    $ResearchRunId = "run_{0}_pit_adjusted_v9_robustness" -f (
        Get-Date -Format "yyyyMMddTHHmmss"
    )
}
if ($ResearchRunId -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "ResearchRunId may contain only letters, digits, dot, underscore and dash"
}

$LogRoot = Join-Path $Root "logs\perception_xalpha_pit_adjusted_robustness"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$StatusPath = Join-Path $LogRoot ("{0}_status.json" -f $ResearchRunId)
$Stdout = Join-Path $LogRoot ("{0}_stdout.log" -f $ResearchRunId)
$Stderr = Join-Path $LogRoot ("{0}_stderr.log" -f $ResearchRunId)

function Write-ResearchStatus {
    param([string]$Status, [string]$Reason = "")
    [ordered]@{
        schemaVersion = "perception_xalpha_pit_adjusted_robustness_runner_v1"
        status = $Status
        reason = $Reason
        updatedAt = (Get-Date).ToString("o")
        backfillRunId = $BackfillRunId
        researchRunId = $ResearchRunId
        stdout = $Stdout
        stderr = $Stderr
        orders = @()
        automaticTradingChanges = @()
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $StatusPath
}

Write-ResearchStatus -Status "waiting_for_clean_data_research_only"
while ($true) {
    if (-not (Test-Path $BackfillManifest)) {
        Start-Sleep -Seconds ([math]::Max(5, $PollSeconds))
        continue
    }
    $Backfill = Get-Content -Raw $BackfillManifest | ConvertFrom-Json
    if ($Backfill.status -eq "running_research_only") {
        Start-Sleep -Seconds ([math]::Max(5, $PollSeconds))
        continue
    }
    if ($Backfill.status -ne "completed_research_only") {
        Write-ResearchStatus -Status "blocked_research_only" -Reason (
            "backfill status: {0}" -f $Backfill.status
        )
        exit 2
    }
    break
}

if (-not (Test-Path $PriceAudit)) {
    Write-ResearchStatus -Status "blocked_research_only" -Reason "price audit missing"
    exit 3
}
$Audit = Get-Content -Raw $PriceAudit | ConvertFrom-Json
if ($Audit.historicalResearchEligible -ne $true) {
    Write-ResearchStatus -Status "blocked_research_only" -Reason (
        "price audit gate failed; coverage={0}" -f $Audit.masterFileCoverage
    )
    exit 4
}

Write-ResearchStatus -Status "running_research_only"
& py -3.13 $ResearchScript `
    --research-data-override $Override `
    --run-id $ResearchRunId `
    1> $Stdout 2> $Stderr
$ResearchExitCode = $LASTEXITCODE
if ($ResearchExitCode -ne 0) {
    Write-ResearchStatus -Status "failed_research_only" -Reason (
        "research exit code: {0}" -f $ResearchExitCode
    )
    exit $ResearchExitCode
}
Write-ResearchStatus -Status "completed_research_only"
