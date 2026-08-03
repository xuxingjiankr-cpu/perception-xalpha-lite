param(
    [string]$BackfillRunId = "",
    [string]$PrerequisiteResearchRunId = "",
    [int]$PollSeconds = 60,
    [string]$EvolutionRunId = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResearchScript = Join-Path $Root (
    "scripts\research_perception_xalpha_autonomous.py"
)
$Config = Join-Path $Root (
    "configs\research\perception_xalpha_all_ashares_v10_pit_adjusted.json"
)
$PriceAudit = Join-Path $Root (
    "outputs\edge_research\ashare_pit_adjusted_data_v1\latest_data_audit.json"
)
$FundamentalSummary = Join-Path $Root (
    "data\market\ashare_research\fundamentals_pit\latest_collection_summary.json"
)
$StateDirectory = Join-Path $Root (
    "outputs\edge_research\perception_xalpha_all_ashares_v10_pit_adjusted\state"
)
if ([string]::IsNullOrWhiteSpace($EvolutionRunId)) {
    $EvolutionRunId = "run_{0}_pit_adjusted_v10_evolution" -f (
        Get-Date -Format "yyyyMMddTHHmmss"
    )
}
if ($EvolutionRunId -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "EvolutionRunId may contain only letters, digits, dot, underscore and dash"
}

$LogRoot = Join-Path $Root "logs\perception_xalpha_pit_adjusted_evolution"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $StateDirectory | Out-Null
$StatusPath = Join-Path $LogRoot ("{0}_status.json" -f $EvolutionRunId)
$Stdout = Join-Path $LogRoot ("{0}_stdout.log" -f $EvolutionRunId)
$Stderr = Join-Path $LogRoot ("{0}_stderr.log" -f $EvolutionRunId)
$Mutex = [System.Threading.Mutex]::new(
    $false,
    "Local\PerceptionXAlphaPITAdjustedV10Evolution"
)
$OwnsMutex = $false

function Write-ResearchStatus {
    param(
        [string]$Status,
        [string]$Reason = "",
        [string]$CycleStatus = ""
    )
    [ordered]@{
        schemaVersion = "perception_xalpha_pit_adjusted_evolution_runner_v1"
        status = $Status
        reason = $Reason
        cycleStatus = $CycleStatus
        updatedAt = (Get-Date).ToString("o")
        backfillRunId = $BackfillRunId
        prerequisiteResearchRunId = $PrerequisiteResearchRunId
        evolutionRunId = $EvolutionRunId
        config = $Config
        stdout = $Stdout
        stderr = $Stderr
        orders = @()
        automaticTradingChanges = @()
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $StatusPath
}

function Wait-ForStatus {
    param(
        [string]$Path,
        [string[]]$WaitingStates,
        [string]$CompletedState,
        [string]$Label
    )
    while ($true) {
        if (-not (Test-Path $Path)) {
            Start-Sleep -Seconds ([math]::Max(5, $PollSeconds))
            continue
        }
        $Payload = Get-Content -Raw $Path | ConvertFrom-Json
        if ($WaitingStates -contains [string]$Payload.status) {
            Start-Sleep -Seconds ([math]::Max(5, $PollSeconds))
            continue
        }
        if ([string]$Payload.status -ne $CompletedState) {
            throw ("{0} status: {1}" -f $Label, $Payload.status)
        }
        return
    }
}

try {
    $OwnsMutex = $Mutex.WaitOne(0)
    if (-not $OwnsMutex) {
        Write-ResearchStatus `
            -Status "skipped_active_runner_research_only" `
            -Reason "another V10 evolution runner is active"
        exit 0
    }
    Write-ResearchStatus -Status "waiting_for_prerequisites_research_only"

    if (-not [string]::IsNullOrWhiteSpace($BackfillRunId)) {
        $BackfillManifest = Join-Path $Root (
            "logs\ashare_pit_adjusted\{0}\launcher_manifest.json" -f $BackfillRunId
        )
        Wait-ForStatus `
            -Path $BackfillManifest `
            -WaitingStates @("running_research_only") `
            -CompletedState "completed_research_only" `
            -Label "price backfill"
    }

    if (-not [string]::IsNullOrWhiteSpace($PrerequisiteResearchRunId)) {
        $PrerequisiteStatus = Join-Path $Root (
            "logs\perception_xalpha_pit_adjusted_robustness\{0}_status.json" -f `
                $PrerequisiteResearchRunId
        )
        Wait-ForStatus `
            -Path $PrerequisiteStatus `
            -WaitingStates @(
                "waiting_for_clean_data_research_only",
                "running_research_only"
            ) `
            -CompletedState "completed_research_only" `
            -Label "frozen V9 robustness audit"
    }

    if (-not (Test-Path $PriceAudit)) {
        throw "price audit missing"
    }
    $Audit = Get-Content -Raw $PriceAudit | ConvertFrom-Json
    if ($Audit.historicalResearchEligible -ne $true) {
        throw ("price audit gate failed; coverage={0}" -f $Audit.masterFileCoverage)
    }
    if (-not (Test-Path $FundamentalSummary)) {
        throw "fundamental collection summary missing"
    }
    $Fundamentals = Get-Content -Raw $FundamentalSummary | ConvertFrom-Json
    if (
        [double]$Fundamentals.coverageOfCurrentMaster -lt 0.98 -or
        [int]$Fundamentals.failedThisRun -ne 0
    ) {
        throw (
            "fundamental gate failed; coverage={0}; failures={1}" -f
                $Fundamentals.coverageOfCurrentMaster,
                $Fundamentals.failedThisRun
        )
    }
    if (Test-Path (Join-Path $StateDirectory "research.lock")) {
        Write-ResearchStatus `
            -Status "skipped_active_cycle_research_only" `
            -Reason "the autonomous engine state lock is already held"
        exit 0
    }

    Write-ResearchStatus -Status "running_research_only"
    $ResearchArguments = @(
        "-3.13",
        ('"{0}"' -f $ResearchScript),
        "--config",
        ('"{0}"' -f $Config),
        "run"
    )
    # Keep native stderr (warnings and diagnostics) in its log. The process
    # exit code, not the mere presence of stderr, controls the fail-closed gate.
    $ResearchProcess = Start-Process `
        -FilePath "py" `
        -ArgumentList $ResearchArguments `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru `
        -Wait
    $ResearchExitCode = $ResearchProcess.ExitCode
    if ($ResearchExitCode -ne 0) {
        throw ("research exit code: {0}" -f $ResearchExitCode)
    }
    $CycleStatus = "completed_research_only"
    try {
        $Result = Get-Content -Raw $Stdout | ConvertFrom-Json
        if ($Result.status -eq "no_new_data") {
            $CycleStatus = "no_new_data"
        }
        elseif ($Result.status) {
            $CycleStatus = [string]$Result.status
        }
    }
    catch {
        $CycleStatus = "completed_research_only"
    }
    Write-ResearchStatus `
        -Status "completed_research_only" `
        -CycleStatus $CycleStatus
}
catch {
    Write-ResearchStatus -Status "blocked_or_failed_research_only" -Reason $_.Exception.Message
    exit 2
}
finally {
    if ($OwnsMutex) {
        $Mutex.ReleaseMutex()
    }
    $Mutex.Dispose()
}
