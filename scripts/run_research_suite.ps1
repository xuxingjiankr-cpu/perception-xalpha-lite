$ErrorActionPreference = "Stop"

# Daily offline research suite (read-only; NO orders, NO broker calls). Runs after the
# A-share close so each new trading day's full-market snapshots are included, letting the
# diagnostic_only signal studies accumulate toward the sample sizes their promotion gates
# require. Each script reads the snapshot history and rewrites its own summary JSON.

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Log = Join-Path $Root "outputs\research_suite\research_suite.log"
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

function Write-SuiteLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $Log -Value "[$stamp] $Message" -Encoding UTF8
}

Write-SuiteLog "research suite start"
$scripts = @(
    "research_early_entry.py",
    "research_lead_lag.py",
    "research_regime.py",
    "research_stops.py",
    "research_sizing.py"
)
foreach ($s in $scripts) {
    $path = Join-Path $Root "scripts\$s"
    try {
        Write-SuiteLog "run $s"
        py -3.13 $path 2>&1 | ForEach-Object { Write-SuiteLog $_ }
    }
    catch {
        Write-SuiteLog "ERROR $s : $($_.Exception.Message)"
    }
}
Write-SuiteLog "research suite end"
