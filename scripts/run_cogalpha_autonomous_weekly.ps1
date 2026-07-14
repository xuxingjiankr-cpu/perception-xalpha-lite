$ErrorActionPreference = "Stop"

# Weekly research-only alpha discovery. This runner does not import or invoke the
# trading agent. The Python program skips the run when the daily-bar fingerprint
# has not changed, so a scheduler retry cannot create duplicate research trials.
$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Script = Join-Path $Root "scripts\research_cogalpha_autonomous.py"
$Config = Join-Path $Root "configs\research\cogalpha_autonomous_v1.json"
$LogRoot = Join-Path $Root "outputs\edge_research\cogalpha_autonomous\logs"
$Log = Join-Path $LogRoot ("weekly_{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$mutex = New-Object System.Threading.Mutex($false, "Local\CodexCogAlphaAutonomousWeekly")
$acquired = $false
try {
    $acquired = $mutex.WaitOne(0)
    if (-not $acquired) {
        Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] skipped: another research run is active" -Encoding UTF8
        exit 0
    }
    Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] start research-only autonomous search" -Encoding UTF8
    $lines = & py -3.13 $Script --config $Config 2>&1
    $code = $LASTEXITCODE
    foreach ($line in $lines) {
        Add-Content -LiteralPath $Log -Value ([string]$line) -Encoding UTF8
        Write-Output $line
    }
    Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] exit_code=$code" -Encoding UTF8
    exit $code
}
finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
