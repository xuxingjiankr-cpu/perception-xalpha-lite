$ErrorActionPreference = "Stop"

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Script = Join-Path $Root "scripts\research_perception_xalpha.py"
$Config = Join-Path $Root "configs\research\perception_xalpha_mvp.json"
$LogRoot = Join-Path $Root "outputs\edge_research\perception_xalpha\logs"
$Log = Join-Path $LogRoot ("weekly_{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$mutex = New-Object System.Threading.Mutex($false, "Local\CodexPerceptionXAlphaWeekly")
$acquired = $false
try {
    $acquired = $mutex.WaitOne(0)
    if (-not $acquired) {
        Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] skipped: research run already active" -Encoding UTF8
        exit 0
    }
    Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] start research-only perception alpha" -Encoding UTF8
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
