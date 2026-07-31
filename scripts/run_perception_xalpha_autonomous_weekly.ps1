$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $root "logs\perception_xalpha_autonomous"
$lockPath = Join-Path $root "outputs\edge_research\perception_xalpha_autonomous\state\weekly.lock"
$python = "py"
$arguments = @("-3.13", (Join-Path $PSScriptRoot "research_perception_xalpha_autonomous.py"), "run")

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $lockPath) -Force | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd"
$logPath = Join-Path $logRoot "perception_xalpha_autonomous_$stamp.log"

try {
    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value "status=skipped_lock_busy"
    exit 0
}

try {
    $env:PYTHONIOENCODING = "utf-8"
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value "start_time=$([DateTimeOffset]::Now.ToString('o'))"
    & $python @arguments *>> $logPath
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value "exit_code=$exitCode"
    exit $exitCode
}
finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
}
