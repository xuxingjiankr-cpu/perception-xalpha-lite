$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
$LogRoot = Join-Path $Root "logs\fundamental_mechanism_forward"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$Day = Get-Date -Format "yyyy-MM-dd"
$Log = Join-Path $LogRoot "fundamental_mechanism_forward_$Day.log"

Push-Location $Root
try {
    & py -3.13 scripts\research_fundamental_mechanism_families.py --mode forward *>> $Log
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
