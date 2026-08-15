param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("score", "settle", "report")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$Workspace = "C:\Users\XU XINGJIAN\Documents\Codex"
$Script = Join-Path $Workspace "scripts\run_stock_event_auction_forward_v1.py"
$LogDirectory = Join-Path $Workspace "logs\stock_event_auction_forward_v1"
$TradeDate = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId(
    [DateTimeOffset]::UtcNow,
    "China Standard Time"
).ToString("yyyy-MM-dd")
$Log = Join-Path $LogDirectory ("{0}_{1}.log" -f $TradeDate, $Mode)

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
Set-Location $Workspace
$env:PYTHONIOENCODING = "utf-8"

& py -3.13 $Script --mode $Mode --trade-date $TradeDate *>&1 |
    Tee-Object -FilePath $Log -Append
exit $LASTEXITCODE
