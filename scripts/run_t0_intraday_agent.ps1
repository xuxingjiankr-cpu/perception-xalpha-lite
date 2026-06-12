$ErrorActionPreference = "Stop"

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Log = Join-Path $Root "outputs\t0_intraday_agent\scheduled_t0_intraday.log"
$Config = Join-Path $Root "configs\t0_intraday_paper_agent.json"

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null

function Write-AgentLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $Log -Value "[$stamp] $Message" -Encoding UTF8
}

Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:HT_APIKEY) {
    $UserKey = [Environment]::GetEnvironmentVariable("HT_APIKEY", "User")
    if ($UserKey) {
        $env:HT_APIKEY = $UserKey
    }
    else {
        Write-AgentLog "HT_APIKEY missing; aborting"
        throw "HT_APIKEY environment variable is required"
    }
}

Write-AgentLog "t0 intraday start"
try {
    py -3.13 .\scripts\run_t0_intraday_agent.py --config $Config --execute 2>&1 |
        ForEach-Object { Write-AgentLog $_ }
    Write-AgentLog "t0 intraday end"
}
catch {
    Write-AgentLog "t0 intraday failed: $($_.Exception.Message)"
    throw
}
