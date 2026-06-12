$ErrorActionPreference = "Stop"

$Root = "C:\Users\XU XINGJIAN\Documents\Codex"
$Log = Join-Path $Root "outputs\paper_trading_agent\scheduled_execute.log"
$Config = Join-Path $Root "configs\etf_paper_trading_agent_execute.json"

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

Write-AgentLog "paper execute start"
try {
    py -3.13 .\scripts\run_etf_paper_trading_agent.py --config $Config --mode paper_execute --execute 2>&1 |
        ForEach-Object { Write-AgentLog $_ }
    Write-AgentLog "paper execute end"
}
catch {
    Write-AgentLog "paper execute failed: $($_.Exception.Message)"
    throw
}
