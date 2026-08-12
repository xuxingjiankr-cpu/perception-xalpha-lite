param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Url = "http://127.0.0.1:$Port/"
$Health = "http://127.0.0.1:$Port/health"
$Running = $false
try {
    $Response = Invoke-RestMethod -Uri $Health -TimeoutSec 1
    $Running = [bool]$Response.ok
}
catch {
    $Running = $false
}

if (-not $Running) {
    $LogRoot = Join-Path $Root "logs\stock_forecast_dashboard"
    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    $ServerScript = Join-Path $Root "scripts\stock_forecast_dashboard.py"
    Start-Process -FilePath "py" `
        -ArgumentList @("-3.13", ('"{0}"' -f $ServerScript), "serve", "--port", "$Port") `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogRoot "server_stdout.log") `
        -RedirectStandardError (Join-Path $LogRoot "server_stderr.log")
    for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $Response = Invoke-RestMethod -Uri $Health -TimeoutSec 1
            if ($Response.ok) {
                $Running = $true
                break
            }
        }
        catch {}
    }
}

if (-not $Running) {
    throw "Stock forecast dashboard did not become ready at $Url"
}

Start-Process $Url
Write-Output $Url
