$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

& py -3.13 scripts\export_eastmoney_choice_top10.py --refresh-source
exit $LASTEXITCODE
