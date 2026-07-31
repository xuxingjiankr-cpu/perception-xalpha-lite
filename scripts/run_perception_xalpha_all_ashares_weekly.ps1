$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

py -3.13 scripts\collect_ashare_research_daily.py --mode master
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.13 scripts\collect_ashare_research_daily.py --mode backfill --workers 4 --max-pages 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.13 scripts\collect_ashare_research_daily.py --mode audit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.13 scripts\research_perception_xalpha_autonomous.py `
  --config configs\research\perception_xalpha_all_ashares_v1.json run
exit $LASTEXITCODE
