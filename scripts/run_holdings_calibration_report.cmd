@echo off
cd /d "%~dp0.."
if not exist "outputs\holdings_calibration" mkdir "outputs\holdings_calibration"
py -3.13 scripts\run_holdings_calibration_report.py >> outputs\holdings_calibration\scheduled_holdings_calibration.log 2>&1
