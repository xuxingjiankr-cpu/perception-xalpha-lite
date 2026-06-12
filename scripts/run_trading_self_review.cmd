@echo off
cd /d "%~dp0.."
if not exist "outputs\trading_self_review" mkdir "outputs\trading_self_review"
py -3.13 scripts\run_trading_self_review.py >> outputs\trading_self_review\scheduled_self_review.log 2>&1
