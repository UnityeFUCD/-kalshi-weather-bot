@echo off
REM daily_run.bat -- One-command daily runner for calibration + scanning
REM Schedule this with Windows Task Scheduler at 8:00 PM MT (10 PM ET)
REM   Why 8 PM? GHCN observations typically update by evening. This gives
REM   the best chance of picking up today's settled data.
REM
REM To set up Task Scheduler:
REM   1. Open Task Scheduler (search "Task Scheduler" in Start)
REM   2. Click "Create Basic Task"
REM   3. Name: "Kalshi Daily Runner"
REM   4. Trigger: Daily at 8:00 PM
REM   5. Action: Start a Program
REM   6. Program/script: C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot\daily_run.bat
REM   7. Start in: C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot
REM   8. Finish -> open Properties -> Settings tab:
REM      - Check "Run whether user is logged on or not"
REM      - Check "Run with highest privileges"
REM
REM This runs: scan + GHCN refresh + calibration builder.
REM Safe to run manually anytime -- idempotent.

cd /d C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot
set KALSHI_API_KEY_ID=37f5c54f-05a2-4e71-aae3-e0f426de5c98

echo [%date% %time%] Daily runner starting... >> logs\daily_runner_scheduler.log

"C:\Users\fycin\AppData\Local\Programs\Python\Python313\python.exe" daily_runner.py >> logs\daily_runner_scheduler.log 2>&1

echo [%date% %time%] Daily runner complete. >> logs\daily_runner_scheduler.log
