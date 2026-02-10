@echo off
REM morning_scan.bat -- Run morning scan at 5:30 AM MT (7:30 AM ET)
REM Schedule this with Windows Task Scheduler
REM
REM To set up:
REM   1. Open Task Scheduler (search "Task Scheduler" in Start)
REM   2. Click "Create Basic Task"
REM   3. Name: "Kalshi Morning Scan"
REM   4. Trigger: Daily at 5:30 AM
REM   5. Action: Start a Program
REM   6. Program: C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot\morning_scan.bat
REM   7. Check "Open properties dialog" -> Settings -> "Run whether user is logged on or not"
REM
REM This does a SCAN ONLY (no trades). Check signals.txt to see what it found.
REM When you're ready for auto-trading, use morning_trade.bat instead.

cd /d C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot

echo [%date% %time%] Morning scan starting... >> logs\scheduler.log

py bot.py scan --once >> logs\scheduler.log 2>&1

echo [%date% %time%] Morning scan complete. Check signals.txt >> logs\scheduler.log
