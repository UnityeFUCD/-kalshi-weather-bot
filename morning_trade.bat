@echo off
REM morning_trade.bat -- Auto-trade at 5:30 AM MT (7:30 AM ET)
REM 
REM WARNING: This places REAL trades with REAL money.
REM Only use this after you've watched scan mode for several days
REM and trust the signals.
REM
REM Safety: max $5/trade, max $15 exposure, max 3 positions
REM
REM To set up in Task Scheduler:
REM   1. Create Basic Task -> "Kalshi Morning Trade"
REM   2. Trigger: Daily at 5:30 AM
REM   3. Action: Start a Program
REM   4. Program: C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot\morning_trade.bat

cd /d C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot

echo ================================================== >> logs\scheduler.log
echo [%date% %time%] LIVE TRADE starting... >> logs\scheduler.log
echo ================================================== >> logs\scheduler.log

py bot.py live --once >> logs\scheduler.log 2>&1

echo [%date% %time%] LIVE TRADE complete. Check signals.txt >> logs\scheduler.log
