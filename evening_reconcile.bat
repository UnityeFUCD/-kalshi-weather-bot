@echo off
REM Evening reconciliation -- run at 6:00 PM MT via Task Scheduler
cd /d "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot"
"C:\Users\fycin\AppData\Local\Programs\Python\Python313\python.exe" bot.py reconcile
