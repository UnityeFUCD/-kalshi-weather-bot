@echo off
REM Morning paper trading scan -- run at 5:30 AM MT via Task Scheduler
cd /d "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot"
"C:\Users\fycin\AppData\Local\Programs\Python\Python313\python.exe" bot.py paper --once
