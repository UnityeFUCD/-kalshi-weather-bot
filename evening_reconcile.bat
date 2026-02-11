@echo off
REM Evening reconciliation -- run at 6:00 PM MT via Task Scheduler
cd /d "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot"
set KALSHI_API_KEY_ID=37f5c54f-05a2-4e71-aae3-e0f426de5c98
"C:\Users\fycin\AppData\Local\Programs\Python\Python313\python.exe" bot.py reconcile
