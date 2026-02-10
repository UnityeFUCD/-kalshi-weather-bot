@echo off
REM paper_loop.bat -- Continuous paper trading during market hours
REM
REM Runs bot.py paper in LOOP mode (scans every 2 minutes)
REM Auto-exits at 4:00 PM MT (6 PM ET) via --until 16
REM
REM Schedule with Task Scheduler at 5:30 AM MT daily.
REM The bot will:
REM   - Scan markets every 2 minutes
REM   - Track forecast revisions (delta_tracker)
REM   - Pull METAR/ASOS observations
REM   - Compute confidence scores
REM   - Fetch ensemble spread
REM   - Paper trade when edge > dynamic threshold
REM   - Auto-exit at 4 PM MT
REM
REM ~300 scans per day (10.5 hours x ~30 scans/hour)

cd /d "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot"

echo [%date% %time%] Paper loop starting... >> logs\paper_loop.log

"C:\Users\fycin\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 bot.py paper --until 16 >> logs\paper_loop.log 2>&1

echo [%date% %time%] Paper loop ended. >> logs\paper_loop.log
