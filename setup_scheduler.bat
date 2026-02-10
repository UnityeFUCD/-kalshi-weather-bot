@echo off
echo Setting up Kalshi Paper Trading scheduled tasks...

schtasks /create /tn "Kalshi Morning Paper" /tr "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot\morning_paper.bat" /sc daily /st 05:30 /f
if %errorlevel% equ 0 (
    echo [OK] Morning paper scan scheduled for 5:30 AM daily
) else (
    echo [FAIL] Could not create morning task -- try running as Administrator
)

schtasks /create /tn "Kalshi Evening Reconcile" /tr "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot\evening_reconcile.bat" /sc daily /st 18:00 /f
if %errorlevel% equ 0 (
    echo [OK] Evening reconcile scheduled for 6:00 PM daily
) else (
    echo [FAIL] Could not create evening task -- try running as Administrator
)

echo.
echo Verifying tasks...
schtasks /query /tn "Kalshi Morning Paper" /fo list
schtasks /query /tn "Kalshi Evening Reconcile" /fo list

pause
