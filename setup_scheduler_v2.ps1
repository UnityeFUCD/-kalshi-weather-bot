# setup_scheduler_v2.ps1 -- Set up Phase 4/5 scheduled tasks
#
# New schedule:
#   1. Paper Loop:       5:30 AM MT daily -- continuous scanning until 4 PM MT
#   2. Evening Reconcile: 6:00 PM MT daily -- settle paper trades
#   3. Daily Runner:      8:00 PM MT daily -- GHCN refresh + calibration
#
# Run this as Administrator:
#   powershell -ExecutionPolicy Bypass -File setup_scheduler_v2.ps1

$botDir = "C:\Users\fycin\Desktop\kelshi-weather-bot\-kalshi-weather-bot"

Write-Host "============================================================"
Write-Host "KALSHI WEATHER BOT -- SCHEDULER SETUP (v2 / Phase 4+5)"
Write-Host "============================================================"
Write-Host ""

# --- 1. Remove old Morning Paper (single-shot) task ---
Write-Host "[1] Removing old 'Kalshi Morning Paper' task (single-shot)..."
try {
    schtasks /delete /tn "Kalshi Morning Paper" /f 2>$null
    Write-Host "    Removed old task."
} catch {
    Write-Host "    (not found or already removed)"
}

# --- 2. Create Paper Loop (continuous) ---
Write-Host "[2] Creating 'Kalshi Paper Loop' task (5:30 AM - 4:00 PM MT)..."
schtasks /create `
    /tn "Kalshi Paper Loop" `
    /tr "$botDir\paper_loop.bat" `
    /sc daily `
    /st 05:30 `
    /f
if ($LASTEXITCODE -eq 0) {
    Write-Host "    [OK] Paper loop scheduled for 5:30 AM daily"
    Write-Host "    Bot will scan every 2 min and auto-exit at 4:00 PM MT"
} else {
    Write-Host "    [FAIL] Could not create task -- try running as Administrator"
}

# --- 3. Evening Reconcile (keep existing) ---
Write-Host "[3] Ensuring 'Kalshi Evening Reconcile' task exists..."
schtasks /create `
    /tn "Kalshi Evening Reconcile" `
    /tr "$botDir\evening_reconcile.bat" `
    /sc daily `
    /st 18:00 `
    /f
if ($LASTEXITCODE -eq 0) {
    Write-Host "    [OK] Evening reconcile at 6:00 PM daily"
} else {
    Write-Host "    [FAIL] Could not create task"
}

# --- 4. Daily Runner (keep existing) ---
Write-Host "[4] Ensuring 'Kalshi Daily Runner' task exists..."
schtasks /create `
    /tn "Kalshi Daily Runner" `
    /tr "$botDir\daily_run.bat" `
    /sc daily `
    /st 20:00 `
    /f
if ($LASTEXITCODE -eq 0) {
    Write-Host "    [OK] Daily runner at 8:00 PM daily"
} else {
    Write-Host "    [FAIL] Could not create task"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "SCHEDULE SUMMARY"
Write-Host "============================================================"
Write-Host "  5:30 AM  Paper Loop     ~300 scans/day (every 2 min)"
Write-Host "  4:00 PM  (auto-exit)    Bot stops scanning"
Write-Host "  6:00 PM  Reconcile      Settle paper trades vs actuals"
Write-Host "  8:00 PM  Daily Runner   GHCN refresh + calibration"
Write-Host "============================================================"
Write-Host ""

Write-Host "Verifying tasks..."
schtasks /query /tn "Kalshi Paper Loop" /fo list 2>$null
schtasks /query /tn "Kalshi Evening Reconcile" /fo list 2>$null
schtasks /query /tn "Kalshi Daily Runner" /fo list 2>$null

Write-Host ""
Write-Host "Done. To run immediately: .\paper_loop.bat"
