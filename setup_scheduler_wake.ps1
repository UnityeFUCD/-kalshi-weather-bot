# Enable wake-from-sleep and run-on-missed for both Kalshi tasks
# Run this as: powershell -ExecutionPolicy Bypass -File setup_scheduler_wake.ps1

$tasks = @("Kalshi Morning Paper", "Kalshi Evening Reconcile")

foreach ($name in $tasks) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "[SKIP] $name not found -- run setup_scheduler.bat first"
        continue
    }

    $settings = $task.Settings
    $settings.WakeToRun = $true
    $settings.StartWhenAvailable = $true
    $settings.RunOnlyIfIdle = $false

    Set-ScheduledTask -TaskName $name -Settings $settings | Out-Null
    Write-Host "[OK] $name -- wake-to-run ON, run-if-missed ON"
}

Write-Host ""
Write-Host "Done. Your PC will wake from SLEEP to run these tasks."
Write-Host "Note: This does NOT work if the PC is fully shut down -- use Sleep instead."
