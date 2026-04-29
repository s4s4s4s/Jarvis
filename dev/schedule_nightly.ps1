# dev/schedule_nightly.ps1
# Регистрирует nightly_self_heal в Windows Task Scheduler
# Запуск: PowerShell -ExecutionPolicy Bypass -File dev\schedule_nightly.ps1

$TaskName   = "Jarvis-NightlySelfHeal"
$JarvisRoot = $env:JARVIS_ROOT
if (-not $JarvisRoot) { $JarvisRoot = "C:\jarvis" }

$PythonExe  = "$JarvisRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python).Source
}

$LogFile    = "$JarvisRoot\logs\nightly_scheduler.log"
$ScriptArgs = "-m dev.nightly_self_heal"

$Action  = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $ScriptArgs `
    -WorkingDirectory $JarvisRoot

# Каждую ночь в 03:00
$Trigger = New-ScheduledTaskTrigger -Daily -At "03:00"

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false

# Запись stdout/stderr в лог через обёртку
$WrapperAction = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c $PythonExe $ScriptArgs >> `"$LogFile`" 2>&1" `
    -WorkingDirectory $JarvisRoot

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[scheduler] Старая задача удалена."
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $WrapperAction `
    -Trigger $Trigger `
    -Settings $Settings `
    -RunLevel Highest `
    -Force | Out-Null

Write-Host "[scheduler] Задача '$TaskName' зарегистрирована — каждую ночь в 03:00."
Write-Host "[scheduler] Лог: $LogFile"
Write-Host "[scheduler] Запустить вручную: Start-ScheduledTask -TaskName '$TaskName'"
