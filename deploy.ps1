<#
.SYNOPSIS
    Jarvis — commit, push to GitHub, then launch.

.PARAMETER msg
    Commit message (default: "update").

.PARAMETER SkipInstall
    Skip pip install step.

.PARAMETER NoPush
    Stage + commit locally without pushing to GitHub.

.PARAMETER NoRun
    Push only, don't start app.py afterwards.

.EXAMPLE
    .\deploy.ps1
    .\deploy.ps1 -msg "фикс таймеров"
    .\deploy.ps1 -SkipInstall -NoRun
#>
param(
    [string]$msg          = "update",
    [switch]$SkipInstall,
    [switch]$NoPush,
    [switch]$NoRun
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$Python     = "$ProjectDir\.venv\Scripts\python.exe"
$Pip        = "$ProjectDir\.venv\Scripts\pip.exe"

Set-Location $ProjectDir

# ---- проверка виртуального окружения ---------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Host "[JARVIS] Виртуальное окружение не найдено. Создайте .venv:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# ---- pip install -------------------------------------------------------
if (-not $SkipInstall) {
    Write-Host ""
    Write-Host "[JARVIS] Устанавливаю зависимости..." -ForegroundColor Cyan
    & $Pip install -r requirements.txt --quiet
}

# ---- git add + commit + push -------------------------------------------
Write-Host ""
Write-Host "[JARVIS] Добавляю файлы..." -ForegroundColor Cyan
git add -A

$status = git status --porcelain
if ($status) {
    Write-Host "[JARVIS] Коммичу: $msg" -ForegroundColor Cyan
    git commit -m $msg

    if (-not $NoPush) {
        Write-Host "[JARVIS] Пушу в main..." -ForegroundColor Cyan
        git push origin main
        Write-Host "[JARVIS] Успешно запушено на GitHub!" -ForegroundColor Green
    } else {
        Write-Host "[JARVIS] Локальный коммит создан (-NoPush)." -ForegroundColor Yellow
    }
} else {
    Write-Host "[JARVIS] Нет изменений для коммита." -ForegroundColor Yellow
}

# ---- запуск -------------------------------------------------------------------
if (-not $NoRun) {
    Write-Host ""
    Write-Host "[JARVIS] Запускаю Jarvis..." -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor DarkGray
    & $Python "app.py"
}
