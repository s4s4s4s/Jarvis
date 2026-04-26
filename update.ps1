<#
.SYNOPSIS
    Jarvis — pull latest from GitHub and launch.

.DESCRIPTION
    1. git pull origin main
    2. (optional) pip install -r requirements.txt
    3. python app.py

.PARAMETER SkipInstall
    Skip pip install step (faster if deps haven't changed).

.EXAMPLE
    .\update.ps1
    .\update.ps1 -SkipInstall
#>
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$Python     = "$ProjectDir\.venv\Scripts\python.exe"
$Pip        = "$ProjectDir\.venv\Scripts\pip.exe"
$Git        = "git"

Set-Location $ProjectDir

# ---- проверка виртуального окружения ---------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Host "[JARVIS] Виртуальное окружение не найдено. Создайте .venv:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# ---- git pull -----------------------------------------------------------
Write-Host ""
Write-Host "[JARVIS] Пуллю последние изменения с GitHub..." -ForegroundColor Cyan

# Сохраняем SHA до пулла чтобы показать новые коммиты
$sha_before = & $Git rev-parse HEAD 2>$null

try {
    & $Git pull origin main --ff-only
} catch {
    Write-Host "[JARVIS] git pull вернул ошибку:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Подсказка: если есть локальные изменения, сперва запусти deploy.ps1" -ForegroundColor Yellow
    exit 1
}

$sha_after = & $Git rev-parse HEAD 2>$null

if ($sha_before -ne $sha_after) {
    Write-Host "[JARVIS] Обновлено! Новые коммиты:" -ForegroundColor Green
    & $Git log --oneline "${sha_before}..${sha_after}"
} else {
    Write-Host "[JARVIS] Уже актуально, новых коммитов нет." -ForegroundColor Yellow
}

# ---- pip install (unless skipped) --------------------------------------
if (-not $SkipInstall) {
    Write-Host ""
    Write-Host "[JARVIS] Проверяю зависимости..." -ForegroundColor Cyan
    & $Pip install -r requirements.txt --quiet
    Write-Host "[JARVIS] Зависимости OK." -ForegroundColor Green
}

# ---- запуск -------------------------------------------------------------------
Write-Host ""
Write-Host "[JARVIS] Запускаю Jarvis..." -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor DarkGray
& $Python "app.py"
