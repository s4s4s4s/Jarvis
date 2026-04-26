# deploy.ps1 — git push + запуск Jarvis
param(
    [string]$msg = "update"
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot   # папка, где лежит deploy.ps1

Set-Location $ProjectDir

Write-Host "`n[JARVIS] Добавляю файлы..." -ForegroundColor Cyan
git add -A

$status = git status --porcelain
if ($status) {
    Write-Host "[JARVIS] Коммичу: $msg" -ForegroundColor Cyan
    git commit -m $msg
    Write-Host "[JARVIS] Пушу в main..." -ForegroundColor Cyan
    git push origin main
    Write-Host "[JARVIS] Успешно залито на GitHub!" -ForegroundColor Green
} else {
    Write-Host "[JARVIS] Нечего коммитить, репо чистое." -ForegroundColor Yellow
}

Write-Host "`n[JARVIS] Запускаю проект..." -ForegroundColor Cyan
python app.py