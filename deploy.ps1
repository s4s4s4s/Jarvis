param(
    [string]$msg = "update",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$Python = "$ProjectDir\.venv\Scripts\python.exe"
$Pip    = "$ProjectDir\.venv\Scripts\pip.exe"

Set-Location $ProjectDir

if (-not $SkipInstall) {
    Write-Host "`n[JARVIS] Installing dependencies..." -ForegroundColor Cyan
    & $Pip install -r requirements.txt --quiet
}

Write-Host "`n[JARVIS] Staging files..." -ForegroundColor Cyan
git add -A

$status = git status --porcelain
if ($status) {
    Write-Host "[JARVIS] Committing: $msg" -ForegroundColor Cyan
    git commit -m $msg
    Write-Host "[JARVIS] Pushing to main..." -ForegroundColor Cyan
    git push origin main
    Write-Host "[JARVIS] Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "[JARVIS] No changes to commit." -ForegroundColor Yellow
}

Write-Host "`n[JARVIS] Starting project..." -ForegroundColor Cyan
& $Python "app.py"
