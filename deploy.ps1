param(
    [string]$msg = "update"
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

Set-Location $ProjectDir

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
& "$ProjectDir\.venv\Scripts\python.exe" "app.py"