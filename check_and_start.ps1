# Check and Start ZTE
Write-Host "=== ZTE System Check ===" -ForegroundColor Cyan

# Check if port 5001 is in use
$port5001 = Get-NetTCPConnection -LocalPort 5001 -ErrorAction SilentlyContinue
if ($port5001) {
    Write-Host "[INFO] Port 5001 is already in use - stopping process..." -ForegroundColor Yellow
    $processId = $port5001.OwningProcess
    Stop-Process -Id $processId -Force
    Start-Sleep -Seconds 2
    Write-Host "[OK] Process stopped" -ForegroundColor Green
} else {
    Write-Host "[OK] Port 5001 is free" -ForegroundColor Green
}

# Check Python
Write-Host "`n[INFO] Checking Python..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion" -ForegroundColor White

# Check if we're in the right directory
$currentDir = Get-Location
Write-Host "`n[INFO] Current directory: $currentDir" -ForegroundColor Cyan

# Check if api_server_trading.py exists
if (Test-Path "api_server_trading.py") {
    Write-Host "[OK] api_server_trading.py found" -ForegroundColor Green
} else {
    Write-Host "[ERROR] api_server_trading.py not found!" -ForegroundColor Red
    exit 1
}

# Start the server
Write-Host "`n[INFO] Starting ZTE API Server on port 5001..." -ForegroundColor Cyan
Write-Host "[INFO] URL: http://localhost:5001" -ForegroundColor White
Write-Host "[INFO] Docs: http://localhost:5001/docs" -ForegroundColor White
Write-Host "`nServer output:" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Yellow

# Run the server
python api_server_trading.py


