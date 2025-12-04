# Restart ZTE Server
Write-Host "=== Restarting ZTE ===" -ForegroundColor Cyan

# Kill all Python processes
Write-Host "[1/3] Stopping all Python processes..." -ForegroundColor Yellow
taskkill /F /IM python.exe 2>&1 | Out-Null
Start-Sleep -Seconds 2
Write-Host "      Done!" -ForegroundColor Green

# Verify port is free
Write-Host "[2/3] Checking port 5001..." -ForegroundColor Yellow
$portCheck = netstat -ano | Select-String ":5001"
if ($portCheck) {
    Write-Host "      WARNING: Port still in use!" -ForegroundColor Red
    Write-Host "      $portCheck" -ForegroundColor Red
} else {
    Write-Host "      Port 5001 is FREE!" -ForegroundColor Green
}

# Start server
Write-Host "[3/3] Starting ZTE API Server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  ZTE Server Starting on Port 5001" -ForegroundColor White
Write-Host "  URL: http://localhost:5001" -ForegroundColor White
Write-Host "  Docs: http://localhost:5001/docs" -ForegroundColor White
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

cd C:\AI-ALL-PRO\ZERO-TRADING-EXPERT
python api_server_trading.py


