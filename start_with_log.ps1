# Start ZTE with logging
$logFile = "C:\AI-ALL-PRO\ZERO-TRADING-EXPERT\startup.log"

"=== ZTE Startup Log ===" | Out-File $logFile
"Time: $(Get-Date)" | Out-File $logFile -Append

# Check port
"Checking port 5001..." | Out-File $logFile -Append
try {
    $port5001 = Get-NetTCPConnection -LocalPort 5001 -ErrorAction Stop
    "Port 5001 is in use by PID: $($port5001.OwningProcess)" | Out-File $logFile -Append
    "Killing process..." | Out-File $logFile -Append
    Stop-Process -Id $port5001.OwningProcess -Force -ErrorAction Stop
    Start-Sleep -Seconds 2
    "Process killed successfully" | Out-File $logFile -Append
} catch {
    "Port 5001 is free (or check failed): $_" | Out-File $logFile -Append
}

# Change to directory
"Changing to ZTE directory..." | Out-File $logFile -Append
Set-Location "C:\AI-ALL-PRO\ZERO-TRADING-EXPERT"

# Check Python
"Checking Python..." | Out-File $logFile -Append
try {
    $pyVersion = python --version 2>&1
    "Python version: $pyVersion" | Out-File $logFile -Append
} catch {
    "Python check failed: $_" | Out-File $logFile -Append
}

# Check if file exists
if (Test-Path "api_server_trading.py") {
    "File api_server_trading.py exists" | Out-File $logFile -Append
} else {
    "ERROR: api_server_trading.py not found!" | Out-File $logFile -Append
    exit 1
}

# Start server and redirect output
"Starting server..." | Out-File $logFile -Append
python api_server_trading.py 2>&1 | Tee-Object -FilePath $logFile -Append


