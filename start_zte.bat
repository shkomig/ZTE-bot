@echo off
REM ================================================
REM Zero Trading Expert (ZTE) - Startup Script
REM ================================================
REM Port: 5001
REM ================================================

echo.
echo ========================================
echo   Zero Trading Expert (ZTE) v1.0.0
echo   Port: 5001
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

REM Check if Ollama is running
echo [INFO] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not running! Start Ollama first.
    echo [INFO] Continuing anyway (will use fallback model)...
)

REM Create logs directory if not exists
if not exist "logs" mkdir logs

echo.
echo [INFO] Starting ZTE API Server...
echo [INFO] URL: http://localhost:5001
echo [INFO] Docs: http://localhost:5001/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python api_server_trading.py

pause

