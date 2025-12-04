# Test ZTE API
Write-Host "=== Testing ZTE API ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "[1] Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:5001/api/health" -Method Get
    Write-Host "✅ Server is healthy!" -ForegroundColor Green
    $health | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Health check failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 2: Memory Stats
Write-Host "[2] Memory Stats..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Uri "http://localhost:5001/api/memory/stats" -Method Get
    Write-Host "✅ Memory stats retrieved!" -ForegroundColor Green
    $stats | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ Stats failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 3: Sentiment Analysis (NVDA)
Write-Host "[3] Sentiment Analysis for NVDA..." -ForegroundColor Yellow
try {
    $sentiment = Invoke-RestMethod -Uri "http://localhost:5001/api/sentiment/NVDA" -Method Get
    Write-Host "✅ Sentiment retrieved!" -ForegroundColor Green
    Write-Host "   Symbol: $($sentiment.symbol)" -ForegroundColor White
    Write-Host "   Score: $($sentiment.score)" -ForegroundColor White
    Write-Host "   Label: $($sentiment.label)" -ForegroundColor White
    Write-Host "   News Count: $($sentiment.news_count)" -ForegroundColor White
} catch {
    Write-Host "❌ Sentiment failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 4: Analysis Request (NVDA)
Write-Host "[4] Full Analysis for NVDA..." -ForegroundColor Yellow
$body = @{
    symbol = "NVDA"
    price = 495.50
    atr = 8.5
    score = 85
    signals = @("MA_CROSS", "VOLUME", "RSI")
    context = "Testing ZTE API"
    prices = @(490, 492, 495, 498, 495.50)
    highs = @(492, 494, 497, 500, 498)
    lows = @(488, 490, 493, 496, 493)
    volumes = @(1000000, 1200000, 1500000, 1800000, 2000000)
} | ConvertTo-Json

try {
    $analysis = Invoke-RestMethod -Uri "http://localhost:5001/api/analyze" -Method Post -Body $body -ContentType "application/json"
    Write-Host "✅ Analysis complete!" -ForegroundColor Green
    Write-Host "   Action: $($analysis.action)" -ForegroundColor White
    Write-Host "   Confidence: $([math]::Round($analysis.confidence * 100, 1))%" -ForegroundColor White
    Write-Host "   Reasoning: $($analysis.reasoning.Substring(0, [Math]::Min(100, $analysis.reasoning.Length)))..." -ForegroundColor White
} catch {
    Write-Host "❌ Analysis failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "👀 Check your ZTE terminal (Terminal 19) - you should see API request logs!" -ForegroundColor Yellow


