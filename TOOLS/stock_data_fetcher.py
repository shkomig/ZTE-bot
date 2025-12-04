"""
Stock Data Fetcher for Zero Trading Expert
==========================================
Fetches real-time and historical market data.

Sources:
- Yahoo Finance (yfinance) - Free, reliable
- Can be extended for other APIs

Usage:
    from TOOLS.stock_data_fetcher import StockDataFetcher
    
    fetcher = StockDataFetcher()
    data = fetcher.get_stock_data("TSLA")
    indicators = fetcher.get_technical_indicators("TSLA")
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

YFINANCE_AVAILABLE = False
yf = None

def _lazy_import_yfinance():
    """Lazy import yfinance to avoid startup errors."""
    global YFINANCE_AVAILABLE, yf
    if yf is None:
        try:
            import yfinance as _yf
            yf = _yf
            YFINANCE_AVAILABLE = True
        except Exception as e:
            print(f"[FETCHER] yfinance error: {e}")
            YFINANCE_AVAILABLE = False
    return YFINANCE_AVAILABLE

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class StockData:
    """Container for stock data."""
    symbol: str
    current_price: float
    previous_close: float
    open_price: float
    high: float
    low: float
    volume: int
    avg_volume: int
    market_cap: float
    
    # Calculated
    change: float = 0.0
    change_pct: float = 0.0
    gap_pct: float = 0.0
    rvol: float = 1.0
    
    # Historical
    prices: List[float] = None
    highs: List[float] = None
    lows: List[float] = None
    volumes: List[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "previous_close": self.previous_close,
            "open": self.open_price,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "market_cap": self.market_cap,
            "change": self.change,
            "change_pct": self.change_pct,
            "gap_pct": self.gap_pct,
            "rvol": self.rvol
        }


class StockDataFetcher:
    """
    Fetches stock market data from various sources.
    """
    
    def __init__(self):
        """Initialize the fetcher."""
        print("[FETCHER] Initialized (yfinance loaded on demand)")
    
    def get_stock_data(self, symbol: str, period: str = "3mo") -> Optional[StockData]:
        """
        Get comprehensive stock data.
        
        Args:
            symbol: Stock symbol (e.g., "TSLA")
            period: Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            
        Returns:
            StockData object or None if failed
        """
        if not _lazy_import_yfinance():
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period=period)
            
            if hist.empty:
                print(f"[FETCHER] No data for {symbol}")
                return None
            
            # Extract values
            current = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
            open_price = hist['Open'].iloc[-1]
            high = hist['High'].iloc[-1]
            low = hist['Low'].iloc[-1]
            volume = int(hist['Volume'].iloc[-1])
            avg_volume = int(hist['Volume'].mean()) if len(hist) > 5 else volume
            
            # Create data object
            data = StockData(
                symbol=symbol.upper(),
                current_price=round(current, 2),
                previous_close=round(prev_close, 2),
                open_price=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                volume=volume,
                avg_volume=avg_volume,
                market_cap=info.get('marketCap', 0),
                prices=hist['Close'].tolist(),
                highs=hist['High'].tolist(),
                lows=hist['Low'].tolist(),
                volumes=hist['Volume'].tolist()
            )
            
            # Calculate derived values
            data.change = round(current - prev_close, 2)
            data.change_pct = round((data.change / prev_close) * 100, 2) if prev_close > 0 else 0
            data.gap_pct = round(((open_price - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0
            data.rvol = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
            
            return data
            
        except Exception as e:
            print(f"[FETCHER] Error fetching {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, StockData]:
        """
        Get data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to StockData
        """
        results = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol)
            if data:
                results[symbol] = data
        return results
    
    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock.
        
        Args:
            symbol: Stock symbol
            period: Historical period
            
        Returns:
            Dict with calculated indicators
        """
        data = self.get_stock_data(symbol, period)
        
        if not data or not data.prices or len(data.prices) < 30:
            return {"error": "Insufficient data"}
        
        prices = np.array(data.prices)
        
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(prices)
        
        # MACD
        macd, signal, hist = self._calculate_macd(prices)
        indicators['macd'] = {
            'macd': macd,
            'signal': signal,
            'histogram': hist,
            'crossover': 'bullish' if macd > signal else 'bearish'
        }
        
        # Moving Averages
        indicators['sma_20'] = round(np.mean(prices[-20:]), 2)
        indicators['sma_50'] = round(np.mean(prices[-50:]), 2) if len(prices) >= 50 else None
        indicators['sma_200'] = round(np.mean(prices[-200:]), 2) if len(prices) >= 200 else None
        
        # Bollinger Bands
        indicators['bollinger'] = self._calculate_bollinger(prices)
        
        # ATR
        if data.highs and data.lows:
            indicators['atr'] = self._calculate_atr(
                np.array(data.highs),
                np.array(data.lows),
                prices
            )
        
        # Trend
        indicators['trend'] = self._determine_trend(prices)
        
        # Current price context
        current = data.current_price
        indicators['price_context'] = {
            'above_sma_20': current > indicators['sma_20'],
            'rsi_signal': 'oversold' if indicators['rsi'] < 30 else ('overbought' if indicators['rsi'] > 70 else 'neutral'),
            'trend': indicators['trend']
        }
        
        return indicators
    
    def get_market_overview(self) -> Dict[str, Any]:
        """
        Get overview of major market indices.
        
        Returns:
            Dict with SPY, QQQ, VIX data
        """
        overview = {}
        
        # Major indices
        for symbol in ['SPY', 'QQQ', 'IWM']:
            data = self.get_stock_data(symbol, period='5d')
            if data:
                overview[symbol] = {
                    'price': data.current_price,
                    'change_pct': data.change_pct,
                    'trend': 'up' if data.change_pct > 0 else 'down'
                }
        
        # VIX (volatility)
        vix = self.get_stock_data('^VIX', period='5d')
        if vix:
            overview['VIX'] = {
                'level': vix.current_price,
                'signal': 'fear' if vix.current_price > 25 else ('complacent' if vix.current_price < 15 else 'normal')
            }
        
        return overview
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        ema_12 = ema(prices, 12)
        ema_26 = ema(prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = macd_line * 0.9  # Simplified
        histogram = macd_line - signal_line
        
        return round(macd_line, 4), round(signal_line, 4), round(histogram, 4)
    
    def _calculate_bollinger(self, prices: np.ndarray, period: int = 20, std: int = 2) -> Dict:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}
        
        window = prices[-period:]
        middle = np.mean(window)
        std_dev = np.std(window)
        
        return {
            'upper': round(middle + (std_dev * std), 2),
            'middle': round(middle, 2),
            'lower': round(middle - (std_dev * std), 2)
        }
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                       closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR."""
        if len(highs) < period + 1:
            return round(np.mean(highs[-10:] - lows[-10:]), 2)
        
        tr_list = []
        for i in range(1, len(highs)):
            h_l = highs[i] - lows[i]
            h_c = abs(highs[i] - closes[i-1])
            l_c = abs(lows[i] - closes[i-1])
            tr_list.append(max(h_l, h_c, l_c))
        
        return round(np.mean(tr_list[-period:]), 2)
    
    def _determine_trend(self, prices: np.ndarray) -> str:
        """Determine trend direction."""
        if len(prices) < 50:
            return 'neutral'
        
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        current = prices[-1]
        
        if current > sma_20 > sma_50:
            return 'bullish'
        elif current < sma_20 < sma_50:
            return 'bearish'
        return 'neutral'


# ===== CLI Interface =====

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch stock market data")
    parser.add_argument('symbol', type=str, nargs='?', default='AAPL',
                       help='Stock symbol')
    parser.add_argument('--indicators', action='store_true',
                       help='Show technical indicators')
    parser.add_argument('--market', action='store_true',
                       help='Show market overview')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Stock Data Fetcher - Zero Trading Expert")
    print("="*60)
    
    fetcher = StockDataFetcher()
    
    if args.market:
        print("\n--- Market Overview ---")
        overview = fetcher.get_market_overview()
        for key, value in overview.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n--- {args.symbol} Data ---")
        data = fetcher.get_stock_data(args.symbol)
        
        if data:
            print(f"  Price: ${data.current_price}")
            print(f"  Change: ${data.change} ({data.change_pct}%)")
            print(f"  Gap: {data.gap_pct}%")
            print(f"  Volume: {data.volume:,} (RVOL: {data.rvol}x)")
            
            if args.indicators:
                print("\n--- Technical Indicators ---")
                indicators = fetcher.get_technical_indicators(args.symbol)
                print(f"  RSI: {indicators.get('rsi')}")
                print(f"  MACD: {indicators.get('macd', {})}")
                print(f"  SMA 20: {indicators.get('sma_20')}")
                print(f"  Trend: {indicators.get('trend')}")
        else:
            print(f"  Failed to fetch data for {args.symbol}")


if __name__ == "__main__":
    main()

