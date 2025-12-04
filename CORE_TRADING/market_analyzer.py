"""
Market Analyzer for Zero Trading Expert
========================================
Technical analysis engine for stock evaluation.

Features:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- VWAP (Volume Weighted Average Price)
- Support/Resistance Detection
- Trend Analysis
- Volume Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARN] pandas not available - some features disabled")

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
            print(f"[WARN] yfinance not available: {e}")
            YFINANCE_AVAILABLE = False
    return YFINANCE_AVAILABLE


@dataclass
class TechnicalIndicators:
    """Container for technical indicators."""
    rsi: float = 0.0
    rsi_signal: str = "neutral"  # oversold, neutral, overbought
    rsi_series: List[float] = None  # RSI history for divergence detection
    
    # RSI Divergence (PHASE 1.1 - 85% Win Rate)
    rsi_divergence: str = "none"  # bullish, bearish, none
    rsi_divergence_strength: float = 0.0  # 0-1 confidence
    
    # TSI - True Strength Index (PHASE 1.2)
    tsi: float = 0.0
    tsi_signal: str = "neutral"  # oversold (<-25), neutral, overbought (>25)
    
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = "none"  # bullish, bearish, none
    # MACD Histogram Divergence (PHASE 1.5)
    macd_histogram_series: List[float] = None
    macd_divergence: str = "none"  # bullish, bearish, none
    macd_divergence_strength: float = 0.0  # 0-1 confidence
    
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: str = "middle"  # below, middle, above
    bb_percent: float = 0.5  # %B indicator (0 = lower band, 1 = upper band)
    
    vwap: float = 0.0
    vwap_position: str = "at"  # below, at, above
    
    trend: str = "neutral"  # bullish, neutral, bearish
    trend_strength: float = 0.0  # 0-1
    
    support: float = 0.0
    resistance: float = 0.0
    
    volume_ratio: float = 1.0  # Current vs average
    volume_signal: str = "normal"  # low, normal, high, extreme
    
    # Volume Profile (PHASE 1.4)
    vpoc: float = 0.0  # Volume Point of Control - highest volume price
    vah: float = 0.0   # Value Area High (70% of volume)
    val: float = 0.0   # Value Area Low (70% of volume)
    volume_profile_signal: str = "neutral"  # above_poc, at_poc, below_poc
    volume_nodes: List[Dict] = None  # High/Low volume nodes
    
    atr: float = 0.0
    atr_percent: float = 0.0


class MarketAnalyzer:
    """
    Technical Analysis Engine.
    Analyzes price data and generates trading signals.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Market Analyzer.
        
        Args:
            config: Configuration dict with indicator settings
        """
        self.config = config or {}
        
        # RSI settings
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        
        # MACD settings
        self.macd_fast = self.config.get("macd_fast", 12)
        self.macd_slow = self.config.get("macd_slow", 26)
        self.macd_signal_period = self.config.get("macd_signal", 9)
        
        # Bollinger settings
        self.bb_period = self.config.get("bollinger_period", 20)
        self.bb_std = self.config.get("bollinger_std", 2)
        
        # ATR settings
        self.atr_period = self.config.get("atr_period", 14)
        
        print("[MARKET_ANALYZER] Initialized")
    
    def analyze(self, symbol: str = None, 
                prices: List[float] = None,
                highs: List[float] = None,
                lows: List[float] = None,
                volumes: List[float] = None,
                current_price: float = None) -> TechnicalIndicators:
        """
        Perform full technical analysis.
        
        Args:
            symbol: Stock symbol (for fetching data if prices not provided)
            prices: List of closing prices (most recent last)
            highs: List of high prices
            lows: List of low prices
            volumes: List of volumes
            current_price: Current price (optional, uses last price if not provided)
            
        Returns:
            TechnicalIndicators object with all indicators
        """
        # Fetch data if symbol provided and no prices
        if symbol and prices is None:
            if _lazy_import_yfinance():
                prices, highs, lows, volumes = self._fetch_data(symbol)
        
        if prices is None or len(prices) < 30:
            print("[WARN] Insufficient data for analysis")
            return TechnicalIndicators()
        
        # Helper function to safely convert to float
        def safe_float(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return None
        
        # Convert to numpy arrays with proper dtype (float64)
        try:
            prices_clean = [safe_float(p) for p in prices]
            prices_clean = [p for p in prices_clean if p is not None]
            prices = np.array(prices_clean, dtype=np.float64)
            
            if highs is not None:
                highs_clean = [safe_float(h) for h in highs]
                highs_clean = [h for h in highs_clean if h is not None]
                highs = np.array(highs_clean, dtype=np.float64)
                
            if lows is not None:
                lows_clean = [safe_float(l) for l in lows]
                lows_clean = [l for l in lows_clean if l is not None]
                lows = np.array(lows_clean, dtype=np.float64)
                
            if volumes is not None:
                volumes_clean = [safe_float(v) for v in volumes]
                volumes_clean = [v for v in volumes_clean if v is not None]
                volumes = np.array(volumes_clean, dtype=np.float64)
        except (ValueError, TypeError) as e:
            print(f"[WARN] Data conversion error: {e}")
            return TechnicalIndicators()
        
        # Validate minimum data after cleaning
        if len(prices) < 30:
            print("[WARN] Insufficient valid data after cleaning")
            return TechnicalIndicators()
        
        # Ensure current is float
        current = float(current_price) if current_price else float(prices[-1])
        
        # Calculate all indicators
        indicators = TechnicalIndicators()
        
        # RSI
        indicators.rsi = self._calculate_rsi(prices)
        indicators.rsi_series = self._calculate_rsi_series(prices)
        if indicators.rsi <= self.rsi_oversold:
            indicators.rsi_signal = "oversold"
        elif indicators.rsi >= self.rsi_overbought:
            indicators.rsi_signal = "overbought"
        else:
            indicators.rsi_signal = "neutral"
        
        # RSI Divergence Detection (85% Win Rate)
        div_type, div_strength = self._detect_rsi_divergence(prices, indicators.rsi_series)
        indicators.rsi_divergence = div_type
        indicators.rsi_divergence_strength = div_strength
        
        # TSI - True Strength Index
        indicators.tsi = self._calculate_tsi(prices)
        if indicators.tsi >= 25:
            indicators.tsi_signal = "overbought"
        elif indicators.tsi <= -25:
            indicators.tsi_signal = "oversold"
        else:
            indicators.tsi_signal = "neutral"
        
        # MACD
        macd, signal, histogram, hist_series = self._calculate_macd_full(prices)
        indicators.macd = macd
        indicators.macd_signal = signal
        indicators.macd_histogram = histogram
        indicators.macd_histogram_series = hist_series
        
        if macd > signal and histogram > 0:
            indicators.macd_crossover = "bullish"
        elif macd < signal and histogram < 0:
            indicators.macd_crossover = "bearish"
        else:
            indicators.macd_crossover = "none"
        
        # MACD Histogram Divergence (Phase 1.5)
        macd_div, macd_div_strength = self._detect_macd_divergence(prices, hist_series)
        indicators.macd_divergence = macd_div
        indicators.macd_divergence_strength = macd_div_strength
        
        # Bollinger Bands
        upper, middle, lower = self._calculate_bollinger(prices)
        indicators.bb_upper = upper
        indicators.bb_middle = middle
        indicators.bb_lower = lower
        
        if current <= lower:
            indicators.bb_position = "below"
        elif current >= upper:
            indicators.bb_position = "above"
        else:
            indicators.bb_position = "middle"
        
        # Calculate %B (Bollinger Band position 0-1)
        if upper != lower:
            indicators.bb_percent = (current - lower) / (upper - lower)
        else:
            indicators.bb_percent = 0.5
        
        # VWAP (if volume available)
        if volumes is not None and len(volumes) > 0:
            indicators.vwap = self._calculate_vwap(prices, volumes)
            if current > indicators.vwap * 1.01:
                indicators.vwap_position = "above"
            elif current < indicators.vwap * 0.99:
                indicators.vwap_position = "below"
            else:
                indicators.vwap_position = "at"
        
        # Trend
        indicators.trend, indicators.trend_strength = self._calculate_trend(prices)
        
        # Support/Resistance
        if highs is not None and lows is not None:
            indicators.support, indicators.resistance = self._calculate_sr(highs, lows, current)
        
        # Volume Analysis
        if volumes is not None and len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            indicators.volume_ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0
            
            if indicators.volume_ratio < 0.5:
                indicators.volume_signal = "low"
            elif indicators.volume_ratio < 1.5:
                indicators.volume_signal = "normal"
            elif indicators.volume_ratio < 3.0:
                indicators.volume_signal = "high"
            else:
                indicators.volume_signal = "extreme"
        
        # ATR
        if highs is not None and lows is not None:
            indicators.atr = self._calculate_atr(highs, lows, prices)
            indicators.atr_percent = (indicators.atr / current) * 100 if current > 0 else 0
        
        # Volume Profile (Phase 1.4)
        if volumes is not None and highs is not None and lows is not None:
            vpoc, vah, val, nodes = self._calculate_volume_profile(prices, highs, lows, volumes)
            indicators.vpoc = vpoc
            indicators.vah = vah
            indicators.val = val
            indicators.volume_nodes = nodes
            
            # Determine position relative to POC
            if current > vpoc * 1.005:
                indicators.volume_profile_signal = "above_poc"
            elif current < vpoc * 0.995:
                indicators.volume_profile_signal = "below_poc"
            else:
                indicators.volume_profile_signal = "at_poc"
        
        return indicators
    
    def _fetch_data(self, symbol: str, period: str = "3mo") -> Tuple:
        """Fetch historical data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return None, None, None, None
            
            return (
                df['Close'].tolist(),
                df['High'].tolist(),
                df['Low'].tolist(),
                df['Volume'].tolist()
            )
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return None, None, None, None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = None) -> float:
        """Calculate RSI."""
        period = period or self.rsi_period
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(float(rsi), 2)
    
    def _calculate_rsi_series(self, prices: np.ndarray, period: int = None) -> List[float]:
        """
        Calculate RSI series for divergence detection.
        Returns list of RSI values for the last 30 periods.
        """
        period = period or self.rsi_period
        
        if len(prices) < period + 30:
            return []
        
        rsi_values = []
        for i in range(30, 0, -1):
            end_idx = len(prices) - i + 1
            if end_idx > period:
                rsi = self._calculate_rsi(prices[:end_idx], period)
                rsi_values.append(rsi)
        
        # Add current RSI
        rsi_values.append(self._calculate_rsi(prices, period))
        
        return rsi_values
    
    def _detect_rsi_divergence(self, prices: np.ndarray, rsi_series: List[float]) -> Tuple[str, float]:
        """
        Detect RSI Divergence - 85% Win Rate Strategy!
        
        Bullish Divergence: Price makes Lower Low, RSI makes Higher Low
        Bearish Divergence: Price makes Higher High, RSI makes Lower High
        
        Returns:
            (divergence_type, strength) - type is bullish/bearish/none, strength 0-1
        """
        if len(prices) < 20 or len(rsi_series) < 10:
            return "none", 0.0
        
        # Get last 20 periods of price
        recent_prices = prices[-20:]
        recent_rsi = rsi_series[-10:] if len(rsi_series) >= 10 else rsi_series
        
        # Find price peaks and troughs
        price_peaks = []
        price_troughs = []
        
        for i in range(2, len(recent_prices) - 2):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1] and \
               recent_prices[i] > recent_prices[i-2] and recent_prices[i] > recent_prices[i+2]:
                price_peaks.append((i, recent_prices[i]))
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1] and \
               recent_prices[i] < recent_prices[i-2] and recent_prices[i] < recent_prices[i+2]:
                price_troughs.append((i, recent_prices[i]))
        
        # Bullish Divergence: Lower Low in price + Higher Low in RSI
        if len(price_troughs) >= 2:
            t1_idx, t1_price = price_troughs[-2]
            t2_idx, t2_price = price_troughs[-1]
            
            if t2_price < t1_price:  # Price made lower low
                # Map to RSI indices (approximate)
                rsi_idx1 = min(len(recent_rsi) - 1, max(0, t1_idx // 2))
                rsi_idx2 = min(len(recent_rsi) - 1, max(0, t2_idx // 2))
                
                if rsi_idx2 > rsi_idx1 and len(recent_rsi) > rsi_idx2:
                    rsi1 = recent_rsi[rsi_idx1]
                    rsi2 = recent_rsi[rsi_idx2]
                    
                    if rsi2 > rsi1:  # RSI made higher low = BULLISH DIVERGENCE!
                        strength = min(1.0, (rsi2 - rsi1) / 20)  # Normalize
                        if strength > 0.1:
                            return "bullish", round(strength, 2)
        
        # Bearish Divergence: Higher High in price + Lower High in RSI
        if len(price_peaks) >= 2:
            p1_idx, p1_price = price_peaks[-2]
            p2_idx, p2_price = price_peaks[-1]
            
            if p2_price > p1_price:  # Price made higher high
                rsi_idx1 = min(len(recent_rsi) - 1, max(0, p1_idx // 2))
                rsi_idx2 = min(len(recent_rsi) - 1, max(0, p2_idx // 2))
                
                if rsi_idx2 > rsi_idx1 and len(recent_rsi) > rsi_idx2:
                    rsi1 = recent_rsi[rsi_idx1]
                    rsi2 = recent_rsi[rsi_idx2]
                    
                    if rsi2 < rsi1:  # RSI made lower high = BEARISH DIVERGENCE!
                        strength = min(1.0, (rsi1 - rsi2) / 20)
                        if strength > 0.1:
                            return "bearish", round(strength, 2)
        
        return "none", 0.0
    
    def _calculate_tsi(self, prices: np.ndarray, r: int = 25, s: int = 13) -> float:
        """
        Calculate True Strength Index (TSI).
        
        TSI = 100 * EMA(EMA(PC, r), s) / EMA(EMA(|PC|, r), s)
        Where PC = Price Change = Close - Close[1]
        
        Args:
            prices: Price array
            r: Long smoothing period (default 25)
            s: Short smoothing period (default 13)
            
        Returns:
            TSI value between -100 and 100
        """
        if len(prices) < r + s + 1:
            return 0.0
        
        # Price changes
        pc = np.diff(prices)
        abs_pc = np.abs(pc)
        
        # Double smoothed price change
        pc_ema_r = self._ema_array(pc, r)
        pc_double_smooth = self._ema_array(pc_ema_r, s)
        
        # Double smoothed absolute price change
        abs_pc_ema_r = self._ema_array(abs_pc, r)
        abs_pc_double_smooth = self._ema_array(abs_pc_ema_r, s)
        
        # TSI
        if abs_pc_double_smooth[-1] == 0:
            return 0.0
        
        tsi = 100 * (pc_double_smooth[-1] / abs_pc_double_smooth[-1])
        return round(tsi, 2)
    
    def _ema_array(self, values: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate EMA for entire array.
        Returns array of EMA values.
        """
        if len(values) < period:
            return values
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(values))
        ema[0] = values[0]
        
        for i in range(1, len(values)):
            ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _calculate_volume_profile(self, prices: np.ndarray, highs: np.ndarray, 
                                   lows: np.ndarray, volumes: np.ndarray,
                                   num_bins: int = 20) -> Tuple[float, float, float, List[Dict]]:
        """
        Calculate Volume Profile - identifies key price levels by volume.
        
        Volume Profile shows where most trading activity occurred,
        helping identify support/resistance levels.
        
        Args:
            prices: Closing prices
            highs: High prices
            lows: Low prices  
            volumes: Trading volumes
            num_bins: Number of price bins (default 20)
            
        Returns:
            (VPOC, VAH, VAL, volume_nodes)
            - VPOC: Volume Point of Control (highest volume price)
            - VAH: Value Area High (upper 70% volume boundary)
            - VAL: Value Area Low (lower 70% volume boundary)
            - volume_nodes: List of high/low volume nodes
        """
        if len(prices) < 10 or len(volumes) < 10:
            return 0.0, 0.0, 0.0, []
        
        # Use last 20-60 bars for volume profile
        lookback = min(60, len(prices))
        prices = prices[-lookback:]
        highs = highs[-lookback:]
        lows = lows[-lookback:]
        volumes = volumes[-lookback:]
        
        # Find price range
        price_high = float(np.max(highs))
        price_low = float(np.min(lows))
        price_range = price_high - price_low
        
        if price_range <= 0:
            return float(prices[-1]), float(prices[-1]), float(prices[-1]), []
        
        # Create price bins
        bin_size = price_range / num_bins
        bins = [price_low + i * bin_size for i in range(num_bins + 1)]
        bin_volumes = [0.0] * num_bins
        
        # Distribute volume across price bins
        for i in range(len(prices)):
            bar_low = lows[i]
            bar_high = highs[i]
            bar_volume = volumes[i]
            
            # Find which bins this bar touches
            for j in range(num_bins):
                bin_low = bins[j]
                bin_high = bins[j + 1]
                
                # Check if bar overlaps with this bin
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)
                
                if overlap_high > overlap_low:
                    # Proportional volume allocation
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        overlap_ratio = (overlap_high - overlap_low) / bar_range
                        bin_volumes[j] += bar_volume * overlap_ratio
        
        # Find VPOC (Volume Point of Control) - highest volume bin
        max_vol_idx = int(np.argmax(bin_volumes))
        vpoc = (bins[max_vol_idx] + bins[max_vol_idx + 1]) / 2
        
        # Calculate Value Area (70% of total volume)
        total_volume = sum(bin_volumes)
        target_volume = total_volume * 0.70
        
        # Start from POC and expand outward
        accumulated_volume = bin_volumes[max_vol_idx]
        low_idx = max_vol_idx
        high_idx = max_vol_idx
        
        while accumulated_volume < target_volume:
            # Expand to the side with higher volume
            expand_low = bin_volumes[low_idx - 1] if low_idx > 0 else 0
            expand_high = bin_volumes[high_idx + 1] if high_idx < num_bins - 1 else 0
            
            if expand_low >= expand_high and low_idx > 0:
                low_idx -= 1
                accumulated_volume += bin_volumes[low_idx]
            elif high_idx < num_bins - 1:
                high_idx += 1
                accumulated_volume += bin_volumes[high_idx]
            else:
                break
        
        val = bins[low_idx]  # Value Area Low
        vah = bins[high_idx + 1]  # Value Area High
        
        # Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        avg_volume = total_volume / num_bins
        volume_nodes = []
        
        for j in range(num_bins):
            node_price = (bins[j] + bins[j + 1]) / 2
            node_volume = bin_volumes[j]
            
            if node_volume > avg_volume * 1.5:
                volume_nodes.append({
                    "type": "HVN",
                    "price": round(node_price, 2),
                    "volume_ratio": round(node_volume / avg_volume, 2)
                })
            elif node_volume < avg_volume * 0.5:
                volume_nodes.append({
                    "type": "LVN",
                    "price": round(node_price, 2),
                    "volume_ratio": round(node_volume / avg_volume, 2)
                })
        
        # Sort by price
        volume_nodes.sort(key=lambda x: x["price"])
        
        return round(vpoc, 2), round(vah, 2), round(val, 2), volume_nodes
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD."""
        if len(prices) < self.macd_slow + self.macd_signal_period:
            return 0.0, 0.0, 0.0
        
        # EMA calculations
        ema_fast = self._ema(prices, self.macd_fast)
        ema_slow = self._ema(prices, self.macd_slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._ema_from_values([macd_line], self.macd_signal_period)
        histogram = macd_line - signal_line
        
        return round(macd_line, 4), round(signal_line, 4), round(histogram, 4)
    
    def _calculate_macd_full(self, prices: np.ndarray) -> Tuple[float, float, float, List[float]]:
        """
        Calculate MACD with histogram series for divergence detection.
        Returns (macd, signal, histogram, histogram_series)
        """
        if len(prices) < self.macd_slow + self.macd_signal_period + 20:
            return 0.0, 0.0, 0.0, []
        
        # Calculate MACD line series
        macd_series = []
        signal_series = []
        histogram_series = []
        
        for i in range(self.macd_slow + self.macd_signal_period, len(prices) + 1):
            subset = prices[:i]
            ema_fast = self._ema(subset, self.macd_fast)
            ema_slow = self._ema(subset, self.macd_slow)
            macd_val = ema_fast - ema_slow
            macd_series.append(macd_val)
        
        # Calculate signal line and histogram
        for i in range(len(macd_series)):
            if i >= self.macd_signal_period - 1:
                signal_val = self._ema(np.array(macd_series[:i+1]), self.macd_signal_period)
            else:
                signal_val = macd_series[i]
            signal_series.append(signal_val)
            histogram_series.append(macd_series[i] - signal_val)
        
        # Return current values and last 20 histogram values
        macd = macd_series[-1] if macd_series else 0.0
        signal = signal_series[-1] if signal_series else 0.0
        histogram = histogram_series[-1] if histogram_series else 0.0
        
        return round(macd, 4), round(signal, 4), round(histogram, 4), histogram_series[-20:]
    
    def _detect_macd_divergence(self, prices: np.ndarray, hist_series: List[float]) -> Tuple[str, float]:
        """
        Detect MACD Histogram Divergence.
        
        Bullish: Price makes Lower Low, MACD Histogram makes Higher Low
        Bearish: Price makes Higher High, MACD Histogram makes Lower High
        
        MACD Divergence is a powerful trend reversal signal.
        """
        if len(prices) < 20 or len(hist_series) < 10:
            return "none", 0.0
        
        recent_prices = prices[-20:]
        recent_hist = hist_series[-10:] if len(hist_series) >= 10 else hist_series
        
        # Find price peaks and troughs
        price_peaks = []
        price_troughs = []
        
        for i in range(2, len(recent_prices) - 2):
            if recent_prices[i] > max(recent_prices[i-2:i]) and \
               recent_prices[i] > max(recent_prices[i+1:i+3]):
                price_peaks.append((i, recent_prices[i]))
            if recent_prices[i] < min(recent_prices[i-2:i]) and \
               recent_prices[i] < min(recent_prices[i+1:i+3]):
                price_troughs.append((i, recent_prices[i]))
        
        # Find histogram peaks and troughs
        hist_peaks = []
        hist_troughs = []
        
        for i in range(1, len(recent_hist) - 1):
            if recent_hist[i] > recent_hist[i-1] and recent_hist[i] > recent_hist[i+1]:
                hist_peaks.append((i, recent_hist[i]))
            if recent_hist[i] < recent_hist[i-1] and recent_hist[i] < recent_hist[i+1]:
                hist_troughs.append((i, recent_hist[i]))
        
        # Bullish Divergence: Lower Low in price + Higher Low in histogram
        if len(price_troughs) >= 2 and len(hist_troughs) >= 2:
            p1, p1_price = price_troughs[-2]
            p2, p2_price = price_troughs[-1]
            h1, h1_val = hist_troughs[-2]
            h2, h2_val = hist_troughs[-1]
            
            if p2_price < p1_price and h2_val > h1_val:
                strength = min(1.0, abs(h2_val - h1_val) * 50)
                if strength > 0.1:
                    return "bullish", round(strength, 2)
        
        # Bearish Divergence: Higher High in price + Lower High in histogram
        if len(price_peaks) >= 2 and len(hist_peaks) >= 2:
            p1, p1_price = price_peaks[-2]
            p2, p2_price = price_peaks[-1]
            h1, h1_val = hist_peaks[-2]
            h2, h2_val = hist_peaks[-1]
            
            if p2_price > p1_price and h2_val < h1_val:
                strength = min(1.0, abs(h1_val - h2_val) * 50)
                if strength > 0.1:
                    return "bearish", round(strength, 2)
        
        return "none", 0.0
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _ema_from_values(self, values: List[float], period: int) -> float:
        """Calculate EMA from a list of values."""
        if len(values) < period:
            return values[-1] if len(values) > 0 else 0
        return self._ema(np.array(values), period)
    
    def _calculate_bollinger(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < self.bb_period:
            return float(prices[-1]), float(prices[-1]), float(prices[-1])
        
        window = prices[-self.bb_period:]
        middle = float(np.mean(window))
        std = float(np.std(window))
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return round(float(upper), 2), round(float(middle), 2), round(float(lower), 2)
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate VWAP."""
        if len(prices) == 0 or len(volumes) == 0:
            return 0.0
        
        cumulative_tp_vol = np.sum(prices * volumes)
        cumulative_vol = np.sum(volumes)
        
        if cumulative_vol == 0:
            return float(prices[-1])
        
        return round(float(cumulative_tp_vol / cumulative_vol), 2)
    
    def _calculate_trend(self, prices: np.ndarray) -> Tuple[str, float]:
        """
        Calculate trend direction and strength.
        
        Returns:
            (trend, strength) - trend is bullish/bearish/neutral, strength is 0-1
        """
        if len(prices) < 50:
            return "neutral", 0.0
        
        # Use SMA 20 vs SMA 50
        sma_20 = float(np.mean(prices[-20:]))
        sma_50 = float(np.mean(prices[-50:]))
        current = float(prices[-1])
        
        # Calculate strength based on distance from SMAs
        avg_price = float(np.mean(prices[-50:]))
        distance = abs(sma_20 - sma_50) / avg_price if avg_price > 0 else 0
        strength = min(distance * 10, 1.0)  # Normalize to 0-1
        
        if current > sma_20 > sma_50:
            return "bullish", strength
        elif current < sma_20 < sma_50:
            return "bearish", strength
        else:
            return "neutral", strength * 0.5
    
    def _calculate_sr(self, highs: np.ndarray, lows: np.ndarray, 
                      current: float) -> Tuple[float, float]:
        """Calculate nearest support and resistance levels."""
        if len(highs) < 20 or len(lows) < 20:
            return current * 0.95, current * 1.05
        
        # Ensure current is a scalar float
        current = float(current)
        
        # Find recent swing highs and lows
        recent_highs = highs[-20:].astype(np.float64)
        recent_lows = lows[-20:].astype(np.float64)
        
        try:
            # Resistance: closest high above current price
            highs_above = recent_highs[recent_highs > current]
            resistance = float(np.min(highs_above)) if len(highs_above) > 0 else float(np.max(recent_highs))
            
            # Support: closest low below current price
            lows_below = recent_lows[recent_lows < current]
            support = float(np.max(lows_below)) if len(lows_below) > 0 else float(np.min(recent_lows))
        except Exception:
            # Fallback to simple calculation
            support = current * 0.95
            resistance = current * 1.05
        
        return round(support, 2), round(resistance, 2)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                       closes: np.ndarray, period: int = None) -> float:
        """Calculate Average True Range."""
        period = period or self.atr_period
        
        if len(highs) < period + 1:
            return float(np.mean(highs[-10:] - lows[-10:])) if len(highs) >= 10 else 0.0
        
        # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        tr_list = []
        for i in range(1, len(highs)):
            h_l = highs[i] - lows[i]
            h_c = abs(highs[i] - closes[i-1])
            l_c = abs(lows[i] - closes[i-1])
            tr_list.append(max(h_l, h_c, l_c))
        
        # Average of last 'period' TRs
        atr = float(np.mean(tr_list[-period:]))
        return round(atr, 2)
    
    def get_summary(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """
        Generate a summary of the analysis.
        
        Args:
            indicators: TechnicalIndicators object
            
        Returns:
            Dict with summary information
        """
        # Count bullish/bearish signals
        bullish_signals = []
        bearish_signals = []
        
        if indicators.rsi_signal == "oversold":
            bullish_signals.append("RSI_Oversold")
        elif indicators.rsi_signal == "overbought":
            bearish_signals.append("RSI_Overbought")
        
        # RSI Divergence (High Win Rate!)
        if indicators.rsi_divergence == "bullish":
            bullish_signals.append(f"RSI_Bullish_Divergence({indicators.rsi_divergence_strength:.0%})")
        elif indicators.rsi_divergence == "bearish":
            bearish_signals.append(f"RSI_Bearish_Divergence({indicators.rsi_divergence_strength:.0%})")
        
        # TSI
        if indicators.tsi_signal == "oversold":
            bullish_signals.append("TSI_Oversold")
        elif indicators.tsi_signal == "overbought":
            bearish_signals.append("TSI_Overbought")
        
        if indicators.macd_crossover == "bullish":
            bullish_signals.append("MACD_Bullish")
        elif indicators.macd_crossover == "bearish":
            bearish_signals.append("MACD_Bearish")
        
        # MACD Histogram Divergence (Phase 1.5)
        if indicators.macd_divergence == "bullish":
            bullish_signals.append(f"MACD_Bullish_Divergence({indicators.macd_divergence_strength:.0%})")
        elif indicators.macd_divergence == "bearish":
            bearish_signals.append(f"MACD_Bearish_Divergence({indicators.macd_divergence_strength:.0%})")
        
        if indicators.bb_position == "below":
            bullish_signals.append("BB_Oversold")
        elif indicators.bb_position == "above":
            bearish_signals.append("BB_Overbought")
        
        if indicators.vwap_position == "above":
            bullish_signals.append("Above_VWAP")
        elif indicators.vwap_position == "below":
            bearish_signals.append("Below_VWAP")
        
        if indicators.trend == "bullish":
            bullish_signals.append("Bullish_Trend")
        elif indicators.trend == "bearish":
            bearish_signals.append("Bearish_Trend")
        
        if indicators.volume_signal in ["high", "extreme"]:
            bullish_signals.append("High_Volume")
        
        # Volume Profile signals
        if indicators.volume_profile_signal == "below_poc":
            bullish_signals.append("Below_VPOC")  # Price below POC often bounces up
        elif indicators.volume_profile_signal == "above_poc":
            bearish_signals.append("Above_VPOC")  # Price above POC may pull back
        
        # Calculate overall score
        bull_count = len(bullish_signals)
        bear_count = len(bearish_signals)
        total = bull_count + bear_count
        
        if total == 0:
            bias = "neutral"
            confidence = 0.5
        else:
            if bull_count > bear_count:
                bias = "bullish"
                confidence = bull_count / total
            elif bear_count > bull_count:
                bias = "bearish"
                confidence = bear_count / total
            else:
                bias = "neutral"
                confidence = 0.5
        
        return {
            "bias": bias,
            "confidence": round(confidence, 2),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "rsi": indicators.rsi,
            "tsi": indicators.tsi,
            "rsi_divergence": indicators.rsi_divergence,
            "macd_histogram": indicators.macd_histogram,
            "trend": indicators.trend,
            "trend_strength": indicators.trend_strength,
            "volume_signal": indicators.volume_signal,
            "vpoc": indicators.vpoc,
            "vah": indicators.vah,
            "val": indicators.val,
            "atr": indicators.atr,
            "atr_percent": indicators.atr_percent,
            "support": indicators.support,
            "resistance": indicators.resistance
        }


# ===== TESTING =====

if __name__ == "__main__":
    print("Testing Market Analyzer...")
    
    analyzer = MarketAnalyzer()
    
    # Test with sample data
    sample_prices = [100 + i * 0.5 + np.random.randn() * 2 for i in range(100)]
    sample_highs = [p + abs(np.random.randn()) for p in sample_prices]
    sample_lows = [p - abs(np.random.randn()) for p in sample_prices]
    sample_volumes = [1000000 + np.random.randint(-200000, 200000) for _ in range(100)]
    
    indicators = analyzer.analyze(
        prices=sample_prices,
        highs=sample_highs,
        lows=sample_lows,
        volumes=sample_volumes
    )
    
    print("\n--- Indicators ---")
    print(f"RSI: {indicators.rsi} ({indicators.rsi_signal})")
    print(f"MACD: {indicators.macd} / Signal: {indicators.macd_signal} / Histogram: {indicators.macd_histogram}")
    print(f"MACD Crossover: {indicators.macd_crossover}")
    print(f"Bollinger: {indicators.bb_lower} / {indicators.bb_middle} / {indicators.bb_upper}")
    print(f"BB Position: {indicators.bb_position}")
    print(f"VWAP: {indicators.vwap} (Position: {indicators.vwap_position})")
    print(f"Trend: {indicators.trend} (Strength: {indicators.trend_strength:.2f})")
    print(f"Support: {indicators.support} / Resistance: {indicators.resistance}")
    print(f"Volume Ratio: {indicators.volume_ratio:.2f}x ({indicators.volume_signal})")
    print(f"ATR: {indicators.atr} ({indicators.atr_percent:.2f}%)")
    
    print("\n--- Summary ---")
    summary = analyzer.get_summary(indicators)
    print(f"Bias: {summary['bias']} (Confidence: {summary['confidence']})")
    print(f"Bullish Signals: {summary['bullish_signals']}")
    print(f"Bearish Signals: {summary['bearish_signals']}")
    
    # Test with real data if yfinance available
    if YFINANCE_AVAILABLE:
        print("\n--- Testing with AAPL ---")
        indicators = analyzer.analyze(symbol="AAPL")
        summary = analyzer.get_summary(indicators)
        print(f"AAPL Bias: {summary['bias']} (Confidence: {summary['confidence']})")
        print(f"RSI: {indicators.rsi}, Trend: {indicators.trend}")

