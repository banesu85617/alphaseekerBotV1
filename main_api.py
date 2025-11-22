# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import time
import ccxt
# import ccxt.async_support as ccxt_async
import logging
from datetime import datetime
from arch import arch_model
from sklearn.preprocessing import StandardScaler
# Silence TensorFlow warnings BEFORE importing Keras/TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# --- LSTM DISABLED: Comment out TF/Keras imports ---
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input
# --- END LSTM DISABLED ---
import warnings
import openai
import json
import re
import asyncio
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
import uvicorn
from dotenv import load_dotenv
import sys

# --- Logging Configuration (Ensure this runs BEFORE any logging happens) ---
# Set level to INFO by default, easy to change to DEBUG if needed
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Use a stream handler to force output to console, overriding potential Uvicorn capture
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Force output to stdout
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
# Remove existing handlers if any to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)

# --- Test Log Message ---
logger.critical("--- Logging Initialized (Level: %s). Logs should now appear on the console. ---", LOG_LEVEL)


# --- LSTM DISABLED: Remove TF_LOCK or keep if other TF usage exists (GARCH doesn't use TF) ---
# TF_LOCK = asyncio.Lock()
# logging.info("TensorFlow Lock initialized.") # Commented out

# --- Configuration ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="ConvergenceWarning", category=UserWarning)

load_dotenv()

# --- LSTM DISABLED: Configuration no longer needed ---
# LSTM_TIME_STEPS = 60
# LSTM_EPOCHS = 15
# LSTM_BATCH_SIZE = 64
# --- END LSTM DISABLED ---

DEFAULT_MAX_CONCURRENT_TASKS = 5

# --- CCXT Initialization (Unchanged) ---
binance_futures = None
try:
    binance_futures = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': { 'adjustForTimeDifference': True },
        'timeout': 30000,
        'rateLimit': 200
    })
    logger.info("CCXT Binance Futures instance created.")
except Exception as e:
    logger.error(f"Error initializing CCXT: {e}", exc_info=True)
    binance_futures = None

# --- Load Markets Function (Unchanged) ---
async def load_exchange_markets(exchange):
    if not exchange: return False
    try:
        logger.info(f"Attempting to load markets for {exchange.id}...")
        markets = await asyncio.to_thread(exchange.load_markets, True)
        if markets:
             logger.info(f"Successfully loaded {len(markets)} markets for {exchange.id}.")
             return True
        else:
             logger.warning(f"Market loading returned empty for {exchange.id}.")
             return False
    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
        logger.error(f"Failed to load markets for {exchange.id} due to network error: {e}", exc_info=False)
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to load markets for {exchange.id} due to exchange error: {e}", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading markets for {exchange.id}: {e}", exc_info=True)
        return False

# --- OpenAI Initialization (Unchanged) ---
openai_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Use getenv
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found. GPT features will be disabled.")
else:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI: {e}", exc_info=True)
        openai_client = None

# --- Test OpenAI Connection (Unchanged) ---
async def test_openai_connection(client):
     if not client: return
     try:
         await asyncio.to_thread(client.models.list)
         logger.info("OpenAI connection test successful.")
     except Exception as e:
         logger.error(f"OpenAI connection test failed: {e}")

# --- FastAPI App (Unchanged) ---
app = FastAPI(
    title="Crypto Trading Analysis & Scanning API",
    description="API for technical analysis, GPT-driven strategy evaluation, backtesting, and market scanning.",
    version="1.3.0 Simplified" # Version bump
)

# --- Pydantic Models ---

# Tickers (Unchanged)
class TickerRequest(BaseModel): pass
class TickersResponse(BaseModel): tickers: List[str]

# AnalysisRequest - LSTM params removed
class AnalysisRequest(BaseModel):
    symbol: str = Field(..., example="BTC/USDT:USDT")
    timeframe: str = Field(default="1h", example="1h")
    lookback: int = Field(default=1000, ge=250)
    accountBalance: float = Field(default=1000.0, ge=0)
    maxLeverage: float = Field(default=10.0, ge=1)
    # --- LSTM DISABLED ---
    # lstm_time_steps: int = Field(default=LSTM_TIME_STEPS, ge=10)
    # lstm_epochs: int = Field(default=LSTM_EPOCHS, ge=1, le=100)
    # lstm_batch_size: int = Field(default=LSTM_BATCH_SIZE, ge=16)
    # --- END LSTM DISABLED ---

# IndicatorsData (Unchanged)
class IndicatorsData(BaseModel):
    RSI: Optional[float] = None; ATR: Optional[float] = None; SMA_50: Optional[float] = None; SMA_200: Optional[float] = None
    EMA_12: Optional[float] = None; EMA_26: Optional[float] = None; MACD: Optional[float] = None; Signal_Line: Optional[float] = None
    Bollinger_Upper: Optional[float] = None; Bollinger_Middle: Optional[float] = None; Bollinger_Lower: Optional[float] = None
    Momentum: Optional[float] = None; Stochastic_K: Optional[float] = None; Stochastic_D: Optional[float] = None
    Williams_R: Optional[float] = Field(None, alias="Williams_%R")
    ADX: Optional[float] = None; CCI: Optional[float] = None; OBV: Optional[float] = None; returns: Optional[float] = None
    model_config = ConfigDict(populate_by_name=True, extra='allow')

# ModelOutputData - LSTM removed
class ModelOutputData(BaseModel):
    # --- LSTM DISABLED ---
    # lstmForecast: Optional[float] = None
    # --- END LSTM DISABLED ---
    garchVolatility: Optional[float] = None
    var95: Optional[float] = None

# GptAnalysisText - LSTM analysis removed
class GptAnalysisText(BaseModel):
    # --- LSTM DISABLED ---
    # lstm_analysis: Optional[str] = None
    # --- END LSTM DISABLED ---
    technical_analysis: Optional[str] = None # Justification for evaluation
    risk_assessment: Optional[str] = None
    market_outlook: Optional[str] = None
    raw_text: Optional[str] = None
    signal_evaluation: Optional[str] = None # Added field for evaluation summary

# GptTradingParams (Unchanged, but interpretation changes)
class GptTradingParams(BaseModel):
    optimal_entry: Optional[float] = None; stop_loss: Optional[float] = None; take_profit: Optional[float] = None
    trade_direction: Optional[str] = None; leverage: Optional[int] = Field(None, ge=1)
    position_size_usd: Optional[float] = Field(None, ge=0); estimated_profit: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1) # Confidence in the *evaluated* technical signal

# Backtest Models (Unchanged)
class BacktestTradeAnalysis(BaseModel):
    total_trades: int = 0; winning_trades: int = 0; losing_trades: int = 0; win_rate: Optional[float] = None
    avg_profit: Optional[float] = None; avg_loss: Optional[float] = None; profit_factor: Optional[float] = None
    total_profit: Optional[float] = None; largest_win: Optional[float] = None; largest_loss: Optional[float] = None
    average_trade_duration: Optional[float] = None

class BacktestResultsData(BaseModel):
    strategy_score: Optional[float] = Field(None, ge=0, le=1); trade_analysis: Optional[BacktestTradeAnalysis] = None
    recommendation: Optional[str] = None; warnings: List[str] = Field([])

# AnalysisResponse - Updated ModelOutput and GptAnalysis
class AnalysisResponse(BaseModel):
    symbol: str; timeframe: str; currentPrice: Optional[float] = None
    indicators: Optional[IndicatorsData] = None
    modelOutput: Optional[ModelOutputData] = None # LSTM removed
    gptParams: Optional[GptTradingParams] = None
    gptAnalysis: Optional[GptAnalysisText] = None # LSTM removed, evaluation added
    backtest: Optional[BacktestResultsData] = None
    error: Optional[str] = None

# ScanRequest - LSTM params removed
class ScanRequest(BaseModel):
    # Core Scan Params (Match user JSON where applicable)
    ticker_start_index: Optional[int] = Field(default=0, ge=0, description="0-based index to start scanning.")
    ticker_end_index: Optional[int] = Field(default=None, ge=0, description="0-based index to end scanning (exclusive).")
    timeframe: str = Field(default="1m", description="Candle timeframe (e.g., '1m', '5m', '1h').") # Default '1m' from user JSON
    max_tickers: Optional[int] = Field(default=100, description="Maximum tickers per run.") # Default 100 from user JSON
    top_n: int = Field(default=10, ge=1, description="Number of top results to return.")

    # Core Filters (Match user JSON)
    min_gpt_confidence: float = Field(default=0.65, ge=0, le=1) # From user JSON
    min_backtest_score: float = Field(default=0.60, ge=0, le=1) # From user JSON
    trade_direction: Optional[str] = Field(default=None, pattern="^(long|short)$") # null in user JSON -> default None
    
        # --- NEW BTC TREND FILTER ---
    filter_by_btc_trend: Optional[bool] = Field(default=True, description="If True, only show LONG signals if BTC is in Uptrend, and SHORT signals if BTC is in Downtrend (based on Price/SMA50/SMA200).")
    # --- END NEW BTC TREND FILTER ---
    
    
    # --- New Backtest Filters (from user JSON) ---
    min_backtest_trades: Optional[int] = Field(default=15, ge=0) # From user JSON
    min_backtest_win_rate: Optional[float] = Field(default=0.52, ge=0, le=1) # From user JSON
    min_backtest_profit_factor: Optional[float] = Field(default=1.5, ge=0) # From user JSON

    # --- New GPT/Risk Filter (from user JSON) ---
    min_risk_reward_ratio: Optional[float] = Field(default=1.8, ge=0) # From user JSON

    # --- New Indicator Filters (from user JSON) ---
    min_adx: Optional[float] = Field(default=25.0, ge=0) # From user JSON
    require_sma_alignment: Optional[bool] = Field(default=True) # From user JSON

    # Analysis Config (Match user JSON)
    lookback: int = Field(default=2000, ge=250) # From user JSON
    accountBalance: float = Field(default=5000.0, ge=0) # From user JSON
    maxLeverage: float = Field(default=20.0, ge=1) # From user JSON
    max_concurrent_tasks: int = Field(default=16, ge=1) # From user JSON

# ScanResultItem (Unchanged)
class ScanResultItem(BaseModel):
    rank: int; symbol: str; timeframe: str; currentPrice: Optional[float] = None
    gptConfidence: Optional[float] = None; backtestScore: Optional[float] = None; combinedScore: Optional[float] = None
    tradeDirection: Optional[str] = None; optimalEntry: Optional[float] = None; stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None; gptAnalysisSummary: Optional[str] = None # Will now contain evaluation summary

# ScanResponse (Unchanged structure, data reflects changes)
class ScanResponse(BaseModel):
    scan_parameters: ScanRequest; total_tickers_attempted: int; total_tickers_succeeded: int
    ticker_start_index: Optional[int] = Field(default=0, ge=0); ticker_end_index: Optional[int] = Field(default=None, ge=0)
    total_opportunities_found: int; top_opportunities: List[ScanResultItem]
    errors: Dict[str, str] = Field(default={})


# --- Helper Functions ---

# --- Data Fetcher (Unchanged) ---
def get_real_time_data(symbol: str, timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data. Raises exceptions on failure."""
    logger.debug(f"[{symbol}] Attempting to fetch {limit} candles for timeframe {timeframe}")
    if binance_futures is None: raise ConnectionError("CCXT exchange instance is not available.")
    if not binance_futures.markets:
         logger.warning(f"[{symbol}] Markets not loaded, attempting synchronous load...")
         try: binance_futures.load_markets(True); logger.info(f"[{symbol}] Markets loaded successfully (sync).")
         except Exception as e: raise ConnectionError(f"Failed to load markets synchronously: {e}") from e
    try:
        ohlcv = binance_futures.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            logger.warning(f"[{symbol}] No OHLCV data returned from fetch_ohlcv.")
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        # Keep rows with valid price/volume, even if some indicators are NaN later
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        if df.empty: logger.warning(f"[{symbol}] DataFrame became empty after type conversion/NaN drop.")
        else: logger.debug(f"[{symbol}] Fetched {len(df)} valid candles.")
        return df
    except ccxt.BadSymbol as e:
        logger.error(f"[{symbol}] Invalid symbol error: {e}")
        raise ValueError(f"Invalid symbol '{symbol}'") from e
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"[{symbol}] Rate limit exceeded: {e}")
        raise ConnectionAbortedError(f"Rate limit exceeded") from e
    except ccxt.NetworkError as e:
        logger.error(f"[{symbol}] Network error: {e}")
        raise ConnectionError(f"Network error fetching {symbol}") from e
    except ccxt.AuthenticationError as e:
        logger.error(f"[{symbol}] Authentication error: {e}")
        raise PermissionError("CCXT Authentication Error") from e
    except Exception as e:
        logger.error(f"[{symbol}] Unexpected error fetching data: {e}", exc_info=True)
        raise RuntimeError(f"Failed to fetch data for {symbol}") from e


# --- Indicator Functions (compute_*, apply_technical_indicators - Unchanged) ---
def compute_rsi(series, window=14):
    delta = series.diff(); gain = delta.where(delta > 0, 0.0).fillna(0); loss = -delta.where(delta < 0, 0.0).fillna(0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean(); avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); rsi = 100.0 - (100.0 / (1.0 + rs)); return rsi.fillna(100) # Fill NaN with 100 (or 0?) maybe fillna(50) better? Let's keep 100 for now.
def compute_atr(df, window=14):
    high_low = df['high'] - df['low']; high_close = abs(df['high'] - df['close'].shift()); low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False); atr = tr.ewm(com=window - 1, min_periods=window).mean(); return atr
def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window, min_periods=window).mean(); std = series.rolling(window=window, min_periods=window).std()
    upper_band = sma + 2 * std; lower_band = sma - 2 * std; return upper_band, sma, lower_band
def compute_stochastic_oscillator(df, window=14, smooth_k=3):
    lowest_low = df['low'].rolling(window=window, min_periods=window).min(); highest_high = df['high'].rolling(window=window, min_periods=window).max()
    range_hh_ll = (highest_high - lowest_low).replace(0, np.nan); k_percent = 100 * ((df['close'] - lowest_low) / range_hh_ll)
    d_percent = k_percent.rolling(window=smooth_k, min_periods=smooth_k).mean(); return k_percent, d_percent
def compute_williams_r(df, window=14):
    highest_high = df['high'].rolling(window=window, min_periods=window).max(); lowest_low = df['low'].rolling(window=window, min_periods=window).min()
    range_ = (highest_high - lowest_low).replace(0, np.nan); williams_r = -100 * ((highest_high - df['close']) / range_); return williams_r
def compute_adx(df, window=14):
    df_adx = df.copy(); df_adx['H-L'] = df_adx['high'] - df_adx['low']; df_adx['H-PC'] = abs(df_adx['high'] - df_adx['close'].shift(1))
    df_adx['L-PC'] = abs(df_adx['low'] - df_adx['low'].shift(1)); df_adx['TR_calc'] = df_adx[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_adx['DMplus'] = np.where((df_adx['high'] - df_adx['high'].shift(1)) > (df_adx['low'].shift(1) - df_adx['low']), df_adx['high'] - df_adx['high'].shift(1), 0)
    df_adx['DMplus'] = np.where(df_adx['DMplus'] < 0, 0, df_adx['DMplus'])
    df_adx['DMminus'] = np.where((df_adx['low'].shift(1) - df_adx['low']) > (df_adx['high'] - df_adx['high'].shift(1)), df_adx['low'].shift(1) - df_adx['low'], 0)
    df_adx['DMminus'] = np.where(df_adx['DMminus'] < 0, 0, df_adx['DMminus'])
    alpha = 1 / window; TR_smooth = df_adx['TR_calc'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    DMplus_smooth = df_adx['DMplus'].ewm(alpha=alpha, adjust=False, min_periods=window).mean(); DMminus_smooth = df_adx['DMminus'].ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    DIplus = 100 * (DMplus_smooth / TR_smooth.replace(0, np.nan)).fillna(0); DIminus = 100 * (DMminus_smooth / TR_smooth.replace(0, np.nan)).fillna(0)
    DIsum = (DIplus + DIminus).replace(0, np.nan); DX = 100 * (abs(DIplus - DIminus) / DIsum); ADX = DX.ewm(alpha=alpha, adjust=False, min_periods=window).mean(); return ADX
def compute_cci(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3; sma = tp.rolling(window=window, min_periods=window).mean()
    mad = tp.rolling(window=window, min_periods=window).apply(lambda x: np.nanmean(np.abs(x - np.nanmean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan)); return cci
def compute_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum(); return obv

def apply_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators."""
    df_copy = df.copy()
    # Ensure 'close' is float before calculations
    df_copy['close'] = df_copy['close'].astype(float)
    df_copy['returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1)).fillna(0)
    df_len = len(df_copy)
    symbol_log = df_copy.get('symbol', 'Unknown') # Get symbol if added earlier
    logger.debug(f"[{symbol_log}] Applying indicators to {df_len} candles.")

    def assign_if_enough_data(col_name, min_len_needed, calculation_lambda):
        if df_len >= min_len_needed:
             try: df_copy[col_name] = calculation_lambda()
             except Exception as e: logger.error(f"[{symbol_log}] Error calculating {col_name}: {e}", exc_info=False); df_copy[col_name] = np.nan
        else:
             logger.debug(f"[{symbol_log}] Skipping {col_name}, need {min_len_needed}, got {df_len}.")
             df_copy[col_name] = np.nan

    assign_if_enough_data('SMA_50', 50, lambda: df_copy['close'].rolling(window=50, min_periods=50).mean())
    assign_if_enough_data('SMA_200', 200, lambda: df_copy['close'].rolling(window=200, min_periods=200).mean())
    assign_if_enough_data('EMA_12', 26, lambda: df_copy['close'].ewm(span=12, adjust=False, min_periods=12).mean())
    assign_if_enough_data('EMA_26', 26, lambda: df_copy['close'].ewm(span=26, adjust=False, min_periods=26).mean())

    # MACD requires EMA_12 and EMA_26
    if df_len >= 26 and 'EMA_12' in df_copy and 'EMA_26' in df_copy and df_copy['EMA_12'].notna().any() and df_copy['EMA_26'].notna().any():
         df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
         # Signal line requires MACD
         if df_len >= 35 and 'MACD' in df_copy and df_copy['MACD'].notna().any():
             assign_if_enough_data('Signal_Line', 35, lambda: df_copy['MACD'].ewm(span=9, adjust=False, min_periods=9).mean())
         else:
             logger.debug(f"[{symbol_log}] Skipping Signal_Line (req MACD/35 candles).")
             df_copy['Signal_Line'] = np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping MACD/Signal_Line (req EMAs/26 candles).")
        df_copy['MACD'], df_copy['Signal_Line'] = np.nan, np.nan

    assign_if_enough_data('RSI', 15, lambda: compute_rsi(df_copy['close'], window=14))
    assign_if_enough_data('ATR', 15, lambda: compute_atr(df_copy, window=14))

    if df_len >= 21:
        try:
            upper, middle, lower = compute_bollinger_bands(df_copy['close'], window=20)
            df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = upper, middle, lower
        except Exception as e:
            logger.error(f"[{symbol_log}] Error calculating Bollinger Bands: {e}", exc_info=False)
            df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping Bollinger Bands (req 21 candles).")
        df_copy['Bollinger_Upper'], df_copy['Bollinger_Middle'], df_copy['Bollinger_Lower'] = np.nan, np.nan, np.nan

    assign_if_enough_data('Momentum', 11, lambda: df_copy['close'] - df_copy['close'].shift(10))

    if df_len >= 17:
        try:
            k, d = compute_stochastic_oscillator(df_copy, window=14, smooth_k=3)
            df_copy['Stochastic_K'], df_copy['Stochastic_D'] = k, d
        except Exception as e:
            logger.error(f"[{symbol_log}] Error calculating Stochastic: {e}", exc_info=False)
            df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan
    else:
        logger.debug(f"[{symbol_log}] Skipping Stochastic (req 17 candles).")
        df_copy['Stochastic_K'], df_copy['Stochastic_D'] = np.nan, np.nan

    assign_if_enough_data('Williams_%R', 15, lambda: compute_williams_r(df_copy, window=14))
    assign_if_enough_data('ADX', 28, lambda: compute_adx(df_copy, window=14))
    assign_if_enough_data('CCI', 21, lambda: compute_cci(df_copy, window=20))
    assign_if_enough_data('OBV', 2, lambda: compute_obv(df_copy))

    logger.debug(f"[{symbol_log}] Finished applying indicators.")
    return df_copy


# --- Statistical Models (GARCH, VaR - LSTM functions removed) ---
def fit_garch_model(returns: pd.Series, symbol_log: str = "Unknown") -> Optional[float]:
    """Fit GARCH(1,1) model. Returns NEXT PERIOD conditional volatility."""
    valid_returns = returns.dropna() * 100
    logger.debug(f"[{symbol_log}] GARCH input len: {len(valid_returns)}")
    if len(valid_returns) < 50:
        logger.warning(f"[{symbol_log}] Skipping GARCH, need 50 returns, got {len(valid_returns)}.")
        return None
    try:
        # Model definition uses upgraded GARCH name
        am = arch_model(valid_returns, vol='GARCH', p=1, q=1, dist='Normal')
        # Suppress convergence warnings during fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(update_freq=0, disp='off', show_warning=False)
        if res.convergence_flag == 0:
            forecasts = res.forecast(horizon=1, reindex=False)
            cond_vol_forecast = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0 # Convert back from variance %^2
            logger.debug(f"[{symbol_log}] GARCH fit successful. Forecast Vol: {cond_vol_forecast:.4f}")
            return float(cond_vol_forecast) if np.isfinite(cond_vol_forecast) else None
        else:
            logger.warning(f"[{symbol_log}] GARCH did not converge (Flag: {res.convergence_flag}).")
            return None
    except Exception as e:
        logger.error(f"[{symbol_log}] GARCH fitting error: {e}", exc_info=False) # Keep logs cleaner
        return None

def calculate_var(returns: pd.Series, confidence_level: float = 0.95, symbol_log: str = "Unknown") -> Optional[float]:
    """Calculate Value at Risk (VaR) at specified confidence level."""
    valid_returns = returns.dropna()
    logger.debug(f"[{symbol_log}] VaR input len: {len(valid_returns)}")
    if len(valid_returns) < 20:
        logger.warning(f"[{symbol_log}] Skipping VaR, need 20 returns, got {len(valid_returns)}.")
        return None
    try:
        var_value = np.percentile(valid_returns, (1.0 - confidence_level) * 100.0)
        logger.debug(f"[{symbol_log}] VaR calculated: {var_value:.4f} at {confidence_level*100:.0f}% confidence.")
        # VaR is usually negative, representing loss
        return float(var_value) if np.isfinite(var_value) else None
    except Exception as e:
        logger.error(f"[{symbol_log}] Error calculating VaR: {e}", exc_info=False)
        return None

# --- LSTM Functions Removed ---
# def prepare_lstm_data(...)
# def build_lstm_model(...)
# def train_lstm_model(...)
# def forecast_with_lstm(...)
# --- END LSTM REMOVED ---


# --- GPT Integration (Prompt Heavily Modified) ---

def gpt_generate_trading_parameters(
    df_with_indicators: pd.DataFrame,
    symbol: str,
    timeframe: str,
    account_balance: float,
    max_leverage: float,
    garch_volatility: Optional[float],
    var95: Optional[float],
    technically_derived_direction: str,
    min_requested_rr: Optional[float] # NEW: Pass the direction from RSI check
) -> str:
    """Generate trading parameters using GPT to EVALUATE a technical signal."""
    log_prefix = f"[{symbol} ({timeframe}) GPT]"
    if openai_client is None:
        logger.warning(f"{log_prefix} OpenAI client not available.")
        return json.dumps({"error": "OpenAI client not available"})

    df_valid = df_with_indicators.dropna(subset=['close', 'RSI', 'ATR']) # Ensure core indicators are present
    if df_valid.empty:
        logger.warning(f"{log_prefix} No valid data (Close/RSI/ATR) for GPT.")
        return json.dumps({"error": "Insufficient indicator data for GPT"})

    latest_data = df_valid.iloc[-1].to_dict()
    technical_indicators = {}

    # Extract finite indicators for context
    for field_name, model_field in IndicatorsData.model_fields.items():
        key_in_df = model_field.alias or field_name
        value = latest_data.get(key_in_df)
        if pd.notna(value) and np.isfinite(value):
             if abs(value) >= 1e4 or (abs(value) < 1e-4 and value != 0):
                 technical_indicators[key_in_df] = f"{value:.3e}"
             else:
                 technical_indicators[key_in_df] = round(float(value), 4)

    current_price = latest_data.get('close')
    if current_price is None or not np.isfinite(current_price):
         logger.error(f"{log_prefix} Missing current price for GPT context")
         return json.dumps({"error": "Missing current price for GPT context"})
    current_price = round(float(current_price), 4)

    garch_vol_str = f"{garch_volatility:.4%}" if garch_volatility is not None else "N/A"
    var95_str = f"{var95:.4%}" if var95 is not None else "N/A"

    # Prepare market context for GPT
    market_info = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": current_price,
        "garch_forecast_volatility": garch_vol_str,
        "value_at_risk_95": var95_str,
        "key_technical_indicators": technical_indicators,
        "potential_signal_direction": technically_derived_direction # Include the signal
    }
    data_json = json.dumps(market_info, indent=2)
    logger.debug(f"{log_prefix} Data prepared for GPT:\n{data_json}")

    # --- REVISED GPT PROMPT ---
    prompt = f"""You are a cryptocurrency trading analyst evaluating a potential trade setup.
A technical signal (RSI + basic SMA trend filter) suggests a potential trade direction: '{technically_derived_direction}'.
Your task is to EVALUATE this signal using the provided market data and technical indicators, and provide actionable parameters if appropriate.

Market Data & Indicators:
{data_json}

Instructions:
1.  **Evaluate Signal:** Assess the provided '{technically_derived_direction}' signal. Look for **confirming** factors (e.g., MACD alignment, price near relevant Bollinger Band, supportive Stochastic/CCI levels) and **strong contradicting** factors (e.g., clear divergence, price hitting major resistance/support against the signal, very low ADX < 15-20 suggesting chop).
2.  **Determine Action:**
    *   If the initial signal has some confirmation OR lacks strong contradictions, **lean towards confirming the `trade_direction`** ('long' or 'short').
    *   Only output `trade_direction: 'hold'` if there are **significant contradictions** from multiple important indicators OR if the market context (e.g., extreme chop, major news event imminent - though you don't have news data) makes the signal very unreliable.
3.  **Refine Parameters (if action suggested):**
    *   `optimal_entry`: Suggest a tactical entry, considering pullbacks/rallies to support/resistance (SMAs, BBands) or breakout/down levels relative to the signal candle. Justify briefly. Use `current_price` only if no better tactical level is apparent.
    *   `stop_loss`: Place logically based on volatility (ATR) or structure.
    # <<< --- CHANGE THIS LINE --- >>>
    *   `take_profit`: Aim for R/R >= {min_requested_rr or 1.5}.
    # <<< --- END OF CHANGE --- >>>
4.  **Provide Confidence:** Assign a `confidence_score` (0.0-1.0) based on the *degree of confirmation* and the *absence of strong contradictions*. A score > 0.6 requires decent confirmation.
5.  **Justify:** Explain your reasoning in the `analysis` sections (`signal_evaluation`, `technical_analysis`, `risk_assessment`, `market_outlook`).

Respond ONLY with a single, valid JSON object containing the specified fields.
"""
    # --- END OF REVISED PROMPT ---

    try:
        logger.info(f"{log_prefix} Sending request to GPT to evaluate '{technically_derived_direction}' signal...")
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Use a capable model
            messages=[
                {"role": "system", "content": "You are a crypto trading analyst evaluating technical signals provided by a user. Respond in JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3, # Lower temperature for more deterministic analysis
            max_tokens=1800 # Allow slightly more tokens for detailed justification
        )
        gpt_output = response.choices[0].message.content
        logger.debug(f"{log_prefix} Raw GPT Output received:\n```json\n{gpt_output}\n```") # Log raw output at DEBUG level
        return gpt_output or "{}"
    except openai.RateLimitError as e:
        logger.error(f"{log_prefix} OpenAI Rate Limit Error: {e}")
        return json.dumps({"error": "OpenAI API rate limit exceeded", "details": str(e)})
    except openai.APIError as e:
        logger.error(f"{log_prefix} OpenAI API Error: {e}", exc_info=False)
        return json.dumps({"error": "OpenAI API error", "details": str(e)})
    except Exception as e:
        logger.error(f"{log_prefix} Error querying OpenAI: {e}", exc_info=False)
        return json.dumps({"error": "Failed to query OpenAI", "details": str(e)})


# --- Parse GPT Parameters (Modified to handle new 'analysis' structure) ---
def parse_gpt_trading_parameters(gpt_output_str: str, symbol_for_log: str = "") -> Dict[str, Any]:
    """Parse GPT-generated trading parameters (evaluation focused)."""
    log_prefix = f"[{symbol_for_log} Parse]"
    # Default structure, including new analysis fields
    parsed_data = {
        'optimal_entry': None, 'stop_loss': None, 'take_profit': None, 'trade_direction': 'hold', # Default to hold
        'leverage': None, 'position_size_usd': None, 'estimated_profit': None, 'confidence_score': 0.0, # Default confidence 0
        'analysis': {'signal_evaluation': None, 'technical_analysis': None, 'risk_assessment': None, 'market_outlook': None, 'raw_text': gpt_output_str}
    }
    try:
        data = json.loads(gpt_output_str)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("GPT output was not a JSON object", gpt_output_str, 0)
        logger.debug(f"{log_prefix} Successfully decoded JSON from GPT.")

        # Helper to safely get float values
        def get_float(key):
            val = data.get(key)
            return float(val) if isinstance(val, (int, float)) and np.isfinite(val) else None

        # Get core trade parameters
        parsed_data['optimal_entry'] = get_float('optimal_entry')
        parsed_data['stop_loss'] = get_float('stop_loss')
        parsed_data['take_profit'] = get_float('take_profit')
        parsed_data['position_size_usd'] = get_float('position_size_usd')
        parsed_data['estimated_profit'] = get_float('estimated_profit')
        parsed_data['confidence_score'] = get_float('confidence_score')

        # Leverage
        leverage_val = data.get('leverage')
        if isinstance(leverage_val, int) and leverage_val >= 1:
            parsed_data['leverage'] = leverage_val
        elif leverage_val is not None:
            logger.warning(f"{log_prefix} Invalid leverage value from GPT: {leverage_val}.")

        # Trade Direction (Crucial - reflects GPT's evaluation)
        direction = data.get('trade_direction')
        if direction in ['long', 'short', 'hold']:
            parsed_data['trade_direction'] = direction
            logger.info(f"{log_prefix} GPT evaluated signal resulted in direction: '{direction}'")
        elif direction:
            logger.warning(f"{log_prefix} Invalid trade_direction from GPT: '{direction}'. Defaulting to 'hold'.")
            parsed_data['trade_direction'] = 'hold'
        else:
             logger.warning(f"{log_prefix} Missing trade_direction from GPT. Defaulting to 'hold'.")
             parsed_data['trade_direction'] = 'hold'


        # Validate parameters *if* GPT suggests a trade
        if parsed_data['trade_direction'] in ['long', 'short']:
             if not all([parsed_data['optimal_entry'], parsed_data['stop_loss'], parsed_data['take_profit'], parsed_data['confidence_score'] is not None]):
                 logger.warning(f"{log_prefix} GPT suggested '{parsed_data['trade_direction']}' but missing Entry/SL/TP/Confidence. Forcing 'hold'.")
                 parsed_data['trade_direction'] = 'hold'
             # Optional: Check R/R ratio validity here
             elif parsed_data['optimal_entry'] and parsed_data['stop_loss'] and parsed_data['take_profit']:
                 risk = abs(parsed_data['optimal_entry'] - parsed_data['stop_loss'])
                 reward = abs(parsed_data['take_profit'] - parsed_data['optimal_entry'])
                 if risk < 1e-9 or reward / risk < 1.0: # Ensure R > 0 and R/R >= 1.0 at least
                      logger.warning(f"{log_prefix} GPT suggested '{parsed_data['trade_direction']}' with invalid R/R ({reward=}, {risk=}). Forcing 'hold'.")
                      parsed_data['trade_direction'] = 'hold'


        # Process analysis section - handle new structure
        analysis_dict = data.get('analysis')
        if isinstance(analysis_dict, dict):
            for key in ['signal_evaluation', 'technical_analysis', 'risk_assessment', 'market_outlook']:
                val = analysis_dict.get(key)
                if isinstance(val, str) and val.strip():
                    parsed_data['analysis'][key] = val.strip()
                elif val is not None:
                    logger.warning(f"{log_prefix} Invalid type or empty value for analysis key '{key}': {type(val)}.")
        elif analysis_dict is not None:
            logger.warning(f"{log_prefix} Invalid type for 'analysis' section: {type(analysis_dict)}.")

        # Add raw text if analysis section was missing/invalid
        if parsed_data['analysis']['signal_evaluation'] is None:
             parsed_data['analysis']['raw_text'] = gpt_output_str # Keep raw if parsing failed

    except json.JSONDecodeError as e:
        logger.error(f"{log_prefix} Failed to decode JSON from GPT: {e}. Raw: {gpt_output_str[:300]}...")
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Failed to parse GPT JSON response. {e}"
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error parsing GPT response: {e}", exc_info=True)
        parsed_data['trade_direction'] = 'hold'
        parsed_data['analysis']['signal_evaluation'] = f"Error: Unexpected error parsing GPT response. {e}"

    # Clear trade params if final decision is 'hold'
    if parsed_data['trade_direction'] == 'hold':
        logger.info(f"{log_prefix} Final direction is 'hold', clearing trade parameters.")
        parsed_data['optimal_entry'] = None
        parsed_data['stop_loss'] = None
        parsed_data['take_profit'] = None
        parsed_data['leverage'] = None
        
