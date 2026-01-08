import pandas as pd
import numpy as np
from binance.client import Client

class ReportService:
    def __init__(self, binance_client: Client):
        self.binance_client = binance_client

    def get_klines(self, timeframe='1d', limit=200):
        klines = self.binance_client.get_klines(symbol='SOLUSDT', interval=timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df

    def calculate_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_sma(self, df, period=50):
        return df['close'].rolling(window=period).mean().iloc[-1]

    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]

    def calculate_fibonacci_levels(self, df):
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        return {
            'level_0': high,
            'level_23.6': high - 0.236 * diff,
            'level_38.2': high - 0.382 * diff,
            'level_50': high - 0.5 * diff,
            'level_61.8': high - 0.618 * diff,
            'level_100': low,
        }

    def identify_candlestick_patterns(self, df):
        # Basic bullish engulfing
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last['close'] > prev['open'] and last['open'] < prev['close'] and last['close'] > prev['close'] and last['open'] < prev['open']:
            return "Bullish Engulfing"
        # Basic bearish engulfing
        if last['open'] > prev['close'] and last['close'] < prev['open'] and last['open'] > prev['open'] and last['close'] < prev['close']:
            return "Bearish Engulfing"
        return "No specific pattern"

    def generate_technical_report(self, timeframe='1d'):
        df = self.get_klines(timeframe=timeframe)
        rsi = self.calculate_rsi(df)
        sma_50 = self.calculate_sma(df, period=50)
        sma_200 = self.calculate_sma(df, period=200)
        macd, signal_line = self.calculate_macd(df)
        fib_levels = self.calculate_fibonacci_levels(df)
        candlestick_pattern = self.identify_candlestick_patterns(df)

        report = {
            'RSI': rsi,
            'SMA_50': sma_50,
            'SMA_200': sma_200,
            'MACD': macd,
            'MACD_Signal': signal_line,
            'Fibonacci_Levels': fib_levels,
            'Candlestick_Pattern': candlestick_pattern,
            'Current_Price': df['close'].iloc[-1]
        }
        return report