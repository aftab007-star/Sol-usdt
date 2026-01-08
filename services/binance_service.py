import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from binance.client import Client

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)
print(
    f"DEBUG: Found Key: {'Yes' if os.getenv('BINANCE_API_KEY') else 'No'} | "
    f"Found Secret: {'Yes' if os.getenv('BINANCE_SECRET') else 'No'}"
)
api_secret = os.getenv("BINANCE_SECRET") or ""
print(f"SECRET CHECK: {api_secret[:4]}...")

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "bot_debug.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Normalize and validate API credentials
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = api_secret

if not API_SECRET:
    print("❌ CRITICAL: NO SECRET KEY FOUND IN .ENV")

TRADE_AMOUNT_USDT_ENV = os.getenv("TRADE_AMOUNT_USDT")


def _init_client() -> Client:
    logger.info("Initializing Binance client")
    client = Client(
        api_key=API_KEY,
        api_secret=API_SECRET,
        requests_params={"timeout": 10},
        verbose=True,
    )
    logger.info("Binance client initialized")
    return client


_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = _init_client()
    return _client


def credentials_valid() -> bool:
    """Return True when both API key and secret are available."""
    return bool(API_KEY and API_SECRET)


def get_live_price(symbol: str) -> Optional[float]:
    """Return the latest price for the symbol (e.g., SOLUSDT or BTCUSDT)."""
    try:
        client = get_client()
        logger.info("Requesting live price for %s", symbol)
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        logger.info("Received price for %s: %s", symbol, price)
        return price
    except Exception as exc:
        logger.exception("Failed to fetch live price for %s: %s", symbol, exc)
        return None


def get_indicators(symbol: str) -> Dict[str, Optional[float]]:
    """Compute 4h RSI and MACD for the symbol."""
    try:
        client = get_client()
        logger.info("Requesting 4h klines for %s", symbol)
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_4HOUR, limit=200)
        closes = pd.Series([float(kline[4]) for kline in klines], dtype=float)

        rsi = _rsi(closes)
        macd, macd_signal, macd_hist = _macd(closes)

        logger.info(
            "Indicators for %s -> RSI: %.2f, MACD: %.4f, Signal: %.4f, Hist: %.4f",
            symbol,
            rsi,
            macd,
            macd_signal,
            macd_hist,
        )
        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }
    except Exception as exc:
        logger.exception("Failed to compute indicators for %s: %s", symbol, exc)
        return {"rsi": None, "macd": None, "macd_signal": None, "macd_hist": None}


def _rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])


def _macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return float(macd.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_hist.iloc[-1])


def get_mock_indicators() -> Dict[str, Optional[float]]:
    """Return mock indicators to trigger signal logic in tests."""
    return {
        "rsi": 32.0,  # Oversold threshold trigger
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
    }


def place_order(side: str, amount: float, symbol: str = "SOLUSDT") -> str:
    """
    Execute a market order on Binance and return the order id.

    Args:
        side: "BUY" or "SELL".
        amount: Quote amount in USDT to spend.
        symbol: Trading pair symbol, defaults to SOLUSDT.
    """
    try:
        client = get_client()
        logger.info("Placing %s order for %s %s", side, amount, symbol)
        print(">>> REACHED BINANCE API CALL")
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quoteOrderQty=amount,
        )
        print(f"RAW BINANCE RESPONSE: {order}")
        order_id = str(order.get("orderId"))
        logger.info("%s order placed. order_id=%s", side, order_id)
        return order_id
    except Exception as exc:
        logger.exception("Failed to place %s order for %s %s: %s", side, amount, symbol, exc)
        raise


def _get_trade_amount(amount_usdt: Optional[float]) -> float:
    """
    Resolves the trade amount. The spend is strictly enforced to what is configured 
    in the .env file (TRADE_AMOUNT_USDT), defaulting to 15.0 USDT if not specified.
    """
    if amount_usdt is not None:
        return float(amount_usdt)

    # Fall back to environment variable, with a default of 15.0, as per user's preference.
    resolved_env = os.getenv("TRADE_AMOUNT_USDT", "15.0")
    
    try:
        resolved_float = float(resolved_env)
        if resolved_float < 10.0:
            logger.warning(
                "Configured TRADE_AMOUNT_USDT is %s, which is below the typical Binance minimum of 10 USDT and may fail.",
                resolved_float
            )
        return resolved_float
    except (ValueError, TypeError):
        logger.error(
            "Invalid value for TRADE_AMOUNT_USDT in .env: '%s'. Falling back to 15.0.",
            resolved_env,
        )
        return 15.0


def execute_market_order(side: str, amount_usdt: Optional[float] = None, symbol: str = "SOLUSDT") -> dict:
    """
    Execute a synchronous market order using quote amount (USDT).

    Args:
        side: "BUY" or "SELL".
        amount_usdt: Quote amount in USDT to spend.
        symbol: Trading pair symbol, defaults to SOLUSDT.
    """
    spend_amount = _get_trade_amount(amount_usdt)
    client = get_client()
    side_upper = side.upper()
    logger.info("Placing %s market order for %s USDT on %s", side_upper, spend_amount, symbol)

    try:
        print(f"[binance_service] Sending MARKET {side_upper} for {symbol} (quote {spend_amount} USDT)")
        order = client.create_order(
            symbol=symbol,
            side=side_upper,
            type=Client.ORDER_TYPE_MARKET,
            quoteOrderQty=spend_amount,
        )
        print(f"[binance_service] Raw order response: {order}")
        order_id = order.get("orderId")
        order_status = str(order.get("status", "")).upper()
        logger.info("Market %s placed: order_id=%s status=%s", side_upper, order_id, order_status)
        return order
    except Exception as exc:
        print("BINANCE ORDER ERROR:", exc)
        response_attr = getattr(exc, "response", None)
        if response_attr:
            try:
                print("ERROR RESPONSE BODY:", response_attr)
            except Exception:
                print("ERROR RESPONSE BODY: <unprintable>")
        elif exc.args:
            print("ERROR ARGS:", exc.args)
        logger.exception("Failed to execute market %s for %s USDT on %s", side_upper, spend_amount, symbol)
        raise


def execute_trade(side: str, symbol: str = "SOLUSDT") -> dict:
    """
    Execute a market buy/sell using Binance helper methods and TRADE_AMOUNT_USDT from env.

    Args:
        side: "BUY" or "SELL".
        symbol: Trading pair symbol, defaults to SOLUSDT.
    """
    amount = _get_trade_amount(None)
    client = get_client()
    side_upper = side.upper()
    try:
        if side_upper == "BUY":
            order = client.order_market_buy(symbol=symbol, quoteOrderQty=amount)
        elif side_upper == "SELL":
            order = client.order_market_sell(symbol=symbol, quoteOrderQty=amount)
        else:
            raise ValueError(f"Unsupported side: {side}")
        logger.info("Market %s executed via execute_trade. order_id=%s", side_upper, order.get("orderId"))
        return order
    except Exception as exc:
        logger.exception("Failed to execute_trade side=%s symbol=%s amount=%s: %s", side, symbol, amount, exc)
        raise


def place_oco_order(
    symbol: str,
    quantity: float,
    take_profit_price: float,
    stop_price: float,
    stop_limit_price: float,
) -> dict:
    """
    Place an OCO order for take-profit and stop-loss.

    Note: quantity must be base asset quantity (not quote).
    """
    client = get_client()
    try:
        order = client.create_oco_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            quantity=quantity,
            price=str(round(take_profit_price, 4)),
            stopPrice=str(round(stop_price, 4)),
            stopLimitPrice=str(round(stop_limit_price, 4)),
            stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC,
        )
        logger.info("OCO order placed: orderListId=%s", order.get("orderListId"))
        return order
    except Exception as exc:
        logger.exception(
            "Failed to place OCO order symbol=%s qty=%s tp=%s sl=%s sllimit=%s: %s",
            symbol,
            quantity,
            take_profit_price,
            stop_price,
            stop_limit_price,
            exc,
        )
        raise
