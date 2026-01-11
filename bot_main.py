import discord  # THIS MUST BE LINE 1
from discord.ext import commands
import datetime
import os
from dotenv import load_dotenv

# Initialize Intents AFTER the import
intents = discord.Intents.default()
intents.message_content = True  # Required for the bot to read your commands
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)
import datetime
import json
import logging
import os
import time
import traceback
from typing import List, Optional, Tuple
import asyncio

import discord
from discord.ext import tasks
from dotenv import load_dotenv

from database import db_manager
from services import binance_service, gemini_service, vision_service
from services.trade_vault import TradeVault
from services.news_service import NewsService
from services.report_service import ReportService

# Ensure fundamental store exists even if the vision service doesn't initialize it
if not hasattr(vision_service, "LATEST_FUNDAMENTAL"):
    vision_service.LATEST_FUNDAMENTAL = {"value": None, "timestamp": 0}


# Logging setup
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "bot_debug.log")
OUTPUT_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "bot_output.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("solana_bot")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
console_handler = logging.StreamHandler()
output_file_handler = logging.FileHandler(OUTPUT_LOG_PATH, encoding="utf-8")
output_logger = logging.getLogger("solana_bot_output")
output_logger.setLevel(logging.INFO)

file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
output_file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
output_logger.addHandler(output_file_handler)


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set in environment")
if not DISCORD_CHANNEL_ID:
    raise RuntimeError("DISCORD_CHANNEL_ID not set in environment")

db_manager.initialize_db()
binance_client = binance_service.get_client()
report_service = ReportService(binance_client)
news_service = NewsService()
trade_vault = TradeVault()
CHANNEL_ID_INT = int(DISCORD_CHANNEL_ID)
FOUR_HOURS_IN_SECONDS = 4 * 60 * 60
DEFAULT_SENTIMENT = {"verdict": "NEUTRAL", "confidence": None}
LAST_TRADE: dict = {"symbol": None, "quantity": None, "entry_price": None, "order_id": None}
LAST_BUY_TIMESTAMP: Optional[datetime.datetime] = None
KILL_SWITCH_OVERRIDE: Optional[bool] = None
TRADING_HALTED: bool = False
NEWS_SENTIMENT_CACHE: dict = {"timestamp": 0.0, "sentiment": dict(DEFAULT_SENTIMENT), "headlines": []}
LAST_TRADE_UPDATE: dict = {"timestamp": 0.0, "trade_id": None}


def _get_trade_mode() -> str:
    return binance_service.get_trade_mode()


def _force_buy_enabled() -> bool:
    raw_value = (os.getenv("FORCE_BUY_ENABLED") or "false").strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _safety_enabled() -> bool:
    raw_value = (os.getenv("SAFETY_ENABLED") or "true").strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _get_safety_limits() -> dict:
    def _int_env(name: str, default_value: int) -> int:
        raw = os.getenv(name, str(default_value))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default_value

    def _float_env(name: str, default_value: float) -> float:
        raw = os.getenv(name, str(default_value))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default_value

    return {
        "cooldown_seconds": _int_env("TRADE_COOLDOWN_SECONDS", 14400),
        "buy_cooldown_seconds": _buy_cooldown_seconds(),
        "max_open_trades": _int_env("MAX_OPEN_TRADES", 1),
        "daily_max_trades": _int_env("DAILY_MAX_TRADES", 1),
        "daily_max_loss": _float_env("DAILY_MAX_LOSS_USDT", 20.0),
        "max_trades_per_day": _max_trades_per_day(),
        "max_daily_loss": _max_daily_loss_usdt(),
        "max_position_usdt": _max_position_usdt(),
    }


def _get_kill_switch() -> bool:
    if KILL_SWITCH_OVERRIDE is not None:
        return KILL_SWITCH_OVERRIDE
    raw_value = (os.getenv("KILL_SWITCH") or "false").strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _live_confirmed() -> bool:
    raw_value = (os.getenv("LIVE_CONFIRMED") or "false").strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _max_trades_per_day() -> int:
    raw = os.getenv("MAX_TRADES_PER_DAY", "2")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 2


def _max_daily_loss_usdt() -> float:
    raw = os.getenv("MAX_DAILY_LOSS_USDT", "50")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 50.0


def _max_position_usdt() -> float:
    raw = os.getenv("MAX_POSITION_USDT")
    if raw is None:
        raw = os.getenv("TRADE_AMOUNT_USDT", "0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _buy_cooldown_seconds() -> int:
    raw = os.getenv("BUY_COOLDOWN_MINUTES", "30")
    try:
        return int(float(raw) * 60)
    except (TypeError, ValueError):
        return 1800


def _sentiment_enabled() -> bool:
    raw_value = (os.getenv("SENTIMENT_ENABLED") or "true").strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _sentiment_min_confidence() -> float:
    raw = os.getenv("SENTIMENT_MIN_CONFIDENCE", "0.60")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.60


def _update_interval_seconds() -> int:
    raw = os.getenv("UPDATE_INTERVAL_MINUTES", "240")
    try:
        return int(float(raw) * 60)
    except (TypeError, ValueError):
        return 240 * 60


def _block_on_bearish() -> bool:
    raw_value = (os.getenv("BLOCK_ON_BEARISH") or "true").strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def _news_limit() -> int:
    raw = os.getenv("NEWS_LIMIT", "6")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 6


def _news_refresh_seconds() -> int:
    raw = os.getenv("NEWS_REFRESH_SECONDS", "600")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 600


def _get_last_buy_timestamp() -> Optional[datetime.datetime]:
    global LAST_BUY_TIMESTAMP
    if LAST_BUY_TIMESTAMP is not None:
        return LAST_BUY_TIMESTAMP
    try:
        last_ts = db_manager.get_last_buy_timestamp()
    except Exception:
        logger.exception("Failed to load last buy timestamp")
        return None
    if not last_ts:
        return None
    try:
        return datetime.datetime.fromisoformat(last_ts)
    except ValueError:
        return None


def _safety_snapshot() -> dict:
    safety_enabled = _safety_enabled()
    limits = _get_safety_limits()
    kill_switch = _get_kill_switch() if safety_enabled else False
    cooldown_seconds = limits["cooldown_seconds"]
    buy_cooldown_seconds = limits["buy_cooldown_seconds"]
    max_open_trades = limits["max_open_trades"]
    daily_max_trades = limits["daily_max_trades"]
    daily_max_loss = limits["daily_max_loss"]
    max_trades_per_day = limits["max_trades_per_day"]
    max_daily_loss = limits["max_daily_loss"]
    max_position_usdt = limits["max_position_usdt"]

    try:
        open_trades = len(db_manager.get_open_trades())
    except Exception:
        logger.exception("Failed to load open trades for safety gate")
        open_trades = 0

    try:
        trades_today = db_manager.get_trades_count_today()
    except Exception:
        logger.exception("Failed to load trades today for safety gate")
        trades_today = 0

    try:
        pnl_today = db_manager.get_realized_pnl_today_usdt()
    except Exception:
        logger.exception("Failed to load pnl today for safety gate")
        pnl_today = 0.0

    last_buy_ts = _get_last_buy_timestamp()
    cooldown_remaining = 0
    buy_cooldown_remaining = 0
    if last_buy_ts and cooldown_seconds > 0:
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - last_buy_ts).total_seconds()
        cooldown_remaining = max(0, int(cooldown_seconds - elapsed))
    if last_buy_ts and buy_cooldown_seconds > 0:
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - last_buy_ts).total_seconds()
        buy_cooldown_remaining = max(0, int(buy_cooldown_seconds - elapsed))

    blocked = False
    reason = "OK"
    if TRADING_HALTED:
        blocked = True
        reason = "TRADING_HALTED"
    elif _get_trade_mode() == "live" and not _live_confirmed():
        blocked = True
        reason = "LIVE mode blocked: set LIVE_CONFIRMED=true"
    elif kill_switch:
        blocked = True
        reason = "KILL_SWITCH"
    elif safety_enabled and cooldown_remaining > 0:
        blocked = True
        reason = f"COOLDOWN {cooldown_remaining}s"
    elif buy_cooldown_remaining > 0:
        blocked = True
        reason = f"BUY_COOLDOWN {buy_cooldown_remaining}s"
    elif safety_enabled and max_open_trades >= 0 and open_trades >= max_open_trades:
        blocked = True
        reason = "MAX_OPEN_TRADES"
    elif safety_enabled and daily_max_trades >= 0 and trades_today >= daily_max_trades:
        blocked = True
        reason = "DAILY_MAX_TRADES"
    elif safety_enabled and daily_max_loss >= 0 and pnl_today <= -abs(daily_max_loss):
        blocked = True
        reason = "DAILY_MAX_LOSS"
    elif max_trades_per_day >= 0 and trades_today >= max_trades_per_day:
        blocked = True
        reason = "MAX_TRADES_PER_DAY"
    elif max_daily_loss >= 0 and pnl_today <= -abs(max_daily_loss):
        blocked = True
        reason = "MAX_DAILY_LOSS"
    else:
        trade_amount = binance_service.get_trade_amount(None)
        if max_position_usdt > 0 and trade_amount > max_position_usdt:
            blocked = True
            reason = "MAX_POSITION_USDT"

    return {
        "enabled": safety_enabled,
        "blocked": blocked,
        "reason": reason if blocked else ("DISABLED" if not safety_enabled else reason),
        "cooldown_remaining": cooldown_remaining,
        "buy_cooldown_remaining": buy_cooldown_remaining,
        "open_trades": open_trades,
        "trades_today": trades_today,
        "pnl_today": pnl_today,
        "limits": limits,
    }


def _safety_check() -> Tuple[bool, str]:
    snapshot = _safety_snapshot()
    if snapshot["blocked"]:
        return False, snapshot["reason"]
    if not snapshot["enabled"]:
        return True, "DISABLED"
    return True, "OK"


def _get_trading_status() -> str:
    if TRADING_HALTED:
        return "HALTED"
    snapshot = _safety_snapshot()
    if snapshot["blocked"]:
        return f"BLOCKED({snapshot['reason']})"
    return "ACTIVE"


def _format_block_message(reason: str) -> str:
    if reason.startswith("LIVE mode blocked"):
        return reason
    return f"Blocked: {reason}"


def _get_trade_folder(trade_id: int) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "data", "trades")
    return os.path.join(base_dir, f"trade_{trade_id}")


class ArtifactStore:
    @staticmethod
    def save_snapshot(trade_folder: str, prefix: str, payload: dict) -> str:
        os.makedirs(trade_folder, exist_ok=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.json"
        path = os.path.join(trade_folder, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        return path


def _simulate_paper_buy(pair: str, fallback_price: float) -> Tuple[dict, float]:
    symbol = pair.replace("/", "")
    price = binance_service.get_live_price(symbol) or fallback_price
    if not price:
        raise RuntimeError("Unable to fetch market price for paper trade.")
    spend_amount = binance_service.get_trade_amount(None)
    executed_qty = spend_amount / float(price)
    order_data = {
        "orderId": f"paper-buy-{int(time.time())}",
        "status": "FILLED",
        "executedQty": executed_qty,
        "cummulativeQuoteQty": spend_amount,
    }
    LAST_TRADE.update(
        {
            "symbol": symbol,
            "quantity": executed_qty,
            "entry_price": float(price),
            "order_id": order_data.get("orderId"),
        }
    )
    global LAST_BUY_TIMESTAMP
    LAST_BUY_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc)
    return order_data, float(price)


def _place_oco_for_trade(
    pair: str,
    entry_price: float,
    quantity: float,
    trade_mode: str,
) -> Tuple[Optional[str], float, float]:
    symbol = pair.replace("/", "")
    tp_price = entry_price * 1.02
    sl_trigger = entry_price * 0.99
    sl_limit = sl_trigger * 0.995

    if trade_mode == "paper":
        try:
            trade_id = db_manager.log_signal(pair, "OCO", float(entry_price))
            db_manager.update_trade_status(trade_id, "OCO_PLACED")
        except Exception:
            logger.exception("DB_ERROR: Failed to log paper OCO for %s", pair)
            trade_id = None
        return str(trade_id) if trade_id else None, tp_price, sl_trigger

    order = binance_service.place_oco_order(
        symbol=symbol,
        quantity=float(quantity),
        take_profit_price=tp_price,
        stop_price=sl_trigger,
        stop_limit_price=sl_limit,
    )
    return str(order.get("orderListId")) if order.get("orderListId") is not None else None, tp_price, sl_trigger


def _extract_fill_details(order_data: dict, fallback_price: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    executed_qty = None
    entry_price = None
    try:
        executed_qty = float(order_data.get("executedQty") or 0)
    except (TypeError, ValueError):
        executed_qty = None

    try:
        quote_qty = float(order_data.get("cummulativeQuoteQty") or 0)
    except (TypeError, ValueError):
        quote_qty = 0.0

    if executed_qty and executed_qty > 0 and quote_qty > 0:
        entry_price = quote_qty / executed_qty
    else:
        fills = order_data.get("fills") or []
        total_qty = 0.0
        total_quote = 0.0
        if isinstance(fills, list):
            for fill in fills:
                if not isinstance(fill, dict):
                    continue
                try:
                    qty = float(fill.get("qty") or 0)
                    price = float(fill.get("price") or 0)
                except (TypeError, ValueError):
                    continue
                total_qty += qty
                total_quote += price * qty
        if total_qty > 0:
            executed_qty = total_qty
            entry_price = total_quote / total_qty

    if entry_price is None and fallback_price is not None:
        entry_price = float(fallback_price)
    return executed_qty, entry_price


class OCOView(discord.ui.View):
    def __init__(self, pair: str, price: float):
        super().__init__(timeout=600)
        self.pair = pair
        self.price = price

    @discord.ui.button(label="🛡️ APPLY OCO", style=discord.ButtonStyle.primary)
    async def apply_oco(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        await interaction.response.defer(ephemeral=True)
        symbol = (LAST_TRADE.get("symbol") or self.pair.replace("/", "")).upper()
        quantity = LAST_TRADE.get("quantity")
        entry_price = LAST_TRADE.get("entry_price") or self.price
        trade_mode = _get_trade_mode()
        safety_ok, safety_reason = _safety_check()
        if not safety_ok:
            logger.info("OCO blocked: %s", safety_reason)
            await interaction.followup.send(_format_block_message(safety_reason), ephemeral=True)
            return

        if not quantity or not entry_price:
            await interaction.followup.send(
                "Unable to place OCO: missing executed trade details.", ephemeral=True
            )
            return

        tp_price = entry_price * 1.02
        sl_trigger = entry_price * 0.99
        sl_limit = sl_trigger * 0.995

        if trade_mode == "paper":
            response = f"PAPER OCO placed: TP={tp_price:.6f}, SL={sl_trigger:.6f}"
            await interaction.followup.send(response, ephemeral=True)
            return

        try:
            order = binance_service.place_oco_order(
                symbol=symbol,
                quantity=float(quantity),
                take_profit_price=tp_price,
                stop_price=sl_trigger,
                stop_limit_price=sl_limit,
            )
        except Exception as exc:
            await interaction.followup.send(f"OCO failed: {exc}", ephemeral=True)
            return

        order_list_id = order.get("orderListId")
        order_reports = order.get("orderReports") or []
        order_ids = []
        if isinstance(order_reports, list):
            for report in order_reports:
                if isinstance(report, dict) and report.get("orderId") is not None:
                    order_ids.append(str(report.get("orderId")))

        response = (
            f"✅ OCO placed for {symbol}\n"
            f"Order List ID: {order_list_id}\n"
            f"Order IDs: {', '.join(order_ids) if order_ids else 'N/A'}\n"
            f"TP: {tp_price:.6f}\n"
            f"SL: {sl_trigger:.6f} (limit {sl_limit:.6f})"
        )
        await interaction.followup.send(response, ephemeral=True)


class SignalView(discord.ui.View):
    def __init__(
        self,
        pair: str,
        price: float,
        signal_type: str = "BUY",
        fundamental_context: Optional[str] = None,
    ):
        super().__init__(timeout=600)
        self.pair = pair
        self.price = price
        self.signal_type = signal_type
        self.fundamental_context = fundamental_context

    @discord.ui.button(label="✅ I BOUGHT", style=discord.ButtonStyle.success)
    async def bought(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        await interaction.response.defer(ephemeral=True)
        signal_upper = self.signal_type.upper()
        order_successful = False
        order_status = ""
        error_details = ""
        oco_view: Optional[OCOView] = None
        trade_mode = _get_trade_mode()
        paper_buy = False

        # Step 1: Execute order if it's a real BUY signal
        if signal_upper == "BUY":
            safety_ok, safety_reason = _safety_check()
            if not safety_ok:
                logger.info("BUY blocked: %s", safety_reason)
                await interaction.followup.send(_format_block_message(safety_reason), ephemeral=True)
                return
            symbol = self.pair.replace("/", "")
            if trade_mode == "live":
                if not binance_service.credentials_valid():
                    await interaction.followup.send("❌ CREDENTIALS MISSING", ephemeral=True)
                    return
                try:
                    logger.info("[BUY] Executing Binance BUY for %s...", symbol)
                    order = binance_service.execute_market_order("BUY", symbol=symbol)

                    order_data = order or {}
                    order_status = str(order_data.get("status", "")).upper()
                    logger.info("LIVE EXECUTION: Binance status is %s", order_status)

                    if order_status == "FILLED":
                        order_successful = True
                        oco_view = OCOView(self.pair, self.price)
                        executed_qty, entry_price = _extract_fill_details(order_data, self.price)
                        LAST_TRADE.update(
                            {
                                "symbol": symbol,
                                "quantity": executed_qty,
                                "entry_price": entry_price,
                                "order_id": order_data.get("orderId"),
                            }
                        )
                        global LAST_BUY_TIMESTAMP
                        LAST_BUY_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc)
                    else:
                        # Capture comprehensive error details for non-FILLED statuses
                        error_details = order_data.get("msg") or str(order_data) or "No error message provided."

                except Exception as e:
                    logger.exception("Binance buy failed for %s", self.pair)
                    error_details = str(e)  # Capture exception as error detail
            else:
                try:
                    order_data, price = _simulate_paper_buy(self.pair, self.price)
                except Exception as exc:
                    await interaction.followup.send(f"❌ {exc}", ephemeral=True)
                    return
                order_status = "FILLED"
                paper_buy = True
                order_successful = True
                oco_view = OCOView(self.pair, float(price))
        
        else: # Non-BUY signals are treated as successful for logging purposes
            order_successful = True

        # Step 2: If order failed, send rejection and stop.
        if not order_successful:
            rejection_embed = discord.Embed(
                title="❌ Binance Order Rejected",
                description=f"The market BUY order for **{self.pair}** was not filled.",
                color=discord.Color.red(),
            )
            rejection_embed.add_field(name="Binance API Response", value=f"```{error_details}```", inline=False)
            await interaction.followup.send(embed=rejection_embed, ephemeral=True)
            return

        # Step 3: Log the signal to the database. A failure here is non-critical.
        trade_id: Optional[int] = None
        confirmation_message = ""
        try:
            trade_id = db_manager.log_signal(self.pair, self.signal_type, self.price, self.fundamental_context)
            db_manager.update_trade_status(trade_id, "CONFIRMED")
            logger.info("[SignalView] CONFIRMED trade #%s for %s at %s", trade_id, self.pair, self.price)
            logger.info("Trade logged and confirmed: id=%s pair=%s price=%s", trade_id, self.pair, self.price)
        except Exception:
            logger.exception("DB_ERROR: Failed to log confirmed trade for %s", self.pair)
            # Send a non-blocking warning to the user
            db_fail_warning = (
                "⚠️ **Database Logging Failed**\n"
                "The trade was executed on Binance, but logging it failed. "
                "Please check the bot's console logs for the error."
            )
            await interaction.followup.send(db_fail_warning, ephemeral=True)

        if trade_id:
            try:
                trade_meta = db_manager.get_trade_metadata(trade_id)
                trade_ts = trade_meta[0] if trade_meta else None
                symbol = self.pair.replace("/", "")
                trade_folder = trade_vault.create_trade_folder(trade_id, symbol, trade_ts)
                safety_snapshot = _safety_snapshot()
                meta_payload = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "pair": self.pair,
                    "trade_mode": trade_mode,
                    "risk": _get_trading_status(),
                    "guards": safety_snapshot,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                trade_vault.save_snapshot("meta", meta_payload, trade_folder)
                indicators = binance_service.get_indicators(symbol)
                latest_sentiment = None
                try:
                    latest_sentiment = db_manager.get_latest_sentiment()
                except Exception:
                    logger.exception("Failed to load latest sentiment for buy vault")
                buy_payload = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "trade_id": trade_id,
                    "pair": self.pair,
                    "price": LAST_TRADE.get("entry_price") or self.price,
                    "quantity": LAST_TRADE.get("quantity"),
                    "indicators": indicators,
                    "sentiment": latest_sentiment,
                }
                trade_vault.save_snapshot("buy", buy_payload, trade_folder)
            except Exception:
                output_logger.exception("TradeVault: failed to store buy vault data for trade %s", trade_id)

        # Step 4: Send the final success message to the user
        if signal_upper == "BUY":
            if paper_buy:
                qty_text = f"{LAST_TRADE.get('quantity'):.6f}" if LAST_TRADE.get("quantity") else "N/A"
                price_text = f"{LAST_TRADE.get('entry_price'):.6f}" if LAST_TRADE.get("entry_price") else "N/A"
                confirmation_message = f"PAPER BUY filled: qty={qty_text}, price={price_text}"
            else:
                confirmation_message = (
                    f"✅ **Trade Executed & Logged** (ID: {trade_id or 'N/A'})\n\n"
                    f"Successfully executed a market BUY for **{self.pair}**.\n\n"
                    "Use the OCO button below to protect your position."
                )
        else: # For TEST or other signals
            confirmation_message = f"✅ Signal for {self.pair} logged as trade #{trade_id or 'N/A'}."

        # The oco_view is already prepared if the order was FILLED
        await interaction.followup.send(content=confirmation_message, view=oco_view, ephemeral=True)

    @discord.ui.button(label="❌ SKIP", style=discord.ButtonStyle.danger)
    async def skip(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        try:
            trade_id = db_manager.log_signal(
                self.pair, self.signal_type, self.price, self.fundamental_context
            )
            db_manager.update_trade_status(trade_id, "SKIPPED")
            logger.info("[SignalView] SKIPPED trade #%s for %s at %s", trade_id, self.pair, self.price)
            await interaction.response.send_message("Trade skipped and logged for backtesting.", ephemeral=True)
            logger.info("Trade skipped: id=%s pair=%s price=%s", trade_id, self.pair, self.price)
        except Exception:
            logger.exception("Failed to log skipped trade")
            await interaction.response.send_message("Error logging skip.", ephemeral=True)


async def send_heartbeat(channel: discord.abc.Messageable, status: str) -> None:
    try:
        await channel.send(status)
    except Exception:
        logger.exception("Failed to send heartbeat message")


def _normalize_sentiment(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    text = str(raw) if raw is not None else DEFAULT_SENTIMENT["verdict"]
    return {"verdict": text, "confidence": DEFAULT_SENTIMENT["confidence"]}


def _get_sentiment_from_image(image_bytes: Optional[bytes]) -> dict:
    """Call vision_service safely; if no bytes are provided, return a neutral default."""
    if not image_bytes:
        return dict(DEFAULT_SENTIMENT)
    try:
        raw = vision_service.get_sentiment(image_bytes)
        return _normalize_sentiment(raw)
    except Exception:
        logger.exception("vision_service.get_sentiment failed")
        return dict(DEFAULT_SENTIMENT)


def _format_confidence(value: Optional[object]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _load_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        output_logger.exception("Failed to read JSON file: %s", path)
        return None


def _build_headline_blob(items: List[dict]) -> str:
    lines = []
    for item in items:
        title = item.get("title") or ""
        source = item.get("source") or ""
        if not title:
            continue
        if source:
            lines.append(f"{title} ({source})")
        else:
            lines.append(str(title))
    return "\n".join(lines)


def _apply_sentiment_confidence(sentiment: dict, min_confidence: float) -> dict:
    verdict = (sentiment.get("verdict") or "").upper()
    confidence = sentiment.get("confidence")
    if confidence is None:
        sentiment["verdict"] = "NEUTRAL"
        return sentiment
    try:
        conf_value = float(confidence)
    except (TypeError, ValueError):
        sentiment["verdict"] = "NEUTRAL"
        return sentiment
    if conf_value < min_confidence:
        sentiment["verdict"] = "NEUTRAL"
        return sentiment
    if verdict not in {"BULLISH", "BEARISH"}:
        sentiment["verdict"] = "NEUTRAL"
    return sentiment


def _get_news_sentiment() -> Tuple[dict, List[dict], bool]:
    if not _sentiment_enabled():
        return dict(DEFAULT_SENTIMENT), [], False

    now = time.time()
    if (now - float(NEWS_SENTIMENT_CACHE.get("timestamp", 0))) < _news_refresh_seconds():
        cached_sentiment = NEWS_SENTIMENT_CACHE.get("sentiment") or dict(DEFAULT_SENTIMENT)
        cached_headlines = NEWS_SENTIMENT_CACHE.get("headlines") or []
        if isinstance(cached_sentiment, dict) and isinstance(cached_headlines, list):
            return dict(cached_sentiment), list(cached_headlines), False

    headlines: List[dict] = []
    try:
        headlines = news_service.get_latest_news(limit=_news_limit(), symbol="SOL")
    except Exception:
        logger.exception("Failed to fetch news headlines")
        headlines = []

    if not headlines:
        sentiment = dict(DEFAULT_SENTIMENT)
    else:
        blob = _build_headline_blob(headlines)
        try:
            sentiment = gemini_service.analyze_text_sentiment(blob)
        except Exception:
            logger.exception("Gemini text sentiment failed")
            sentiment = dict(DEFAULT_SENTIMENT)

    NEWS_SENTIMENT_CACHE["timestamp"] = now
    NEWS_SENTIMENT_CACHE["sentiment"] = dict(sentiment)
    NEWS_SENTIMENT_CACHE["headlines"] = list(headlines)
    return dict(sentiment), list(headlines), True


def _get_trend_from_rsi(rsi_value: Optional[float]) -> str:
    """Helper to convert RSI number to a text trend."""
    if rsi_value is None:
        return "NEUTRAL"
    if rsi_value < 35:
        return "BULLISH"
    if rsi_value > 65:
        return "BEARISH"
    return "NEUTRAL"


def _get_fundamental_sentiment() -> Tuple[str, Optional[str], float]:
    """
    Parse the latest vision analysis into a sentiment verdict.

    Returns (fundamental_sentiment, raw_vision_value, vision_timestamp).
    """
    vision_data = vision_service.LATEST_FUNDAMENTAL.get("value")
    vision_timestamp = vision_service.LATEST_FUNDAMENTAL.get("timestamp", 0) or 0
    fundamental_sentiment = "NEUTRAL"

    if vision_data:
        age = time.time() - vision_timestamp
        if age < FOUR_HOURS_IN_SECONDS:
            try:
                parts = vision_data.split(",")
                score_part = parts[0]
                score = float(score_part.split(":")[1].strip())
                if score > 0.3:
                    fundamental_sentiment = "BULLISH"
                elif score < -0.3:
                    fundamental_sentiment = "BEARISH"
            except (IndexError, ValueError) as e:
                logger.error("Could not parse LATEST_FUNDAMENTAL: %s, error: %s", vision_data, e)
        else:
            # Sentiment is stale, reset it
            vision_service.LATEST_FUNDAMENTAL["value"] = None
            vision_service.LATEST_FUNDAMENTAL["timestamp"] = 0
            vision_data = None
            vision_timestamp = 0

    return fundamental_sentiment, vision_data, vision_timestamp


def _contains_fundamental_url(text: str) -> bool:
    url_tokens = (
        "x.com",
        "twitter.com",
        "cryptopanic",
        "coindesk",
        "cointelegraph",
    )
    return any(token in text for token in url_tokens)


def _looks_like_fundamental_text(content: str) -> bool:
    if not content:
        return False
    stripped = content.strip()
    lowered = stripped.lower()
    prefixes = (
        "fa:",
        "fundamental:",
        "news:",
        "x:",
        "cryptopanic:",
        "analysis:",
    )
    if any(lowered.startswith(prefix) for prefix in prefixes):
        return True
    if _contains_fundamental_url(lowered):
        return True
    if len(stripped) >= 200 and not lowered.startswith("!"):
        return True
    return False


async def _maybe_send_trade_update(channel: discord.abc.Messageable) -> None:
    interval_seconds = _update_interval_seconds()
    if interval_seconds <= 0:
        return
    try:
        open_trades = db_manager.get_open_trades()
    except Exception:
        logger.exception("Failed to load open trades for 4H update")
        return

    if not open_trades:
        return

    latest_trade = max(open_trades, key=lambda row: row[0])
    trade_id, pair, signal_type, entry_price, status = latest_trade
    now = time.time()
    last_ts = float(LAST_TRADE_UPDATE.get("timestamp") or 0.0)
    last_trade_id = LAST_TRADE_UPDATE.get("trade_id")
    if last_trade_id == trade_id and (now - last_ts) < interval_seconds:
        return

    symbol = str(pair).replace("/", "")
    price = binance_service.get_live_price(symbol)
    indicators = binance_service.get_indicators(symbol)
    rsi = indicators.get("rsi")
    macd = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    trend = _get_trend_from_rsi(rsi)

    latest_sentiment = None
    try:
        latest_sentiment = db_manager.get_latest_sentiment()
    except Exception:
        logger.exception("Failed to load latest sentiment for 4H update")

    price_text = f"{price:.4f}" if price is not None else "N/A"
    rsi_text = f"{rsi:.2f}" if rsi is not None else "N/A"
    macd_text = f"{macd:.4f}" if macd is not None else "N/A"
    macd_signal_text = f"{macd_signal:.4f}" if macd_signal is not None else "N/A"

    message = (
        f"4H Update | Trade {trade_id} | Price: {price_text} | "
        f"RSI: {rsi_text} | MACD: {macd_text}/{macd_signal_text} | Trend: {trend}"
    )
    if latest_sentiment:
        verdict = str(latest_sentiment.get("verdict") or "N/A").upper()
        conf_text = _format_confidence(latest_sentiment.get("confidence"))
        message += f" | Fundamental: {verdict} ({conf_text})"

    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "trade_id": trade_id,
        "pair": pair,
        "signal_type": signal_type,
        "status": status,
        "entry_price": entry_price,
        "price": price,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": indicators.get("macd_hist"),
        "trend": trend,
        "sentiment": latest_sentiment,
    }

    trade_folder = _get_trade_folder(trade_id)
    ArtifactStore.save_snapshot(trade_folder, "update_4h", payload)
    vault_folder = trade_vault.get_trade_folder(trade_id)
    if not vault_folder:
        try:
            trade_meta = db_manager.get_trade_metadata(trade_id)
            trade_ts = trade_meta[0] if trade_meta else None
        except Exception:
            logger.exception("Failed to load trade metadata for 4H vault update")
            trade_ts = None
        vault_folder = trade_vault.create_trade_folder(trade_id, str(pair), trade_ts)
    trade_vault.save_snapshot("update_4h", payload, vault_folder)
    await channel.send(message)

    LAST_TRADE_UPDATE["timestamp"] = now
    LAST_TRADE_UPDATE["trade_id"] = trade_id


@tasks.loop(minutes=10)
async def monitor_market():
    try:
        channel = bot.get_channel(CHANNEL_ID_INT)
        if channel is None:
            channel = await bot.fetch_channel(CHANNEL_ID_INT)

        sol_price = binance_service.get_live_price("SOLUSDT")
        sol_indicators = binance_service.get_indicators("SOLUSDT")
        sol_rsi = sol_indicators.get("rsi")

        btc_price = binance_service.get_live_price("BTCUSDT")
        btc_indicators = binance_service.get_indicators("BTCUSDT")
        btc_trend = _get_trend_from_rsi(btc_indicators.get("rsi"))
        btc_ok = gemini_service.check_market_gravity(btc_price or 0, btc_trend)

        sentiment, headlines, sentiment_refreshed = _get_news_sentiment()
        sentiment = _apply_sentiment_confidence(sentiment, _sentiment_min_confidence())
        verdict = (sentiment.get("verdict") or "").upper()

        fundamental_sentiment, vision_data, vision_timestamp = _get_fundamental_sentiment()

        safety_snapshot = _safety_snapshot()
        safety_line = (
            "OK" if not safety_snapshot["blocked"] else f"Blocked ({safety_snapshot['reason']})"
        )
        trade_mode = _get_trade_mode()
        trading_status = _get_trading_status()
        heartbeat = _format_heartbeat(
            sol_price,
            sol_rsi,
            btc_price,
            btc_trend,
            btc_ok,
            sentiment,
            len(headlines),
            fundamental_sentiment,
            vision_timestamp,
            safety_line,
            trade_mode,
            trading_status,
        )
        await send_heartbeat(channel, heartbeat)
        await _maybe_send_trade_update(channel)

        if sentiment_refreshed and headlines:
            for item in headlines[:3]:
                title = item.get("title", "")
                if title:
                    output_logger.info("[NEWS] %s", title)

        # Safety Logic: Skip signal if risk is high
        if fundamental_sentiment == "BEARISH":
            logger.info("Skipping signal generation due to HIGH risk (BEARISH fundamental sentiment).")
            return

        if verdict == "BEARISH" and _block_on_bearish():
            confidence = sentiment.get("confidence")
            conf_value = float(confidence) if confidence is not None else 0.0
            if conf_value >= _sentiment_min_confidence():
                reason = (
                    f"Blocking BUY: news sentiment BEARISH (confidence {conf_value:.2f} >= "
                    f"{_sentiment_min_confidence():.2f})."
                )
                logger.info(reason)
                await channel.send(reason)
                return

        if sol_rsi is not None and sol_rsi < 35 and verdict == "BULLISH":
            price = sol_price or 0
            target_profit = price * 1.014
            stop_loss = price * 0.98

            signal_message = (
                f"**Potential BUY Signal for SOL/USDT**\n\n"
                f"**RSI:** {sol_rsi:.2f}\n"
                f"**News Sentiment:** {verdict}\n"
                f"**Fundamental Risk:** {fundamental_sentiment}\n\n"
                f"**Current Price:** ${price:.4f}\n"
                f"**Target Profit:** `${target_profit:.4f}` (+1.4%)\n"
                f"**Stop Loss:** `${stop_loss:.4f}` (-2.0%)\n\n"
                f"Consider placing a BUY order."
            )

            view = SignalView("SOL/USDT", price, fundamental_context=vision_data)
            await channel.send(signal_message, view=view)

    except Exception:
        logger.exception("Error in monitor_market loop")


@monitor_market.before_loop
async def before_monitor():
    await bot.wait_until_ready()
    logger.info("Monitor loop starting")


@bot.event
async def on_ready():
    logger.info("Bot logged in as %s", bot.user)
    if not monitor_market.is_running():
        monitor_market.start()
    if not daily_combo_report.is_running():
        daily_combo_report.start()
    if not track_paper_trades.is_running():
        track_paper_trades.start()
    try:
        await bot.tree.sync()
    except Exception:
        logger.exception("Failed to sync app commands")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # Handle image attachments
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    await message.channel.send("Analyzing image for fundamental sentiment...")
                    image_bytes = await attachment.read()
                    vision_response = vision_service.analyze_image(image_bytes)
                    # The result is stored in vision_service.LATEST_FUNDAMENTAL
                    await message.channel.send("Analysis complete. The next signal will consider this information.")
                    trade_folder = trade_vault.get_active_trade_folder(db_manager) or trade_vault.get_unassigned_folder()
                    trade_vault.save_screenshot(image_bytes, attachment.filename or "image.png", trade_folder)
                    latest_sentiment = None
                    try:
                        latest_sentiment = db_manager.get_latest_sentiment()
                    except Exception:
                        logger.exception("Failed to load latest sentiment for image vault")
                    payload = {
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "source": "image",
                        "raw_response": vision_response,
                        "sentiment": latest_sentiment,
                        "filename": attachment.filename,
                    }
                    trade_vault.save_snapshot("fundamental_image", payload, trade_folder)
                except Exception as e:
                    logger.exception("Failed to process image attachment")
                    await message.channel.send(f"Error processing image: {e}")
    elif message.channel.id == CHANNEL_ID_INT and _looks_like_fundamental_text(message.content or ""):
        try:
            result = gemini_service.analyze_text_sentiment(message.content or "")
            verdict = str(result.get("verdict") or "NEUTRAL").upper()
            confidence = result.get("confidence")
            db_manager.store_text_sentiment(
                raw_text=message.content or "",
                verdict=verdict,
                confidence=confidence,
                source="text",
                channel_id=str(message.channel.id),
                author=str(message.author),
            )
            trade_folder = trade_vault.get_active_trade_folder(db_manager) or trade_vault.get_unassigned_folder()
            text_ts = trade_vault.save_fundamental_text(message.content or "", trade_folder)
            payload = {
                "timestamp": text_ts or datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": "text",
                "verdict": verdict,
                "confidence": confidence,
                "channel_id": str(message.channel.id),
                "author": str(message.author),
                "text_timestamp": text_ts,
            }
            trade_vault.save_snapshot("fundamental_text", payload, trade_folder)
            conf_text = _format_confidence(confidence)
            await message.channel.send(
                "Text analysis complete. Stored as fundamental input "
                f"(sentiment={verdict}, confidence={conf_text})."
            )
        except Exception:
            logger.exception("Failed to process fundamental text")
            await message.channel.send("Error processing fundamental text.")

    await bot.process_commands(message)


@bot.command(name="status")
async def status(ctx):
    try:
        logs = _get_last_logs(5)
        if not logs:
            await ctx.send("No trades logged yet.")
            return
        lines = [
            f"#{row[0]} | {row[1]} | {row[2]} | {row[3]} @ {row[4]} | {row[5]} | dur={row[6]}"
            for row in logs
        ]
        await ctx.send("Last trades:\n" + "\n".join(lines))
    except Exception:
        logger.exception("Failed to fetch status logs")
        await ctx.send("Error fetching logs.")


@bot.command(name="trade_mode")
async def trade_mode(ctx):
    mode = _get_trade_mode()
    force_enabled = _force_buy_enabled()
    await ctx.send(f"TRADE_MODE={mode}, FORCE_BUY_ENABLED={str(force_enabled).lower()}")


@bot.command(name="kill_on")
async def kill_on(ctx):
    global KILL_SWITCH_OVERRIDE
    KILL_SWITCH_OVERRIDE = True
    await ctx.send("Kill switch enabled.")


@bot.command(name="kill_off")
async def kill_off(ctx):
    global KILL_SWITCH_OVERRIDE
    KILL_SWITCH_OVERRIDE = False
    await ctx.send("Kill switch disabled.")


@bot.hybrid_command(name="kill_trading")
async def kill_trading(ctx):
    global TRADING_HALTED
    TRADING_HALTED = True
    await ctx.send("Trading halted. All BUY/OCO blocked until restart.")


@bot.command(name="safety_status")
async def safety_status(ctx):
    snapshot = _safety_snapshot()
    limits = snapshot["limits"]
    kill_switch = _get_kill_switch()
    message = (
        "Safety Status:\n"
        f"enabled={str(snapshot['enabled']).lower()} | kill_switch={str(kill_switch).lower()} | "
        f"cooldown_remaining={snapshot['cooldown_remaining']}s | "
        f"buy_cooldown_remaining={snapshot['buy_cooldown_remaining']}s | "
        f"open_trades={snapshot['open_trades']} | trades_today={snapshot['trades_today']} | "
        f"pnl_today={snapshot['pnl_today']:.2f}\n"
        f"limits: cooldown={limits['cooldown_seconds']}s, buy_cooldown={limits['buy_cooldown_seconds']}s, "
        f"max_open={limits['max_open_trades']}, daily_max_trades={limits['daily_max_trades']}, "
        f"daily_max_loss={limits['daily_max_loss']:.2f}, max_trades_day={limits['max_trades_per_day']}, "
        f"max_daily_loss={limits['max_daily_loss']:.2f}, max_position={limits['max_position_usdt']:.2f}"
    )
    await ctx.send(message)


@bot.command(name="force_buy")
async def force_buy(ctx):
    if not _force_buy_enabled():
        await ctx.send("disabled")
        return
    safety_ok, safety_reason = _safety_check()
    if not safety_ok:
        logger.info("Force BUY blocked: %s", safety_reason)
        await ctx.send(_format_block_message(safety_reason))
        return

    symbol = "SOLUSDT"
    price = binance_service.get_live_price(symbol)
    if price is None:
        await ctx.send("Unable to fetch price for force buy.")
        return

    view = SignalView("SOL/USDT", price, signal_type="BUY")
    await ctx.send(f"Force BUY ready for SOL/USDT at ${price:.4f}.", view=view)


@bot.command(name="e2e_buy_oco")
async def e2e_buy_oco(ctx):
    trade_mode = _get_trade_mode()
    force_enabled = _force_buy_enabled()
    if not force_enabled:
        await ctx.send("disabled")
        return
    safety_ok, safety_reason = _safety_check()
    if not safety_ok:
        logger.info("E2E BUY blocked: %s", safety_reason)
        await ctx.send(_format_block_message(safety_reason))
        return

    pair = "SOL/USDT"
    symbol = "SOLUSDT"
    buy_ok = False
    oco_ok = False
    buy_id = None
    oco_id = None
    qty = None
    entry = None
    tp = None
    sl = None

    if trade_mode == "live" and not binance_service.credentials_valid():
        await ctx.send("❌ CREDENTIALS MISSING")
        return

    try:
        if trade_mode == "paper":
            price = binance_service.get_live_price(symbol)
            if price is None:
                await ctx.send("❌ Unable to fetch market price for paper trade.")
                return
            order_data, entry = _simulate_paper_buy(pair, price)
            qty = LAST_TRADE.get("quantity")
            trade_id = None
            try:
                trade_id = db_manager.log_signal(pair, "BUY", float(entry))
                db_manager.update_trade_status(trade_id, "CONFIRMED")
            except Exception:
                logger.exception("DB_ERROR: Failed to log paper BUY for %s", pair)
            buy_id = str(trade_id) if trade_id else str(order_data.get("orderId"))
            buy_ok = True
        else:
            order = binance_service.execute_market_order("BUY", symbol=symbol)
            order_data = order or {}
            order_status = str(order_data.get("status", "")).upper()
            if order_status != "FILLED":
                await ctx.send(f"❌ Live BUY not filled: {order_data}")
                return
            qty, entry = _extract_fill_details(order_data, None)
            if not qty or not entry:
                await ctx.send("❌ Live BUY missing fill details.")
                return
            LAST_TRADE.update(
                {
                    "symbol": symbol,
                    "quantity": qty,
                    "entry_price": entry,
                    "order_id": order_data.get("orderId"),
                }
            )
            buy_id = str(order_data.get("orderId"))
            buy_ok = True
            global LAST_BUY_TIMESTAMP
            LAST_BUY_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc)

        if not qty or not entry:
            await ctx.send("❌ Unable to resolve quantity/entry for OCO.")
            return

        oco_id, tp, sl = _place_oco_for_trade(pair, float(entry), float(qty), trade_mode)
        oco_ok = oco_id is not None
    except Exception as exc:
        await ctx.send(f"❌ E2E failed: {exc}")
        return

    buy_ok_text = "ok" if buy_ok else "fail"
    oco_ok_text = "ok" if oco_ok else "fail"
    await ctx.send(
        "E2E OK ✅ | "
        f"mode={trade_mode} | "
        f"qty={float(qty):.6f} | "
        f"entry={float(entry):.6f} | "
        f"TP={float(tp):.6f} | "
        f"SL={float(sl):.6f} | "
        f"BUY={buy_ok_text} | "
        f"OCO={oco_ok_text} | "
        f"buy_id={buy_id or 'N/A'} | "
        f"oco_id={oco_id or 'N/A'}"
    )


@bot.hybrid_command(name="open_trades")
async def open_trades(ctx):
    try:
        open_trades = db_manager.get_open_trades()
    except Exception:
        logger.exception("Failed to load open trades")
        await ctx.send("Error loading open trades.")
        return

    if not open_trades:
        await ctx.send("No open trades.")
        return

    trade_amount = binance_service.get_trade_amount(None)
    lines = []
    for trade_id, pair, signal_type, entry_price, status in open_trades:
        if not entry_price:
            continue
        symbol = str(pair).replace("/", "")
        current_price = binance_service.get_live_price(symbol)
        tp_price = float(entry_price) * 1.02
        sl_price = float(entry_price) * 0.99
        qty = trade_amount / float(entry_price)
        if current_price is None:
            pnl_text = "N/A"
        else:
            pnl = (float(current_price) - float(entry_price)) * qty
            pnl_text = f"{pnl:.6f}"

        lines.append(
            f"{pair} | entry={float(entry_price):.6f} | TP={tp_price:.6f} | "
            f"SL={sl_price:.6f} | uPnL={pnl_text}"
        )

    if not lines:
        await ctx.send("No open trades.")
        return

    await ctx.send("\n".join(lines))


def _format_heartbeat(
    sol_price: Optional[float],
    sol_rsi: Optional[float],
    btc_price: Optional[float],
    btc_trend: str,
    btc_ok: bool,
    sentiment: dict,
    headline_count: int,
    fundamental_sentiment: str,
    vision_timestamp: float,
    safety_line: str,
    trade_mode: str,
    trading_status: str,
) -> str:
    sol_rsi_text = f"{sol_rsi:.2f}" if sol_rsi is not None else "N/A"
    sol_price_text = f"{sol_price:.4f}" if sol_price is not None else "N/A"
    btc_price_text = f"{btc_price:.2f}" if btc_price is not None else "N/A"

    if fundamental_sentiment == "BULLISH":
        risk_level = "🟢 LOW"
    elif fundamental_sentiment == "NEUTRAL":
        risk_level = "🟡 MEDIUM"
    else:  # BEARISH
        risk_level = "🔴 HIGH"

    vision_ts_text = (
        datetime.datetime.fromtimestamp(vision_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        if vision_timestamp
        else "N/A"
    )

    return (
        f"**Heartbeat** | **Risk Level:** {risk_level}\n"
        f"SOL: ${sol_price_text} (RSI: {sol_rsi_text}) | "
        f"BTC: ${btc_price_text} (Trend: {btc_trend}, Safe: {btc_ok})\n"
        f"News Sentiment: {sentiment.get('verdict', 'N/A')} "
        f"(Confidence: {_format_confidence(sentiment.get('confidence'))}) | "
        f"Headlines: {headline_count}\n"
        f"Last Vision Analysis: {vision_ts_text}\n"
        f"Trading Mode: {trade_mode} | Trading Status: {trading_status}\n"
        f"Safety: {safety_line}"
    )


def _get_last_logs(limit: int) -> List[Tuple]:
    with db_manager._get_connection() as conn:  # type: ignore[attr-defined]
        cursor = conn.execute(
            "SELECT id, timestamp, pair, signal_type, price, status, duration "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()


@tasks.loop(seconds=30)
async def track_paper_trades():
    if _get_trade_mode() != "paper":
        return

    try:
        open_trades = db_manager.get_open_trades()
    except Exception:
        logger.exception("Failed to load open trades")
        return

    trade_amount = binance_service.get_trade_amount(None)

    for trade_id, pair, signal_type, entry_price, status in open_trades:
        if str(signal_type).upper() != "BUY":
            continue
        if not entry_price:
            continue
        symbol = str(pair).replace("/", "")
        current_price = binance_service.get_live_price(symbol)
        if current_price is None:
            continue

        tp_price = float(entry_price) * 1.02
        sl_price = float(entry_price) * 0.99
        close_status = None
        close_price = None

        if current_price >= tp_price:
            close_status = "CLOSED_TP"
            close_price = current_price
        elif current_price <= sl_price:
            close_status = "CLOSED_SL"
            close_price = current_price

        if not close_status:
            continue

        qty = trade_amount / float(entry_price)
        pnl_usdt = (float(close_price) - float(entry_price)) * qty
        pnl_pct = ((float(close_price) - float(entry_price)) / float(entry_price)) * 100
        closed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        try:
            db_manager.close_trade_with_details(
                trade_id=trade_id,
                status=close_status,
                pnl_usdt=pnl_usdt,
                pnl_pct=pnl_pct,
                closed_price=float(close_price),
                closed_at=closed_at,
            )
            reason = "TP" if close_status == "CLOSED_TP" else "SL"
            output_logger.info(
                "CLOSE | id=%s | reason=%s | entry=%.6f | close=%.6f | pnl_usdt=%.6f | pnl_pct=%.4f",
                trade_id,
                reason,
                float(entry_price),
                float(close_price),
                pnl_usdt,
                pnl_pct,
            )
            trade_folder = trade_vault.get_trade_folder(trade_id)
            if not trade_folder:
                trade_meta = db_manager.get_trade_metadata(trade_id)
                trade_ts = trade_meta[0] if trade_meta else None
                trade_symbol = trade_meta[1] if trade_meta else pair
                trade_folder = trade_vault.create_trade_folder(trade_id, str(trade_symbol), trade_ts)
            sell_payload = {
                "timestamp": closed_at,
                "trade_id": trade_id,
                "pair": pair,
                "price": float(close_price),
                "pnl_usdt": pnl_usdt,
                "pnl_pct": pnl_pct,
                "reason": reason,
            }
            trade_vault.save_snapshot("sell", sell_payload, trade_folder)

            entry_sentiment = None
            buy_data = _load_json_file(os.path.join(trade_folder, "buy.json"))
            if isinstance(buy_data, dict):
                entry_sentiment = buy_data.get("sentiment")
            final_sentiment = None
            try:
                final_sentiment = db_manager.get_latest_sentiment()
            except Exception:
                logger.exception("Failed to load latest sentiment for close summary")
            duration_seconds = None
            try:
                trade_meta = db_manager.get_trade_metadata(trade_id)
                if trade_meta and trade_meta[0]:
                    opened_at = datetime.datetime.fromisoformat(trade_meta[0])
                    closed_dt = datetime.datetime.fromisoformat(closed_at)
                    duration_seconds = int((closed_dt - opened_at).total_seconds())
            except Exception:
                logger.exception("Failed to compute trade duration for summary")

            summary_payload = {
                "timestamp": closed_at,
                "trade_id": trade_id,
                "pair": pair,
                "duration_seconds": duration_seconds,
                "pnl_usdt": pnl_usdt,
                "pnl_pct": pnl_pct,
                "entry_sentiment": entry_sentiment,
                "final_sentiment": final_sentiment,
            }
            trade_vault.finalize_trade(trade_folder, summary_payload)
        except Exception:
            logger.exception("Failed to close trade_id=%s", trade_id)


@tasks.loop(time=datetime.time(hour=0, minute=0, tzinfo=datetime.timezone.utc))
async def daily_combo_report():
    """Generates and sends a daily summary report."""
    channel = bot.get_channel(CHANNEL_ID_INT)
    if channel is None:
        channel = await bot.fetch_channel(CHANNEL_ID_INT)

    try:
        # 1. Get Technical Report
        technical_report = report_service.generate_technical_report(timeframe='1d')
        
        # 2. Get Sentiment Summary
        sentiment_summary = db_manager.get_sentiment_summary(hours=24)

        # 3. Format Embed
        embed = discord.Embed(
            title="Daily Combo Report for SOL/USDT",
            color=discord.Color.blue(),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        
        # Technicals
        embed.add_field(name="📊 Technical Analysis (1D)", value=(
            f"**Price:** ${technical_report['Current_Price']:.4f}\n"
            f"**RSI:** {technical_report['RSI']:.2f}\n"
            f"**SMA 50/200:** ${technical_report['SMA_50']:.2f} / ${technical_report['SMA_200']:.2f}\n"
            f"**MACD:** {technical_report['MACD']:.2f} (Signal: {technical_report['MACD_Signal']:.2f})\n"
            f"**Pattern:** {technical_report['Candlestick_Pattern']}"
        ), inline=False)
        
        # Sentiment
        total_sentiments = sentiment_summary['total']
        if total_sentiments > 0:
            bull_perc = (sentiment_summary['BULLISH']['count'] / total_sentiments) * 100
            bear_perc = (sentiment_summary['BEARISH']['count'] / total_sentiments) * 100
            neut_perc = (sentiment_summary['NEUTRAL']['count'] / total_sentiments) * 100
            embed.add_field(name="📰 Sentiment Analysis (24h)", value=(
                f"**Bullish:** {bull_perc:.1f}% (Avg. Conf: {sentiment_summary['BULLISH']['confidence']:.1f}%)\n"
                f"**Bearish:** {bear_perc:.1f}% (Avg. Conf: {sentiment_summary['BEARISH']['confidence']:.1f}%)\n"
                f"**Neutral:** {neut_perc:.1f}% (Avg. Conf: {sentiment_summary['NEUTRAL']['confidence']:.1f}%)\n"
                f"({total_sentiments} total analyses)"
            ), inline=False)
        else:
            embed.add_field(name="📰 Sentiment Analysis (24h)", value="No image sentiments analyzed in the last 24 hours.", inline=False)

        await channel.send(embed=embed)

    except Exception as e:
        logger.exception("Failed to generate daily combo report")
        await channel.send(f"Error generating daily report: {e}")


@bot.command(name="report")
async def report(ctx, timeframe: str = "daily"):
    """Generates a technical and sentiment report. Timeframe: [daily/weekly/monthly]"""
    timeframe_map = {
        "daily": "1d",
        "weekly": "1w",
        "monthly": "1M",
    }
    if timeframe.lower() not in timeframe_map:
        await ctx.send("Invalid timeframe. Use 'daily', 'weekly', or 'monthly'.")
        return

    try:
        # 1. Get Technical Report
        technical_report = report_service.generate_technical_report(timeframe=timeframe_map[timeframe.lower()])
        
        # 2. Get Sentiment Summary (always last 24h for now)
        sentiment_summary = db_manager.get_sentiment_summary(hours=24)
        
        # 3. Decision Recommendation Logic
        score = 0
        # RSI
        if technical_report['RSI'] < 30: score += 2
        elif technical_report['RSI'] < 40: score += 1
        elif technical_report['RSI'] > 70: score -= 2
        elif technical_report['RSI'] > 60: score -= 1
        # MACD
        if technical_report['MACD'] > technical_report['MACD_Signal']: score += 1
        else: score -= 1
        # Sentiment
        if sentiment_summary['total'] > 0:
            if sentiment_summary['BULLISH']['count'] > sentiment_summary['BEARISH']['count']: score += 1
            elif sentiment_summary['BEARISH']['count'] > sentiment_summary['BULLISH']['count']: score -=1

        if score >= 3: recommendation = "🟢 Strong Buy Signal"
        elif score >= 1: recommendation = "🟩 Leaning Bullish"
        elif score <= -3: recommendation = "🔴 Strong Sell Signal"
        elif score <= -1: recommendation = "🟥 Leaning Bearish"
        else: recommendation = "🟡 Neutral / Hold"


        # 4. Format Embed
        embed = discord.Embed(
            title=f"{timeframe.capitalize()} Report for SOL/USDT",
            color=discord.Color.purple(),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        
        embed.add_field(name="📊 Technical Analysis", value=(
            f"**Price:** ${technical_report['Current_Price']:.4f}\n"
            f"**RSI:** {technical_report['RSI']:.2f}\n"
            f"**SMA 50/200:** ${technical_report['SMA_50']:.2f} / ${technical_report['SMA_200']:.2f}\n"
            f"**MACD:** {technical_report['MACD']:.2f} (Signal: {technical_report['MACD_Signal']:.2f})\n"
            f"**Pattern:** {technical_report['Candlestick_Pattern']}"
        ), inline=False)

        total_sentiments = sentiment_summary['total']
        if total_sentiments > 0:
            bull_perc = (sentiment_summary['BULLISH']['count'] / total_sentiments) * 100
            bear_perc = (sentiment_summary['BEARISH']['count'] / total_sentiments) * 100
            embed.add_field(name="📰 Sentiment Analysis (24h)", value=(
                f"**Bullish:** {bull_perc:.1f}%\n"
                f"**Bearish:** {bear_perc:.1f}%"
            ), inline=True)
        
        embed.add_field(name="🧠 Recommendation", value=f"**{recommendation}** (Score: {score})", inline=True)


        await ctx.send(embed=embed)

    except Exception as e:
        logger.exception(f"Failed to generate {timeframe} report")
        await ctx.send(f"Error generating report: {e}")


def main():
    logger.info("Starting Discord bot")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
