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
from services.report_service import ReportService

# Ensure fundamental store exists even if the vision service doesn't initialize it
if not hasattr(vision_service, "LATEST_FUNDAMENTAL"):
    vision_service.LATEST_FUNDAMENTAL = {"value": None, "timestamp": 0}


# Logging setup
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "bot_debug.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("solana_bot")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set in environment")
if not DISCORD_CHANNEL_ID:
    raise RuntimeError("DISCORD_CHANNEL_ID not set in environment")

db_manager.initialize_db()
report_service = ReportService(binance_service.client)
CHANNEL_ID_INT = int(DISCORD_CHANNEL_ID)
FOUR_HOURS_IN_SECONDS = 4 * 60 * 60
DEFAULT_SENTIMENT = {"verdict": "NEUTRAL", "confidence": "N/A"}


class OCOView(discord.ui.View):
    def __init__(self, pair: str, price: float):
        super().__init__(timeout=600)
        self.pair = pair
        self.price = price

    @discord.ui.button(label="🛡️ APPLY OCO", style=discord.ButtonStyle.primary)
    async def apply_oco(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        await interaction.response.send_message(
            f"OCO controls available for {self.pair} at price {self.price}.", ephemeral=True
        )


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

        # Step 1: Execute order if it's a real BUY signal
        if signal_upper == "BUY":
            if not binance_service.credentials_valid():
                await interaction.followup.send("❌ CREDENTIALS MISSING", ephemeral=True)
                return
            try:
                symbol = self.pair.replace("/", "")
                logger.info("[BUY] Executing Binance BUY for %s...", symbol)
                order = binance_service.execute_market_order("BUY", symbol=symbol)
                
                order_data = order or {}
                order_status = str(order_data.get("status", "")).upper()
                logger.info("LIVE EXECUTION: Binance status is %s", order_status)

                if order_status == "FILLED":
                    order_successful = True
                    oco_view = OCOView(self.pair, self.price)
                else:
                    # Capture comprehensive error details for non-FILLED statuses
                    error_details = order_data.get("msg") or str(order_data) or "No error message provided."

            except Exception as e:
                logger.exception("Binance buy failed for %s", self.pair)
                error_details = str(e) # Capture exception as error detail
        
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

        # Step 4: Send the final success message to the user
        if signal_upper == "BUY":
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

        # Use vision-based sentiment exclusively
        sentiment = _get_sentiment_from_image(None)
        verdict = (sentiment.get("verdict") or "").upper()

        fundamental_sentiment, vision_data, vision_timestamp = _get_fundamental_sentiment()

        heartbeat = _format_heartbeat(
            sol_price, sol_rsi, btc_price, btc_trend, btc_ok, sentiment, fundamental_sentiment, vision_timestamp
        )
        await send_heartbeat(channel, heartbeat)

        # Safety Logic: Skip signal if risk is high
        if fundamental_sentiment == "BEARISH":
            logger.info("Skipping signal generation due to HIGH risk (BEARISH fundamental sentiment).")
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
                    vision_service.analyze_image(image_bytes)
                    # The result is stored in vision_service.LATEST_FUNDAMENTAL
                    await message.channel.send("Analysis complete. The next signal will consider this information.")
                except Exception as e:
                    logger.exception("Failed to process image attachment")
                    await message.channel.send(f"Error processing image: {e}")

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


def _format_heartbeat(
    sol_price: Optional[float],
    sol_rsi: Optional[float],
    btc_price: Optional[float],
    btc_trend: str,
    btc_ok: bool,
    sentiment: dict,
    fundamental_sentiment: str,
    vision_timestamp: float,
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
        f"News Sentiment: {sentiment.get('verdict', 'N/A')} (Confidence: {sentiment.get('confidence', 'N/A')})\n"
        f"Last Vision Analysis: {vision_ts_text}"
    )


def _get_last_logs(limit: int) -> List[Tuple]:
    with db_manager._get_connection() as conn:  # type: ignore[attr-defined]
        cursor = conn.execute(
            "SELECT id, timestamp, pair, signal_type, price, status, duration "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()


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
