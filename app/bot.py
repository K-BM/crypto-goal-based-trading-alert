# app/bot.py

import os
import asyncio
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler,
    MessageHandler, filters, ConversationHandler, ContextTypes
)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# States
TARGET, ASSET, TIME_HORIZON, RISK = range(4)
user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    user_states[user_id] = {}
    await update.message.reply_text("Welcome! What's your profit target in % (e.g., 3)?")
    return TARGET

async def set_target(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    try:
        # Replace commas with periods and convert to float
        target_pct = float(update.message.text.replace(',', '.'))
        user_states[user_id]['target_pct'] = target_pct
        await update.message.reply_text("Which asset? (e.g., BTC)")
        return ASSET
    except ValueError:
        # Handle invalid input
        await update.message.reply_text("Invalid input. Please enter a valid number (e.g., 3 or 3.5).")
        return TARGET

async def set_asset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    user_states[user_id]['asset'] = update.message.text.upper()
    await update.message.reply_text("What’s your time horizon? (e.g., 2 days, 4 hours)")
    return TIME_HORIZON

async def set_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    user_states[user_id]['time_horizon'] = update.message.text
    keyboard = [['low', 'medium', 'high']]
    await update.message.reply_text(
        "Select your risk level:",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    )
    return RISK

async def set_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    user_states[user_id]['risk'] = update.message.text.lower()
    await update.message.reply_text(f"✅ Got it:\n\n{user_states[user_id]}")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END

async def start_bot():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            TARGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_target)],
            ASSET: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_asset)],
            TIME_HORIZON: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_time)],
            RISK: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_risk)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)
    print("🤖 Telegram bot is starting...")

    # Properly initialize the application
    await app.initialize()
    await app.start()
    print("🤖 Telegram bot is polling...")
    await app.updater.start_polling()
    await asyncio.Event().wait()  # Keep the bot running indefinitely
    await app.stop()
    await app.shutdown()
