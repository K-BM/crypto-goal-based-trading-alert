# Telegram bot logic
# app/bot.py
# app/bot.py

import os
from dotenv import load_dotenv
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Telegram bot token from the BotFather

# Start command handler
def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    update.message.reply_text(f"Hi {user.first_name}, I am your goal-based trading assistant!")

# Help command handler
def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Use /setgoal to set your trading goal.")

# Function to start the bot
async def start_bot():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    # Start polling to get messages
    updater.start_polling()

    updater.idle()
