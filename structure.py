import requests
from config import TELEGRAM_CONFIG

# Verify token
me = requests.get(f"https://api.telegram.org/bot{TELEGRAM_CONFIG['bot_token']}/getMe").json()
print("Bot Status:", me)

# Get updates with long polling
updates = requests.get(
    f"https://api.telegram.org/bot{TELEGRAM_CONFIG['bot_token']}/getUpdates",
    params={'timeout': 10}  # Wait for new messages
).json()
print("Raw Updates:", updates)

# Extract chat IDs
if updates['result']:
    print("Active Chat IDs:", [m['message']['chat']['id'] for m in updates['result']])
else:
    print("No messages received - Have you sent /start to the bot?")