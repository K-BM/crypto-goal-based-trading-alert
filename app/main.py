# app/main.py
import asyncio
from bot import start_bot  # Import the start_bot function

if __name__ == "__main__":
    # Directly run the start_bot function using asyncio.run()
    asyncio.run(start_bot())