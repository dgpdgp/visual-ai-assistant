# This is the main entry point for the application.
import asyncio
from bot import dp, bot
from db import init_db

async def on_startup():
    """Initializes the database when the bot starts."""
    print("Initializing database...")
    init_db()
    print("Database initialized.")
    print("Bot started.")

async def main():
    dp.startup.register(on_startup)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
