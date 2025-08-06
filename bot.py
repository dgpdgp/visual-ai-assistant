# This file will contain the Telegram bot logic.
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from config import TELEGRAM_BOT_TOKEN
import db
from exchanges import ExchangeManager
from arbitrage import ArbitrageEngine

# States for the /add_api command
class Form(StatesGroup):
    exchange = State()
    api_key = State()
    api_secret = State()

bot = Bot(token=TELEGRAM_BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# A dictionary to hold the running arbitrage engines for each user
arbitrage_engines = {}

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """Handles the /start command."""
    db.add_user(message.from_user.id)
    await message.reply("Welcome to CryptoArbitrage AI PRO!\n"
                        "Use /add_api to add your exchange API keys.\n"
                        "Use /run to start the bot and /stop to stop it.")

@dp.message_handler(commands=['add_api'])
async def add_api_start(message: types.Message):
    """Starts the process of adding API keys."""
    await Form.exchange.set()
    await message.reply("Which exchange do you want to add? (e.g., binance, bybit, kraken)")

@dp.message_handler(state=Form.exchange)
async def process_exchange(message: types.Message, state: FSMContext):
    """Processes the exchange name."""
    await state.update_data(exchange=message.text.lower())
    await Form.next()
    await message.reply("Please enter your API key.")

@dp.message_handler(state=Form.api_key)
async def process_api_key(message: types.Message, state: FSMContext):
    """Processes the API key."""
    await state.update_data(api_key=message.text)
    await Form.next()
    await message.reply("Please enter your API secret.")

@dp.message_handler(state=Form.api_secret)
async def process_api_secret(message: types.Message, state: FSMContext):
    """Processes the API secret and saves the settings."""
    data = await state.get_data()
    exchange = data['exchange']
    api_key = data['api_key']
    api_secret = message.text

    user_id = db.add_user(message.from_user.id)
    db.add_api_settings(user_id, exchange, api_key, api_secret)

    await message.reply(f"API keys for {exchange} saved successfully!")
    await state.finish()

@dp.message_handler(commands=['run'])
async def run_bot(message: types.Message):
    """Starts the arbitrage engine for the user."""
    user_id = db.add_user(message.from_user.id)
    if user_id in arbitrage_engines and arbitrage_engines[user_id].running:
        await message.reply("The bot is already running.")
        return

    settings = db.get_api_settings(user_id)
    if not settings:
        await message.reply("You haven't added any API keys. Use /add_api first.")
        return

    exchange_manager = ExchangeManager()
    for exchange, api_key, api_secret in settings:
        await exchange_manager.add_exchange(exchange, api_key, api_secret)

    # For MVP, we'll hardcode the pairs
    pairs = ['BTC/USDT', 'ETH/USDT']

    engine = ArbitrageEngine(exchange_manager, pairs)
    arbitrage_engines[user_id] = engine
    asyncio.create_task(engine.find_opportunities())

    await message.reply("Arbitrage bot started!")

@dp.message_handler(commands=['stop'])
async def stop_bot(message: types.Message):
    """Stops the arbitrage engine for the user."""
    user_id = db.add_user(message.from_user.id)
    if user_id in arbitrage_engines and arbitrage_engines[user_id].running:
        arbitrage_engines[user_id].stop()
        del arbitrage_engines[user_id]
        await message.reply("Arbitrage bot stopped.")
    else:
        await message.reply("The bot is not running.")

@dp.message_handler(commands=['stats'])
async def show_stats(message: types.Message):
    """Shows some basic stats."""
    # Placeholder for stats
    await message.reply("Stats are not implemented yet.")

def setup_bot(dp):
    # This function can be used to set up all handlers
    # For now, we are defining them globally
    pass

if __name__ == '__main__':
    # This is for running the bot directly for testing
    # In production, we'll use main.py
    import asyncio
    async def main():
        await dp.start_polling(bot)
    asyncio.run(main())
