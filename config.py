# This file will contain the configuration for the application.
# For example, database connection details and other settings.
import os

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/arbitrage_db")
