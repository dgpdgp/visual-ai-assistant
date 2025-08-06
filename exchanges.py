# This file will contain the logic for interacting with exchanges.
import ccxt.async_support as ccxt

class ExchangeManager:
    def __init__(self):
        self.exchanges = {}

    async def add_exchange(self, exchange_id, api_key, secret):
        """Adds and initializes an exchange instance."""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchanges[exchange_id] = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True, # Important for not getting banned
            })
            print(f"Exchange '{exchange_id}' added.")
        except AttributeError:
            print(f"Error: Exchange '{exchange_id}' not found in ccxt.")

    async def get_order_book(self, exchange_id, pair):
        """Fetches the order book for a given pair from an exchange."""
        if exchange_id not in self.exchanges:
            return None
        try:
            order_book = await self.exchanges[exchange_id].fetch_order_book(pair)
            return order_book
        except Exception as e:
            print(f"Error fetching order book from {exchange_id}: {e}")
            return None

    async def get_balance(self, exchange_id, currency):
        """Fetches the balance for a specific currency from an exchange."""
        if exchange_id not in self.exchanges:
            return None
        try:
            balance = await self.exchanges[exchange_id].fetch_balance()
            return balance.get(currency, {}).get('free', 0)
        except Exception as e:
            print(f"Error fetching balance from {exchange_id}: {e}")
            return 0

    async def create_market_order(self, exchange_id, pair, side, amount):
        """Creates a market order on an exchange."""
        if exchange_id not in self.exchanges:
            return None
        try:
            order = await self.exchanges[exchange_id].create_market_order(pair, side, amount)
            return order
        except Exception as e:
            print(f"Error creating market order on {exchange_id}: {e}")
            return None

    async def close(self):
        """Closes all exchange connections."""
        for exchange in self.exchanges.values():
            await exchange.close()
