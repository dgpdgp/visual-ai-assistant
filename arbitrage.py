# This file will contain the core arbitrage logic.
import asyncio

class ArbitrageEngine:
    def __init__(self, exchange_manager, pairs, min_spread=0.5):
        self.exchange_manager = exchange_manager
        self.pairs = pairs
        self.min_spread = min_spread
        self.running = False

    async def find_opportunities(self):
        """The main loop to find arbitrage opportunities."""
        self.running = True
        print("Arbitrage engine started.")

        while self.running:
            for pair in self.pairs:
                # Get order books from all exchanges for the pair
                order_books = {}
                exchanges = list(self.exchange_manager.exchanges.keys())
                for exchange_id in exchanges:
                    order_book = await self.exchange_manager.get_order_book(exchange_id, pair)
                    if order_book and order_book['bids'] and order_book['asks']:
                        order_books[exchange_id] = order_book

                # Compare prices between exchanges
                for i in range(len(exchanges)):
                    for j in range(i + 1, len(exchanges)):
                        ex1 = exchanges[i]
                        ex2 = exchanges[j]

                        if ex1 in order_books and ex2 in order_books:
                            sell_price = order_books[ex1]['bids'][0][0]  # Highest bid on ex1
                            buy_price = order_books[ex2]['asks'][0][0]   # Lowest ask on ex2

                            spread = ((sell_price - buy_price) / buy_price) * 100

                            if spread >= self.min_spread:
                                print(f"Arbitrage Opportunity Found!")
                                print(f"  Pair: {pair}")
                                print(f"  Buy on {ex2} at {buy_price}")
                                print(f"  Sell on {ex1} at {sell_price}")
                                print(f"  Spread: {spread:.2f}%")
                                # In a real scenario, you would also check volumes and execute trades here.

            await asyncio.sleep(5) # Wait for 5 seconds before the next check

    def stop(self):
        """Stops the arbitrage engine."""
        self.running = False
        print("Arbitrage engine stopped.")
