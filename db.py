# This file will contain the database setup and query functions.
import psycopg2
from config import DATABASE_URL

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def add_user(telegram_id):
    """Adds a new user to the database if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (telegram_id) VALUES (%s) ON CONFLICT (telegram_id) DO NOTHING;", (telegram_id,))
    conn.commit()
    cur.execute("SELECT id FROM users WHERE telegram_id = %s;", (telegram_id,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()
    return user_id

def add_api_settings(user_id, exchange, api_key, api_secret):
    """Adds or updates API settings for a user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO settings (user_id, exchange, api_key, api_secret)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id, exchange) DO UPDATE SET
            api_key = EXCLUDED.api_key,
            api_secret = EXCLUDED.api_secret;
    """, (user_id, exchange, api_key, api_secret))
    conn.commit()
    cur.close()
    conn.close()

def get_api_settings(user_id):
    """Gets all API settings for a user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT exchange, api_key, api_secret FROM settings WHERE user_id = %s;", (user_id,))
    settings = cur.fetchall()
    cur.close()
    conn.close()
    return settings

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()

    # WARNING: API keys are stored in plain text in this MVP.
    # In a production environment, they should be encrypted.

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            exchange VARCHAR(255) NOT NULL,
            api_key VARCHAR(255) NOT NULL,
            api_secret VARCHAR(255) NOT NULL,
            min_spread FLOAT DEFAULT 0.5,
            trade_volume FLOAT DEFAULT 0.01,
            UNIQUE(user_id, exchange)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            exchange_buy VARCHAR(255),
            exchange_sell VARCHAR(255),
            pair VARCHAR(255),
            buy_price FLOAT,
            sell_price FLOAT,
            volume FLOAT,
            profit FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    print("Database initialized.")
