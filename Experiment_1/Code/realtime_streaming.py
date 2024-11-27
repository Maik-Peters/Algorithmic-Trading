from alpaca_trade_api.stream import Stream
from utils import log_message

# API-Schlüssel und Basis-URL
API_KEY = "PKS2735LEPXOLUV8MFDM"
SECRET_KEY = "EPLKx7mQseQfJPkYxLoK6iQeqbvpLtK3XKI2nRW9"
BASE_URL = "https://paper-api.alpaca.markets/v2"  # Für Paper Trading

# Websocket-Client einrichten
stream = Stream(API_KEY, SECRET_KEY, BASE_URL)


async def on_quote(data):
    """
    Callback-Funktion für Echtzeitdaten.
    """
    log_message(f"Live-Daten: {data}")

# Hauptfunktion
if __name__ == "__main__":
    symbol = "AAPL"
    stream.subscribe_quotes(on_quote, symbol)
    log_message(f"Streaming gestartet für {symbol}...")
    stream.run()
