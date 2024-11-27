from alpaca_trade_api.rest import REST, TimeFrame
from utils import log_message, ensure_directory_exists

# API-Schlüssel und Basis-URL
API_KEY = "PKS2735LEPXOLUV8MFDM"
SECRET_KEY = "EPLKx7mQseQfJPkYxLoK6iQeqbvpLtK3XKI2nRW9"
BASE_URL = "https://paper-api.alpaca.markets/v2"

# Verbindung einrichten
api = REST(API_KEY, SECRET_KEY, BASE_URL)

def fetch_historical_data(symbol, start_date, end_date, save_path):
    """
    Lädt historische OHLCV-Daten von Alpaca herunter und speichert sie als CSV.
    """
    ensure_directory_exists(save_path.rsplit('/', 1)[0])
    historical_data = api.get_bars(
        symbol, TimeFrame.Day, start=start_date, end=end_date
    ).df
    historical_data.to_csv(save_path)
    log_message(f"Historische Daten für {symbol} gespeichert: {save_path}")

def fetch_latest_price(symbol):
    """
    Ruft den aktuellen Preis einer Aktie ab.
    """
    latest_trade = api.get_latest_trade(symbol)
    log_message(f"Aktuelle Daten für {symbol}: {latest_trade}")

if __name__ == "__main__":
    fetch_historical_data("AAPL", "2022-01-01", "2022-12-31", "Data/raw_data.csv")
    fetch_latest_price("AAPL")
