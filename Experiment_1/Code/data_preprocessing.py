import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from utils import log_message, ensure_directory_exists

def preprocess_data(input_path, output_path):
    """
    Lädt die Rohdaten, normalisiert sie und fügt technische Indikatoren hinzu.
    """
    data = pd.read_csv(input_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.drop(columns=['trade_count', 'vwap'])
    scaler = MinMaxScaler()
    data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        data[['open', 'high', 'low', 'close', 'volume']]
    )
    data['RSI'] = ta.rsi(data['close'], length=14)
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal'] = macd['MACDs_12_26_9']
    data.fillna(0, inplace=True)
    ensure_directory_exists(output_path.rsplit('/', 1)[0])
    data.to_csv(output_path, index=False)
    log_message(f"Vorverarbeitete Daten gespeichert: {output_path}")

if __name__ == "__main__":
    preprocess_data("Data/raw_data.csv", "Data/preprocessed_data.csv")
