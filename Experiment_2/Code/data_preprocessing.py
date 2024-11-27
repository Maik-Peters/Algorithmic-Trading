import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import log_message, ensure_directory_exists

def preprocess_data(output_path):
    """
    Kombiniert Daten von mehreren Symbolen und skaliert sie.
    """
    # Daten laden
    aapl = pd.read_csv("Data/AAPL_raw.csv")
    msft = pd.read_csv("Data/MSFT_raw.csv")
    spy = pd.read_csv("Data/SPY_raw.csv")

    # Skaliere relevante Spalten
    scaler = MinMaxScaler()
    aapl_scaled = scaler.fit_transform(aapl[['open', 'high', 'low', 'close', 'volume']])
    msft_scaled = scaler.fit_transform(msft[['close']])
    spy_scaled = scaler.fit_transform(spy[['close']])

    # Kombinierte Daten
    combined = pd.DataFrame(aapl_scaled, columns=['aapl_open', 'aapl_high', 'aapl_low', 'aapl_close', 'aapl_volume'])
    combined['msft_close'] = msft_scaled
    combined['spy_close'] = spy_scaled

    ensure_directory_exists(output_path.rsplit('/', 1)[0])
    combined.to_csv(output_path, index=False)
    log_message(f"Vorverarbeitete Daten gespeichert: {output_path}")

if __name__ == "__main__":
    preprocess_data("Data/preprocessed_data.csv")
