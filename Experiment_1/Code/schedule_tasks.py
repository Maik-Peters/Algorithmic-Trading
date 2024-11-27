import schedule
import time
from data_acquisition import fetch_historical_data
from utils import log_message

def scheduled_task():
    """
    Regelmäßige Aufgabe, um Daten herunterzuladen.
    """
    fetch_historical_data("AAPL", "2022-01-01", "2022-12-31", "Experiment_1/Data/raw_data.csv")

# Zeitplan festlegen
schedule.every().day.at("18:00").do(scheduled_task)

log_message("Scheduler läuft...")
while True:
    schedule.run_pending()
    time.sleep(1)