# Experiment 1: Baseline-Modell

## **Beschreibung**
In diesem Experiment wird ein einfaches LSTM-Modell entwickelt, um historische Aktienkursdaten zu analysieren und die Preisänderung des nächsten Tages vorherzusagen. Das Ziel ist es, eine Baseline-Performance zu schaffen, die als Vergleichspunkt für komplexere Modelle in zukünftigen Experimenten dient.

---

## **Datenakquise**
- **Datenquelle:** Alpaca API
- **Abrufmethoden:**
  - Historische Daten: Tagesbasierte OHLCV-Daten (Open, High, Low, Close, Volume).
  - Echtzeitdaten: Websockets für aktuelle Marktpreise.
- **Skripte:**
  - `data_acquisition.py` - Lädt historische Daten herunter und speichert sie in `Data/raw_data.csv`&#8203;:contentReference[oaicite:0]{index=0}.
  - `realtime_streaming.py` - Ruft Echtzeitdaten ab und zeigt sie in der Konsole an&#8203;:contentReference[oaicite:1]{index=1}.

---

## **Datenvorverarbeitung**
- **Schritte:**
  1. **Normierung:** Min-Max-Skalierung für numerische Werte (z. B. Open, High, Close).
  2. **Technische Indikatoren:**
     - **RSI (Relative Strength Index):** Misst die Kursstärke.
     - **MACD (Moving Average Convergence Divergence):** Zeigt Markttrends an.
  3. **Fehlende Werte:** Werden mit `0` aufgefüllt.
- **Skript:**
  - `data_preprocessing.py` - Führt die Vorverarbeitung der Rohdaten durch und speichert sie in `Data/preprocessed_data.csv`&#8203;:contentReference[oaicite:2]{index=2}.

---

## **Features**
Die wichtigsten Features und Zielgrößen sind:
1. **Features:**
   - Open, High, Low, Close, Volume
   - RSI, MACD, Signal
2. **Zielgröße:**
   - `Next Day Close`: Vorhergesagter Schlusskurs des nächsten Handelstags.


---

## **Modellarchitektur**
- **Typ:** Long Short-Term Memory (LSTM).
- **Schichten:**
  - Eingabeschicht: 7 Features (Open, High, Low, Close, Volume, RSI, MACD).
  - Versteckte Schicht: 64 Neuronen.
  - Ausgabeschicht: 1 Neuron (Preisvorhersage).
- **Optimierer:** Adam mit einer Lernrate von 0.001.
- **Verlustfunktion:** Mean Squared Error (MSE).
- **Sequenzlänge:** 1 Tag (univariate Zeitreihe).

---

## **Ausführung**
### **1. Datenakquise**
- Laden Sie historische Daten:
  ```bash
  python Code/data_acquisition.py
  
- Echtzeitdaten streamen:
  ```bash
  python Code/realtime_streaming.py
  
- Vorverarbeiten der Rohdaten:
  ```bash
  python Code/data_preprocessing.py

- Trainieren Sie das Modell:
  ```bash
  python Code/train_model.py
