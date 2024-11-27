# Experiment 2: Multi-Input LSTM mit Attention

## **Beschreibung**
In diesem Experiment wird ein Multi-Input LSTM-Modell entwickelt, das mehrere Datenquellen wie Zielaktie, korrelierte Aktien und Marktindizes kombiniert. Ein Aufmerksamkeitsmechanismus wird verwendet, um wichtige Faktoren stärker zu gewichten und die Vorhersagegenauigkeit zu verbessern.

---

## **Datenakquise**
- **Datenquellen:** Alpaca API
- **Symbole:**
  - Zielaktie: `AAPL`
  - Korrelierte Aktien: `MSFT`
  - Marktindex: `SPY`
- **Abrufmethoden:**
  - Historische Daten: Tagesbasierte OHLCV-Daten.
  - Echtzeitdaten: Über Websockets (optional).
- **Skripte:**
  - `data_acquisition.py`: Lädt historische Daten für alle Symbole herunter und speichert sie in `Data/{SYMBOL}_raw.csv`.

---

## **Datenvorverarbeitung**
- **Ziel:** Kombination von Haupt- und Nebenfaktoren in einer Multi-Input-Struktur.
- **Schritte:**
  1. Normierung aller numerischen Werte mit Min-Max-Skalierung.
  2. Kombination der Daten von `AAPL`, `MSFT` und `SPY` in eine einzige CSV-Datei.
- **Skript:**
  - `data_preprocessing.py`: Führt die Vorverarbeitung durch und speichert die Ergebnisse in `Data/preprocessed_data.csv`.

---

## **Features**
Die wichtigsten Features und Zielgrößen sind:
1. **Hauptfaktor (AAPL):**
   - Open, High, Low, Close, Volume
2. **Nebenfaktoren:**
   - `MSFT`: Schlusskurs.
   - `SPY`: Schlusskurs.
3. **Zielgröße:**
   - `AAPL Close`: Vorhergesagter Schlusskurs des nächsten Handelstags.

Weitere Details finden sich in der Datei `features.json`.

---

## **Modellarchitektur**
- **Typ:** Multi-Input LSTM mit Attention.
- **Eingaben:**
  - **Hauptfaktor:** LSTM verarbeitet die AAPL-Daten.
  - **Nebenfaktoren:** LSTM verarbeitet die Daten von MSFT und SPY.
- **Schichten:**
  - Zwei LSTM-Schichten: Eine für Hauptfaktor, eine für Nebenfaktoren.
  - Aufmerksamkeitsmechanismus zur Gewichtung wichtiger Sequenzen.
  - Fully Connected Layer kombiniert die Ausgaben der LSTMs.
- **Optimierer:** Adam mit einer Lernrate von 0.001.
- **Verlustfunktion:** Mean Squared Error (MSE).

---

## **Ausführung**
### **1. Datenakquise**
- Laden Sie historische Daten:
  ```bash
  python Code/data_acquisition.py

- Vorverarbeiten der Rohdaten:
  ```bash
  python Code/data_preprocessing.py

- Trainieren Sie das Modell:
  ```bash
  python Code/train_model.py
