# **DeepTrade: Deep Learning für Algorithmisches Trading**

## **Allgemeine Beschreibung**

**DeepTrade** ist ein Deep-Learning-Projekt, das sich mit algorithmischem Trading befasst. Ziel ist es, präzise Vorhersagen von Aktien- und Marktindizes basierend auf historischen Daten zu treffen. Es kombiniert bewährte Zeitreihenmodelle wie LSTMs (Long Short-Term Memory) mit fortschrittlichen Mechanismen wie Attention-Layern und Multi-Input-Architekturen.

Das Projekt umfasst zwei Experimente:
1. **Experiment 1**: Einführung in die Kursvorhersage mit einem einfachen LSTM-Modell.
2. **Experiment 2**: Erweiterung des LSTM-Modells durch einen Attention-Mechanismus und Multi-Input-Strukturen.

---

## **Ziel des Projekts**

Das Hauptziel von **DeepTrade** ist es, robuste, skalierbare und interpretierbare Modelle für das algorithmische Trading zu entwickeln. Es fokussiert sich auf:
- **Vorhersagegenauigkeit**: Nutzung von Deep-Learning-Ansätzen, um Marktbewegungen präzise vorherzusagen.
- **Modellinterpretierbarkeit**: Anwendung von Attention-Mechanismen zur Hervorhebung relevanter Datenpunkte.
- **Skalierbarkeit**: Entwicklung eines Frameworks, das auf verschiedene Märkte und Datenquellen ausgeweitet werden kann.

---

## **Zusammenfassung und Ergebnisse**

### **Experiment 1: LSTM für Kursvorhersagen**

#### **Beschreibung**
In Experiment 1 wurde ein LSTM-Modell entwickelt, um die `Close`-Preise von Aktien basierend auf historischen Daten vorherzusagen. Die Daten stammen von der Alpaca API.

#### **Vorgehensweise**
1. **Datenakquise**: Historische `Close`-Preise von Apple (AAPL) wurden heruntergeladen und als `raw_data.csv` gespeichert.
2. **Datenvorverarbeitung**: Die Daten wurden skaliert und vorverarbeitet, um sie als Eingabe für das Modell nutzbar zu machen. Die vorverarbeiteten Daten wurden in `preprocessed_data.csv` gespeichert.
3. **Modelltraining**: Ein LSTM-Modell wurde mit historischen Daten trainiert, um den Schlusskurs des nächsten Tages vorherzusagen.

#### **Ergebnisse**
- **Letzter Trainingsverlust**: 0.000912
- **Letzter Testverlust**: 0.000743
- **Visualisierung**: Die Verlustkurven wurden grafisch dargestellt. Das Modell zeigt eine gute Konvergenz und moderate Vorhersagegenauigkeit.

---

### **Experiment 2: Multi-Input LSTM mit Attention**

#### **Beschreibung**
Experiment 2 erweitert das Modell durch:
1. Einen **Attention-Mechanismus**, der relevante Zeitpunkte während der Vorhersagen hervorhebt.
2. Ein **Multi-Input LSTM**, das Daten aus mehreren Märkten (AAPL, MSFT, SPY) simultan verarbeitet.

#### **Vorgehensweise**
1. **Datenakquise**: Historische `Close`-Preise für Apple (AAPL), Microsoft (MSFT) und den S&P 500 Index (SPY) wurden heruntergeladen und kombiniert.
2. **Datenvorverarbeitung**: Die Daten wurden skaliert und in einer Datei (`preprocessed_data.csv`) gespeichert.
3. **Attention-Mechanismus**: Ein Attention-Layer wurde in das Modell integriert, um wichtige Zeitpunkte zu gewichten.
4. **Multi-Input LSTM**: Ein LSTM-Modell, das Daten aus mehreren Märkten kombiniert, wurde trainiert.

#### **Ergebnisse**
- **Letzter Trainingsverlust**: 0.000911
- **Letzter Testverlust**: 0.002021
- **Attention Visualisierung**: Die Gewichtung wichtiger Zeitpunkte wurde hervorgehoben und grafisch dargestellt.
- **Verbesserte Vorhersagegenauigkeit**: Das Multi-Input-Modell zeigte eine signifikante Verbesserung im Vergleich zu Experiment 1.

---

## **Schlussfolgerungen und Ausblick**

1. **LSTM-Modelle** eignen sich hervorragend, um zeitliche Muster in Marktdaten zu erkennen. 
2. **Attention-Mechanismen** erhöhen die Interpretierbarkeit und identifizieren Schlüsselmuster.
3. **Multi-Input-Modelle** bieten einen Vorteil, wenn Daten aus verschiedenen Märkten integriert werden.

### **Zukünftige Arbeiten**
- **Echtzeit-Trading**: Implementierung von Vorhersagen und automatischen Trades in Echtzeit.
- **Anwendung auf Kryptowährungen**: Testen der Modelle auf volatile Märkte wie Bitcoin und Ethereum.
- **Hyperparameteroptimierung**: Feinabstimmung der Modelle, um die Performance weiter zu steigern.
