# **DeepTrade: Deep Learning für Algorithmisches Trading**

## **Allgemeine Beschreibung**

**DeepTrade** ist ein Deep-Learning-Projekt, das sich mit algorithmischem Trading befasst. Ziel ist es, präzise Vorhersagen von Aktien- und Kryptowährungskursen basierend auf historischen Daten zu treffen. Es kombiniert bewährte Zeitreihenmodelle wie LSTMs (Long Short-Term Memory) mit fortschrittlichen Mechanismen wie Attention-Layern und Multi-Input-Architekturen. 

Das Projekt ist in zwei Experimente gegliedert:
1. **Experiment 1** untersucht die Grundlagen der Kursvorhersage mit einem einfachen LSTM-Modell.
2. **Experiment 2** erweitert diese Ansätze mit einem Attention-Mechanismus und einem Multi-Input-Framework, um Daten aus mehreren Märkten zu kombinieren.

---

## **Ziel des Projekts**

Das Hauptziel von **DeepTrade** ist es, robuste, skalierbare und interpretierbare Modelle für das algorithmische Trading zu entwickeln. Es fokussiert sich dabei auf:
- **Vorhersagegenauigkeit**: Nutzung von Deep-Learning-Ansätzen, um Marktbewegungen präzise vorherzusagen.
- **Modell interpretierbarkeit**: Anwendung von Attention-Mechanismen zur Hervorhebung relevanter Datenpunkte.
- **Skalierbarkeit**: Entwicklung eines Frameworks, das auf verschiedene Märkte und Datenquellen ausgeweitet werden kann.

---

## **Zusammenfassung und Ergebnisse**

### **Experiment 1: LSTM für Kursvorhersagen**

#### **Beschreibung**
In Experiment 1 wurde ein LSTM-Modell entwickelt, um die `Close`-Preise von Aktien basierend auf historischen Daten vorherzusagen. Als Datenquelle diente Alpaca API.

#### **Vorgehensweise**
1. **Datenakquise**: Historische `Close`-Preise von Apple (AAPL) wurden heruntergeladen und als `raw_data.csv` gespeichert.
2. **Datenvorverarbeitung**: Die Daten wurden skaliert, um sie als Eingabe für das Modell nutzbar zu machen. Die vorverarbeiteten Daten wurden in `preprocessed_data.csv` gespeichert.
3. **Modelltraining**: Ein LSTM-Modell mit einem 60-Tage-Zeitfenster wurde trainiert, um den nächsten Tagespreis vorherzusagen.

#### **Ergebnisse**
- **Training Loss**: Der Trainingsverlust zeigte eine schnelle Konvergenz.
- **Vorhersagegenauigkeit**: Das Modell konnte die Kursverläufe moderat genau vorhersagen, wobei eine leichte Überanpassung auf die Trainingsdaten beobachtet wurde.
- **Visualisierung**: Die Ergebnisse wurden grafisch dargestellt, einschließlich des Vergleichs zwischen tatsächlichen und vorhergesagten Werten.

---

### **Experiment 2: Multi-Input LSTM mit Attention**

#### **Beschreibung**
Experiment 2 erweiterte das grundlegende Modell durch:
1. Einen **Attention-Mechanismus**, der relevante Zeitpunkte während der Vorhersagen hervorhebt.
2. Ein **Multi-Input LSTM**, das Daten aus mehreren Märkten (AAPL, MSFT, SPY) simultan verarbeitet.

#### **Vorgehensweise**
1. **Datenakquise**: Historische `Close`-Preise für Apple (AAPL), Microsoft (MSFT) und den S&P 500 Index (SPY) wurden heruntergeladen.
2. **Datenvorverarbeitung**: Die Daten jeder Aktie wurden individuell skaliert und in einer Datei (`preprocessed_data.csv`) gespeichert.
3. **Attention-Mechanismus**: Ein Attention-Layer wurde in das Modell integriert, um die Wichtigkeit einzelner Zeitpunkte zu berücksichtigen.
4. **Multi-Input LSTM**: Ein LSTM-Modell, das alle drei Datenquellen kombiniert, wurde entwickelt und trainiert.

#### **Ergebnisse**
- **Attention Visualisierung**: Der Mechanismus hob Schlüsselzeiträume hervor, die für die Modellentscheidungen entscheidend waren.
- **Vorhersagegenauigkeit**: Das Multi-Input-Modell zeigte eine signifikante Verbesserung der Vorhersagegenauigkeit im Vergleich zu Experiment 1.
- **Multi-Markt-Korrelation**: Die Nutzung mehrerer Datenquellen (AAPL, MSFT, SPY) erlaubte es dem Modell, die Marktinteraktionen besser zu verstehen und zu nutzen.

---

## **Schlussfolgerungen und Ausblick**

1. **LSTM-Modelle sind leistungsstark**, wenn es darum geht, zeitliche Muster in Marktdaten zu erkennen. 
2. **Attention-Mechanismen erhöhen die Interpretierbarkeit** und erlauben es, Schlüsselmuster in den Daten zu identifizieren.
3. **Multi-Input-Modelle bieten einen Vorteil**, wenn Daten aus verschiedenen Märkten oder Vermögenswerten integriert werden.

### **Zukünftige Arbeiten**
- **Echtzeit-Trading**: Implementierung von Vorhersagen und automatischen Trades in Echtzeit.
- **Kryptowährungen**: Anwendung der Modelle auf volatile Märkte wie Bitcoin und Ethereum.
- **Hyperparameteroptimierung**: Feinabstimmung der Modelle, um die Performance weiter zu steigern.
