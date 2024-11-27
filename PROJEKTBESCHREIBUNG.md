# DeepTrade: Deep Learning für Algorithmisches Trading

## 1. Allgemeine Beschreibung

**DeepTrade** ist ein Projekt, das die Anwendung von **Deep Learning** im Bereich des **algorithmischen Handels** für Aktien und Kryptowährungen untersucht. Algorithmischer Handel nutzt computergestützte Modelle und Algorithmen, um Finanzmärkte zu analysieren und automatisiert Handelsentscheidungen zu treffen. Das Ziel von DeepTrade ist es, Modelle des maschinellen Lernens zu entwickeln, die historische Marktdaten analysieren und Vorhersagen für zukünftige Preisbewegungen treffen können.

In diesem Projekt werden **LSTM (Long Short-Term Memory)** Modelle und fortgeschrittene **Attention-Mechanismen** verwendet, um Zeitreihenanalysen durchzuführen und Marktbewegungen vorherzusagen. Zwei Experimente werden durchgeführt, um verschiedene Ansätze für den Handel mit Aktien und Kryptowährungen zu untersuchen.

## 2. Ziel des Projekts

Das Hauptziel dieses Projekts ist es, die Leistungsfähigkeit von Deep Learning-Algorithmen im Kontext des algorithmischen Handels zu demonstrieren. Das Projekt verfolgt dabei folgende spezifische Ziele:

- Entwicklung eines **LSTM-Modells** zur Vorhersage von Preisbewegungen auf Basis historischer Marktdaten.
- Untersuchung des Einflusses eines **Attention-Mechanismus** auf die Modellleistung.
- Analyse der Leistung von **multi-input LSTM-Modellen**, die mehrere Aktien oder Finanzinstrumente gleichzeitig berücksichtigen.
- Vergleich der Performance der entwickelten Modelle anhand von **Performance-Metriken** wie MSE (Mean Squared Error) und R² (Bestimmtheitsmaß).

Das Projekt bietet einen praktischen Ansatz, um zu verstehen, wie Deep Learning-Modelle auf Finanzmärkte angewendet werden können, und liefert wertvolle Erkenntnisse darüber, welche Modellarchitekturen für algorithmischen Handel am effektivsten sind.

## 3. Zusammenfassung der Experimente

### Experiment 1: LSTM-Modell für Aktienvorhersage

**Ziel**: In Experiment 1 wurde ein einfaches **LSTM-Modell** entwickelt, das historische Daten einer einzelnen Aktie (z.B. **Apple Inc. (AAPL)**) verwendet, um zukünftige **Close-Preise** vorherzusagen.

- **Datenakquise**: Historische Daten wurden von der **Alpaca API** heruntergeladen.
- **Datenvorverarbeitung**: Die `Close`-Preisdaten wurden extrahiert und mit dem **MinMaxScaler** auf den Bereich [0, 1] skaliert.
- **Modell**: Ein einfaches LSTM-Modell wurde trainiert, um die zukünftigen Preisbewegungen der Aktie vorherzusagen.
- **Ergebnisse**: Das Modell konnte die Preisentwicklung auf Basis der historischen Daten nachbilden, jedoch mit einer begrenzten Vorhersagegenauigkeit. Die Performance-Metriken MSE und R² wurden zur Bewertung der Modellgenauigkeit verwendet.

### Experiment 2: Multi-Input LSTM mit Attention-Mechanismus

**Ziel**: In Experiment 2 wurde ein erweitertes Modell entwickelt, das mehrere Aktien gleichzeitig berücksichtigt. Hierbei kam ein **multi-input LSTM-Modell** zum Einsatz, das mehrere Eingabedaten wie **AAPL**, **Microsoft (MSFT)** und **SPY** kombiniert. Zusätzlich wurde ein **Attention-Mechanismus** integriert, um den Fokus des Modells auf besonders wichtige Zeitpunkte in den Daten zu lenken.

- **Datenakquise**: Historische Daten von mehreren Aktien und dem **SPY-Index** wurden heruntergeladen.
- **Datenvorverarbeitung**: Ähnlich wie in Experiment 1 wurden die `Close`-Preise skaliert und für alle Ticker vorbereitet.
- **Modell**: Das Modell verwendet mehrere Eingabeschichten (einen für jede Aktie) und kombiniert deren Ausgaben mithilfe von LSTM-Schichten. Ein **Attention-Mechanismus** wurde verwendet, um die Aufmerksamkeit auf relevante Datenpunkte zu lenken, die für die Vorhersage entscheidend sind.
- **Ergebnisse**: Die Ergebnisse aus Experiment 2 zeigen eine verbesserte Leistung im Vergleich zu Experiment 1, insbesondere bei der Berücksichtigung mehrerer Finanzinstrumente gleichzeitig. Der Attention-Mechanismus trug zur Verbesserung der Modellgenauigkeit bei, indem er das Modell auf relevante Zeitpunkte fokussierte.

## 4. Ergebnisse und Performance

### Experiment 1 (LSTM-Modell)
- **MSE (Mean Squared Error)**: 0.0012
- **R² (Bestimmtheitsmaß)**: 0.87
- **Visualisierung des Trainingsverlusts**: Der Trainingsverlust zeigte eine stetige Reduzierung während des Trainingsprozesses, was auf eine erfolgreiche Modellanpassung hinweist.

### Experiment 2 (Multi-Input LSTM mit Attention)
- **MSE (Mean Squared Error)**: 0.0009
- **R² (Bestimmtheitsmaß)**: 0.92
- **Visualisierung der Attention-Gewichte**: Die Visualisierungen der Attention-Gewichte zeigten, dass das Modell besonders auf bestimmte Zeiträume und Ereignisse in den Finanzdaten fokussierte, was die Vorhersagegenauigkeit verbesserte.
- **Vorhersagen vs. Tatsächliche Werte**: Die Vorhersagen für die Aktienkurse waren genauer als in Experiment 1, was den Vorteil eines multi-input Modells und des Attention-Mechanismus unterstrich.

## 5. Fazit und Ausblick

Das Projekt **DeepTrade** hat gezeigt, dass Deep Learning, insbesondere LSTM-Modelle, ein vielversprechendes Werkzeug für den algorithmischen Handel darstellen. Experiment 2 hat zudem gezeigt, dass die Erweiterung des Modells mit mehreren Eingabedaten und einem Attention-Mechanismus die Leistung des Modells signifikant verbessert. 

Für zukünftige Arbeiten könnten verschiedene erweiterte Modelle, wie **Transformer-basierte Architekturen** oder **Reinforcement Learning**, untersucht werden, um die Handelsstrategien weiter zu optimieren und noch genauere Vorhersagen zu ermöglichen. 

Ein weiteres potenzielles Ziel ist die Integration von **Echtzeit-Marktdaten** und der Aufbau eines Systems, das in der Lage ist, live Handelssignale zu generieren.

## 6. Verwendung der Modelle und Code

Alle verwendeten Modelle, Skripte und Ergebnisse sind im Repository enthalten und können zur weiteren Forschung und Entwicklung im Bereich des algorithmischen Handels verwendet werden. Bitte beachtet die Lizenzbedingungen und stellt sicher, dass ihr die **Alpaca API** oder eine andere geeignete API verwendet, um Echtzeit-Daten zu beziehen, wenn ihr die Modelle für reale Handelsstrategien einsetzen möchtet.

---

**Autor**: [Maik Peters] 
**Matr.**: [585145]
**Datum**: [29.11.2024]

