import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import json
from utils import ensure_directory_exists, log_message


# LSTM-Modell
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Überprüfen Sie, dass die Eingabe 3D ist (batch_size, seq_len, input_dim)
        if len(x.shape) != 3:
            raise ValueError(f"Erwartet eine 3D-Eingabe, erhalten: {x.shape}")

        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        output = self.fc(lstm_out[:, -1, :])  # Letzte Ausgabe der Sequenz verwenden
        return output


# Dataset-Klasse
class StockDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.features = data[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']].values
        self.target = data['close'].values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Reshape die Eingabe in (seq_len, input_dim), wobei seq_len = 1
        input_data = self.features[idx].reshape(1, -1)  # seq_len = 1
        target_data = self.target[idx]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)


# Training-Funktion
def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    training_loss = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            # Eingabe überprüfen und sicherstellen, dass sie 3D ist
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)  # Form: (batch_size, seq_len, input_dim)

            outputs = model(inputs)  # Vorhersagen
            loss = criterion(outputs.squeeze(), targets)  # Verlust berechnen
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        training_loss.append(avg_loss)
        log_message(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return training_loss


# Evaluation-Funktion
def evaluate_model(model, dataloader, criterion):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Eingabe überprüfen
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
    avg_loss = total_loss / len(dataloader)
    return predictions, actuals, avg_loss


# Ergebnisse speichern
def save_results(training_loss, predictions, actuals, performance_metrics, results_dir):
    ensure_directory_exists(results_dir)

    # Trainingsverlust plotten
    plt.plot(training_loss)
    plt.title("Trainingsverlust")
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.savefig(f"{results_dir}/training_loss.png")
    plt.close()

    # Vorhersagen speichern
    predictions_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions})
    predictions_df.to_csv(f"{results_dir}/predictions.csv", index=False)

    # Performance-Metriken speichern
    with open(f"{results_dir}/performance_metrics.json", "w") as f:
        json.dump(performance_metrics, f)

    log_message("Ergebnisse gespeichert.")


# Hauptfunktion
if __name__ == "__main__":
    # Daten laden
    train_dataset = StockDataset("Data/preprocessed_data.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Modell initialisieren
    input_dim = 7  # Anzahl der Features
    hidden_dim = 64
    output_dim = 1
    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Verlustfunktion und Optimierer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    log_message("Training beginnt...")
    training_loss = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

    # Evaluation
    log_message("Evaluation beginnt...")
    predictions, actuals, test_loss = evaluate_model(model, train_loader, criterion)

    # Performance-Metriken berechnen
    performance_metrics = {
        "training_loss_last_epoch": training_loss[-1],
        "test_loss": test_loss
    }

    # Ergebnisse speichern
    save_results(training_loss, predictions, actuals, performance_metrics, "Results")
    log_message("Experiment abgeschlossen.")
