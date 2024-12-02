import predictions
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
    def __init__(self, data):
        self.features = data[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']].values
        self.target = data['close'].values

        # Sicherstellen, dass alle Eingaben die gleiche Form haben
        self.seq_len = 1
        self.input_dim = self.features.shape[1]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Input- und Ziel-Daten
        input_data = self.features[idx]
        target_data = self.target[idx]

        # Sicherstellen, dass die Eingaben korrekt reshaped werden
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(self.seq_len, self.input_dim)

        # Validierungen
        assert input_data.shape == (1, 7), f"Unexpected input shape: {input_data.shape}"
        assert isinstance(target_data, (float, int)), f"Unexpected target type: {type(target_data)}"

        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)


# Training-Funktion
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    training_loss = []
    test_loss_history = []
    final_predictions, final_actuals = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Sicherstellen, dass die Eingabe 3D ist
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        training_loss.append(avg_train_loss)

        # Testverlust nach jeder Epoche berechnen
        predictions, actuals, test_loss = evaluate_model(model, test_loader, criterion)
        test_loss_history.append(test_loss)

        # Speichere die finalen Vorhersagen und Zielwerte nur für die letzte Epoche
        if epoch == num_epochs - 1:
            final_predictions = predictions
            final_actuals = actuals

        log_message(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return training_loss, test_loss_history, final_predictions, final_actuals


# Evaluation-Funktion
def evaluate_model(model, dataloader, criterion):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Eingabe überprüfen
            print(f"Inputs shape before: {inputs.shape}, Targets shape: {targets.shape}")
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)  # Sicherstellen, dass die Eingabe 3D ist
            print(f"Inputs shape after: {inputs.shape}")

            # Zielwerte überprüfen
            assert len(targets.shape) == 1, f"Unexpected target shape: {targets.shape}"

            # Vorhersagen und Verlust berechnen
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()

            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
    avg_loss = total_loss / len(dataloader)
    return predictions, actuals, avg_loss


# Ergebnisse speichern
def save_results(training_loss, test_loss_history, predictions, actuals, performance_metrics, results_dir):
    ensure_directory_exists(results_dir)

    # Trainings- und Testverlust plotten
    plt.plot(training_loss, label="Trainingsverlust")
    plt.plot(test_loss_history, label="Testverlust", linestyle='--')
    plt.title("Trainings- und Testverlust")
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{results_dir}/training_and_test_loss.png")
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
    full_data = pd.read_csv("Data/preprocessed_data.csv")
    split_index = int(len(full_data) * 0.8)

    # Training- und Test-Daten aufteilen
    train_data = full_data.iloc[:split_index]
    test_data = full_data.iloc[split_index:]

    # Train- und Test-Datasets erstellen
    train_dataset = StockDataset(train_data)
    test_dataset = StockDataset(test_data)

    # Dataloader erstellen
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modell initialisieren
    input_dim = 7
    hidden_dim = 64
    output_dim = 1
    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Verlustfunktion und Optimierer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    log_message("Training beginnt...")
    training_loss, test_loss_history, predictions, actuals = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=50
    )

    # Performance-Metriken
    performance_metrics = {
        "training_loss_last_epoch": training_loss[-1],
        "test_loss_last_epoch": test_loss_history[-1]
    }

    # Ergebnisse speichern
    save_results(training_loss, test_loss_history, predictions, actuals, performance_metrics, "Results")
    log_message("Experiment abgeschlossen.")

