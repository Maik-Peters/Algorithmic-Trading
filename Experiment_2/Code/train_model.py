import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import json
from multi_input_lstm import MultiInputLSTM
from utils import ensure_directory_exists, log_message
from attention import visualize_attention

class MultiInputDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.main_features = data[['aapl_open', 'aapl_high', 'aapl_low', 'aapl_close', 'aapl_volume']].values
        self.aux_features = data[['msft_close', 'spy_close']].values
        self.target = data['aapl_close'].values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        main_input = self.main_features[idx].reshape(1, -1)  # seq_len = 1
        aux_input = self.aux_features[idx].reshape(1, -1)    # seq_len = 1
        target = self.target[idx]
        return (
            torch.tensor(main_input, dtype=torch.float32),
            torch.tensor(aux_input, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    training_loss = []
    test_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for main_input, aux_input, targets in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(main_input, aux_input)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        training_loss.append(avg_train_loss)

        # Berechne den Testverlust
        test_loss = evaluate_model(model, test_loader, criterion, results_dir=None, log_attention=False)
        test_loss_history.append(test_loss)

        log_message(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return training_loss, test_loss_history

def evaluate_model(model, dataloader, criterion, results_dir=None, log_attention=True):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    all_main_weights = []
    all_aux_weights = []

    with torch.no_grad():
        for main_input, aux_input, targets in dataloader:
            outputs, main_weights, aux_weights = model(main_input, aux_input)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())
            all_main_weights.append(main_weights)
            all_aux_weights.append(aux_weights)

    avg_loss = total_loss / len(dataloader)

    if results_dir and log_attention:
        ensure_directory_exists(results_dir)
        predictions_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions})
        predictions_df.to_csv(f"{results_dir}/predictions.csv", index=False)
        log_message(f"Vorhersagen in {results_dir}/predictions.csv gespeichert.")

        # Visualisierung der Attention-Gewichte
        all_main_weights = torch.cat(all_main_weights, dim=0)  # (batch_size, seq_len)
        all_aux_weights = torch.cat(all_aux_weights, dim=0)  # (batch_size, seq_len)
        visualize_attention(all_main_weights, f"{results_dir}/main_attention_weights.png")
        visualize_attention(all_aux_weights, f"{results_dir}/aux_attention_weights.png")

    return avg_loss

def plot_training_loss(training_loss, save_path):
    """
    Plottet den Trainingsverlust und speichert ihn.
    """
    ensure_directory_exists(save_path.rsplit('/', 1)[0])
    plt.plot(training_loss)
    plt.title("Trainingsverlust")
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()
    log_message(f"Trainingsverlust in {save_path} gespeichert.")

def plot_combined_loss(training_loss, test_loss, save_path):
    """
    Plottet Trainings- und Testverlust in einem Diagramm und speichert es.
    """
    ensure_directory_exists(save_path.rsplit('/', 1)[0])
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, label="Trainingsverlust")
    plt.plot(epochs, test_loss, label="Testverlust", linestyle='--')
    plt.title("Trainings- und Testverlust")
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    log_message(f"Trainings- und Testverlust in {save_path} gespeichert.")


if __name__ == "__main__":
    # Daten laden
    dataset = MultiInputDataset("Data/preprocessed_data.csv")
    split_index = int(len(dataset) * 0.8)
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_index))
    test_dataset = torch.utils.data.Subset(dataset, range(split_index, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultiInputLSTM(input_dim_main=5, input_dim_aux=2, hidden_dim=64, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_message("Training beginnt...")
    training_loss, test_loss_history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)

    # Kombinierte Visualisierung der Verluste
    plot_combined_loss(training_loss, test_loss_history, "Results/training_and_test_loss.png")

    # Speichern der Performance-Metriken
    performance_metrics = {
        "training_loss_last_epoch": training_loss[-1],
        "test_loss_last_epoch": test_loss_history[-1]
    }
    with open("Results/performance_metrics.json", "w") as f:
        json.dump(performance_metrics, f)

    log_message(f"Performance-Metriken in Results/performance_metrics.json gespeichert.")
    log_message("Experiment abgeschlossen.")
