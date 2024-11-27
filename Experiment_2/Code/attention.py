import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import log_message


class Attention(nn.Module):
    """
    Implementierung eines Attention-Mechanismus.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Lineares Layer zur Berechnung der Attention-Gewichte
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Berechnet die Attention-Gewichte und den Kontextvektor.

        Args:
            lstm_output: Tensor der Form (batch_size, seq_len, hidden_dim)

        Returns:
            context: Kontextvektor der Form (batch_size, hidden_dim)
            weights: Attention-Gewichte der Form (batch_size, seq_len, 1)
        """
        # Berechne die Gewichte (batch_size, seq_len, 1)
        weights = torch.softmax(self.attention(lstm_output), dim=1)

        # Kontextvektor: gewichtete Summe der LSTM-Ausgaben (batch_size, hidden_dim)
        context = torch.sum(weights * lstm_output, dim=1)

        return context, weights


def visualize_attention(weights, save_path):
    """
    Visualisiert die Attention-Gewichte als Heatmap.

    Args:
        weights: Tensor der Form (batch_size, seq_len) oder (batch_size, seq_len, 1)
        save_path: Speicherpfad für die Heatmap

    Returns:
        None
    """
    # Überprüfen der Form und vorbereiten der Gewichte
    if len(weights.shape) == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)  # Entferne die letzte Dimension, wenn sie 1 ist
    elif len(weights.shape) != 2:
        raise ValueError(f"Unerwartete Form von weights: {weights.shape}")

    weights = weights.detach().numpy()  # Konvertiere in NumPy-Array

    # Normalisierung der Attention-Werte
    range_weights = weights.max() - weights.min()
    if range_weights > 0:  # Nur normalisieren, wenn der Bereich größer als 0 ist
        weights = (weights - weights.min()) / range_weights

    # Heatmap erstellen
    plt.figure(figsize=(10, 6))
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.title("Attention Weights")
    plt.xlabel("Sequence Step")
    plt.ylabel("Batch Sample")
    plt.savefig(save_path)
    plt.close()

    log_message(f"Attention-Gewichte in {save_path} visualisiert.")
