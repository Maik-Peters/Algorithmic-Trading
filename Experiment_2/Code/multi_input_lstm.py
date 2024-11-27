import torch
import torch.nn as nn
from attention import Attention  # Import der Attention-Klasse

class MultiInputLSTM(nn.Module):
    def __init__(self, input_dim_main, input_dim_aux, hidden_dim, output_dim):
        super(MultiInputLSTM, self).__init__()
        self.lstm_main = nn.LSTM(input_dim_main, hidden_dim, batch_first=True)
        self.lstm_aux = nn.LSTM(input_dim_aux, hidden_dim, batch_first=True)
        self.attention_main = Attention(hidden_dim)  # Unterschiedlicher Attention-Layer
        self.attention_aux = Attention(hidden_dim)  # Unterschiedlicher Attention-Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, main_input, aux_input):
        # Hauptfaktor: LSTM und Attention
        main_out, _ = self.lstm_main(main_input)
        main_context, main_weights = self.attention_main(main_out)

        # Nebenfaktor: LSTM und Attention
        aux_out, _ = self.lstm_aux(aux_input)
        aux_context, aux_weights = self.attention_aux(aux_out)

        # Kombination der Kontextvektoren
        combined = torch.cat((main_context, aux_context), dim=1)

        # Ausgabe
        output = self.fc(combined)
        return output, main_weights, aux_weights
