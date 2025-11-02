import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        non_padding = torch.abs(x).sum(dim=2) != 0
        lengths = non_padding.sum(dim=1)
        lengths = lengths.clamp(min=1)
        last_indices = lengths - 1
        batch_indices = torch.arange(x.size(0)).to(x.device)
        last_hidden_states = out[batch_indices, last_indices, :]
        out = self.fc(last_hidden_states)
        return out
