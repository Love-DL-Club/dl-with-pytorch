import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=16)

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=64,
            num_layers=5,
            batch_first=True,
        )

        self.fc1 = nn.Linear(128, num_embeddings)
        self.fc2 = nn.Linear(num_embeddings, num_embeddings)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embed(x)

        x, _ = self.lstm(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
