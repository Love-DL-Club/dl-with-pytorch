import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, h):
        x = self.embedding(x).view(1, 1, -1)
        output, hidden = self.gru(x, h)

        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=11):
        super().__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.attention = nn.Linear(hidden_size * 2, max_length)

        self.context = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h, encoder_outputs):
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)

        attn_weights = self.softmax(self.attention(torch.cat(x[0], h[0], -1)))

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqeeze(0))

        output = torch.cat((x[0], attn_applied[0]), 1)
        output = self.context(output).unsqeeze(0)
        output = self.relu(output)

        output, _ = self.gru(output, h)

        output = self.out(output[0])

        return output
