# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dl-with-pytorch
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

from lib.path import data_path

path = f'{data_path()}/netflix/train.csv'
df = pd.read_csv(path)  # 데이터 파일을 읽어옴
df.head()

# %%
df.info()

# %%
import matplotlib.pyplot as plt

df_used = df.iloc[:, 1:4]
df_used['Close'] = df['Close']

hist = df_used.hist()
plt.show()

# %%
import numpy as np
from torch.utils.data.dataset import Dataset


class Netflix(Dataset):
    def __init__(self):
        self.csv = pd.read_csv(path)

        self.data = self.csv.iloc[:, 1:4].values
        self.data = self.data / np.max(self.data)

        self.label = df['Close'].values
        self.label = self.label / np.max(self.label)

    def __len__(self):
        return len(self.data) - 30

    def __getitem__(self, i):
        data = self.data[i : i + 30]
        label = self.label[i + 30]

        return data, label


# %%
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(input_size=3, hidden_size=8, num_layers=5, batch_first=True)

        self.fc1 = nn.Linear(in_features=240, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)

        x = torch.reshape(x, (x.shape[0], -1))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        x = torch.flatten(x)

        return x


# %%
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from lib.device import available_device
from lib.path import model_path

device = available_device()

model = RNN().to(device)

dataset = Netflix()

loader = DataLoader(dataset, batch_size=32)

criterion = nn.MSELoss()

optim = Adam(params=model.parameters(), lr=1e-4)

for ep in range(200):
    iterator = tqdm.tqdm(loader)
    for data, label in iterator:
        optim.zero_grad()

        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        pred = model(data.type(torch.FloatTensor).to(device), h0)

        loss = criterion(pred, label.type(torch.FloatTensor).to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f'epoch : {ep} loss : {loss.item()}')

torch.save(model.state_dict(), model_path('rnn.pth'))

# %%
import matplotlib.pyplot as plt

loader = DataLoader(dataset, batch_size=1)

preds = []
total_loss = 0

with torch.no_grad():
    model.load_state_dict(torch.load(model_path('rnn.pth'), map_location=device))

    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item())

        loss = criterion(pred, label.type(torch.FloatTensor).to(device))

        total_loss += loss / len(loader)

total_loss.item()

# %%
plt.plot(preds, label='prediction')
plt.plot(dataset.label[30:], label='actual')
plt.legend()
plt.show()
