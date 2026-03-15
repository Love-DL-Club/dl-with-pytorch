import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from lib.utils.device import available_device
from lib.utils.path import model_path
from lstm.dataset import TextGeneration
from lstm.models import LSTM


def run(epoch=200):
    device = available_device()

    dataset = TextGeneration()
    loader = DataLoader(dataset, batch_size=64)

    model = LSTM(num_embeddings=len(dataset.BOW)).to(device)

    optim = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        iterator = tqdm.tqdm(loader)

        for data, label in iterator:
            optim.zero_grad()

            pred = model(torch.tensor(data, dtype=torch.long).to(device))

            loss = criterion(pred, torch.tensor(label, dtype=torch.long).to(device))

            loss.backward()
            optim.step()

            iterator.set_description(f'epoch{ep} loss:{loss.item()}')

    torch.save(model.state_dict(), model_path('lstm.pth'))
