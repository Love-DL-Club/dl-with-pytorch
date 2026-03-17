import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from crnn.dataset import Captcha
from crnn.models import CRNN
from lib.utils.device import available_device
from lib.utils.path import data_path, model_path


def main():
    device = available_device()

    dataset = Captcha(path=str(data_path() / 'CH12'))
    loader = DataLoader(dataset, batch_size=8)

    model = CRNN(output_size=len(dataset.BOW)).to(device)

    optim = Adam(model.parameters(), lr=1e-4)

    criterion = nn.CTCLoss(blank=0)

    for ep in range(200):
        iterator = tqdm.tqdm(loader)

        for data, label in iterator:
            optim.zero_grad()
            preds = model(data.to(device))

            current_batch_size = data.size(0)

            preds_size = torch.IntTensor([preds.size(0)] * current_batch_size).to(
                device
            )
            target_len = torch.IntTensor([len(txt) for txt in label]).to(device)

            loss = criterion(preds, label.to(device), preds_size, target_len)

            loss.backward()
            optim.step()

            iterator.set_description(f'epoch{ep + 1} loss: {loss.item()}')

    torch.save(model.state_dict(), model_path('CRNN.pth'))
