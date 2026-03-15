import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from auto_coloring.dataset import AutoColoring
from auto_coloring.models import AutoColoringModel
from lib.utils.device import available_device
from lib.utils.path import model_path


def run():
    device = available_device()

    model = AutoColoringModel().to(device)

    dataset = AutoColoring()
    print('dataset', len(dataset))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optim = Adam(params=model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for ep in range(200):
        iterator = tqdm.tqdm(loader)

        for L, AB in iterator:
            L = torch.unsqueeze(L, dim=1).to(device)
            optim.zero_grad()

            pred = model(L)

            loss = criterion(pred, AB.to(device))
            loss.backward()
            optim.step()

            iterator.set_description(f'epoch{ep} loss:{loss.item()}')

    torch.save(model.state_dict(), model_path('AutoColor.pth'))


if __name__ == '__main__':
    run()
