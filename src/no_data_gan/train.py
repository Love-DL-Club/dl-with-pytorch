import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torchvision.models.resnet import (
    resnet34,
)

from lib.utils.device import available_device
from lib.utils.path import model_path
from no_data_gan.dataset import train_loader


def main(epoch=30):
    loader = train_loader()

    device = available_device()

    teacher = resnet34(weights=None, num_classes=10).to(device)

    lr = 1e-5

    optim = Adam(teacher.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        iterator = tqdm.tqdm(loader)

        for data, label in iterator:
            optim.zero_grad()

            preds = teacher(data.to(device))

            loss = criterion(preds, label.to(device))
            loss.backward()
            optim.step()

            iterator.set_description(f'epoch:{ep + 1} loss:{loss.item()}')

    torch.save(teacher.state_dict(), model_path('teacher.pth'))

    return teacher
