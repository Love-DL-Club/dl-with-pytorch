import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.models.resnet import (
    resnet34,
)

from lib.dataset.cifar10_data import load_data
from lib.utils.device import available_device
from lib.utils.path import data_path, model_path
from no_data_gan.transforms import create


def main(epoch=30):
    transforms = create()

    train_data = load_data(data_path(), transform=transforms)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    device = available_device()

    teacher = resnet34(weights=None, num_classes=10).to(device)

    lr = 1e-5

    optim = Adam(teacher.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        iterator = tqdm.tqdm(train_loader)

        for data, label in iterator:
            optim.zero_grad()

            preds = teacher(data.to(device))

            loss = criterion(preds, label.to(device))
            loss.backward()
            optim.step()

            iterator.set_description(f'epoch:{ep + 1} loss:{loss.item()}')

    torch.save(teacher.state_dict(), model_path('teacher.pth'))

    return teacher
