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
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

from lib.cifar10_data import load_data

train_data = load_data(root='../data', train=True, transform=ToTensor())
test_data = load_data(root='../data', train=False, transform=ToTensor())

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_data.data[i])

plt.show()

# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

from lib.cifar10_data import load_data
from lib.show_image import show_image

transforms = Compose(
    [
        T.ToPILImage(),
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
    ]
)

train_data = load_data(
    root='../data',
    train=True,
    transform=transforms,
)

test_data = load_data(
    root='../data',
    train=False,
    transform=transforms,
)

show_image(transforms, train_data)


# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip

from lib.cifar10_data import load_data
from lib.show_image import show_image

transforms = Compose(
    [
        T.ToPILImage(),
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4464), std=(0.247, 0.243, 0.261)),
        T.ToPILImage(),
    ]
)

train_data = load_data(
    root='../data',
    train=True,
    transform=transforms,
)

test_data = load_data(
    root='../data',
    train=False,
    transform=transforms,
)

show_image(transforms, train_data)


# %%
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


# %%
import torch


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
        self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
        self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)

        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


# %%
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from lib.cifar10_data import load_data

transforms = Compose(
    [
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ]
)

train_data = load_data('../data', train=True, transforms=transforms)
test_data = load_data('../data', train=False, transforms=transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN(num_classes=10)

model.to(device)

lr = 1e-3

optim = Adam(model.parameters(), lr=lr)

for ep in range(100):
    for data, label in train_loader:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    if ep == 0 or ep % 10 == 9:
        print(f'epoch {ep + 1} loss: {loss.item()}')

torch.save(model.state_dict(), '../data/models/CiFAR.pth')

# %%
model.load_state_dict(torch.load('CIFAR.pth', map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

print(f'Accuracy: {num_corr / len(test_data)}')

# %%
import torch.nn as nn
import torchvision.models as models

from lib.device import available_device

device = available_device()
print(device)

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
fc = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 10),
)

model.classifier = fc
model.to(device)

# %%
import torch
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from lib.cifar10_data import load_data

transforms = Compose(
    [
        Resize(224),
        RandomCrop((224, 224), padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ]
)

train_data = load_data(root='../data', train=True, transform=transforms)
test_data = load_data(root='../data', train=False, transform=transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)


for ep in range(1):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f'epoch: {ep + 1} loss: {loss.item()}')

torch.save(model.state_dict(), '../data/models/CIFAR_pretrained.pth')

# %%
model.load_state_dict(
    torch.load('../data/models/CIFAR_pretrained.pth', map_location=device)
)

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

print(f'Accuracy: {num_corr / len(test_data)}')
