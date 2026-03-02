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
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

train_data = MNIST(root='../data', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='../data', train=False, download=True, transform=ToTensor())

print(len(train_data))
print(len(test_data))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_data.data[i])

plt.show()

# %%
from torch.utils.data.dataloader import DataLoader

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# %%
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from lib.device import available_device
from lib.path import model_path

device = available_device()

model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)
)
model.to(device)

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for ep in range(20):
    for data, label in train_loader:
        optim.zero_grad()

        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    print(f'epoch {ep + 1} loss : {loss.item()}')

torch.save(model.state_dict(), model_path('MNIST.pth'))

# %%
model.load_state_dict(torch.load(model_path('MNIST.pth'), map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)

        output = model(data.to(device))
        # 최근에 data를 직접 건드리는것 보다 argmax를 사용하는걸 선호
        preds = output.argmax(dim=1)

        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

print(f'Accuray: {num_corr / len(test_data)}')
