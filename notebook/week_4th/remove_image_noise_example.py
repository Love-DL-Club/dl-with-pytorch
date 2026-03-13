# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: study-club (3.11.14)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch


def gaussian_noise(x, scale=0.8):
    gaussian_data_x = x + np.random.normal(loc=0, scale=scale, size=x.shape)

    gaussian_data_x = np.clip(gaussian_data_x, 0, 1)

    gaussian_data_x = torch.tensor(gaussian_data_x)
    gaussian_data_x = gaussian_data_x.type(torch.FloatTensor)

    return gaussian_data_x



# %%
import matplotlib.pyplot as plt

from lib.dataset.mnist import load_data

train_data = load_data()
test_data = load_data(train=False)

img = train_data.data[0]
gaussian = gaussian_noise(img)

plt.subplot(1, 2, 1)
plt.title('original')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title('noisy')
plt.imshow(gaussian)
plt.show()

# %%
from torch.utils.data.dataset import Dataset

from lib.dataset.mnist import load_data


class Denoising(Dataset):
    def __init__(self):
        self.mnist = load_data()

        self.data = []

        for i in range(len(self.mnist)):
            noisy_input = gaussian_noise(self.mnist.data[i])
            input_tensor = torch.tensor(noisy_input)
            self.data.append(torch.unsqueeze(input_tensor, dim=0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]

        label = self.mnistdata[i] / 255

        return data, label


# %%
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x



# %%
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = BasicBlock(in_channels=1, out_channels=16, hidden_dim=16)
        self.conv2 = BasicBlock(in_channels=16, out_channels=8, hidden_dim=16)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        return x

