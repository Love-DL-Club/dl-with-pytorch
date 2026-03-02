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
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor

# fmt: off
# SIFAR-10 데이터셋 불러오기
training_data = CIFAR10(
    root="../data",
    train = True,
    download = True,
    transform=ToTensor()
)

test_data = CIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor()
)
# fmt: on

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(training_data.data[i])
plt.show()

# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

transforms = Compose(
    [T.ToPILImage(), RandomCrop((32, 32), padding=4), RandomHorizontalFlip(p=0.5)]
)
training_data = CIFAR10(root='../data', train=True, download=True, transform=transforms)

test_data = CIFAR10(root='../data', train=False, download=True, transform=transforms)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(transforms(training_data.data[i]))

plt.show()

# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip

transforms = Compose(
    [
        T.ToPILImage(),
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        T.ToPILImage(),
    ]
)

training_data = CIFAR10(root='../data', train=True, download=True, transform=transforms)

test_data = CIFAR10(root='../data', train=False, download=True, transform=transforms)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(transforms(training_data.data[0]))
plt.show()

# %%
import torch

training_data = CIFAR10(root='../data', train=True, download=True, transform=ToTensor())

imgs = [item[0] for item in training_data]

imgs = torch.stack(imgs, dim=0).numpy()

mean_r = imgs[:, 0, :, :].mean()
mean_g = imgs[:, 1, :, :].mean()
mean_b = imgs[:, 2, :, :].mean()
print(mean_r, mean_g, mean_b)

std_r = imgs[:, 0, :, :].std()
std_g = imgs[:, 1, :, :].std()
std_b = imgs[:, 2, :, :].std()
print(std_r, std_g, std_b)
