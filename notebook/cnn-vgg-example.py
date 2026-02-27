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

