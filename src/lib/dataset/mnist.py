from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from lib.utils.path import data_path


def load_data(train=True):
    return MNIST(data_path(), train=train, download=True, transform=ToTensor())
