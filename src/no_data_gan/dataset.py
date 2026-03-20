from torch.utils.data.dataloader import DataLoader

from lib.dataset.cifar10_data import load_data
from lib.utils.path import data_path
from no_data_gan.transforms import create


def train_loader():
    data = load_data(data_path(), transform=create())

    return (data, DataLoader(data, batch_size=32, shuffle=True))


def test_loader():
    data = load_data(data_path(), train=False, transform=create())

    return (data, DataLoader(data, batch_size=32, shuffle=False))
