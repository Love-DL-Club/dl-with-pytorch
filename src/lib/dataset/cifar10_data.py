from torchvision.datasets.cifar import CIFAR10


def load_data(root='./', train=True, transform=None):
    return CIFAR10(root=root, train=train, download=True, transform=transform)
