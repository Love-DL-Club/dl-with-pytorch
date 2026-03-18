import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder

from lib.utils.path import data_path


def load_data():
    transforms = T.Compose(
        [
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = ImageFolder(root=data_path() / 'images/GAN', transform=transforms)

    return DataLoader(dataset, batch_size=128, shuffle=True)
