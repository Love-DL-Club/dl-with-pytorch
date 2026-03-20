import torch
from torch.utils.data.dataloader import DataLoader

from lib.dataset.cifar10_data import load_data
from lib.utils.device import available_device
from lib.utils.path import data_path, model_path
from no_data_gan.transforms import create


def main(model):
    device = available_device()

    model.load_state_dict(torch.load(model_path('teacher.pth'), map_location=device))

    transforms = create()
    test_data = load_data(data_path(), train=False, transform=transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    num_corr = 0

    with torch.no_grad():
        for data, label in test_loader:
            output = model(data.to(device))

            preds = output.data.max(1)[1]

            corr = preds.eq(label.to(device).data).sum().item()
            num_corr += corr

        print(f'Accuracy: {num_corr / len(test_data)}')
