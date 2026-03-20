import torch

import no_data_gan.dataset as dataset
from lib.utils.device import available_device
from lib.utils.path import model_path


def main(model):
    device = available_device()

    num_corr = 0

    model.load_state_dict(torch.load(model_path('student.pth'), map_location=device))

    train_data, train_loader = dataset.train_loader()

    with torch.no_grad():
        for data, label in train_loader:
            output = model(data.to(device))
            preds = output.data.max(1)[1]
            corr = preds.eq(label.to(device).data).sum().item()
            num_corr += corr

        print(f'Accuracy:{num_corr / len(train_data)}')

    num_corr = 0

    test_data, test_loader = dataset.test_loader()

    with torch.no_grad():
        for data, label in test_loader:
            output = model(data.to(device))
            preds = output.data.max(1)[1]
            corr = preds.eq(label.to(device).data).sum().item()
            num_corr += corr

        print(f'Accuracy:{num_corr / len(test_data)}')
