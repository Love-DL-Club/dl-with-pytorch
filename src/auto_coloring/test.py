import numpy as np
import torch

from auto_coloring.dataset import AutoColoring
from auto_coloring.models import AutoColoringModel
from auto_coloring.utils import lab2rgb
from lib.utils.device import available_device
from lib.utils.path import model_path


def run():
    device = available_device()
    dataset = AutoColoring()

    model = AutoColoringModel().to(device)

    test_L, test_AB = dataset[0]
    test_L = np.expand_dims(test_L, axis=0)
    real_img = np.concatenate([test_L, test_AB])
    real_img = real_img.transpose(1, 2, 0).astype(np.uint8)
    real_img = lab2rgb(real_img)
    pred_img = None

    with torch.no_grad():
        model.load_state_dict(
            torch.load(model_path('AutoColor.pth'), map_location=device)
        )

        input_tensor = torch.tensor(test_L)
        input_tensor = torch.unsqueeze(input_tensor, dim=0).to(device)
        pred_AB = model(input_tensor)

        pred_LAB = torch.cat([input_tensor, pred_AB], dim=1)
        pred_LAB = torch.squeeze(pred_LAB)
        pred_LAB = pred_LAB.permute(1, 2, 0).cpu().numpy()
        pred_LAB = lab2rgb(pred_LAB.astype(np.uint8))
        pred_img = pred_LAB

    return real_img, pred_img


if __name__ == '__main__':
    run()
