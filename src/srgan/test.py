import matplotlib.pyplot as plt
import torch

from lib.utils.device import available_device
from lib.utils.path import model_path
from srgan.dataset import CelebA
from srgan.models import Generator


def main():
    device = available_device()
    dataset = CelebA()
    G = Generator().to(device)

    G.load_state_dict(torch.load(model_path('SRGAN_G.pth'), map_location=device))

    with torch.no_grad():
        low_res, _ = dataset[0]

        input_tensor = torch.unsqueeze(low_res, dim=0).to(device)

        pred = G(input_tensor)
        pred = pred.squeeze()
        pred = pred.permute(1, 2, 0).cpu().numpy()

        low_res = low_res.permute(1, 2, 0).numpy()

        plt.subplot(1, 2, 1)
        plt.title('low resolution image')
        plt.imshow(low_res)
        plt.subplot(1, 2, 2)
        plt.imshow(pred)
        plt.title('predicted high resolution image')
        plt.show()
