import matplotlib.pyplot as plt
import torch

from gan.models import Discriminator, Generator, weights_init
from lib.utils.device import available_device
from lib.utils.path import model_path


def main():
    device = available_device()

    G = Generator().to(device)
    G.apply(weights_init)

    D = Discriminator()
    D.apply(weights_init)

    with torch.no_grad():
        G.load_state_dict(torch.load(model_path('Generator.pth'), map_location=device))
        feature_vector = torch.randn(1, 100, 1, 1).to(device)

        pred = G(feature_vector).squeeze()
        pred = pred.permute(1, 2, 0).cpu().numpy()

        plt.imshow(pred)
        plt.title('predicted image')
        plt.show()
