import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam

from gan.dataset import load_data
from gan.models import Discriminator, Generator, weights_init
from lib.utils.device import available_device
from lib.utils.path import model_path


def main():
    device = available_device()

    G = Generator().to(device)
    G.apply(weights_init)

    D = Discriminator().to(device)
    D.apply(weights_init)

    loader = load_data()

    criterion = nn.BCELoss()

    G_optim = Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    D_optim = Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for ep in range(50):
        iterator = tqdm.tqdm(enumerate(loader, 0), total=len(loader))

        for i, data in iterator:
            D_optim.zero_grad()

            label = torch.ones_like(data[1], dtype=torch.float32).to(device)
            label_fake = torch.zeros_like(data[1], dtype=torch.float32).to(device)

            real = D(data[0].to(device))

            Dloss_real = criterion(torch.squeeze(real), label)
            Dloss_real.backward()

            noise = torch.randn(label.shape[0], 100, 1, 1, device=device)
            fake = G(noise)

            output = D(fake.detach())

            Dloss_fake = criterion(torch.squeeze(output), label_fake)
            Dloss_fake.backward()

            Dloss = Dloss_real + Dloss_fake
            D_optim.step()

            G_optim.zero_grad()
            output = D(fake)
            Gloss = criterion(torch.squeeze(output), label)
            Gloss.backward()

            G_optim.step()

            iterator.set_description(
                f'epoch:{ep} iteration:{i} D_loss:{Dloss} G_loss:{Gloss}'
            )

    torch.save(G.state_dict(), model_path('Generator.pth'))
    torch.save(D.state_dict(), model_path('Discriminator.pth'))
