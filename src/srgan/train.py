import torch
import torch.nn as nn
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

from lib.utils.device import available_device
from lib.utils.path import model_path
from srgan.dataset import CelebA
from srgan.models import Discriminator, FeatureExtractor, Generator


def main():
    device = available_device()

    dataset = CelebA()
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    G_optim = Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    D_optim = Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    for ep in range(1):
        iterator = tqdm.tqdm(loader)

        for i, (low_res, high_res) in enumerate(iterator):
            G_optim.zero_grad()
            D_optim.zero_grad()

            label_true = torch.ones(batch_size, dtype=torch.float32).to(device)
            label_false = torch.zeros(batch_size, dtype=torch.float32).to(device)

            fake_hr = G(low_res.to(device))
            GAN_loss = criterion_mse(D(fake_hr), label_true)

            fake_features = feature_extractor(fake_hr)
            real_features = feature_extractor(high_res.to(device))
            content_loss = criterion_l1(fake_features, real_features)

            loss_G = content_loss + 1e-3 * GAN_loss
            loss_G.backward()
            G_optim.step()

            real_loss = criterion_mse(D(high_res.to(device)), label_true)
            fake_loss = criterion_mse(D(fake_hr.detach()), label_false)
            loss_D = (real_loss + fake_loss) / 2
            loss_D.backward()
            D_optim.step()

            iterator.set_description(
                f'epoch:{ep} iterator:{i} G_loss:{GAN_loss} D_loss:{loss_D}'
            )

    torch.save(G.state_dict(), model_path('SRGAN_G.pth'))
    torch.save(D.state_dict(), model_path('SRGAN_D.pth'))
