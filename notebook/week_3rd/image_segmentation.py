# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: study-club (3.11.14)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
from PIL import Image

from lib.utils.path import data_path

# https://drive.google.com/drive/folders/1Qd-zNPa-09b_2CPN4Lj7wqDyFl_mxjpH?usp=drive_link
# https://drive.google.com/drive/folders/1ckoEygLFntQbRqUsrD20AA6TwzRaRLqc?usp=drive_link
base_path = f'{data_path()}/CH07'

path_to_annotations = f'{base_path}/annotations/trimaps/'
path_to_image = f'{base_path}/images/'


annotation = Image.open(path_to_annotations + 'Abyssinian_1.png')
plt.subplot(1, 2, 1)
plt.title('annotation')
plt.imshow(annotation)

image = Image.open(path_to_image + 'Abyssinian_1.jpg')
plt.subplot(1, 2, 2)
plt.title('image')
plt.imshow(image)

plt.show()

# %%
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class Pets(Dataset):
    def __init__(
        self,
        path_to_img,
        path_to_anno,
        train=True,
        transforms=None,
        input_size=(128, 128),
    ):

        self.images = sorted(glob.glob(path_to_img + '/*.jpg'))
        self.annotations = sorted(glob.glob(path_to_anno + '/*.png'))

        self.X_train = self.images[: int(0.8 * len(self.images))]
        self.X_test = self.images[int(0.8 * len(self.images)) :]
        self.y_train = self.annotations[: int(0.8 * len(self.annotations))]
        self.y_test = self.annotations[int(0.8 * len(self.annotations)) :]

        self.train = train
        self.transforms = transforms
        self.input_size = input_size

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)

    def preprocess_mask(self, mask):
        mask = mask.resize(self.input_size)
        mask = np.array(mask).astype(np.float32)
        mask[mask != 2.0] = 1.0
        mask[mask == 2.0] = 0.0
        mask = torch.tensor(mask)
        return mask

    def __getitem__(self, i):
        if self.train:
            X_train = Image.open(self.X_train[i])
            X_train = self.transforms(X_train)
            y_train = Image.open(self.y_train[i])
            y_train = self.preprocess_mask(y_train)

            return X_train, y_train
        else:
            X_test = Image.open(self.X_test[i])
            X_test = self.transforms(X_test)
            y_test = Image.open(self.y_test[i])
            y_test = self.preprocess_mask(y_test)

            return X_test, y_test


# %%
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.enc5_2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.upsample_4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec4_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.upsample_3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec3_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.upsample_2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upsample_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.enc1_1(x)
        x = self.relu(x)
        e1 = self.enc1_2(x)
        e1 = self.relu(e1)
        x = self.pool_1(e1)

        x = self.enc2_1(x)
        x = self.relu(x)
        e2 = self.enc2_2(x)
        e2 = self.relu(e2)
        x = self.pool_2(e2)

        x = self.enc3_1(x)
        x = self.relu(x)
        e3 = self.enc3_2(x)
        e3 = self.relu(e3)
        x = self.pool_3(e3)

        x = self.enc4_1(x)
        x = self.relu(x)
        e4 = self.enc4_2(x)
        e4 = self.relu(e4)
        x = self.pool_4(e4)

        x = self.enc5_1(x)
        x = self.relu(x)
        x = self.enc5_2(x)
        x = self.relu(x)

        x = self.upsample_4(x)

        x = torch.cat([x, e4], dim=1)
        x = self.dec4_1(x)
        x = self.relu(x)
        x = self.dec4_2(x)
        x = self.relu(x)

        x = self.upsample_3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3_1(x)
        x = self.relu(x)
        x = self.dec3_2(x)
        x = self.relu(x)

        x = self.upsample_2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2_1(x)
        x = self.relu(x)
        x = self.dec2_2(x)
        x = self.relu(x)

        x = self.upsample_1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1_1(x)
        x = self.relu(x)
        x = self.dec1_2(x)
        x = self.relu(x)
        x = self.dec1_3(x)

        x = torch.squeeze(x)

        return x


# %%
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from lib.utils.device import available_device
from lib.utils.path import model_path

device = available_device()

transform = Compose([Resize((128, 128)), ToTensor()])

train_set = Pets(
    path_to_img=path_to_image, path_to_anno=path_to_annotations, transforms=transform
)
test_set = Pets(
    path_to_img=path_to_image,
    path_to_anno=path_to_annotations,
    transforms=transform,
    train=False,
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set)

model = UNet().to(device)

lr = 1e-4

optim = Adam(params=model.parameters(), lr=lr)

criterion = nn.BCEWithLogitsLoss()

for ep in range(200):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = criterion(preds, label.type(torch.FloatTensor).to(device))
        loss.backward()

        optim.step()

        iterator.set_description(f'epoch {ep + 1} loss : {loss.item()}')

torch.save(model.state_dict(), model_path('UNet.pth'))


# %%
import matplotlib.pyplot as plt

model.load_state_dict(torch.load(model_path('UNet.pth'), map_location='cpu'))

data, label = test_set[1]

pred = model(torch.unsqueeze(data.to(device), dim=0)) > 0.5
pred_for_plot = pred.detach().cpu().squeeze().numpy()

with torch.no_grad():
    plt.subplot(1, 2, 1)
    plt.title('Predicted')
    plt.imshow(pred_for_plot)

    plt.subplot(1, 2, 2)
    plt.title('Real')
    plt.imshow(label)

    plt.show()
