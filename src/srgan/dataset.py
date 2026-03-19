import glob

import torchvision.transforms as T
from PIL import Image
from torch.utils.data.dataset import Dataset

from lib.utils.path import data_path


class CelebA(Dataset):
    def __init__(self):
        self.imgs = glob.glob(
            str(data_path() / 'images' / 'img_align_celeba' / '*.jpg')
        )

        mean_std = (0.5, 0.5, 0.5)

        self.low_res_tf = T.Compose(
            [T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean_std, mean_std)]
        )

        self.high_res_tf = T.Compose(
            [T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean_std, mean_std)]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i])

        img_low_res = self.low_res_tf(img)
        img_high_res = self.high_res_tf(img)

        return [img_low_res, img_high_res]
