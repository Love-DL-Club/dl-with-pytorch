import glob

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from auto_coloring.utils import rgb2lab
from lib.utils.path import data_path


class AutoColoring(Dataset):
    def __init__(self):
        self.data = glob.glob(str(data_path() / 'CH09' / '*.jpg'))
        if len(self.data) == 0:
            raise FileNotFoundError(
                '데이터셋 경로에 이미지가 없습니다. 경로를 확인하세요.'
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rgb = np.array(Image.open(self.data[i]).resize((256, 256)))

        lab = rgb2lab(rgb)

        lab = lab.transpose(2, 0, 1).astype(np.float32)

        return lab[0], lab[1:]
