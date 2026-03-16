import glob
import string

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


def get_bow(corpus):
    BOW = {'<pad>': 0}

    for letter in corpus:
        if letter not in BOW:
            BOW[letter] = len(BOW.keys())

    return BOW


class Captcha(Dataset):
    def __init__(self, path, train=True):
        self.corpus = string.ascii_lowercase + string.digits
        self.BOW = get_bow(self.corpus)

        self.imgfiles = glob.glob(path + '/*.png')

        self.train = train
        self.trainset = self.imgfiles[: int(len(self.imgfiles) * 0.8)]
        self.testset = self.imgfiles[int(len(self.imgfiles) * 0.8) :]

    def get_seq(self, line):
        label = []

        for letter in line:
            label.append(self.BOW[letter])

        return label

    def __len__(self):
        if self.train:
            return len(self.imgfiles)
        else:
            return len(self.testset)

    def __getitem__(self, i):
        if self.train:
            data = Image.open(self.trainset[i]).convert('RGB')

            label = self.trainset[i].split('/')[-1]
            label = label.split('.png')[0]
            label = self.get_seq(label)

            data = np.array(data).astype(np.float32)
            data = np.transpose(data, (2, 0, 1))
            label = np.array(label)

            return data, label
        else:
            data = Image.open(self.testset[i]).convert('RGB')
            label = self.testset[i].split('/')[-1]
            label = label.split('.png')[0]
            label = self.get_seq(label)

            data = np.array(data).astype(np.float32)
            label = np.array(label)

            return data, label
