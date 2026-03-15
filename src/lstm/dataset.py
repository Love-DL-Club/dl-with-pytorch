import glob
import string

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

from lib.utils.path import data_path


class TextGeneration(Dataset):
    def clean_text(self, txt):
        txt = ''.join(v for v in txt if v not in string.punctuation).lower()

        return txt

    def __init__(self):
        all_headlines = []

        for filename in glob.glob(str(data_path() / 'CH10' / '*.csv')):
            if 'Articles' in filename:
                article_df = pd.read_csv(filename)

                all_headlines.extend(list(article_df.headline.values))
                break

        all_headlines = [h for h in all_headlines if h != 'Unknown']

        self.corpus = [self.clean_text(x) for x in all_headlines]
        self.BOW = {}

        for line in self.corpus:
            for word in line.split():
                if word not in self.BOW:
                    self.BOW[word] = len(self.BOW.keys())

        self.data = self.generate_sequence(self.corpus)

    def generate_sequence(self, txt):
        seq = []

        for line in txt:
            line = line.split()
            line_bow = [self.BOW[word] for word in line]

            data = [
                ([line_bow[i], line_bow[i + 1]], line_bow[i + 2])
                for i in range(len(line_bow) - 2)
            ]

            seq.extend(data)

        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = np.array(self.data[i][0])
        label = np.array(self.data[i][1]).astype(np.float32)

        return data, label
