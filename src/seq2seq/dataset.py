import string

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def get_bow(corpus):
    BOW = {'<SOS>': 0, '<EOS>': 1}

    for line in corpus:
        for word in line.split():
            if word not in BOW:
                BOW[word] = len(BOW.keys())

    return BOW


class Eng2Kor(Dataset):
    def __init__(self, pth2txt):
        self.eng_corpus = []
        self.kor_corpus = []

        with open(pth2txt, encoding='utf-8') as f:
            lines = f.read().split('\n')

            for line in lines:
                txt = ''.join(v for v in line if v not in string.punctuation).lower()

                eng_txt = txt.split('\t')[0]
                kor_txt = txt.split('\t')[1]

                if len(eng_txt.split()) <= 10 and len(kor_txt.split()) <= 10:
                    self.eng_corpus.append(eng_txt)
                    self.kor_corpus.append(kor_txt)

        self.eng_bow = get_bow(self.eng_corpus)
        self.kor_bow = get_bow(self.kor_corpus)

    def gen_seq(self, line):
        seq = line.split()
        seq.append('<EOS>')

        return seq

    def __len__(self):
        return len(self.eng_corpus)

    def __getitem__(self, i):
        data = np.array([self.eng_bow[txt] for txt in self.gen_seq(self.eng_corpus[i])])

        label = np.array(
            [self.kor_bow[txt] for txt in self.gen_seq(self.kor_corpus[i])]
        )

        return data, label


def loader(dataset):
    for i in range(len(dataset)):
        data, label = dataset[i]

        yield torch.tensor(data), torch.tensor(label)
