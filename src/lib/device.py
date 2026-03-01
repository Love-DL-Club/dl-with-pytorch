import sys

import torch

CPU = 'cpu'


def available_device():
    if sys.platform == 'wind32' or sys.platform == 'linux':
        return 'cuda' if torch.cuda.is_available() else CPU

    return 'mps' if torch.backends.mps.is_available() else CPU
