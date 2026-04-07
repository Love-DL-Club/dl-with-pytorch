# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dl-with-pytorch (3.11.14)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


# %%
# 데이터 전처리기 생성
transforms = Compose(
    [
        T.ToPILImage(),
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ]
)

# %%
# CIFAR-10 데이터셋 불러오기
training_data = CIFAR10(root='./', train=True, download=True, transform=ToTensor())
# 테스트 데이터에 똑같은 전처리기를 써도 되나?
test_data = CIFAR10(root='./', train=False, download=True, transform=ToTensor())

# %%
# 정규화된 이미지가 아닌 원본이미지로 나오는 이유가 뭐지...
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(training_data.data[i])
plt.show()
