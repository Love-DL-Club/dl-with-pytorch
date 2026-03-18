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
import glob
import os

import matplotlib.pyplot as plt
from PIL import Image

from lib.utils.path import data_path

image_path = data_path() / 'images' / 'GAN' / 'img_align_celeba'
imgs = glob.glob(os.path.join(image_path, '*'))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = Image.open(imgs[i])
    plt.imshow(img)

plt.show()

# %%
from gan.test import main

main()
