# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dl-with-pytorch
#     language: python
#     name: python3
# ---

# %%
import glob

import matplotlib.pyplot as plt
from PIL import Image

from lib.utils.path import data_path

imgs = glob.glob(str(data_path() / 'CH09' / '*.jpg'))

for i in range(9):
    img = Image.open(imgs[i])
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)

plt.show()

# %%
from auto_coloring.train import run

run()

# %%
import matplotlib.pyplot as plt

from auto_coloring.test import run

image, pred = run()

print(pred)

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('real image')

plt.subplot(1, 2, 2)
plt.imshow(pred)
plt.title('predicted image')

plt.show()
