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

import matplotlib.pyplot as plt
from PIL import Image

from lib.utils.path import data_path

imgfile = glob.glob(str(data_path() / 'CH12' / '*.png'))[0]

imgfile = Image.open(imgfile)

plt.imshow(imgfile)
plt.show()

# %%
from crnn.test import main

main()
