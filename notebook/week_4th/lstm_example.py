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
import pandas as pd

from lib.utils.path import data_path

df = pd.read_csv(data_path() / 'CH10' / 'ArticlesApril2017.csv')
print(df.columns)

# %%
from lstm.train import run

run()

# %%
from lstm.test import run

run()
