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
import string

from lib.utils.path import data_path

letters = []

with open(data_path() / 'Ch11.txt', encoding='utf-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        txt = ''.join(v for v in line if v not in string.punctuation).lower()
        letters.append(txt)

print(letters[:5])
