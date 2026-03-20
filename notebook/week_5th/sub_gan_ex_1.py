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
from no_data_gan.train_sub import main as teacher

model = teacher()

# %%
from no_data_gan.test_sub import main as student

student(model)
