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
from no_data_gan.train import main

model = main()

# %%
from no_data_gan.test import main

main(model)
