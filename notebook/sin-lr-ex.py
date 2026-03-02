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
import math

import koreanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import torch

x = torch.linspace(-math.pi, math.pi, 1000)

y = torch.sin(x)

a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

y_random = a * x**3 + b * x**2 + c * x + d

plt.subplot(2, 1, 1)
plt.title('y true')
plt.plot(x, y)

plt.subplot(2, 1, 2)
plt.title('y random')
plt.plot(x, y_random)

plt.tight_layout()
plt.show()

# %%
learning_rate = 1e-6

for ep in range(2000):
    y_pred = a * x**3 + b * x**2 + c * x + d

    loss = (y_pred - y).pow(2).sum().item()
    if ep % 100 == 0:
        print(f'epoch {ep + 1} loss: {loss}')

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (grad_y_pred * x**3).sum()
    grad_b = (grad_y_pred * x**2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


plt.subplot(3, 1, 1)
plt.title('y true')
plt.plot(x, y)

plt.subplot(3, 1, 2)
plt.title('y pred')
plt.plot(x, y_pred)

plt.subplot(3, 1, 3)
plt.title('y random')
plt.plot(y_random)

plt.tight_layout()
plt.show()
