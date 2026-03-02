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
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
# 데이터는 housing.data, 타겟(집값)은 housing.target에 들어있습니다.
print(dataset.keys())

df = pd.DataFrame(dataset['data'])
df.columns = dataset['feature_names']
df['target'] = dataset['target']

print(df.head())

# %%
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.adam import Adam

X = df.iloc[:, :8].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 0~1 사이 혹은 표준편차 기반으로 정규화

# 모델 정의 (입력 크기를 8로 변경)
model = nn.Sequential(
    nn.Linear(8, 100),  # 13 -> 8로 수정
    nn.ReLU(),
    nn.Linear(100, 1),
)

batch_size = 100
learning_rate = 1e-3
optim = Adam(model.parameters(), lr=learning_rate)

for ep in range(200):
    for i in range(len(X_scaled) // batch_size):
        start = i * batch_size
        end = start + batch_size

        x = torch.FloatTensor(X_scaled[start:end])
        # y의 차원을 (batch_size, 1)로 맞춰줍니다.
        y = torch.FloatTensor(Y[start:end]).view(-1, 1)

        optim.zero_grad()
        preds = model(x)

        # Loss 함수를 반복문 밖에서 정의하거나 아래처럼 사용
        loss = nn.MSELoss()(preds, y)

        loss.backward()
        optim.step()

    if ep % 20 == 0:
        print(f'epoch {ep} loss: {loss.item()}')

# %%
pred = model(torch.FloatTensor(X_scaled[0, :13]))
real = Y[0]

print(f'pred: {pred.item()} real: {real}')

# %%
from sklearn.metrics import mean_squared_error, r2_score

# 평가 모드로 전환
model.eval()

with torch.no_grad():
    # 전체 데이터 X에 대한 예측값 생성
    # (실제로는 학습에 쓰지 않은 '테스트 데이터'로 하는 것이 정석입니다)
    X_all = torch.FloatTensor(X_scaled)
    y_true = Y

    # 모델 예측
    preds = model(X_all)
    # 2차원인 preds를 계산을 위해 1차원으로 다시 바꿈
    preds = preds.detach().numpy().flatten()

# 성능 지표 계산 (Scikit-learn 활용)

mse = mean_squared_error(y_true, preds)
r2 = r2_score(y_true, preds)

print(f'Final MSE: {mse:.4f}')
print(f'Final R2 Score: {r2:.4f}')
