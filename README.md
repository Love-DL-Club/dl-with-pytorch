# 사랑의 딥러닝단 
## 텐초의 파이토치 특강 스터디

[예제 코드](https://colab.research.google.com/drive/11k-SAzMs6mw_0gTMxmIAIY2gCfB8VbqB)

python version : 3.11
```py
uv sync
```

src/lib 안의 py를 import 할려면
```py
uv pip install -e .

# import 가 잘 안되는경우 vscode reload
```

## vscode 확장 프로그램 ruff 설치

# jupytext 실행법
|![alt text](image.png)|![alt text](image-1.png)|

화면대로 enter, enter

![alt text](image-2.png)
Jupytext 설치

# nbstripout 활성화
```
uv run nbstripout --install
```

# lib.path 사용법
```
from lib.path import model_path

torch.load(model_path('xxx.pth'))

from lib.path import data_path

# 어느 위치에 있던 root/data 폴더 경로를 읽어옵니다.
train_data = MNIST(root=data_path(), train=True, download=True, transform=ToTensor())
```

git test