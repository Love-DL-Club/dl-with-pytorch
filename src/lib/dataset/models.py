from pathlib import Path

import gdown

root = Path.cwd().parent
url = 'https://drive.google.com/uc?id=1kXhRuN_EvgxzfvFhNPvoYmH541C6xgIQ'

default_output = root / 'data' / 'models'

print('default_output', default_output)


def download(output=default_output, file_name='best_model.pth'):
    output = output / file_name
    output.parent.mkdir(parents=True, exist_ok=True)

    gdown.download(url, str(output), quiet=False)
