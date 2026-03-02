from pathlib import Path

root = Path.cwd().parent


def model_path(file_name):
    model_path = root / 'data' / 'models'
    Path(model_path).mkdir(parents=True, exist_ok=True)

    return str(model_path / file_name)
