from pathlib import Path

root = Path.cwd().parent


def model_path(file_name):
    return str(root / 'data' / 'models' / file_name)
