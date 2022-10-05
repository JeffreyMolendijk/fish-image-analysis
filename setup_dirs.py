from pathlib import Path

def setup_dirs():
    Path('./output/crop').mkdir(parents=True, exist_ok=True)
    Path('./output/magenta').mkdir(parents=True, exist_ok=True)
    Path('./output/cluster').mkdir(parents=True, exist_ok=True)

