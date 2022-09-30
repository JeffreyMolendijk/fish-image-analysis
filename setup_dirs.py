from pathlib import Path

def setup_dirs():
    Path('./input/crop').mkdir(parents=True, exist_ok=True)
    Path('./input/magenta').mkdir(parents=True, exist_ok=True)

