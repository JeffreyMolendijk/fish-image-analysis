from pathlib import Path


def setup_dirs():
    """
    Creates the necessary directory structure for output files.

    This function ensures the existence of the following directories:
    - './output/crop' for storing cropped images.
    - './output/magenta' for storing images analyzed for magenta content.
    - './output/cluster' for storing images or data related to clustering.

    If the directories already exist, it does nothing (due to `exist_ok=True`).
    """
    Path("./output/crop").mkdir(parents=True, exist_ok=True)
    Path("./output/magenta").mkdir(parents=True, exist_ok=True)
    Path("./output/cluster").mkdir(parents=True, exist_ok=True)
