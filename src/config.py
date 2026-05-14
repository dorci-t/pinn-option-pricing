"""
TOML config loader.
"""

import tomllib
from pathlib import Path


def load_config(path):
    with open(Path(path), "rb") as f:
        return tomllib.load(f)
