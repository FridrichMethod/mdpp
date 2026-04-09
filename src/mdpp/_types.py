"""Shared type aliases for the mdpp package."""

import os
from pathlib import Path

import numpy as np

type StrPath = str | os.PathLike[str]
type PathLike = str | Path
type FloatDType = type[np.float32] | type[np.float64]
