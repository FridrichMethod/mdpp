"""Shared type aliases for the mdpp package."""

import os
from pathlib import Path

type StrPath = str | os.PathLike[str]
type PathLike = str | Path
