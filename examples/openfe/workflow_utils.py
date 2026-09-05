"""Input fingerprints for the OpenFE example campaigns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def input_digest(files: Iterable[Path], *, settings: Mapping[str, object]) -> str:
    """Fingerprint exact file contents and settings before reusing calculations.

    Args:
        files: Input files, including poses or charged SDFs as appropriate.
        settings: JSON-serializable settings, including software versions.

    Returns:
        SHA-256 digest suitable for a cache or campaign directory name.

    Raises:
        OSError: If an input file cannot be read.
        TypeError: If settings cannot be serialized as JSON.
    """
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in sorted(files, key=str):
        # Length prefixes prevent ambiguous concatenation of adjacent files.
        name = path.name.encode()
        contents = path.read_bytes()
        for data in (name, contents):
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()
