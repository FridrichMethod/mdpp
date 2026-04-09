"""Tests for mdpp._dtype module."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp._dtype import (
    get_default_dtype,
    resolve_dtype,
    set_default_dtype,
)


class TestGetDefaultDtype:
    """Tests for get_default_dtype."""

    def test_default_is_float32(self) -> None:
        assert get_default_dtype() == np.dtype(np.float32)


class TestSetDefaultDtype:
    """Tests for set_default_dtype."""

    def test_set_and_restore(self) -> None:
        try:
            set_default_dtype(np.float64)
            assert get_default_dtype() == np.dtype(np.float64)
        finally:
            set_default_dtype(np.float32)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            set_default_dtype(np.int32)  # type: ignore[arg-type]


class TestResolveDtype:
    """Tests for resolve_dtype."""

    def test_explicit_overrides(self) -> None:
        assert resolve_dtype(np.float64) == np.dtype(np.float64)

    def test_none_uses_default(self) -> None:
        assert resolve_dtype(None) == np.dtype(np.float32)

    def test_none_respects_global(self) -> None:
        try:
            set_default_dtype(np.float64)
            assert resolve_dtype(None) == np.dtype(np.float64)
        finally:
            set_default_dtype(np.float32)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            resolve_dtype(np.int64)  # type: ignore[arg-type]
