"""Tests for mdpp._dtype module."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mdpp._dtype import (
    default_dtype,
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

    def test_set_default_dtype_float64(self) -> None:
        with default_dtype(np.float32):  # restore on exit
            set_default_dtype(np.float64)
            assert get_default_dtype() == np.dtype(np.float64)

    def test_set_default_dtype_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            set_default_dtype(np.int32)  # type: ignore[arg-type]


class TestDefaultDtypeContextManager:
    """Tests for default_dtype context manager."""

    def test_context_manager_restores(self) -> None:
        assert get_default_dtype() == np.dtype(np.float32)
        with default_dtype(np.float64):
            assert get_default_dtype() == np.dtype(np.float64)
        assert get_default_dtype() == np.dtype(np.float32)

    def test_context_manager_restores_on_exception(self) -> None:
        assert get_default_dtype() == np.dtype(np.float32)
        try:
            with default_dtype(np.float64):
                assert get_default_dtype() == np.dtype(np.float64)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert get_default_dtype() == np.dtype(np.float32)


class TestResolveDtype:
    """Tests for resolve_dtype."""

    def test_resolve_dtype_explicit_overrides(self) -> None:
        result = resolve_dtype(np.float64)
        assert result == np.dtype(np.float64)

    def test_resolve_dtype_none_uses_default(self) -> None:
        result = resolve_dtype(None)
        assert result == np.dtype(np.float32)

    def test_resolve_dtype_none_respects_context(self) -> None:
        with default_dtype(np.float64):
            assert resolve_dtype(None) == np.dtype(np.float64)

    def test_resolve_dtype_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            resolve_dtype(np.int64)  # type: ignore[arg-type]


class TestThreadIsolation:
    """Tests for thread safety via ContextVar."""

    def test_thread_isolation(self) -> None:
        """Two threads with different dtypes do not interfere."""
        results: dict[str, np.dtype] = {}  # type: ignore[type-arg]
        barrier = threading.Barrier(2)

        def worker(name: str, dtype: type[np.floating]) -> None:  # type: ignore[type-arg]
            set_default_dtype(dtype)
            barrier.wait()  # Ensure both threads have set their dtype.
            results[name] = get_default_dtype()

        t1 = threading.Thread(target=worker, args=("t1", np.float32))
        t2 = threading.Thread(target=worker, args=("t2", np.float64))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == np.dtype(np.float32)
        assert results["t2"] == np.dtype(np.float64)
