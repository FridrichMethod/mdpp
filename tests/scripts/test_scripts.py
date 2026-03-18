"""Tests for mdpp.data resource APIs and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mdpp.data import copy_mdp_files, get_mdp_template, list_mdp_templates


class TestListMdpTemplates:
    """Tests for list_mdp_templates."""

    def test_returns_five(self) -> None:
        result = list_mdp_templates()
        assert len(result) == 5

    def test_contains_production(self) -> None:
        result = list_mdp_templates()
        assert any("step5_production" in name for name in result)


class TestGetMdpTemplate:
    """Tests for get_mdp_template."""

    def test_reads_by_short_name(self) -> None:
        content = get_mdp_template("step5_production")
        assert "integrator" in content

    def test_reads_with_extension(self) -> None:
        content = get_mdp_template("step5_production.mdp")
        assert "integrator" in content

    def test_reads_with_prefix(self) -> None:
        content = get_mdp_template("mdps/step5_production.mdp")
        assert "integrator" in content

    def test_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_mdp_template("nonexistent")


class TestCopyMdpFiles:
    """Tests for copy_mdp_files."""

    def test_copies_all(self, tmp_path: Path) -> None:
        written = copy_mdp_files(tmp_path)
        assert len(written) == 5
        names = {p.name for p in written}
        assert "step5_production.mdp" in names


class TestCli:
    """Tests for the mdpp CLI."""

    def test_mdps_command(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mdpp._cli", "mdps", str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "step5_production.mdp" in result.stdout
