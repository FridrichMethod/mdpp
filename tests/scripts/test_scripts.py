"""Tests for mdpp.data resource APIs and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mdpp.data import FORCE_FIELDS, copy_mdp_files, get_mdp_template, list_mdp_templates


class TestListMdpTemplates:
    """Tests for list_mdp_templates."""

    @pytest.mark.parametrize("ff", sorted(FORCE_FIELDS))
    def test_returns_five(self, ff: str) -> None:
        result = list_mdp_templates(ff)
        assert len(result) == 5

    @pytest.mark.parametrize("ff", sorted(FORCE_FIELDS))
    def test_contains_production(self, ff: str) -> None:
        result = list_mdp_templates(ff)
        assert any("step5_production" in name for name in result)

    def test_default_is_charmm(self) -> None:
        result = list_mdp_templates()
        assert all("charmm" in name for name in result)

    def test_invalid_ff_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown force field"):
            list_mdp_templates("opls")


class TestGetMdpTemplate:
    """Tests for get_mdp_template."""

    def test_reads_by_short_name(self) -> None:
        content = get_mdp_template("step5_production")
        assert "integrator" in content

    def test_reads_with_extension(self) -> None:
        content = get_mdp_template("step5_production.mdp")
        assert "integrator" in content

    def test_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_mdp_template("nonexistent")

    def test_amber_ff(self) -> None:
        content = get_mdp_template("step5_production", ff="amber")
        assert "integrator" in content
        assert "EnerPres" in content

    def test_charmm_ff(self) -> None:
        content = get_mdp_template("step5_production", ff="charmm")
        assert "Force-switch" in content

    def test_amber_has_no_force_switch(self) -> None:
        content = get_mdp_template("step4.0_minimization", ff="amber")
        assert "Force-switch" not in content
        assert "Potential-shift" in content


class TestCopyMdpFiles:
    """Tests for copy_mdp_files."""

    @pytest.mark.parametrize("ff", sorted(FORCE_FIELDS))
    def test_copies_all(self, tmp_path: Path, ff: str) -> None:
        written = copy_mdp_files(tmp_path, ff=ff)
        assert len(written) == 5
        names = {p.name for p in written}
        assert "step5_production.mdp" in names

    def test_default_is_charmm(self, tmp_path: Path) -> None:
        copy_mdp_files(tmp_path)
        content = (tmp_path / "step5_production.mdp").read_text()
        assert "Force-switch" in content


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

    def test_mdps_command_amber(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mdpp._cli", "mdps", "--ff", "amber", str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "step5_production.mdp" in result.stdout
        content = (tmp_path / "step5_production.mdp").read_text()
        assert "EnerPres" in content
