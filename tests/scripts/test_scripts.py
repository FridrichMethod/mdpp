"""Tests for mdpp.scripts and mdpp.data resource APIs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mdpp.data import copy_mdp_files, get_mdp_template, list_mdp_templates
from mdpp.scripts import copy_scripts, get_script_path, list_scripts, read_script


class TestListScripts:
    """Tests for list_scripts."""

    def test_returns_nonempty(self) -> None:
        result = list_scripts()
        assert len(result) > 0

    def test_contains_known_script(self) -> None:
        result = list_scripts()
        assert "gromacs/analysis/gmx_rmsd.sh" in result

    def test_filter_by_prefix(self) -> None:
        result = list_scripts("gromacs/analysis")
        assert all(s.startswith("gromacs/analysis") for s in result)
        assert len(result) >= 8  # 8 analysis scripts + sherlock

    def test_filter_by_runtime(self) -> None:
        result = list_scripts("gromacs/runtime")
        assert "gromacs/runtime/restart.sh" in result
        assert "gromacs/runtime/check_status.sh" in result

    def test_empty_prefix_returns_all(self) -> None:
        assert list_scripts("") == list_scripts()


class TestGetScriptPath:
    """Tests for get_script_path."""

    def test_returns_existing_path(self) -> None:
        path = get_script_path("gromacs/analysis/gmx_rmsd.sh")
        assert path.exists()
        assert path.is_file()

    def test_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_script_path("nonexistent/script.sh")


class TestReadScript:
    """Tests for read_script."""

    def test_reads_content(self) -> None:
        content = read_script("gromacs/analysis/gmx_rmsd.sh")
        assert "#!/" in content


class TestCopyScripts:
    """Tests for copy_scripts."""

    def test_copies_category(self, tmp_path: Path) -> None:
        written = copy_scripts("gromacs/runtime", tmp_path)
        assert len(written) >= 4
        names = {p.name for p in written}
        assert "restart.sh" in names
        assert "check_status.sh" in names

    def test_raises_on_missing_category(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            copy_scripts("nonexistent/category", tmp_path)

    def test_raises_on_overwrite(self, tmp_path: Path) -> None:
        copy_scripts("gromacs/runtime", tmp_path)
        with pytest.raises(FileExistsError):
            copy_scripts("gromacs/runtime", tmp_path)

    def test_overwrite_flag(self, tmp_path: Path) -> None:
        copy_scripts("gromacs/runtime", tmp_path)
        written = copy_scripts("gromacs/runtime", tmp_path, overwrite=True)
        assert len(written) >= 4


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

    def test_list_command(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mdpp.scripts._cli", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "gromacs/analysis/gmx_rmsd.sh" in result.stdout

    def test_list_with_prefix(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mdpp.scripts._cli", "list", "gromacs/runtime"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "restart.sh" in result.stdout

    def test_show_command(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mdpp.scripts._cli", "show", "gromacs/analysis/gmx_rmsd.sh"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "#!/" in result.stdout

    def test_copy_command(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mdpp.scripts._cli",
                "copy",
                "gromacs/runtime",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "restart.sh" in result.stdout

    def test_mdps_command(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mdpp.scripts._cli", "mdps", str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "step5_production.mdp" in result.stdout
