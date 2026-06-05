"""Tests for mdpp.core.parsers (XVG parsing)."""

from __future__ import annotations

import numpy as np

from mdpp.core.parsers import read_xvg


class TestReadXvgMetadata:
    def test_subtitle_does_not_overwrite_title(self, tmp_path):
        # A loose substring match would let the "@ subtitle" directive clobber
        # the title (both contain "title"); the tokenized parser must not.
        xvg = tmp_path / "rmsd.xvg"
        xvg.write_text(
            '@    title "Backbone RMSD"\n'
            '@    subtitle "after least-squares fit"\n'
            '@    xaxis  label "Time (ps)"\n'
            '@    yaxis  label "RMSD (nm)"\n'
            "0.0   0.10\n"
            "10.0  0.12\n"
        )
        df = read_xvg(xvg)
        assert df.attrs["title"] == "Backbone RMSD"
        assert df.attrs["xlabel"] == "Time (ps)"
        assert df.attrs["ylabel"] == "RMSD (nm)"

    def test_ticklabel_does_not_set_axis_label(self, tmp_path):
        # "@ xaxis ticklabel ..." must not be mistaken for "@ xaxis label ...".
        xvg = tmp_path / "t.xvg"
        xvg.write_text(
            "@    xaxis  ticklabel char size 1.0\n"
            '@    xaxis  label "Time (ps)"\n'
            "0.0   1.0\n"
            "1.0   2.0\n"
        )
        df = read_xvg(xvg)
        assert df.attrs["xlabel"] == "Time (ps)"


class TestReadXvgShape:
    def test_single_column_multi_row(self, tmp_path):
        # A single-column file must stay (n_rows, 1), not be flattened to (1, n).
        xvg = tmp_path / "single_col.xvg"
        xvg.write_text("1.0\n2.0\n3.0\n4.0\n")
        df = read_xvg(xvg)
        assert df.shape == (4, 1)
        np.testing.assert_allclose(df.iloc[:, 0].to_numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_single_row_multi_column(self, tmp_path):
        xvg = tmp_path / "single_row.xvg"
        xvg.write_text("0.0  1.0  2.0\n")
        df = read_xvg(xvg)
        assert df.shape == (1, 3)

    def test_multi_row_multi_column(self, tmp_path):
        xvg = tmp_path / "grid.xvg"
        xvg.write_text("0.0  1.0\n10.0  2.0\n20.0  3.0\n")
        df = read_xvg(xvg)
        assert df.shape == (3, 2)
