import logging
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
from skimage import io

from app import Opener
from render_jpg import render_color_tiles

testimages_path = Path(__file__).parent.parent / "testimages"


def test_ome_tif_rendered_output():
    logger = logging.getLogger("app")
    filepath = testimages_path / "2048x2048_ome6_tiled.ome.tif"
    group_dir = "group_label_0__DAPI--1__panCK--2__S100"
    opener = Opener(str(filepath))
    groups = [
        {
            "Group Path": group_dir,
            "Channel Number": ["0", "1", "2"],
            "High": [1, 2, 4],
            "Low": [0, 0, 0],
            "Color": [[1.0, 0, 0, 1.0], [0, 1.0, 0, 1.0], [0, 0, 1.0, 1.0]],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        render_color_tiles(
            opener,
            output_dir=tmpdir,
            tile_size=1024,
            config_rows=groups,
            logger=logger,
            progress_callback=None,
        )

        gdir = Path(tmpdir) / group_dir

        filename = gdir / "1_0_0.jpg"
        img = io.imread(filename)
        assert img.shape == (1024, 1024, 3)
        _assert_approx_color(img[200][200], [0, 128, 0])
        _assert_approx_color(img[200][500], [255, 128, 0])
        _assert_approx_color(img[200][800], [255, 0, 0])
        _assert_approx_color(img[550][350], [0, 128, 64])
        _assert_approx_color(img[450][500], [255, 128, 64])
        _assert_approx_color(img[550][700], [255, 0, 64])
        _assert_approx_color(img[800][500], [0, 0, 64])

        filename = gdir / "0_0_0.jpg"
        img = io.imread(filename)
        assert img.shape == (1024, 1024, 3)
        _assert_approx_color(img[700][700], [0, 128, 0])

        filename = gdir / "0_0_1.jpg"
        img = io.imread(filename)
        assert img.shape == (1024, 1024, 3)
        _assert_approx_color(img[300][700], [0, 0, 64])

        filename = gdir / "0_1_0.jpg"
        img = io.imread(filename)
        assert img.shape == (1024, 1024, 3)
        _assert_approx_color(img[800][400], [255, 0, 0])

        filename = gdir / "0_1_1.jpg"
        img = io.imread(filename)
        assert img.shape == (1024, 1024, 3)
        _assert_approx_color(img[300][200], [0, 0, 64])


def test_ome_tif_rendered_output_2():
    logger = logging.getLogger("app")
    filepath = testimages_path / "64x64.ome.tif"
    group_dir = "group_label_0__DAPI--1__panCK"
    opener = Opener(str(filepath))
    groups = [
        {
            "Group Path": group_dir,
            "Channel Number": ["0", "1"],
            "High": [1, 1],
            "Low": [0, 0],
            "Color": [[1.0, 0, 0, 1.0], [0.1, 1.0, 0, 1.0]],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        render_color_tiles(
            opener,
            output_dir=tmpdir,
            tile_size=64,
            config_rows=groups,
            logger=logger,
            progress_callback=None,
        )

        filename = Path(tmpdir) / group_dir / "0_0_0.jpg"
        img = io.imread(filename)
        _assert_approx_color(img[0][0], [255, 0, 0])
        _assert_approx_color(img[0][63], [26, 255, 0])
        _assert_approx_color(img[0][29], [255, 255, 0], 2)


def _assert_approx_color(actual, expected, tolerance=1):
    """
    Compares two colors, a margin of error is allowed.
    Jpeg compression will result in approximate colors, so we can't make exact assertions
    """
    assert actual == pytest.approx(expected, abs=tolerance)
