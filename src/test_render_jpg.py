import logging
import os

from skimage import io

from app import Opener
from render_jpg import render_color_tiles

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_ome_tif_rendered_output():
    logger = logging.getLogger("app")
    filepath = "../testimages/2048x2048_ome6_tiled.ome.tif"
    group_dir = "group_label_0__DAPI--1__panCK--2__S100"
    opener = Opener(filepath)
    groups = [
        {
            "Group Path": group_dir,
            "Channel Number": ["0", "1", "2"],
            "High": [1, 2, 4],
            "Low": [0, 0, 0],
            "Color": [[1.0, 0, 0, 1.0], [0, 1.0, 0, 1.0], [0, 0, 1.0, 1.0]],
        }
    ]
    render_color_tiles(
        opener,
        output_dir="tmp",
        tile_size=1024,
        config_rows=groups,
        logger=logger,
        progress_callback=None,
    )

    filename = os.path.join("tmp", group_dir, "1_0_0.jpg")
    img = io.imread(filename)
    assert img.shape == (1024, 1024, 3)
    _assert_approx_color(img[200][200], [0, 128, 0])
    _assert_approx_color(img[200][500], [255, 128, 0])
    _assert_approx_color(img[200][800], [255, 0, 0])
    _assert_approx_color(img[550][350], [0, 128, 64])
    _assert_approx_color(img[450][500], [255, 128, 64])
    _assert_approx_color(img[550][700], [255, 0, 64])
    _assert_approx_color(img[800][500], [0, 0, 64])

    filename = os.path.join("tmp", group_dir, "0_0_0.jpg")
    img = io.imread(filename)
    assert img.shape == (1024, 1024, 3)
    _assert_approx_color(img[700][700], [0, 128, 0])

    filename = os.path.join("tmp", group_dir, "0_0_1.jpg")
    img = io.imread(filename)
    assert img.shape == (1024, 1024, 3)
    _assert_approx_color(img[300][700], [0, 0, 64])

    filename = os.path.join("tmp", group_dir, "0_1_0.jpg")
    img = io.imread(filename)
    assert img.shape == (1024, 1024, 3)
    _assert_approx_color(img[800][400], [255, 0, 0])

    filename = os.path.join("tmp", group_dir, "0_1_1.jpg")
    img = io.imread(filename)
    assert img.shape == (1024, 1024, 3)
    _assert_approx_color(img[300][200], [0, 0, 64])


def _assert_approx_color(expected, actual, margin=1):
    """
    Compares two colors, a margin of error is allowed.
    Jpeg compression will result in approximate colors, so we can't make exact assertions
    """
    assert expected[0] - margin <= actual[0] <= expected[0] + margin
    assert expected[1] - margin <= actual[1] <= expected[1] + margin
    assert expected[2] - margin <= actual[2] <= expected[2] + margin
