import os
from render_jpg import render_color_tiles
from app import Opener
from skimage import io

def test_ome_tif_rendered_output():
    opener = Opener('../testimages/64x64.ome.tif')

    groups = [
        {
            'Group': 'group_label',
            'Marker Name': 'DAPI',
            'Channel Number': '0',
            'Low': 0,
            'High': 65535,
            'Color': [1.0, 0, 0, 1.0],
        },
        {
            'Group': 'group_label',
            'Marker Name': 'panCK',
            'Channel Number': '1',
            'Low': 0,
            'High': 65535,
            'Color': [0, 1.0, 0, 1.0],
        },
        {
            'Group': 'group_label',
            'Marker Name': 'S100',
            'Channel Number': '2',
            'Low': 0,
            'High': 65535,
            'Color': [0, 0, 1.0, 1.0],
        }
    ]
    render_color_tiles(opener, output_dir="tmp", tile_size=1024, num_channels=1, config_rows=groups, progress_callback=None)
    filename = os.path.join("tmp", "group_label_0__DAPI--1__panCK--2__S100", "0_0_0.jpg")
    img = io.imread(filename)
    assert img.shape == (64, 64, 3)
    # Jpeg compression will result in approximate colors, so we can't make exact assertions
    _assert_approx_color(img[0][0], [255, 0, 0])
    _assert_approx_color(img[63][0], [0, 0, 255])
    _assert_approx_color(img[0][63], [0, 255, 0])
    _assert_approx_color(img[63][63], [0, 0, 0])
    _assert_approx_color(img[32][0], [255, 0, 255])
    _assert_approx_color(img[0][32], [255, 255, 0])

def _assert_approx_color(actual, expected, margin=1):
    assert expected[0] - margin <= actual[0] <= expected[0] + margin
    assert expected[1] - margin <= actual[1] <= expected[1] + margin
    assert expected[2] - margin <= actual[2] <= expected[2] + margin