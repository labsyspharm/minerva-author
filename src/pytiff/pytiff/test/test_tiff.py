from hypothesis import HealthCheck
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from pytiff import *
import hypothesis.strategies as st
import numpy as np
import pytest
import subprocess
import tifffile

TILED_GREY = "test_data/small_example_tiled.tif"
NOT_TILED_GREY = "test_data/small_example.tif"
TILED_RGB = "test_data/tiled_rgb_sample.tif"
NOT_TILED_RGB = "test_data/rgb_sample.tif"
TILED_BIG = "test_data/bigtif_example_tiled.tif"
NOT_TILED_BIG = "test_data/bigtif_example.tif"
NO_FILE = "test_data/not_here.tif"

def test_open():
    tif = Tiff(TILED_GREY)
    assert True

def test_open_fail():
    with pytest.raises(IOError):
        tif = Tiff(NO_FILE)

def test_greyscale_tiled():
    with Tiff(TILED_GREY) as tif:
        assert tif.is_tiled()
    read_methods(TILED_GREY)

def test_large_slice():
    with Tiff(TILED_GREY) as tif:
        assert tif.is_tiled()
        data = tif[:]

    with Tiff(TILED_GREY) as tif:
        shape = tif.shape
        large = tif[0:shape[0]+100, 0:shape[1]+33]

    assert np.all(data == large)

def test_greyscale_not_tiled():
    with Tiff(NOT_TILED_GREY) as tif:
        assert not tif.is_tiled()
    read_methods(NOT_TILED_GREY)

def test_rgb_tiled():
    with tifffile.TiffFile(TILED_RGB) as tif:
        for page in tif:
            first_page = page.asarray()
            break

    with Tiff(TILED_RGB) as tif:
        assert tif.is_tiled()
        data = tif[:]
        # test reading whole page
        np.testing.assert_array_equal(first_page, data)
        # test reading a chunk
        chunk = tif[100:200, :, :3]
        np.testing.assert_array_equal(first_page[100:200, :], chunk)

        chunk = tif[:, 250:350]
        np.testing.assert_array_equal(first_page[:, 250:350], chunk)

        chunk = tif[100:200, 250:350]
        np.testing.assert_array_equal(first_page[100:200, 250:350], chunk)

def test_rgb_not_tiled():
    with tifffile.TiffFile(NOT_TILED_RGB) as tif:
        for page in tif:
            first_page = page.asarray()
            break

    with Tiff(NOT_TILED_RGB) as tif:
        assert not tif.is_tiled()
        data = tif[:]
        # test reading whole page
        np.testing.assert_array_equal(first_page, data)
        # test reading a chunk
        chunk = tif[100:200, :]
        np.testing.assert_array_equal(first_page[100:200, :], chunk)

        chunk = tif[:, 250:350]
        np.testing.assert_array_equal(first_page[:, 250:350], chunk)

        chunk = tif[100:200, 250:350]
        np.testing.assert_array_equal(first_page[100:200, 250:350], chunk)

def test_big_tiled():
    with Tiff(TILED_BIG) as tif:
        assert tif.is_tiled()
    read_methods(TILED_BIG)

def test_big_not_tiled():
    with Tiff(NOT_TILED_BIG) as tif:
        assert not tif.is_tiled()
    read_methods(NOT_TILED_BIG)

def test_to_array():
    import numpy as np
    with Tiff(TILED_GREY) as tif:
        data = np.array(tif)
    with Tiff(TILED_GREY) as t:
        assert np.all(data == t[:])

MULTI_PAGE = "test_data/multi_page.tif"
N_PAGES = 4
SIZE = [(500, 500), (500, 500), (400, 640),(500, 500)]
MODE = ["greyscale", "greyscale", "rgb","greyscale"]
TYPE = [np.uint8, np.uint8, np.uint8, np.uint16]

def test_multi_page():
    with Tiff(MULTI_PAGE) as tif:
        assert tif.number_of_pages == N_PAGES
        assert tif.current_page == 0
        tif.set_page(2)
        assert tif.current_page == 2
        tif.set_page(N_PAGES + 1)
        assert tif.current_page == 3
        for i in range(N_PAGES):
            tif.set_page(i)
            assert tif.size[:2] == SIZE[i]
            assert tif.mode == MODE[i]
            assert tif.dtype == TYPE[i]

def test_multi_page_generator():
    with Tiff(MULTI_PAGE) as tif:
        for i, page in enumerate(tif.pages):
            assert i == page.current_page
            assert page.size[:2] == SIZE[i]
            assert page.mode == MODE[i]
            assert page.dtype == TYPE[i]
        with pytest.raises(SinglePageError):
            page.set_page(0)

def test_shape():
    with Tiff(TILED_GREY) as tif:
        assert len(tif.shape) == 2
    with Tiff(TILED_RGB) as tif:
        assert len(tif.shape) == 3

def read_methods(filename):
    with tifffile.TiffFile(filename) as tif:
        for page in tif:
            first_page = page.asarray()
            break

    with Tiff(filename) as tif:
        data = tif[:]
        # test reading whole page
        np.testing.assert_array_equal(first_page, data)
        # test reading a chunk
        chunk = tif[100:200, :, :3]
        np.testing.assert_array_equal(first_page[100:200, :], chunk)

        chunk = tif[:, 250:350]
        np.testing.assert_array_equal(first_page[:, 250:350], chunk)

        chunk = tif[100:200, 250:350]
        np.testing.assert_array_equal(first_page[100:200, 250:350], chunk)

