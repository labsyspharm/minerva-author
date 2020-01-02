from hypothesis import HealthCheck
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp
from pytiff import *
import hypothesis.strategies as st
import numpy as np
import pytest
import subprocess
import tifffile
from skimage.data import coffee

def test_write_rgb(tmpdir_factory):
    img = coffee()
    filename = str(tmpdir_factory.mktemp("write").join("rgb_img.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(img, method="tile")
    with Tiff(filename) as handle:
        data = handle[:]
        assert np.all(img == data[:, :, :3])

    with Tiff(filename, "w") as handle:
        handle.write(img, method="scanline")
    with Tiff(filename) as handle:
        data = handle[:]
        assert np.all(img == data[:, :, :3])

# scanline integer tests

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=st.one_of(hnp.integer_dtypes(endianness="="), hnp.unsigned_integer_dtypes(endianness="=")),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50)))
def test_write_int_scanline(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("int_img.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(data, method="scanline")

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data, img)

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=st.one_of(hnp.integer_dtypes(endianness="="), hnp.unsigned_integer_dtypes(endianness="=")),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50)))
def test_write_int_scanline_set_rows_per_strip(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("int_img.tif"))
    rows_per_strip = 1
    with Tiff(filename, "w") as handle:
        handle.write(data, method="scanline", rows_per_strip=rows_per_strip)

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data, img)
        assert rows_per_strip == handle[0].tags["rows_per_strip"].value

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=st.one_of(hnp.integer_dtypes(endianness="="), hnp.unsigned_integer_dtypes(endianness="=")),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=20, max_side=20)))
def test_write_int_slices_scanline(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("int_img_scanline.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(data[:, :], method="scanline")

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data[:,:], img)

# tile integer tests

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=st.one_of(hnp.integer_dtypes(endianness="="), hnp.unsigned_integer_dtypes(endianness="=")),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50)))
def test_write_int_tile(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("int_tile_img.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(data, method="tile", tile_width=16, tile_length=16)

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data, img)

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=hnp.floating_dtypes(endianness="="),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50), elements=st.floats(0, 1)))
def test_write_float_scanline(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("float_img.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(data, method="scanline")

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data, img)

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=hnp.floating_dtypes(endianness="="),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50), elements=st.floats(0, 1)))
def test_write_float_tile(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("float_tile_img.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(data, method="tile", tile_length=16, tile_width=16)

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data, img)

@settings(buffer_size=11000000)
@given(data=hnp.arrays(dtype=st.one_of(hnp.integer_dtypes(endianness="="), hnp.unsigned_integer_dtypes(endianness="=")),
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=50)))
def test_append_int_tile(data, tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("append_img.tif"))
    with Tiff(filename, "w") as handle:
        handle.write(data, method="tile", tile_width=16, tile_length=16)

    with Tiff(filename, "a") as handle:
        handle.write(data, method="tile", tile_width=16, tile_length=16)

    with Tiff(filename, "r") as handle:
        assert handle.number_of_pages == 2

    with tifffile.TiffFile(filename) as handle:
        img = handle.asarray()
        np.testing.assert_array_equal(data, img[0])
        np.testing.assert_array_equal(data, img[1])

def test_write_chunk(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("chunk_img.tif"))
    filename = "test_chunk.tif"
    data1 = np.ones((64,64), dtype=np.uint8) * 1
    data2 = np.ones((64,64), dtype=np.uint8) * 2
    data3 = np.ones((64,64), dtype=np.uint8) * 3
    data4 = np.ones((64,64), dtype=np.uint8) * 4
    with Tiff(filename, "w") as handle:
        chunks = [data1, data2, data3, data4]
        handle.new_page((300, 300), dtype=np.uint8, tile_length=16, tile_width=16)
        row = 0
        col = 0
        max_row_end = 0
        positions = []
        for c in chunks:
            shape = c.shape
            row_end, col_end = row + shape[0], col + shape[1]
            max_row_end = max(max_row_end, row_end)
            handle[row:row_end, col:col_end] = c
            # save for reading chunks
            positions.append([row, row_end, col, col_end])
            if col_end >= handle.shape[1]:
                col = 0
                row = max_row_end
            else:
                col = col_end

        handle.save_page()

    with Tiff(filename) as handle:
        for pos, chunk in zip(positions, chunks):
            row, row_end, col, col_end = pos
            data = handle[row:row_end, col:col_end]
            assert np.all(data == chunk)

    with Tiff(filename) as handle:
        with pytest.raises(ValueError):
            handle.new_page((50, 50), np.dtype("uint8"))
            handle[:, :] = np.random.rand(50, 50)
            handle.save_page()

def test_write_chunk_multiple_pages(tmpdir_factory):
    filename = str(tmpdir_factory.mktemp("write").join("multi_page_chunk_img.tif"))
    data1 = np.ones((64,64), dtype=np.uint8) * 1
    data2 = np.ones((64,64), dtype=np.uint8) * 2
    data3 = np.ones((64,64), dtype=np.uint8) * 3
    data4 = np.ones((64,64), dtype=np.uint8) * 4
    with Tiff(filename, "w")as handle:
        chunks = [data1, data2, data3, data4]

        for c in chunks:
            shape = c.shape
            handle.new_page(shape, dtype=np.uint8, tile_length=16, tile_width=16)
            handle[:] = c

    with Tiff(filename) as handle:
        for page, chunk in enumerate(chunks):
            handle.set_page(page)
            data = handle[:]
            assert data.shape == chunk.shape
            assert np.all(data == chunk)

