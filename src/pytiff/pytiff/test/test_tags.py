from hypothesis import given, settings
import hypothesis.strategies as st
from pytiff import Tiff
import pytiff
import tifffile
import os
import numpy as np
import sys
import logging

def test_tags(tmpdir_factory):
    testfile = "test_data/small_example_tiled.tif"
    basename = os.path.basename(testfile)

    with Tiff(testfile) as handle:
        tags = handle.read_tags()

    with tifffile.TiffFile(testfile) as handle:
        page1 = handle.pages[0]
        tifffile_tags = {}
        for tag in page1.tags.values():
            name, value = tag.name, tag.value
            tifffile_tags[name] = value
        image = page1.asarray()

    compare_tags(tags, tifffile_tags)

    filename = str(tmpdir_factory.mktemp("write_tags").join(basename))
    with Tiff(filename, "w") as handle:
        handle.set_tags(tags)
        handle.write(image, method="tile", tile_length=tags[pytiff.tags.tile_length], tile_width=tags[pytiff.tags.tile_width])

    with Tiff(filename) as handle:
        written_tags = handle.read_tags()
        written_image = handle[:]

    assert np.all(written_image == image)
    check_written_tags(written_tags, tags)

def test_tags_strips():
    testfile = "test_data/small_example.tif"
    basename = os.path.basename(testfile)
    with Tiff(testfile) as handle:
        tags = handle.read_tags()

    with tifffile.TiffFile(testfile) as handle:
        page1 = handle.pages[0]
        tifffile_tags = {}
        for tag in page1.tags.values():
            name, value = tag.name, tag.value
            tifffile_tags[name] = value
        image = page1.asarray()

    compare_tags(tags, tifffile_tags)

def test_tags_rbg():
    testfile = "test_data/rgb_sample.tif"
    basename = os.path.basename(testfile)
    with Tiff(testfile) as handle:
        tags = handle.read_tags()

    with tifffile.TiffFile(testfile) as handle:
        page1 = handle.pages[0]
        tifffile_tags = {}
        for tag in page1.tags.values():
            name, value = tag.name, tag.value
            tifffile_tags[name] = value
        image = page1.asarray()

    compare_tags(tags, tifffile_tags)

def test_tags_rbg_tiled(tmpdir_factory):
    testfile = "test_data/tiled_rgb_sample.tif"
    basename = os.path.basename(testfile)
    with Tiff(testfile) as handle:
        tags = handle.read_tags()

    with tifffile.TiffFile(testfile) as handle:
        page1 = handle.pages[0]
        tifffile_tags = {}
        for tag in page1.tags.values():
            name, value = tag.name, tag.value
            tifffile_tags[name] = value
        image = page1.asarray()

    compare_tags(tags, tifffile_tags)

    filename = str(tmpdir_factory.mktemp("write_tags").join(basename))
    with Tiff(filename, "w") as handle:
        handle.set_tags(tags)
        handle.write(image, method="tile", tile_length=tags[pytiff.tags.tile_length], tile_width=tags[pytiff.tags.tile_width])

    with Tiff(filename) as handle:
        written_tags = handle.read_tags()
        written_image = handle[:]

    assert np.all(written_image == image)
    check_written_tags(written_tags, tags)

def test_tags_bigtif(tmpdir_factory):
    testfile = "test_data/bigtif_example_tiled.tif"
    basename = os.path.basename(testfile)

    with Tiff(testfile) as handle:
        tags = handle.read_tags()

    with tifffile.TiffFile(testfile) as handle:
        page1 = handle.pages[0]
        tifffile_tags = {}
        for tag in page1.tags.values():
            name, value = tag.name, tag.value
            tifffile_tags[name] = value
        image = page1.asarray()

    compare_tags(tags, tifffile_tags)

    filename = str(tmpdir_factory.mktemp("write_tags").join(basename))
    with Tiff(filename, "w") as handle:
        handle.set_tags(tags)
        handle.write(image, method="tile", tile_length=tags[pytiff.tags.tile_length], tile_width=tags[pytiff.tags.tile_width])

    with Tiff(filename) as handle:
        written_tags = handle.read_tags()
        written_image = handle[:]

    assert np.all(written_image == image)
    check_written_tags(written_tags, tags)

def check_written_tags(written_tags, tags):
    for k in written_tags:
        if "_offsets" in k.name:
            o1 = written_tags[k] - written_tags[k][0]
            o2 = tags[k] - tags[k][0]
            assert np.all(o1 == o2)
            continue
        # skip sample_format if set to default value
        if k.name == "sample_format":
            if written_tags[k] == 1:
                continue
        if isinstance(written_tags[k], np.ndarray):
            assert np.all(written_tags[k] == tags[k]), "*** failed for {} ***".format(k.name)
        else:
            assert written_tags[k] == tags[k]

def compare_tags(pytiff_tags, tifffile_tags):
    assert len(pytiff_tags) == len(tifffile_tags), "missing keys: {}".format(set([t.name for t in pytiff_tags]) - set(tifffile_tags.keys()))
    for name in tifffile_tags:
        value = tifffile_tags[name]
        if isinstance(value, bytes):
            value = value.decode()
            assert value == pytiff_tags[pytiff.tags[name]], "key {}: {} == {}".format(name, value, pytiff_tags[pytiff.tags[name]])
        elif name == "x_resolution" or name == "y_resolution":
            assert value[0] / value[1] == pytiff_tags[pytiff.tags[name]]
        else:
            value = np.array([value])
            value.squeeze()
            assert np.all(value == pytiff_tags[pytiff.tags[name]]), "key {}: {} == {}".format(name, value, pytiff_tags[pytiff.tags[name]])


