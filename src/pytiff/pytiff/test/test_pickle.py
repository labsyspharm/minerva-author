from pytiff import Tiff
import pytest
import pickle
import sys
if sys.version_info.major > 2:
    import pickle as cPickle
else:
    import cPickle
import numpy as np

TILED_GREY = "test_data/small_example_tiled.tif"
MULTI_PAGE = "test_data/multi_page.tif"

def test_pickle():
    t = Tiff(TILED_GREY)
    data = t[:]
    t.close()
    saved = pickle.dumps(t)
    loaded = pickle.loads(saved)
    loaded_data = loaded[:]
    loaded.close()
    np.testing.assert_array_equal(data, loaded_data)

def test_pickle_multipage():
    t = Tiff(MULTI_PAGE)
    t.set_page(2)
    data = t[:]
    t.close()
    saved = pickle.dumps(t)
    loaded = pickle.loads(saved)
    loaded_data = loaded[:]
    loaded.close()
    np.testing.assert_array_equal(data, loaded_data)

def test_cpickle():
    t = Tiff(TILED_GREY)
    data = t[:]
    t.close()
    saved = cPickle.dumps(t)
    loaded = cPickle.loads(saved)
    loaded_data = loaded[:]
    loaded.close()
    np.testing.assert_array_equal(data, loaded_data)

def test_cpickle_multipage():
    t = Tiff(MULTI_PAGE)
    t.set_page(2)
    data = t[:]
    t.close()
    saved = cPickle.dumps(t)
    loaded = cPickle.loads(saved)
    loaded_data = loaded[:]
    loaded.close()
    np.testing.assert_array_equal(data, loaded_data)
