import pytest
import app
import numpy as np
from skimage import io
from io import BytesIO
import importlib

@pytest.fixture
def client():
    importlib.reload(app)
    with app.app.test_client() as client:
        yield client

def test_import_ome(client):
    form_data = {
        "filepath": '../testimages/2048x2048_ome6_tiled.ome.tif',
        "csvpath": "../testimages/markers.csv",
        "dataset": "test"
    }
    res = client.post('/api/import', content_type='application/x-www-form-urlencoded', data=form_data)
    assert res.status_code == 200

    _assert_tile_exists(client, 0, 1, 0, 0)
    _assert_tile_exists(client, 0, 0, 0, 0)
    _assert_tile_exists(client, 0, 0, 0, 1)
    _assert_tile_exists(client, 0, 0, 1, 0)
    _assert_tile_exists(client, 0, 0, 1, 1)

    _assert_tile_exists(client, 1, 1, 0, 0)
    _assert_tile_exists(client, 1, 0, 0, 0)
    _assert_tile_exists(client, 1, 0, 0, 1)
    _assert_tile_exists(client, 1, 0, 1, 0)
    _assert_tile_exists(client, 1, 0, 1, 1)

    _assert_tile_exists(client, 2, 1, 0, 0)
    _assert_tile_exists(client, 2, 0, 0, 0)
    _assert_tile_exists(client, 2, 0, 0, 1)
    _assert_tile_exists(client, 2, 0, 1, 0)
    _assert_tile_exists(client, 2, 0, 1, 1)

def test_import_svs(client):
    form_data = {
        "filepath": '../testimages/CMU-1-Small-Region.svs',
        "csvpath": "",
        "dataset": "test"
    }
    res = client.post('/api/import', content_type='application/x-www-form-urlencoded', data=form_data)
    assert res.status_code == 200

    _assert_rgb_tile_exists(client, 2, 0, 0, width=555, height=742)

    _assert_rgb_tile_exists(client, 1, 0, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, 1, 1, 0, width=86, height=1024)
    _assert_rgb_tile_exists(client, 1, 0, 1, width=1024, height=460)
    _assert_rgb_tile_exists(client, 1, 1, 1, width=86, height=460)

    _assert_rgb_tile_exists(client, 0, 0, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, 0, 1, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, 0, 0, 1, width=1024, height=1024)
    _assert_rgb_tile_exists(client, 0, 1, 1, width=1024, height=1024)
    _assert_rgb_tile_exists(client, 0, 2, 1, width=172, height=1024)
    _assert_rgb_tile_exists(client, 0, 1, 2, width=1024, height=919)
    _assert_rgb_tile_exists(client, 0, 2, 2, width=172, height=919)

def _assert_tile_exists(client, channel, level, x, y, width=1024, height=1024):
    res = client.get(f'/api/u16/{channel}/{level}_{x}_{y}.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == (height, width)
    assert img.dtype == np.uint16

def _assert_rgb_tile_exists(client, level, x, y, width=1024, height=1024):
    res = client.get(f'/api/u16/0/{level}_{x}_{y}.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == (height, width, 3)
    assert img.dtype == np.uint8
