import pytest
import app
import json
import numpy as np
from skimage import io
from io import BytesIO
import importlib

@pytest.fixture
def client():
    importlib.reload(app)
    with app.app.test_client() as client:
        yield client

def test_make_exhibit_config():

    ome_tif_in = '../testimages/2048x2048_ome6_tiled.ome.tif'
    for i in [0, 1]:
        exhibit_in = f'../testimages/exhibit{i}_in.json'
        exhibit_out = f'../testimages/exhibit{i}_out.json'
        exhibit_name = 'test'

        with open(exhibit_in) as f:
            data_in = json.load(f)
        with open(exhibit_out) as f:
            data_out = json.load(f)

        opener = app.Opener(ome_tif_in)

        exhibit_config = app.make_exhibit_config(opener, exhibit_name, data_in)
        assert exhibit_config['Groups'] == data_out['Groups']
        assert exhibit_config['Header'] == data_out['Header']
        assert exhibit_config['Images'] == data_out['Images']
        assert exhibit_config['Layout'] == data_out['Layout']
        assert exhibit_config['Rotation'] == data_out['Rotation']
        assert exhibit_config['Stories'] == data_out['Stories']
        assert exhibit_config['Masks'] == data_out['Masks']

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

    _assert_rgb_tile_exists(client, 2, 0, 0, width=555, height=741)

    _assert_rgb_tile_exists(client, 1, 0, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, 1, 1, 0, width=86, height=1024)
    _assert_rgb_tile_exists(client, 1, 0, 1, width=1024, height=459)
    _assert_rgb_tile_exists(client, 1, 1, 1, width=86, height=459)

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
