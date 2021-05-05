import pytest
import app
import uuid
import json
import pathlib
import numpy as np
from urllib.parse import quote
from skimage import io
from io import BytesIO
import importlib


def escape_url(url):
    return quote(quote(url, safe=''), safe='')


@pytest.fixture
def client():
    importlib.reload(app)
    with app.app.test_client() as client:
        yield client


def test_make_exhibit_config():

    ome_tif_in = '../testimages/2048x2048_ome6_tiled.ome.tif'
    for i in [0, 1, 2]:
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


def test_ome_preview(client):
    out_name = "output"
    group_name = "group name"
    channel_name = "channel name"
    test_header = "test header"
    test_description = "test description"
    test_channels = [{
        "color": "FFFFFF",
        "label": channel_name,
        "min": 0,
        "max": 1,
        "id": 0
    }]
    test_groups = [{
        "channels": test_channels,
        "render": test_channels,
        "label": group_name
    }]
    group_path = app.make_group_path(test_groups, test_groups[0])

    session = uuid.uuid4().hex

    filepath = '../testimages/2048x2048_ome6_tiled.ome.tif'
    post_data = json.dumps({
        "in_file": str(pathlib.Path(filepath).resolve()),
        "out_name": out_name,
        "groups": test_groups,
        "image": {
            "description": test_description
        },
        "header": test_header,
        "rotation": 0,
        "waypoints": [],
        "masks": [],
    })

    client.post(f'/api/preview/{session}', content_type='application/json', data=post_data)
    _assert_cache_keys(session, {
        'index.html', 'exhibit.json',
        f'images/{out_name}/{group_path}/0_0_0.jpg',
        f'images/{out_name}/{group_path}/0_0_1.jpg',
        f'images/{out_name}/{group_path}/0_1_0.jpg',
        f'images/{out_name}/{group_path}/0_1_1.jpg',
        f'images/{out_name}/{group_path}/1_0_0.jpg'
    })


def test_ome_tiles(client):
    path = '../testimages/2048x2048_ome6_tiled.ome.tif'

    _assert_tile_exists(client, path, 0, 1, 0, 0)
    _assert_tile_exists(client, path, 0, 0, 0, 0)
    _assert_tile_exists(client, path, 0, 0, 0, 1)
    _assert_tile_exists(client, path, 0, 0, 1, 0)
    _assert_tile_exists(client, path, 0, 0, 1, 1)

    _assert_tile_exists(client, path, 1, 1, 0, 0)
    _assert_tile_exists(client, path, 1, 0, 0, 0)
    _assert_tile_exists(client, path, 1, 0, 0, 1)
    _assert_tile_exists(client, path, 1, 0, 1, 0)
    _assert_tile_exists(client, path, 1, 0, 1, 1)

    _assert_tile_exists(client, path, 2, 1, 0, 0)
    _assert_tile_exists(client, path, 2, 0, 0, 0)
    _assert_tile_exists(client, path, 2, 0, 0, 1)
    _assert_tile_exists(client, path, 2, 0, 1, 0)
    _assert_tile_exists(client, path, 2, 0, 1, 1)


def test_import_svs(client):
    form_data = {
        "csvpath": "",
        "dataset": "test",
        "filepath": '../testimages/CMU-1-Small-Region.svs'
    }
    res = client.post('/api/import', content_type='application/x-www-form-urlencoded', data=form_data)
    assert res.status_code == 200


def test_svs_tiles(client):
    path = '../testimages/CMU-1-Small-Region.svs'

    _assert_rgb_tile_exists(client, path, 2, 0, 0, width=555, height=742)

    _assert_rgb_tile_exists(client, path, 1, 0, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, path, 1, 1, 0, width=86, height=1024)
    _assert_rgb_tile_exists(client, path, 1, 0, 1, width=1024, height=460)
    _assert_rgb_tile_exists(client, path, 1, 1, 1, width=86, height=460)

    _assert_rgb_tile_exists(client, path, 0, 0, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, path, 0, 1, 0, width=1024, height=1024)
    _assert_rgb_tile_exists(client, path, 0, 0, 1, width=1024, height=1024)
    _assert_rgb_tile_exists(client, path, 0, 1, 1, width=1024, height=1024)
    _assert_rgb_tile_exists(client, path, 0, 2, 1, width=172, height=1024)
    _assert_rgb_tile_exists(client, path, 0, 1, 2, width=1024, height=919)
    _assert_rgb_tile_exists(client, path, 0, 2, 2, width=172, height=919)


def _assert_cache_keys(session, key_set):
    from app import G
    cache_dict_keys = set(G['preview_cache'][session].keys())
    cache_diff = cache_dict_keys.symmetric_difference(key_set)
    assert len(cache_diff) == 0


def _assert_tile_exists(client, path, channel, level, x, y, width=1024, height=1024):
    key = escape_url(str(pathlib.Path(path).resolve()))
    res = client.get(f'/api/u16/{key}/{channel}/{level}_{x}_{y}.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == (height, width)
    assert img.dtype == np.uint16


def _assert_rgb_tile_exists(client, path, level, x, y, width=1024, height=1024):
    key = escape_url(str(pathlib.Path(path).resolve()))
    res = client.get(f'/api/u16/{key}/0/{level}_{x}_{y}.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == (height, width, 3)
    assert img.dtype == np.uint8
