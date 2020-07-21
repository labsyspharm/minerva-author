import pytest
from app import app
import numpy as np
from skimage import io
from io import BytesIO

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

TILE_SIZE = (1024, 1024)

def test_raw_render(client):

    form_data = {
        "filepath": '../testimages/2048x2048_ome6_tiled.ome.tif',
        "csvpath": "../testimages/markers.csv",
        "dataset": "test"
    }
    res = client.post('/api/import', content_type='application/x-www-form-urlencoded', data=form_data)
    assert res.status_code == 200

    _assert_tile_exists(client, 1, 0, 0)
    _assert_tile_exists(client, 0, 0, 0)
    _assert_tile_exists(client, 0, 0, 1)
    _assert_tile_exists(client, 0, 1, 0)
    _assert_tile_exists(client, 0, 1, 1)

def _assert_tile_exists(client, level, x, y):
    res = client.get(f'/api/u16/2/{level}_{x}_{y}.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == TILE_SIZE
    assert img.dtype == np.uint16
