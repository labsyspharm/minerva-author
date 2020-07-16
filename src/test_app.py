import pytest
from app import app
import numpy as np
from skimage import io
from io import BytesIO

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_raw_render(client):
    IMAGE_SIZE = (64, 64)

    form_data = {
        "filepath": "../testimages/64x64.ome.tif",
        "csvpath": "../testimages/markers.csv",
        "dataset": "test"
    }
    res = client.post('/api/import', content_type='application/x-www-form-urlencoded', data=form_data)
    assert res.status_code == 200

    res = client.get('/api/u16/0/0_0_0.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == IMAGE_SIZE
    assert img.dtype == np.uint16

    res = client.get('/api/u16/1/0_0_0.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == IMAGE_SIZE
    assert img.dtype == np.uint16

    res = client.get('/api/u16/2/0_0_0.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == IMAGE_SIZE
    assert img.dtype == np.uint16

    res = client.get('/api/u16/3/0_0_0.png')
    assert res.status_code == 200
    img = io.imread(BytesIO(res.data))
    assert img.shape == IMAGE_SIZE
    assert img.dtype == np.uint16

