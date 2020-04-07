""" Render PNG tiles
"""
import itertools
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
from skimage.exposure import adjust_gamma
from PIL import Image
import numpy as np
import pytiff
import io

def render_tile(tiff, tile_size, num_channels, level, tx, ty, channel_number):
    iy = ty * tile_size
    ix = tx * tile_size
    page_base = level * num_channels

    tiff.set_page(page_base + channel_number)
    tile = tiff[iy:iy+tile_size, ix:ix+tile_size]
    # tile = adjust_gamma(tile, 1/2.2)

    array_buffer = tile.tobytes()
    img = Image.new("I", tile.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")

    img_io = io.BytesIO()
    img.save(img_io, 'PNG', compress_level=1)
    img_io.seek(0)
    return img_io


def render_tiles(input_file, output_dir, tile_size, num_channels):

    print('Processing:', str(input_file))

    output_path = pathlib.Path(output_dir)
    tiff = pytiff.Tiff(str(input_file), encoding='utf-8')

    assert tiff.number_of_pages % num_channels == 0, "Pyramid/channel mismatch"

    num_levels = tiff.number_of_pages // num_channels

    for level in range(num_levels):

        page_base = level * num_channels
        tiff.set_page(page_base)
        ny = int(np.ceil(tiff.shape[0] / tile_size))
        nx = int(np.ceil(tiff.shape[1] / tile_size))
        print('    level {} ({} x {})'.format(level, ny, nx))

        for ty, tx in itertools.product(range(0, ny), range(0, nx)):

            iy = ty * tile_size
            ix = tx * tile_size
            filename = '{}_{}_{}.png'.format(level, tx, ty)

            for channel_number in range(num_channels):

                group_dir = str(channel_number)
                if not (output_path / group_dir).exists():
                    (output_path / group_dir).mkdir(parents=True)

                tiff.set_page(page_base + channel_number)
                tile = tiff[iy:iy+tile_size, ix:ix+tile_size]
                # tile = adjust_gamma(tile, 1/2.2)

                array_buffer = tile.tobytes()
                img = Image.new("I", tile.T.shape)
                img.frombytes(array_buffer, 'raw', "I;16")

                img.save(str(output_path / group_dir / filename))

if __name__ == "__main__":

    TILE_SIZE = 1024
    NUM_CHANNELS = 24

    INPUT_FILE_DIR = '/Users/john/data/n/imstor/sorger/data/computation/Jeremy/ashlar_examples/'
    INPUT_FILE = sorted(pathlib.Path(INPUT_FILE_DIR).glob('*.ome.tif'))[0]

    render_tiles(INPUT_FILE, './output', TILE_SIZE, NUM_CHANNELS)
