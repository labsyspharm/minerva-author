from __future__ import print_function, division
import sys
import itertools
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
from skimage.exposure import adjust_gamma
import json
import numpy as np
import pytiff
from PIL import Image
from matplotlib import colors
import csv
import json
import yaml
from collections import OrderedDict

def composite_channel(target, image, color, range_min, range_max):
    ''' Render _image_ in pseudocolor and composite into _target_
    Args:
        target: Numpy float32 array containing composition target image
        image: Numpy uint16 array of image to render and composite
        color: Color as r, g, b float array, 0-1
        range_min: Threshhold range minimum, 0-65535
        range_max: Threshhold range maximum, 0-65535
    '''
    f_image = (image.astype('float32') - range_min) / (range_max - range_min)
    f_image = f_image.clip(0,1, out=f_image)
    for i, component in enumerate(color):
        target[:, :, i] += f_image * component


def _calculate_total_tiles(tiff, tile_size, num_levels, num_channels):
    tiles = 0
    for level in range(num_levels):
        page_base = level * num_channels
        tiff.set_page(page_base)
        ny = int(np.ceil(tiff.shape[0] / tile_size))
        nx = int(np.ceil(tiff.shape[1] / tile_size))
        tiles += (nx * ny)
    return tiles

def render_color_tiles(input_file, output_dir, tile_size, num_channels, config_rows, progress_callback=None):

    EXT = 'jpg'

    print('Processing:', str(input_file))

    output_path = pathlib.Path(output_dir)
    tiff = pytiff.Tiff(str(input_file), encoding='utf-8')

    assert tiff.number_of_pages % num_channels == 0, "Pyramid/channel mismatch"

    num_levels = tiff.number_of_pages // num_channels

    render_groups = OrderedDict()
    for i in config_rows:
        if i['Group'] not in render_groups.keys():
            render_groups[i['Group']] = {
                'Channel Number': [],
                'Low': [],
                'High': [],
                'Color': [],
                'Marker Name': []
            }
        for key in render_groups[i['Group']]:
            render_groups[i['Group']][key].append(i[key])

    total_tiles = _calculate_total_tiles(tiff, tile_size, num_levels, num_channels)
    progress = 0

    for level in range(num_levels):

        page_base = level * num_channels
        tiff.set_page(page_base)
        ny = int(np.ceil(tiff.shape[0] / tile_size))
        nx = int(np.ceil(tiff.shape[1] / tile_size))
        print('    level {} ({} x {})'.format(level, ny, nx))

        for ty, tx in itertools.product(range(0, ny), range(0, nx)):

            iy = ty * tile_size
            ix = tx * tile_size
            filename = '{}_{}_{}.{}'.format(level, tx, ty, EXT)

            for name, settings in render_groups.items():

                channels = '--'.join(
                    settings['Channel Number'][i] + '__' + settings['Marker Name'][i]
                    for i in range(len(settings['Color']))
                )

                group_dir = name.replace(' ', '-') + '_' + channels
                if not (output_path / group_dir).exists():
                    (output_path / group_dir).mkdir(parents=True)
                for i, (marker, color, start, end) in enumerate(zip(
                    settings['Channel Number'], settings['Color'],
                    settings['Low'], settings['High']
                )):
                    tiff.set_page(page_base + int(marker))
                    tile = tiff[iy:iy+tile_size, ix:ix+tile_size]
                    tile = adjust_gamma(tile, 1/2.2)
                    if i == 0:
                        target = np.zeros(tile.shape + (3,), np.float32)
                    composite_channel(
                        target, tile, colors.to_rgb(color), float(start), float(end)
                    )
                np.clip(target, 0, 1, out=target)
                target_u8 = (target * 255).astype(np.uint8)
                img = Image.frombytes('RGB', target.T.shape[1:], target_u8.tobytes())
                img.save(str(output_path / group_dir / filename), quality=85)

                progress += 1
                if progress_callback is not None:
                    progress_callback(progress, total_tiles)

