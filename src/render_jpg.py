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
from PIL import Image
from matplotlib import colors
import csv
import json
import yaml
import time
import os

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


def _calculate_total_tiles(opener, tile_size, num_levels, num_channels):
    tiles = 0
    for level in range(num_levels):
        (nx, ny) = opener.get_level_tiles(level, tile_size)
        tiles += nx * ny

    return tiles

def render_color_tiles(opener, output_dir, tile_size, num_channels, config_rows, logger, progress_callback=None):
    EXT = 'jpg'

    for settings in config_rows:
        settings['Source'] = opener.path

    print('Processing:', str(opener.path))

    output_path = pathlib.Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    config_path = output_path / 'config.json'
    old_rows = []

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                old_rows = json.load(f)
            except json.decoder.JSONDecodeError as err:
                print(err)

    with open(config_path, 'w') as f:
        json.dump(config_rows, f)

    if opener.reader == 'pytiff':
        assert opener.io.number_of_pages % num_channels == 0, "Pyramid/channel mismatch"

    num_levels = opener.get_shape()[1]

    total_tiles = _calculate_total_tiles(opener, tile_size, num_levels, num_channels)
    progress = 0

    if num_levels < 2:
        logger.warning(f'Number of levels {num_levels} < 2')

    for level in range(num_levels):

        (nx, ny) = opener.get_level_tiles(level, tile_size)
        print('    level {} ({} x {})'.format(level, ny, nx))

        for ty, tx in itertools.product(range(0, ny), range(0, nx)):

            filename = '{}_{}_{}.{}'.format(level, tx, ty, EXT)

            for settings in config_rows:

                group_dir = settings['Group Path'] 
                if not (output_path / group_dir).exists():
                    (output_path / group_dir).mkdir(parents=True)
                output_file = str(output_path / group_dir / filename)

                # Only save file if change in config rows
                if not (os.path.exists(output_file) and config_rows == old_rows):
                    try:
                        opener.save_tile(output_file, settings, tile_size, level, tx, ty)
                    except AttributeError as e:
                        logger.error(f'{level} ty {ty} tx {tx}: {e}')
                else:
                    logger.warning(f'Not saving tile level {level} ty {ty} tx {tx}')
                    logger.warning(f'Path {output_file} exists and config rows match {config_path}')

                progress += 1
                if progress_callback is not None:
                    progress_callback(progress, len(config_rows)*total_tiles)
