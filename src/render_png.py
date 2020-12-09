""" Render PNG tiles
"""
import itertools
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
from PIL import Image
import numpy as np
import tifffile
import json
import os
import io
from threading import Lock

tiff_lock = Lock()

def render_tile(opener, level, tx, ty, channel_number, fmt=None):
    with tiff_lock:

        (num_channels, num_levels, width, height) = opener.get_shape()
        tilesize = opener.tilesize
    
        img_io = None
        if level < num_levels and channel_number < num_channels:
            if tx <= width // tilesize and ty <= height // tilesize:
                img = opener.get_tile(num_channels, level, tx, ty, channel_number, fmt)
                img_io = io.BytesIO()
                img.save(img_io, 'PNG', compress_level=1)
                img_io.seek(0)

    return img_io

def mix(v1, v2, a):
    return v1 * (1 - a) + v2 * a

def hsv2rgba(hsv_buff):
    K = np.array((1, 2/3, 1/3), np.float64)
    alpha = np.expand_dims(hsv_buff[:,:,3], 2)
    hue_buff = np.repeat(hsv_buff[:,:,0][:, :, np.newaxis], 3, axis=2)
    sat_buff = np.repeat(hsv_buff[:,:,1][:, :, np.newaxis], 3, axis=2)
    val_buff = np.repeat(hsv_buff[:,:,2][:, :, np.newaxis], 3, axis=2)
    rgb = np.clip(np.abs(np.modf(hue_buff + K)[0] * 6 - 3) - 1, 0, 1);
    rgb = val_buff * mix(1, rgb, sat_buff)
    return np.concatenate((rgb,alpha), axis=2)

def spike(image):
    buff = np.zeros(image.shape + (4,), np.float64)
    star_hue = 1/3 + 1/100
    star_sat = 1/7 + 1/1000
    star_val = 1/2 + 1/100
    buff[:,:,0] = np.modf(image * star_hue)[0]
    buff[:,:,1] = mix(0.6, 1.0, np.modf(image * star_sat)[0])
    buff[:,:,2] = mix(0.2, 0.9, np.modf(image * star_val)[0])
    buff[:,:,3][image != 0] = 1.0
    return buff

def colorize_mask(target, image, color):
    ''' Render _image_ in pseudocolor into _target_
    Args:
        target: Numpy uint8 array containing RGBA composition target
        image: Numpy uint32 array of image to render and composite
        color: Color as r, g, b float array, 0-1
    '''
    hsv_buff = spike(image)
    rgba_buff = hsv2rgba(hsv_buff)
    target[:] = np.around(255 * rgba_buff).astype(np.uint8)

    # Manually set color for ID 1
    uint8_color1 = tuple(np.uint8( color ) * 255)
    target[image == 1] = uint8_color1 + (255,)

def render_u32_tiles(opener, output_dir, tile_size, colors, logger, progress_callback=None):
    EXT = 'png'

    settings = {'Color': colors, 'Source': str(opener.path)}
    print('Processing:', str(opener.path))

    output_path = pathlib.Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    config_path = output_path / 'config.json'
    old_settings = {}

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                old_settings = json.load(f)
            except json.decoder.JSONDecodeError as err:
                print(err)

    with open(config_path, 'w') as f:
        json.dump(settings, f)

    num_levels = opener.get_shape()[1]

    progress = 0

    if num_levels < 2:
        logger.warning(f'Number of levels {num_levels} < 2')

    for level in range(num_levels):

        (nx, ny) = opener.get_level_tiles(level, tile_size)
        print('    level {} ({} x {})'.format(level, ny, nx))

        for ty, tx in itertools.product(range(0, ny), range(0, nx)):

            filename = '{}_{}_{}.{}'.format(level, tx, ty, EXT)
            output_file = str(output_path / filename)

            # Only save file if change in config rows
            if not (os.path.exists(output_file) and settings == old_settings):
                try:
                    opener.save_tile(output_file, settings, tile_size, level, tx, ty, is_mask=True)
                except AttributeError as e:
                    logger.error(f'{level} ty {ty} tx {tx}: {e}')
            else:
                logger.warning(f'Not saving tile level {level} ty {ty} tx {tx}')
                logger.warning(f'Path {output_file} exists and config rows match {config_path}')

            progress += 1
            if progress_callback is not None:
                progress_callback(progress)
