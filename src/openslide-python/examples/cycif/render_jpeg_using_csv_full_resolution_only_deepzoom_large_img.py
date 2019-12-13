from __future__ import print_function, division
import sys
import itertools
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import numpy as np
import math
import pytiff
from PIL import Image
from matplotlib import colors
import csv
import json
import yaml
from collections import OrderedDict
from deepzoom_tile_cycif import DeepZoomStaticTiler
from time import time
import skimage.io

Image.MAX_IMAGE_PIXELS = int(80000 * 80000) 

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

TILE_SIZE = 1024
EXT = 'jpg'
num_channels = 52
csv_path = './mskcc_rendering_setting.csv'

input_file_dir = pathlib.Path('Z:/Connor/Z136/10b_324/registration/')
input_filepaths = sorted(input_file_dir.rglob('*.ome.tif'))

def main():

    if len(input_filepaths) is 0: print('no file to process')

    for input_file_path in input_filepaths[:]:
        print('Processing:', str(input_file_path))
        time_start = time()

        input_filename = input_file_path.name.split('.')[0]
        input_filename = input_filename.replace('_', '')
        output_path = pathlib.Path('rendered') / input_filename
        tiff = pytiff.Tiff(str(input_file_path))

        # assert tiff.number_of_pages % num_channels == 0, "Pyramid/channel mismatch"
        # num_levels = tiff.number_of_pages // num_channels

        with open(csv_path) as group_config:
            reader = csv.DictReader(group_config)
            config_rows = [dict(row) for row in reader]

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


        for name, settings in list(render_groups.items())[1:]:

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
                tiff.set_page(int(marker))
                tile = tiff[:, :]

                if i == 0:
                    target = np.zeros(tile.shape + (3,), np.float32)
                composite_channel(
                    target, tile, colors.to_rgb(color), float(start), float(end)
                )
            np.clip(target, 0, 1, out=target)
            target_u8 = (target * 255).astype(np.uint8)
            img = Image.frombytes('RGB', target.T.shape[1:], target_u8.tobytes())

            # Multiprocessing depends on pickle which is constrained by 32 bit on Windows machines.
            # The underlining C in Windows is only 32 bit disregard the OS.
            # The highest resolution level is not parallelized to prevent overflowing pickle
            ny = int(np.ceil(img.height / TILE_SIZE))
            nx = int(np.ceil(img.width / TILE_SIZE))

            for ty, tx in itertools.product(range(0, ny), range(0, nx)):
                iy = ty * TILE_SIZE
                ix = tx * TILE_SIZE
                level_0_filename = '{}_{}_{}.{}'.format(0, tx, ty, EXT)
                max_x = ix + TILE_SIZE if ix + TILE_SIZE < img.width else img.width
                max_y = iy + TILE_SIZE if iy + TILE_SIZE < img.height else img.height
                img.crop((ix, iy, max_x, max_y)).save(str(output_path / group_dir) + '/' + level_0_filename, quality=85)

            # Rest of the levels are parallelized usind Openslide DeepZoom
            img = img.resize((int(img.width / 2), int(img.height / 2)))

            print('    Writing tiles to:', str(output_path / group_dir))
            DeepZoomStaticTiler(img, str(output_path / group_dir), 'jpg', 1024, 0, False, 85, 8, False, is_PIL_Image_obj=True, level_offset=1).run()

        print('Used', str(int(time() - time_start)), 'sec for', input_filename)

if __name__ == '__main__':
    main()




    

    