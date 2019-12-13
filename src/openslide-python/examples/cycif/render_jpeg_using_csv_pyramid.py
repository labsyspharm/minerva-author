from __future__ import print_function, division
import sys
import itertools
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
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

'''
Architecture Panel: S100, aSMA, CD45, DNA
Differentiation State: MITF, NGFR, AXL, CD45
T cell infiltrate: CD4, CD8a, HLAA, Granzyme/FOXP3
Survival: Catenin, survivin, LAMP2, gH2AX
Cell Cycle: KI67, pRB, DNA, Catenin
Other immune: CD11b, LC3, aSMA
Exhaustion: PDL1, PD1, CD3, Ki67
Other signaling: cMET, pERK
'''

TILE_SIZE = 1024
EXT = 'jpg'
num_channels = 24
csv_path = './settings.csv'

input_file_dir = pathlib.Path('/Users/john/data/n/imstor/sorger/data/computation/Jeremy/ashlar_examples/')
input_filepaths = sorted(input_file_dir.glob('*.ome.tif'))

for input_file_path in input_filepaths[:2]:
    print('Processing:', str(input_file_path))

# input_file_path = pathlib.Path('Z:/048-BENBIOPSY1-2017APR/D13_07880_B2.ome.tif')
    input_filename = input_file_path.name.split('.')[0]
    output_path = pathlib.Path('rendered') / input_filename
    tiff = pytiff.Tiff(str(input_file_path))

    assert tiff.number_of_pages % num_channels == 0, "Pyramid/channel mismatch"

    num_levels = tiff.number_of_pages // num_channels

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

    for level in range(num_levels)[:1]:

        page_base = level * num_channels
        tiff.set_page(page_base)
        ny = int(np.ceil(tiff.shape[0] / TILE_SIZE))
        nx = int(np.ceil(tiff.shape[1] / TILE_SIZE))
        print('    level {} ({} x {})'.format(level, ny, nx))

        for ty, tx in itertools.product(range(0, ny), range(0, nx)):

            iy = ty * TILE_SIZE
            ix = tx * TILE_SIZE
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
                    tile = tiff[iy:iy+TILE_SIZE, ix:ix+TILE_SIZE]
                    if i == 0:
                        target = np.zeros(tile.shape + (3,), np.float32)
                    composite_channel(
                        target, tile, colors.to_rgb(color), float(start), float(end)
                    )
                np.clip(target, 0, 1, out=target)
                target_u8 = (target * 255).astype(np.uint8)
                img = Image.frombytes('RGB', target.T.shape[1:], target_u8.tobytes())
                img.save(str(output_path / group_dir / filename), quality=85)
