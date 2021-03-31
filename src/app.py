from imagecodecs import _zlib # Needed for pyinstaller
from imagecodecs import _imcd # Needed for pyinstaller
from imagecodecs import _jpeg8 # Needed for pyinstaller
from imagecodecs import _jpeg2k # Needed for pyinstaller
from numcodecs import compat_ext # Needed for pyinstaller
from numcodecs import blosc # Needed for pyinstaller
from urllib.parse import unquote
from pyramid_assemble import main as make_ome
from concurrent.futures import ThreadPoolExecutor
import ome_types
import pathlib
import re
import string
import sys
import os
import csv
import json
import math
import time
import zarr
from distutils.errors import DistutilsFileError
from distutils import file_util
from tifffile import TiffFile
from tifffile import create_output
from tifffile.tifffile import TiffFileError
import pickle
import webbrowser
import numpy as np
import imagecodecs
from PIL import Image
from matplotlib import colors
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from threading import Timer
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file
from flask import make_response
from render_png import render_tile
from render_png import colorize_mask
from render_png import colorize_integer
from render_png import render_u32_tiles
from render_jpg import render_color_tiles
from render_jpg import composite_channel
from render_jpg import _calculate_total_tiles
from storyexport import create_story_base, get_story_folders
from storyexport import deduplicate_data, label_to_dir
from storyexport import mask_path_from_index
from storyexport import mask_label_from_index
from storyexport import group_path_from_label
from flask_cors import CORS, cross_origin
from pathlib import Path
from waitress import serve
from functools import wraps, update_wrapper
from datetime import datetime
import multiprocessing
import logging
import atexit

if os.name == 'nt':
    from ctypes import windll


PORT = 2020

FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def check_ext(path):
    base, ext1 = os.path.splitext(path)
    ext2 = os.path.splitext(base)[1]
    return ext2 + ext1

def tif_path_to_ome_path(path):
    base, ext = os.path.splitext(path)
    return f'{base}.ome{ext}'

def extract_story_json_stem(input_file):
    default_out_name = input_file.stem
    # Handle extracting the actual stem from .story.json files
    if (pathlib.Path(default_out_name).suffix in ['.story']):
        default_out_name = pathlib.Path(default_out_name).stem
    return default_out_name

def copy_vis_csv_files(waypoint_data, json_path):
    input_dir = json_path.parent
    author_stem = extract_story_json_stem(json_path)
    vis_data_dir = f'{author_stem}-story-infovis'

    vis_path_dict_out = deduplicate_data(waypoint_data, input_dir / vis_data_dir)

    if not len(vis_path_dict_out):
        return

    if not (input_dir / vis_data_dir).exists():
        (input_dir / vis_data_dir).mkdir(parents=True)

    # Copy the visualization csv files to an infovis directory
    for in_path, out_path in vis_path_dict_out.items():
        try:
            file_util.copy_file(in_path, out_path)
        except DistutilsFileError as e:
            print(f'Cannot copy {in_path}')
            print(e)

class Opener:

    def __init__(self, path):
        self.warning = ''
        self.path = path
        self.reader = None
        self.tilesize = 1024
        self.ext = check_ext(path)
        self.default_dtype = np.uint16

        if self.ext == '.ome.tif' or self.ext == '.ome.tiff':
            self.io = TiffFile(self.path, is_ome=False)
            self.group = zarr.open(self.io.series[0].aszarr())
            self.reader = 'tifffile'
            self.ome_version = self._get_ome_version()
            print("OME ", self.ome_version)
            num_channels = self.get_shape()[0]
            tile_0 = self.get_tifffile_tile(num_channels, 0,0,0,0, 1024)
            if tile_0 is not None:
                self.default_dtype = tile_0.dtype

            if (num_channels == 3 and tile_0.dtype == 'uint8'):
                self.rgba = True
                self.rgba_type = '3 channel'
            elif (num_channels == 1 and tile_0.dtype == 'uint8'):
                self.rgba = True
                self.rgba_type = '1 channel'
            else:
                self.rgba = False
                self.rgba_type = None

            print("RGB ", self.rgba)
            print("RGB type ", self.rgba_type)

        elif self.ext == '.svs':
            self.io = OpenSlide(self.path)
            self.dz = DeepZoomGenerator(self.io, tile_size=1024, overlap=0, limit_bounds=True)
            self.reader = 'openslide'
            self.rgba = True
            self.rgba_type = None
            self.default_dtype = np.uint8

            print("RGB ", self.rgba)
            print("RGB type ", self.rgba_type)

        else:
            self.reader = None


    def _get_ome_version(self):
        try:
            software = self.io.pages[0].tags[305].value
            sub_ifds = self.io.pages[0].tags[330].value
            if "Faas" in software or sub_ifds is None:
                return 5

            m = re.search('OME\\sBio-Formats\\s(\\d+)\\.\\d+\\.\\d+', software)
            if m is None:
                return 5
            return int(m.group(1))
        except Exception as e:
            print(e)
            return 5

    def load_xml_markers(self):
        if self.ext == '.ome.tif' or self.ext == '.ome.tiff':
            try:
                metadata = ome_types.from_tiff(self.path)
            except Exception as e:
                return []

            if not metadata or not metadata.images or not metadata.images[0]:
                return []

            metadata_pixels = metadata.images[0].pixels
            if not metadata_pixels or not metadata_pixels.channels:
                return []

            return [c.name for c in metadata_pixels.channels]
        else:
            return []

    def close(self):
        self.io.close()

    def is_rgba(self, rgba_type=None):
        if rgba_type is None:
            return self.rgba
        else:
            return self.rgba and rgba_type == self.rgba_type

    def get_level_tiles(self, level, tile_size):
        if self.reader == 'tifffile':

            # Negative indexing to support shape len 3 or len 2
            ny = int(np.ceil(self.group[level].shape[-2] / tile_size))
            nx = int(np.ceil(self.group[level].shape[-1] / tile_size))
            return (nx, ny)
        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            return self.dz.level_tiles[l]

    def get_shape(self):
        def parse_shape(shape):
            if len(shape) >= 3:
                (num_channels, shape_y, shape_x) = shape[-3:]
            else:
                (shape_y, shape_x) = shape
                num_channels = 1

            return (num_channels, shape_x, shape_y)

        if self.reader == 'tifffile':

            (num_channels, shape_x, shape_y) = parse_shape(self.group[0].shape)
            all_levels = [parse_shape(v.shape) for v in self.group.values()]
            num_levels = len([shape for shape in all_levels if max(shape[1:]) > 512])
            return (num_channels, num_levels, shape_x, shape_y)

        elif self.reader == 'openslide':

            (width, height) = self.io.dimensions

            def has_one_tile(counts):
                return max(counts) == 1

            small_levels = list(filter(has_one_tile, self.dz.level_tiles))
            level_count = self.dz.level_count - len(small_levels) + 1

            return (3, level_count, width, height)

    def read_tiles(self, level, channel_number, tx, ty, tilesize):
        ix = tx * tilesize
        iy = ty * tilesize

        num_channels = self.get_shape()[0]
        try:
            if num_channels == 1:
                tile = self.group[level][iy:iy+tilesize, ix:ix+tilesize]
            else:
                tile = self.group[level][channel_number, iy:iy+tilesize, ix:ix+tilesize]
            tile = np.squeeze(tile)
            return tile
        except Exception as e:
            G['logger'].error(e)
            return None

    def get_tifffile_tile(self, num_channels, level, tx, ty, channel_number, tilesize=1024):

        if self.reader == 'tifffile':

            tile = self.read_tiles(level, channel_number, tx, ty, tilesize)

            if tile is None:
                return np.zeros((tilesize, tilesize), dtype=self.default_dtype)

            return tile

    def get_tile(self, num_channels, level, tx, ty, channel_number, fmt=None):

        if self.reader == 'tifffile':

            if self.is_rgba('3 channel'):
                tile_0 = self.get_tifffile_tile(num_channels, level, tx, ty, 0, 1024)
                tile_1 = self.get_tifffile_tile(num_channels, level, tx, ty, 1, 1024)
                tile_2 = self.get_tifffile_tile(num_channels, level, tx, ty, 2, 1024)
                tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
                tile[:, :, 0] = tile_0
                tile[:, :, 1] = tile_1
                tile[:, :, 2] = tile_2
                _format = 'I;8'
            else:
                tile = self.get_tifffile_tile(num_channels, level, tx, ty, channel_number, 1024)
                _format = fmt if fmt else 'I;16'

                if (_format == 'RGBA' and tile.dtype != np.uint32):
                    tile = tile.astype(np.uint32)

                if (_format == 'I;16' and tile.dtype != np.uint16):
                    if tile.dtype == np.uint8:
                        tile = 255 * tile.astype(np.uint16)
                    else:
                        # TODO: real support for uint32, signed values, and floats
                        tile = np.clip(tile, 0, 65535).astype(np.uint16)

            return Image.fromarray(tile, _format)

        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            img = self.dz.get_tile(l, (tx, ty))
            return img

    def save_mask_tiles(self, filename, mask_params, logger, tile_size, level, tx, ty):

        should_skip_tile = {}

        def get_empty_path(path):
            basename = os.path.splitext(path)[0]
            return pathlib.Path(f'{basename}_tmp.txt')

        for image_params in mask_params['images']:

            output_file = str(image_params['out_path'] / filename)
            path_exists = os.path.exists(output_file) or os.path.exists(get_empty_path(output_file))
            should_skip = path_exists and image_params['is_up_to_date']
            should_skip_tile[output_file] = should_skip

        if all(should_skip_tile.values()):
            logger.warning(f'Not saving tile level {level} ty {ty} tx {tx}')
            logger.warning(f'Every mask {filename} exists with same rendering settings')
            return

        if self.reader == 'tifffile':
            num_channels = self.get_shape()[0]
            tile = self.get_tifffile_tile(num_channels, level, tx, ty, 0, tile_size)

            for image_params in mask_params['images']:

                output_file = str(image_params['out_path'] / filename)
                if should_skip_tile[output_file]:
                    continue

                target = np.zeros(tile.shape + (4,), np.uint8)
                skip_empty_tile = True

                for channel in image_params['settings']['channels']:
                    rgba_color = [ int(255 * i) for i in (colors.to_rgba(channel['color'])) ]
                    ids = channel['ids']

                    if len(ids) > 0:
                        bool_tile = np.isin(tile, ids)
                        # Signal that we must actually save the image
                        if not skip_empty_tile or np.any(bool_tile):
                            skip_empty_tile = False
                        target[bool_tile] = rgba_color
                    else:
                        # Note, any channel without ids to map will override all others
                        target = colorize_mask(target, tile)
                        skip_empty_tile = False

                if skip_empty_tile:
                    empty_file = get_empty_path(output_file)
                    if not os.path.exists(empty_file):
                        with open(empty_file, 'w') as fp:
                            pass
                else:
                    img = Image.frombytes('RGBA', target.T.shape[1:], target.tobytes())
                    img.save(output_file, quality=85)

    def save_tile(self, output_file, settings, tile_size, level, tx, ty):
        if self.reader == 'tifffile' and self.is_rgba('3 channel'):

            num_channels = self.get_shape()[0]
            tile_0 = self.get_tifffile_tile(num_channels, level, tx, ty, 0, tile_size)
            tile_1 = self.get_tifffile_tile(num_channels, level, tx, ty, 1, tile_size)
            tile_2 = self.get_tifffile_tile(num_channels, level, tx, ty, 2, tile_size)
            tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
            tile[:,:,0] = tile_0
            tile[:,:,1] = tile_1
            tile[:,:,2] = tile_2

            img = Image.fromarray(tile, 'RGB')
            img.save(output_file, quality=85)

        elif self.reader == 'tifffile' and self.is_rgba('1 channel'):

            num_channels = self.get_shape()[0]
            tile = self.get_tifffile_tile(num_channels, level, tx, ty, 0, tile_size)

            img = Image.fromarray(tile, 'RGB')
            img.save(output_file, quality=85)

        elif self.reader == 'tifffile':
            target = None
            for i, (marker, color, start, end) in enumerate(zip(
                    settings['Channel Number'], settings['Color'],
                    settings['Low'], settings['High']
            )):
                num_channels = self.get_shape()[0]
                tile = self.get_tifffile_tile(num_channels, level, tx, ty, int(marker), tile_size)

                if (tile.dtype != np.uint16):
                    if tile.dtype == np.uint8:
                        tile = 255 * tile.astype(np.uint16)
                    else:
                        tile = tile.astype(np.uint16)

                if i == 0 or target is None:
                    target = np.zeros(tile.shape + (3,), np.float32)

                composite_channel(
                    target, tile, colors.to_rgb(color), float(start), float(end)
                )

            if target is not None:
                np.clip(target, 0, 1, out=target)
                target_u8 = (target * 255).astype(np.uint8)
                img = Image.frombytes('RGB', target.T.shape[1:], target_u8.tobytes())
                img.save(output_file, quality=85)

        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            img = self.dz.get_tile(l, (tx, ty))
            img.save(output_file, quality=85)

def api_error(status, message):
    return jsonify({
        "error": message
    }), status

def reset_globals():
    _g = {
        'in_file': None,
        'opener': None,
        'csv_file': None,
        'out_dir': None,
        'out_dat': None,
        'out_yaml': None,
        'out_log': None,
        'logger': logging.getLogger('app'),
        'import_pool': ThreadPoolExecutor(max_workers=1),
        'sample_info': {
            'rotation': 0,
            'name': '',
            'text': ''
        },
        'mask_openers': {},
        'masks': [],
        'groups': [],
        'waypoints': [],
        'channels': [],
        'loaded': False,
        'maxLevel': None,
        'tilesize': 1024,
        'height': 1024,
        'width': 1024,
        'save_progress': {},
        'save_progress_max': {}
    }
    _g['logger'].setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(FORMATTER)
    _g['logger'].addHandler(ch)
    return _g

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder at _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

G = reset_globals()
tiff_lock = multiprocessing.Lock()
mask_lock = multiprocessing.Lock()
app = Flask(__name__,
            static_folder=resource_path('static'),
            static_url_path='')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def open_input_file(path):
    global G
    tiff_lock.acquire()
    if G['opener'] is None and path is not None:
        opener = Opener(path)
        if opener.reader is not None:
            G['opener'] = Opener(path)
    tiff_lock.release()

def make_mask_opener(path):
    if path not in G['mask_openers']:
        try:
            return Opener(path)
        except (FileNotFoundError, TiffFileError) as e:
            print(e)

def convert_mask(path):
    sys.stdout.reconfigure(line_buffering=True)

    ome_path = tif_path_to_ome_path(path)
    if os.path.exists(ome_path):
        return

    print(f'Converting {path}')
    tmp_dir = 'minerva_author_tmp_dir'
    tmp_dir = os.path.join(os.path.dirname(path), tmp_dir)
    tmp_path = os.path.join(tmp_dir, 'tmp.tif')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if (os.path.exists(tmp_path)):
        os.remove(tmp_path)
    make_ome(
        [pathlib.Path(path)], pathlib.Path(tmp_path),
        is_mask=True, pixel_size=1
    )
    os.rename(tmp_path, ome_path)
    if os.path.exists(tmp_dir) and not len(os.listdir(tmp_dir)):
        os.rmdir(tmp_dir)
    print(f'Done creating {ome_path}')

def open_input_mask(path, convert=False):
    global G
    opener = None
    invalid = True
    ext = check_ext(path)
    if ext == '.ome.tif' or ext == '.ome.tiff':
        opener = make_mask_opener(path)
    elif ext == '.tif' or ext == '.tiff':
        ome_path = tif_path_to_ome_path(path)
        convertable = os.path.exists(path) and not os.path.exists(ome_path)
        if convert and convertable:
            executor = G['import_pool'].submit(convert_mask, path)
        elif os.path.exists(ome_path):
            opener = make_mask_opener(ome_path)
            path = ome_path
        invalid = False

    if isinstance(opener, Opener):
        mask_lock.acquire()
        G['mask_openers'][path] = opener
        mask_lock.release()
        return False

    return invalid

def get_mask_opener(path):
    opener = None
    ext = check_ext(path)
    invalid = not os.path.exists(path)

    if ext == '.ome.tif' or ext == '.ome.tiff':
        opener = G['mask_openers'].get(path)
    elif ext == '.tif' or ext == '.tiff':
        ome_path = tif_path_to_ome_path(path)
        opener = G['mask_openers'].get(ome_path)

    if invalid and opener:
        mask_lock.acquire()
        opener.close()
        G['mask_openers'].pop(opener.path, None)
        mask_lock.release()
        return None

    return opener

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)

def load_mask_state_subsets(filename):
    all_mask_states = {}
    path = pathlib.Path(filename)
    if not path.is_file() or path.suffix != '.csv':
        return None

    with open(path) as cf:
        state_labels = []
        for row in csv.DictReader(cf):
            try:
                cell_id = int(row.get('CellID', None))
            except ValueError as e:
                print(f'Cannot parse CellID "{cell_id}" in {filename}')
                continue

            # Determine whether to use State or sequentially numbered State
            if not len(state_labels):
                state_labels = ['State']
                if state_labels[0] not in row:
                    state_labels = []
                    for i in range(1, 10):
                        state_i = f'State{i}'
                        if state_i not in row:
                            break
                        state_labels.append(state_i)

                if not len(state_labels):
                    print(f'No State headers found in {filename}')
                    break

            # Load from each State label
            for state_i in state_labels:
                cell_state = row.get(state_i, '')
                if cell_state == '':
                    print(f'Empty {state_i} for CellID "{cell_id}" in {filename}')
                    continue

                mask_subsets = all_mask_states.get(state_i, {})
                mask_group = mask_subsets.get(cell_state, set())
                mask_group.add(cell_id)

                mask_subsets[cell_state] = mask_group
                all_mask_states[state_i] = mask_subsets

    if not len(all_mask_states):
        return None

    return {
        state: {
            k: sorted(v) for (k,v) in mask_subsets.items()
        } for (state, mask_subsets) in all_mask_states.items()
    }

def reload_all_mask_state_subsets(masks):
    all_mask_state_subsets = {}

    def is_mask_ok(mask):
        return 'map_path' in mask and 'channels' in mask

    for mask in masks:
        if is_mask_ok(mask):
            all_mask_state_subsets[mask['map_path']] = {}

    for map_path in all_mask_state_subsets:
        mask_state_subsets = load_mask_state_subsets(map_path)
        if mask_state_subsets is not None:
            all_mask_state_subsets[map_path] = mask_state_subsets

    for mask in masks:
        if not is_mask_ok(mask):
            continue

        mask_state_subsets = all_mask_state_subsets.get(mask['map_path'], {})

        # Support version 1.5.0 or lower
        mask_label = mask.get('label')
        default_label = mask.get('original_label')
        default_label = default_label if default_label else mask_label

        for chan in mask['channels']:
            state_label = chan.get('state_label', 'State')
            original_label = chan.get('original_label')
            original_label = original_label if original_label else default_label
            chan['ids'] = mask_state_subsets.get(state_label, {}).get(original_label, [])
            chan['original_label'] = original_label
            chan['state_label'] = state_label

    return masks


@app.route('/')
def root():
    """
    Serves the minerva-author web UI
    """
    global G
    close_tiff()
    close_masks()
    close_import_pool()
    tiff_lock.acquire()
    mask_lock.acquire()
    G = reset_globals()
    tiff_lock.release()
    mask_lock.release()
    return app.send_static_file('index.html')

@app.route('/story/', defaults={'path': 'index.html'})
@app.route('/story/<path:path>')
@cross_origin()
@nocache
def out_story(path):

    not_valid = (G['out_yaml'] is None)

    out_dir = os.path.dirname(G['out_yaml'] or '')
    out_file = os.path.join(out_dir, path)

    if not_valid or not os.path.exists(out_file):
        response = make_response('Not found', 404)
        response.mimetype = "text/plain"
        return response

    return send_file(out_file)


@app.route('/api/validate/u32/<key>')
@cross_origin()
@nocache
def u32_validate(key):
    """
    Returns status for given image mask
    Args:
        key: URL-escaped path to mask

    Returns: status dict
        invalid: whether the original path does not exist
        ready: whether the ome-tiff version of the path is ready
        path: the ome-tiff version of the path
    """
    img_io = None
    path = unquote(key)
    invalid = True

    # Open the input file on the first request only
    if get_mask_opener(path) is None:
        invalid = open_input_mask(path, convert=True)

    opener = get_mask_opener(path)

    return jsonify({
        "invalid": invalid,
        "ready": True if isinstance(opener, Opener) else False,
        "path": opener.path if isinstance(opener, Opener) else ''
    })


@app.route('/api/mask_subsets/<key>')
@cross_origin()
@nocache
def mask_subsets(key):
    """
    Returns the dictionary of mask subsets
    Args:
        key: URL-escaped path to mask group csv file

    Returns: Dictionary mapping mask subsets to cell ids

    """
    path = unquote(key)

    if not os.path.exists(path):
        response = make_response('Not found', 404)
        response.mimetype = "text/plain"
        return response

    mask_state_subsets = load_mask_state_subsets(path)
    if mask_state_subsets is None:
        response = make_response('Not found', 404)
        response.mimetype = "text/plain"
        return response

    mask_states = []
    mask_subsets = []
    for (mask_state, state_subsets) in mask_state_subsets.items():
        for (k,v) in state_subsets.items():
            mask_states.append(mask_state)
            mask_subsets.append([k, v])

    return jsonify({
        'mask_states': mask_states,
        'mask_subsets': mask_subsets,
        'subset_colors': [ colorize_integer(v[0]) for [k,v] in mask_subsets ]
    })

@app.route('/api/u32/<key>/<level>_<x>_<y>.png')
@cross_origin()
@nocache
def u32_image(key, level, x, y):
    """
    Returns a 32-bit tile from given image mask
    Args:
        key: URL-escaped path to mask
        level: Pyramid level
        x: Tile coordinate x
        y: Tile coordinate y

    Returns: Tile image in png format

    """
    img_io = None
    path = unquote(key)

    # Open the input file on the first request only
    if get_mask_opener(path) is None:
        open_input_mask(path, convert=False)

    opener = get_mask_opener(path)
    if isinstance(opener, Opener):
        img_io = render_tile(opener, int(level),
                        int(x), int(y), 0, 'RGBA')

    if img_io is None:
        response = make_response('Not found', 404)
        response.mimetype = "text/plain"
        return response

    return send_file(img_io, mimetype='image/png')


@app.route('/api/u16/<channel>/<level>_<x>_<y>.png')
@cross_origin()
@nocache
def u16_image(channel, level, x, y):
    """
    Returns a single channel 16-bit tile from the image
    Args:
        channel: Image channel
        level: Pyramid level
        x: Tile coordinate x
        y: Tile coordinate y

    Returns: Tile image in png format

    """

    # Open the input file on the first request only
    if G['opener'] is None:
        open_input_file(G['in_file'])

    img_io = None
    if G['opener'] is not None:
        img_io = render_tile(G['opener'], int(level),
                         int(x), int(y), int(channel))

    if img_io is None:
        response = make_response('Not found', 404)
        response.mimetype = "text/plain"
        return response

    return send_file(img_io, mimetype='image/png')

@app.route('/api/out/<path:path>')
@cross_origin()
@nocache
def out_image(path):
    image_path = os.path.join(G['out_dir'], path)
    return send_file(image_path, mimetype='image/jpeg')

def make_saved_chan(chan):
    # We consider ids too large to store
    return {k:v for (k,v) in chan.items() if k != 'ids'}

def make_saved_mask(mask):
    new_mask = {k:v for (k,v) in mask.items() if k != 'channels'}
    new_mask['channels'] = list(map(make_saved_chan, mask.get('channels', [])))
    return new_mask

def make_saved_file(data):
    new_copy = {k:v for (k,v) in data.items() if k != 'masks'}
    new_copy['masks'] = list(map(make_saved_mask, data.get('masks', [])))
    return new_copy

@app.route('/api/save', methods=['POST'])
@cross_origin()
@nocache
def api_save():
    """
    Saves minerva-author project information in json file.
    Returns: OK on success

    """
    if request.method == 'POST':
        data = request.json
        data['in_file'] = G['in_file']
        data['csv_file'] = G['csv_file']
        # Set globals whether saving or autosaving
        G['sample_info'] = data['sample_info']
        G['waypoints'] = data['waypoints']
        G['groups'] = data['groups']
        G['masks'] = data['masks']
        # This should only happen after G is set
        data = make_saved_file(data)

        out_dir, out_yaml, out_dat, out_log = get_story_folders(G['out_name'])

        G['out_dat'] = out_dat
        G['out_yaml'] = out_yaml
        G['out_dir'] = out_dir
        G['out_log'] = out_log

        saved = load_saved_file(out_dat)[0]
        # Only relegate to autosave if save file exists
        if saved and data.get('is_autosave'):
            # Copy new data to autosave and copy old saved to data
            data['autosave'] = copy_saved_states(data, {})
            data = copy_saved_states(saved, data)
            # Set the autosave timestamp
            data['autosave']['timestamp'] = time.time()
        else:
            # Set the current timestamp
            data['timestamp'] = time.time()
            # Persist old autosaves just in case
            if saved and 'autosave' in saved:
                data['autosave'] = saved['autosave']
            # Make a copy of the visualization csv files
            # for use with save_exhibit_pyramid.py
            copy_vis_csv_files(data['waypoints'], pathlib.Path(out_dat))

        with open(G['out_dat'], 'w') as out_file:
            json.dump(data, out_file)

        return 'OK'

def render_progress_callback(current, maximum, key='default'):
    G['save_progress'][key] = current
    G['save_progress_max'][key] = maximum

def create_progress_callback(maximum, key='default'):
    def progress_callback(current):
        render_progress_callback(current, maximum, key)
    progress_callback(0)
    return progress_callback

@app.route('/api/render/progress', methods=['GET'])
@cross_origin()
@nocache
def get_render_progress():
    """
    Returns progress of rendering of tiles (0-100). The progress bar in minerva-author-ui uses this endpoint.
    Returns: JSON which contains progress and max

    """
    return jsonify({
        "progress": sum(G['save_progress'].values()),
        "max": sum(G['save_progress_max'].values())
    })

def format_arrow(a):
    return {
        'Text': a['text'],
        'HideArrow': a['hide'],
        'Point': a['position'],
        'Angle': 60 if a['angle'] == '' else a['angle']
    }

def format_overlay(o):
    return {
        'x': o[0],
        'y': o[1],
        'width': o[2],
        'height': o[3]
    }

def make_waypoints(d, mask_data, vis_path_dict={}):

    for waypoint in d:
        mask_labels = []
        if len(mask_data) > 0:
            wp_masks = waypoint['masks']
            mask_labels = [mask_label_from_index(mask_data, i) for i in wp_masks]
        wp = {
            'Name': waypoint['name'],
            'Description': waypoint['text'],
            'Arrows': list(map(format_arrow, waypoint['arrows'])),
            'Overlays': list(map(format_overlay, waypoint['overlays'])),
            'Group': waypoint['group'],
            'Masks': mask_labels,
            'ActiveMasks': mask_labels,
            'Zoom': waypoint['zoom'],
            'Pan': waypoint['pan']
        }
        for vis in ['VisScatterplot', 'VisCanvasScatterplot', 'VisMatrix']:
            if vis in waypoint:
                wp[vis] = waypoint[vis]
                wp[vis]['data'] = vis_path_dict[wp[vis]['data']]

        if 'VisBarChart' in waypoint:
            wp['VisBarChart'] = vis_path_dict[waypoint['VisBarChart']]

        yield wp

def make_stories(d, mask_data=[], vis_path_dict={}):
    return [{
        'Name': '',
        'Description': '',
        'Waypoints': list(make_waypoints(d, mask_data, vis_path_dict))
    }]

def make_mask_yaml(mask_data):
    for (i, mask) in enumerate(mask_data):
        yield {
            'Path': mask_path_from_index(mask_data, i),
            'Name': mask_label_from_index(mask_data, i),
            'Colors': [c['color'] for c in mask['channels']],
            'Channels': [c['label'] for c in mask['channels']]
        }

def make_group_path(groups, group):
    c_path = '--'.join(
        str(c['id']) + '__' + label_to_dir(c['label'])
        for c in group['channels']
    )
    g_path = group_path_from_label(groups, group['label'])
    return  g_path + '_' + c_path


def make_groups(d):
    for group in d:
        yield {
            'Name': group['label'],
            'Path': make_group_path(d, group),
            'Colors': [c['color'] for c in group['channels']],
            'Channels': [c['label'] for c in group['channels']]
        }

def make_rows(d):
    for group in d:
        channels = group['channels']
        yield {
            'Group Path': make_group_path(d, group),
            'Channel Number': [str(c['id']) for c in channels],
            'Low': [int(65535 * c['min']) for c in channels],
            'High': [int(65535 * c['max']) for c in channels],
            'Color': ['#' + c['color'] for c in channels]
        }

def make_exhibit_config(opener, out_name, json):

    data = json['groups']
    mask_data = json['masks']
    waypoint_data = json['waypoints']
    vis_path_dict = deduplicate_data(waypoint_data, 'data')

    (num_channels, num_levels, width, height) = opener.get_shape()

    _config = {
        'Images': [{
            'Name': 'i0',
            'Description': json['image']['description'],
            'Path': 'images/' + out_name,
            'Width': width,
            'Height': height,
            'MaxLevel': num_levels - 1
        }],
        'Header': json['header'],
        'Rotation': json['rotation'],
        'Layout': {'Grid': [['i0']]},
        'Stories': make_stories(waypoint_data, mask_data, vis_path_dict),
        'Masks': list(make_mask_yaml(mask_data)),
        'Groups': list(make_groups(data))
    }
    return _config


@app.route('/api/render', methods=['POST'])
@cross_origin()
@nocache
def api_render():
    """
    Renders all image tiles and saves them under new minerva-story instance.
    Returns: OK on success

    """
    G['save_progress'] = {}
    G['save_progress_max'] = {}

    if request.method == 'POST':
        data = request.json['groups']
        mask_data = request.json['masks']
        waypoint_data = request.json['waypoints']
        create_story_base(G['out_name'], waypoint_data, mask_data)
        config_rows = list(make_rows(data))
        exhibit_config = make_exhibit_config(G['opener'], G['out_name'], request.json)

        out_dir, out_yaml, out_dat, out_log = get_story_folders(G['out_name'])
        G['out_yaml'] = out_yaml

        with open(G['out_yaml'], 'w') as wf:
            json.dump(exhibit_config, wf)

        all_mask_params = {}

        for (i, mask) in enumerate(mask_data):

            mask_params = {}
            mask_path = mask['path']

            if mask_path in all_mask_params:
                mask_params = all_mask_params[mask_path]
            else:
                if mask_path not in G['mask_openers']:
                    open_input_mask(mask_path, convert=False)
                mask_params['images'] = []
                mask_params['opener'] = get_mask_opener(mask_path)

            if isinstance(mask_params['opener'], Opener):
                mask_opener = mask_params['opener']
                num_levels = mask_opener.get_shape()[1]
                mask_total = _calculate_total_tiles(mask_opener, 1024, num_levels)
                mask_params['images'].append({
                    'settings': {
                        'channels': [{
                            'ids': c['ids'],
                            'color': '#'+c['color']
                        } for c in mask['channels']],
                        'source': str(mask_path)
                    },
                    'progress': create_progress_callback(mask_total, i),
                    'out_path': pathlib.Path(mask_path_from_index(mask_data, i, out_dir))
                })
                all_mask_params[mask_path] = mask_params
            else:
                print(f'Skipping render of {mask_path}')

        # Render all uint16 image channels
        render_color_tiles(G['opener'], G['out_dir'], 1024, config_rows, G['logger'],
                           progress_callback=render_progress_callback)

        # Render all uint32 segmentation masks
        for mask_params in all_mask_params.values():
            render_u32_tiles(mask_params, 1024, G['logger'])

        return 'OK'

@app.route('/api/import/groups', methods=['POST'])
@cross_origin()
@nocache
def api_import_groups():
    if request.method == 'POST':
        data = request.json
        input_file = pathlib.Path(data['filepath'])
        if not os.path.exists(input_file):
            return api_error(404, 'File not found: ' + str(input_file))

        saved = load_saved_file(input_file)[0]
        if saved and 'groups' in saved:
            G['groups'] = saved['groups']

        return jsonify({
            'groups': G['groups']
        })

def load_saved_file(input_file):
    saved = None
    autosaved = None
    input_path = pathlib.Path(input_file)
    if not input_path.exists():
        return (None, None)

    if input_path.suffix == '.dat':
        saved = pickle.load( open( input_path, "rb" ) )
    else:
        with open(input_path) as json_file:
            saved = json.load(json_file)
            autosaved = saved.get("autosave")

    return (saved, autosaved)

def copy_saved_states(from_save, to_save):
    saved_keys = [
        'sample_info', 'waypoints', 'groups', 'masks'
    ]
    for saved_key in saved_keys:
        if saved_key in from_save:
            to_save[saved_key] = from_save[saved_key]

    return to_save

def is_new_autosave(saved, autosaved):
    if saved is None or autosaved is None:
        return False

    autosaved_time = autosaved.get("timestamp")
    saved_time = saved.get("timestamp")
    if autosaved_time:
        if saved_time:
            # Decide if new autosave
            return autosaved_time > saved_time
        else:
            # Save file from before v1.6.0
            return True
    else:
        # Malformed autosave
        return False

@app.route('/api/import', methods=['GET', 'POST'])
@cross_origin()
@nocache
def api_import():
    if request.method == 'GET':

        return jsonify({
            'loaded': G['loaded'],
            'waypoints': G['waypoints'],
            'sample_info': G['sample_info'],
            'groups': G['groups'],
            'masks': G['masks'],
            'channels': G['channels'],
            'tilesize': G['tilesize'],
            'maxLevel': G['maxLevel'],
            'height': G['height'],
            'width': G['width'],
            'output_save_file': G['out_dat'],
            'warning': G['opener'].warning if G['opener'] else '',
            'rgba': G['opener'].is_rgba() if G['opener'] else False
        })

    if request.method == 'POST':
        chanLabel = {}
        data = request.form
        default_out_name = 'out'
        input_file = pathlib.Path(data['filepath'])
        input_image_file = pathlib.Path(data['filepath'])
        loading_saved_file = input_file.suffix in ['.dat', '.json']

        if not os.path.exists(input_file):
            return api_error(404, 'Image file not found: ' + str(input_file))

        if (loading_saved_file):
            default_out_name = extract_story_json_stem(input_file)
            # autosave_logic should be "ask", "skip", or "load"
            autosave_logic = data.get("autosave_logic", "skip")
            autosave_error = autosave_logic == "ask"

            (saved, autosaved) = load_saved_file(input_file)

            if is_new_autosave(saved, autosaved):
                # We need to know whether to use autosave file
                if autosave_error:
                    action = 'AUTO ASK ERR'
                    return api_error(400, f'{action}: Autosave Error')
                # We will load a new autosave file
                elif autosave_logic == "load":
                    saved = copy_saved_states(autosaved, saved)

            input_image_file = pathlib.Path(saved['in_file'])
            if (data['csvpath']):
                csv_file = pathlib.Path(data['csvpath'])
                if not os.path.exists(csv_file):
                    return api_error(404, 'Marker csv file not found: ' + str(csv_file))
            else:
                csv_file = pathlib.Path(saved['csv_file'])
            if 'sample_info' in saved:
                G['sample_info'] = saved['sample_info']
            try:
                G['sample_info']['rotation']
            except KeyError:
                G['sample_info']['rotation'] = 0

            if 'masks' in saved:
                # This step could take up to a minute
                G['masks'] = reload_all_mask_state_subsets(saved['masks'])

            G['waypoints'] = saved['waypoints']
            G['groups'] = saved['groups']
            for group in saved['groups']:
                for chan in group['channels']:
                    chanLabel[str(chan['id'])] = chan['label']
        else:
            csv_file = pathlib.Path(data['csvpath'])

        out_name = label_to_dir(data['dataset'], empty=default_out_name)
        if out_name == '':
            out_name = default_out_name

        out_dir, out_yaml, out_dat, out_log = get_story_folders(out_name)

        if not loading_saved_file and os.path.exists(out_dat):
            action = 'OUT ASK ERR'
            verb = 'provide an' if out_name == default_out_name else 'change the'
            return api_error(400, f'{action}: Please {verb} output name, as {out_dat} exists.')
        elif loading_saved_file and os.path.exists(out_dat):
            if not os.path.samefile(input_file, out_dat):
                action = 'OUT ASK ERR'
                verb = 'provide an' if out_name == default_out_name else 'change the'
                command = f'Please {verb} output name or directly load {out_dat}';
                return api_error(400, f'{action}: {command}, as that file already exists.')

        try:
            print("Opening file: ", str(input_image_file))

            if G['opener'] is None:
                open_input_file(str(input_image_file))

            (num_channels, num_levels, width, height) = G['opener'].get_shape()
            tilesize = G['opener'].tilesize

            G['maxLevel'] = num_levels - 1
            G['tilesize'] = tilesize
            G['height'] = height
            G['width'] = width

        except Exception as e:
            print (e)
            return api_error(500, 'Invalid tiff file')

        def yield_labels(num_channels):
            label_num = 0
            if str(csv_file) != '.':
                with open(csv_file) as cf:
                    for row in csv.DictReader(cf):
                        if label_num < num_channels:
                            default = row.get('marker_name', str(label_num))
                            default = row.get('Marker Name', default)
                            yield chanLabel.get(str(label_num), default)
                            label_num += 1
            else:
                for label in G['opener'].load_xml_markers():
                    yield label
                    label_num += 1

            while label_num < num_channels:
                yield chanLabel.get(str(label_num), str(label_num))
                label_num += 1

        try:
            labels = list(yield_labels(num_channels))
        except Exception as e:
            return api_error(500, "Error in loading channel marker names")

        fh = logging.FileHandler(str(out_log))
        fh.setLevel(logging.INFO)
        fh.setFormatter(FORMATTER)
        G['logger'].addHandler(fh)

        if os.path.exists(input_image_file):
            G['out_yaml'] = str(out_yaml)
            G['out_dat'] = str(out_dat)
            G['out_dir'] = str(out_dir)
            G['out_name'] = str(out_name)
            G['in_file'] = str(input_image_file)
            G['csv_file'] = str(csv_file)
            G['channels'] = labels
            G['loaded'] = True
        else:
            G['logger'].error(f'Input file {input_image_file} does not exist')

        return 'OK'

@app.route('/api/filebrowser', methods=['GET'])
@cross_origin()
@nocache
def file_browser():
    """
    Endpoint which allows browsing the local file system

    Url parameters:
        path: path to a directory
        parent: if true, returns the contents of parent directory of given path
    Returns:
        Contents of the directory specified by path
        (or parent directory, if parent parameter is set)
    """
    folder = request.args.get('path')
    orig_folder = folder
    parent = request.args.get('parent')
    if folder is None or folder == "":
        folder = Path.home()
    elif parent == 'true':
        folder = Path(folder).parent

    if not os.path.exists(folder):
        return api_error(404, 'Path not found')

    response = {
        "entries": [],
        "path": str(folder)
    }

    # Windows: When navigating back from drive root, we have to show a list of available drives
    if os.name == 'nt' and folder is not None and str(orig_folder) == str(folder) and parent == 'true':
        match = re.search('[A-Za-z]:\\\\$', str(folder))  # C:\ or D:\ etc.
        if match:
            drives = _get_drives_win()
            for drive in drives:
                new_entry = {
                    "name": drive + ":\\",
                    "path": drive + ":\\",
                    "isDir": True
                }
                response["entries"].append(new_entry)
            return jsonify(response)

    # Return a list of folders and files within the requested folder
    for entry in os.scandir(folder):
        try:
            is_directory = entry.is_dir()
            new_entry = {
                "name": entry.name,
                "path": entry.path,
                "isDir": is_directory
            }

            is_broken = False
            is_hidden = entry.name[0] == '.'

            if not is_directory:
                try:
                    stat_result = entry.stat()
                    new_entry["size"] = stat_result.st_size
                    new_entry["ctime"] = stat_result.st_ctime
                    new_entry["mtime"] = stat_result.st_mtime
                except FileNotFoundError as e:
                    is_broken = True

            if not is_hidden and not is_broken:
                response["entries"].append(new_entry)
        except PermissionError as e:
            pass

    return jsonify(response)

# Returns a list of drive letters in Windows
# https://stackoverflow.com/a/827398
def _get_drives_win():
    drives = []
    bitmask = windll.kernel32.GetLogicalDrives()
    for letter in string.ascii_uppercase:
        if bitmask & 1:
            drives.append(letter)
        bitmask >>= 1

    return drives

def close_tiff():
    print("Closing tiff file")
    if G['opener'] is not None:
        try:
            G['opener'].close()
        except Exception as e:
            print(e)

def close_masks():
    print("Closing mask files")
    for opener in G['mask_openers'].values():
        try:
            opener.close()
        except Exception as e:
            print(e)

def close_import_pool():
    print("Closing import pool")
    if G['import_pool'] is not None:
        try:
            G['import_pool'].shutdown()
        except Exception as e:
            print(e)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:' + str(PORT) + '/')

if __name__ == '__main__':
    Timer(1, open_browser).start()

    atexit.register(close_tiff)
    atexit.register(close_masks)
    atexit.register(close_import_pool)

    sys.stdout.reconfigure(line_buffering=True)

    if '--dev' in sys.argv:
        app.run(debug=False, port=PORT)
    else:
        serve(app, listen="127.0.0.1:" + str(PORT), threads=10)
