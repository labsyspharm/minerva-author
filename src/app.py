from imagecodecs import _zlib # Needed for pyinstaller
from imagecodecs import _imcd # Needed for pyinstaller
from imagecodecs import _jpeg8 # Needed for pyinstaller
from imagecodecs import _jpeg2k # Needed for pyinstaller
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
from tifffile import TiffFile
from tifffile import create_output
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
from render_jpg import render_color_tiles
from render_jpg import composite_channel
from storyexport import create_story_base, get_story_folders, deduplicate_data
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


class Opener:

    def __init__(self, path):
        self.warning = ''
        self.path = path
        self.tilesize = 1024
        base, ext1 = os.path.splitext(path)
        ext2 = os.path.splitext(base)[1]
        ext = ext2 + ext1

        if ext == '.ome.tif' or ext == '.ome.tiff':
            self.io = TiffFile(self.path, is_ome=False)
            self.group = zarr.open(self.io.series[0].aszarr())
            self.reader = 'tifffile'
            self.ome_version = self._get_ome_version()
            print("OME ", self.ome_version)
            num_channels = self.get_shape()[0]
            tile_0 = self.get_tifffile_tile(num_channels, 0,0,0,0)
            if (num_channels == 3 and tile_0.dtype == 'uint8'):
                self.rgba = True
                self.rgba_type = '3 channel'
            elif (num_channels == 1 and tile_0.dtype == 'uint8'):
                self.rgba = True
                self.rgba_type = '1 channel'
            else:
                self.rgba = False
                self.rgba_type = None

        else:
            self.io = OpenSlide(self.path)
            self.dz = DeepZoomGenerator(self.io, tile_size=1024, overlap=0, limit_bounds=True) 
            self.reader = 'openslide'
            self.rgba = True
            self.rgba_type = None

        print("RGB ", self.rgba)
        print("RGB type ", self.rgba_type)

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

    def close(self):
        self.io.close()

    def is_rgba(self, rgba_type=None):
        if rgba_type is None:
            return self.rgba
        else:
            return self.rgba and rgba_type == self.rgba_type

    def get_level_tiles(self, level, tile_size):
        if self.reader == 'tifffile':

            ny = int(np.ceil(self.group[level].shape[1] / tile_size))
            nx = int(np.ceil(self.group[level].shape[2] / tile_size))
            print((nx, ny))
            return (nx, ny)
        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            return self.dz.level_tiles[l]

    def get_shape(self):
        if self.reader == 'tifffile':

            num_levels = len(self.group)
            shape = self.group[0].shape
            (num_channels, shape_y, shape_x) = shape
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

        try:
            tile = self.group[level][channel_number, iy:iy+tilesize, ix:ix+tilesize]
            tile = np.squeeze(tile)
            return tile
        except Exception as e:
            G['logger'].error(e)
            return None

    def get_tifffile_tile(self, num_channels, level, tx, ty, channel_number, tilesize=None):

        if self.reader == 'tifffile':

            self.tilesize = max(self.io.series[0].pages[0].chunks)

            if (tilesize is None) and self.tilesize == 0:
                # Warning... return untiled planes as all-black
                self.tilesize = 1024
                self.warning = f'Level {level} is not tiled. It will show as all-black.'
                tile = np.zeros((1024, 1024), dtype=ifd.dtype)

            elif (tilesize is not None) and self.tilesize == 0:
                self.tilesize = tilesize
                tile = self.read_tiles(level, channel_number, tx, ty, tilesize)

            elif (tilesize is not None) and (self.tilesize != tilesize):
                tile = self.read_tiles(level, channel_number, tx, ty, tilesize)

            else:
                self.tilesize = self.tilesize if self.tilesize else 1024
                tile = self.read_tiles(level, channel_number, tx, ty, self.tilesize)

            if tile is None:
                return None

            return tile

    def get_tile(self, num_channels, level, tx, ty, channel_number):
        
        if self.reader == 'tifffile':
 
            if self.is_rgba('3 channel'):
                tile_0 = self.get_tifffile_tile(num_channels, level, tx, ty, 0)
                tile_1 = self.get_tifffile_tile(num_channels, level, tx, ty, 1)
                tile_2 = self.get_tifffile_tile(num_channels, level, tx, ty, 2)
                tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
                tile[:, :, 0] = tile_0
                tile[:, :, 1] = tile_1
                tile[:, :, 2] = tile_2
                format = 'I;8'
            else:
                tile = self.get_tifffile_tile(num_channels, level, tx, ty, channel_number)
                format = 'I;16'

            return Image.fromarray(tile, format)

        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            img = self.dz.get_tile(l, (tx, ty))
            return img

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
            for i, (marker, color, start, end) in enumerate(zip(
                    settings['Channel Number'], settings['Color'],
                    settings['Low'], settings['High']
            )):
                num_channels = self.get_shape()[0]
                tile = self.get_tifffile_tile(num_channels, level, tx, ty, int(marker), tile_size)

                if i == 0:
                    target = np.zeros(tile.shape + (3,), np.float32)

                composite_channel(
                    target, tile, colors.to_rgb(color), float(start), float(end)
                )

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
        'sample_info': {
            'rotation': 0,
            'name': '',
            'text': ''
        },
        'groups': [],
        'waypoints': [],
        'channels': [],
        'loaded': False,
        'maxLevel': None,
        'tilesize': 1024,
        'height': 1024,
        'width': 1024,
        'save_progress': 0,
        'save_progress_max': 0
    }
    _yaml = {
        'Images': [],
        'Layout': {'Grid': [['i0']]},
        'Groups': []
    }
    return (_g, _yaml)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder at _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

G, YAML = reset_globals()
tiff_lock = multiprocessing.Lock()
app = Flask(__name__,
            static_folder=resource_path('static'),
            static_url_path='')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def open_input_file(path):
    tiff_lock.acquire()
    if G['opener'] is None and path is not None:
        G['opener'] = Opener(path)
    tiff_lock.release()


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

@app.route('/')
def root():
    """
    Serves the minerva-author web UI
    """
    global G
    global YAML
    close_tiff()
    G, YAML = reset_globals()
    return app.send_static_file('index.html')

@app.route('/story/', defaults={'path': 'index.html'})
@app.route('/story/<path:path>')
@cross_origin()
@nocache
def out_story(path):
    if G['out_yaml'] is None:
        response = make_response('Not found', 404)
        response.mimetype = "text/plain"
        return response
    out_dir = os.path.dirname(G['out_yaml'])
    return send_file(os.path.join(out_dir, path))

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

    img_io = render_tile(G['opener'], len(G['channels']),
                        int(level), int(x), int(y), int(channel))
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

@app.route('/api/stories', methods=['POST'])
@cross_origin()
@nocache
def api_stories():

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

    def make_waypoints(d):

        data_dict = deduplicate_data(d, 'data')
        for waypoint in d:
            wp = {
                'Name': waypoint['name'],
                'Description': waypoint['text'],
                'Arrows': list(map(format_arrow, waypoint['arrows'])),
                'Overlays': list(map(format_overlay, waypoint['overlays'])),
                'Group': waypoint['group'],
                'Zoom': waypoint['zoom'],
                'Pan': waypoint['pan'],
            }
            for vis in ['VisScatterplot', 'VisCanvasScatterplot', 'VisMatrix']:
                if vis in waypoint:
                    wp[vis] = waypoint[vis]
                    wp[vis]['data'] = data_dict[wp[vis]['data']]

            if 'VisBarChart' in waypoint:
                wp['VisBarChart'] = data_dict[waypoint['VisBarChart']]

            yield wp

    def make_stories(d):
        for story in d:
            yield {
                'Name': story['name'],
                'Description': story['text'],
                'Waypoints': list(make_waypoints(story['waypoints']))
            }

    if request.method == 'POST':
        data = request.json['stories']
        YAML['Stories'] = list(make_stories(data))
        return 'OK'

@app.route('/api/save', methods=['POST'])
@cross_origin()
@nocache
def api_save():
    """
    Saves minerva-author project information in dat-file.
    Returns: OK on success

    """
    if request.method == 'POST':
        data = request.json
        data['in_file'] = G['in_file']
        data['csv_file'] = G['csv_file']
        G['sample_info'] = data['sample_info']
        G['waypoints'] = data['waypoints']
        G['groups'] = data['groups']

        out_dir, out_yaml, out_dat, out_log = get_story_folders(G['out_name'])

        G['out_dat'] = out_dat
        G['out_yaml'] = out_yaml
        G['out_dir'] = out_dir
        G['out_log'] = out_log

        with open(G['out_dat'], 'w') as out_file:
            json.dump(data, out_file)

        return 'OK'
    

def render_progress_callback(current, max):
    G['save_progress'] = current
    G['save_progress_max'] = max

@app.route('/api/render/progress', methods=['GET'])
@cross_origin()
@nocache
def get_render_progress():
    """
    Returns progress of rendering of tiles (0-100). The progress bar in minerva-author-ui uses this endpoint.
    Returns: JSON which contains progress and max

    """
    return jsonify({
        "progress": G['save_progress'],
        "max": G['save_progress_max']
    })

@app.route('/api/render', methods=['POST'])
@cross_origin()
@nocache
def api_render():
    """
    Renders all image tiles and saves them under new minerva-story instance.
    Returns: OK on success

    """
    G['save_progress'] = 0
    G['save_progress_max'] = 0

    def make_yaml(d):
        for group in d:
            channels = group['channels']
            c_path = '--'.join(
                str(channels[i]['id']) + '__' + channels[i]['label']
                for i in range(len(channels))
            )
            g_path = group['label'].replace(' ', '-') + '_' + c_path

            yield {
                'Path': g_path,
                'Name': group['label'],
                'Colors': [c['color'] for c in channels],
                'Channels': [c['label'] for c in channels]
            }

    def make_rows(d):
        for group in d:
            for channel in group['channels']:
                yield {
                    'Group': group['label'],
                    'Marker Name': channel['label'],
                    'Channel Number': str(channel['id']),
                    'Low': int(65535 * channel['min']),
                    'High': int(65535 * channel['max']),
                    'Color': '#' + channel['color'],
                }
    
    if request.method == 'POST':
        data = request.json['groups']
        config_rows = list(make_rows(data))
        YAML['Groups'] = list(make_yaml(data))
        YAML['Header'] = request.json['header']
        YAML['Rotation'] = request.json['rotation']
        sample_name = request.json['image']['description']
        YAML['Images'][0]['Description'] = sample_name
        YAML['Images'][0]['Path'] = 'images/' + G['out_name']

        create_story_base(G['out_name'], request.json['waypoints'])

        out_dir, out_yaml, out_dat, out_log = get_story_folders(G['out_name'])
        G['out_yaml'] = out_yaml

        with open(G['out_yaml'], 'w') as wf:
            json_text = json.dumps(YAML, ensure_ascii=False)
            wf.write(json_text)

        render_color_tiles(G['opener'], G['out_dir'], 1024,
                           len(G['channels']), config_rows, G['logger'], progress_callback=render_progress_callback)

        return 'OK'

@app.route('/api/import/groups', methods=['POST'])
@cross_origin()
@nocache
def api_import_groups():
    if request.method == 'POST':
        data = request.json
        input_file = pathlib.Path(data['filepath'])
        if not os.path.exists(input_file):
            return api_error(404, 'Dat file not found: ' + str(input_file))

        if (input_file.suffix == '.dat'):
            saved = pickle.load( open( input_file, "rb" ) )
            G['groups'] = saved['groups']

        return jsonify({
            'groups': G['groups']
        })

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
            'channels': G['channels'],
            'tilesize': G['tilesize'],
            'maxLevel': G['maxLevel'],
            'height': G['height'],
            'width': G['width'],
            'warning': G['opener'].warning if G['opener'] else '',
            'rgba': G['opener'].is_rgba() if G['opener'] else False
        })

    if request.method == 'POST':
        chanLabel = {}
        data = request.form
        input_file = pathlib.Path(data['filepath'])
        if not os.path.exists(input_file):
            return api_error(404, 'Image file not found: ' + str(input_file))

        if (input_file.suffix == '.dat' or input_file.suffix == '.json'):
            if input_file.suffix == '.dat':
                saved = pickle.load( open( input_file, "rb" ) )
            else:
                with open(input_file) as json_file:
                    saved = json.load(json_file)

            input_file = pathlib.Path(saved['in_file'])
            if (data['csvpath']):
                csv_file = pathlib.Path(data['csvpath'])
                if not os.path.exists(csv_file):
                    return api_error(404, 'Marker csv file not found: ' + str(csv_file))
            else:
                csv_file = pathlib.Path(saved['csv_file'])
            G['sample_info'] = saved['sample_info']
            try:
                G['sample_info']['rotation']
            except KeyError:
                G['sample_info']['rotation'] = 0

            G['waypoints'] = saved['waypoints']
            G['groups'] = saved['groups']
            for group in saved['groups']:
                for chan in group['channels']:
                    chanLabel[str(chan['id'])] = chan['label']
        else:
            csv_file = pathlib.Path(data['csvpath'])

        out_name = data['dataset']
        if out_name == '':
            out_name = 'out'

        out_dir, out_yaml, out_dat, out_log = get_story_folders(out_name)

        try:
            print("Opening file: ", str(input_file))

            if G['opener'] is None:
                open_input_file(str(input_file))


            (num_channels, num_levels, width, height) = G['opener'].get_shape()
            tilesize = G['opener'].tilesize

            YAML['Images'] = [{
                'Name': 'i0',
                'Description': '',
                'Path': 'http://127.0.0.1:2020/api/out',
                'Width': width,
                'Height': height,
                'MaxLevel': num_levels - 1
            }]
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
            while label_num < num_channels:
                yield chanLabel.get(str(label_num), str(label_num))
                label_num += 1

        try:
            labels = list(yield_labels(num_channels))
        except Exception as e:
            return api_error(500, "Error in opening marker csv file")

        fh = logging.FileHandler(str(out_log))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        G['logger'].addHandler(ch)
        G['logger'].addHandler(fh)

        if os.path.exists(input_file):
            G['out_yaml'] = str(out_yaml)
            G['out_dat'] = str(out_dat)
            G['out_dir'] = str(out_dir)
            G['out_name'] = out_name
            G['in_file'] = str(input_file)
            G['csv_file'] = str(csv_file)
            G['channels'] = labels
            G['loaded'] = True
        else:
            G['logger'].error(f'Input file {input_file} does not exist')

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

def open_browser():
    webbrowser.open_new('http://127.0.0.1:' + str(PORT) + '/')


if __name__ == '__main__':
    Timer(1, open_browser).start()

    atexit.register(close_tiff)
    if '--dev' in sys.argv:
        app.run(debug=False, port=PORT)
    else:
        serve(app, listen="127.0.0.1:" + str(PORT), threads=10)
