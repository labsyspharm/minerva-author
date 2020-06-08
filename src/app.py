import pathlib
import re
import string
import sys
import os
import csv
import yaml
import pytiff
import pickle
import webbrowser
import numpy as np
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
from render_jpg import composite_channel
from render_jpg import render_color_tiles
from flask_cors import CORS, cross_origin
from pathlib import Path
from waitress import serve
import multiprocessing
import atexit
if os.name == 'nt':
    from ctypes import windll


PORT = 2020

class Opener:

    def __init__(self, path):
        self.path = path
        base, ext1 = os.path.splitext(path)
        ext2 = os.path.splitext(base)[1]
        ext = ext2 + ext1

        if ext == '.ome.tif':
            self.io = pytiff.Tiff(self.path, "r", encoding='utf-8')
            self.reader = 'pytiff'
            num_channels = self.get_shape()[0]
            tile_0 = self.get_pytiff_tile(1024, num_channels, 0,0,0,0)
            if (num_channels == 3 and tile_0.dtype == 'uint8'):
                self.rgba = True
            else:
                self.rgba = False
        else:
            self.io = OpenSlide(self.path)
            self.dz = DeepZoomGenerator(self.io, tile_size=1024, overlap=0, limit_bounds=True) 
            self.reader = 'openslide'
            self.rgba = True

    def close(self):
        self.io.close()

    def is_rgba(self):
        return self.rgba

    def get_level_tiles(self, level, tile_size):
        if self.reader == 'pytiff':
            num_channels = self.get_shape()[0]
            page_base = level * num_channels
            self.io.set_page(page_base)
            ny = int(np.ceil(self.io.shape[0] / tile_size))
            nx = int(np.ceil(self.io.shape[1] / tile_size))
            return (nx, ny)
        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            return self.dz.level_tiles[l]

    def get_shape(self):
        if self.reader == 'pytiff':

            num_channels = 0
            for page in self.io.pages:
                if self.io.shape == page.shape:
                    num_channels += 1

            num_levels = self.io.number_of_pages // num_channels

            return (num_channels, num_levels, self.io.shape[1], self.io.shape[0])

        elif self.reader == 'openslide':

            (width, height) = self.io.dimensions

            def has_one_tile(counts):
                return max(counts) == 1

            small_levels = list(filter(has_one_tile, self.dz.level_tiles))
            level_count = self.dz.level_count - len(small_levels)

            return (3, level_count, width, height)

    def get_pytiff_tile(self, tile_size, num_channels, level, tx, ty, channel_number):
        
        if self.reader == 'pytiff':

            page_base = level * num_channels
            page = page_base + channel_number

            iy = ty * tile_size
            ix = tx * tile_size

            self.io.set_page(page)
            return self.io[iy:iy+tile_size, ix:ix+tile_size]

    def get_tile(self, tile_size, num_channels, level, tx, ty, channel_number):
        
        if self.reader == 'pytiff':
 
            if self.is_rgba():
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                tile[:,:,0] = self.get_pytiff_tile(tile_size, num_channels, level, tx, ty, 0)
                tile[:,:,1] = self.get_pytiff_tile(tile_size, num_channels, level, tx, ty, 1)
                tile[:,:,2] = self.get_pytiff_tile(tile_size, num_channels, level, tx, ty, 2)
            else:
                tile = self.get_pytiff_tile(tile_size, num_channels, level, tx, ty, channel_number)

            return Image.fromarray(tile)

       
        elif self.reader == 'openslide':
            l = self.dz.level_count - 1 - level
            img = self.dz.get_tile(l, (tx, ty))
            return img

    def save_tile(self, output_file, settings, tile_size, level, tx, ty):
        if self.reader == 'pytiff':
            iy = ty * tile_size
            ix = tx * tile_size
            for i, (marker, color, start, end) in enumerate(zip(
                settings['Channel Number'], settings['Color'],
                settings['Low'], settings['High']
            )):
                num_channels = self.get_shape()[0]
                page_base = level * num_channels
                self.io.set_page(page_base + int(marker))
                tile = self.io[iy:iy+tile_size, ix:ix+tile_size]
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
        'sample_info': {
            'rotation': 0,
            'name': '',
            'text': ''
        },
        'groups': [],
        'waypoints': [],
        'channels': [],
        'loaded': False,
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

def open_input_file(path=None):
    tiff_lock.acquire()
    if G['opener'] is None:
        G['opener'] = Opener(path if path else G['in_file'])
    tiff_lock.release()

@app.route('/')
def root():
    global G
    global YAML
    close_tiff()
    G, YAML = reset_globals()
    return app.send_static_file('index.html')

@app.route('/api/u16/<channel>/<level>_<x>_<y>.png')
@cross_origin()
def u16_image(channel, level, x, y):

    # Open the input file on the first request only
    if G['opener'] is None:
        open_input_file()

    img_io = render_tile(G['opener'], 1024, len(G['channels']),
                        int(level), int(x), int(y), int(channel))
    return send_file(img_io, mimetype='image/png')

@app.route('/api/out/<path:path>')
@cross_origin()
def out_image(path):
    image_path = os.path.join(G['out_dir'], path)
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/api/yaml', methods=['GET', 'POST'])
@cross_origin()
def api_yaml():
    yaml_text = yaml.dump({'Exhibit': YAML}, allow_unicode=True)
    response = make_response(yaml_text, 200)
    response.mimetype = "text/plain"
    return response


@app.route('/api/stories', methods=['POST'])
@cross_origin()
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
        for waypoint in d:
            yield {
                'Name': waypoint['name'],
                'Description': waypoint['text'],
                'Arrows': list(map(format_arrow, waypoint['arrows'])),
                'Overlays': list(map(format_overlay, waypoint['overlays'])),
                'Group': waypoint['group'],
                'Zoom': waypoint['zoom'],
                'Pan': waypoint['pan'],
            }

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

@app.route('/api/minerva/yaml', methods=['POST'])
@cross_origin()
def api_minerva_yaml():

    def make_yaml(d):
        for group in d:
            channels = group['channels']

            yield {
                'Path': group['id'],
                'Name': group['label'],
                'Colors': [c['color'] for c in channels],
                'Channels': [c['label'] for c in channels]
            }

    if request.method == 'POST':
        groups = request.json['groups']
        img = request.json['image']

        YAML['Groups'] = list(make_yaml(groups))
        if not img['url'].endswith('/'):
            img['url'] = img['url'] + '/'

        YAML['Images'] = [{
            'Name': 'i0',
            'Description': '',
            'Provider': 'minerva',
            'Path': img['url'] + img['id'] + '/prerendered-tile/',
            'Width': img['width'],
            'Height': img['height'],
            'MaxLevel': img['maxLevel']
        }]

        with open('out.yaml', 'w') as wf:
            yaml_text = yaml.dump({'Exhibit': YAML}, allow_unicode=True)
            wf.write(yaml_text)

        return 'OK'


@app.route('/api/save', methods=['POST'])
@cross_origin()
def api_save():
    
    if request.method == 'POST':
        data = request.json
        data['in_file'] = G['in_file']
        data['csv_file'] = G['csv_file']
        G['sample_info'] = data['sample_info']
        G['waypoints'] = data['waypoints']
        G['groups'] = data['groups']

        pickle.dump( data, open( G['out_dat'], 'wb' ) )

        return 'OK'
    

def render_progress_callback(current, max):
    G['save_progress'] = current
    G['save_progress_max'] = max

@app.route('/api/render/progress', methods=['GET'])
@cross_origin()
def get_render_progress():
    return jsonify({
        "progress": G['save_progress'],
        "max": G['save_progress_max']
    })

@app.route('/api/render', methods=['POST'])
@cross_origin()
def api_render():
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
        YAML['Images'][0]['Description'] = request.json['image']['description']

        with open(G['out_yaml'], 'w') as wf:
            yaml_text = yaml.dump({'Exhibit': YAML}, allow_unicode=True)
            wf.write(yaml_text)

        render_color_tiles(G['in_file'], G['opener'], G['out_dir'], 1024,
                           len(G['channels']), config_rows, progress_callback=render_progress_callback)

        return 'OK'

@app.route('/api/import', methods=['GET', 'POST'])
@cross_origin()
def api_import():
    if request.method == 'GET':

        return jsonify({
            'loaded': G['loaded'],
            'waypoints': G['waypoints'],
            'sample_info': G['sample_info'],
            'groups': G['groups'],
            'channels': G['channels'],
            'height': G['height'],
            'width': G['width'],
            'rgba': G['opener'].is_rgba() if G['opener'] else False
        })

    if request.method == 'POST':
        data = request.form
        input_file = pathlib.Path(data['filepath'])
        if not os.path.exists(input_file):
            return api_error(404, 'Image file not found: ' + str(input_file))

        if (input_file.suffix == '.dat'):
            saved = pickle.load( open( input_file, "rb" ) )
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
        else:
            csv_file = pathlib.Path(data['csvpath'])

        out_name = data['dataset']
        if out_name == '':
            out_name = 'out'

        out_name = out_name.replace(' ', '_')
        out_dir = os.path.join(input_file.parent, out_name)
        out_yaml = os.path.join(input_file.parent, out_name+'.yaml')
        out_dat = os.path.join(input_file.parent, out_name+'.dat')

        try:
            print("Opening file: ", str(input_file))

            if G['opener'] is None:
                open_input_file(str(input_file))


            (num_channels, num_levels, width, height) = G['opener'].get_shape()

            YAML['Images'] = [{
                'Name': 'i0',
                'Description': '',
                'Path': 'http://127.0.0.1:2020/api/out',
                'Width': width,
                'Height': height,
                'MaxLevel': num_levels - 1
            }]
            G['height'] = height
            G['width'] = width

        except Exception as e:
            print (e)
            return api_error(500, 'Invalid tiff file')

        def yield_labels(num_channels): 
            num_labels = 0
            try:
                with open(csv_file) as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        if num_labels < num_channels:
                            default = row.get('marker_name', str(num_labels))
                            yield row.get('Marker Name', default)
                            num_labels += 1
            except Exception as e:
                if (str(csv_file) != '.'):
                    return []
            while num_labels < num_channels:
                yield str(num_labels)
                num_labels += 1

        labels = list(yield_labels(num_channels))
        if labels == []:
            return api_error(500, "Error in opening marker csv file")

        if os.path.exists(input_file):
            G['out_yaml'] = str(out_yaml)
            G['out_dat'] = str(out_dat)
            G['out_dir'] = str(out_dir)
            G['in_file'] = str(input_file)
            G['csv_file'] = str(csv_file)
            G['channels'] = labels
            G['loaded'] = True

        return 'OK'

@app.route('/api/filebrowser', methods=['GET'])
@cross_origin()
def file_browser():
    """
    Browse local file system

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
            if not is_directory:
                stat_result = entry.stat()
                new_entry["size"] = stat_result.st_size
                new_entry["ctime"] = stat_result.st_ctime
                new_entry["mtime"] = stat_result.st_mtime
            response["entries"].append(new_entry)
        except PermissionError as e:
            pass

    return jsonify(response)

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
