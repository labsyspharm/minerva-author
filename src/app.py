import pathlib
import sys
import os
import csv
import yaml
import pytiff
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file 
from flask import make_response
from render_png import render_tile
from render_png import render_tiles
from render_jpg import render_color_tiles
from flask_cors import CORS, cross_origin


G = {
    'in_file': None,
    'channels': [],
    'loaded': False,
    'height': 1024,
    'width': 1024
}
YAML = {
    'Images': [],
    'Layout': {'Grid': [['i0']]},
    'Groups': []
}

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder at _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


app = Flask(__name__,
            static_folder=resource_path('static'),
            static_url_path='')

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/u16/<channel>/<level>_<x>_<y>.png')
@cross_origin()
def u16_image(channel, level, x, y):
    img_io = render_tile(G['in_file'], 1024, len(G['channels']),
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
            'Point': a
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



@app.route('/api/render', methods=['POST'])
@cross_origin()
def api_render():

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

        with open(G['out_yaml'], 'w') as wf:
            yaml_text = yaml.dump({'Exhibit': YAML}, allow_unicode=True)
            wf.write(yaml_text)

        render_color_tiles(G['in_file'], G['out_dir'], 1024,
                           len(G['channels']), config_rows)

        return 'OK'

@app.route('/api/import', methods=['GET', 'POST'])
@cross_origin()
def api_import():
    if request.method == 'GET':

        return jsonify({
            'loaded': G['loaded'],
            'channels': G['channels'],
            'height': G['height'],
            'width': G['width']
        })

    if request.method == 'POST':
        data = request.form
        csv_file = pathlib.Path(data['csvpath'])
        input_file = pathlib.Path(data['filepath'])
        out_dir = os.path.join(input_file.parent, 'out')
        out_yaml = os.path.join(input_file.parent, 'out.yaml')

        num_channels = 0

        with pytiff.Tiff(str(input_file), encoding='utf-8') as handle:

            for page in handle.pages:
                if handle.shape == page.shape:
                    num_channels += 1
    
            num_levels = handle.number_of_pages // num_channels

            YAML['Images'] = [{
                'Name': 'i0',
                'Description': '',
                'Path': 'http://localhost:2020/api/out',
                'Width': handle.shape[1],
                'Height': handle.shape[0],
                'MaxLevel': num_levels - 1
            }]
            G['height'] = handle.shape[0]
            G['width'] = handle.shape[1]

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
                pass
            while num_labels < num_channels:
                yield str(num_labels)
                num_labels += 1

        labels = list(yield_labels(num_channels))

        if os.path.exists(input_file):
            G['out_yaml'] = str(out_yaml)
            G['out_dir'] = str(out_dir)
            G['in_file'] = str(input_file)
            G['channels'] = labels
            G['loaded'] = True

        return 'OK'


if __name__ == '__main__':
    app.run(debug=True, port=2020)
