import pathlib
import sys
import os
import yaml
import pytiff
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file 
from flask import make_response
from render_png import render_tiles
from render_jpg import render_color_tiles

G = {
    'u16_dir': None,
    'in_file': None,
    'channels': 0,
    'loaded': False
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

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/u16/<path:path>')
def u16_image(path):
    image_path = os.path.join(G['u16_dir'], path)
    return send_file(image_path, mimetype='image/png')

@app.route('/api/yaml', methods=['GET', 'POST'])
def api_yaml():
    yaml_text = yaml.dump({'Exhibit': YAML}, allow_unicode=True)
    response = make_response(yaml_text, 200)
    response.mimetype = "text/plain"
    return response

@app.route('/api/render', methods=['GET', 'POST'])
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
                    'Low': channel['range']['min'],
                    'High': channel['range']['max'],
                    'Color': '#' + channel['color'],
                }
    
    if request.method == 'POST':
        data = request.json['groups']
        config_rows = list(make_rows(data))
        YAML['Groups'] = list(make_yaml(data))

        render_color_tiles(G['in_file'], G['out_dir'], 1024,
                           G['channels'], config_rows)
        
        return 'OK'

@app.route('/api/import', methods=['GET', 'POST'])
def api_import():
    if request.method == 'GET':

        return jsonify({
            'loaded': G['loaded'],
            'channels': G['channels']
        })

    if request.method == 'POST':
        data = request.form
        input_file = pathlib.Path(data['filepath'])
        u16_dir = os.path.join(input_file.parent, 'u16')
        out_dir = os.path.join(input_file.parent, 'out')

        num_channels = 0

        with pytiff.Tiff(str(input_file)) as handle:
            for page in handle.pages:
                if handle.shape == page.shape:
                    num_channels += 1
    
            num_levels = handle.number_of_pages // num_channels

            YAML['Images'] = [{
                'Name': 'i0',
                'Description': '',
                'Path': 'http://localhost:8000',
                'Width': handle.shape[1],
                'Height': handle.shape[0],
                'MaxLevel': num_levels - 1
            }]

        if os.path.exists(input_file):
            # render_tiles(input_file, u16_dir, 1024, num_channels)
            G['out_dir'] = str(out_dir)
            G['u16_dir'] = str(u16_dir)
            G['in_file'] = str(input_file)
            G['channels'] = num_channels
            G['loaded'] = True

        return 'OK'


if __name__ == '__main__':
    app.run(debug=True, port=2020)
