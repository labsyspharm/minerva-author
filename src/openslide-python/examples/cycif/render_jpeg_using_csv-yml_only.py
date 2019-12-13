from __future__ import print_function, division
import sys
import itertools
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import math
import pytiff
from matplotlib import colors
import csv
import json
import yaml
from collections import OrderedDict
from time import time


TILE_SIZE = 1024
EXT = 'jpg'
num_channels = 44
csv_path = './marker_settings_20191016.csv'

input_file_dir = pathlib.Path('P:/datasets/natureprotocols/LUNG-3-PR/')
input_filepaths = sorted(input_file_dir.rglob('LUNG-3-PR_40X.ome.tif'))


def parse_clinical_info():
    clinical_info_dict = {}
    with open(clinical_info_csv_path) as clinical_info:
        reader = csv.DictReader(clinical_info)
        clinical_info_rows = [dict(row) for row in reader]
        for i in clinical_info_rows:
            clinical_info_dict[i['CycifID']] = i
    return clinical_info_dict

def calculate_pan_x(width, height):
    return width / float(height * 2)

def main():

    for input_file_path in input_filepaths[:]:
        
        input_filename = input_file_path.name.split('.')[0]
        input_filename = input_filename.replace('_', '')

        print('Processing:', str(input_file_path))
        time_start = time()
        
        tiff = pytiff.Tiff(str(input_file_path))
        
        # Deidentify IDs
        output_filename = input_filename
        
        output_path = pathlib.Path('rendered') / output_filename
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

            channel_groups = []
            for name, settings in render_groups.items():

                channels = '--'.join(
                    settings['Channel Number'][i] + '__' + settings['Marker Name'][i]
                    for i in range(len(settings['Color']))
                )

                channel_groups += [{
                    'Path': name.replace(' ', '-') + '_' + channels,
                    'Name': name,
                    'Colors': [colors.to_hex(c)[1:] for c in settings['Color']],
                    'Channels': settings['Marker Name'],
                }]

            json_str = ''
            with open('default_json.json') as f:
                json_str = f.read()
            
            pan_x = calculate_pan_x(tiff.shape[1], tiff.shape[0])
            json_str = (json_str
                .replace('placeholder_slide_name', output_filename)
                .replace(
                    'placeholder_exhibit_name', output_filename
                    )
                .replace('"placeholder_width"', str(tiff.shape[1]))
                .replace('"placeholder_height"', str(tiff.shape[0]))
                .replace('"placeholder_max_level"', str(num_levels - 1))
                .replace('placeholder_first_ch_group', channel_groups[0]['Name'])
                .replace('"placeholder_pan_x"', str(pan_x))
                .replace('"placeholder_pan_y"', str(0.5))
            )

            out_json = json.loads(json_str, object_pairs_hook=OrderedDict)
            out_json['Exhibit']['Groups'] = channel_groups
            out_json['Exhibit']['Layout']['Grid'] = [['i0']]
            with open(str(output_path / (output_filename + '.json')), 'w') as outfile:
                json.dump(out_json, outfile)

            with open(str(output_path / (output_filename + '.yml')), 'w') as outfile:
                yaml.SafeDumper.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
                yaml.safe_dump(out_json, outfile, default_flow_style=False, canonical=False)
            
        print('Used', str(int(time() - time_start)), 'sec for', output_filename)


if __name__ == '__main__':
    main()




    

    