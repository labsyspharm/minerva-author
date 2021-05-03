import pathlib
import json
import os
import re
import logging
import argparse
from distutils import file_util
from distutils.errors import DistutilsFileError
from tifffile.tifffile import TiffFileError
from json.decoder import JSONDecodeError
from render_jpg import render_color_tiles
from storyexport import deduplicate_data
from app import make_groups, make_rows, make_stories
from app import extract_story_json_stem
from app import Opener

def render(opener, saved, output_dir, logger):
    config_rows = list(make_rows(saved['groups']))
    render_color_tiles(opener, output_dir, 1024, config_rows, logger, None, False)

def copy_vis_csv_files(waypoint_data, json_path, output_dir, vis_dir):
    input_dir = json_path.parent
    author_stem = extract_story_json_stem(json_path)
    vis_data_dir = vis_dir if vis_dir else f'{author_stem}-story-infovis'

    vis_path_dict_in = deduplicate_data(waypoint_data, input_dir / vis_data_dir)
    vis_path_dict_out = deduplicate_data(waypoint_data, output_dir / 'data')

    if not (output_dir / 'data').exists():
        (output_dir / 'data').mkdir(parents=True)

    # Copy the visualization csv files to a "data" directory
    for key_path, in_path in vis_path_dict_in.items():
        if pathlib.Path(in_path).suffix in ['.csv']:
            try:
                out_path = vis_path_dict_out[key_path]
                file_util.copy_file(in_path, out_path)
            except DistutilsFileError as e:
                print(f'Cannot copy {in_path}')
                print(e)
        else:
            print(f'Refusing to copy non-csv infovis: {in_path}')

def set_if_not_none(exhibit, key, value):
    if value is not None:
        exhibit[key] = value
    return exhibit

def make_exhibit_config(opener, root_url, saved):

    waypoint_data = saved['waypoints']
    (num_channels, num_levels, width, height) = opener.get_shape()
    vis_path_dict = deduplicate_data(waypoint_data, 'data')

    exhibit = {
        'Images': [{
            'Name': 'i0',
            'Description': saved['sample_info']['name'],
            'Path': root_url if root_url else '.',
            'Width': width,
            'Height': height,
            'MaxLevel': num_levels - 1
        }],
        'Header': saved['sample_info']['text'],
        'Rotation': saved['sample_info']['rotation'],
        'Layout': {'Grid': [['i0']]},
        'Stories': make_stories(waypoint_data, [], vis_path_dict),
        'Groups': list(make_groups(saved['groups'])),
        'Masks': []
    }
    first_group = saved['sample_info'].get('first_group', None)
    default_group = saved['sample_info'].get('default_group', None)
    pixels_per_micron = saved['sample_info'].get('pixels_per_micron', None)
    exhibit = set_if_not_none(exhibit, 'PixelsPerMicron', pixels_per_micron)
    exhibit = set_if_not_none(exhibit, 'DefaultGroup', default_group)
    exhibit = set_if_not_none(exhibit, 'FirstGroup', first_group)

    return exhibit

def main(ome_tiff, author_json, output_dir, root_url, vis_dir, force=False):
    FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('app')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)

    opener = None
    saved = None

    try:
        opener = Opener(ome_tiff)
    except (FileNotFoundError, TiffFileError) as e:
        logger.error(e)
        logger.error(f'Invalid ome-tiff file: cannot parse {ome_tiff}')
        return

    try:
        with open(author_json) as json_file:
            saved = json.load(json_file)
        groups = saved['groups']
    except (FileNotFoundError, JSONDecodeError, KeyError) as e:
        logger.error(e)
        logger.error(f'Invalid save file: cannot parse {author_json}')
        return

    if not force and os.path.exists(output_dir):
        logger.error(f'Refusing to overwrite output directory {output_dir}')
        return
    elif force and os.path.exists(output_dir):
        logger.warning(f'Writing to existing output directory {output_dir}')

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    exhibit_config = make_exhibit_config(opener, root_url, saved)
    copy_vis_csv_files(saved['waypoints'], author_json, output_dir, vis_dir)

    with open(output_dir / 'exhibit.json', 'w') as wf:
        json.dump(exhibit_config, wf)

    render(opener, saved, output_dir, logger)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ome_tiff", metavar="ome_tiff", type=pathlib.Path,
        help="Input path to OME-TIFF with all channel groups",
    )
    parser.add_argument(
        "author_json", metavar="author_json", type=pathlib.Path,
        help="Input Minerva Author save file with channel configuration",
    )
    parser.add_argument(
        "output_dir", metavar="output_dir", type=pathlib.Path,
        help="Output directory for exhibit and rendered JPEG pyramid",
    )
    parser.add_argument(
        "--url", metavar="url", default=None,
        help="URL to planned hosting location of rendered JPEG pyramid",
    )
    parser.add_argument(
        "--vis", metavar="vis",
        type=pathlib.Path, default=None,
        help="Input data visualization directory (default constructed from author .json)",
    )
    parser.add_argument('--force', help='Overwrite output', action='store_true')
    args = parser.parse_args()

    ome_tiff = args.ome_tiff
    author_json = args.author_json
    output_dir = args.output_dir
    root_url = args.url
    vis_dir = args.vis
    force = args.force

    main(ome_tiff, author_json, output_dir, root_url, vis_dir, force)
