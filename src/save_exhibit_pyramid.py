import pathlib
import json
import os
import re
import logging
import argparse
from json.decoder import JSONDecodeError
from render_jpg import render_color_tiles
from app import make_groups, make_rows, make_stories
from app import Opener

def render(opener, saved, output_dir, logger):
    config_rows = list(make_rows(saved['groups']))
    render_color_tiles(opener, output_dir, 1024, config_rows, logger, None, False)

def make_exhibit_config(opener, root_url, saved):

    (num_channels, num_levels, width, height) = opener.get_shape()

    return {
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
        'Stories': make_stories(saved['waypoints'], [], True),
        'Groups': list(make_groups(saved['groups'])),
        'Masks': []
    }

def main(ome_tiff, author_json, output_dir, root_url, force=False):
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
       logger.error(f'Invalid save file: cannot parse {json_file}')
       return

   if not force and os.path.exists(output_dir):
      logger.error(f'Refusing to overwrite output directory {output_dir}')
      return
   elif force and os.path.exists(output_dir):
      logger.warning(f'Writing to existing output directory {output_dir}')

   output_path = pathlib.Path(output_dir)
   if not output_path.exists():
        output_path.mkdir(parents=True)

   exhibit_config = make_exhibit_config(opener, root_url, saved)

   with open(output_dir / 'exhibit.json', 'w') as wf:
       json_text = json.dumps(exhibit_config, ensure_ascii=False)
       wf.write(json_text)

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
        help="Output directory for rendered JPEG pyramid",
    )
    parser.add_argument(
        "--url", metavar="url", default=None,
        help="URL to planned hosting location of rendered JPEG pyramid",
    )
    parser.add_argument('--force', help='Overwrite output', action='store_true')
    args = parser.parse_args()

    ome_tiff = args.ome_tiff
    author_json = args.author_json
    output_dir = args.output_dir
    root_url = args.url
    force = args.force

    main(ome_tiff, author_json, output_dir, root_url, force)
