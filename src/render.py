import argparse
import json
import logging
import csv
import os
import re
import pathlib
import threading
import concurrent.futures
from itertools import cycle
from distutils import file_util
from distutils.errors import DistutilsFileError
from json.decoder import JSONDecodeError

import numpy as np
from PIL import Image
import tifffile as tiff
from matplotlib import colors
from tifffile.tifffile import TiffFileError

from thumbnail import find_group_tiles
from thumbnail import merge_tiles_and_save_image
from app import Opener, extract_story_json_stem, make_channels, make_groups, make_rows, make_stories, make_mask_rows
from storyexport import deduplicate_data, copy_vega_csv
from render_jpg import render_color_tiles, composite_channel
from render_png import render_u32_tiles


def json_to_html(exhibit):
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en-US">\n'
        '\n'
        '    <head>\n'
        '        <meta charset="utf-8">\n'
        '        <meta http-equiv="X-UA-Compatible" content="IE=edge">\n'
        '        <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '    </head>\n'
        '\n'
        '    <body>\n'
        '        <div id="minerva-browser" style="position: absolute; top: 0; left: 0; height: 100%; width: 100%;"></div>\n'
        '        <script defer src="https://use.fontawesome.com/releases/v5.2.0/js/all.js" integrity="sha384-4oV5EgaV02iISL2ban6c/RmotsABqE4yZxZLcYMAdG7FAPsyHYAPpywE9PJo+Khy" crossorigin="anonymous"></script>\n'
        '        <script src="https://cdn.jsdelivr.net/npm/minerva-browser@3.20.0/build/bundle.js"></script>\n'
        '        <script>\n'
        '         window.viewer = MinervaStory.default.build_page({\n'
                     f'exhibit: {exhibit},\n'
        '             id: "minerva-browser",\n'
        '             embedded: true,\n'
        '             markerData: [],\n'
        '             cellTypeData: [],\n'
        '         });\n'
        '        </script>\n'
        '    </body>\n'
        '\n'
        '</html>'
    )


def render(opener, saved, output_dir, rgba, n_threads, logger):

    threads = []

    print(f'Using {n_threads} threads')

    config_rows = list(make_rows(saved["groups"], rgba))

    for thread in range(n_threads):
        th_args = (opener, output_dir, opener.to_tsize(), config_rows, logger, None, False)
        th_kwargs = {'thread': thread,'n_threads': n_threads}
        th = threading.Thread(target=render_color_tiles, args=th_args, kwargs=th_kwargs)
        threads.append(th)

    for th in threads:
        th.start()

    for th in threads:
        th.join()

    print(f'{n_threads} threads complete')

def copy_vis_csv_files(waypoint_data, json_path, output_dir, vis_dir):
    input_dir = json_path.parent
    author_stem = extract_story_json_stem(json_path)
    vis_data_dir = vis_dir if vis_dir else f"{author_stem}-story-infovis"

    vis_path_dict_in = deduplicate_data(waypoint_data, input_dir / vis_data_dir)
    vis_path_dict_out = deduplicate_data(waypoint_data, output_dir / "data")

    if not (output_dir / "data").exists():
        (output_dir / "data").mkdir(parents=True)

    # Copy the visualization csv files to a "data" directory
    for key_path, in_path in vis_path_dict_in.items():
        if pathlib.Path(in_path).suffix in [".csv"]:
            try:
                out_path = vis_path_dict_out[key_path]
                # Modify matrix CSV files if needed
                copy_vega_csv(waypoint_data, in_path, out_path)
            except DistutilsFileError as e:
                print(f"Cannot copy {in_path}")
                print(e)
        else:
            print(f"Refusing to copy non-csv infovis: {in_path}")


def set_if_not_none(exhibit, key, value):
    if value is not None:
        exhibit[key] = value
    return exhibit


def make_exhibit_config(in_shape, root_url, saved, rgba):

    levels = in_shape['levels']
    height = in_shape['height']
    width = in_shape['width']
    waypoint_data = saved["waypoints"]
    vis_path_dict = deduplicate_data(waypoint_data, "data")

    exhibit = {
        "Images": [
            {
                "Name": "i0",
                "Description": saved["sample_info"]["name"],
                "Path": root_url if root_url else ".",
                "Width": width,
                "Height": height,
                "MaxLevel": levels - 1,
            }
        ],
        "Header": saved["sample_info"]["text"],
        "Rotation": saved["sample_info"]["rotation"],
        "Layout": {"Grid": [["i0"]]},
        "Stories": make_stories(waypoint_data, [], vis_path_dict),
        "Channels": list(make_channels(saved["groups"], rgba)),
        "Groups": list(make_groups(saved["groups"])),
        "Masks": [],
    }
    first_group = saved.get("first_group", None)
    default_group = saved.get("default_group", None)
    first_viewport = saved.get("first_viewport", None)
    pixels_per_micron = saved["sample_info"].get("pixels_per_micron", None)
    exhibit = set_if_not_none(exhibit, "PixelsPerMicron", pixels_per_micron)
    exhibit = set_if_not_none(exhibit, "FirstViewport", first_viewport)
    exhibit = set_if_not_none(exhibit, "DefaultGroup", default_group)
    exhibit = set_if_not_none(exhibit, "FirstGroup", first_group)

    return exhibit


def to_one_tile(one_tile, settings):

    (height, width) = one_tile.shape[:2]
    target = np.zeros((height, width, 3), np.float32)

    for i, (marker, color, start, end) in enumerate(
        zip(
            settings["Channel Number"],
            settings["Color"],
            settings["Low"],
            settings["High"],
        )
    ):
        tile = one_tile[:, :, int(marker)]

        if np.issubdtype(tile.dtype, np.unsignedinteger):
            iinfo = np.iinfo(tile.dtype)
            start *= iinfo.max
            end *= iinfo.max

        composite_channel(
            target, tile, colors.to_rgb(color), float(start), float(end)
        )

    np.clip(target, 0, 1, out=target)
    target_u8 = np.rint(target * 255).astype(np.uint8)
    return Image.frombytes("RGB", target.T.shape[1:], target_u8.tobytes())


def render_one_tile(one_tile, output_dir, config_rows):
    print("    level {} ({} x {})".format(0, 0, 0))
    filename = "{}_{}_{}.{}".format(0, 0, 0, "jpg")

    output_path = pathlib.Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for settings in config_rows:
        group_dir = settings["Group Path"]
        if not (output_path / group_dir).exists():
            (output_path / group_dir).mkdir(parents=True)
        output_file = str(output_path / group_dir / filename)
        img = to_one_tile(one_tile, settings)
        img.save(output_file, quality=85)


def main(ome_tiff, author_json, output_dir, root_url, vis_dir, mask_tiff, mask_map, n_threads, force=False):
    FORMATTER = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("app")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)

    one_tile = None
    opener = None
    saved = None

    try:
        opener = Opener(ome_tiff)
    except (FileNotFoundError, TiffFileError) as e:
        logger.error(e)
        logger.error(f"Invalid ome-tiff file: cannot parse {ome_tiff}")
        return

    in_shape = None
    # Treat as static tiff
    if opener.reader is None:
        print('Opening single tile plain .tif')
        one_tile = tiff.imread(ome_tiff)
        in_shape = {
            'levels': 1,
            'height': one_tile.shape[0],
            'width': one_tile.shape[1],
        }
    else:
        (levels, width, height) = opener.get_shape()[1:]
        in_shape = {
            'levels': levels,
            'height': height,
            'width': width,
        }

    try:
        with open(author_json) as json_file:
            saved = json.load(json_file)
    except (FileNotFoundError, JSONDecodeError, KeyError) as e:
        logger.error(e)
        logger.error(f"Invalid save file: cannot parse {author_json}")
        return

    if not force and os.path.exists(output_dir):
        logger.error(f"Refusing to overwrite output directory {output_dir}")
        return
    elif force and os.path.exists(output_dir):
        logger.warning(f"Writing to existing output directory {output_dir}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    rgba = opener.rgba
    exhibit_config = make_exhibit_config(in_shape, root_url, saved, rgba)
    copy_vis_csv_files(saved["waypoints"], author_json, output_dir, vis_dir)

    if opener.reader is None:
        config_rows = list(make_rows(saved["groups"], rgba))
        render_one_tile(one_tile, output_dir, config_rows)
    else:
        if mask_tiff is not None:
            mask_id_lists = dict()
            with open(mask_map, encoding="utf-8-sig") as cf:
                for row in csv.DictReader(cf):
                    mask_label = re.sub('[/\\\]', '-', row['State'])
                    mask_id_list = mask_id_lists.get(mask_label, [])
                    mask_id_list.append(int(row['CellID']))
                    mask_id_lists[mask_label] = mask_id_list
            # Find matching IDs for the saved masks
            mask_config = [
                {
                    'label': mask_item["label"],
                    'path': mask_tiff,
                    'channels': [{
                        'color': mask_item["color"],
                        'ids': mask_id_lists[mask_item["original_label"]],
                        'opacity': mask_item.get("opacity", 1)
                     }]
                }
                for mask_item in saved.get("masks", [])
                if mask_item["original_label"] in mask_id_lists
            ]
            exhibit_config['Masks'] = [
                {
                    "Path": config['label'],
                    "Name": config['label'],
                    "Channels": [config['label']],
                    "Colors": [config['channels'][0]['color']]
                }
                for config in mask_config
            ]
            mask_config_rows = [
                {**config, 'images': [ image ]}
                for config in list(
                    make_mask_rows(output_dir, mask_config, None)
                )
                for image in config['images']
            ]
            def render_mask(mask_params):
                mask_out_path = mask_params['images'][0]['out_path']
                ids = mask_params['images'][0]['settings']['channels'][0]['ids']
                print(
                    f'{mask_out_path}: Rendering {len(ids)} segmentation IDs'
                )
                render_u32_tiles(
                    mask_params, opener.to_tsize(), logger 
                )
            with concurrent.futures.ThreadPoolExecutor(n_threads) as pool:
                list(pool.map(render_mask, mask_config_rows))

        with open(output_dir / "exhibit.json", "w") as wf:
            json.dump(exhibit_config, wf)

        with open(output_dir / "index.html", "w") as wf:
            exhibit_string = json.dumps(exhibit_config)
            wf.write(json_to_html(exhibit_string))

        render(opener, saved, output_dir, rgba, n_threads, logger)

        # Render thumbnail
        groups = exhibit_config["Groups"]
        if len(groups) > 0:
            group = exhibit_config.get("FirstGroup", groups[0]["Name"])
            tiles = find_group_tiles(output_dir, output_dir / "exhibit.json", group)
            merge_tiles_and_save_image(output_dir, tiles)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ome_tiff",
        metavar="ome_tiff",
        type=pathlib.Path,
        help="Input path to OME-TIFF with all channel groups",
    )
    parser.add_argument(
        "author_json",
        metavar="author_json",
        type=pathlib.Path,
        help="Input Minerva Author save file with channel configuration",
    )
    parser.add_argument(
        "output_dir",
        metavar="output_dir",
        type=pathlib.Path,
        help="Output directory for exhibit and rendered JPEG pyramid",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        metavar="threads",
        help="Number of threads to use rendering the JPEG pyramid",
    )
    parser.add_argument(
        "--url",
        metavar="url",
        default=None,
        help="URL to planned hosting location of rendered JPEG pyramid",
    )
    parser.add_argument(
        "--mask",
        metavar="mask",
        type=pathlib.Path,
        default=None,
        help="Input segmentation mask image file",
    )
    parser.add_argument(
        "--mask-ids",
        metavar="mask_ids",
        type=pathlib.Path,
        default=None,
        help="Input segmentation mask ID csv file",
    )
    parser.add_argument(
        "--vis",
        metavar="vis",
        type=pathlib.Path,
        default=None,
        help="Input data visualization directory (default constructed from author .json)",
    )
    parser.add_argument("--force", help="Overwrite output", action="store_true")
    args = parser.parse_args()

    ome_tiff = args.ome_tiff
    author_json = args.author_json
    output_dir = args.output_dir
    mask_tiff = args.mask
    mask_map = args.mask_ids
    n_threads = args.threads
    root_url = args.url
    vis_dir = args.vis
    force = args.force

    if mask_tiff != None:
        assert mask_map != None

    main(ome_tiff, author_json, output_dir, root_url, vis_dir, mask_tiff, mask_map, n_threads, force)
