import os
import json
import argparse
import numpy as np
import skimage as ski

def linear_blend(images):
    float_images = [
        (color, ski.img_as_float(i))
        for (_, color, i) in images
    ]
    # Use first channel to define output
    (first_color, first_float_image) = float_images[0]
    first_float_color = parse_hex_color(first_color)
    float_out = first_float_color * first_float_image
    # Linear blending
    for (color, float_image) in float_images[1:]:
        float_color = parse_hex_color(color)
        float_out += float_color * float_image
    # Uint8 output
    out = np.clip(float_out, 0, 1)
    return ski.img_as_ubyte(out)

def parse_hex_color(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(
        int(hex_color[i:i+2], 16) / 255
        for i in (0, 2, 4)
    )

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def load_jpeg(filename):
    return ski.io.imread(filename)

def save_jpeg(filename, image):
    ski.io.imsave(filename, image)

def to_path_zoom(name):
    zoom = name.split('_')[0]
    try:
        return int(zoom)
    except ValueError:
        pass
    return -1

def find_jpegs(folder):
    return [
        filename for filename in os.listdir(folder)
        if filename.endswith('.jpeg')
        or filename.endswith('.jpg')
    ]

def find_top_tiles(folder, channels):
    for (channel, name, color) in channels:
        subdir = os.path.join(folder, channel)
        names = [
            name for name in find_jpegs(subdir)
            if to_path_zoom(name) >= 0
        ]
        names.sort(key=lambda name: to_path_zoom(name))
        filename = os.path.join(subdir, names[-1])
        thumb_tile = load_jpeg(filename)
        yield (name, color, thumb_tile)

def find_group_channels(exhibit, group):
    ex = load_json(exhibit)
    chan_paths = {
        channel['Name']: channel['Path']
        for channel in ex.get('Channels', [])
        if not channel['Rendered']
    }
    for item in ex['Groups']:
        if item['Name'] == group:
            chan_cols = list(
                zip(item['Channels'], item['Colors'])
            )
            return [
                (chan_paths.get(c[0], None), c[0], c[1])
                for c in chan_cols if c[0]
            ]
    return []

def find_group_tiles(root, exhibit, group):
    channel_list = find_group_channels(exhibit, group)
    return list(find_top_tiles(root, channel_list))

def name_blend(tiles):
    return '-'.join([
        '_'.join([name, color])
        for (name, color, _) in tiles
    ])+'.jpg'

def merge_tiles_and_save_image(root, tiles):
    image_name = name_blend(tiles)
    image_file = os.path.join(root, image_name)
    image = linear_blend(tiles)
    # Make output directory and save image
    os.makedirs(root, exist_ok=True)
    save_jpeg(image_file, image)
    print(f'Saved image size {image.shape} to {image_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--group', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    tiles = find_group_tiles(args.root, args.json, args.group)
    merge_tiles_and_save_image(args.output, tiles)
