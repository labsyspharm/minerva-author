import csv
import json
import numpy as np
import ome_types
import threading
import sklearn.mixture
import sys
import tifffile
import zarr
import argparse
import os

def auto_threshold(img):

    assert img.ndim == 2

    yi, xi = np.floor(np.linspace(0, img.shape, 200, endpoint=False)).astype(int).T
    # Slice one dimension at a time. Should generally use less memory than a meshgrid.
    img = img[yi]
    img = img[:, xi]
    img_log = np.log(img[img > 0])
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    _, i1, i2 = np.argsort(means)
    mean1, mean2 = means[[i1, i2]]
    std1, std2 = gmm.covariances_[[i1, i2], 0, 0] ** 0.5

    x = np.linspace(mean1, mean2, 50)
    n_pdf = lambda m,s,x: np.exp(-0.5 * ((x - m) / s)**2) / (s * np.sqrt(2*np.pi))
    y1 = n_pdf(mean1, std1, x) * gmm.weights_[i1]
    y2 = n_pdf(mean2, std2, x) * gmm.weights_[i2]

    lmax = mean2 + 2 * std2
    lmin = x[np.argmin(np.abs(y1 - y2))]
    if lmin >= mean2:
        lmin = mean2 - 2 * std2
    vmin = max(np.exp(lmin), img.min(), 0)
    vmax = min(np.exp(lmax), img.max())

    return vmin, vmax

def to_heuristic(channel_names, step_size):
    names = channel_names[::step_size]
    # Preferences from most to least important
    # Few trigrams, few bigrams, low modulo
    return (
        len(set([ name[:3] for name in names ]))/len(names),
        len(set([ name[:2] for name in names ]))/len(names),
        len(channel_names) % step_size
    )

def group_channels(n_channels, channel_names):
    size_options = [3, 4, 5, 6]
    # Group with fewest initial ngrams
    size_stats = sorted([
        (*to_heuristic(channel_names, size), size)
        for size in size_options 
    ])
    group_size = size_stats[0][-1]
    # Groups must be at least half size
    min_size = 2 + group_size // 2
    group_starts = list(range(0, n_channels, group_size))
    group_iter = zip(group_starts, group_starts[1:]+[None])
    channel_groups = []
    for gi, pair in enumerate(group_iter):
        channel_range = list(range(n_channels)[slice(*pair)])
        if gi > 0 and len(channel_range) < min_size:
            last_range = channel_groups[-1][-1]
            last_range += channel_range
            continue
        channel_groups.append([gi, channel_range])
    return dict(channel_groups)


def color_cycle(iterable):
    colors = (
        'ffffff', 'ff0000', '00ff00', '0000ff',
        'ff00ff', 'ffff00', '00ffff'
    )
    def cycle():
        i = 0
        while True:
            yield colors[i % len(colors)]
            i += 1
    # Pair each iterable entry with a color
    return zip(iterable, cycle())


def main(opener, channel_names, n_workers=1):

    try:
        metadata = opener.read_metadata()
        pixels = metadata.images[0].pixels
        pixel_microns = pixels.physical_size_x_quantity.to('um').m
        pixels_per_micron = 1/pixel_microns if pixel_microns > 0 else 0
    except:
        print(f"Unable to read metadata", file=sys.stderr)
        pixels_per_micron = None

    story = {
        "sample_info": {
            "name": "",
            "rotation": 0,
            "text": "",
            "pixels_per_micron": pixels_per_micron,
        },
        "groups": [],
        "waypoints": [],
    }

    dtype = opener.default_dtype
    n_levels = opener.get_shape()[1]
    n_channels = opener.get_shape()[0]
    scale = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    level = n_levels - 1

    output_groups = dict()

    def record_thresholds(*args_list):
        for recorder, thresholder in args_list:
            recorder(thresholder())

    def to_recorder(gi, ci):
        def recorder(channel):
            group = output_groups.get(gi, {})
            group[ci] = channel
            output_groups[gi] = group
        return recorder

    def to_thresholder(ci, color):
        img = opener.wrapper[level, :, :, 0, ci, 0]
        if img.min() < 0:
            print("  WARNING: Ignoring negative pixel values", file=sys.stderr)
        def thresholder():
            vmin, vmax = auto_threshold(img)
            return {
                "color": color,
                "id": ci,
                "label": channel_names[ci],
                "min": vmin / scale,
                "max": vmax / scale,
            }
        return thresholder

    channel_groups = group_channels(n_channels, channel_names) 
    group_args = [
        (to_recorder(gi, ci), to_thresholder(ci, color))
        for gi, channel_range in channel_groups.items()
        for ci, color in color_cycle(channel_range)
    ]
    n_groups = len(channel_groups)
    n_workers = min(n_workers, n_groups)
    multi_group_args = [
        group_args[i::n_workers]
        for i in range(len(group_args) // n_workers)
    ]
    group_sizes = set(
        len(channel_range) for channel_range in channel_groups.values()
    )
    group_range = '-'.join([
        str(size) for size in sorted(group_sizes)
    ])
    print(f'''Auto-Grouping:
    {n_groups} groups of {group_range} channels
    over {n_workers} threads
    ''')

    threads = [
        threading.Thread(target=record_thresholds, args=args)
        for args in multi_group_args
    ]
    for th in threads:
        th.start()

    for th in threads:
        th.join()
   
    auto_groups = [
        output_groups[gi] for gi in
        sorted(channel_groups.keys())
    ]

    for gi, val in enumerate(auto_groups):
        story["groups"].append({
            "label": f"Group {gi+1}",
            "channels": [
                v for _,v in sorted(val.items())
            ]
        })

    return story
