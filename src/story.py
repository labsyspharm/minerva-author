import concurrent.futures
import csv
import itertools
import json
import numpy as np
import ome_types
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
    # Pair each iterable entry with a color
    return zip(iterable, itertools.cycle(colors))


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

    def threshold(ci):
        img = opener.wrapper[level, :, :, 0, ci, 0]
        if img.min() < 0:
            print(
                f"  WARNING: Ignoring negative pixel values in channel {ci}",
                file=sys.stderr,
            )
        res = auto_threshold(img)
        return res

    with concurrent.futures.ThreadPoolExecutor(n_workers) as pool:
        thresholds = list(pool.map(threshold, range(n_channels)))

    channel_groups = group_channels(n_channels, channel_names)
    output_groups = {}
    for gi, channel_range in channel_groups.items():
        group = {}
        for ci, color in color_cycle(channel_range):
            vmin, vmax = thresholds[ci]
            group[ci] = {
                "color": color,
                "id": ci,
                "label": channel_names[ci],
                "min": vmin / scale,
                "max": vmax / scale,
            }
        output_groups[gi] = group
   
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
