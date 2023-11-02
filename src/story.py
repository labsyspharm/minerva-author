import csv
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


def main(opener, channel_names):

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
    num_levels = opener.get_shape()[1]
    num_channels = opener.get_shape()[0]
    color_cycle = 'ffffff', 'ff0000', '00ff00', '0000ff'
    scale = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    level = num_levels - 1

    for gi, idx_start in enumerate(range(0, num_channels, 4), 1):
        idx_end = min(idx_start + 4, num_channels)
        channel_numbers = range(idx_start, idx_end)
        channel_defs = []
        for ci, color in zip(channel_numbers, color_cycle):
            print(
                f"analyzing channel {ci + 1}/{num_channels}", file=sys.stderr
            )
            img = opener.wrapper[level, :, :, 0, ci, 0]
            if img.min() < 0:
                print("  WARNING: Ignoring negative pixel values", file=sys.stderr)
            vmin, vmax = auto_threshold(img)
            vmin /= scale
            vmax /= scale
            channel_defs.append({
                "color": color,
                "id": ci,
                "label": channel_names[ci],
                "min": vmin,
                "max": vmax,
            })
        story["groups"].append({
            "label": f"Group {gi}",
            "channels": channel_defs,
        })

    return story
