import skimage.io
import pandas as pd
import numpy as np
import matplotlib.colors
from skimage.color import label2rgb


seg_mask = skimage.io.imread('../LUNG-3-PR_40X_Seg_labeled.tiff')
cluster4 = pd.read_csv('../LUNG-3-PR_clust4cellids.csv')

color_list = np.full_like(cluster4.clust_ID, '#000000', dtype=object)

def set_mask_color(input_color_list, full_cluster_ids, target_id, color):
    colored_mask = np.copy(input_color_list)
    colored_mask[full_cluster_ids == target_id] = color
    return colored_mask

# List of colors for the masks
material_colors = ['#F44336', '#673AB7', '#4CAF50', '#03A9F4']
# Set target cluster ID
for cluster_id in range(4)[:]:

    cluster_id += 1

    color_cluster1 = set_mask_color(
        color_list,
        cluster4.clust_ID,
        cluster_id,
        material_colors[cluster_id - 1]
    )

    c1img = label2rgb(seg_mask, colors=[matplotlib.colors.to_rgb(c) for c in color_cluster1], bg_label=0)

    c1img_max = c1img.max(axis=-1)
    c1img_alpha = np.zeros_like(c1img_max)
    c1img_alpha[c1img_max != 0] = 1

    skimage.io.imsave('cluster_{}_test.png'.format(cluster_id), np.dstack((c1img, c1img_alpha)))

# Conver png image into pyramid
import deepzoom_tile_cycif
from pathlib2 import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = 30000 * 30000

imgs = sorted(Path('./cluster_mask/all_cell_types_demo/full_img').rglob('cluster_*.png'))
if len(imgs) is 0:
    print('No images found.')

# Run deepzoom
for i in imgs:
    outpath = i.parent.parent / i.name.replace('.png', '')
    tiler = deepzoom_tile_cycif.DeepZoomStaticTiler(
        str(i), outpath, 'png', 1024, 0, False, 100, 8, False, True
    )
    tiler.run()