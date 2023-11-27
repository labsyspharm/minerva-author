from tifffile.tifffile import TiffFileError
from story import main as auto_minerva
from app import Opener
import argparse
import pathlib
import json
import sys


def yield_numeric_labels(num_channels):
    for label_num in range(num_channels):
        yield str(label_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ome_tiff",
        metavar="ome_tiff",
        type=pathlib.Path,
        help="Input path to OME-TIFF with all channel groups",
    )
    parser.add_argument(
        "output_file",
        metavar="output_file",
        type=pathlib.Path,
        help="output file file to story.json with fit settings",
    )
    args = parser.parse_args()

    opener = None
    try:
        opener = Opener(args.ome_tiff)
        num_channels = opener.get_shape()[0]
        labels = yield_numeric_labels(num_channels)
        results = auto_minerva(opener, list(labels))
        print(results)
        with open(args.output_file, 'w') as wf:
            json.dump(results, wf)
    except (FileNotFoundError, TiffFileError) as e:
        print(f"Invalid ome-tiff file: cannot parse {ome_tiff}", file=sys.stderr)
