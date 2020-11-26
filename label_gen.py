import os
import os.path
from argparse import ArgumentParser

import numpy as np
from PIL import Image

def parse_slice_name(name):
    """Assume name has form <name>_<loc_axis>_<loc>.<ext>"""
    # Remove extension if needed
    parts = name.split('.')
    if len(parts) > 1:
        name = parts[-2]

    # Get parts
    parts = name.split('_')
    return parts[-2], int(parts[-1])


def insert_slices_from_files(vol_size, files):
    labels = np.zeros(vol_size, dtype=np.uint8)

    label_map = {}
    max_label = 0

    for f in files:
        loc = parse_slice_name(f)[1]  # For now, ignore axes part

        slice_ = Image.open(os.path.join(args.slice_dir, f)).convert('L')
        slice_ = np.array(slice_)

        crnt_labels = np.unique(slice_)
        # Ensure all labels are in the label map
        for lbl in crnt_labels:
            # We assume 0 means no label
            if lbl > 0 and lbl not in label_map:
                max_label += 1
                label_map[lbl] = max_label

        # Convert slice labels to proper labels
        for lbl in crnt_labels:
            if lbl > 0:
                slice_[slice_ == lbl] = label_map[lbl]

        # Insert slice into label volume
        labels[:, :, loc] = slice_

    return labels


def main(args):
    vol_size = (args.vol_size_x, args.vol_size_y, args.vol_size_z)

    files = os.listdir(args.slice_dir)
    labels = insert_slices_from_files(vol_size, files)
    np.save(args.ofile, labels)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('slice_dir')
    parser.add_argument('vol_size_x', type=int)
    parser.add_argument('vol_size_y', type=int)
    parser.add_argument('vol_size_z', type=int)
    parser.add_argument('ofile')
    args = parser.parse_args()

    main(args)
