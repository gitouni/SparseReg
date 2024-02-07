import numpy as np
import argparse
from utils import plot_marker_displacement,refresh_dir
import os
from PIL import Image


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markerless_dir",type=str,default="res/markerless_test")
    parser.add_argument("--marker_shift_dir",type=str,default="res/gan_test_shift")
    parser.add_argument("--output_dir",type=str,default="res/gan_arrowed")
    parser.add_argument("--pt_color",type=int, nargs=3,default=[0,255,0])
    parser.add_argument("--arrow_color",type=int, nargs=3,default=[255,0,0])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    markerless_files = sorted(os.listdir(args.markerless_dir))
    markershift_files = sorted(os.listdir(args.marker_shift_dir))
    refresh_dir(args.output_dir)
    for img_file, shift_file in zip(markerless_files, markershift_files):
        img = Image.open(os.path.join(args.markerless_dir, img_file))
        kpt_shift = np.loadtxt(os.path.join(args.marker_shift_dir, shift_file))
        kpt_coords = kpt_shift[:, :2].astype(np.int32)
        end_coords = (kpt_shift[:, :2] + kpt_shift[:, 2:]).astype(np.int32)
        Image.fromarray(plot_marker_displacement(img, kpt_coords, end_coords, args.pt_color, args.arrow_color)).save(os.path.join(args.output_dir, img_file))