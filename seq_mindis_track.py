import numpy as np
from utils import find_marker_centers, refresh_dir
from debug_track_marker_min import vec_mindis
import cv2
import argparse
import os
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--ref_marker",type=str,default="ex_data/ref_marker.png")
    io_parser.add_argument("--ref_marker_center",type=str,default="ex_data/ref_marker_center.txt")
    io_parser.add_argument("--tracked_marker_dir",type=str, default="res/gan_test")
    io_parser.add_argument("--output_dir",type=str,default="res/gan_test_shift")
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    tracked_marker_list = sorted(os.listdir(args.tracked_marker_dir))
    ref_marker = cv2.imread(args.ref_marker, cv2.IMREAD_GRAYSCALE)
    ref_mask_marker = np.array(ref_marker/255,dtype=np.float32)
    if not os.path.isfile(args.ref_marker_center):
        p0 = np.array(find_marker_centers(ref_marker),dtype=np.float32)
        np.savetxt(args.ref_marker_center, p0, fmt='%.4f')
    else:
        p0 = np.loadtxt(args.ref_marker_center,dtype=np.float32)
    refresh_dir(args.output_dir)
    for i, track_marker_file in tqdm(enumerate(tracked_marker_list),total=len(tracked_marker_list)):
        track_marker = cv2.imread(os.path.join(args.tracked_marker_dir, track_marker_file), cv2.IMREAD_GRAYSCALE)
        track_marker[track_marker > 0] = 255
        p1 = np.array(find_marker_centers(track_marker),dtype=np.int32)
        p1_idx = vec_mindis(p0, p1)
        res = np.hstack((p0, p1[p1_idx]-p0))
        np.savetxt(os.path.join(args.output_dir, "%04d.txt"%i), res, fmt="%.4f")