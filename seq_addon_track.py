import numpy as np
from utils import find_marker_centers, refresh_dir
import cv2
import argparse
import os
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--ref_marker",type=str,default="ex_data/ref_marker.png")
    io_parser.add_argument("--ref_markerless",type=str,default="ex_data/ref_markerless.png")
    io_parser.add_argument("--ref_marker_center",type=str,default="ex_data/ref_marker_center.txt")
    io_parser.add_argument("--tracked_img_dir", type=str, default="res/markerless_test")
    io_parser.add_argument("--tracked_marker_dir",type=str, default="res/gan_test")
    io_parser.add_argument("--output_dir",type=str,default="res/gan_test_shift")
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    tracked_img_list = sorted(os.listdir(args.tracked_img_dir))
    tracked_marker_list = sorted(os.listdir(args.tracked_marker_dir))
    ref_marker = cv2.imread(args.ref_marker, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(args.ref_markerless, cv2.IMREAD_GRAYSCALE)
    ref_mask_marker = np.array(ref_marker/255,dtype=np.float32)
    ref_img = np.array(ref_img * (1.0 - ref_mask_marker) + ref_marker * (255-ref_mask_marker), dtype=np.uint8)
    if not os.path.isfile(args.ref_marker_center):
        p0 = np.array(find_marker_centers(ref_marker),dtype=np.float32)
        np.savetxt(args.ref_marker_center, p0, fmt='%.4f')
    else:
        p0 = np.loadtxt(args.ref_marker_center,dtype=np.float32)
    refresh_dir(args.output_dir)
    for i, (track_img_file,track_marker_file) in tqdm(enumerate(zip(tracked_img_list,tracked_marker_list)),total=len(tracked_img_list)):
        track_img = cv2.imread(os.path.join(args.tracked_img_dir, track_img_file), cv2.IMREAD_GRAYSCALE)
        track_marker = cv2.imread(os.path.join(args.tracked_marker_dir, track_marker_file), cv2.IMREAD_GRAYSCALE)
        track_marker[track_marker > 0] = 255
        track_mask_marker = np.array(track_marker/255,dtype=np.float32)
        track_img = np.array(track_img * (1-track_mask_marker) + track_mask_marker * (255-track_marker),dtype=np.uint8)
        p1, st, err = cv2.calcOpticalFlowPyrLK(ref_img, track_img, p0, None, winSize=args.winSize, maxLevel=args.maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        st = st.reshape(-1)
        kpt0 = p0[st == 1]
        kpt1 = p1[st == 1]
        kpt0 = p0
        kpt1 = p1
        res = np.hstack((kpt0, kpt1-kpt0))
        np.savetxt(os.path.join(args.output_dir, "%04d.txt"%i), res, fmt="%.4f")