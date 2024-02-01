import numpy as np
from utils import find_marker_centers, refresh_dir
import cv2
import argparse
import os
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--ref_marker",type=str,default="data/ref_marker.png")
    io_parser.add_argument("--ref_markered",type=str,default="data/ref_markered.png")
    io_parser.add_argument("--tracked_img_dir", type=str, default="data/markered")
    io_parser.add_argument("--output_dir",type=str,default="data/marker_shift")
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    tracked_img_list = sorted(os.listdir(args.tracked_img_dir))
    ref_marker = cv2.imread(args.ref_marker, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(args.ref_markered, cv2.IMREAD_GRAYSCALE)
    p0 = np.array(find_marker_centers(ref_marker),dtype=np.float32)
    refresh_dir(args.output_dir)
    for i,track_img_file in tqdm(enumerate(tracked_img_list),total=len(tracked_img_list)):
        track_img = cv2.imread(os.path.join(args.tracked_img_dir, track_img_file), cv2.IMREAD_GRAYSCALE)
        p1, st, err = cv2.calcOpticalFlowPyrLK(ref_img, track_img, p0, None, winSize=args.winSize, maxLevel=args.maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        st = st.reshape(-1)
        kpt0 = p0[st == 1]
        kpt1 = p1[st == 1]
        res = np.hstack((kpt0, kpt1-kpt0))
        np.savetxt(os.path.join(args.output_dir, "%04d.txt"%i), res, fmt="%.4f")