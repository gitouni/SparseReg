import numpy as np
from utils import find_marker_centers, plot_marker_delta, plot_marker_center, plot_marker_displacement
import cv2
import argparse

def options():
    parser = argparse.ArgumentParser()
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    ref_marker = cv2.imread("ex_data/ref_marker.png")
    img1 = cv2.imread("ex_data/ref_markered.png")
    img2 = cv2.imread("ex_data/markered/0105.png")
    img3 = cv2.imread("ex_data/markerless/0105.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    p0 = np.array(find_marker_centers(ref_marker),dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, winSize=args.winSize, maxLevel=args.maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    st = st.reshape(-1)
    kpt0 = p0[st == 1]
    kpt1 = p1[st == 1]
    out_ref = plot_marker_center(img1, p0)
    out1 = plot_marker_displacement(img2, kpt0, kpt1)
    out2 = plot_marker_displacement(img3, kpt0, kpt1)
    # Display the blended image
    cv2.imwrite('debug/marker_centers.png', out_ref)
    cv2.imwrite('debug/tracked_markered.png', out1)
    cv2.imwrite('debug/tracked_markerless.png', out2)