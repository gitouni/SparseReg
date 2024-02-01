import numpy as np
from utils import find_marker_centers, plot_marker_displacement
import cv2
import argparse

def options():
    parser = argparse.ArgumentParser()
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=2)
    return parser.parse_args()

def vec_mindis(p0:np.ndarray, p1:np.ndarray):
    N = p0.shape[0]
    M = p1.shape[0]
    p0 = np.repeat(p0[None,:,:],N,axis=0)
    p1 = np.repeat(p1[:,None,:],M,axis=1)
    dis = np.sum((p0 - p1)**2, axis=-1)  # (N, N)
    index = np.argmin(dis, axis=0)  # p1 <-> p0[index]
    return index


if __name__ == "__main__":
    args = options()
    ref_marker = cv2.imread("data/ref_marker.png")
    marker = cv2.imread("data/marker/0008.png")
    img1 = cv2.imread("data/ref3_gan_marker.png")
    img2 = cv2.imread("data/markered/0008.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    p0 = np.array(find_marker_centers(ref_marker))
    p1 = np.array(find_marker_centers(marker))
    print("p0: {}, p1: {}".format(len(p0), len(p1)))
    index = vec_mindis(p0, p1)
    out = plot_marker_displacement(img2, p0, p1[index])
    # Display the blended image
    cv2.imwrite('debug/tracked_mindis.png', out)