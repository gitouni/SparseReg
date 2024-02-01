import cv2
from utils import find_marker_props
from matplotlib import pyplot as plt
import math
img = cv2.imread("data/marker/0003.png", cv2.IMREAD_GRAYSCALE)
H,W = img.shape
props = find_marker_props(img)
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)
for prop in props:
    y0, x0 = prop.centroid
    orientation = prop.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * prop.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * prop.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * prop.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * prop.axis_major_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = prop.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, W, H, 0))
plt.savefig("marker_biased.png")
