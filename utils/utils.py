import os
import shutil
import cv2
import numpy as np
from skimage import measure


def refresh_dir(dirname:str):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def find_marker(frame,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        mask_range=(150, 255), min_value:int=70,
        morphop_iter=1, morphclose_iter=2, dilate_iter=1):
    """find markers in the tactile iamge

    Args:
        frame (np.ndarray): input image (can be RGB or grayscale)
        morphop_kernel (cv.MatLike, optional): kernel of MORPH_OPEN. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        morphclose_kernel (cv.MatLike, optional): kernel of MORPH_CLOSE. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        dilate_kernel (cv.MatLike, optional): kernel of dilation. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        mask_range (tuple, optional): range of mask segementation. Defaults to (150, 255).
        min_value (int, optional): minimum value to segement marker from HSV (V-chan). Defaults to 70.
        morphop_iter (int, optional): iteration of MORPH_OPEN operation. Defaults to 1.
        morphclose_iter (int, optional): iteration of MORPH_CLOSE operation. Defaults to 2.
        dilate_iter (int, optional): iteration of DILATION operation. Defaults to 1.

    Returns:
        np.ndarray: final mask (0, 255) np.unint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    # img_sblur = cv2.GaussianBlur(gray,(3,3),5)
    img_lblur = cv2.GaussianBlur(gray, (15,15),5)
    im_blur_sub = img_lblur - gray + 128
    blur_mask = np.logical_and(im_blur_sub >= mask_range[0], im_blur_sub <= mask_range[1])
    value_mask = value < min_value
    mask = np.logical_or(blur_mask, value_mask)
    mask255 = np.array(255 * mask,dtype=np.uint8)
    mask255_op = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, morphop_kernel, iterations=morphop_iter)
    if dilate_iter > 0:
        dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    else:
        dilate_mask = mask255_op
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=morphclose_iter)
    return morph_close

def find_marker_props(marker_img:np.ndarray):
    label_img = measure.label(marker_img)
    props = measure.regionprops(label_img)
    return props

def find_marker_centers(marker_img:np.ndarray):
    label_img = measure.label(marker_img)
    props = measure.regionprops(label_img)
    return [[prop.centroid[1], prop.centroid[0]] for prop in props]

def vec_mindis(p0:np.ndarray, p1:np.ndarray):
    N = p0.shape[0]
    M = p1.shape[0]
    p0 = np.repeat(p0[None,:,:],M,axis=0)  # M, N, 2
    p1 = np.repeat(p1[:,None,:],N,axis=1)  # M, N, 2
    dis = np.sum((p0 - p1)**2, axis=-1)  # (M, N)
    index = np.argmin(dis, axis=0)  # p0 <-> p1[index]
    return index

def bilateral_vec_mindis(p0:np.ndarray, p1:np.ndarray):
    N = p0.shape[0]
    M = p1.shape[0]
    p0 = np.repeat(p0[None,:,:],M,axis=0)  # M, N, 2
    p1 = np.repeat(p1[:,None,:],N,axis=1)  # M, N, 2
    dis = np.sum((p0 - p1)**2, axis=-1)  # (M, N)
    p1_index = np.argmin(dis, axis=0)  # p0 <-> p1[index]
    p0_index = np.argmin(dis, axis=1)  # p1 <-> p0[index]
    return p0_index, p1_index

def plot_marker_center(img:np.ndarray, centers:np.ndarray):
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        out = np.copy(img)
    for pt in centers:
        cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), thickness=-1)
    return out

def plot_marker_displacement(img:np.ndarray, st_points:np.ndarray, end_points:np.ndarray, pt_color=(0,255,0), arrow_corlor=(255,0,0)):
    out_img = np.ascontiguousarray(img).copy()
    for st_pt, end_pt in zip(st_points, end_points):
        cv2.circle(out_img, (int(st_pt[0]), int(st_pt[1])), 3, pt_color, thickness=-1)
        cv2.arrowedLine(out_img, (int(st_pt[0]), int(st_pt[1])), (int(end_pt[0]), int(end_pt[1])), arrow_corlor, thickness=1, line_type=cv2.LINE_AA, tipLength=1)
    return out_img

def plot_marker_displacement2(img:np.ndarray, mask:np.ndarray, shift:np.ndarray, pt_color=(0,255,0), arrow_corlor=(255,0,0)):
    out_img = np.ascontiguousarray(img).copy() # image need to be contiguous
    ylist, xlist = np.nonzero(mask)
    for x, y in zip(xlist, ylist):
        dx, dy = shift[y, x]
        cv2.circle(out_img, (int(x), int(y)), 3, pt_color, thickness=-1)
        cv2.arrowedLine(out_img, (int(x), int(y)), (int(x + dx), int(y + dy)), arrow_corlor, thickness=1, line_type=cv2.LINE_AA, tipLength=1)
    return out_img

def plot_marker_delta(img:np.ndarray, end_points:np.ndarray, delta_vecs:np.ndarray, pt_color=(0, 255, 0), arrow_color=(255,0,0), scale:float=3.0):
    out_img = np.ascontiguousarray(img).copy()
    for end_pt, delta in zip(end_points, delta_vecs):
        cv2.circle(out_img, (int(end_pt[0]), int(end_pt[1])), 3, pt_color, thickness=-1)
        cv2.arrowedLine(out_img, (int(end_pt[0]), int(end_pt[1])), (int(end_pt[0] + delta[0]*scale), int(end_pt[1] + delta[1]*scale)), arrow_color, thickness=2, line_type=cv2.LINE_AA, tipLength=0.2)
    return out_img

