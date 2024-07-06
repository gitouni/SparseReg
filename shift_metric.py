import numpy as np
import argparse
import os
import json
from collections import OrderedDict

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_shift_dir", type=str, default="res/pred_shift")
    parser.add_argument("--gt_shift_dir",type=str,default="res/gt_shift")
    parser.add_argument("--save_file",type=str,default="metrics/marker_adding/deeplabv3.json")
    parser.add_argument("--contact_threshold",type=float,default=3.0)
    parser.add_argument("--img_size",type=int,default=[480, 640])
    return parser.parse_args()

def norm(arr:np.ndarray, axis:int) -> np.ndarray:
    return np.sqrt(np.sum(arr**2, axis=axis))

if __name__ == "__main__":
    args = options()
    pred_shift_files = sorted(os.listdir(args.pred_shift_dir))
    gt_shift_files = sorted(os.listdir(args.gt_shift_dir))
    x_err = []
    y_err = []
    norm_err = []
    contact_x_err = []
    contact_y_err = []
    contact_ori_err = []
    contact_norm_err = []
    err_files = 0
    for pred_file, gt_file in zip(pred_shift_files, gt_shift_files):
        gt_coord_mat = np.zeros(args.img_size+[2], dtype=np.float32)
        pred_coord_mat = np.zeros_like(gt_coord_mat)
        pred_data = np.loadtxt(os.path.join(args.pred_shift_dir, pred_file))
        gt_data = np.loadtxt(os.path.join(args.gt_shift_dir, gt_file))
        pred_kpts = pred_data[:, :2].astype(np.int32)
        gt_kpts = gt_data[:,:2].astype(np.int32)
        pred_shift = pred_data[:, 2:].astype(np.float32)
        gt_shift = gt_data[:, 2:].astype(np.float32)
        gt_shift_norm = norm(gt_shift, 1)
        pred_shift_norm = norm(pred_shift, 1)
        if len(gt_shift_norm) != len(pred_shift_norm):
            err_files += 1
        gt_contact_idx = gt_shift_norm >= args.contact_threshold
        gt_x, gt_y = gt_kpts[:,0], gt_kpts[:,1]
        pred_x, pred_y = pred_kpts[:, 0], pred_kpts[:, 1]
        gt_coord_mat[gt_y, gt_x] = gt_shift
        pred_coord_mat[pred_y, pred_x] = pred_shift
        xy_err = gt_coord_mat[gt_y, gt_x] - pred_coord_mat[gt_y, gt_x]
        pred_shift_norm = norm(pred_coord_mat[gt_y, gt_x], 1)
        gt_contact_shift = gt_shift[gt_contact_idx]
        gt_contact_ori = np.arctan2(gt_contact_shift[:,1],gt_contact_shift[:,0])
        gt_contact_norm = norm(gt_contact_shift, 1)
        pred_contact_shift = pred_coord_mat[gt_y[gt_contact_idx], gt_x[gt_contact_idx]]
        pred_contact_ori = np.arctan2(pred_contact_shift[:,1],pred_contact_shift[:,0])
        pred_contact_norm = norm(pred_contact_shift, 1)
        norm_err.extend(gt_shift_norm - pred_shift_norm)
        x_err.extend(xy_err[:, 0].tolist())
        y_err.extend(xy_err[:, 1].tolist())
        contact_x_err.extend(xy_err[gt_contact_idx,0].tolist())
        contact_y_err.extend(xy_err[gt_contact_idx,1].tolist())
        contact_ori_err_item = np.arccos(np.cos((gt_contact_ori-pred_contact_ori)))
        contact_ori_err.extend(contact_ori_err_item.tolist())
        contact_norm_err.extend((gt_contact_norm - pred_contact_norm).tolist())
    rmse = np.sqrt(np.array(x_err)**2 + np.array(y_err)**2)
    rmse_median = np.median(rmse)
    rmse_mean = np.mean(rmse)
    abs_norm_err = np.abs(np.array(norm_err))
    abs_contact_norm_err = np.abs(np.array(contact_norm_err))
    norm_mean = np.mean(abs_norm_err)
    ori_mean = np.mean(contact_ori_err)
    contact_norm_mean = np.mean(abs_contact_norm_err)
    norm_median = np.median(abs_norm_err)
    ori_median = np.median(contact_ori_err)
    contact_norm_median = np.median(abs_contact_norm_err)
    contact_rmse = np.sqrt(np.array(contact_x_err)**2 + np.array(contact_y_err)**2)
    crmse_mean = np.mean(contact_rmse)
    crmse_median = np.median(contact_rmse)
    json.dump(OrderedDict(
        err_files=err_files,
        rmse_mean=rmse_mean.item(),rmse_median=rmse_median.item(),
        norm_mean=norm_mean.item(),norm_median=norm_median.item(),
        ori_mean=ori_mean.item(), ori_median=ori_median.item(),
        contact_norm_mean=contact_norm_mean.item(), contact_norm_median=contact_norm_median.item(),
        crmse_mean=crmse_mean.item(), crmse_median=crmse_median.item()),
        open(args.save_file,'w'),indent=4)