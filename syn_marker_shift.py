import json
import argparse
from utils import find_marker_centers, bilateral_vec_mindis, refresh_dir
import cv2
import numpy as np
import os
from utils.io import match_dirs

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker_dir",type=str,default="data/marker_extra/marker2")
    parser.add_argument("--syn_file",type=str,default="data/marker_extra/syn_pair.json")
    parser.add_argument("--output_dir",type=str,default="data/marker_extra/marker_shift")
    return parser.parse_args()



if __name__ == "__main__":
    args = options()
    syn_file = json.load(open(args.syn_file,'r'))
    marker_files = sorted(os.listdir(args.marker_dir))
    markered_files = [pair[0] for pair in syn_file]
    _, matched_subdirs = match_dirs(markered_files)
    refresh_dir(args.output_dir)
    for s_name in matched_subdirs.keys():
        for ss_name in matched_subdirs[s_name].keys():
            matched_subdirs[s_name][ss_name]['normal'].sort()  # filename sort
            matched_subdirs[s_name][ss_name]['shear'].sort()  # filename sort
            print("{}/{}/normal has {} files".format(s_name, ss_name, len(matched_subdirs[s_name][ss_name]['normal'])))
            print("{}/{}/shear has {} files".format(s_name, ss_name, len(matched_subdirs[s_name][ss_name]['shear'])))
            # normal force marker motion
            for type_cls in ['normal','shear']:
                ref_img_name = marker_files[matched_subdirs[s_name][ss_name][type_cls][0]]
                ref_img_fullpath = os.path.join(args.marker_dir, ref_img_name)
                ref_marker_img = cv2.imread(ref_img_fullpath, cv2.IMREAD_GRAYSCALE)
                p0 = np.array(find_marker_centers(ref_marker_img))
                marker_shift = np.zeros_like(p0)
                marker_valid = np.ones(len(p0), dtype=np.bool_)
                save_data = np.hstack((p0, marker_shift))
                np.savetxt(os.path.join(args.output_dir, os.path.splitext(ref_img_name)[0]+'.txt'),
                        save_data, fmt="%0.6f", header="x y dx dy")
                for normal_file_idx in matched_subdirs[s_name][ss_name][type_cls][1:]:
                    normal_file = marker_files[normal_file_idx]
                    normal_fullpath = os.path.join(args.marker_dir, normal_file)
                    marker_img = cv2.imread(normal_fullpath, cv2.IMREAD_GRAYSCALE)
                    p1 = np.array(find_marker_centers(marker_img))
                    p0_index, p1_index = bilateral_vec_mindis(p0, p1)
                    cycle_index = p0_index[p1_index]
                    p0_ii = np.arange(len(p0))
                    valid = p0_ii == cycle_index
                    marker_valid = np.logical_and(marker_valid, valid)
                    marker_shift[valid] += p1[p1_index[valid]] - p0[valid]
                    save_data = np.hstack((p0[marker_valid], marker_shift[marker_valid]))
                    np.savetxt(os.path.join(args.output_dir, os.path.splitext(normal_file)[0]+'.txt'),
                            save_data, fmt="%0.6f", header="x y dx dy")
            

