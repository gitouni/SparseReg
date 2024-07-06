import json
import re
import argparse
import numpy as np
import os
from utils.io import match_dirs


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker_dir",type=str,default="data/marker_extra/marker2")
    parser.add_argument("--syn_file",type=str,default="data/marker_extra/syn_pair.json")
    parser.add_argument("--output_dir",type=str,default="data/marker_extra/marker_shift")
    parser.add_argument("--train_idx_save",type=str,default='data/marker_extra/train.txt')
    parser.add_argument("--test_idx_save",type=str,default="data/marker_extra/test.txt")
    parser.add_argument("--train_seq_idx_save",type=str,default="data/marker_extra/train_seq.json")
    parser.add_argument("--test_seq_idx_save",type=str,default="data/marker_extra/test_seq.json")
    parser.add_argument("--test_num",type=int,default=2)
    parser.add_argument("--seed",default=2414)
    return parser.parse_args()



if __name__ == "__main__":
    args = options()
    np.random.seed(args.seed)
    syn_file = json.load(open(args.syn_file,'r'))
    marker_files = sorted(os.listdir(args.marker_dir))
    markered_files = [pair[0] for pair in syn_file]
    matched_idxdict, matched_subdirs = match_dirs(markered_files)
    indenter_keys = list(matched_idxdict.keys())
    Nindenter = len(indenter_keys)
    Idxindenter = np.random.permutation(Nindenter)
    test_indenter_idx = Idxindenter[:args.test_num]
    train_indenter_idx = Idxindenter[args.test_num:]
    train_indenter_names = [indenter_keys[idx_] for idx_ in train_indenter_idx]
    test_indenter_names = [indenter_keys[idx_] for idx_ in test_indenter_idx]
    train_idx = []
    test_idx = []
    train_idx_dict = dict()
    test_idx_dict = dict()
    for indidx in train_indenter_idx:
        key = indenter_keys[indidx]
        train_idx.extend(matched_idxdict[key])
        train_idx_dict[key] = matched_subdirs[key]
    for indidx in test_indenter_idx:
        key = indenter_keys[indidx]
        test_idx.extend(matched_idxdict[indenter_keys[indidx]])
        test_idx_dict[key] = matched_subdirs[key]
    np.savetxt(args.train_idx_save, train_idx, fmt='%d',header="indenter: {}".format(train_indenter_names))
    np.savetxt(args.test_idx_save, test_idx, fmt="%d", header="indenter: {}".format(test_indenter_names))
    json.dump(train_idx_dict, open(args.train_seq_idx_save,'w'))
    json.dump(test_idx_dict, open(args.test_seq_idx_save,'w'))
    