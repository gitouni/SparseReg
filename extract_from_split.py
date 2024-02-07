import numpy as np
import argparse
import os
import shutil

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir",default="ex_data/markerless")
    parser.add_argument("--target_dir",default="res/markerless_test")
    parser.add_argument("--split",default="ex_data/test.txt")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    split = np.loadtxt(args.split, dtype=np.int32)
    source_files = sorted(os.listdir(args.source_dir))
    if os.path.exists(args.target_dir):
        shutil.rmtree(args.target_dir)
    os.makedirs(args.target_dir)
    for index in split:
        shutil.copy(os.path.join(args.source_dir, source_files[index]), args.target_dir)