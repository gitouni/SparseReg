import numpy as np
import os
import argparse
import json

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str,default="ex_data/marker")
    parser.add_argument("--save",type=str,nargs=2,default=['ex_data/train.txt','ex_data/test.txt'])
    parser.add_argument("--test_ratio",type=float,default=0.15)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    N = len(os.listdir(args.input_dir))
    teN = int(N * args.test_ratio)
    trN = N - teN
    np.random.seed(0)
    random_index = np.random.permutation(N)
    np.savetxt(args.save[0],random_index[:trN],fmt='%d')
    np.savetxt(args.save[1],np.sort(random_index[trN:]),fmt='%d')